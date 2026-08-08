from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
import re
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from sqlalchemy import Integer, String, bindparam, cast, func, or_, select, text, true
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.storage.models import (
    BbotListenerCacheEntry,
    ConversationEpisode,
    DevSession,
    DevTask,
    DevTaskArtifact,
    EpisodeMessage,
    Group,
    Job,
    MemoryBackfillRun,
    MemoryItem,
    MemoryItemSemanticVector,
    Message,
    RetrievalDocument,
    RetrievalDocumentMessage,
    RetrievalIndexState,
    Summary,
    UsageRecord,
    User,
)
from app.providers.embeddings import hashed_text_embedding
from app.storage.db import validate_retrieval_vector_table_name
from app.core.time_utils import shanghai_naive, shanghai_now_naive


_INELIGIBLE_DELIVERY_STATES = (
    "reserved",
    "blocked",
    "uncertain",
    "deleted",
)

ASIA_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _as_local_day(value: datetime) -> date:
    if value.tzinfo is None:
        return value.date()
    return value.astimezone(ASIA_SHANGHAI).date()

# sqlite-vec rejects KNN queries above this engine limit.  Hard filters such
# as speaker/time are validated after vector ranking, so expansion may reach
# the cap on large groups even when the caller only requests 150 final hits.
_SQLITE_VEC_MAX_K = 4096


def _next_vector_fetch_limit(
    current: int,
    *,
    requested: int,
    available: int,
) -> int:
    searchable = min(max(0, int(available)), _SQLITE_VEC_MAX_K)
    if searchable <= 0:
        return 0
    return min(
        searchable,
        max(int(current) + max(1, int(requested)), int(current) * 2),
    )


def _initial_vector_fetch_limit(
    *,
    requested: int,
    available: int,
    has_post_filters: bool,
) -> int:
    searchable = min(max(0, int(available)), _SQLITE_VEC_MAX_K)
    if searchable <= 0:
        return 0
    if has_post_filters:
        # sqlite-vec can partition by group but sparse positive filters such
        # as speaker/time/mention are validated against canonical source rows
        # afterwards. Starting at the cap avoids repeated KNN scans for them.
        return searchable
    return min(max(1, int(requested)), searchable)


def _vector_fetch_ceiling(
    *,
    requested: int,
    available: int,
    has_sparse_post_filters: bool,
    has_exclusion_filter: bool,
) -> int:
    searchable = min(max(0, int(available)), _SQLITE_VEC_MAX_K)
    if searchable <= 0:
        return 0
    if has_sparse_post_filters:
        return searchable
    if has_exclusion_filter:
        return min(
            searchable,
            max(max(1, int(requested)) * 4, max(1, int(requested)) + 32),
        )
    return min(max(1, int(requested)), searchable)


@dataclass(frozen=True, slots=True)
class RetrievalDocumentHit:
    document_id: int
    group_id: int
    document_kind: str
    episode_id: int | None
    source_msg_ids: tuple[str, ...]
    start_at: datetime
    end_at: datetime
    score: float
    lexical_exact: bool = False


@dataclass(frozen=True, slots=True)
class RetrievalEmbeddingCoverage:
    total_documents: int
    ready_documents: int
    failed_documents: int

    @property
    def coverage(self) -> float:
        if self.total_documents == 0:
            return 1.0
        return self.ready_documents / self.total_documents


def _normalize_utc_sqlite_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(ASIA_SHANGHAI).replace(tzinfo=None)


def _mentioned_user_ids(raw_json: object) -> frozenset[str]:
    if not isinstance(raw_json, dict):
        return frozenset()
    segments = raw_json.get("message", raw_json.get("raw_message"))
    if not isinstance(segments, list):
        return frozenset()
    mentioned: set[str] = set()
    for segment in segments:
        if not isinstance(segment, dict) or str(segment.get("type") or "") != "at":
            continue
        data = segment.get("data")
        if not isinstance(data, dict):
            continue
        for key in ("qq", "uin", "target"):
            value = data.get(key)
            if value is None or isinstance(value, bool):
                continue
            normalized = str(value).strip()
            if normalized:
                mentioned.add(normalized)
    return frozenset(mentioned)


def _normalize_optional_string_filter(
    values: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    return tuple(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )


def _normalize_optional_integer_filter(
    values: Sequence[str] | None,
) -> tuple[int, ...] | None:
    if values is None:
        return None
    return tuple(
        dict.fromkeys(
            int(value)
            for value in values
            if str(value).strip().lstrip("-").isdigit()
        )
    )


def _retrieval_source_prefilters(
    *,
    group_id: int,
    speaker_ids: Sequence[str] | None,
    excluded_speaker_ids: Sequence[str] | None = None,
) -> list[Any]:
    """SQL prefilters applied before ranking/limit on document channels."""

    unsafe_document_ids = (
        select(RetrievalDocumentMessage.document_id)
        .join(
            Message,
            (Message.id == RetrievalDocumentMessage.message_id)
            & (Message.group_id == RetrievalDocumentMessage.group_id),
        )
        .where(
            RetrievalDocumentMessage.group_id == int(group_id),
            Message.group_id == int(group_id),
            func.json_extract(Message.raw_json, "$.delivery_state").in_(
                _INELIGIBLE_DELIVERY_STATES
            ),
        )
    )
    filters: list[Any] = [
        RetrievalDocument.id.not_in(unsafe_document_ids),
    ]
    normalized_speakers = _normalize_optional_integer_filter(speaker_ids)
    if normalized_speakers is not None:
        if not normalized_speakers:
            filters.append(RetrievalDocument.id < 0)
        else:
            mismatched_document_ids = (
                select(RetrievalDocumentMessage.document_id)
                .join(
                    Message,
                    (Message.id == RetrievalDocumentMessage.message_id)
                    & (Message.group_id == RetrievalDocumentMessage.group_id),
                )
                .where(
                    RetrievalDocumentMessage.group_id == int(group_id),
                    Message.group_id == int(group_id),
                    Message.user_id.not_in(normalized_speakers),
                )
            )
            filters.append(
                RetrievalDocument.id.not_in(mismatched_document_ids)
            )
    normalized_excluded_speakers = _normalize_optional_integer_filter(
        excluded_speaker_ids
    )
    if normalized_excluded_speakers:
        excluded_document_ids = (
            select(RetrievalDocumentMessage.document_id)
            .join(
                Message,
                (Message.id == RetrievalDocumentMessage.message_id)
                & (Message.group_id == RetrievalDocumentMessage.group_id),
            )
            .where(
                RetrievalDocumentMessage.group_id == int(group_id),
                Message.group_id == int(group_id),
                Message.user_id.in_(normalized_excluded_speakers),
            )
        )
        filters.append(RetrievalDocument.id.not_in(excluded_document_ids))
    return filters


def _retrieval_mention_document_ids(
    *,
    group_id: int,
    mentioned_user_ids: Sequence[str],
):
    """Select documents backed by a real OneBot ``at`` segment."""

    normalized_mentions = _normalize_optional_string_filter(mentioned_user_ids)
    if not normalized_mentions:
        return select(RetrievalDocument.id).where(RetrievalDocument.id < 0)
    segments = func.json_each(
        Message.raw_json,
        "$.message",
    ).table_valued("value").alias("mention_segment")
    return (
        select(RetrievalDocumentMessage.document_id)
        .join(
            Message,
            (Message.id == RetrievalDocumentMessage.message_id)
            & (Message.group_id == RetrievalDocumentMessage.group_id),
        )
        .join(segments, true())
        .where(
            RetrievalDocumentMessage.group_id == int(group_id),
            Message.group_id == int(group_id),
            func.json_extract(segments.c.value, "$.type") == "at",
            cast(
                func.json_extract(segments.c.value, "$.data.qq"),
                String,
            ).in_(normalized_mentions),
        )
    )


def _delete_active_retrieval_vectors(
    session: Session,
    *,
    document_ids: list[int],
) -> None:
    if not document_ids:
        return
    active_states = list(
        session.scalars(
            select(RetrievalIndexState).where(
                RetrievalIndexState.channel == "vector",
                RetrievalIndexState.is_active.is_(True),
            )
        )
    )
    for state in active_states:
        physical_table = validate_retrieval_vector_table_name(
            state.physical_table,
            generation=state.generation,
        )
        try:
            session.execute(
                text(
                    f"DELETE FROM {physical_table} "
                    "WHERE document_id IN :document_ids"
                ).bindparams(bindparam("document_ids", expanding=True)),
                {
                    "document_ids": tuple(
                        int(document_id) for document_id in document_ids
                    )
                },
            )
        except SQLAlchemyError:
            # The active metadata can outlive optional extension availability
            # in a process. Canonical status still prevents use by fallback
            # channels; vector cleanup can be retried when sqlite-vec returns.
            continue


def _deactivate_memory_retrieval_documents(
    session: Session,
    *,
    memory_id: int,
    keep_document_id: int | None = None,
) -> int:
    filters = [
        RetrievalDocument.document_kind == "memory",
        RetrievalDocument.source_table == "memory_items",
        RetrievalDocument.source_id == str(int(memory_id)),
        RetrievalDocument.status == "active",
    ]
    if keep_document_id is not None:
        filters.append(RetrievalDocument.id != int(keep_document_id))
    documents = list(session.scalars(select(RetrievalDocument).where(*filters)))
    if not documents:
        return 0
    now = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
    for document in documents:
        document.status = "inactive"
        document.embedding_status = "stale"
        document.updated_at = now
        session.add(document)
        try:
            session.execute(
                text("DELETE FROM retrieval_documents_fts WHERE document_id = :document_id"),
                {"document_id": str(document.id)},
            )
        except SQLAlchemyError:
            pass
    _delete_active_retrieval_vectors(
        session,
        document_ids=[document.id for document in documents],
    )
    return len(documents)


class GroupRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_group(self, *, group_id: int, group_name: str, enabled: bool, speak_enabled: bool) -> Group:
        group = self.session.get(Group, group_id) or Group(group_id=group_id)
        group.group_name = group_name
        group.enabled = enabled
        group.speak_enabled = speak_enabled
        self.session.add(group)
        return group

    def set_speak_enabled(self, group_id: int, value: bool) -> None:
        group = self.session.get(Group, group_id) or Group(group_id=group_id)
        group.speak_enabled = value
        self.session.add(group)

    def set_enabled(self, group_id: int, value: bool) -> None:
        group = self.session.get(Group, group_id) or Group(group_id=group_id)
        group.enabled = value
        self.session.add(group)

    def get_group(self, group_id: int) -> Group | None:
        return self.session.get(Group, group_id)


class UserRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_user(self, *, user_id: int, nickname: str, group_card: str) -> User:
        user = self.session.get(User, user_id) or User(user_id=user_id)
        user.nickname = nickname
        user.group_card = group_card
        now = datetime.now().astimezone()
        user.first_seen_at = user.first_seen_at or now
        user.last_seen_at = now
        self.session.add(user)
        return user

    def get_users_by_ids(self, user_ids: list[int]) -> dict[int, User]:
        unique_ids = list(dict.fromkeys(user_ids))
        if not unique_ids:
            return {}
        stmt = select(User).where(User.user_id.in_(unique_ids))
        return {user.user_id: user for user in self.session.scalars(stmt)}


class MessageRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def _is_reserved_outbound(message: Message) -> bool:
        raw_json = message.raw_json
        return isinstance(raw_json, dict) and raw_json.get("delivery_state") == "reserved"

    @staticmethod
    def is_reserved_outbound(message: Message) -> bool:
        return MessageRepository._is_reserved_outbound(message)

    @staticmethod
    def is_qq_blocked_outbound(message: Message) -> bool:
        raw_json = message.raw_json
        return (
            isinstance(raw_json, dict)
            and raw_json.get("delivery_state") == "blocked"
            and raw_json.get("failure_kind") == "qq_sensitive_content"
        )

    @staticmethod
    def is_delivery_uncertain_outbound(message: Message) -> bool:
        raw_json = message.raw_json
        return (
            isinstance(raw_json, dict)
            and raw_json.get("delivery_state") == "uncertain"
            and raw_json.get("failure_kind") == "delivery_result_unknown"
        )

    @staticmethod
    def _is_unconfirmed_outbound(message: Message) -> bool:
        return (
            MessageRepository.is_qq_blocked_outbound(message)
            or MessageRepository.is_delivery_uncertain_outbound(message)
        )

    def add_group_message(
        self,
        *,
        platform_msg_id: str,
        group_id: int,
        user_id: int,
        timestamp: datetime,
        plain_text: str,
        raw_json: dict[str, Any],
        msg_type: str,
        reply_to_msg_id: str | None,
        mentioned_bot: bool,
    ) -> Message:
        self.session.flush()
        message = Message(
            platform_msg_id=platform_msg_id,
            group_id=group_id,
            user_id=user_id,
            timestamp=shanghai_naive(timestamp),
            plain_text=plain_text,
            raw_json=raw_json,
            msg_type=msg_type,
            reply_to_msg_id=reply_to_msg_id,
            mentioned_bot=mentioned_bot,
        )
        self.session.add(message)
        return message

    def add_private_message(
        self,
        *,
        platform_msg_id: str,
        user_id: int,
        timestamp: datetime,
        plain_text: str,
        raw_json: dict[str, Any],
        msg_type: str = "text",
        reply_to_msg_id: str | None = None,
        mentioned_bot: bool = False,
    ) -> Message:
        self.session.flush()
        message = Message(
            platform_msg_id=platform_msg_id,
            group_id=None,
            user_id=user_id,
            timestamp=shanghai_naive(timestamp),
            plain_text=plain_text,
            raw_json=raw_json,
            msg_type=msg_type,
            reply_to_msg_id=reply_to_msg_id,
            mentioned_bot=mentioned_bot,
        )
        self.session.add(message)
        return message

    def get_by_platform_msg_id(self, platform_msg_id: str) -> Message | None:
        stmt = select(Message).where(Message.platform_msg_id == platform_msg_id).limit(1)
        return self.session.execute(stmt).scalar_one_or_none()

    def mark_group_message_deleted(
        self,
        *,
        group_id: int,
        platform_msg_id: str,
        reason: str = "group_recall",
    ) -> Message | None:
        message = self.session.scalars(
            select(Message).where(
                Message.group_id == int(group_id),
                Message.platform_msg_id == str(platform_msg_id),
            )
        ).first()
        if message is None:
            return None
        raw_json = (
            dict(message.raw_json)
            if isinstance(message.raw_json, dict)
            else {}
        )
        raw_json["delivery_state"] = "deleted"
        raw_json["deletion_reason"] = str(reason or "group_recall")[:64]
        message.raw_json = raw_json
        self.session.add(message)
        return message

    def get_group_messages_by_platform_msg_ids(
        self,
        *,
        group_id: int,
        platform_msg_ids: list[str],
    ) -> dict[str, Message]:
        identifiers = list(dict.fromkeys(str(item).strip() for item in platform_msg_ids if str(item).strip()))
        if not identifiers:
            return {}
        rows = self.session.scalars(
            select(Message).where(
                Message.group_id == int(group_id),
                Message.platform_msg_id.in_(identifiers),
            )
        )
        return {str(message.platform_msg_id): message for message in rows}

    def list_direct_group_replies(
        self,
        *,
        group_id: int,
        parent_platform_msg_ids: list[str],
        scan_limit_per_parent: int,
    ) -> list[Message]:
        """Load a bounded direct-reply scan for scoped parents in stable order.

        The reply quota is applied after the query plan's delivery, subject,
        and time eligibility checks. This larger scan bound prevents unbounded
        ORM hydration without consuming the final two-reply eligibility quota.
        """
        if scan_limit_per_parent < 1:
            raise ValueError("direct reply scan limit must be positive")
        parent_ids = list(
            dict.fromkeys(
                str(item).strip()
                for item in parent_platform_msg_ids
                if str(item).strip()
            )
        )
        if not parent_ids:
            return []
        ranked = (
            select(
                Message.id.label("message_id"),
                func.row_number()
                .over(
                    partition_by=Message.reply_to_msg_id,
                    order_by=(Message.timestamp.asc(), Message.id.asc()),
                )
                .label("reply_rank"),
            )
            .where(
                Message.group_id == int(group_id),
                Message.reply_to_msg_id.in_(parent_ids),
                ~func.coalesce(
                    func.json_extract(Message.raw_json, "$.delivery_state"),
                    "",
                ).in_(_INELIGIBLE_DELIVERY_STATES),
            )
            .subquery()
        )
        stmt = (
            select(Message)
            .join(ranked, ranked.c.message_id == Message.id)
            .where(ranked.c.reply_rank <= int(scan_limit_per_parent))
            .order_by(
                Message.reply_to_msg_id.asc(),
                Message.timestamp.asc(),
                Message.id.asc(),
            )
        )
        return list(self.session.scalars(stmt))

    def is_late_group_message(
        self,
        *,
        group_id: int,
        message_id: int,
        timestamp: datetime,
    ) -> bool:
        normalized_timestamp = _normalize_utc_sqlite_timestamp(timestamp)
        stmt = (
            select(Message.id)
            .where(
                Message.group_id == int(group_id),
                Message.id < int(message_id),
                Message.timestamp > normalized_timestamp,
            )
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none() is not None

    def list_recent_group_messages(self, *, group_id: int, limit: int) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.group_id == group_id)
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        recent_messages = []
        for message in self.session.scalars(stmt):
            if self._is_reserved_outbound(message):
                continue
            recent_messages.append(message)
            if len(recent_messages) >= limit:
                break
        return list(reversed(recent_messages))

    def list_recent_group_messages_for_summarization(self, *, group_id: int, limit: int) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.group_id == group_id)
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        recent_messages = []
        for message in self.session.scalars(stmt):
            if self._is_reserved_outbound(message) or self._is_unconfirmed_outbound(message):
                continue
            recent_messages.append(message)
            if len(recent_messages) >= limit:
                break
        return list(reversed(recent_messages))

    def list_group_messages_chronological(
        self,
        *,
        group_id: int,
        exclude_platform_msg_id: str | None = None,
    ) -> list[Message]:
        """Return every delivered group message in its original order."""
        stmt = (
            select(Message)
            .where(Message.group_id == group_id)
            .order_by(Message.timestamp.asc(), Message.id.asc())
        )
        return [
            message
            for message in self.session.scalars(stmt)
            if not self._is_reserved_outbound(message) and message.platform_msg_id != exclude_platform_msg_id
        ]

    def count_group_messages(self, group_id: int) -> int:
        stmt = select(func.count()).select_from(Message).where(Message.group_id == group_id)
        return self.session.scalar(stmt) or 0

    def count_group_inbound_messages(self, *, group_id: int, bot_user_id: int) -> int:
        stmt = (
            select(func.count())
            .select_from(Message)
            .where(
                Message.group_id == group_id,
                Message.user_id != bot_user_id,
                text(
                    "(json_extract(messages.raw_json, '$.delivery_state') IS NULL "
                    "OR json_extract(messages.raw_json, '$.delivery_state') <> 'reserved')"
                ),
            )
        )
        return int(self.session.scalar(stmt) or 0)

    def list_group_messages_for_day(
        self,
        *,
        group_id: int,
        day,
        excluded_user_ids: set[int] | None = None,
    ) -> list[Message]:
        excluded = {int(user_id) for user_id in (excluded_user_ids or set())}
        stmt = (
            select(Message)
            .where(Message.group_id == group_id)
            .order_by(Message.id.asc())
        )
        return [
            message
            for message in self.session.scalars(stmt)
            if _as_local_day(message.timestamp) == day
            and message.user_id not in excluded
            and not self._is_reserved_outbound(message)
            and not self._is_unconfirmed_outbound(message)
        ]

    def list_group_ids(self) -> list[int]:
        stmt = select(Message.group_id).where(Message.group_id.is_not(None)).distinct().order_by(Message.group_id.asc())
        return [int(group_id) for group_id in self.session.scalars(stmt) if group_id is not None]

    def list_recent_group_message_windows(
        self,
        *,
        group_id: int,
        batch_size: int,
        limit_windows: int,
        excluded_user_ids: set[int] | None = None,
    ) -> list[list[Message]]:
        excluded = {int(user_id) for user_id in (excluded_user_ids or set())}
        stmt = (
            select(Message)
            .where(Message.group_id == group_id)
            .order_by(Message.id.asc())
        )
        rows = [
            message
            for message in self.session.scalars(stmt)
            if not self._is_reserved_outbound(message)
            and not self._is_unconfirmed_outbound(message)
            and message.user_id not in excluded
        ]
        windows = [
            rows[index : index + batch_size]
            for index in range(0, len(rows), batch_size)
            if len(rows[index : index + batch_size]) == batch_size
        ]
        return windows[-max(1, limit_windows) :]

    def list_recent_group_inbound_messages(
        self,
        *,
        group_id: int,
        bot_user_id: int,
        limit: int,
    ) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.group_id == group_id, Message.user_id != bot_user_id)
            .order_by(Message.id.desc())
        )
        rows: list[Message] = []
        for message in self.session.scalars(stmt):
            if self._is_reserved_outbound(message) or self._is_unconfirmed_outbound(message):
                continue
            rows.append(message)
            if len(rows) >= max(1, limit):
                break
        return list(reversed(rows))

    def list_group_messages_by_id_range(
        self,
        *,
        group_id: int,
        start_id: int,
        end_id: int,
        limit: int = 200,
    ) -> list[Message]:
        stmt = (
            select(Message)
            .where(
                Message.group_id == group_id,
                Message.id >= start_id,
                Message.id <= end_id,
            )
            .order_by(Message.id.asc())
            .limit(max(1, limit))
        )
        return [
            message
            for message in self.session.scalars(stmt)
            if not self._is_reserved_outbound(message) and not self._is_unconfirmed_outbound(message)
        ]

    def list_recent_group_messages_for_user(self, *, group_id: int, user_id: int, limit: int) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.group_id == group_id, Message.user_id == user_id)
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        recent_messages = []
        for message in self.session.scalars(stmt):
            if self._is_reserved_outbound(message) or self._is_unconfirmed_outbound(message):
                continue
            recent_messages.append(message)
            if len(recent_messages) >= limit:
                break
        return list(reversed(recent_messages))

    def list_recent_group_messages_for_user_since(
        self,
        *,
        group_id: int,
        user_id: int,
        since: datetime,
        limit: int,
    ) -> list[Message]:
        stmt = (
            select(Message)
            .where(
                Message.group_id == group_id,
                Message.user_id == user_id,
                Message.timestamp >= since,
            )
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        recent_messages = []
        for message in self.session.scalars(stmt):
            if self._is_reserved_outbound(message):
                continue
            recent_messages.append(message)
            if len(recent_messages) >= limit:
                break
        return list(reversed(recent_messages))

    def list_recent_private_messages_for_user_since(
        self,
        *,
        user_id: int,
        since: datetime,
        limit: int,
    ) -> list[Message]:
        stmt = (
            select(Message)
            .where(
                Message.group_id.is_(None),
                Message.user_id == user_id,
                Message.timestamp >= since,
            )
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        recent_messages = []
        for message in self.session.scalars(stmt):
            if self._is_reserved_outbound(message):
                continue
            recent_messages.append(message)
            if len(recent_messages) >= limit:
                break
        return list(reversed(recent_messages))

    def list_group_messages_matching_terms(
        self,
        *,
        group_id: int,
        terms: list[str],
        exclude_platform_msg_ids: set[str],
        limit: int,
    ) -> list[Message]:
        normalized_terms = list(dict.fromkeys(term.strip().lower() for term in terms if len(term.strip()) >= 2))
        if limit <= 0 or not normalized_terms:
            return []
        stmt = (
            select(Message)
            .where(
                Message.group_id == group_id,
                Message.plain_text.is_not(None),
                or_(*(Message.plain_text.ilike(f"%{term}%") for term in normalized_terms)),
            )
            .order_by(Message.timestamp.desc(), Message.id.desc())
            .limit(max(limit * 4, limit))
        )
        matched: list[Message] = []
        for message in self.session.scalars(stmt):
            if message.platform_msg_id in exclude_platform_msg_ids or self._is_reserved_outbound(message):
                continue
            matched.append(message)
            if len(matched) >= limit:
                break
        return matched

    def list_recent_group_user_ids(self, *, group_id: int, limit: int) -> list[int]:
        latest_message_at = func.max(Message.timestamp).label("latest_message_at")
        stmt = (
            select(Message.user_id, latest_message_at)
            .where(Message.group_id == group_id)
            .group_by(Message.user_id)
            .order_by(latest_message_at.desc())
            .limit(limit)
        )
        return [int(user_id) for user_id, _latest in self.session.execute(stmt)]

    def list_recent_group_member_messages(
        self,
        *,
        group_id: int | None,
        limit: int | None,
    ) -> list[Message]:
        """Return latest eligible sender snapshots, optionally across all groups."""

        if limit is not None and limit <= 0:
            return []
        rank_partition = (
            (Message.group_id, Message.user_id)
            if group_id is None
            else Message.user_id
        )
        filters = (
            [Message.group_id.is_not(None)]
            if group_id is None
            else [Message.group_id == int(group_id)]
        )
        ranked = (
            select(
                Message.id.label("message_id"),
                func.row_number()
                .over(
                    partition_by=rank_partition,
                    order_by=(Message.timestamp.desc(), Message.id.desc()),
                )
                .label("member_rank"),
            )
            .where(
                *filters,
                ~func.coalesce(
                    func.json_extract(Message.raw_json, "$.delivery_state"),
                    "",
                ).in_(_INELIGIBLE_DELIVERY_STATES),
            )
            .subquery()
        )
        stmt = (
            select(Message)
            .join(ranked, ranked.c.message_id == Message.id)
            .where(ranked.c.member_rank == 1)
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        if limit is not None:
            stmt = stmt.limit(int(limit))
        return list(self.session.scalars(stmt))

    def last_bot_reply_at(self, *, group_id: int, bot_user_id: int) -> datetime | None:
        stmt = (
            select(Message)
            .where(Message.group_id == group_id, Message.user_id == bot_user_id)
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        timestamp = None
        for message in self.session.scalars(stmt):
            if self._is_reserved_outbound(message) or self.is_qq_blocked_outbound(message):
                continue
            timestamp = message.timestamp
            break
        if timestamp is None:
            return None
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=ASIA_SHANGHAI).astimezone(UTC)
        return timestamp


class BbotListenerCacheRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_entry(
        self,
        *,
        group_id: int,
        platform: str,
        external_id: str,
        canonical_name: str,
        aliases: list[str],
        source: str,
        updated_at: datetime,
    ) -> BbotListenerCacheEntry:
        stmt = (
            select(BbotListenerCacheEntry)
            .where(
                BbotListenerCacheEntry.group_id == group_id,
                BbotListenerCacheEntry.platform == platform,
                BbotListenerCacheEntry.external_id == external_id,
            )
            .limit(1)
        )
        entry = self.session.execute(stmt).scalar_one_or_none()
        if entry is None:
            entry = BbotListenerCacheEntry(
                group_id=group_id,
                platform=platform,
                external_id=external_id,
            )
        entry.canonical_name = canonical_name
        entry.aliases_json = aliases
        entry.source = source
        entry.updated_at = updated_at
        self.session.add(entry)
        return entry

    def find_best_match(self, *, group_id: int, platform: str, query: str) -> BbotListenerCacheEntry | None:
        stmt = (
            select(BbotListenerCacheEntry)
            .where(
                BbotListenerCacheEntry.group_id == group_id,
                BbotListenerCacheEntry.platform == platform,
            )
            .order_by(BbotListenerCacheEntry.updated_at.desc(), BbotListenerCacheEntry.id.desc())
        )
        normalized_query = self._normalize(query)
        if not normalized_query:
            return None

        best_entry = None
        best_score = -1
        for entry in self.session.scalars(stmt):
            names = [str(entry.canonical_name or "")] + [str(alias) for alias in (entry.aliases_json or [])]
            normalized_candidates = [candidate for candidate in (self._normalize(name) for name in names) if candidate]
            score = self._score_candidates(normalized_query, normalized_candidates)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_score <= 0:
            return None
        return best_entry

    def _score_candidates(self, query: str, candidates: list[str]) -> int:
        score = 0
        for candidate in candidates:
            if query == candidate:
                score = max(score, 100)
            elif query in candidate:
                score = max(score, 80)
            elif candidate in query:
                score = max(score, 60)
        return score

    def _normalize(self, value: str) -> str:
        lowered = value.strip().lower()
        return "".join(character for character in lowered if character.isalnum() or "\u4e00" <= character <= "\u9fff")


class SummaryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_summary(
        self,
        *,
        scope_type: str,
        scope_id: str,
        summary_level: str,
        start_at: datetime,
        end_at: datetime,
        content: str,
        source_count: int,
    ) -> Summary:
        summary = Summary(
            scope_type=scope_type,
            scope_id=scope_id,
            summary_level=summary_level,
            start_at=start_at,
            end_at=end_at,
            content=content,
            source_count=source_count,
        )
        self.session.add(summary)
        return summary

    def upsert_summary(
        self,
        *,
        scope_type: str,
        scope_id: str,
        summary_level: str,
        summary_key: str,
        start_at: datetime,
        end_at: datetime,
        content: str,
        source_count: int,
        source_start_msg_id: str | None = None,
        source_end_msg_id: str | None = None,
        source_summary_ids: list[int] | None = None,
        status: str = "active",
    ) -> Summary:
        """Idempotently replace a recursive summary identified by its stable key."""
        if not summary_key.strip():
            raise ValueError("summary_key is required for upsert_summary")
        stmt = select(Summary).where(
            Summary.scope_type == scope_type,
            Summary.scope_id == scope_id,
            Summary.summary_level == summary_level,
            Summary.summary_key == summary_key,
        )
        summary = self.session.scalars(stmt).first()
        if summary is None:
            summary = Summary(
                scope_type=scope_type,
                scope_id=scope_id,
                summary_level=summary_level,
                summary_key=summary_key,
            )
            self.session.add(summary)
        summary.start_at = start_at
        summary.end_at = end_at
        summary.content = content
        summary.source_count = source_count
        summary.source_start_msg_id = source_start_msg_id
        summary.source_end_msg_id = source_end_msg_id
        summary.source_summary_ids = list(source_summary_ids or [])
        summary.status = status
        return summary

    def list_recent_group_summaries(self, scope_id: str, limit: int) -> list[str]:
        stmt = (
            select(Summary)
            .where(Summary.scope_type == "group", Summary.scope_id == scope_id)
            .order_by(Summary.end_at.desc(), Summary.id.desc())
            .limit(limit)
        )
        summaries = [summary.content for summary in self.session.scalars(stmt)]
        return list(reversed(summaries))

    def list_group_summaries(
        self,
        *,
        scope_id: str,
        limit: int,
        summary_levels: list[str] | None = None,
        summary_key: str | None = None,
    ) -> list[Summary]:
        if limit <= 0:
            return []
        filters = [Summary.scope_type == "group", Summary.scope_id == scope_id, Summary.status == "active"]
        if summary_levels:
            filters.append(Summary.summary_level.in_(summary_levels))
        if summary_key is not None:
            filters.append(Summary.summary_key == summary_key)
        stmt = select(Summary).where(*filters).order_by(Summary.end_at.desc(), Summary.id.desc()).limit(limit)
        return list(reversed(list(self.session.scalars(stmt))))

class MemoryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_memory(
        self,
        *,
        scope_type: str,
        scope_id: str,
        subject_type: str,
        subject_id: str,
        memory_kind: str,
        content: str,
        importance: int,
        confidence: float,
        source_msg_id: str,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        status: str = "active",
    ) -> MemoryItem:
        memory = MemoryItem(
            scope_type=scope_type,
            scope_id=scope_id,
            subject_type=subject_type,
            subject_id=subject_id,
            memory_kind=memory_kind,
            content=content,
            importance=importance,
            confidence=confidence,
            source_msg_id=source_msg_id,
            valid_from=valid_from,
            valid_until=valid_until,
            expires_at=valid_until,
            status=status,
        )
        self.session.add(memory)
        self.session.flush()
        self._sync_memory_indexes(memory)
        return memory

    def upsert_memory(
        self,
        *,
        scope_type: str,
        scope_id: str,
        subject_type: str,
        subject_id: str,
        memory_kind: str,
        content: str,
        importance: int,
        confidence: float,
        source_msg_id: str,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        status: str = "active",
        supersedes_id: int | None = None,
    ) -> MemoryItem:
        """Idempotently persist one extracted memory while retaining its source message."""
        stmt = select(MemoryItem).where(
            MemoryItem.scope_type == scope_type,
            MemoryItem.scope_id == scope_id,
            MemoryItem.subject_type == subject_type,
            MemoryItem.subject_id == subject_id,
            MemoryItem.memory_kind == memory_kind,
            MemoryItem.content == content,
            MemoryItem.source_msg_id == source_msg_id,
        )
        memory = self.session.scalars(stmt).first()
        if memory is None:
            memory = MemoryItem(
                scope_type=scope_type,
                scope_id=scope_id,
                subject_type=subject_type,
                subject_id=subject_id,
                memory_kind=memory_kind,
                content=content,
                source_msg_id=source_msg_id,
            )
            self.session.add(memory)
        memory.importance = importance
        memory.confidence = confidence
        memory.valid_from = valid_from
        memory.valid_until = valid_until
        memory.expires_at = valid_until
        memory.status = status
        memory.supersedes_id = supersedes_id
        self.session.flush()
        if supersedes_id is not None:
            self.mark_superseded(memory_id=supersedes_id, superseded_by_id=memory.id, valid_until=valid_from)
        self._sync_memory_indexes(memory)
        return memory

    def upsert_canonical_memory(
        self,
        *,
        scope_type: str,
        scope_id: str,
        subject_type: str,
        subject_id: str,
        memory_kind: str,
        canonical_key: str,
        predicate: str,
        object_text: str,
        content: str,
        importance: int,
        confidence: float,
        source_msg_ids: list[str],
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        replace_previous: bool = False,
    ) -> MemoryItem:
        """Merge repeated evidence into one compact fact and keep its provenance."""
        normalized_sources = list(dict.fromkeys(str(item).strip() for item in source_msg_ids if str(item).strip()))
        if not canonical_key.strip():
            raise ValueError("canonical_key is required")
        memory = self.session.scalars(
            select(MemoryItem).where(
                MemoryItem.scope_type == scope_type,
                MemoryItem.scope_id == scope_id,
                MemoryItem.canonical_key == canonical_key,
                MemoryItem.status == "active",
            )
        ).first()
        previous_content: str | None = str(memory.content) if memory is not None else None
        if memory is None:
            legacy_memory = None
            if normalized_sources:
                legacy_memory = self.session.scalars(
                    select(MemoryItem).where(
                        MemoryItem.scope_type == scope_type,
                        MemoryItem.scope_id == scope_id,
                        MemoryItem.subject_id == subject_id,
                        MemoryItem.memory_kind == memory_kind,
                        MemoryItem.status == "active",
                        MemoryItem.canonical_key == "",
                        MemoryItem.source_msg_id.in_(normalized_sources),
                    )
                ).first()
            primary_source = normalized_sources[0] if normalized_sources else f"canonical:{canonical_key}"
            if legacy_memory is not None:
                memory = legacy_memory
                previous_content = str(memory.content)
                memory.canonical_key = canonical_key
                memory.predicate = predicate
                memory.object_text = object_text
                memory.content = content
                memory.source_msg_ids = normalized_sources
                memory.mention_count = max(1, len(normalized_sources))
            else:
                memory = MemoryItem(
                    scope_type=scope_type,
                    scope_id=scope_id,
                    subject_type=subject_type,
                    subject_id=subject_id,
                    memory_kind=memory_kind,
                    canonical_key=canonical_key,
                    predicate=predicate,
                    object_text=object_text,
                    content=content,
                    source_msg_id=primary_source,
                    source_msg_ids=normalized_sources,
                    mention_count=max(1, len(normalized_sources)),
                    status="active",
                )
                self.session.add(memory)
        else:
            existing_sources = [str(item) for item in (memory.source_msg_ids or []) if str(item).strip()]
            merged_sources = list(dict.fromkeys([*existing_sources, *normalized_sources]))
            memory.source_msg_ids = merged_sources
            memory.mention_count = max(int(memory.mention_count or 1), len(merged_sources))
            memory.content = content
            memory.predicate = predicate
            memory.object_text = object_text
        memory.importance = max(int(memory.importance or 1), int(importance))
        memory.confidence = max(float(memory.confidence or 0.0), float(confidence))
        memory.valid_from = memory.valid_from or valid_from
        memory.valid_until = valid_until
        memory.expires_at = valid_until
        memory.last_seen_at = valid_from or shanghai_now_naive()
        self.session.flush()
        if previous_content is not None and previous_content != content:
            _deactivate_memory_retrieval_documents(
                self.session,
                memory_id=memory.id,
            )

        legacy_candidates = list(
            self.session.scalars(
                select(MemoryItem).where(
                    MemoryItem.scope_type == scope_type,
                    MemoryItem.scope_id == scope_id,
                    MemoryItem.subject_id == subject_id,
                    MemoryItem.memory_kind == memory_kind,
                    MemoryItem.status == "active",
                    MemoryItem.canonical_key == "",
                    MemoryItem.id != memory.id,
                )
            )
        )
        normalized_object = str(object_text or "").strip().casefold()
        normalized_content = " ".join(str(content or "").casefold().split())
        for duplicate in legacy_candidates:
            duplicate_content = " ".join(str(duplicate.content or "").casefold().split())
            same_source = duplicate.source_msg_id in normalized_sources
            same_object = len(normalized_object) >= 2 and normalized_object in duplicate_content
            same_content = bool(normalized_content) and duplicate_content == normalized_content
            if same_source or same_object or same_content:
                self.mark_superseded(
                    memory_id=duplicate.id,
                    superseded_by_id=memory.id,
                    valid_until=valid_from,
                )

        if replace_previous and predicate.strip():
            previous = list(
                self.session.scalars(
                    select(MemoryItem).where(
                        MemoryItem.scope_type == scope_type,
                        MemoryItem.scope_id == scope_id,
                        MemoryItem.subject_id == subject_id,
                        MemoryItem.predicate == predicate,
                        MemoryItem.status == "active",
                        MemoryItem.id != memory.id,
                    )
                )
            )
            for older in previous:
                self.mark_superseded(
                    memory_id=older.id,
                    superseded_by_id=memory.id,
                    valid_until=valid_from,
                )
        self._sync_memory_indexes(memory)
        return memory

    def mark_superseded(
        self,
        *,
        memory_id: int,
        superseded_by_id: int | None = None,
        valid_until: datetime | None = None,
    ) -> MemoryItem | None:
        memory = self.session.get(MemoryItem, memory_id)
        if memory is None:
            return None
        memory.status = "superseded"
        memory.superseded_by_id = superseded_by_id
        if valid_until is not None:
            memory.valid_until = valid_until
            memory.expires_at = valid_until
        self._sync_memory_indexes(memory)
        _deactivate_memory_retrieval_documents(
            self.session,
            memory_id=memory.id,
        )
        return memory

    def find_unique_correction_candidate(
        self,
        *,
        scope_id: str,
        predicate: str,
        object_text: str,
        replacement_memory_id: int,
        as_of: datetime,
        subject_id: str | None = None,
    ) -> MemoryItem | None:
        instant = _normalize_utc_sqlite_timestamp(as_of)
        memory_kind = "preference" if predicate == "likes" else "taboo" if predicate == "dislikes" else ""
        if not memory_kind:
            return None
        filters = [
            MemoryItem.scope_type == "group",
            MemoryItem.scope_id == scope_id,
            MemoryItem.subject_type == "user",
            MemoryItem.memory_kind == memory_kind,
            or_(MemoryItem.predicate == predicate, MemoryItem.predicate == ""),
            MemoryItem.status == "active",
            MemoryItem.id != int(replacement_memory_id),
            or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
            or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
        ]
        if subject_id is not None:
            filters.append(MemoryItem.subject_id == subject_id)
        rows = list(self.session.scalars(select(MemoryItem).where(*filters)))
        target = " ".join(str(object_text or "").casefold().split())
        if target:
            rows = [
                row
                for row in rows
                if self.correction_objects_are_related(
                    target,
                    row.object_text if str(row.object_text or "").strip() else row.content,
                )
            ]
        return rows[0] if len(rows) == 1 else None

    @staticmethod
    def correction_objects_are_related(target: str, candidate: str) -> bool:
        normalized = " ".join(str(candidate or "").casefold().split())
        if not target or not normalized:
            return False
        if target == normalized:
            return True
        return min(len(target), len(normalized)) >= 4 and (
            target in normalized or normalized in target
        )

    def supersede_current_memories(
        self,
        *,
        scope_id: str,
        subject_id: str,
        predicate: str,
        valid_until: datetime | None,
        object_text: str = "",
    ) -> int:
        rows = list(
            self.session.scalars(
                select(MemoryItem).where(
                    MemoryItem.scope_type == "group",
                    MemoryItem.scope_id == scope_id,
                    MemoryItem.subject_id == subject_id,
                    MemoryItem.predicate == predicate,
                    MemoryItem.status == "active",
                )
            )
        )
        normalized_object = str(object_text or "").strip().casefold()
        if normalized_object:
            rows = [
                memory
                for memory in rows
                if str(memory.object_text or "").strip().casefold() == normalized_object
            ]
        for memory in rows:
            self.mark_superseded(memory_id=memory.id, valid_until=valid_until)
        return len(rows)

    def list_group_memories(self, scope_id: str, limit: int) -> list[MemoryItem]:
        stmt = (
            select(MemoryItem)
            .where(MemoryItem.scope_type == "group", MemoryItem.scope_id == scope_id)
            .order_by(MemoryItem.importance.desc(), MemoryItem.id.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def list_group_memories_by_source_msg_ids(
        self,
        *,
        scope_id: str,
        source_msg_ids: list[str],
    ) -> list[MemoryItem]:
        identifiers = {
            str(item).strip()
            for item in source_msg_ids
            if str(item).strip()
        }
        if not identifiers:
            return []
        rows = self.session.scalars(
            select(MemoryItem)
            .where(
                MemoryItem.scope_type == "group",
                MemoryItem.scope_id == scope_id,
            )
            .order_by(MemoryItem.id)
        )
        return [
            memory
            for memory in rows
            if str(memory.source_msg_id) in identifiers
            or identifiers.intersection(
                str(item) for item in (memory.source_msg_ids or [])
            )
        ]

    def list_current_group_memories(
        self,
        *,
        scope_id: str,
        limit: int,
        as_of: datetime | None = None,
        subject_id: str | None = None,
    ) -> list[MemoryItem]:
        if limit <= 0:
            return []
        instant = _normalize_utc_sqlite_timestamp(as_of or datetime.now(UTC))
        filters = [
            MemoryItem.scope_type == "group",
            MemoryItem.scope_id == scope_id,
            MemoryItem.status == "active",
            or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
            or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
        ]
        if subject_id is not None:
            filters.append(MemoryItem.subject_id == subject_id)
        stmt = (
            select(MemoryItem)
            .where(*filters)
            .order_by(MemoryItem.importance.desc(), MemoryItem.confidence.desc(), MemoryItem.id.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def search_group_memories_fts(
        self,
        *,
        scope_id: str,
        query: str,
        limit: int,
        as_of: datetime | None = None,
        subject_ids: Sequence[str] | None = None,
    ) -> list[MemoryItem]:
        """Return group-scoped lexical candidates with FTS5 as an accelerator.

        The SQL fallback is intentional: short Chinese terms cannot be indexed
        by every FTS tokenizer, and missing a source-backed memory is worse
        than spending a bounded query on the source-of-truth table.
        """
        if limit <= 0:
            return []
        normalized_subject_ids = tuple(
            dict.fromkeys(
                str(subject_id).strip()
                for subject_id in (subject_ids or ())
                if str(subject_id).strip()
            )
        )
        if subject_ids is not None and not normalized_subject_ids:
            return []
        terms = _fts_search_terms(query)
        if not terms:
            return []
        # OR candidates tolerate natural-language Chinese questions, whose full
        # token sequence rarely appears verbatim in a stored atomic memory.
        fts_terms = [term for term in terms if len(term) >= 3]
        ids: list[int] = []
        if fts_terms:
            match_query = " OR ".join(f'"{term}"' for term in fts_terms)
            try:
                rows = self.session.execute(
                    text(
                        "SELECT memory_id FROM memory_items_fts "
                        "WHERE memory_items_fts MATCH :query AND scope_type = 'group' AND scope_id = :scope_id "
                        "ORDER BY bm25(memory_items_fts) LIMIT :limit"
                    ),
                    {"query": match_query, "scope_id": scope_id, "limit": limit},
                )
                ids = [int(row[0]) for row in rows]
            except (SQLAlchemyError, ValueError):
                ids = []
        instant = _normalize_utc_sqlite_timestamp(as_of or datetime.now(UTC))
        active_filters = [
            MemoryItem.scope_type == "group",
            MemoryItem.scope_id == scope_id,
            MemoryItem.status == "active",
            or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
            or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
        ]
        if subject_ids is not None:
            active_filters.append(MemoryItem.subject_id.in_(normalized_subject_ids))
        memories = self.session.scalars(
            select(MemoryItem).where(
                MemoryItem.id.in_(ids),
                *active_filters,
            )
        ).all()
        by_id = {memory.id: memory for memory in memories}
        ordered = [by_id[memory_id] for memory_id in ids if memory_id in by_id]
        fallback_matches = self.session.scalars(
            select(MemoryItem)
            .where(*active_filters, or_(*(MemoryItem.content.ilike(f"%{term}%") for term in terms)))
            .order_by(MemoryItem.importance.desc(), MemoryItem.confidence.desc(), MemoryItem.id.desc())
            .limit(max(32, limit * 4))
        ).all()
        fallback_matches.sort(
            key=lambda memory: (
                sum(1 for term in terms if term in memory.content.lower()),
                memory.importance,
                memory.confidence,
                memory.id,
            ),
            reverse=True,
        )
        seen = {memory.id for memory in ordered}
        ordered.extend(memory for memory in fallback_matches if memory.id not in seen)
        return ordered[:limit]

    def search_group_memories_vector(
        self,
        *,
        scope_id: str,
        query: str,
        limit: int,
        as_of: datetime | None = None,
    ) -> list[MemoryItem]:
        if limit <= 0 or not str(query or "").strip():
            return []
        instant = _normalize_utc_sqlite_timestamp(as_of or datetime.now(UTC))
        group_ids = list(
            self.session.scalars(
                select(MemoryItem.id).where(
                    MemoryItem.scope_type == "group",
                    MemoryItem.scope_id == scope_id,
                    MemoryItem.status == "active",
                    or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
                    or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
                )
            )
        )
        if not group_ids:
            return []
        id_filter = ",".join(str(int(memory_id)) for memory_id in group_ids)
        try:
            rows = self.session.execute(
                text(
                    "SELECT memory_id, vec_distance_cosine(embedding, :embedding) AS distance "
                    f"FROM memory_items_vec WHERE memory_id IN ({id_filter}) "
                    "ORDER BY distance LIMIT :limit"
                ),
                {
                    "embedding": json.dumps(hashed_text_embedding(query)),
                    "limit": max(1, min(len(group_ids), int(limit * 3))),
                },
            )
            ids = [int(row[0]) for row in rows]
        except (SQLAlchemyError, ValueError):
            return []
        if not ids:
            return []
        memories = self.session.scalars(
            select(MemoryItem).where(
                MemoryItem.id.in_(ids),
                MemoryItem.scope_type == "group",
                MemoryItem.scope_id == scope_id,
                MemoryItem.status == "active",
                or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
                or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
            )
        ).all()
        by_id = {memory.id: memory for memory in memories}
        return [by_id[memory_id] for memory_id in ids if memory_id in by_id][:limit]

    def find_current_memory_for_supersession(
        self,
        *,
        scope_id: str,
        subject_type: str,
        subject_id: str,
        memory_kind: str,
        replacement_content: str,
        as_of: datetime | None = None,
    ) -> MemoryItem | None:
        candidates = [
            memory
            for memory in self.list_current_group_memories(
                scope_id=scope_id,
                subject_id=subject_id,
                as_of=as_of,
                limit=20,
            )
            if memory.subject_type == subject_type and memory.memory_kind == memory_kind
        ]
        if not candidates:
            return None
        ignored_terms = {
            "计划", "取消", "决定", "改变", "现在", "之前", "打算", "不打算", "算了",
            "plan", "cancel", "decision", "decided", "planning",
        }
        replacement_body = re.sub(r"^[^:\uff1a]{1,80}[:\uff1a]\s*", "", replacement_content).strip()
        terms = [term for term in _fts_search_terms(replacement_body) if term not in ignored_terms]
        ranked = [
            (
                sum(
                    1
                    for term in terms
                    if term in re.sub(r"^[^:\uff1a]{1,80}[:\uff1a]\s*", "", memory.content).lower()
                ),
                memory.id,
                memory,
            )
            for memory in candidates
        ]
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if not ranked or ranked[0][0] <= 0:
            return None
        if len(ranked) > 1 and ranked[1][0] == ranked[0][0]:
            return None
        return ranked[0][2]

    def list_group_memories_for_subject(self, *, scope_id: str, subject_id: str, limit: int) -> list[MemoryItem]:
        instant = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        stmt = (
            select(MemoryItem)
            .where(
                MemoryItem.scope_type == "group",
                MemoryItem.scope_id == scope_id,
                MemoryItem.subject_id == subject_id,
                MemoryItem.status == "active",
                or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
                or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
            )
            .order_by(MemoryItem.importance.desc(), MemoryItem.id.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def find_active_memory_by_content(
        self,
        *,
        scope_id: str,
        subject_id: str,
        memory_kind: str,
        content: str,
    ) -> MemoryItem | None:
        return self.session.scalars(
            select(MemoryItem)
            .where(
                MemoryItem.scope_type == "group",
                MemoryItem.scope_id == scope_id,
                MemoryItem.subject_id == subject_id,
                MemoryItem.memory_kind == memory_kind,
                MemoryItem.content == content,
                MemoryItem.status == "active",
            )
            .order_by(MemoryItem.id.desc())
            .limit(1)
        ).first()

    def upsert_memory_item_semantic_vectors(
        self,
        rows: Sequence[dict[str, Any]],
    ) -> int:
        now = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        count = 0
        for row in rows:
            memory_id = int(row["memory_id"])
            existing = self.session.get(MemoryItemSemanticVector, memory_id)
            if existing is None:
                existing = MemoryItemSemanticVector(
                    memory_id=memory_id,
                    group_id=int(row["group_id"]),
                )
                self.session.add(existing)
            existing.group_id = int(row["group_id"])
            existing.provider = str(row.get("provider") or "")
            existing.model = str(row.get("model") or "")
            existing.dimensions = int(row.get("dimensions") or 0)
            existing.version = str(row.get("version") or "")
            existing.vector_json = str(row.get("vector_json") or "")
            existing.updated_at = now
            count += 1
        self.session.flush()
        return count

    def load_memory_item_semantic_vectors(
        self,
        memory_ids: Sequence[int],
        *,
        provider: str = "",
        model: str = "",
        dimensions: int = 0,
        version: str = "",
    ) -> dict[int, list[float]]:
        normalized_ids = tuple(
            dict.fromkeys(int(memory_id) for memory_id in memory_ids if memory_id)
        )
        if not normalized_ids:
            return {}
        rows = self.session.scalars(
            select(MemoryItemSemanticVector).where(
                MemoryItemSemanticVector.memory_id.in_(normalized_ids)
            )
        ).all()
        result: dict[int, list[float]] = {}
        for row in rows:
            if provider and row.provider != provider:
                continue
            if model and row.model != model:
                continue
            if dimensions and row.dimensions != dimensions:
                continue
            if version and row.version != version:
                continue
            try:
                vector = json.loads(row.vector_json or "[]")
            except ValueError:
                continue
            if (
                isinstance(vector, list)
                and vector
                and all(isinstance(value, (int, float)) for value in vector)
            ):
                result[int(row.memory_id)] = [float(value) for value in vector]
        return result

    def delete_memory_item_semantic_vectors(
        self,
        memory_ids: Sequence[int],
    ) -> int:
        normalized = [int(memory_id) for memory_id in memory_ids if memory_id]
        if not normalized:
            return 0
        rows = self.session.scalars(
            select(MemoryItemSemanticVector).where(
                MemoryItemSemanticVector.memory_id.in_(normalized)
            )
        ).all()
        for row in rows:
            self.session.delete(row)
        self.session.flush()
        return len(rows)

    def list_active_memory_items_for_indexing(
        self,
        *,
        after_id: int = 0,
        limit: int = 500,
        scope_id: str | None = None,
    ) -> list[MemoryItem]:
        filters = [
            MemoryItem.status == "active",
            MemoryItem.id > int(after_id),
        ]
        if scope_id is not None:
            filters.append(MemoryItem.scope_id == scope_id)
        return list(
            self.session.scalars(
                select(MemoryItem)
                .where(*filters)
                .order_by(MemoryItem.id.asc())
                .limit(max(1, int(limit)))
            )
        )

    def deactivate_memory_items(
        self,
        memory_ids: Sequence[int],
        *,
        valid_until: datetime,
    ) -> int:
        normalized = [int(memory_id) for memory_id in memory_ids if memory_id]
        if not normalized:
            return 0
        rows = self.session.scalars(
            select(MemoryItem).where(
                MemoryItem.id.in_(normalized),
                MemoryItem.status == "active",
            )
        ).all()
        for row in rows:
            row.status = "inactive"
            row.valid_until = valid_until
            row.expires_at = valid_until
            _deactivate_memory_retrieval_documents(
                self.session,
                memory_id=int(row.id),
            )
        self.session.flush()
        return len(rows)

    def _sync_memory_indexes(self, memory: MemoryItem) -> None:
        self._sync_fts(memory)
        self._sync_vector(memory)

    def _sync_fts(self, memory: MemoryItem) -> None:
        try:
            self.session.execute(text("DELETE FROM memory_items_fts WHERE memory_id = :memory_id"), {"memory_id": str(memory.id)})
            if memory.status != "active":
                return
            self.session.execute(
                text(
                    "INSERT INTO memory_items_fts (content, scope_type, scope_id, memory_id) "
                    "VALUES (:content, :scope_type, :scope_id, :memory_id)"
                ),
                {
                    "content": memory.content,
                    "scope_type": memory.scope_type,
                    "scope_id": memory.scope_id,
                    "memory_id": str(memory.id),
                },
            )
        except SQLAlchemyError:
            # FTS5 is optional and must not make source-of-truth writes fail.
            return

    def _sync_vector(self, memory: MemoryItem) -> None:
        try:
            self.session.execute(text("DELETE FROM memory_items_vec WHERE memory_id = :memory_id"), {"memory_id": memory.id})
            if memory.status != "active":
                return
            self.session.execute(
                text("INSERT INTO memory_items_vec(memory_id, embedding) VALUES (:memory_id, :embedding)"),
                {
                    "memory_id": memory.id,
                    "embedding": json.dumps(hashed_text_embedding(memory.content)),
                },
            )
        except SQLAlchemyError:
            return


def _fts_search_terms(query: str) -> list[str]:
    """Derive FTS-safe lexical candidates for both Latin and Chinese queries."""
    normalized = str(query or "").lower().replace('"', " ")
    terms: list[str] = []
    for chinese_run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        terms.append(chinese_run)
        terms.extend(chinese_run[index : index + 2] for index in range(len(chinese_run) - 1))
        terms.extend(chinese_run[index : index + 3] for index in range(len(chinese_run) - 2))
    terms.extend(re.findall(r"[a-z0-9_]{2,}", normalized))
    return list(dict.fromkeys(term for term in terms if term))[:16]


class EpisodeRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_episode(
        self,
        *,
        group_id: int,
        start_message_id: int,
        started_at: datetime,
        segmentation_version: str,
        status: str = "open",
        boundary_reason: str = "",
    ) -> ConversationEpisode:
        episode = ConversationEpisode(
            group_id=group_id,
            segmentation_version=segmentation_version,
            status=status,
            is_current=True,
            start_message_id=start_message_id,
            end_message_id=None,
            started_at=_normalize_utc_sqlite_timestamp(started_at),
            ended_at=None,
            boundary_reason=boundary_reason,
            message_count=0,
            token_count=0,
        )
        self.session.add(episode)
        return episode

    def get_open_episode(self, *, group_id: int) -> ConversationEpisode | None:
        return self.session.scalars(
            select(ConversationEpisode)
            .where(
                ConversationEpisode.group_id == group_id,
                ConversationEpisode.status == "open",
                ConversationEpisode.is_current.is_(True),
            )
            .order_by(ConversationEpisode.id.desc())
            .limit(1)
        ).first()

    def get_episode(self, episode_id: int) -> ConversationEpisode | None:
        return self.session.get(ConversationEpisode, episode_id)

    def list_unassigned_messages(
        self,
        *,
        group_id: int | None = None,
        after_message_id: int | None = None,
        watermark_message_id: int | None = None,
        limit: int = 500,
    ) -> list[Message]:
        stmt = (
            select(Message)
            .outerjoin(
                EpisodeMessage,
                (EpisodeMessage.message_id == Message.id)
                & (EpisodeMessage.group_id == Message.group_id),
            )
            .where(
                Message.group_id.is_not(None),
                EpisodeMessage.message_id.is_(None),
                text(
                    "(json_extract(messages.raw_json, '$.delivery_state') IS NULL "
                    "OR json_extract(messages.raw_json, '$.delivery_state') "
                    "NOT IN ('reserved', 'uncertain'))"
                ),
            )
        )
        if group_id is not None:
            stmt = stmt.where(Message.group_id == int(group_id))
        if after_message_id is not None:
            stmt = stmt.where(Message.id > int(after_message_id))
        if watermark_message_id is not None:
            stmt = stmt.where(Message.id <= int(watermark_message_id))
        return list(
            self.session.scalars(
                stmt.order_by(
                    Message.group_id.asc(),
                    Message.timestamp.asc(),
                    Message.id.asc(),
                ).limit(max(1, int(limit)))
            )
        )

    def list_idle_open_episodes(
        self,
        *,
        idle_before: datetime,
        group_id: int | None = None,
        limit: int = 100,
    ) -> list[ConversationEpisode]:
        normalized_idle_before = _normalize_utc_sqlite_timestamp(idle_before)
        last_message_at = (
            select(func.max(Message.timestamp))
            .join(
                EpisodeMessage,
                (EpisodeMessage.message_id == Message.id)
                & (EpisodeMessage.group_id == Message.group_id),
            )
            .where(
                EpisodeMessage.episode_id == ConversationEpisode.id,
                EpisodeMessage.group_id == ConversationEpisode.group_id,
            )
            .correlate(ConversationEpisode)
            .scalar_subquery()
        )
        stmt = select(ConversationEpisode).where(
            ConversationEpisode.status == "open",
            ConversationEpisode.is_current.is_(True),
            func.coalesce(last_message_at, ConversationEpisode.started_at)
            <= normalized_idle_before,
        )
        if group_id is not None:
            stmt = stmt.where(ConversationEpisode.group_id == int(group_id))
        return list(
            self.session.scalars(
                stmt.order_by(
                    ConversationEpisode.group_id.asc(),
                    ConversationEpisode.started_at.asc(),
                    ConversationEpisode.id.asc(),
                ).limit(max(1, int(limit)))
            )
        )

    def list_processable_episodes(
        self,
        *,
        group_id: int | None = None,
        statuses: tuple[str, ...] = ("closed", "failed"),
        compaction_version: str | None = None,
        limit: int = 100,
    ) -> list[ConversationEpisode]:
        normalized_statuses = tuple(dict.fromkeys(str(value) for value in statuses))
        if not normalized_statuses:
            return []
        stmt = select(ConversationEpisode).where(
            ConversationEpisode.is_current.is_(True),
            ConversationEpisode.status.in_(normalized_statuses),
        )
        if group_id is not None:
            stmt = stmt.where(ConversationEpisode.group_id == int(group_id))
        if compaction_version is not None:
            stmt = stmt.where(
                ConversationEpisode.compaction_version == str(compaction_version)
            )
        return list(
            self.session.scalars(
                stmt.order_by(
                    ConversationEpisode.group_id.asc(),
                    ConversationEpisode.ended_at.asc(),
                    ConversationEpisode.id.asc(),
                ).limit(max(1, int(limit)))
            )
        )

    def compare_and_set_status(
        self,
        *,
        episode_id: int,
        group_id: int,
        expected_statuses: tuple[str, ...],
        new_status: str,
        compaction_version: str | None = None,
    ) -> bool:
        normalized_statuses = tuple(
            dict.fromkeys(str(value) for value in expected_statuses)
        )
        if not normalized_statuses:
            return False
        values: dict[str, Any] = {
            "new_status": str(new_status),
            "updated_at": _normalize_utc_sqlite_timestamp(datetime.now(UTC)),
            "episode_id": int(episode_id),
            "group_id": int(group_id),
        }
        assignments = "status = :new_status, updated_at = :updated_at"
        if compaction_version is not None:
            assignments += ", compaction_version = :compaction_version"
            values["compaction_version"] = str(compaction_version)
        status_params: list[str] = []
        for index, status in enumerate(normalized_statuses):
            key = f"expected_status_{index}"
            values[key] = status
            status_params.append(f":{key}")
        result = self.session.execute(
            text(
                f"UPDATE conversation_episodes SET {assignments} "
                "WHERE id = :episode_id AND group_id = :group_id "
                f"AND status IN ({','.join(status_params)}) AND is_current = 1"
            ),
            values,
        )
        self.session.expire_all()
        return int(result.rowcount or 0) == 1

    def find_episode_for_late_arrival(
        self,
        *,
        group_id: int,
        timestamp: datetime,
        segmentation_version: str | None = None,
    ) -> ConversationEpisode | None:
        resolved_timestamp = _normalize_utc_sqlite_timestamp(timestamp)
        stmt = select(ConversationEpisode).where(
            ConversationEpisode.group_id == int(group_id),
            ConversationEpisode.is_current.is_(True),
            ConversationEpisode.started_at <= resolved_timestamp,
            or_(
                ConversationEpisode.ended_at.is_(None),
                ConversationEpisode.ended_at >= resolved_timestamp,
            ),
        )
        if segmentation_version is not None:
            normalized_generation = str(segmentation_version)
            stmt = stmt.where(
                or_(
                    ConversationEpisode.segmentation_version
                    == normalized_generation,
                    ConversationEpisode.segmentation_version.like(
                        f"{normalized_generation}:late:%"
                    ),
                )
            )
        containing = self.session.scalars(
            stmt.order_by(
                ConversationEpisode.started_at.desc(),
                ConversationEpisode.id.desc(),
            ).limit(1)
        ).first()
        if containing is not None:
            return containing
        future_stmt = select(ConversationEpisode).where(
            ConversationEpisode.group_id == int(group_id),
            ConversationEpisode.is_current.is_(True),
            ConversationEpisode.started_at > resolved_timestamp,
        )
        if segmentation_version is not None:
            normalized_generation = str(segmentation_version)
            future_stmt = future_stmt.where(
                or_(
                    ConversationEpisode.segmentation_version
                    == normalized_generation,
                    ConversationEpisode.segmentation_version.like(
                        f"{normalized_generation}:late:%"
                    ),
                )
            )
        return self.session.scalars(
            future_stmt.order_by(
                ConversationEpisode.started_at.asc(),
                ConversationEpisode.id.asc(),
            ).limit(1)
        ).first()

    def supersede_episode(
        self,
        *,
        episode_id: int,
        group_id: int,
        expected_current: bool = True,
    ) -> bool:
        result = self.session.execute(
            text(
                "UPDATE conversation_episodes SET status = 'superseded', "
                "is_current = 0, updated_at = :updated_at "
                "WHERE id = :episode_id AND group_id = :group_id "
                "AND is_current = :expected_current"
            ),
            {
                "episode_id": int(episode_id),
                "group_id": int(group_id),
                "expected_current": bool(expected_current),
                "updated_at": _normalize_utc_sqlite_timestamp(datetime.now(UTC)),
            },
        )
        self.session.expire_all()
        return int(result.rowcount or 0) == 1

    def prepare_late_arrival_resegment(
        self,
        *,
        group_id: int,
        message_id: int,
        timestamp: datetime,
        segmentation_version: str | None = None,
        compaction_version: str | None = None,
    ) -> list[int]:
        """Atomically supersede the affected suffix and release its memberships."""

        resolved_segmentation = str(segmentation_version or "")
        if not resolved_segmentation:
            raise ValueError("late-arrival preparation requires a segmentation generation")
        preparation_id = self.session.execute(
            text(
                "INSERT INTO memory_late_arrival_preparations ("
                "group_id, message_id, segmentation_generation, "
                "compaction_generation, created_at"
                ") VALUES ("
                ":group_id, :message_id, :segmentation_generation, "
                ":compaction_generation, :created_at"
                ") ON CONFLICT(group_id, message_id, segmentation_generation) "
                "DO NOTHING RETURNING id"
            ),
            {
                "group_id": int(group_id),
                "message_id": int(message_id),
                "segmentation_generation": resolved_segmentation,
                "compaction_generation": str(compaction_version or ""),
                "created_at": _normalize_utc_sqlite_timestamp(datetime.now(UTC)),
            },
        ).scalar_one_or_none()
        if preparation_id is None:
            return []
        affected = self.find_episode_for_late_arrival(
            group_id=group_id,
            timestamp=timestamp,
            segmentation_version=segmentation_version,
        )
        if affected is None:
            return [int(message_id)]
        suffix_ids = list(
            self.session.scalars(
                select(ConversationEpisode.id)
                .where(
                    ConversationEpisode.group_id == int(group_id),
                    ConversationEpisode.is_current.is_(True),
                    ConversationEpisode.started_at >= affected.started_at,
                )
                .order_by(
                    ConversationEpisode.started_at.asc(),
                    ConversationEpisode.id.asc(),
                )
            )
        )
        if not suffix_ids:
            return [int(message_id)]
        replay_message_ids = list(
            self.session.scalars(
                select(Message.id)
                .join(
                    EpisodeMessage,
                    (EpisodeMessage.message_id == Message.id)
                    & (EpisodeMessage.group_id == Message.group_id),
                )
                .where(
                    EpisodeMessage.group_id == int(group_id),
                    EpisodeMessage.episode_id.in_(suffix_ids),
                    Message.group_id == int(group_id),
                )
                .order_by(Message.timestamp.asc(), Message.id.asc())
            )
        )
        values: dict[str, Any] = {
            "updated_at": _normalize_utc_sqlite_timestamp(datetime.now(UTC)),
            "group_id": int(group_id),
            "episode_ids": tuple(int(value) for value in suffix_ids),
        }
        compaction_assignment = ""
        if compaction_version is not None:
            compaction_assignment = ", compaction_version = :compaction_version"
            values["compaction_version"] = str(compaction_version)
        superseded = self.session.execute(
            text(
                "UPDATE conversation_episodes SET status = 'superseded', "
                f"is_current = 0, updated_at = :updated_at{compaction_assignment} "
                "WHERE group_id = :group_id AND is_current = 1 "
                "AND id IN :episode_ids"
            ).bindparams(bindparam("episode_ids", expanding=True)),
            values,
        )
        if int(superseded.rowcount or 0) != len(suffix_ids):
            raise RuntimeError("late-arrival episode generation CAS failed")
        document_ids = list(
            self.session.scalars(
                select(RetrievalDocument.id).where(
                    RetrievalDocument.group_id == int(group_id),
                    RetrievalDocument.episode_id.in_(suffix_ids),
                    RetrievalDocument.status == "active",
                )
            )
        )
        if document_ids:
            self.session.execute(
                text(
                    "UPDATE retrieval_documents SET status = 'inactive', "
                    "embedding_status = 'stale', updated_at = :updated_at "
                    "WHERE group_id = :group_id AND id IN :document_ids"
                ).bindparams(bindparam("document_ids", expanding=True)),
                {
                    "group_id": int(group_id),
                    "document_ids": tuple(int(value) for value in document_ids),
                    "updated_at": values["updated_at"],
                },
            )
            try:
                self.session.execute(
                    text(
                        "DELETE FROM retrieval_documents_fts "
                        "WHERE document_id IN :document_ids"
                    ).bindparams(bindparam("document_ids", expanding=True)),
                    {
                        "document_ids": tuple(
                            str(value) for value in document_ids
                        )
                    },
                )
            except SQLAlchemyError:
                pass
            _delete_active_retrieval_vectors(
                self.session,
                document_ids=[int(value) for value in document_ids],
            )
        self.session.execute(
            text(
                "DELETE FROM episode_messages WHERE group_id = :group_id "
                "AND episode_id IN :episode_ids"
            ).bindparams(bindparam("episode_ids", expanding=True)),
            {
                "group_id": int(group_id),
                "episode_ids": tuple(int(value) for value in suffix_ids),
            },
        )
        self.session.expire_all()
        return [int(value) for value in replay_message_ids]

    def add_message(
        self,
        *,
        episode_id: int,
        group_id: int,
        message_id: int,
        ordinal: int,
        estimated_tokens: int,
    ) -> EpisodeMessage:
        membership = EpisodeMessage(
            episode_id=episode_id,
            group_id=group_id,
            message_id=message_id,
            ordinal=ordinal,
        )
        self.session.add(membership)
        episode = self.session.get(ConversationEpisode, episode_id)
        if episode is not None:
            episode.message_count = max(int(episode.message_count or 0), ordinal + 1)
            episode.token_count = int(episode.token_count or 0) + max(0, int(estimated_tokens))
            episode.updated_at = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
            self.session.add(episode)
        return membership

    def add_message_if_current(
        self,
        *,
        episode_id: int,
        group_id: int,
        message_id: int,
        estimated_tokens: int,
    ) -> bool:
        """Atomically append only while the target remains the current open episode."""
        added_at = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        inserted = self.session.execute(
            text(
                "INSERT INTO episode_messages ("
                "episode_id, message_id, group_id, ordinal, added_at"
                ") SELECT id, :message_id, group_id, message_count, :added_at "
                "FROM conversation_episodes "
                "WHERE id = :episode_id AND group_id = :group_id "
                "AND status = 'open' AND is_current = 1 "
                "ON CONFLICT DO NOTHING"
            ),
            {
                "episode_id": int(episode_id),
                "group_id": int(group_id),
                "message_id": int(message_id),
                "added_at": added_at,
            },
        )
        if int(inserted.rowcount or 0) != 1:
            return False
        updated = self.session.execute(
            text(
                "UPDATE conversation_episodes SET "
                "message_count = message_count + 1, "
                "token_count = token_count + :estimated_tokens, "
                "updated_at = :updated_at "
                "WHERE id = :episode_id AND group_id = :group_id "
                "AND status = 'open' AND is_current = 1"
            ),
            {
                "episode_id": int(episode_id),
                "group_id": int(group_id),
                "estimated_tokens": max(0, int(estimated_tokens)),
                "updated_at": added_at,
            },
        )
        if int(updated.rowcount or 0) != 1:
            raise RuntimeError("current episode disappeared after guarded append")
        self.session.expire_all()
        return True

    def close_episode(
        self,
        *,
        episode_id: int,
        ended_at: datetime,
        end_message_id: int,
        boundary_reason: str,
        content_hash: str,
    ) -> ConversationEpisode | None:
        episode = self.session.get(ConversationEpisode, episode_id)
        if (
            episode is None
            or episode.status != "open"
            or not bool(episode.is_current)
        ):
            return None
        closed_at = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        normalized_ended_at = _normalize_utc_sqlite_timestamp(ended_at)
        segmentation_version = str(episode.segmentation_version or "")
        duplicate_id = self.session.scalar(
            select(ConversationEpisode.id).where(
                ConversationEpisode.id != int(episode_id),
                ConversationEpisode.group_id == int(episode.group_id),
                ConversationEpisode.segmentation_version
                == segmentation_version,
                ConversationEpisode.start_message_id
                == int(episode.start_message_id),
                ConversationEpisode.end_message_id == int(end_message_id),
            )
        )
        if duplicate_id is not None:
            # A late-arrival replay can deterministically reproduce an older
            # superseded boundary. Keep both audit rows, but identify the
            # replay as a derived segmentation generation.
            segmentation_version = (
                f"{segmentation_version}:late:{int(episode_id)}"
            )
        result = self.session.execute(
            text(
                "UPDATE conversation_episodes SET "
                "status = 'closed', segmentation_version = :segmentation_version, "
                "ended_at = :ended_at, end_message_id = :end_message_id, "
                "boundary_reason = :boundary_reason, content_hash = :content_hash, "
                "closed_at = :closed_at, updated_at = :updated_at "
                "WHERE id = :episode_id AND group_id = :group_id "
                "AND status = 'open' AND is_current = 1"
            ),
            {
                "segmentation_version": segmentation_version,
                "ended_at": normalized_ended_at,
                "end_message_id": int(end_message_id),
                "boundary_reason": str(boundary_reason),
                "content_hash": str(content_hash),
                "closed_at": closed_at,
                "updated_at": closed_at,
                "episode_id": int(episode_id),
                "group_id": int(episode.group_id),
            },
        )
        self.session.expire_all()
        if int(result.rowcount or 0) != 1:
            return None
        return self.session.get(ConversationEpisode, episode_id)

    def list_episode_messages(self, *, episode_id: int, group_id: int) -> list[Message]:
        return list(
            self.session.scalars(
                select(Message)
                .join(
                    EpisodeMessage,
                    (EpisodeMessage.message_id == Message.id)
                    & (EpisodeMessage.group_id == Message.group_id),
                )
                .where(
                    EpisodeMessage.episode_id == episode_id,
                    EpisodeMessage.group_id == group_id,
                    Message.group_id == group_id,
                )
                .order_by(EpisodeMessage.ordinal.asc(), Message.id.asc())
            )
        )


class RetrievalDocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_document(
        self,
        *,
        scope_type: str,
        scope_id: str,
        group_id: int,
        episode_id: int | None,
        document_kind: str,
        source_table: str,
        source_id: str,
        start_at: datetime,
        end_at: datetime,
        content: str,
        metadata_json: dict[str, Any],
        content_hash: str,
        source_message_ids: list[int],
        status: str = "active",
        embedding_provider: str = "",
        embedding_model: str = "",
        embedding_version: str = "",
        embedding_dimensions: int | None = None,
        embedding_generation: int | None = None,
        embedding_eligible: bool = False,
        embedding_status: str = "disabled",
    ) -> RetrievalDocument:
        memory_source_id: int | None = None
        if document_kind == "memory" and source_table == "memory_items":
            try:
                memory_source_id = int(source_id)
            except ValueError:
                raise ValueError("memory retrieval document source_id must be numeric") from None
            source_memory = self.session.get(MemoryItem, memory_source_id)
            if (
                source_memory is None
                or scope_type != "group"
                or source_memory.scope_type != "group"
                or source_memory.scope_id != str(scope_id)
                or str(scope_id) != str(int(group_id))
            ):
                raise ValueError("memory retrieval document source scope mismatch")
        document = self.session.scalars(
            select(RetrievalDocument).where(
                RetrievalDocument.scope_type == scope_type,
                RetrievalDocument.scope_id == scope_id,
                RetrievalDocument.group_id == group_id,
                RetrievalDocument.document_kind == document_kind,
                RetrievalDocument.source_table == source_table,
                RetrievalDocument.source_id == source_id,
                RetrievalDocument.content_hash == content_hash,
            )
        ).first()
        if document is None:
            conflicting = self.session.scalars(
                select(RetrievalDocument).where(
                    RetrievalDocument.scope_type == scope_type,
                    RetrievalDocument.scope_id == scope_id,
                    RetrievalDocument.document_kind == document_kind,
                    RetrievalDocument.source_table == source_table,
                    RetrievalDocument.source_id == source_id,
                    RetrievalDocument.content_hash == content_hash,
                )
            ).first()
            if conflicting is not None:
                raise ValueError("retrieval document identity is bound to another group")
            document = RetrievalDocument(
                scope_type=scope_type,
                scope_id=scope_id,
                group_id=group_id,
                episode_id=episode_id,
                document_kind=document_kind,
                source_table=source_table,
                source_id=source_id,
                start_at=_normalize_utc_sqlite_timestamp(start_at),
                end_at=_normalize_utc_sqlite_timestamp(end_at),
                content=content,
                metadata_json=metadata_json,
                content_hash=content_hash,
                status=status,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_version=embedding_version,
                embedding_dimensions=embedding_dimensions,
                embedding_generation=embedding_generation,
                embedding_eligible=bool(embedding_eligible),
                embedding_status=embedding_status,
            )
            self.session.add(document)
            self.session.flush()
        else:
            if document.group_id != group_id or document.episode_id != episode_id:
                raise ValueError("retrieval document identity cannot move across group or episode")
            document.status = status
            document.metadata_json = metadata_json
            document.embedding_provider = embedding_provider
            document.embedding_model = embedding_model
            document.embedding_version = embedding_version
            document.embedding_dimensions = embedding_dimensions
            document.embedding_generation = embedding_generation
            document.embedding_eligible = bool(embedding_eligible)
            document.embedding_status = embedding_status
            document.updated_at = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
            self.session.add(document)

        if document.document_kind == "memory" and document.source_table == "memory_items":
            _deactivate_memory_retrieval_documents(
                self.session,
                memory_id=int(memory_source_id),
                keep_document_id=document.id,
            )

        existing_message_ids = set(
            self.session.scalars(
                select(RetrievalDocumentMessage.message_id).where(
                    RetrievalDocumentMessage.document_id == document.id
                )
            )
        )
        for ordinal, message_id in enumerate(dict.fromkeys(source_message_ids)):
            if message_id in existing_message_ids:
                continue
            self.session.add(
                RetrievalDocumentMessage(
                    document_id=document.id,
                    group_id=group_id,
                    message_id=message_id,
                    ordinal=ordinal,
                    role="source",
                )
            )
        self._sync_fts(document)
        return document

    def project_raw_message_v3(
        self,
        *,
        group_id: int,
        message_id: int,
        embedding_generation: int | None = None,
    ) -> RetrievalDocument | None:
        """Idempotently project one canonical message into immediate V3 retrieval.

        The caller owns the transaction, so the canonical document, provenance,
        and best-effort FTS row become visible together. Unsafe outbound
        placeholders revoke any older projection without mutating the message.
        """
        message = self.session.scalars(
            select(Message).where(
                Message.id == int(message_id),
                Message.group_id == int(group_id),
            )
        ).first()
        if message is None:
            raise ValueError("raw message projection source is missing or out of scope")

        existing = list(
            self.session.scalars(
                select(RetrievalDocument).where(
                    RetrievalDocument.group_id == int(group_id),
                    RetrievalDocument.document_kind == "raw_message_v3",
                    RetrievalDocument.source_table == "messages",
                    RetrievalDocument.source_id == str(int(message_id)),
                )
            )
        )
        content = str(message.plain_text or "").strip()
        delivery_state = (
            str(message.raw_json.get("delivery_state") or "")
            .strip()
            .casefold()
            if isinstance(message.raw_json, dict)
            else ""
        )
        unsafe = delivery_state in _INELIGIBLE_DELIVERY_STATES
        if unsafe or not content:
            self._deactivate_documents(existing)
            return None

        generation = (
            int(embedding_generation)
            if embedding_generation is not None
            else None
        )
        hash_input = json.dumps(
            {
                "kind": "raw_message_v3",
                "message_id": int(message.id),
                "group_id": int(message.group_id),
                "platform_msg_id": str(message.platform_msg_id),
                "user_id": int(message.user_id),
                "timestamp": _normalize_utc_sqlite_timestamp(message.timestamp).isoformat(),
                "content": content,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        content_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        matching = next(
            (row for row in existing if row.content_hash == content_hash),
            None,
        )
        if (
            generation is not None
            and matching is not None
            and matching.embedding_generation is not None
            and int(matching.embedding_generation) > generation
        ):
            claimed_state = self.session.execute(
                text(
                    "SELECT status, is_active, document_family "
                    "FROM retrieval_index_state "
                    "WHERE channel = 'vector' AND generation = :generation"
                ),
                {"generation": int(matching.embedding_generation)},
            ).one_or_none()
            requested_state = self.session.execute(
                text(
                    "SELECT is_active, document_family "
                    "FROM retrieval_index_state "
                    "WHERE channel = 'vector' AND generation = :generation"
                ),
                {"generation": generation},
            ).one_or_none()
            if (
                claimed_state is not None
                and requested_state is not None
                and str(claimed_state.document_family) == "raw_message_v3"
                and str(requested_state.document_family) == "raw_message_v3"
                and str(claimed_state.status) in {"building", "ready"}
                and not bool(claimed_state.is_active)
                and bool(requested_state.is_active)
            ):
                # A worker pinned to the older active generation must not
                # erase a newer generation's in-progress document claim.
                generation = int(matching.embedding_generation)
        if generation is None and matching is not None:
            embedding_provider = matching.embedding_provider
            embedding_model = matching.embedding_model
            embedding_version = matching.embedding_version
            embedding_dimensions = matching.embedding_dimensions
            effective_generation = matching.embedding_generation
            embedding_eligible = bool(matching.embedding_eligible)
            embedding_status = matching.embedding_status
        elif (
            matching is not None
            and matching.embedding_generation == generation
            and matching.embedding_status == "ready"
        ):
            embedding_provider = matching.embedding_provider
            embedding_model = matching.embedding_model
            embedding_version = matching.embedding_version
            embedding_dimensions = matching.embedding_dimensions
            effective_generation = generation
            embedding_eligible = True
            embedding_status = "ready"
        else:
            embedding_provider = ""
            embedding_model = ""
            embedding_version = ""
            embedding_dimensions = None
            effective_generation = generation
            embedding_eligible = generation is not None
            embedding_status = "pending" if generation is not None else "disabled"
        document = self.upsert_document(
            scope_type="group",
            scope_id=str(int(group_id)),
            group_id=int(group_id),
            episode_id=None,
            document_kind="raw_message_v3",
            source_table="messages",
            source_id=str(int(message.id)),
            start_at=message.timestamp,
            end_at=message.timestamp,
            content=content,
            metadata_json={
                "platform_msg_id": str(message.platform_msg_id),
                "speaker_id": str(message.user_id),
                "reply_to_msg_id": str(message.reply_to_msg_id or ""),
                "mentioned_bot": bool(message.mentioned_bot),
                "index_generation": effective_generation,
            },
            content_hash=content_hash,
            source_message_ids=[int(message.id)],
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
            embedding_dimensions=embedding_dimensions,
            embedding_generation=effective_generation,
            embedding_eligible=embedding_eligible,
            embedding_status=embedding_status,
        )
        self._deactivate_documents(
            [row for row in existing if row.id != document.id]
        )
        return document

    def list_raw_message_v3_projection_gaps(
        self,
        *,
        projection_job_type: str,
        embedding_generation: int | None,
        limit: int,
        group_ids: Sequence[int] | None = None,
    ) -> list[tuple[int, int]]:
        """Return eligible ledger rows lacking a projection or live repair job."""
        projection_filters = [
            RetrievalDocumentMessage.message_id == Message.id,
            RetrievalDocumentMessage.group_id == Message.group_id,
            RetrievalDocument.id == RetrievalDocumentMessage.document_id,
            RetrievalDocument.group_id == Message.group_id,
            RetrievalDocument.document_kind == "raw_message_v3",
            RetrievalDocument.source_table == "messages",
            RetrievalDocument.source_id == cast(Message.id, String),
            RetrievalDocument.status == "active",
        ]
        if embedding_generation is not None:
            projection_filters.extend(
                [
                    RetrievalDocument.embedding_generation
                    == int(embedding_generation),
                    RetrievalDocument.embedding_eligible.is_(True),
                ]
            )
        matching_projection = (
            select(RetrievalDocument.id)
            .select_from(RetrievalDocumentMessage)
            .join(
                RetrievalDocument,
                RetrievalDocument.id == RetrievalDocumentMessage.document_id,
            )
            .where(*projection_filters)
            .correlate(Message)
            .exists()
        )
        live_or_terminal_failure_job = (
            select(Job.id)
            .where(
                Job.job_type == str(projection_job_type),
                Job.status.in_(("queued", "running", "failed")),
                cast(
                    func.json_extract(Job.payload_json, "$.group_id"),
                    Integer,
                )
                == Message.group_id,
                cast(
                    func.json_extract(Job.payload_json, "$.message_id"),
                    Integer,
                )
                == Message.id,
            )
            .correlate(Message)
            .exists()
        )
        delivery_state = func.coalesce(
            func.json_extract(Message.raw_json, "$.delivery_state"),
            "",
        )
        filters = [
            Message.group_id.is_not(None),
            func.trim(func.coalesce(Message.plain_text, "")) != "",
            ~delivery_state.in_(_INELIGIBLE_DELIVERY_STATES),
            ~matching_projection,
            ~live_or_terminal_failure_job,
        ]
        if group_ids:
            filters.append(Message.group_id.in_(tuple(int(value) for value in group_ids)))
        rows = self.session.execute(
            select(Message.group_id, Message.id)
            .where(*filters)
            .order_by(Message.id.asc())
            .limit(max(1, int(limit)))
        )
        return [
            (int(group_id), int(message_id))
            for group_id, message_id in rows
            if group_id is not None
        ]

    def revoke_unsafe_raw_message_v3_projections(
        self,
        *,
        limit: int,
        group_ids: Sequence[int] | None = None,
    ) -> int:
        """Deactivate projections whose canonical source became ineligible."""
        delivery_state = func.coalesce(
            func.json_extract(Message.raw_json, "$.delivery_state"),
            "",
        )
        filters = [
            RetrievalDocument.document_kind == "raw_message_v3",
            RetrievalDocument.source_table == "messages",
            RetrievalDocument.status == "active",
            or_(
                func.trim(func.coalesce(Message.plain_text, "")) == "",
                delivery_state.in_(_INELIGIBLE_DELIVERY_STATES),
            ),
        ]
        if group_ids:
            filters.append(
                RetrievalDocument.group_id.in_(
                    tuple(int(value) for value in group_ids)
                )
            )
        documents = list(
            self.session.scalars(
                select(RetrievalDocument)
                .join(
                    RetrievalDocumentMessage,
                    (
                        RetrievalDocumentMessage.document_id
                        == RetrievalDocument.id
                    )
                    & (
                        RetrievalDocumentMessage.group_id
                        == RetrievalDocument.group_id
                    ),
                )
                .join(
                    Message,
                    (Message.id == RetrievalDocumentMessage.message_id)
                    & (
                        Message.group_id
                        == RetrievalDocumentMessage.group_id
                    ),
                )
                .where(*filters)
                .distinct()
                .order_by(RetrievalDocument.id.asc())
                .limit(max(1, int(limit)))
            )
        )
        self._deactivate_documents(documents)
        return len(documents)

    def deactivate_raw_message_v3(
        self,
        *,
        group_id: int,
        message_id: int,
    ) -> int:
        documents = list(
            self.session.scalars(
                select(RetrievalDocument).where(
                    RetrievalDocument.group_id == int(group_id),
                    RetrievalDocument.document_kind == "raw_message_v3",
                    RetrievalDocument.source_table == "messages",
                    RetrievalDocument.source_id == str(int(message_id)),
                    RetrievalDocument.status == "active",
                )
            )
        )
        self._deactivate_documents(documents)
        return len(documents)

    def load_raw_message_embedding_document(
        self,
        *,
        group_id: int,
        message_id: int,
        document_id: int,
        embedding_generation: int,
    ) -> RetrievalDocument | None:
        document = self.session.scalars(
            select(RetrievalDocument).where(
                RetrievalDocument.id == int(document_id),
                RetrievalDocument.group_id == int(group_id),
                RetrievalDocument.document_kind == "raw_message_v3",
                RetrievalDocument.source_table == "messages",
                RetrievalDocument.source_id == str(int(message_id)),
                RetrievalDocument.status == "active",
                RetrievalDocument.embedding_eligible.is_(True),
                RetrievalDocument.embedding_generation == int(embedding_generation),
            )
        ).first()
        if document is None:
            return None
        source = self.session.scalars(
            select(Message)
            .join(
                RetrievalDocumentMessage,
                (RetrievalDocumentMessage.message_id == Message.id)
                & (RetrievalDocumentMessage.group_id == Message.group_id),
            )
            .where(
                RetrievalDocumentMessage.document_id == document.id,
                RetrievalDocumentMessage.group_id == int(group_id),
                Message.id == int(message_id),
                Message.group_id == int(group_id),
            )
        ).first()
        if (
            source is None
            or MessageRepository.is_reserved_outbound(source)
            or MessageRepository.is_qq_blocked_outbound(source)
            or MessageRepository.is_delivery_uncertain_outbound(source)
            or (
                isinstance(source.raw_json, dict)
                and str(source.raw_json.get("delivery_state") or "")
                .strip()
                .casefold()
                in _INELIGIBLE_DELIVERY_STATES
            )
        ):
            self._deactivate_documents([document])
            return None
        return document

    def _deactivate_documents(
        self,
        documents: Sequence[RetrievalDocument],
    ) -> None:
        if not documents:
            return
        now = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        for document in documents:
            if document.status != "inactive":
                document.status = "inactive"
                document.embedding_status = "stale"
                document.updated_at = now
                self.session.add(document)
            self._sync_fts(document)
        _delete_active_retrieval_vectors(
            self.session,
            document_ids=[document.id for document in documents],
        )

    def list_source_message_ids(self, *, document_id: int, group_id: int) -> list[int]:
        return list(
            self.session.scalars(
                select(RetrievalDocumentMessage.message_id)
                .join(
                    RetrievalDocument,
                    (RetrievalDocument.id == RetrievalDocumentMessage.document_id)
                    & (RetrievalDocument.group_id == RetrievalDocumentMessage.group_id),
                )
                .where(
                    RetrievalDocumentMessage.document_id == document_id,
                    RetrievalDocumentMessage.group_id == group_id,
                    RetrievalDocument.group_id == group_id,
                )
                .order_by(RetrievalDocumentMessage.ordinal.asc())
            )
        )

    def deactivate_episode_documents(
        self,
        *,
        group_id: int,
        episode_id: int,
        embedding_generation: int | None = None,
    ) -> int:
        stmt = select(RetrievalDocument).where(
            RetrievalDocument.group_id == int(group_id),
            RetrievalDocument.episode_id == int(episode_id),
            RetrievalDocument.status == "active",
        )
        if embedding_generation is not None:
            stmt = stmt.where(
                RetrievalDocument.embedding_generation == int(embedding_generation)
            )
        documents = list(self.session.scalars(stmt))
        for document in documents:
            document.status = "inactive"
            document.embedding_status = "stale"
            document.updated_at = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
            self.session.add(document)
            self._sync_fts(document)
        _delete_active_retrieval_vectors(
            self.session,
            document_ids=[document.id for document in documents],
        )
        return len(documents)

    def embedding_coverage(
        self,
        *,
        group_id: int | None = None,
        generation: int | None = None,
    ) -> RetrievalEmbeddingCoverage:
        filters = [
            RetrievalDocument.status == "active",
            RetrievalDocument.embedding_eligible.is_(True),
        ]
        if group_id is not None:
            filters.append(RetrievalDocument.group_id == int(group_id))
        if generation is not None:
            filters.append(
                RetrievalDocument.embedding_generation == int(generation)
            )
        total = int(
            self.session.scalar(
                select(func.count(RetrievalDocument.id)).where(*filters)
            )
            or 0
        )
        ready = int(
            self.session.scalar(
                select(func.count(RetrievalDocument.id)).where(
                    *filters,
                    RetrievalDocument.embedding_status == "ready",
                )
            )
            or 0
        )
        failed = int(
            self.session.scalar(
                select(func.count(RetrievalDocument.id)).where(
                    *filters,
                    RetrievalDocument.embedding_status == "failed",
                )
            )
            or 0
        )
        return RetrievalEmbeddingCoverage(
            total_documents=total,
            ready_documents=ready,
            failed_documents=failed,
        )

    def search_group_documents_fts_hits(
        self,
        *,
        group_id: int,
        query: str,
        limit: int,
        subject_ids: Sequence[str] | None = None,
        document_kinds: Sequence[str] | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        speaker_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
    ) -> list[RetrievalDocumentHit]:
        resolved_limit = max(1, int(limit))
        documents = self.search_group_documents_fts(
            group_id=group_id,
            query=query,
            limit=resolved_limit,
            document_kinds=document_kinds,
            start_at=start_at,
            end_at=end_at,
            speaker_ids=speaker_ids,
            excluded_speaker_ids=excluded_speaker_ids,
            mentioned_user_ids=mentioned_user_ids,
        )
        ranked = [
            (document.id, float(resolved_limit - rank))
            for rank, document in enumerate(documents)
        ]
        hits = self._validated_hits(
            group_id=group_id,
            ranked_document_ids=ranked,
            subject_ids=subject_ids,
            document_kinds=document_kinds,
            start_at=start_at,
            end_at=end_at,
            speaker_ids=speaker_ids,
            excluded_speaker_ids=excluded_speaker_ids,
            mentioned_user_ids=mentioned_user_ids,
        )
        normalized_query = str(query or "").strip().casefold()
        exact_ids = {
            int(document.id)
            for document in documents
            if normalized_query
            and normalized_query in str(document.content or "").casefold()
        }
        return [
            replace(hit, lexical_exact=int(hit.document_id) in exact_ids)
            for hit in hits
        ]

    def search_group_documents_temporal_hits(
        self,
        *,
        group_id: int,
        start_at: datetime | None,
        end_at: datetime | None,
        limit: int,
        subject_ids: Sequence[str] | None = None,
        document_kinds: Sequence[str] | None = None,
        speaker_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
        allow_unbounded: bool = False,
        sample_time_coverage: bool | None = None,
    ) -> list[RetrievalDocumentHit]:
        if start_at is None and end_at is None and not allow_unbounded:
            return []
        normalized_mentioned_user_ids = _normalize_optional_string_filter(
            mentioned_user_ids
        )
        if normalized_mentioned_user_ids == ():
            return []
        stmt = select(RetrievalDocument.id).where(
            RetrievalDocument.group_id == int(group_id),
            RetrievalDocument.status == "active",
            *_retrieval_source_prefilters(
                group_id=group_id,
                speaker_ids=speaker_ids,
                excluded_speaker_ids=excluded_speaker_ids,
            ),
        )
        if normalized_mentioned_user_ids is not None:
            stmt = stmt.where(
                RetrievalDocument.id.in_(
                    _retrieval_mention_document_ids(
                        group_id=group_id,
                        mentioned_user_ids=normalized_mentioned_user_ids,
                    )
                )
            )
        normalized_document_kinds = tuple(
            dict.fromkeys(
                str(kind).strip()
                for kind in (document_kinds or ())
                if str(kind).strip()
            )
        )
        if document_kinds is not None:
            if not normalized_document_kinds:
                return []
            stmt = stmt.where(
                RetrievalDocument.document_kind.in_(normalized_document_kinds)
            )
        if start_at is not None:
            stmt = stmt.where(
                RetrievalDocument.end_at
                >= _normalize_utc_sqlite_timestamp(start_at)
            )
        if end_at is not None:
            stmt = stmt.where(
                RetrievalDocument.start_at
                < _normalize_utc_sqlite_timestamp(end_at)
            )
        resolved_limit = max(1, int(limit))
        use_time_coverage = (
            bool(allow_unbounded)
            if sample_time_coverage is None
            else bool(sample_time_coverage)
        )
        ordered_stmt = stmt.order_by(
            RetrievalDocument.start_at.asc()
            if use_time_coverage
            else RetrievalDocument.end_at.desc(),
            RetrievalDocument.id.asc()
            if use_time_coverage
            else RetrievalDocument.id.desc(),
        )
        if not use_time_coverage:
            ordered_stmt = ordered_stmt.limit(resolved_limit * 4)
        document_ids = list(self.session.scalars(ordered_stmt))
        hits = self._validated_hits(
            group_id=group_id,
            ranked_document_ids=[
                (int(document_id), float(len(document_ids) - rank))
                for rank, document_id in enumerate(document_ids)
            ],
            subject_ids=subject_ids,
            document_kinds=document_kinds,
            start_at=start_at,
            end_at=end_at,
            speaker_ids=speaker_ids,
            excluded_speaker_ids=excluded_speaker_ids,
            mentioned_user_ids=mentioned_user_ids,
        )
        if len(hits) <= resolved_limit:
            return hits
        if not use_time_coverage or resolved_limit == 1:
            return hits[:resolved_limit]
        # Deterministic coverage sampling over the full eligible chronology.
        # Endpoints are pinned and interior slots are evenly distributed.
        indices = [
            (slot * (len(hits) - 1)) // (resolved_limit - 1)
            for slot in range(resolved_limit)
        ]
        return [hits[index] for index in dict.fromkeys(indices)]

    def search_group_documents_entity_hits(
        self,
        *,
        group_id: int,
        entities: tuple[str, ...],
        speaker_ids: Sequence[str] | None,
        limit: int,
        subject_ids: Sequence[str] | None = None,
        document_kinds: Sequence[str] | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
    ) -> list[RetrievalDocumentHit]:
        normalized_entities = tuple(
            dict.fromkeys(value.strip() for value in entities if value.strip())
        )[:12]
        normalized_speaker_ids = tuple(
            int(value)
            for value in dict.fromkeys(speaker_ids or ())
            if str(value).strip().lstrip("-").isdigit()
        )
        if not normalized_entities and not normalized_speaker_ids:
            return []
        conditions = []
        for entity in normalized_entities:
            conditions.extend(
                (
                    RetrievalDocument.content.contains(entity),
                    RetrievalDocument.metadata_json.contains(entity),
                )
            )
        identity_conditions = []
        if normalized_speaker_ids:
            identity_conditions.append(Message.user_id.in_(normalized_speaker_ids))
        if normalized_entities and not normalized_speaker_ids:
            identity_user_ids = select(User.user_id).where(
                or_(
                    User.nickname.in_(normalized_entities),
                    User.group_card.in_(normalized_entities),
                )
            )
            identity_conditions.append(Message.user_id.in_(identity_user_ids))
        if identity_conditions:
            identity_document_ids = (
                select(RetrievalDocumentMessage.document_id)
                .join(
                    Message,
                    (Message.id == RetrievalDocumentMessage.message_id)
                    & (Message.group_id == RetrievalDocumentMessage.group_id),
                )
                .where(
                    RetrievalDocumentMessage.group_id == int(group_id),
                    Message.group_id == int(group_id),
                    or_(*identity_conditions),
                )
            )
            conditions.append(RetrievalDocument.id.in_(identity_document_ids))
        stmt = (
            select(RetrievalDocument.id)
            .where(
                RetrievalDocument.group_id == int(group_id),
                RetrievalDocument.status == "active",
                or_(*conditions),
                *_retrieval_source_prefilters(
                    group_id=group_id,
                    speaker_ids=speaker_ids,
                    excluded_speaker_ids=excluded_speaker_ids,
                ),
            )
            .order_by(
                RetrievalDocument.end_at.desc(),
                RetrievalDocument.id.desc(),
            )
        )
        normalized_document_kinds = _normalize_optional_string_filter(
            document_kinds
        )
        if normalized_document_kinds == ():
            return []
        if normalized_document_kinds is not None:
            stmt = stmt.where(
                RetrievalDocument.document_kind.in_(normalized_document_kinds)
            )
        if start_at is not None:
            stmt = stmt.where(
                RetrievalDocument.end_at
                >= _normalize_utc_sqlite_timestamp(start_at)
            )
        if end_at is not None:
            stmt = stmt.where(
                RetrievalDocument.start_at
                < _normalize_utc_sqlite_timestamp(end_at)
            )
        stmt = stmt.limit(
            2_147_483_647
            if mentioned_user_ids is not None
            else max(1, int(limit))
        )
        document_ids = list(self.session.scalars(stmt))
        return self._validated_hits(
            group_id=group_id,
            ranked_document_ids=[
                (int(document_id), float(len(document_ids) - rank))
                for rank, document_id in enumerate(document_ids)
            ],
            subject_ids=subject_ids,
            document_kinds=document_kinds,
            start_at=start_at,
            end_at=end_at,
            speaker_ids=speaker_ids,
            excluded_speaker_ids=excluded_speaker_ids,
            mentioned_user_ids=mentioned_user_ids,
        )

    def search_group_fact_hits(
        self,
        *,
        group_id: int,
        query: str,
        entities: tuple[str, ...],
        limit: int,
        subject_ids: Sequence[str] | None = None,
        document_kinds: Sequence[str] | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        speaker_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
    ) -> list[RetrievalDocumentHit]:
        terms = tuple(
            dict.fromkeys(
                [
                    *(value.strip() for value in entities if value.strip()),
                    *(
                        value
                        for value in _fts_search_terms(query)
                        if len(value) >= 2
                    ),
                ]
            )
        )
        instant = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        stmt = (
            select(RetrievalDocument.id)
            .join(
                MemoryItem,
                MemoryItem.id
                == func.cast(RetrievalDocument.source_id, Integer),
            )
            .where(
                RetrievalDocument.group_id == int(group_id),
                RetrievalDocument.status == "active",
                RetrievalDocument.document_kind == "memory",
                RetrievalDocument.source_table == "memory_items",
                RetrievalDocument.scope_type == "group",
                RetrievalDocument.scope_id == str(int(group_id)),
                MemoryItem.scope_type == "group",
                MemoryItem.scope_id == str(int(group_id)),
                MemoryItem.status == "active",
                or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
                or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
            )
        )
        if terms:
            stmt = stmt.where(
                or_(
                    *(
                        RetrievalDocument.content.contains(term)
                        for term in terms
                    )
                )
            )
        document_ids = list(
            self.session.scalars(
                stmt.order_by(
                    RetrievalDocument.end_at.desc(),
                    RetrievalDocument.id.desc(),
                ).limit(max(1, int(limit)))
            )
        )
        return self._validated_hits(
            group_id=group_id,
            ranked_document_ids=[
                (int(document_id), float(len(document_ids) - rank))
                for rank, document_id in enumerate(document_ids)
            ],
            subject_ids=subject_ids,
            document_kinds=document_kinds,
            start_at=start_at,
            end_at=end_at,
            speaker_ids=speaker_ids,
            excluded_speaker_ids=excluded_speaker_ids,
            mentioned_user_ids=mentioned_user_ids,
        )

    def search_group_reference_hits(
        self,
        *,
        group_id: int,
        reference_msg_ids: tuple[str, ...],
        include_replies: bool,
        limit: int,
        subject_ids: Sequence[str] | None = None,
        document_kinds: Sequence[str] | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        speaker_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
    ) -> list[RetrievalDocumentHit]:
        references = tuple(
            dict.fromkeys(value.strip() for value in reference_msg_ids if value.strip())
        )
        if not references:
            return []
        reference_condition = (
            Message.reply_to_msg_id.in_(references)
            if include_replies
            else Message.platform_msg_id.in_(references)
        )
        stmt = (
            select(RetrievalDocument.id)
                .join(
                    RetrievalDocumentMessage,
                    (RetrievalDocumentMessage.document_id == RetrievalDocument.id)
                    & (RetrievalDocumentMessage.group_id == RetrievalDocument.group_id),
                )
                .join(
                    Message,
                    (Message.id == RetrievalDocumentMessage.message_id)
                    & (Message.group_id == RetrievalDocumentMessage.group_id),
                )
                .where(
                    RetrievalDocument.group_id == int(group_id),
                    RetrievalDocument.status == "active",
                    RetrievalDocumentMessage.group_id == int(group_id),
                    Message.group_id == int(group_id),
                    reference_condition,
                    *_retrieval_source_prefilters(
                        group_id=group_id,
                        speaker_ids=speaker_ids,
                        excluded_speaker_ids=excluded_speaker_ids,
                    ),
                )
                .distinct()
                .order_by(
                    RetrievalDocument.end_at.desc(),
                    RetrievalDocument.id.desc(),
                )
            )
        normalized_document_kinds = _normalize_optional_string_filter(
            document_kinds
        )
        if normalized_document_kinds == ():
            return []
        if normalized_document_kinds is not None:
            stmt = stmt.where(
                RetrievalDocument.document_kind.in_(normalized_document_kinds)
            )
        if start_at is not None:
            stmt = stmt.where(
                RetrievalDocument.end_at
                >= _normalize_utc_sqlite_timestamp(start_at)
            )
        if end_at is not None:
            stmt = stmt.where(
                RetrievalDocument.start_at
                < _normalize_utc_sqlite_timestamp(end_at)
            )
        stmt = stmt.limit(
            2_147_483_647
            if mentioned_user_ids is not None
            else max(1, int(limit))
        )
        document_ids = list(self.session.scalars(stmt))
        return self._validated_hits(
            group_id=group_id,
            ranked_document_ids=[
                (int(document_id), float(len(document_ids) - rank))
                for rank, document_id in enumerate(document_ids)
            ],
            subject_ids=subject_ids,
            document_kinds=document_kinds,
            start_at=start_at,
            end_at=end_at,
            speaker_ids=speaker_ids,
            excluded_speaker_ids=excluded_speaker_ids,
            mentioned_user_ids=mentioned_user_ids,
        )

    def search_group_documents_vector_hits(
        self,
        *,
        group_id: int,
        embedding: list[float],
        provider: str,
        model: str,
        dimensions: int,
        version: str,
        generation: int | None = None,
        limit: int,
        subject_ids: Sequence[str] | None = None,
        document_kinds: Sequence[str] | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        speaker_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
    ) -> list[RetrievalDocumentHit]:
        if len(embedding) != int(dimensions):
            raise ValueError("query embedding dimensions are incompatible")
        normalized_document_kinds = _normalize_optional_string_filter(document_kinds)
        normalized_subject_ids = _normalize_optional_string_filter(subject_ids)
        normalized_speaker_ids = _normalize_optional_integer_filter(speaker_ids)
        normalized_excluded_speaker_ids = _normalize_optional_integer_filter(
            excluded_speaker_ids
        )
        normalized_mentioned_user_ids = _normalize_optional_string_filter(
            mentioned_user_ids
        )
        if (
            normalized_document_kinds == ()
            or normalized_speaker_ids == ()
            or normalized_mentioned_user_ids == ()
            or (
                normalized_subject_ids == ()
                and normalized_document_kinds == ("raw_message_v3",)
            )
        ):
            return []
        generation_document_family = (
            "raw_message_v3"
            if normalized_document_kinds == ("raw_message_v3",)
            else ""
        )
        generation_clause = (
            "AND generation = :generation "
            if generation is not None
            else "AND is_active = 1 "
        )
        state = self.session.execute(
            text(
                "SELECT generation, physical_table "
                "FROM retrieval_index_state "
                "WHERE channel = 'vector' AND status = 'ready' "
                f"{generation_clause}"
                "AND provider = :provider AND model = :model "
                "AND dimensions = :dimensions AND version = :version "
                "AND document_family = :document_family LIMIT 1"
            ),
            {
                "generation": (
                    int(generation) if generation is not None else None
                ),
                "provider": str(provider),
                "model": str(model),
                "dimensions": int(dimensions),
                "version": str(version),
                "document_family": generation_document_family,
            },
        ).one_or_none()
        if state is None:
            return []
        physical_table = validate_retrieval_vector_table_name(
            state.physical_table,
            generation=state.generation,
        )
        resolved_limit = max(1, int(limit))
        available = int(
            self.session.execute(
                text(
                    f"SELECT count(*) FROM {physical_table} "
                    "WHERE group_id = :group_id"
                ),
                {"group_id": int(group_id)},
            ).scalar_one()
            or 0
        )
        if available <= 0:
            return []
        has_sparse_post_filters = any(
            value is not None
            for value in (
                normalized_subject_ids,
                normalized_speaker_ids,
                normalized_mentioned_user_ids,
                start_at,
                end_at,
            )
        )
        fetch_ceiling = _vector_fetch_ceiling(
            requested=resolved_limit,
            available=available,
            has_sparse_post_filters=has_sparse_post_filters,
            has_exclusion_filter=normalized_excluded_speaker_ids is not None,
        )
        fetch_limit = _initial_vector_fetch_limit(
            requested=resolved_limit,
            available=fetch_ceiling,
            has_post_filters=has_sparse_post_filters,
        )
        serialized_embedding = json.dumps(
            [float(value) for value in embedding],
            separators=(",", ":"),
        )
        while True:
            ranked = [
                (int(document_id), -float(distance))
                for document_id, distance in self.session.execute(
                    text(
                        "SELECT document_id, distance "
                        f"FROM {physical_table} "
                        "WHERE embedding MATCH :embedding "
                        "AND group_id = :group_id AND k = :limit"
                    ),
                    {
                        "embedding": serialized_embedding,
                        "group_id": int(group_id),
                        "limit": fetch_limit,
                    },
                )
            ]
            hits = self._validated_hits(
                group_id=group_id,
                ranked_document_ids=ranked,
                subject_ids=normalized_subject_ids,
                document_kinds=normalized_document_kinds,
                start_at=start_at,
                end_at=end_at,
                speaker_ids=(
                    tuple(str(value) for value in normalized_speaker_ids)
                    if normalized_speaker_ids is not None
                    else None
                ),
                excluded_speaker_ids=(
                    tuple(str(value) for value in normalized_excluded_speaker_ids)
                    if normalized_excluded_speaker_ids is not None
                    else None
                ),
                mentioned_user_ids=normalized_mentioned_user_ids,
            )
            if len(hits) >= resolved_limit or fetch_limit >= fetch_ceiling:
                return hits[:resolved_limit]
            fetch_limit = _next_vector_fetch_limit(
                fetch_limit,
                requested=resolved_limit,
                available=fetch_ceiling,
            )

    def _validated_hits(
        self,
        *,
        group_id: int,
        ranked_document_ids: list[tuple[int, float]],
        subject_ids: Sequence[str] | None = None,
        document_kinds: Sequence[str] | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        speaker_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
    ) -> list[RetrievalDocumentHit]:
        if not ranked_document_ids:
            return []
        normalized_document_kinds = _normalize_optional_string_filter(document_kinds)
        normalized_subject_ids = _normalize_optional_string_filter(subject_ids)
        normalized_speaker_ids = _normalize_optional_integer_filter(speaker_ids)
        normalized_excluded_speaker_ids = _normalize_optional_integer_filter(
            excluded_speaker_ids
        )
        normalized_mentioned_user_ids = _normalize_optional_string_filter(
            mentioned_user_ids
        )
        if (
            normalized_document_kinds == ()
            or normalized_speaker_ids == ()
            or normalized_mentioned_user_ids == ()
            or (
                normalized_subject_ids == ()
                and normalized_document_kinds == ("raw_message_v3",)
            )
        ):
            return []
        ordered_ids = list(
            dict.fromkeys(int(document_id) for document_id, _ in ranked_document_ids)
        )
        document_filters = [
            RetrievalDocument.id.in_(ordered_ids),
            RetrievalDocument.group_id == int(group_id),
            RetrievalDocument.status == "active",
        ]
        if normalized_document_kinds is not None:
            document_filters.append(
                RetrievalDocument.document_kind.in_(normalized_document_kinds)
            )
        if start_at is not None:
            document_filters.append(
                RetrievalDocument.end_at
                >= _normalize_utc_sqlite_timestamp(start_at)
            )
        if end_at is not None:
            document_filters.append(
                RetrievalDocument.start_at
                < _normalize_utc_sqlite_timestamp(end_at)
            )
        documents = {
            document.id: document
            for document in self.session.scalars(
                select(RetrievalDocument).where(*document_filters)
            )
        }
        memory_documents = {
            document_id: document
            for document_id, document in documents.items()
            if document.document_kind == "memory"
        }
        if memory_documents:
            if normalized_subject_ids == ():
                for document_id in memory_documents:
                    documents.pop(document_id, None)
                memory_documents = {}
            memory_ids = {
                int(document.source_id)
                for document in memory_documents.values()
                if document.source_table == "memory_items"
                and str(document.source_id).lstrip("-").isdigit()
            }
            instant = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
            current_filters = [
                MemoryItem.id.in_(memory_ids),
                MemoryItem.scope_type == "group",
                MemoryItem.scope_id == str(int(group_id)),
                MemoryItem.status == "active",
                or_(MemoryItem.valid_from.is_(None), MemoryItem.valid_from <= instant),
                or_(MemoryItem.valid_until.is_(None), MemoryItem.valid_until > instant),
            ]
            if normalized_subject_ids is not None:
                current_filters.append(MemoryItem.subject_id.in_(normalized_subject_ids))
            current_memory_ids = set(
                self.session.scalars(
                    select(MemoryItem.id).where(*current_filters)
                )
            )
            for document_id, document in memory_documents.items():
                source_id = str(document.source_id)
                if (
                    document.source_table != "memory_items"
                    or not source_id.lstrip("-").isdigit()
                    or int(source_id) not in current_memory_ids
                ):
                    documents.pop(document_id, None)
        all_counts = {
            int(document_id): int(count)
            for document_id, count in self.session.execute(
                select(
                    RetrievalDocumentMessage.document_id,
                    func.count(),
                )
                .where(RetrievalDocumentMessage.document_id.in_(ordered_ids))
                .group_by(RetrievalDocumentMessage.document_id)
            )
        }
        source_rows = list(
            self.session.execute(
                select(
                    RetrievalDocumentMessage.document_id,
                    RetrievalDocumentMessage.group_id,
                    Message.group_id,
                    Message.platform_msg_id,
                    Message.user_id,
                    Message.timestamp,
                    Message.raw_json,
                    RetrievalDocumentMessage.ordinal,
                )
                .join(
                    RetrievalDocument,
                    (RetrievalDocument.id == RetrievalDocumentMessage.document_id)
                    & (
                        RetrievalDocument.group_id
                        == RetrievalDocumentMessage.group_id
                    ),
                )
                .join(
                    Message,
                    (Message.id == RetrievalDocumentMessage.message_id)
                    & (Message.group_id == RetrievalDocumentMessage.group_id),
                )
                .where(
                    RetrievalDocumentMessage.document_id.in_(ordered_ids),
                    RetrievalDocumentMessage.group_id == int(group_id),
                    RetrievalDocument.group_id == int(group_id),
                    Message.group_id == int(group_id),
                )
                .order_by(
                    RetrievalDocumentMessage.document_id.asc(),
                    RetrievalDocumentMessage.ordinal.asc(),
                    RetrievalDocumentMessage.message_id.asc(),
                )
            )
        )
        sources: dict[int, list[str]] = {}
        for (
            document_id,
            provenance_group_id,
            message_group_id,
            platform_msg_id,
            user_id,
            message_timestamp,
            raw_json,
            _ordinal,
        ) in source_rows:
            if (
                int(provenance_group_id) != int(group_id)
                or int(message_group_id) != int(group_id)
            ):
                raise ValueError("retrieval provenance failed group validation")
            delivery_state = (
                str(raw_json.get("delivery_state") or "")
                if isinstance(raw_json, dict)
                else ""
            )
            if delivery_state in _INELIGIBLE_DELIVERY_STATES:
                documents.pop(int(document_id), None)
                continue
            if (
                normalized_speaker_ids is not None
                and int(user_id) not in normalized_speaker_ids
            ):
                documents.pop(int(document_id), None)
                continue
            if (
                normalized_excluded_speaker_ids is not None
                and int(user_id) in normalized_excluded_speaker_ids
            ):
                documents.pop(int(document_id), None)
                continue
            if (
                normalized_mentioned_user_ids is not None
                and not set(normalized_mentioned_user_ids).intersection(
                    _mentioned_user_ids(raw_json)
                )
            ):
                documents.pop(int(document_id), None)
                continue
            normalized_timestamp = _normalize_utc_sqlite_timestamp(message_timestamp)
            if (
                start_at is not None
                and normalized_timestamp
                < _normalize_utc_sqlite_timestamp(start_at)
            ):
                documents.pop(int(document_id), None)
                continue
            if (
                end_at is not None
                and normalized_timestamp
                >= _normalize_utc_sqlite_timestamp(end_at)
            ):
                documents.pop(int(document_id), None)
                continue
            sources.setdefault(int(document_id), []).append(str(platform_msg_id))
        score_by_id = {
            int(document_id): float(score)
            for document_id, score in ranked_document_ids
        }
        hits: list[RetrievalDocumentHit] = []
        for document_id in ordered_ids:
            document = documents.get(document_id)
            if document is None:
                continue
            source_msg_ids = tuple(dict.fromkeys(sources.get(document_id, ())))
            if not source_msg_ids or len(sources.get(document_id, ())) != all_counts.get(
                document_id,
                0,
            ):
                raise ValueError("retrieval provenance is incomplete or unscoped")
            hits.append(
                RetrievalDocumentHit(
                    document_id=document.id,
                    group_id=document.group_id,
                    document_kind=document.document_kind,
                    episode_id=document.episode_id,
                    source_msg_ids=source_msg_ids,
                    start_at=document.start_at,
                    end_at=document.end_at,
                    score=score_by_id[document_id],
                )
            )
        return hits

    def search_group_documents_fts(
        self,
        *,
        group_id: int,
        query: str,
        limit: int,
        document_kinds: Sequence[str] | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        speaker_ids: Sequence[str] | None = None,
        excluded_speaker_ids: Sequence[str] | None = None,
        mentioned_user_ids: Sequence[str] | None = None,
    ) -> list[RetrievalDocument]:
        resolved_limit = max(1, int(limit))
        normalized_document_kinds = _normalize_optional_string_filter(document_kinds)
        normalized_speaker_ids = _normalize_optional_integer_filter(speaker_ids)
        normalized_excluded_speaker_ids = _normalize_optional_integer_filter(
            excluded_speaker_ids
        )
        normalized_mentioned_user_ids = _normalize_optional_string_filter(
            mentioned_user_ids
        )
        if any(
            values == ()
            for values in (
                normalized_document_kinds,
                normalized_speaker_ids,
                normalized_mentioned_user_ids,
            )
        ):
            return []
        terms = [term for term in _fts_search_terms(query) if len(term) >= 3]
        if terms:
            try:
                clauses = [
                    "retrieval_documents_fts MATCH :match_query",
                    "retrieval_documents_fts.group_id = :group_id",
                    "d.group_id = :group_id_int",
                    "d.status = 'active'",
                    "NOT EXISTS ("
                    "SELECT 1 FROM retrieval_document_messages AS safety_rdm "
                    "JOIN messages AS safety_m "
                    "ON safety_m.id = safety_rdm.message_id "
                    "AND safety_m.group_id = safety_rdm.group_id "
                    "WHERE safety_rdm.document_id = d.id "
                    "AND safety_rdm.group_id = :group_id_int "
                    "AND json_extract(safety_m.raw_json, '$.delivery_state') "
                    "IN ('reserved','blocked','uncertain','deleted'))",
                ]
                parameters: dict[str, Any] = {
                    "match_query": " OR ".join(f'"{term}"' for term in terms),
                    "group_id": str(group_id),
                    "group_id_int": int(group_id),
                    "limit": resolved_limit,
                }
                statement = text(
                    "SELECT d.id FROM retrieval_documents_fts "
                    "JOIN retrieval_documents AS d "
                    "ON d.id = CAST(retrieval_documents_fts.document_id AS INTEGER) "
                    "WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY bm25(retrieval_documents_fts), d.id ASC LIMIT :limit"
                )
                if normalized_document_kinds is not None:
                    clauses.append("d.document_kind IN :document_kinds")
                    parameters["document_kinds"] = normalized_document_kinds
                if start_at is not None:
                    clauses.append("d.end_at >= :start_at")
                    parameters["start_at"] = _normalize_utc_sqlite_timestamp(start_at)
                if end_at is not None:
                    clauses.append("d.start_at < :end_at")
                    parameters["end_at"] = _normalize_utc_sqlite_timestamp(end_at)
                if normalized_speaker_ids is not None:
                    clauses.append(
                        "NOT EXISTS ("
                        "SELECT 1 FROM retrieval_document_messages AS speaker_rdm "
                        "JOIN messages AS speaker_m "
                        "ON speaker_m.id = speaker_rdm.message_id "
                        "AND speaker_m.group_id = speaker_rdm.group_id "
                        "WHERE speaker_rdm.document_id = d.id "
                        "AND speaker_rdm.group_id = :group_id_int "
                        "AND speaker_m.user_id NOT IN :speaker_ids)"
                    )
                    parameters["speaker_ids"] = normalized_speaker_ids
                if normalized_excluded_speaker_ids:
                    clauses.append(
                        "NOT EXISTS ("
                        "SELECT 1 FROM retrieval_document_messages AS excluded_rdm "
                        "JOIN messages AS excluded_m "
                        "ON excluded_m.id = excluded_rdm.message_id "
                        "AND excluded_m.group_id = excluded_rdm.group_id "
                        "WHERE excluded_rdm.document_id = d.id "
                        "AND excluded_rdm.group_id = :group_id_int "
                        "AND excluded_m.user_id IN :excluded_speaker_ids)"
                    )
                    parameters["excluded_speaker_ids"] = normalized_excluded_speaker_ids
                if normalized_mentioned_user_ids is not None:
                    clauses.append(
                        "EXISTS ("
                        "SELECT 1 FROM retrieval_document_messages AS mention_rdm "
                        "JOIN messages AS mention_m "
                        "ON mention_m.id = mention_rdm.message_id "
                        "AND mention_m.group_id = mention_rdm.group_id "
                        "JOIN json_each(CASE "
                        "WHEN json_type(mention_m.raw_json, '$.message') = 'array' "
                        "THEN json_extract(mention_m.raw_json, '$.message') "
                        "ELSE json_extract(mention_m.raw_json, '$.raw_message') END) "
                        "AS mention_segment "
                        "WHERE mention_rdm.document_id = d.id "
                        "AND mention_rdm.group_id = :group_id_int "
                        "AND json_extract(mention_segment.value, '$.type') = 'at' "
                        "AND CAST(coalesce("
                        "json_extract(mention_segment.value, '$.data.qq'), "
                        "json_extract(mention_segment.value, '$.data.uin'), "
                        "json_extract(mention_segment.value, '$.data.target')) AS TEXT) "
                        "IN :mentioned_user_ids)"
                    )
                    parameters["mentioned_user_ids"] = normalized_mentioned_user_ids
                statement = text(
                    "SELECT d.id FROM retrieval_documents_fts "
                    "JOIN retrieval_documents AS d "
                    "ON d.id = CAST(retrieval_documents_fts.document_id AS INTEGER) "
                    "WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY bm25(retrieval_documents_fts), d.id ASC LIMIT :limit"
                )
                expanding = []
                if normalized_document_kinds is not None:
                    expanding.append(bindparam("document_kinds", expanding=True))
                if normalized_speaker_ids is not None:
                    expanding.append(bindparam("speaker_ids", expanding=True))
                if normalized_excluded_speaker_ids:
                    expanding.append(
                        bindparam("excluded_speaker_ids", expanding=True)
                    )
                if normalized_mentioned_user_ids is not None:
                    expanding.append(
                        bindparam("mentioned_user_ids", expanding=True)
                    )
                if expanding:
                    statement = statement.bindparams(*expanding)
                rows = self.session.execute(
                    statement,
                    parameters,
                ).scalars()
                document_ids = [int(document_id) for document_id in rows]
                if document_ids:
                    documents = {
                        document.id: document
                        for document in self.session.scalars(
                            select(RetrievalDocument).where(
                                RetrievalDocument.id.in_(document_ids),
                                RetrievalDocument.group_id == group_id,
                                RetrievalDocument.status == "active",
                            )
                        )
                    }
                    return [
                        documents[hit.document_id]
                        for hit in self._validated_hits(
                            group_id=group_id,
                            ranked_document_ids=[
                                (document_id, float(len(document_ids) - rank))
                                for rank, document_id in enumerate(document_ids)
                            ],
                            document_kinds=normalized_document_kinds,
                            start_at=start_at,
                            end_at=end_at,
                            speaker_ids=(
                                tuple(str(value) for value in normalized_speaker_ids)
                                if normalized_speaker_ids is not None
                                else None
                            ),
                            excluded_speaker_ids=(
                                tuple(str(value) for value in normalized_excluded_speaker_ids)
                                if normalized_excluded_speaker_ids is not None
                                else None
                            ),
                            mentioned_user_ids=normalized_mentioned_user_ids,
                        )
                        if hit.document_id in documents
                    ]
            except SQLAlchemyError:
                # FTS is optional. The canonical group-scoped LIKE fallback
                # preserves availability without weakening isolation.
                pass
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []
        document_filters = [
            RetrievalDocument.group_id == int(group_id),
            RetrievalDocument.status == "active",
            RetrievalDocument.content.contains(normalized_query),
            *_retrieval_source_prefilters(
                group_id=group_id,
                speaker_ids=(
                    tuple(str(value) for value in normalized_speaker_ids)
                    if normalized_speaker_ids is not None
                    else None
                ),
                excluded_speaker_ids=(
                    tuple(str(value) for value in normalized_excluded_speaker_ids)
                    if normalized_excluded_speaker_ids is not None
                    else None
                ),
            ),
        ]
        if normalized_document_kinds is not None:
            document_filters.append(
                RetrievalDocument.document_kind.in_(normalized_document_kinds)
            )
        if start_at is not None:
            document_filters.append(
                RetrievalDocument.end_at
                >= _normalize_utc_sqlite_timestamp(start_at)
            )
        if end_at is not None:
            document_filters.append(
                RetrievalDocument.start_at
                < _normalize_utc_sqlite_timestamp(end_at)
            )
        candidates = list(
            self.session.scalars(
                select(RetrievalDocument)
                .where(*document_filters)
                .order_by(RetrievalDocument.start_at.desc(), RetrievalDocument.id.desc())
                .limit(
                    max(resolved_limit * 4, resolved_limit)
                    if (
                        normalized_speaker_ids is None
                        and normalized_mentioned_user_ids is None
                    )
                    else 2_147_483_647
                )
            )
        )
        candidates_by_id = {document.id: document for document in candidates}
        return [
            candidates_by_id[hit.document_id]
            for hit in self._validated_hits(
                group_id=group_id,
                ranked_document_ids=[
                    (document.id, float(len(candidates) - rank))
                    for rank, document in enumerate(candidates)
                ],
                document_kinds=normalized_document_kinds,
                start_at=start_at,
                end_at=end_at,
                speaker_ids=(
                    tuple(str(value) for value in normalized_speaker_ids)
                    if normalized_speaker_ids is not None
                    else None
                ),
                excluded_speaker_ids=(
                    tuple(str(value) for value in normalized_excluded_speaker_ids)
                    if normalized_excluded_speaker_ids is not None
                    else None
                ),
                mentioned_user_ids=normalized_mentioned_user_ids,
            )
            if hit.document_id in candidates_by_id
        ][:resolved_limit]

    def _sync_fts(self, document: RetrievalDocument) -> None:
        try:
            self.session.execute(
                text(
                    "DELETE FROM retrieval_documents_fts "
                    "WHERE document_id = :document_id"
                ),
                {"document_id": str(document.id)},
            )
            if document.status != "active":
                return
            self.session.execute(
                text(
                    "INSERT INTO retrieval_documents_fts "
                    "(content, group_id, document_id, content_hash) "
                    "VALUES (:content, :group_id, :document_id, :content_hash)"
                ),
                {
                    "content": document.content,
                    "group_id": str(document.group_id),
                    "document_id": str(document.id),
                    "content_hash": document.content_hash,
                },
            )
        except SQLAlchemyError:
            return


class RetrievalIndexStateRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_generation(
        self,
        *,
        channel: str,
        generation: int,
        physical_table: str,
        provider: str,
        model: str,
        dimensions: int | None,
        version: str,
        status: str,
        total_documents: int,
        indexed_documents: int,
    ) -> RetrievalIndexState:
        if channel == "vector":
            validate_retrieval_vector_table_name(
                physical_table,
                generation=generation,
            )
            if dimensions is None or int(dimensions) <= 0:
                raise ValueError("vector generation dimensions are required")
        state = self.session.scalars(
            select(RetrievalIndexState).where(
                RetrievalIndexState.channel == channel,
                RetrievalIndexState.generation == generation,
            )
        ).first()
        if state is None:
            state = RetrievalIndexState(
                channel=channel,
                generation=generation,
                physical_table=physical_table,
            )
        elif channel == "vector":
            persisted_identity = (
                state.physical_table,
                state.provider,
                state.model,
                state.dimensions,
                state.version,
            )
            requested_identity = (
                physical_table,
                provider,
                model,
                dimensions,
                version,
            )
            if persisted_identity != requested_identity:
                raise ValueError("vector generation identity is immutable")
        state.provider = provider
        state.model = model
        state.dimensions = dimensions
        state.version = version
        state.status = status
        state.total_documents = max(0, int(total_documents))
        state.indexed_documents = max(0, int(indexed_documents))
        state.updated_at = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        self.session.add(state)
        self.session.flush()
        return state

    def get_active_generation(self, *, channel: str) -> RetrievalIndexState | None:
        return self.session.scalars(
            select(RetrievalIndexState)
            .where(
                RetrievalIndexState.channel == channel,
                RetrievalIndexState.is_active.is_(True),
            )
            .limit(1)
        ).first()

    def activate_generation(
        self,
        *,
        channel: str,
        generation: int,
        expected_active_generation: int | None,
    ) -> bool:
        target = self.session.scalars(
            select(RetrievalIndexState).where(
                RetrievalIndexState.channel == channel,
                RetrievalIndexState.generation == generation,
                RetrievalIndexState.status == "ready",
            )
        ).first()
        if target is None:
            return False
        if channel == "vector":
            validate_retrieval_vector_table_name(
                target.physical_table,
                generation=target.generation,
            )
            if int(target.total_documents or 0) != int(
                target.indexed_documents or 0
            ):
                return False
        activated_at = _normalize_utc_sqlite_timestamp(datetime.now(UTC))
        if expected_active_generation is None:
            result = self.session.execute(
                text(
                    "UPDATE retrieval_index_state "
                    "SET is_active = 1, activated_at = :activated_at, updated_at = :activated_at "
                    "WHERE channel = :channel AND generation = :generation AND status = 'ready' "
                    "AND NOT EXISTS ("
                    "SELECT 1 FROM retrieval_index_state AS active "
                    "WHERE active.channel = :channel AND active.is_active = 1"
                    ")"
                ),
                {
                    "channel": channel,
                    "generation": generation,
                    "activated_at": activated_at,
                },
            )
            self.session.expire_all()
            return int(result.rowcount or 0) == 1

        if expected_active_generation == generation:
            active = self.get_active_generation(channel=channel)
            return active is not None and active.generation == generation

        deactivated = self.session.execute(
            text(
                "UPDATE retrieval_index_state SET is_active = 0 "
                "WHERE channel = :channel AND generation = :expected_generation "
                "AND is_active = 1"
            ),
            {
                "channel": channel,
                "expected_generation": expected_active_generation,
            },
        )
        if int(deactivated.rowcount or 0) != 1:
            return False
        activated = self.session.execute(
            text(
                "UPDATE retrieval_index_state "
                "SET is_active = 1, activated_at = :activated_at, updated_at = :activated_at "
                "WHERE channel = :channel AND generation = :generation AND status = 'ready'"
            ),
            {
                "channel": channel,
                "generation": generation,
                "activated_at": activated_at,
            },
        )
        if int(activated.rowcount or 0) != 1:
            raise RuntimeError("ready retrieval generation disappeared during activation")
        self.session.expire_all()
        return True


class MemoryBackfillRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_run(
        self,
        *,
        run_key: str,
        snapshot_watermarks: dict[str, int],
        manifest: dict[str, Any],
        segmentation_generation: str,
        compaction_generation: str,
        index_generation: str,
        created_at: datetime,
    ) -> MemoryBackfillRun:
        frozen_watermarks = {
            str(group_id): int(watermark)
            for group_id, watermark in snapshot_watermarks.items()
        }
        self.session.execute(
            text(
                "INSERT OR IGNORE INTO memory_backfill_runs ("
                "run_key, status, snapshot_watermarks_json, manifest_json, "
                "segmentation_generation, compaction_generation, index_generation, "
                "created_at, last_error_code"
                ") VALUES ("
                ":run_key, 'pending', :snapshot_watermarks_json, :manifest_json, "
                ":segmentation_generation, :compaction_generation, :index_generation, "
                ":created_at, ''"
                ")"
            ),
            {
                "run_key": run_key,
                "snapshot_watermarks_json": json.dumps(frozen_watermarks),
                "manifest_json": json.dumps(manifest),
                "segmentation_generation": segmentation_generation,
                "compaction_generation": compaction_generation,
                "index_generation": index_generation,
                "created_at": _normalize_utc_sqlite_timestamp(created_at),
            },
        )
        self.session.expire_all()
        run = self.session.scalars(
            select(MemoryBackfillRun).where(MemoryBackfillRun.run_key == run_key)
        ).first()
        if run is None:
            raise RuntimeError("backfill run upsert did not return a persisted row")
        return run

    def get_run(self, *, run_id: int) -> MemoryBackfillRun | None:
        return self.session.get(MemoryBackfillRun, run_id)

    def update_status(
        self,
        *,
        run_id: int,
        status: str,
        completed_at: datetime | None,
        last_error_code: str,
    ) -> MemoryBackfillRun | None:
        run = self.session.get(MemoryBackfillRun, run_id)
        if run is None:
            return None
        run.status = status
        run.completed_at = (
            _normalize_utc_sqlite_timestamp(completed_at)
            if completed_at is not None
            else None
        )
        run.last_error_code = str(last_error_code or "")[:96]
        self.session.add(run)
        return run


class JobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_job(
        self,
        *,
        job_type: str,
        payload_json: dict[str, Any],
        run_at: datetime,
        status: str = "queued",
        job_key: str = "",
    ) -> Job:
        if job_key:
            self.session.execute(
                text(
                    "INSERT OR IGNORE INTO jobs (job_type, job_key, payload_json, status, run_at) "
                    "VALUES (:job_type, :job_key, :payload_json, :status, :run_at)"
                ),
                {
                    "job_type": job_type,
                    "job_key": job_key,
                    "payload_json": json.dumps(payload_json),
                    "status": status,
                    "run_at": _normalize_utc_sqlite_timestamp(run_at),
                },
            )
            existing = self.session.scalars(
                select(Job).where(Job.job_type == job_type, Job.job_key == job_key)
            ).first()
            if existing is not None:
                return existing
        job = Job(job_type=job_type, job_key=job_key, payload_json=payload_json, run_at=run_at, status=status)
        self.session.add(job)
        self.session.flush()
        return job

    def enqueue_coalescing_job(
        self,
        *,
        job_type: str,
        job_key: str,
        payload_json: dict[str, Any],
        run_at: datetime,
        backfill_run_id: int | None = None,
        target_generation: str = "",
        max_attempts: int = 3,
    ) -> Job:
        if not job_key:
            raise ValueError("coalescing jobs require a stable job_key")
        job_id = self.session.execute(
            text(
                "INSERT INTO jobs ("
                "job_type, job_key, payload_json, status, run_at, "
                "requested_generation, processed_generation, claimed_generation, "
                "backfill_run_id, target_generation, max_attempts"
                ") VALUES ("
                ":job_type, :job_key, :payload_json, 'queued', :run_at, 1, 0, 0, "
                ":backfill_run_id, :target_generation, :max_attempts"
                ") ON CONFLICT(job_type, job_key) WHERE job_key <> '' DO UPDATE SET "
                "requested_generation = jobs.requested_generation + 1, "
                "payload_json = excluded.payload_json, "
                "backfill_run_id = coalesce("
                "excluded.backfill_run_id, jobs.backfill_run_id), "
                "target_generation = CASE WHEN excluded.target_generation <> '' "
                "THEN excluded.target_generation ELSE jobs.target_generation END, "
                "max_attempts = excluded.max_attempts, "
                "status = CASE WHEN jobs.status = 'running' THEN 'running' ELSE 'queued' END, "
                "run_at = CASE WHEN jobs.status = 'running' THEN jobs.run_at ELSE excluded.run_at END, "
                "completed_at = NULL, last_error_code = '' "
                "RETURNING id"
            ),
            {
                "job_type": job_type,
                "job_key": job_key,
                "payload_json": json.dumps(payload_json),
                "run_at": _normalize_utc_sqlite_timestamp(run_at),
                "backfill_run_id": backfill_run_id,
                "target_generation": str(target_generation or ""),
                "max_attempts": max(1, int(max_attempts)),
            },
        ).scalar_one()
        self.session.expire_all()
        job = self.session.get(Job, int(job_id))
        if job is None:
            raise RuntimeError("coalescing job upsert did not return a persisted row")
        return job

    def claim_coalescing_job(
        self,
        *,
        job_type: str,
        worker_id: str,
        now: datetime,
        lease_seconds: int,
        target_generation: str | None = None,
        include_derived_generations: bool = False,
    ) -> Job | None:
        claimed_at = _normalize_utc_sqlite_timestamp(now)
        lease_until = _normalize_utc_sqlite_timestamp(
            now + timedelta(seconds=max(1, int(lease_seconds)))
        )
        generation_predicate = ""
        parameters: dict[str, Any] = {
            "job_type": job_type,
            "worker_id": worker_id,
            "claimed_at": claimed_at,
            "lease_until": lease_until,
        }
        if target_generation is not None:
            parameters["target_generation"] = str(target_generation)
            if include_derived_generations:
                parameters["derived_generation_pattern"] = (
                    f"{str(target_generation)}:late:%"
                )
                generation_predicate = (
                    "AND (target_generation = :target_generation "
                    "OR target_generation LIKE :derived_generation_pattern) "
                )
            else:
                generation_predicate = (
                    "AND target_generation = :target_generation "
                )
        job_id = self.session.execute(
            text(
                "UPDATE jobs SET "
                "status = 'running', locked_by = :worker_id, locked_at = :claimed_at, "
                "lease_until = :lease_until, claimed_generation = requested_generation "
                "WHERE id = ("
                "SELECT id FROM jobs "
                "WHERE job_type = :job_type AND status = 'queued' AND run_at <= :claimed_at "
                f"{generation_predicate}"
                "ORDER BY run_at ASC, id ASC LIMIT 1"
                ") AND status = 'queued' RETURNING id"
            ),
            parameters,
        ).scalar_one_or_none()
        if job_id is None:
            return None
        self.session.expire_all()
        return self.session.get(Job, int(job_id))

    def complete_coalescing_job(
        self,
        *,
        job_id: int,
        worker_id: str,
        claimed_generation: int,
        now: datetime,
    ) -> Job | None:
        completed_at = _normalize_utc_sqlite_timestamp(now)
        completed_id = self.session.execute(
            text(
                "UPDATE jobs SET "
                "processed_generation = CASE "
                "WHEN processed_generation < :claimed_generation THEN :claimed_generation "
                "ELSE processed_generation END, "
                "status = CASE "
                "WHEN requested_generation = :claimed_generation THEN 'completed' "
                "ELSE 'queued' END, "
                "run_at = CASE "
                "WHEN requested_generation = :claimed_generation THEN run_at "
                "ELSE :completed_at END, "
                "completed_at = CASE "
                "WHEN requested_generation = :claimed_generation THEN :completed_at "
                "ELSE NULL END, "
                "locked_by = NULL, locked_at = NULL, lease_until = NULL "
                "WHERE id = :job_id AND status = 'running' "
                "AND locked_by = :worker_id AND claimed_generation = :claimed_generation "
                "RETURNING id"
            ),
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "claimed_generation": claimed_generation,
                "completed_at": completed_at,
            },
        ).scalar_one_or_none()
        if completed_id is None:
            return None
        self.session.expire_all()
        return self.session.get(Job, int(completed_id))

    def fail_coalescing_job(
        self,
        *,
        job_id: int,
        worker_id: str,
        claimed_generation: int,
        error_code: str,
        now: datetime,
        retry_at: datetime,
    ) -> Job | None:
        failed_id = self.session.execute(
            text(
                "UPDATE jobs SET "
                "attempt_count = attempt_count + 1, "
                "status = CASE WHEN attempt_count + 1 >= max_attempts "
                "THEN 'failed' ELSE 'queued' END, "
                "run_at = CASE WHEN attempt_count + 1 >= max_attempts "
                "THEN run_at ELSE :retry_at END, "
                "last_error_code = :error_code, "
                "completed_at = CASE WHEN attempt_count + 1 >= max_attempts "
                "THEN :failed_at ELSE NULL END, "
                "locked_by = NULL, locked_at = NULL, lease_until = NULL "
                "WHERE id = :job_id AND status = 'running' "
                "AND locked_by = :worker_id "
                "AND claimed_generation = :claimed_generation "
                "RETURNING id"
            ),
            {
                "job_id": int(job_id),
                "worker_id": str(worker_id),
                "claimed_generation": int(claimed_generation),
                "error_code": str(error_code or "")[:96],
                "retry_at": _normalize_utc_sqlite_timestamp(retry_at),
                "failed_at": _normalize_utc_sqlite_timestamp(now),
            },
        ).scalar_one_or_none()
        if failed_id is None:
            return None
        self.session.expire_all()
        return self.session.get(Job, int(failed_id))

    def update_coalescing_job_payload(
        self,
        *,
        job_id: int,
        worker_id: str,
        claimed_generation: int,
        payload_json: dict[str, Any],
    ) -> Job | None:
        updated_id = self.session.execute(
            text(
                "UPDATE jobs SET payload_json = :payload_json "
                "WHERE id = :job_id AND status = 'running' "
                "AND locked_by = :worker_id "
                "AND claimed_generation = :claimed_generation "
                "RETURNING id"
            ),
            {
                "job_id": int(job_id),
                "worker_id": str(worker_id),
                "claimed_generation": int(claimed_generation),
                "payload_json": json.dumps(payload_json),
            },
        ).scalar_one_or_none()
        if updated_id is None:
            return None
        self.session.expire_all()
        return self.session.get(Job, int(updated_id))

    def retry_failed_coalescing_job(
        self,
        *,
        job_id: int,
        run_at: datetime,
        reset_attempts: bool = True,
    ) -> Job | None:
        result = self.session.execute(
            text(
                "UPDATE jobs SET status = 'queued', run_at = :run_at, "
                "attempt_count = CASE WHEN :reset_attempts THEN 0 ELSE attempt_count END, "
                "last_error_code = '', completed_at = NULL, "
                "locked_by = NULL, locked_at = NULL, lease_until = NULL "
                "WHERE id = :job_id AND status = 'failed' RETURNING id"
            ),
            {
                "job_id": int(job_id),
                "run_at": _normalize_utc_sqlite_timestamp(run_at),
                "reset_attempts": bool(reset_attempts),
            },
        ).scalar_one_or_none()
        if result is None:
            return None
        self.session.expire_all()
        return self.session.get(Job, int(result))

    def requeue_stale_coalescing_jobs(
        self,
        *,
        job_type: str,
        now: datetime,
    ) -> int:
        stale_before = _normalize_utc_sqlite_timestamp(now)
        result = self.session.execute(
            text(
                "UPDATE jobs SET status = 'queued', run_at = :stale_before, "
                "locked_by = NULL, locked_at = NULL, lease_until = NULL "
                "WHERE job_type = :job_type AND status = 'running' "
                "AND lease_until IS NOT NULL AND lease_until <= :stale_before"
            ),
            {"job_type": job_type, "stale_before": stale_before},
        )
        self.session.expire_all()
        return int(result.rowcount or 0)

    def count_active_jobs(self, *, job_type: str, statuses: list[str] | None = None) -> int:
        active_statuses = statuses or ["queued", "running"]
        stmt = select(func.count(Job.id)).where(Job.job_type == job_type, Job.status.in_(active_statuses))
        return int(self.session.execute(stmt).scalar_one() or 0)

    def list_jobs(self, *, job_type: str, statuses: list[str]) -> list[Job]:
        if not statuses:
            return []
        stmt = (
            select(Job)
            .where(Job.job_type == job_type, Job.status.in_(statuses))
            .order_by(Job.id.asc())
        )
        return list(self.session.scalars(stmt))

    def claim_oldest_queued_job(self, *, job_type: str, now: datetime | None = None) -> Job | None:
        run_before = _normalize_utc_sqlite_timestamp(now or datetime.now().astimezone())
        lease_until = _normalize_utc_sqlite_timestamp(
            (now or datetime.now(UTC)) + timedelta(minutes=15)
        )
        job_id = self.session.execute(
            text(
                "UPDATE jobs SET status = 'running', run_at = :lease_until, "
                "locked_by = 'legacy-worker', locked_at = :run_before, lease_until = :lease_until "
                "WHERE id = ("
                "SELECT id FROM jobs WHERE job_type = :job_type AND status = 'queued' AND run_at <= :run_before "
                "ORDER BY id ASC LIMIT 1"
                ") AND status = 'queued' RETURNING id"
            ),
            {"job_type": job_type, "run_before": run_before, "lease_until": lease_until},
        ).scalar_one_or_none()
        if job_id is None:
            return None
        return self.session.get(Job, int(job_id))

    def mark_job_status(self, *, job_id: int, status: str, payload_json: dict[str, Any] | None = None) -> Job | None:
        job = self.session.get(Job, job_id)
        if job is None:
            return None
        job.status = status
        if payload_json is not None:
            job.payload_json = payload_json
        self.session.add(job)
        return job

    def retry_job(self, *, job_id: int, payload_json: dict[str, Any], run_at: datetime) -> Job | None:
        job = self.session.get(Job, job_id)
        if job is None:
            return None
        job.status = "queued"
        job.payload_json = payload_json
        job.run_at = run_at
        self.session.add(job)
        return job

    def next_queued_job_at(self, *, job_type: str) -> datetime | None:
        value = self.session.execute(
            select(func.min(Job.run_at)).where(Job.job_type == job_type, Job.status == "queued")
        ).scalar_one_or_none()
        return value

    def requeue_running_jobs(self, *, job_type: str) -> int:
        jobs = self.list_jobs(job_type=job_type, statuses=["running"])
        for job in jobs:
            job.status = "queued"
            self.session.add(job)
        return len(jobs)

    def requeue_stale_running_jobs(self, *, job_type: str, now: datetime | None = None) -> int:
        stale_before = _normalize_utc_sqlite_timestamp(now or datetime.now(UTC))
        jobs = list(
            self.session.scalars(
                select(Job).where(
                    Job.job_type == job_type,
                    Job.status == "running",
                    func.coalesce(Job.lease_until, Job.run_at) <= stale_before,
                )
            )
        )
        for job in jobs:
            job.status = "queued"
            job.locked_by = None
            job.locked_at = None
            job.lease_until = None
            self.session.add(job)
        return len(jobs)


class UsageRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_usage(
        self,
        *,
        timestamp: datetime,
        model: str,
        endpoint: str,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
    ) -> UsageRecord:
        record = UsageRecord(
            timestamp=_normalize_utc_sqlite_timestamp(timestamp),
            model=model,
            endpoint=endpoint,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
        )
        self.session.add(record)
        return record

    def summarize_usage(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        model: str | None = None,
    ) -> dict[str, int]:
        normalized_start_at = _normalize_utc_sqlite_timestamp(start_at)
        normalized_end_at = _normalize_utc_sqlite_timestamp(end_at)
        stmt = select(
            func.count(UsageRecord.id),
            func.sum(UsageRecord.input_tokens),
            func.sum(UsageRecord.cached_input_tokens),
            func.sum(UsageRecord.output_tokens),
        ).where(
            UsageRecord.timestamp >= normalized_start_at,
            UsageRecord.timestamp <= normalized_end_at,
        )
        if model is not None:
            stmt = stmt.where(UsageRecord.model == model)
        call_count, input_tokens, cached_input_tokens, output_tokens = self.session.execute(stmt).one()
        return {
            "call_count": int(call_count or 0),
            "input_tokens": int(input_tokens or 0),
            "cached_input_tokens": int(cached_input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
        }


class DevSessionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_latest_owner_session(self, *, owner_qq: int, session_mode: str = "project") -> DevSession | None:
        stmt = (
            select(DevSession)
            .where(DevSession.owner_qq == owner_qq, DevSession.session_mode == session_mode)
            .order_by(DevSession.last_active_at.desc(), DevSession.id.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def create_owner_session(self, *, owner_qq: int, session_mode: str = "project") -> DevSession:
        now = datetime.now().astimezone()
        dev_session = DevSession(
            owner_qq=owner_qq,
            session_mode=session_mode,
            started_at=now,
            last_active_at=now,
            summary="",
        )
        self.session.add(dev_session)
        self.session.flush()
        return dev_session

    def get_or_create_owner_session(self, *, owner_qq: int, session_mode: str = "project") -> DevSession:
        dev_session = self.get_latest_owner_session(owner_qq=owner_qq, session_mode=session_mode)
        if dev_session is None:
            dev_session = self.create_owner_session(owner_qq=owner_qq, session_mode=session_mode)
        else:
            dev_session.last_active_at = datetime.now().astimezone()
            self.session.add(dev_session)
        self.session.add(dev_session)
        self.session.flush()
        return dev_session

    def list_recent_owner_sessions(
        self,
        *,
        owner_qq: int,
        limit: int,
        session_modes: list[str] | None = None,
    ) -> list[DevSession]:
        stmt = select(DevSession).where(DevSession.owner_qq == owner_qq)
        if session_modes:
            stmt = stmt.where(DevSession.session_mode.in_(session_modes))
        stmt = stmt.order_by(DevSession.last_active_at.desc(), DevSession.id.desc()).limit(limit)
        return list(self.session.scalars(stmt))

    def update_session(
        self,
        *,
        session_id: int,
        summary: str | None = None,
        last_task_id: int | None = None,
    ) -> DevSession | None:
        dev_session = self.session.get(DevSession, session_id)
        if dev_session is None:
            return None
        dev_session.last_active_at = datetime.now().astimezone()
        if summary is not None:
            dev_session.summary = summary
        if last_task_id is not None:
            dev_session.last_task_id = last_task_id
        self.session.add(dev_session)
        return dev_session


class DevTaskRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_task(
        self,
        *,
        session_id: int,
        requested_by_qq: int,
        raw_request_text: str,
        intent_type: str,
        status: str = "queued",
    ) -> DevTask:
        task = DevTask(
            session_id=session_id,
            requested_by_qq=requested_by_qq,
            requested_at=datetime.now().astimezone(),
            raw_request_text=raw_request_text,
            intent_type=intent_type,
            status=status,
            summary="",
            files_read_json=[],
            files_changed_json=[],
            commands_run_json=[],
            restart_required=False,
            restart_result="",
            failure_reason="",
            checkpoint_dir="",
            result_text="",
        )
        self.session.add(task)
        self.session.flush()
        return task

    def list_tasks_by_status(self, status: str) -> list[DevTask]:
        stmt = select(DevTask).where(DevTask.status == status).order_by(DevTask.id.asc())
        return list(self.session.scalars(stmt))

    def list_tasks_for_session_by_status(self, *, session_id: int, statuses: list[str]) -> list[DevTask]:
        if not statuses:
            return []
        stmt = (
            select(DevTask)
            .where(DevTask.session_id == session_id, DevTask.status.in_(statuses))
            .order_by(DevTask.id.asc())
        )
        return list(self.session.scalars(stmt))

    def list_tasks_by_statuses(self, *, statuses: list[str], intent_types: list[str] | None = None) -> list[DevTask]:
        if not statuses:
            return []
        stmt = select(DevTask).where(DevTask.status.in_(statuses))
        if intent_types:
            stmt = stmt.where(DevTask.intent_type.in_(intent_types))
        stmt = stmt.order_by(DevTask.id.asc())
        return list(self.session.scalars(stmt))

    def list_recent_tasks_for_session(self, *, session_id: int, limit: int) -> list[DevTask]:
        stmt = (
            select(DevTask)
            .where(DevTask.session_id == session_id)
            .order_by(DevTask.id.desc())
            .limit(limit)
        )
        return list(reversed(list(self.session.scalars(stmt))))

    def claim_oldest_queued_task(self, *, intent_types: list[str] | None = None) -> DevTask | None:
        stmt = select(DevTask).where(DevTask.status == "queued")
        if intent_types:
            stmt = stmt.where(DevTask.intent_type.in_(intent_types))
        stmt = stmt.order_by(DevTask.id.asc()).limit(1)
        task = self.session.execute(stmt).scalar_one_or_none()
        if task is None:
            return None
        task.status = "running"
        self.session.add(task)
        self.session.flush()
        return task

    def get_task(self, task_id: int) -> DevTask | None:
        return self.session.get(DevTask, task_id)

    def mark_completed(
        self,
        *,
        task_id: int,
        summary: str,
        result_text: str,
        files_read: list[str],
        files_changed: list[str],
        commands_run: list[str],
        restart_required: bool,
        restart_result: str,
        checkpoint_dir: str,
    ) -> DevTask | None:
        task = self.session.get(DevTask, task_id)
        if task is None:
            return None
        task.status = "completed"
        task.summary = summary
        task.result_text = result_text
        task.files_read_json = files_read
        task.files_changed_json = files_changed
        task.commands_run_json = commands_run
        task.restart_required = restart_required
        task.restart_result = restart_result
        task.checkpoint_dir = checkpoint_dir
        self.session.add(task)
        return task

    def mark_failed(self, *, task_id: int, failure_reason: str, checkpoint_dir: str = "") -> DevTask | None:
        task = self.session.get(DevTask, task_id)
        if task is None:
            return None
        task.status = "failed"
        task.failure_reason = failure_reason
        if checkpoint_dir:
            task.checkpoint_dir = checkpoint_dir
        self.session.add(task)
        return task

    def mark_status(self, *, task_id: int, status: str) -> DevTask | None:
        task = self.session.get(DevTask, task_id)
        if task is None:
            return None
        task.status = status
        self.session.add(task)
        return task


class DevTaskArtifactRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_artifact(
        self,
        *,
        task_id: int,
        artifact_type: str,
        artifact_path: str,
        metadata_json: dict[str, Any],
    ) -> DevTaskArtifact:
        artifact = DevTaskArtifact(
            task_id=task_id,
            artifact_type=artifact_type,
            artifact_path=artifact_path,
            metadata_json=metadata_json,
        )
        self.session.add(artifact)
        self.session.flush()
        return artifact
