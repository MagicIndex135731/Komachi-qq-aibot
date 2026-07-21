from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import re
from typing import Any

from sqlalchemy import func, or_, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.storage.models import (
    BbotListenerCacheEntry,
    DevSession,
    DevTask,
    DevTaskArtifact,
    Group,
    Job,
    MemoryItem,
    Message,
    Summary,
    UsageRecord,
    User,
)
from app.providers.embeddings import hashed_text_embedding


def _normalize_utc_sqlite_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


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
    def is_qq_blocked_outbound(message: Message) -> bool:
        raw_json = message.raw_json
        return (
            isinstance(raw_json, dict)
            and raw_json.get("delivery_state") == "blocked"
            and raw_json.get("failure_kind") == "qq_sensitive_content"
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
            timestamp=timestamp,
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
            timestamp=timestamp,
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
            if self._is_reserved_outbound(message) or self.is_qq_blocked_outbound(message):
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
            if message.timestamp.date() == day
            and message.user_id not in excluded
            and not self._is_reserved_outbound(message)
            and not self.is_qq_blocked_outbound(message)
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
            and not self.is_qq_blocked_outbound(message)
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
            if self._is_reserved_outbound(message) or self.is_qq_blocked_outbound(message):
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
            if not self._is_reserved_outbound(message) and not self.is_qq_blocked_outbound(message)
        ]

    def list_recent_group_messages_for_user(self, *, group_id: int, user_id: int, limit: int) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.group_id == group_id, Message.user_id == user_id)
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        recent_messages = []
        for message in self.session.scalars(stmt):
            if self._is_reserved_outbound(message) or self.is_qq_blocked_outbound(message):
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

    def list_group_messages_since(
        self,
        *,
        group_id: int,
        since: datetime,
        bot_user_id: int,
        limit: int,
    ) -> list[Message]:
        if limit <= 0:
            return []
        stmt = (
            select(Message)
            .where(
                Message.group_id == group_id,
                Message.timestamp >= since,
            )
            .order_by(Message.timestamp.asc(), Message.id.asc())
        )
        kept_messages = []
        for message in self.session.scalars(stmt):
            if self._is_reserved_outbound(message):
                continue
            if message.user_id == bot_user_id:
                continue
            if str(message.plain_text or "").strip() == "":
                continue
            kept_messages.append(message)
            if len(kept_messages) >= limit:
                break
        return kept_messages

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
            return timestamp.replace(tzinfo=UTC)
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
        memory.last_seen_at = valid_from or datetime.now(UTC)
        self.session.flush()

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
        return memory

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
    ) -> list[MemoryItem]:
        """Return group-scoped lexical candidates with FTS5 as an accelerator.

        The SQL fallback is intentional: short Chinese terms cannot be indexed
        by every FTS tokenizer, and missing a source-backed memory is worse
        than spending a bounded query on the source-of-truth table.
        """
        if limit <= 0:
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
                "UPDATE jobs SET status = 'running', run_at = :lease_until WHERE id = ("
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
                    Job.run_at <= stale_before,
                )
            )
        )
        for job in jobs:
            job.status = "queued"
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
