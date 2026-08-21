"""Scoped executor for the memory function-calling tools."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import logging
import re
from typing import Any, Mapping, Sequence

from app.core.memory_compaction import canonical_key, is_addressing_rule
from app.core.time_utils import ASIA_SHANGHAI
from app.core.memory_tools import (
    MEMORY_TOOL_KINDS,
    MEMORY_TOOL_LIMIT_MAX,
    validate_memory_search_args,
    validate_memory_write_args,
)
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.repositories import (
    MemoryRepository,
    MessageRepository,
    RetrievalDocumentRepository,
    SummaryRepository,
)


logger = logging.getLogger(__name__)

PROFILE_KINDS = frozenset({"profile", "fact", "preference", "taboo", "relationship"})
SUMMARY_LEVELS = ("episode", "semantic_window", "semantic_daily")
MAX_ITEM_CHARS = 500
_RELATIVE_DAY_PATTERN = re.compile(r"昨天|今天|前天|上周")


def _relative_day_range(text: str, now: datetime) -> tuple[datetime, datetime] | None:
    """Resolve 昨天/今天/前天/上周 to an aware UTC range, if present."""
    if not _RELATIVE_DAY_PATTERN.search(text):
        return None
    local_now = now.astimezone(ASIA_SHANGHAI)
    local_day = datetime(local_now.year, local_now.month, local_now.day, tzinfo=ASIA_SHANGHAI)
    if "昨天" in text:
        start = local_day - timedelta(days=1)
    elif "今天" in text:
        start = local_day
    elif "前天" in text:
        start = local_day - timedelta(days=2)
    elif "上周" in text:
        start = local_day - timedelta(days=local_day.weekday() + 7)
    else:
        return None
    if "上周" in text:
        end = start + timedelta(days=7)
    else:
        end = start + timedelta(days=1)
    return start.astimezone(UTC), end.astimezone(UTC)


def _trim(value: str, *, limit: int = MAX_ITEM_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


class MemoryToolExecutor:
    """Execute memory tools strictly scoped to one group and conversation."""

    def __init__(
        self,
        *,
        engine,
        group_id: int,
        current_user_id: int,
        now: datetime,
        recent_source_msg_ids: Sequence[str] = (),
        member_names: Mapping[str, int] | None = None,
        timeout_seconds: float = 2.0,
        max_results: int = 5,
    ) -> None:
        self._engine = engine
        self._group_id = int(group_id)
        self._current_user_id = int(current_user_id)
        self._now = now
        self._recent_source_msg_ids = frozenset(
            str(value) for value in recent_source_msg_ids if str(value).strip()
        )
        self._member_names = dict(member_names or {})
        self._timeout_seconds = max(0.1, float(timeout_seconds))
        self._max_results = max(
            1, min(MEMORY_TOOL_LIMIT_MAX, int(max_results))
        )

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        def run() -> str:
            if name == "memory_search":
                return self._search(arguments)
            if name == "memory_read":
                return self._read(arguments)
            if name == "memory_write":
                return self._write(arguments)
            return '{"error":"unknown_tool"}'

        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="memory-tool",
        ) as pool:
            future = pool.submit(run)
            try:
                return future.result(timeout=self._timeout_seconds)
            except TimeoutError:
                return '{"error":"tool_timeout"}'
            except Exception as exc:
                logger.warning(
                    "memory_tool_error tool=%s error=%s",
                    name,
                    type(exc).__name__,
                )
                return '{"error":"tool_execution_failed"}'

    def _resolve_member(self, member: str | None) -> int | None:
        if member is None:
            return None
        label = str(member).strip()
        if not label:
            return None
        if label.isdigit():
            return int(label)
        resolved = self._member_names.get(label)
        if resolved is None:
            normalized = label.casefold()
            for candidate, user_id in self._member_names.items():
                if str(candidate).casefold() == normalized:
                    return user_id
        return resolved

    def _search(self, arguments: dict[str, Any]) -> str:
        error = validate_memory_search_args(arguments)
        if error is not None:
            return f'{{"error":"{error}"}}'
        query = str(arguments["query"]).strip()
        layer = str(arguments.get("layer") or "all").strip()
        limit = max(1, min(self._max_results, int(arguments.get("limit", self._max_results))))
        member = arguments.get("member")
        if isinstance(member, str) and not member.strip():
            # The model may pass an empty member string to mean "no
            # restriction"; treat it exactly like an absent member.
            member = None
        member_id = self._resolve_member(member)
        if member is not None and member_id is None:
            return '{"error":"member_unresolved"}'

        time_window = _relative_day_range(query, self._now)
        lines: list[str] = []
        with session_scope(self._engine) as session:
            if layer in {"raw", "all"}:
                documents = RetrievalDocumentRepository(session)
                rows = documents.search_group_documents_fts(
                    group_id=self._group_id,
                    query=query,
                    limit=limit,
                    document_kinds=("raw_message_v3",),
                    speaker_ids=(str(member_id),) if member_id is not None else None,
                    start_at=time_window[0] if time_window else None,
                    end_at=time_window[1] if time_window else None,
                )
                for document in rows:
                    message_row = session.get(Message, int(document.source_id))
                    source = (
                        str(message_row.platform_msg_id)
                        if message_row is not None
                        else str(document.source_id)
                    )
                    lines.append(
                        f"Raw message (source: {source}): "
                        f"{_trim(document.content)}"
                    )
            if layer in {"facts", "all"}:
                memories = MemoryRepository(session)
                rows = memories.search_group_memories_fts(
                    scope_id=str(self._group_id),
                    query=query,
                    limit=limit,
                    as_of=self._now,
                    subject_ids=(str(member_id),) if member_id is not None else None,
                )
                for row in rows:
                    sources = ",".join(
                        str(value) for value in (row.source_msg_ids or [])
                    )
                    if not sources and row.source_msg_id:
                        sources = str(row.source_msg_id)
                    lines.append(
                        f"Memory fact ({row.memory_kind}; source: {sources}): "
                        f"{_trim(row.content)}"
                    )
            if layer in {"summaries", "all"}:
                summaries = SummaryRepository(session)
                rows = summaries.list_group_summaries(
                    scope_id=str(self._group_id),
                    limit=limit,
                    summary_levels=SUMMARY_LEVELS,
                )
                for row in rows:
                    sources = ",".join(
                        str(value)
                        for value in (row.source_start_msg_id, row.source_end_msg_id)
                        if value
                    )
                    lines.append(
                        f"Summary ({row.summary_level}; source: {sources}): "
                        f"{_trim(row.content)}"
                    )
        if not lines:
            return "No memory evidence found for this query in the current group."
        return "\n".join(lines[: limit * 2])

    def _read(self, arguments: dict[str, Any]) -> str:
        member = arguments.get("member")
        member_id = self._resolve_member(member)
        if member_id is None:
            return '{"error":"member_unresolved"}'
        lines: list[str] = []
        with session_scope(self._engine) as session:
            memories = MemoryRepository(session)
            rows = memories.list_group_memories_for_subject(
                scope_id=str(self._group_id),
                subject_id=str(member_id),
                limit=self._max_results * 2,
            )
            profile_rows = [
                row for row in rows if row.memory_kind in PROFILE_KINDS
            ]
            for row in profile_rows[: self._max_results]:
                sources = ",".join(str(value) for value in (row.source_msg_ids or []))
                if not sources and row.source_msg_id:
                    sources = str(row.source_msg_id)
                lines.append(
                    f"{row.memory_kind} (source: {sources}): {_trim(row.content)}"
                )
            messages = MessageRepository(session)
            recent_count = len(
                messages.list_recent_group_messages_for_user(
                    group_id=self._group_id,
                    user_id=member_id,
                    limit=200,
                )
            )
        if not lines:
            return (
                f"No profile facts found for member {member} in this group. "
                "Recent messages are not preference evidence."
            )
        return (
            f"Profile facts for {member} (recent messages: {recent_count}):\n"
            + "\n".join(lines)
        )

    def _write(self, arguments: dict[str, Any]) -> str:
        error = validate_memory_write_args(arguments)
        if error is not None:
            return f'{{"error":"{error}"}}'
        kind = str(arguments["kind"])
        subject = str(arguments["subject"]).strip()
        predicate = str(arguments["predicate"]).strip()
        object_text = str(arguments["object_text"]).strip()
        content = str(arguments["content"]).strip()
        source_ids = [str(value).strip() for value in arguments["source_msg_ids"]]
        if subject != "group" and subject != str(self._current_user_id):
            return '{"error":"subject_out_of_scope"}'
        if any(source_id not in self._recent_source_msg_ids for source_id in source_ids):
            return '{"error":"source_not_in_conversation"}'

        with session_scope(self._engine) as session:
            messages = MessageRepository(session)
            for source_id in source_ids:
                row = messages.get_by_platform_msg_id(source_id)
                if row is None or int(row.group_id or 0) != self._group_id:
                    return '{"error":"source_not_in_group"}'
            memory = MemoryRepository(session).upsert_canonical_memory(
                scope_type="group",
                scope_id=str(self._group_id),
                subject_type="group" if subject == "group" else "user",
                subject_id=subject,
                memory_kind=kind,
                canonical_key=canonical_key(
                    kind,
                    subject,
                    predicate,
                    object_text,
                ),
                predicate=predicate,
                object_text=object_text,
                content=content,
                importance=1,
                confidence=0.6,
                source_msg_ids=source_ids,
                valid_from=self._now,
                valid_until=None,
                replace_previous=(
                    kind == "preference"
                    and is_addressing_rule(predicate, object_text, content)
                ),
            )
            memory_id = int(memory.id)
        return f'{{"memory_id":{memory_id}}}'
