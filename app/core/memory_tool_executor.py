"""Scoped executor for the memory function-calling tools."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from datetime import UTC, datetime, timedelta
import logging
import re
from threading import BoundedSemaphore
from time import monotonic
from typing import Any, Mapping, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.adapters.outbound_guard import current_group_dispatch

from app.core.memory_compaction import (
    canonical_key,
    is_addressing_rule,
    is_single_value_profile_attribute,
    profile_sources_directly_support,
    single_value_profile_attribute_predicates,
)
from app.core.memory_fact_ranking import select_diverse_portrait_facts
from app.core.memory_source_guard import is_usable_memory_source, memory_provenance_ids
from app.core.memory_tools import (
    MEMORY_TOOL_KINDS,
    MEMORY_TOOL_LIMIT_MAX,
    validate_memory_search_args,
    validate_memory_write_args,
)
from app.core.time_utils import ASIA_SHANGHAI
from app.storage.db import session_scope
from app.storage.models import Message, RetrievalDocument, Summary
from app.storage.repositories import (
    MemoryRepository,
    MessageRepository,
    RetrievalDocumentRepository,
    SummaryRepository,
)


logger = logging.getLogger(__name__)

SUMMARY_LEVELS = ("episode", "semantic_window", "semantic_daily")
MAX_ITEM_CHARS = 500
_RELATIVE_DAY_PATTERN = re.compile(r"昨天|今天|前天|上周")


class _ToolBusyError(RuntimeError):
    pass


class _ToolCancelledError(RuntimeError):
    pass


class _BoundedReadPool:
    """No queue: cancelled reads occupy their slot until they really finish."""

    def __init__(self, *, max_workers: int = 4) -> None:
        self._slots = BoundedSemaphore(max_workers)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="memory-tool-read")

    def run(self, function, *, timeout: float) -> str:
        if not self._slots.acquire(blocking=False):
            raise _ToolBusyError
        try:
            future = self._pool.submit(copy_context().run, function)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)


_READ_POOL = _BoundedReadPool()
_WRITE_SLOTS = BoundedSemaphore(1)


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
        current_user_id: int | None,
        now: datetime,
        authorized_subject_ids: Sequence[int] | None = None,
        recent_source_msg_ids: Sequence[str] = (),
        member_names: Mapping[str, int] | None = None,
        timeout_seconds: float = 2.0,
        max_results: int = 5,
    ) -> None:
        self._engine = engine
        self._group_id = int(group_id)
        self._current_user_id = int(current_user_id) if current_user_id is not None else None
        # A batch may authorize several real authors. A source-backed personal
        # write still has to be authored entirely by that selected subject.
        self._authorized_subject_ids = frozenset(
            str(int(item)) for item in (
                authorized_subject_ids
                if authorized_subject_ids is not None
                else (() if current_user_id is None else (current_user_id,))
            )
            if not isinstance(item, bool) and int(item) > 0
        )
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
        if name not in {"memory_search", "memory_read", "memory_write"}:
            return '{"error":"unknown_tool"}'
        deadline = monotonic() + self._timeout_seconds
        try:
            if name == "memory_write":
                if not _WRITE_SLOTS.acquire(timeout=self._timeout_seconds):
                    raise TimeoutError
                try:
                    # Writes never outlive this call in a detached thread.
                    return self._write(arguments, deadline=deadline)
                finally:
                    _WRITE_SLOTS.release()
            function = self._search if name == "memory_search" else self._read
            return _READ_POOL.run(lambda: function(arguments), timeout=self._timeout_seconds)
        except _ToolBusyError:
            return '{"error":"tool_busy"}'
        except _ToolCancelledError:
            return '{"error":"tool_cancelled"}'
        except TimeoutError:
            return '{"error":"tool_timeout"}'
        except Exception as exc:
            if monotonic() >= deadline:
                return '{"error":"tool_timeout"}'
            logger.warning("memory_tool_error tool=%s error=%s", name, type(exc).__name__)
            return '{"error":"tool_execution_failed"}'

    def _check_write_allowed(self, deadline: float) -> None:
        if monotonic() >= deadline:
            raise TimeoutError
        scope = current_group_dispatch(self._group_id)
        if scope is not None and not scope.can_send():
            raise _ToolCancelledError

    @contextmanager
    def _write_session(self, deadline: float):
        """Bound SQLite lock/statement work and revoke immediately before commit."""
        self._check_write_allowed(deadline)
        session = Session(self._engine, expire_on_commit=False)
        connection = None
        previous_busy_timeout = None
        try:
            if self._engine.dialect.name == "sqlite":
                connection = session.connection().connection.driver_connection
                previous_busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
                remaining_ms = max(1, int((deadline - monotonic()) * 1000))
                connection.execute(f"PRAGMA busy_timeout={remaining_ms}")
                connection.set_progress_handler(lambda: int(monotonic() >= deadline), 100)
            yield session
            self._check_write_allowed(deadline)
            session.commit()
        except BaseException:
            if connection is not None:
                connection.set_progress_handler(None, 0)
            session.rollback()
            raise
        finally:
            if connection is not None:
                connection.set_progress_handler(None, 0)
                connection.execute(f"PRAGMA busy_timeout={int(previous_busy_timeout)}")
            session.close()

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

    def _usable_provenance(self, session, source_ids: Sequence[str]) -> bool:
        if not source_ids:
            return False
        sources = MessageRepository(session).get_group_messages_by_platform_msg_ids(
            group_id=self._group_id, platform_msg_ids=list(source_ids)
        )
        return set(sources) == set(source_ids) and all(
            is_usable_memory_source(row, group_id=self._group_id)
            for row in sources.values()
        )

    def _usable_summary_sources(
        self, session, summary: Summary, *, ancestors: frozenset[int] = frozenset(),
    ) -> bool:
        """Verify the complete projection or an unambiguous canonical window.

        Legacy semantic summaries store only bounds and a source count. Those
        bounds do not prove that an intermediate source is still available;
        sparse or incomplete windows must therefore be withheld.
        """
        if (
            summary.scope_type != "group"
            or summary.scope_id != str(self._group_id)
            or summary.status != "active"
            or summary.id in ancestors
            or len(ancestors) >= 16
            or not isinstance(summary.source_count, int)
            or isinstance(summary.source_count, bool)
            or summary.source_count <= 0
            or not summary.source_start_msg_id
            or not summary.source_end_msg_id
        ):
            return False
        messages = MessageRepository(session)
        bounds = messages.get_group_messages_by_platform_msg_ids(
            group_id=self._group_id,
            platform_msg_ids=[summary.source_start_msg_id, summary.source_end_msg_id],
        )
        if set(bounds) != {summary.source_start_msg_id, summary.source_end_msg_id}:
            return False
        start = bounds[summary.source_start_msg_id]
        end = bounds[summary.source_end_msg_id]
        if start.id > end.id or any(
            not is_usable_memory_source(row, group_id=self._group_id) for row in bounds.values()
        ):
            return False
        if summary.source_summary_ids:
            # Rollups are derived from these summaries. Their source count can
            # describe daily/filtered rows rather than a contiguous raw window.
            path = ancestors | {int(summary.id)}
            for child_id in summary.source_summary_ids:
                if isinstance(child_id, bool) or not isinstance(child_id, int) or child_id <= 0:
                    return False
                child = session.get(Summary, child_id)
                if child is None or not self._usable_summary_sources(session, child, ancestors=path):
                    return False
                child_bounds = messages.get_group_messages_by_platform_msg_ids(
                    group_id=self._group_id,
                    platform_msg_ids=[child.source_start_msg_id, child.source_end_msg_id],
                )
                if (
                    any(not start.id <= row.id <= end.id for row in child_bounds.values())
                    or child.start_at < summary.start_at
                    or child.end_at > summary.end_at
                ):
                    return False
            return True
        document = session.scalars(select(RetrievalDocument).where(
            RetrievalDocument.scope_type == "group",
            RetrievalDocument.scope_id == str(self._group_id),
            RetrievalDocument.group_id == self._group_id,
            RetrievalDocument.document_kind == "episode_summary",
            RetrievalDocument.source_table == "summaries",
            RetrievalDocument.source_id == str(summary.id),
            RetrievalDocument.status == "active",
            RetrievalDocument.content == summary.content,
        )).first()
        if document is not None:
            source_ids = RetrievalDocumentRepository(session).list_source_message_ids(
                document_id=document.id, group_id=self._group_id,
            )
            if len(source_ids) != summary.source_count or len(set(source_ids)) != len(source_ids):
                return False
            rows = list(session.scalars(select(Message).where(
                Message.group_id == self._group_id, Message.id.in_(source_ids),
            )))
        else:
            # Count before the repository's delivery filters, so an excluded
            # source cannot be silently replaced by a different row in range.
            count = session.scalar(select(func.count()).select_from(Message).where(
                Message.group_id == self._group_id,
                Message.id >= start.id, Message.id <= end.id,
            ))
            if count != summary.source_count:
                return False
            rows = messages.list_group_messages_by_id_range(
                group_id=self._group_id, start_id=start.id, end_id=end.id,
                limit=summary.source_count + 1,
            )
        if (
            len(rows) != summary.source_count
            or not {start.id, end.id}.issubset({row.id for row in rows})
            or any(not is_usable_memory_source(row, group_id=self._group_id) for row in rows)
        ):
            return False
        return True

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
                    if not is_usable_memory_source(message_row, group_id=self._group_id):
                        continue
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
                    if not self._usable_provenance(session, memory_provenance_ids(row)):
                        continue
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
                    if not self._usable_summary_sources(session, row):
                        continue
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
                limit=max(200, self._max_results),
            )
            profile_rows = select_diverse_portrait_facts(
                [row for row in rows if self._usable_provenance(session, memory_provenance_ids(row))],
                limit=self._max_results,
            )
            for row in profile_rows:
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
                f"No stable portrait facts found for member {member} in this group. "
                "Recent messages are not preference evidence."
            )
        return (
            f"Stable portrait for {member} (recent messages: {recent_count}):\n"
            + "\n".join(lines)
        )

    def _write(self, arguments: dict[str, Any], *, deadline: float | None = None) -> str:
        deadline = deadline if deadline is not None else monotonic() + self._timeout_seconds
        error = validate_memory_write_args(arguments)
        if error is not None:
            return f'{{"error":"{error}"}}'
        kind = str(arguments["kind"])
        subject = str(arguments["subject"]).strip()
        predicate = str(arguments["predicate"]).strip()
        object_text = str(arguments["object_text"]).strip()
        content = str(arguments["content"]).strip()
        source_ids = [str(value).strip() for value in arguments["source_msg_ids"]]
        if subject != "group" and subject not in self._authorized_subject_ids:
            return '{"error":"subject_out_of_scope"}'
        if any(source_id not in self._recent_source_msg_ids for source_id in source_ids):
            return '{"error":"source_not_in_conversation"}'

        with self._write_session(deadline) as session:
            messages = MessageRepository(session)
            source_rows = []
            for source_id in source_ids:
                row = messages.get_by_platform_msg_id(source_id)
                if row is None or int(row.group_id or 0) != self._group_id:
                    return '{"error":"source_not_in_group"}'
                if not is_usable_memory_source(row, group_id=self._group_id):
                    return '{"error":"source_ineligible"}'
                source_rows.append(row)
            if subject != "group" and any(
                str(row.user_id) != subject for row in source_rows
            ):
                return '{"error":"source_author_mismatch"}'
            if subject == "group" and any(
                str(row.user_id) not in self._authorized_subject_ids for row in source_rows
            ):
                # A group subject must not turn the assistant's own generated
                # text, or another member's words, into a current-user fact.
                return '{"error":"source_author_not_authorized"}'
            if kind == "profile" and not profile_sources_directly_support(
                predicate=predicate,
                object_text=object_text,
                content=content,
                source_texts=tuple(str(row.plain_text or "") for row in source_rows),
            ):
                return '{"error":"profile_source_not_direct"}'
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
                    (
                        kind == "preference"
                        and is_addressing_rule(predicate, object_text, content)
                    )
                    or (
                        kind == "profile"
                        and is_single_value_profile_attribute(predicate)
                    )
                ),
                replacement_predicates=(
                    single_value_profile_attribute_predicates(predicate)
                    if kind == "profile"
                    else ()
                ),
            )
            memory_id = int(memory.id)
        return f'{{"memory_id":{memory_id}}}'
