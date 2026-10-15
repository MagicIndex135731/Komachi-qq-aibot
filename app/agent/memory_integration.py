"""Read-only Agent access to the existing scoped V3 memory pipeline.

Canonical storage, hybrid channels, query resolution and token packing remain
owned by the memory runtime. This adapter only supplies a canonical request,
validates complete provenance and registers it with the Agent's revocation
guard before any evidence enters its persistent session.
"""
from __future__ import annotations

import asyncio
import base64
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import hmac
import json
import logging
from time import perf_counter
from typing import TYPE_CHECKING, Any
import zlib

from sqlalchemy import select

from app.agent.tool_agent import ToolOutcome
from app.core.legacy_memory_context import GroupMemoryContextRequest
from app.core.member_identity import (
    classify_group_member_reference,
    group_member_identities_from_messages,
)
from app.core.memory_context_packer import PackedMemoryContext
from app.core.memory_query_resolver import MemoryQueryResolver
from app.core.memory_source_guard import is_usable_memory_source
from app.core.memory_tool_executor import MemoryToolExecutor
from app.core.memory_tools import DEFAULT_MEMORY_TOOL_TIMEOUT_SECONDS, validate_memory_search_args
from app.core.time_utils import shanghai_aware, stored_as_utc
from app.runtime.blocking import AgentModelBusyError
from app.runtime.memory_composition import _evidence_messages_from_rows
from app.storage.db import session_scope
from app.storage.models import MemoryItem, Message, RetrievalDocument, Summary
from app.storage.repositories import (
    MessageRepository,
    RetrievalDocumentRepository,
    UserRepository,
)

if TYPE_CHECKING:
    from app.agent.service_tools import ServiceToolContext


logger = logging.getLogger(__name__)
SOURCE_LIMIT = 200
DEPENDENCY_LIMIT = 200


def _dependency_identifiers(dependencies: Mapping[str, str]) -> dict[str, dict[int, str]]:
    if not isinstance(dependencies, Mapping) or len(dependencies) > DEPENDENCY_LIMIT:
        raise ValueError("invalid_memory_dependencies")
    identifiers = {"fact": {}, "summary": {}}
    for key, revision in dependencies.items():
        if (not isinstance(key, str) or not isinstance(revision, str) or len(revision) != 64
                or any(character not in "0123456789abcdef" for character in revision)):
            raise ValueError("invalid_memory_dependencies")
        kind, separator, raw_id = key.partition(":")
        if (kind not in identifiers or separator != ":" or not raw_id.isascii()
                or not raw_id.isdigit() or len(raw_id) > 19):
            raise ValueError("invalid_memory_dependencies")
        identifier = int(raw_id)
        if not 1 <= identifier <= 2**63 - 1 or str(identifier) != raw_id:
            raise ValueError("invalid_memory_dependencies")
        identifiers[kind][identifier] = revision
    return identifiers


def create_memory_dependency_receipt(dependencies: Mapping[str, str]) -> dict[str, Any]:
    """Compact content-free revisions for an atomic, size-bounded work note.

    Only bounded canonical identifiers and an aggregate revision hash persist.
    Restoring re-reads their current semantic revisions before comparing hashes.
    """
    _dependency_identifiers(dependencies)
    ordered = dict(sorted(dependencies.items()))
    encoded = base64.b64encode(zlib.compress(json.dumps(list(ordered),
        separators=(",", ":")).encode("ascii"))).decode("ascii")
    digest = hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode("ascii")).hexdigest()
    return {"v": 1, "keys": encoded, "revision": digest}


def restore_memory_dependency_receipt(engine, *, group_id: int, receipt: Any, now) -> dict[str, str] | None:
    """Restore only unchanged active group dependencies; malformed data fails closed."""
    if (not isinstance(receipt, dict) or set(receipt) != {"v", "keys", "revision"}
            or type(receipt.get("v")) is not int or receipt["v"] != 1
            or not isinstance(receipt.get("keys"), str) or len(receipt["keys"]) > 4096
            or not isinstance(receipt.get("revision"), str) or len(receipt["revision"]) != 64
            or any(character not in "0123456789abcdef" for character in receipt["revision"])):
        return None
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(base64.b64decode(receipt["keys"], validate=True), 8192)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            return None
        keys = json.loads(raw)
        if (not isinstance(keys, list) or len(keys) > DEPENDENCY_LIMIT
                or any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys)):
            return None
        identifiers = _dependency_identifiers(dict.fromkeys(keys, "0" * 64))
    except (ValueError, TypeError, zlib.error):
        return None
    dependencies = {}
    with session_scope(engine) as session:
        for kind, model in (("fact", MemoryItem), ("summary", Summary)):
            wanted = identifiers[kind]
            if not wanted:
                continue
            rows = tuple(session.scalars(select(model).where(model.scope_type == "group",
                model.scope_id == str(group_id), model.id.in_(wanted))))
            if len(rows) != len(wanted):
                return None
            for row in rows:
                revision = _revision(row, now)
                if revision is None:
                    return None
                dependencies[f"{kind}:{row.id}"] = revision
    current = create_memory_dependency_receipt(dependencies)["revision"]
    return dependencies if hmac.compare_digest(receipt["revision"], current) else None


def _revision(row: MemoryItem | Summary, now) -> str | None:
    """Semantic changes invalidate context; counters and vector caches do not."""
    if row.status != "active":
        return None
    if isinstance(row, MemoryItem):
        if any(value is not None and stored_as_utc(value) <= stored_as_utc(now)
               for value in (row.valid_until, row.expires_at)):
            return None
        if row.valid_from is not None and stored_as_utc(row.valid_from) > stored_as_utc(now):
            return None
        payload = (row.scope_type, row.scope_id, row.subject_type, row.subject_id,
                   row.memory_kind, row.content, row.source_msg_id, row.source_msg_ids,
                   row.valid_from, row.valid_until, row.expires_at, row.superseded_by_id)
    else:
        payload = (row.scope_type, row.scope_id, row.summary_level, row.content,
                   row.source_start_msg_id, row.source_end_msg_id, row.source_count,
                   row.source_summary_ids, row.start_at, row.end_at)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, default=str,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def memory_dependencies_valid(engine, *, group_id: int, dependencies: Mapping[str, str], now) -> bool:
    """Recheck only facts/summaries actually admitted to this native session.

    The persisted values are content-free revision hashes. Unrelated new raw
    messages and facts cannot rotate this session. Missing, expired, superseded
    or edited dependencies fail closed, including changes to summary children.
    """
    try:
        identifiers = _dependency_identifiers(dependencies)
    except ValueError:
        return False
    with session_scope(engine) as session:
        for kind, model in (("fact", MemoryItem), ("summary", Summary)):
            wanted = identifiers[kind]
            if not wanted:
                continue
            rows = tuple(session.scalars(select(model).where(model.scope_type == "group",
                model.scope_id == str(group_id), model.id.in_(wanted))))
            if len(rows) != len(wanted) or any(_revision(row, now) != wanted[row.id] for row in rows):
                return False
    return True


def _summary_dependencies(session, row: Summary, now, *, ancestors=frozenset()) -> dict[str, str]:
    if row.id in ancestors or len(ancestors) >= 16:
        raise _MemoryReadError("summary_provenance_invalid")
    revision = _revision(row, now)
    if revision is None:
        raise _MemoryReadError("summary_provenance_invalid")
    dependencies = {f"summary:{row.id}": revision}
    for child_id in row.source_summary_ids or ():
        child = session.get(Summary, child_id)
        if child is None or child.scope_type != "group" or child.scope_id != row.scope_id:
            raise _MemoryReadError("summary_provenance_invalid")
        dependencies.update(_summary_dependencies(session, child, now, ancestors=ancestors | {row.id}))
        if len(dependencies) > DEPENDENCY_LIMIT:
            raise _MemoryReadError("memory_dependency_budget_exhausted")
    return dependencies


class _MemoryReadError(ValueError):
    pass


def _member_id(context: ServiceToolContext, member: str) -> str:
    """Resolve only a real member of this group, rejecting ambiguous aliases."""
    group_id = context.observation.job.scope_id
    bot_id = context.router.runtime.settings.bot_qq
    with session_scope(context.router.engine) as session:
        rows = MessageRepository(session).list_recent_group_member_messages(
            group_id=group_id, limit=None,
        )
        if member.isdigit():
            user_id = int(member)
            if user_id != bot_id and any(row.user_id == user_id for row in rows):
                return str(user_id)
        else:
            resolution = classify_group_member_reference(
                member, group_member_identities_from_messages(rows, target_group_id=group_id),
                match_mode="exact", exclude_user_ids={bot_id},
            )
            if resolution.status == "resolved" and resolution.member is not None:
                return str(resolution.member.user_id)
    raise _MemoryReadError("member_unresolved")


def _request(
    context: ServiceToolContext, *, query: str | None, addressed: bool,
    layer: str = "", subject_ids: tuple[str, ...] | None = None,
) -> GroupMemoryContextRequest:
    router, observation = context.router, context.observation
    settings, group_id = router.runtime.settings, observation.job.scope_id
    with session_scope(router.engine) as session:
        messages = MessageRepository(session)
        target = None
        if context.target_message_id:
            target = messages.get_group_messages_by_platform_msg_ids(
                group_id=group_id, platform_msg_ids=[context.target_message_id],
            ).get(context.target_message_id)
            if not is_usable_memory_source(target, group_id=group_id):
                raise _MemoryReadError("source_unavailable")
            if context.target_user_id is not None and target.user_id != context.target_user_id:
                raise _MemoryReadError("source_author_mismatch")
        recent_limit = min(30, SOURCE_LIMIT, max(1, int(getattr(settings, "memory_recent_snapshot_limit", 60))))
        statement = select(Message).where(Message.group_id == group_id)
        if target is not None:
            statement = statement.where(Message.id <= target.id)
        rows = list(session.scalars(statement.order_by(Message.id.desc()).limit(recent_limit)))
        rows = [row for row in reversed(rows) if is_usable_memory_source(row, group_id=group_id)]
        if target is None:
            target = next((row for row in reversed(rows) if row.user_id != settings.bot_qq), None)
        if query is None:
            query = str(target.plain_text or "") if target is not None else ""
        quoted = None
        if target is not None and target.reply_to_msg_id:
            quoted = messages.get_group_messages_by_platform_msg_ids(
                group_id=group_id, platform_msg_ids=[target.reply_to_msg_id],
            ).get(target.reply_to_msg_id)
            if not is_usable_memory_source(quoted, group_id=group_id):
                quoted = None
        users = UserRepository(session).get_users_by_ids(
            [row.user_id for row in rows] + ([quoted.user_id] if quoted is not None else [])
        )
        evidence_kwargs = dict(users_by_id=users, messages=messages, settings=settings,
                               bot_display_name=str(router.runtime.persona.get("name") or "小町"))
        recent = _evidence_messages_from_rows(rows=rows, **evidence_kwargs)
        quoted_evidence = (_evidence_messages_from_rows(rows=(quoted,), **evidence_kwargs)[0]
                           if quoted is not None else None)
        target_id = str(target.platform_msg_id) if target is not None else None
        requester_id = int(target.user_id) if target is not None else int(context.target_user_id or settings.bot_qq)
        mentioned_bot = bool(target is not None and target.mentioned_bot)
    persona = getattr(router, "_active_persona", lambda _group_id: {})(group_id)
    impersonated = None
    persona_group = persona.get("source_group_id")
    if (getattr(router, "_impersonating", lambda _group_id: False)(group_id)
            and not isinstance(persona_group, bool)
            and str(persona_group or "").isdigit() and int(persona_group) == group_id):
        candidate = persona.get("source_user_id")
        if not isinstance(candidate, bool) and str(candidate or "").isdigit() and int(candidate) > 0:
            impersonated = int(candidate)
    available = max(1, int(getattr(settings, "llm_context_window_tokens", 258000))
                    - int(getattr(settings, "llm_max_output_tokens", 8192))
                    - int(getattr(settings, "llm_context_safety_margin_tokens", 2048))
                    - int(getattr(settings, "llm_tool_context_reserve_tokens", 0)))
    if getattr(context, 'available_input_budget', None) is not None:
        available = min(available, max(1, context.available_input_budget))
    return GroupMemoryContextRequest(
        group_id=group_id, query=str(query or ""), recent_messages=recent,
        quoted_message=quoted_evidence, target_message_id=target_id if not layer else None,
        available_input=available, now=context.clock() if context.clock else observation.now, current_user_id=requester_id,
        impersonated_subject_id=impersonated,
        impersonated_subject_addressed=bool(addressed or mentioned_bot) if impersonated is not None else None,
        bot_user_id=int(settings.bot_qq) if mentioned_bot else None,
        force_memory_retrieval=bool(layer), memory_subject_ids=subject_ids,
        memory_layer=layer, read_only=True,
    )


def _summary_sources(session, group_id: int, summary: Summary) -> tuple[str, ...]:
    """Load full provenance after the shared executor validated this summary."""
    messages = MessageRepository(session)
    if summary.source_summary_ids:
        return tuple(dict.fromkeys(
            source for child_id in summary.source_summary_ids
            for source in _summary_sources(session, group_id, session.get(Summary, child_id))
        ))
    document = session.scalars(select(RetrievalDocument).where(
        RetrievalDocument.group_id == group_id,
        RetrievalDocument.scope_type == "group", RetrievalDocument.scope_id == str(group_id),
        RetrievalDocument.document_kind == "episode_summary",
        RetrievalDocument.source_table == "summaries", RetrievalDocument.source_id == str(summary.id),
        RetrievalDocument.status == "active", RetrievalDocument.content == summary.content,
    )).first()
    if document is not None:
        row_ids = RetrievalDocumentRepository(session).list_source_message_ids(
            document_id=document.id, group_id=group_id,
        )
        rows = session.scalars(select(Message).where(Message.group_id == group_id, Message.id.in_(row_ids)))
    else:
        bounds = messages.get_group_messages_by_platform_msg_ids(
            group_id=group_id, platform_msg_ids=[summary.source_start_msg_id, summary.source_end_msg_id],
        )
        rows = messages.list_group_messages_by_id_range(
            group_id=group_id, start_id=bounds[summary.source_start_msg_id].id,
            end_id=bounds[summary.source_end_msg_id].id, limit=summary.source_count + 1,
        )
    return tuple(dict.fromkeys(row.platform_msg_id for row in rows))


def _safe_packet(context: ServiceToolContext, request, result, *, limit: int | None):
    packed = result.packed_context
    if not isinstance(packed, PackedMemoryContext):
        raise _MemoryReadError("memory_pipeline_unavailable")
    provider = getattr(context.router.memory_orchestrator, "v2_provider", None)
    packer = getattr(provider, "_packer", None)
    if packer is None:
        raise _MemoryReadError("memory_pipeline_unavailable")
    group_id = request.group_id
    known = set(context.observed_sources)
    selected = {"recent": [], "raw": [], "facts": [], "summaries": []}
    returned_sources: set[int] = set()
    selected_dependencies: dict[str, str] = {}
    item_dependencies: dict[tuple[str, int], dict[str, str]] = {}
    tool_time = (MemoryQueryResolver._parse_time_range(
        request.query, request.now or context.observation.now, force_history=True,
    ) if request.memory_layer else None)
    with session_scope(context.router.engine) as session:
        messages = MessageRepository(session)
        validator = MemoryToolExecutor(engine=context.router.engine, group_id=group_id,
            current_user_id=request.current_user_id, now=request.now)

        def register(kind, item, source_ids, dependencies=None):
            if not source_ids:
                return
            scoped_items = item.messages if kind == "raw" else (item,)
            if any(value.group_id != group_id for value in scoped_items):
                raise _MemoryReadError("memory_scope_mismatch")
            rows = messages.get_group_messages_by_platform_msg_ids(group_id=group_id, platform_msg_ids=list(source_ids))
            if set(rows) != set(source_ids) or any(not is_usable_memory_source(row, group_id=group_id) for row in rows.values()):
                return
            if tool_time is not None and any(
                (tool_time.start is not None and stored_as_utc(row.timestamp) < tool_time.start)
                or (tool_time.end is not None and stored_as_utc(row.timestamp) >= tool_time.end)
                for row in rows.values()
            ):
                return
            ids = {row.id for row in rows.values()}
            if len(known | returned_sources | ids) > SOURCE_LIMIT:
                return
            dependencies = dependencies or {}
            if len(set(context.memory_dependencies) | set(selected_dependencies) | set(dependencies)) > DEPENDENCY_LIMIT:
                return
            returned_sources.update(ids)
            selected_dependencies.update(dependencies)
            if kind in {"facts", "summaries"}:
                item_id = item.memory_id if kind == "facts" else item.summary_id
                item_dependencies[(kind, item_id)] = dependencies
            selected[kind].append(item)

        def summary_candidate(item):
            rows = session.scalars(select(Summary).where(
                Summary.scope_type == "group", Summary.scope_id == str(group_id),
                Summary.status == "active", Summary.content == item.text,
                Summary.source_start_msg_id.in_(item.source_msg_ids),
                Summary.source_end_msg_id.in_(item.source_msg_ids),
            ))
            for row in rows:
                if validator._usable_summary_sources(session, row):
                    sources = _summary_sources(session, group_id, row)
                    if request.memory_subject_ids is not None:
                        canonical = messages.get_group_messages_by_platform_msg_ids(group_id=group_id, platform_msg_ids=list(sources))
                        if any(str(source.user_id) not in request.memory_subject_ids for source in canonical.values()):
                            continue
                    return replace(item, source_msg_ids=sources, summary_id=int(row.id)), _summary_dependencies(session, row, request.now)
            return None, {}

        # Initial context keeps the configured adaptive packet. Explicit tools
        # retain ranked units and share a bounded result limit across layers.
        recent = packed.recent_messages if not request.memory_layer else ()
        for message in recent:
            register("recent", message, (message.source_msg_id,))
        buckets = [("facts", tuple(packed.facts)), ("raw", tuple(packed.evidence_segments)),
                   ("summaries", tuple(packed.summaries))]
        candidates = []
        if limit is None:
            candidates = [(kind, item) for kind, items in buckets for item in items]
        else:
            for index in range(max((len(items) for _, items in buckets), default=0)):
                candidates.extend((kind, items[index]) for kind, items in buckets if index < len(items))
        for kind, item in candidates:
            if limit is not None:
                if sum(len(selected[value]) for value in ("raw", "facts", "summaries")) >= limit * (2 if request.memory_layer == "all" else 1):
                    break
                if len(selected[kind]) >= limit:
                    continue
            dependencies = {}
            if kind == "summaries":
                item, dependencies = summary_candidate(item)
                if item is None:
                    continue
            elif kind == "facts":
                canonical = session.get(MemoryItem, item.memory_id) if item.memory_id is not None else None
                if (canonical is None or canonical.scope_type != "group" or canonical.scope_id != str(group_id)
                        or canonical.content != item.text):
                    continue
                revision = _revision(canonical, request.now)
                if revision is None:
                    continue
                dependencies = {f"fact:{canonical.id}": revision}
            sources = (tuple(message.source_msg_id for message in item.messages)
                       if kind == "raw" else item.source_msg_ids)
            register(kind, item, sources, dependencies)
    safe = packer.pack(packed.mode, available_input=request.available_input,
        target_message_id=request.target_message_id, recent_messages=selected["recent"],
        evidence_segments=selected["raw"], facts=selected["facts"], summaries=selected["summaries"],
        recent_only=result.resolved_quote_role == "conversation_anchor")
    # A second packing pass may shrink the packet. Register only what was
    # actually returned, never an earlier oversized intermediate selection.
    with session_scope(context.router.engine) as session:
        rows = MessageRepository(session).get_group_messages_by_platform_msg_ids(
            group_id=group_id, platform_msg_ids=list(safe.source_msg_ids),
        )
        if set(rows) != set(safe.source_msg_ids) or any(not is_usable_memory_source(row, group_id=group_id) for row in rows.values()):
            raise _MemoryReadError("source_unavailable")
        source_ids = tuple(sorted(row.id for row in rows.values()))
    dependencies = {}
    for kind, items in (("facts", safe.facts), ("summaries", safe.summaries)):
        for item in items:
            item_id = item.memory_id if kind == "facts" else item.summary_id
            dependencies.update(item_dependencies[(kind, item_id)])
    return safe, source_ids, dependencies


def _read(context: ServiceToolContext, *, query, addressed, layer="", member=None, limit=None):
    context.guard.require_valid()
    subject_ids = (_member_id(context, member),) if member else None
    request = _request(context, query=query, addressed=addressed, layer=layer, subject_ids=subject_ids)
    orchestrator = getattr(context.router, "memory_orchestrator", None)
    if orchestrator is None:
        raise _MemoryReadError("memory_pipeline_unavailable")
    result = orchestrator.build_context(request)
    if result.group_id != request.group_id:
        raise _MemoryReadError("memory_scope_mismatch")
    packed, source_ids, dependencies = _safe_packet(context, request, result, limit=limit)
    context.guard.require_valid()
    with session_scope(context.router.engine) as session:
        sources = MessageRepository(session).get_group_messages_by_platform_msg_ids(
            group_id=request.group_id, platform_msg_ids=list(packed.source_msg_ids))
        if set(sources) != set(packed.source_msg_ids) or any(
                not is_usable_memory_source(row, group_id=request.group_id) for row in sources.values()):
            raise _MemoryReadError("source_unavailable")
        source_timestamps = {key: shanghai_aware(row.timestamp).isoformat(timespec="seconds")
                             for key, row in sources.items()}
        # Only final packed recent evidence can replace a duplicate observation
        # line. A historical fact's sources do not imply its raw text was shown.
        recent_source_ids = [sources[message.source_msg_id].id for message in packed.recent_messages]
    return {
        "evidence": packed.text, "evidence_blocks": list(packed.evidence_blocks),
        "kind": "source_bound_memory_evidence", "mode": result.mode,
        "source_ids": list(source_ids), "source_message_ids": list(packed.source_msg_ids),
        "recent_source_ids": recent_source_ids,
        "source_timestamps": source_timestamps,
        "memory_dependencies": dependencies,
        "estimated_tokens": packed.estimated_tokens,
        "layer_counts": {"raw": len(packed.evidence_segments), "facts": len(packed.facts), "summaries": len(packed.summaries)},
        "resolved_subject_ids": result.resolved_subject_ids,
        "resolved_subject_binding": result.resolved_subject_binding,
        "resolved_personal_memory_intent": result.resolved_personal_memory_intent,
        "memory_search_allowed": result.memory_search_allowed,
    }


async def _run_read(context: ServiceToolContext, **kwargs) -> ToolOutcome:
    context.guard.require_valid()
    job = context.observation.job
    if job.kind != "group":
        return ToolOutcome({"error": "memory_out_of_scope"}, "blocked")
    if not getattr(context.router, "_group_memory_enabled", lambda **_: True)(group_id=job.scope_id):
        return ToolOutcome({"error": "memory_disabled"}, "blocked")
    timeout = max(0.1, float(getattr(context.router.runtime.settings,
        "memory_memory_tool_timeout_seconds", DEFAULT_MEMORY_TOOL_TIMEOUT_SECONDS)))
    started = perf_counter()
    try:
        value = await asyncio.wait_for(context.blocking.run(_read, context, **kwargs), timeout=timeout)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        logger.warning("agent_memory_read_timeout group_id=%s layer=%s budget_seconds=%.3f elapsed_ms=%.3f",
            job.scope_id, kwargs.get("layer") or "preload", timeout, (perf_counter()-started)*1000)
        return ToolOutcome({"error": "tool_timeout"}, "failed")
    except AgentModelBusyError:
        return ToolOutcome({"error": "tool_busy"}, "failed")
    except _MemoryReadError as exc:
        return ToolOutcome({"error": str(exc)}, "blocked")
    except Exception as exc:
        logger.warning("agent_memory_read_failed error_type=%s", type(exc).__name__)
        return ToolOutcome({"error": "memory_read_failed"}, "failed")
    context.guard.require_valid()
    candidate_ids = set(value["source_ids"])
    if len(context.observed_sources | candidate_ids) > SOURCE_LIMIT:
        return ToolOutcome({"error": "source_budget_exhausted"}, "blocked")
    dependencies = value["memory_dependencies"]
    if (len(set(context.memory_dependencies) | set(dependencies)) > DEPENDENCY_LIMIT
            or not memory_dependencies_valid(context.router.engine, group_id=job.scope_id,
                dependencies=dependencies, now=context.clock() if context.clock else context.observation.now)):
        return ToolOutcome({"error": "memory_dependency_unavailable"}, "blocked")
    # Revalidate after the worker completed and before admitting the evidence
    # into persistent native context; a recall can happen while retrieval runs.
    with session_scope(context.router.engine) as session:
        rows = tuple(session.scalars(select(Message).where(Message.group_id == job.scope_id, Message.id.in_(candidate_ids))))
        if len(rows) != len(candidate_ids) or any(not is_usable_memory_source(row, group_id=job.scope_id) for row in rows):
            return ToolOutcome({"error": "source_unavailable"}, "blocked")
    if perf_counter()-started >= timeout:
        logger.warning("agent_memory_read_timeout group_id=%s layer=%s budget_seconds=%.3f elapsed_ms=%.3f",
            job.scope_id, kwargs.get("layer") or "preload", timeout, (perf_counter()-started)*1000)
        return ToolOutcome({"error": "tool_timeout"}, "failed")
    context.observed_sources.update(candidate_ids)
    context.memory_dependencies.update(dependencies)
    context.guard.require_valid()
    # Revisions are guard/checkpoint metadata, outside the model's token budget.
    evidence = dict(value)
    evidence.pop("memory_dependencies", None)
    logger.info("agent_memory_read_completed group_id=%s layer=%s budget_seconds=%.3f elapsed_ms=%.3f sources=%s",
        job.scope_id, kwargs.get("layer") or "preload", timeout, (perf_counter()-started)*1000, len(candidate_ids))
    return ToolOutcome(evidence)


async def build_agent_memory(
    context: ServiceToolContext, *, query: str | None = None, addressed: bool = False,
) -> ToolOutcome:
    """Build a normal group wake's complete existing memory context."""
    return await _run_read(context, query=query, addressed=addressed)


async def call_agent_memory_tool(context: ServiceToolContext, name: str, arguments: dict[str, Any]) -> ToolOutcome:
    """Search/portrait tools share the full pipeline; writes remain elsewhere."""
    maximum = max(1, min(20, int(getattr(context.router.runtime.settings, "memory_memory_tool_max_results", 5))))
    if name == "memory_search":
        error = validate_memory_search_args(arguments)
        if error is not None:
            return ToolOutcome({"error": "invalid_memory_search_arguments"}, "failed")
        member = arguments.get("member")
        requested_limit = arguments.get("limit", maximum)
        result_limit = min(maximum, requested_limit)
        outcome = await _run_read(context, query=arguments["query"].strip(), addressed=False,
            layer=str(arguments.get("layer") or "all"), member=member.strip() if member else None,
            limit=result_limit)
        if outcome.status == "ok":
            return replace(outcome, data={**outcome.data, "requested_limit": requested_limit,
                "result_limit": result_limit, "limit_capped": result_limit < requested_limit})
        return outcome
    if name == "memory_read":
        member = arguments.get("member")
        if not isinstance(member, str) or not member.strip():
            return ToolOutcome({"error": "member_unresolved"}, "failed")
        return await _run_read(context, query="完整个人画像", addressed=False,
            layer="facts", member=member.strip(), limit=maximum)
    return ToolOutcome({"error": "unknown_tool"}, "failed")
