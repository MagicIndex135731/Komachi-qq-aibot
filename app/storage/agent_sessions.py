"""Framework-neutral, scoped checkpoints with bounded closed-turn context.

The framework still owns its execution loop and JSONL session. This store records
the path, recent closed exchanges and real tool receipts; it never executes or
replays tools. The caller must keep platform action ledgers as the write authority.
"""
from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import re
from pathlib import Path
from uuid import uuid4
from datetime import timedelta

from sqlalchemy import DateTime, Integer, JSON, String, delete, select
from sqlalchemy.orm import Mapped, mapped_column

from app.storage.autonomous_agent import AutonomousRepository
from app.storage.models import Base
from app.storage.social_agent import utc_naive


class AgentSessionCheckpoint(Base):
    __tablename__ = "agent_session_checkpoints"

    key: Mapped[str] = mapped_column(String(240), primary_key=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    scope: Mapped[str] = mapped_column(String(160))
    items: Mapped[list] = mapped_column(JSON, default=list)
    checkpoint: Mapped[dict] = mapped_column(JSON, default=dict)
    source_message_ids: Mapped[list] = mapped_column(JSON, default=list)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    dropped_items: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class SessionRevisionConflict(RuntimeError):
    """A newer consumer or source invalidation already changed this session."""


class AgentSessionDependency(Base):
    """Content-free provenance retained independently of the current turn cap.

    Compacted summaries conservatively inherit these dependencies. When an
    invalid dependency cannot be removed precisely, the complete native input
    is invalidated; it is never kept with a shortened, misleading source list.
    """
    __tablename__ = 'agent_session_dependencies'
    session_key: Mapped[str] = mapped_column(String(240), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), primary_key=True)
    identifier: Mapped[str] = mapped_column(String(80), primary_key=True)
    revision_value: Mapped[str] = mapped_column(String(64), default='')


class AgentContextSnapshot(Base):
    """Immutable provenance for work that can outlive the main turn."""
    __tablename__ = 'agent_context_snapshots'
    id: Mapped[str] = mapped_column(String(24), primary_key=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    scope: Mapped[str] = mapped_column(String(160), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)


CONTEXT_DEPENDENCY_LIMIT = 16384


def _memory_revisions(value):
    if not isinstance(value, dict) or len(value) > CONTEXT_DEPENDENCY_LIMIT:
        raise ValueError('invalid context memory dependencies')
    for key, revision in value.items():
        if (not isinstance(key, str) or not re.fullmatch(r'(fact|summary):[1-9][0-9]{0,18}', key)
                or int(key.split(':')[1]) > 2**63 - 1
                or not isinstance(revision, str) or not re.fullmatch(r'[a-f0-9]{64}', revision)):
            raise ValueError('invalid context memory dependency')
    return dict(value)


def _retained_context(value):
    if value is None:
        return set(), {}
    sources = value.get('sources', ())
    if (not isinstance(sources, (list, tuple, set)) or len(sources) > CONTEXT_DEPENDENCY_LIMIT
            or any(type(source) is not int or not 1 <= source <= 2**63 - 1 for source in sources)):
        raise ValueError('invalid retained context sources')
    return set(sources), _memory_revisions(value.get('memory', {}))


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("checkpoint must contain JSON values") from exc


def _json_size(value) -> int:
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


def _validate_sources(sources) -> list[int]:
    if (not isinstance(sources, (list, tuple)) or len(sources) > 200
            or any(type(value) is not int or not 1 <= value <= 2**63 - 1 for value in sources)
            or len(set(sources)) != len(sources)):
        raise ValueError("invalid checkpoint sources")
    return list(sources)


def _closed_turns(items: list[dict]) -> list[list[dict]]:
    turns: list[list[dict]] = []
    current: list[dict] = []
    for item in items:
        if item.get("role") == "user" and current:
            turns.append(current)
            current = []
        current.append(item)
    if current:
        turns.append(current)
    for turn in turns:
        pending: set[str] = set()
        seen: set[str] = set()
        for item in turn:
            calls = []
            if item.get("type") == "function_call":
                calls.append(item.get("call_id"))
            elif item.get("role") == "assistant":
                raw_calls = item.get("tool_calls", [])
                if not isinstance(raw_calls, list) or any(not isinstance(call, dict) for call in raw_calls):
                    raise ValueError("invalid tool calls")
                calls.extend(call.get("id") for call in raw_calls)
            for call_id in calls:
                if not isinstance(call_id, str) or not call_id or call_id in seen:
                    raise ValueError("invalid or duplicate tool call identifier")
                seen.add(call_id)
                pending.add(call_id)
            if item.get("type") == "function_call_output" or item.get("role") == "tool":
                call_id = item.get("call_id") if item.get("type") == "function_call_output" else item.get("tool_call_id")
                if not isinstance(call_id, str) or call_id not in pending:
                    raise ValueError("orphan tool output in checkpoint")
                pending.remove(call_id)
        if pending:
            raise ValueError("checkpoint tool calls must have closed outputs")
    return turns


def _safe_receipts(checkpoint: dict) -> dict:
    copied = _json_copy(checkpoint)
    receipts = copied.get("receipts", [])
    if not isinstance(receipts, list) or len(receipts) > 200 or any(not isinstance(receipt, dict) for receipt in receipts):
        raise ValueError("invalid tool receipts")
    for receipt in receipts:
        if "status" in receipt and not isinstance(receipt["status"], str):
            raise ValueError("invalid tool receipt status")
        if "call_id" in receipt and (not isinstance(receipt["call_id"], str) or not receipt["call_id"]):
            raise ValueError("invalid tool receipt call identifier")
        if receipt.get("status") in {"dispatching", "uncertain", "unknown"}:
            if receipt.get("status") == "dispatching":
                receipt.setdefault("recovery_reason", "checkpoint_after_dispatch")
            receipt["status"] = "uncertain"
            receipt["replay_allowed"] = False
    if "receipts" in copied:
        copied["receipts"] = receipts
    return copied


def _retain_uncertain_receipts(previous: dict, checkpoint: dict) -> dict:
    """Context pruning/invalidation cannot accidentally erase write fences."""
    retained = previous.get("receipts", [])
    incoming = checkpoint.get("receipts", [])
    def identity(receipt):
        # Provider call IDs need only be unique inside one model request.
        return json.dumps([receipt.get('run_id'), receipt.get('reply_source_id'), receipt.get('call_id')], sort_keys=True)
    by_id = {identity(receipt): receipt for receipt in incoming if receipt.get("call_id")}
    for receipt in retained:
        if receipt.get("status") != "uncertain":
            continue
        replacement = by_id.get(identity(receipt))
        # A confirmed platform receipt may resolve the uncertainty. Merely
        # claiming a different status cannot make the original call replayable.
        if replacement and replacement.get("status") == "sent" and replacement.get("receipt_id"):
            replacement["replay_allowed"] = False
            continue
        if replacement:
            replacement.update(receipt)
        elif receipt not in incoming:
            incoming.append(receipt)
    if incoming or "receipts" in checkpoint:
        checkpoint["receipts"] = incoming
    return _safe_receipts(checkpoint)


class AgentSessionRepository:
    """CAS checkpoints isolated by account, data scope and framework path."""

    def __init__(self, engine, *, account_id: int, session_root: Path,
                 clock=None, max_items: int = 256, max_characters: int = 60000):
        AutonomousRepository(engine, account_id=account_id)
        if type(max_items) is not int or not 1 <= max_items <= 1024:
            raise ValueError("invalid session item budget")
        if type(max_characters) is not int or not 128 <= max_characters <= 128000:
            raise ValueError("invalid session character budget")
        self.engine, self.account_id = engine, account_id
        self.session_root = Path(session_root).resolve()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.max_items, self.max_characters = max_items, max_characters

    def session_path(self, scope: str) -> Path:
        AutonomousRepository._validate_scope(scope)
        directory = self.session_root / str(self.account_id) / sha256(scope.encode("utf-8")).hexdigest()
        if not directory.resolve().is_relative_to(self.session_root):
            raise ValueError("session path escapes root")
        return directory / "session.jsonl"

    def _validate_checkpoint(self, scope: str, checkpoint: dict) -> dict:
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be an object")
        copied = _safe_receipts(checkpoint)
        if "session_path" in copied:
            path = copied["session_path"]
            if not isinstance(path, str) or not path or not Path(path).is_absolute():
                raise ValueError("session path must be absolute")
            isolated_directory = self.session_path(scope).parent.resolve()
            if not Path(path).resolve().is_relative_to(isolated_directory):
                raise ValueError("session path belongs to another data scope")
        if _json_size(copied) > 16000:
            raise ValueError("checkpoint exceeds character budget")
        return copied

    @staticmethod
    def _snapshot(row: AgentSessionCheckpoint | None) -> dict:
        if row is None:
            return {"items": [], "checkpoint": {}, "source_message_ids": [], "revision": 0, "dropped_items": 0}
        return {"items": _json_copy(row.items), "checkpoint": _safe_receipts(row.checkpoint),
                "source_message_ids": list(row.source_message_ids), "revision": row.revision,
                "dropped_items": row.dropped_items}

    def read(self, scope: str) -> dict:
        from app.storage.db import session_scope

        AutonomousRepository._validate_scope(scope)
        with session_scope(self.engine) as session:
            row = session.get(AgentSessionCheckpoint, f"{self.account_id}:{scope}")
            return self._snapshot(row) | self._context_dependencies(session, scope, row)

    def _context_dependencies(self, session, scope, row):
        sources = set(row.source_message_ids if row else ())
        memory = _memory_revisions(row.checkpoint.get('memory_dependencies', {}) if row else {})
        rows = session.scalars(select(AgentSessionDependency).where(
            AgentSessionDependency.session_key == f'{self.account_id}:{scope}').limit(CONTEXT_DEPENDENCY_LIMIT + 1))
        for dependency in rows:
            if dependency.kind == 'source':
                sources.add(int(dependency.identifier))
            elif dependency.kind == 'memory':
                if dependency.identifier in memory and memory[dependency.identifier] != dependency.revision_value:
                    raise ValueError('context memory revision conflict')
                memory[dependency.identifier] = dependency.revision_value
            else:
                raise ValueError('invalid context dependency kind')
        if len(sources) + len(memory) > CONTEXT_DEPENDENCY_LIMIT:
            raise ValueError('context dependency resource limit')
        _memory_revisions(memory)
        return {'context_source_message_ids': sorted(sources), 'context_memory_dependencies': memory}

    def freeze_dependencies(self, scope, sources=(), memory=None, *, authority_scope=None, before_commit=None,
                            retained_context=None):
        """Freeze current and retained inputs; a later reset cannot erase them."""
        from app.storage.db import session_scope
        AutonomousRepository._validate_scope(scope)
        authority_scope = authority_scope or scope
        AutonomousRepository._validate_scope(authority_scope)
        current_sources = _validate_sources(sources)
        current_memory = _memory_revisions(memory or {})
        inherited_sources, inherited_memory = _retained_context(retained_context)
        with session_scope(self.engine) as session:
            session.connection().exec_driver_sql('BEGIN IMMEDIATE')
            # Snapshots and background jobs have the same seven-day lifetime.
            expired = list(session.scalars(select(AgentContextSnapshot).where(
                AgentContextSnapshot.account_id == self.account_id,
                AgentContextSnapshot.created_at < utc_naive(self.clock() - timedelta(days=7))).limit(64)))
            for old in expired:
                session.execute(delete(AgentSessionDependency).where(
                    AgentSessionDependency.session_key == f'{self.account_id}:snapshot:{old.id}'))
                session.delete(old)
            count = len(list(session.scalars(select(AgentContextSnapshot.id).where(
                AgentContextSnapshot.account_id == self.account_id).limit(1025))))
            if count >= 1024:
                raise ValueError('context snapshot capacity')
            row = session.get(AgentSessionCheckpoint, f'{self.account_id}:{scope}')
            deps = self._context_dependencies(session, scope, row)
            all_sources = set(deps['context_source_message_ids']) | set(current_sources) | inherited_sources
            all_memory = deps['context_memory_dependencies']
            if any(key in all_memory and all_memory[key] != value for key, value in inherited_memory.items()):
                raise ValueError('context memory revision conflict')
            all_memory.update(inherited_memory)
            if any(key in all_memory and all_memory[key] != value for key, value in current_memory.items()):
                raise ValueError('context memory revision conflict')
            all_memory.update(current_memory)
            if len(all_sources) + len(all_memory) > CONTEXT_DEPENDENCY_LIMIT:
                raise ValueError('context dependency resource limit')
            identifier = uuid4().hex[:24]
            session.add(AgentContextSnapshot(id=identifier, account_id=self.account_id,
                scope=authority_scope, created_at=utc_naive(self.clock())))
            frozen_key = f'{self.account_id}:snapshot:{identifier}'
            for kind, key, revision in [*(('source', str(value), '') for value in all_sources),
                    *(('memory', key, value) for key, value in all_memory.items())]:
                session.add(AgentSessionDependency(session_key=frozen_key, kind=kind,
                    identifier=key, revision_value=revision))
            session.flush()
            if before_commit:
                before_commit()
        return identifier


    def begin_turn(self, scope: str, checkpoint: dict, sources=(), *, expected_revision: int,
                   before_commit=None) -> dict:
        """Fence an in-flight framework JSONL before its first model request.

        On restart the caller rotates a checkpoint marked in_flight, rather
        than replaying framework content whose newer provenance was never
        committed. Closed prior items and durable effect receipts survive.
        """
        state = self.read(scope)
        if state["revision"] != expected_revision:
            raise SessionRevisionConflict("agent checkpoint revision changed")
        checked = _validate_sources(sources)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be an object")
        beginning = {**checkpoint, "in_flight": True, "last_status": "running"}
        return self.save(scope, state["items"], beginning, checked,
                         expected_revision=expected_revision, before_commit=before_commit)

    def save(self, scope: str, items: list[dict], checkpoint: dict, sources=(), *, expected_revision: int,
             before_commit=None, reset_dependencies=False, retained_context=None) -> dict:
        from app.storage.db import session_scope

        AutonomousRepository._validate_scope(scope)
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("invalid expected session revision")
        if not isinstance(items, list) or len(items) > 4096 or any(not isinstance(item, dict) for item in items):
            raise ValueError("invalid session items")
        copied_items = _json_copy(items)
        if _json_size(copied_items) > 2_000_000:
            raise ValueError("session input exceeds compaction budget")
        turns = _closed_turns(copied_items)
        dropped = 0
        # Compute suffix costs once. Repeatedly serializing a 4096-turn input
        # while pruning would make deterministic compression quadratic.
        counts = [len(turn) for turn in turns]
        sizes = [sum(_json_size(item) for item in turn) for turn in turns]
        remaining_count, remaining_size = sum(counts), sum(sizes)
        first_retained = 0
        while (first_retained < len(turns) and (remaining_count > self.max_items
                or 2 + remaining_size + max(0, remaining_count - 1) > self.max_characters)):
            dropped += counts[first_retained]
            remaining_count -= counts[first_retained]
            remaining_size -= sizes[first_retained]
            first_retained += 1
        packed = [item for turn in turns[first_retained:] for item in turn]
        copied_checkpoint = self._validate_checkpoint(scope, checkpoint)
        checked_sources = _validate_sources(sources)
        inherited_sources, inherited_memory = _retained_context(retained_context)
        with session_scope(self.engine) as session:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            key = f"{self.account_id}:{scope}"
            row = session.get(AgentSessionCheckpoint, key)
            if (row.revision if row else 0) != expected_revision:
                raise SessionRevisionConflict("agent checkpoint revision changed")
            retained = (self._context_dependencies(session, scope, row) if not reset_dependencies else
                {'context_source_message_ids': [], 'context_memory_dependencies': {}})
            retained_sources = set(retained['context_source_message_ids']) | set(checked_sources) | inherited_sources
            retained_memory = retained['context_memory_dependencies']
            if any(key in retained_memory and retained_memory[key] != value for key, value in inherited_memory.items()):
                raise ValueError('context memory revision conflict')
            retained_memory.update(inherited_memory)
            incoming_memory = _memory_revisions(copied_checkpoint.get('memory_dependencies', {}))
            if any(key in retained_memory and retained_memory[key] != value for key, value in incoming_memory.items()):
                raise ValueError('context memory revision conflict')
            retained_memory.update(incoming_memory)
            if len(retained_sources) + len(retained_memory) > CONTEXT_DEPENDENCY_LIMIT:
                raise ValueError('context dependency resource limit')
            if reset_dependencies:
                session.execute(delete(AgentSessionDependency).where(AgentSessionDependency.session_key == key))
            existing = {(dep.kind, dep.identifier) for dep in session.scalars(select(AgentSessionDependency).where(
                AgentSessionDependency.session_key == key))}
            for kind, identifier, revision in [
                *(('source', str(source), '') for source in retained_sources),
                *(('memory', name, version) for name, version in retained_memory.items())]:
                if (kind, identifier) not in existing:
                    session.add(AgentSessionDependency(session_key=key, kind=kind,
                        identifier=identifier, revision_value=revision))
            if row is None:
                row = AgentSessionCheckpoint(key=key, account_id=self.account_id, scope=scope)
                session.add(row)
            else:
                copied_checkpoint = self._validate_checkpoint(scope, _retain_uncertain_receipts(
                    _safe_receipts(row.checkpoint), copied_checkpoint))
            row.items, row.checkpoint, row.source_message_ids = packed, copied_checkpoint, checked_sources
            row.revision, row.dropped_items = expected_revision + 1, dropped
            row.updated_at = utc_naive(self.clock())
            session.flush()
            if before_commit is not None:
                before_commit()
            return self._snapshot(row) | {'context_source_message_ids': sorted(retained_sources),
                'context_memory_dependencies': dict(retained_memory)}

    def complete_turn(self, scope: str, items: list[dict], checkpoint: dict, sources=(), *,
                      native_revision: int, parallel=False, before_commit=None, retained_context=None) -> dict:
        """Append one closed result; a concurrent completion never replaces history.

        The normal writer owns native replay. Snapshot workers only append
        closed application data; the next native turn bootstraps the merged
        history when another worker completed outside that native session.
        """
        incoming_sources = _validate_sources(sources)
        inherited_sources, inherited_memory = _retained_context(retained_context)
        retained_context = {'sources': inherited_sources | set(incoming_sources), 'memory': inherited_memory}
        for _ in range(5):
            current = self.read(scope)
            merged = dict(current['checkpoint']) if parallel else dict(checkpoint)
            receipts = [*current['checkpoint'].get('receipts', []), *checkpoint.get('receipts', [])]
            merged['receipts'] = list({json.dumps(r, sort_keys=True): r for r in receipts}.values())
            # This is a bounded context cache; the effect ledger remains the
            # authority. Never let accumulating confirmed receipts prevent a
            # completed turn from closing its 16k-character checkpoint.
            while len(merged['receipts']) > 100 or _json_size(merged['receipts']) > 8000:
                oldest = next((i for i, receipt in enumerate(merged['receipts'])
                    if receipt.get('status') not in {'dispatching','uncertain','unknown'}), None)
                if oldest is None:
                    break
                del merged['receipts'][oldest]
            if items and (parallel or current['revision'] != native_revision):
                merged['shared_native_dirty'] = True
            try:
                return self.save(scope, [*current['items'], *items], merged,
                    current['source_message_ids'] if parallel and current['checkpoint'].get('in_flight') else sources,
                    expected_revision=current['revision'], before_commit=before_commit,
                    retained_context=retained_context)
            except SessionRevisionConflict:
                if before_commit:
                    before_commit()
        raise SessionRevisionConflict('concurrent completion could not commit')

    def clear_if_source(self, scope: str, message_id: int) -> bool:
        from app.storage.db import session_scope

        AutonomousRepository._validate_scope(scope)
        _validate_sources((message_id,))
        with session_scope(self.engine) as session:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            row = session.get(AgentSessionCheckpoint, f"{self.account_id}:{scope}")
            if row is None:
                return False
            dependencies = self._context_dependencies(session, scope, row)
            if message_id not in dependencies['context_source_message_ids']:
                return False
            session.execute(delete(AgentSessionDependency).where(
                AgentSessionDependency.session_key == f'{self.account_id}:{scope}'))
            row.items, row.source_message_ids = [], []
            receipts = _safe_receipts(row.checkpoint).get("receipts", [])
            row.checkpoint = {"receipts": receipts, "last_status": "source_invalidated"}
            row.revision += 1
            row.updated_at = utc_naive(self.clock())
            return True


def read_context_snapshot(engine, *, account_id, scope, identifier, now):
    """Return only a server-created, same-account/scope, unexpired manifest."""
    from app.storage.db import session_scope
    if not isinstance(identifier, str) or not re.fullmatch(r'[a-f0-9]{24}', identifier):
        return None
    with session_scope(engine) as session:
        row = session.get(AgentContextSnapshot, identifier)
        if (row is None or row.account_id != account_id or row.scope != scope
                or row.created_at < utc_naive(now - timedelta(days=7))):
            return None
        entries = list(session.scalars(select(AgentSessionDependency).where(
            AgentSessionDependency.session_key == f'{account_id}:snapshot:{identifier}')
            .limit(CONTEXT_DEPENDENCY_LIMIT + 1)))
        if len(entries) > CONTEXT_DEPENDENCY_LIMIT:
            return None
        sources, memory = set(), {}
        for entry in entries:
            if entry.kind == 'source':
                sources.add(int(entry.identifier))
            elif entry.kind == 'memory':
                memory[entry.identifier] = entry.revision_value
            else:
                return None
        _memory_revisions(memory)
        return {'sources': sorted(sources), 'memory': memory}
