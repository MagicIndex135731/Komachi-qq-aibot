"""Framework-neutral, scoped checkpoints with bounded closed-turn context.

The framework still owns its execution loop and JSONL session. This store records
the path, recent closed exchanges and real tool receipts; it never executes or
replays tools. The caller must keep platform action ledgers as the write authority.
"""
from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path

from sqlalchemy import DateTime, Integer, JSON, String
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
    by_id = {receipt.get("call_id"): receipt for receipt in incoming if receipt.get("call_id")}
    for receipt in retained:
        if receipt.get("status") != "uncertain":
            continue
        replacement = by_id.get(receipt.get("call_id"))
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
            return self._snapshot(session.get(AgentSessionCheckpoint, f"{self.account_id}:{scope}"))

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
        merged = sorted(set(state["source_message_ids"]) | set(checked))
        if len(merged) > 200:
            raise ValueError("in-flight checkpoint source budget exhausted")
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be an object")
        beginning = {**checkpoint, "in_flight": True, "last_status": "running"}
        return self.save(scope, state["items"], beginning, merged,
                         expected_revision=expected_revision, before_commit=before_commit)

    def save(self, scope: str, items: list[dict], checkpoint: dict, sources=(), *, expected_revision: int,
             before_commit=None) -> dict:
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
        with session_scope(self.engine) as session:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            key = f"{self.account_id}:{scope}"
            row = session.get(AgentSessionCheckpoint, key)
            if (row.revision if row else 0) != expected_revision:
                raise SessionRevisionConflict("agent checkpoint revision changed")
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
            return self._snapshot(row)

    def clear_if_source(self, scope: str, message_id: int) -> bool:
        from app.storage.db import session_scope

        AutonomousRepository._validate_scope(scope)
        _validate_sources((message_id,))
        with session_scope(self.engine) as session:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            row = session.get(AgentSessionCheckpoint, f"{self.account_id}:{scope}")
            if row is None or message_id not in row.source_message_ids:
                return False
            row.items, row.source_message_ids = [], []
            receipts = _safe_receipts(row.checkpoint).get("receipts", [])
            row.checkpoint = {"receipts": receipts, "last_status": "source_invalidated"}
            row.revision += 1
            row.updated_at = utc_naive(self.clock())
            return True
