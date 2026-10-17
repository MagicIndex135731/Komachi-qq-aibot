"""Durable per-tool external-effect fencing, separate from wake envelopes."""
from __future__ import annotations

from datetime import UTC, date as calendar_date, datetime, timedelta
from hashlib import sha256
import json
import math
import re

from sqlalchemy import DateTime, Index, Integer, JSON, String, case, or_, select, func, union_all
from sqlalchemy.orm import Mapped, mapped_column

from app.core.time_utils import ASIA_SHANGHAI
from app.storage.autonomous_agent import AutonomousRepository
from app.storage.models import Base, Message
from app.storage.social_agent import SocialActionRecord, utc_naive


class AgentToolEffect(Base):
    __tablename__ = "agent_tool_effects"
    __table_args__ = (Index("ix_agent_tool_effect_budget", "account_id", "scope", "dispatched_at"),)

    action_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    account_id: Mapped[int] = mapped_column(Integer)
    scope: Mapped[str] = mapped_column(String(160))
    run_id: Mapped[str] = mapped_column(String(160))
    wake_id: Mapped[str] = mapped_column(String(160), default="", server_default="")
    tool_call_id: Mapped[str] = mapped_column(String(200))
    operation: Mapped[str] = mapped_column(String(80))
    payload_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), default="reserved")
    reservation_attempts: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    source_message_ids: Mapped[list] = mapped_column(JSON, default=list)
    receipt: Mapped[dict | list | str | None] = mapped_column(JSON, nullable=True)
    reason: Mapped[str] = mapped_column(String(160), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime)
    dispatched_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


def _json_copy(value):
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tool receipt must contain JSON values") from exc
    if len(encoded) > 16000:
        raise ValueError("tool receipt exceeds budget")
    return json.loads(encoded)


class AgentToolEffectRepository:
    TERMINAL = frozenset({"sent", "uncertain", "failed", "blocked"})

    def __init__(self, engine, *, account_id: int, clock=None):
        self._transactions = AutonomousRepository(engine, account_id=account_id, clock=clock)
        self.engine, self.account_id = engine, account_id
        self.clock = clock or (lambda: datetime.now(UTC))

    @staticmethod
    def _snapshot(row: AgentToolEffect, *, fresh: bool = False) -> dict:
        return {"action_id": row.action_id, "effect_id": row.action_id, "fresh": fresh,
                "account_id": row.account_id,
                "scope": row.scope, "run_id": row.run_id, "wake_id": row.wake_id, "tool_call_id": row.tool_call_id,
                "operation": row.operation, "payload_hash": row.payload_hash, "status": row.status,
                "reservation_attempts": row.reservation_attempts,
                "source_message_ids": list(row.source_message_ids), "receipt": _json_copy(row.receipt),
                "reason": row.reason, "replay_allowed": row.status == "reserved",
                "created_at": row.created_at.replace(tzinfo=UTC).isoformat(),
                "updated_at": row.updated_at.replace(tzinfo=UTC).isoformat(),
                "dispatched_at": row.dispatched_at.replace(tzinfo=UTC).isoformat() if row.dispatched_at else None}

    def _unresolved(self, session, scope: str, operation: str, payload_hash: str, *, exclude=None):
        # Public platform effects cannot be replayed by changing authority or
        # the per-friend budget namespace. Quotas remain separate elsewhere.
        public_scope = scope.removeprefix('owner:')
        scopes = (scope,)
        if public_scope.startswith('public:'):
            base=public_scope.split(':friend:',1)[0]
            scopes = tuple(dict.fromkeys((public_scope, 'owner:' + public_scope, base, 'owner:' + base)))
        query = select(AgentToolEffect).where(
            AgentToolEffect.account_id == self.account_id, AgentToolEffect.scope.in_(scopes),
            AgentToolEffect.operation == operation, AgentToolEffect.payload_hash == payload_hash,
            AgentToolEffect.status.in_(("dispatching", "uncertain")))
        if exclude is not None:
            query = query.where(AgentToolEffect.action_id != exclude)
        return session.scalar(query.order_by(AgentToolEffect.updated_at.desc(), AgentToolEffect.action_id).limit(1))

    def _owned(self, session, effect_id: str):
        if not isinstance(effect_id, str) or not effect_id or len(effect_id) > 80:
            raise ValueError("invalid effect identifier")
        row = session.get(AgentToolEffect, effect_id)
        return row if row and row.account_id == self.account_id else None

    def reserve(self, scope: str, *, run_id: str, tool_call_id: str, operation: str,
                payload_hash: str, sources=(), initial_receipt=None) -> dict:
        """Persist initial claim metadata atomically, only for a fresh attempt.

        Metadata is not a platform receipt: only a settled sent effect with
        its actual receipt confirms delivery. Existing unknown claims retain
        their original metadata and fence.
        """
        AutonomousRepository._validate_scope(scope)
        for value, limit in ((run_id, 160), (tool_call_id, 200), (operation, 80)):
            if not isinstance(value, str) or not value.strip() or len(value) > limit or any(ord(c) < 32 for c in value):
                raise ValueError("invalid effect identity")
        if not isinstance(payload_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", payload_hash):
            raise ValueError("invalid effect payload hash")
        if (not isinstance(sources, (tuple, list)) or len(sources) > 200
                or any(type(source) is not int or not 1 <= source <= 2**63 - 1 for source in sources)
                or len(set(sources)) != len(sources)):
            raise ValueError("invalid effect sources")
        copied_receipt = _json_copy(initial_receipt)
        identity = json.dumps([self.account_id, run_id, tool_call_id], separators=(",", ":"))
        effect_id = "effect-" + sha256(identity.encode("utf-8")).hexdigest()
        with self._transactions._write_scope() as session:
            row = session.get(AgentToolEffect, effect_id)
            if row is not None:
                if row.scope != scope or row.operation != operation:
                    raise ValueError("tool effect identity changed")
                repairable = (row.status in {"blocked", "failed"} and row.dispatched_at is None
                              and row.reservation_attempts < 2)
                if row.payload_hash != payload_hash and not repairable:
                    raise ValueError("tool effect identity changed")
                if repairable:
                    unresolved = self._unresolved(session, scope, operation, payload_hash)
                    if unresolved is not None:
                        return self._snapshot(unresolved)
                    row.status, row.reason, row.receipt = "reserved", "", copied_receipt
                    row.payload_hash, row.source_message_ids = payload_hash, list(sources)
                    row.reservation_attempts += 1
                    row.updated_at = utc_naive(self.clock())
                    return self._snapshot(row, fresh=True)
                return self._snapshot(row)
            # A new wake/call identity cannot resolve an unknown platform result.
            # Keep the original sources and receipt rather than manufacturing a
            # new authority to repeat the same external write.
            unresolved = self._unresolved(session, scope, operation, payload_hash)
            if unresolved is not None:
                return self._snapshot(unresolved)
            now = utc_naive(self.clock())
            row = AgentToolEffect(action_id=effect_id, account_id=self.account_id, scope=scope,
                run_id=run_id, tool_call_id=tool_call_id, operation=operation, payload_hash=payload_hash,
                status="reserved", reservation_attempts=1, source_message_ids=list(sources), receipt=copied_receipt, reason="",
                created_at=now, updated_at=now)
            session.add(row)
            session.flush()
            return self._snapshot(row, fresh=True)

    def read(self, effect_id: str) -> dict:
        from app.storage.db import session_scope
        with session_scope(self.engine) as session:
            row = self._owned(session, effect_id)
            return self._snapshot(row) if row else {}

    get = read

    def media_delivery_claims(self, scope: str, run_id: str):
        """Read legacy bundle fences for this account and canonical request."""
        from app.storage.db import session_scope
        with session_scope(self.engine) as session:
            rows=session.scalars(select(AgentToolEffect).where(AgentToolEffect.account_id==self.account_id,
                AgentToolEffect.scope==scope,AgentToolEffect.run_id==run_id,
                AgentToolEffect.operation=='media_delivery_once').limit(64))
            return tuple(self._snapshot(row) for row in rows)

    def recent(self, scope: str, *, limit: int = 16) -> tuple[dict, ...]:
        """Return scoped real effect metadata for a resumed/rotated session."""
        from app.storage.db import session_scope
        AutonomousRepository._validate_scope(scope)
        if type(limit) is not int or not 1 <= limit <= 64:
            raise ValueError("invalid recent tool effect limit")
        with session_scope(self.engine) as session:
            query = select(AgentToolEffect).where(
                AgentToolEffect.account_id == self.account_id, AgentToolEffect.scope == scope
            ).order_by(AgentToolEffect.updated_at.desc(), AgentToolEffect.action_id).limit(limit)
            return tuple(self._snapshot(row) for row in session.scalars(query))

    def self_publications(self, *, date: str | None = None) -> dict:
        """Only this account's actual publication receipts, never wake prose.

        Counts cover the selected Shanghai day; record details are bounded.
        Absence covers this system's writes, not manual activity on QQ.
        """
        from app.storage.db import session_scope
        now = self.clock().astimezone(ASIA_SHANGHAI)
        if date is None or date == 'today':
            day = now.date()
        else:
            if not isinstance(date, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
                raise ValueError('invalid self activity date')
            day = calendar_date.fromisoformat(date)
            if day > now.date():
                raise ValueError('self activity date is in the future')
        start = datetime.combine(day, datetime.min.time(), ASIA_SHANGHAI)
        end = start + timedelta(days=1)

        def state(status, dispatched, receipt_valid):
            return case(
                ((status == 'sent') & dispatched.is_not(None) & receipt_valid, 'confirmed'),
                (status.in_(('sent', 'dispatching', 'uncertain')), 'unknown'),
                (status.in_(('failed', 'blocked')), 'failed'),
                else_='pending')

        effect = AgentToolEffect
        actual_receipt = func.json_extract(effect.receipt, '$')
        effect_at = func.coalesce(effect.dispatched_at, effect.created_at)
        effects = select(effect.action_id.label('action_id'), effect_at.label('at'),
            actual_receipt.label('receipt_id'), state(effect.status, effect.dispatched_at,
                (func.json_type(effect.receipt) == 'text') & (func.length(func.trim(actual_receipt)) > 0)).label('state'))
        effects = effects.where(effect.account_id == self.account_id, effect.operation == 'publish_qzone',
            effect.scope.in_(('public:qzone_publish', 'owner:public:qzone_publish', 'qzone_publish', 'public')),
            effect_at >= utc_naive(start), effect_at < utc_naive(end))

        old = SocialActionRecord
        old_at = func.coalesce(old.dispatched_at, old.created_at)
        # A tool wake can have a sent envelope: only its fenced tool is a fact.
        has_tool = select(effect.action_id).where(effect.account_id == self.account_id,
            effect.operation == 'publish_qzone', effect.run_id == old.action_id).exists()
        legacy = select(old.action_id.label('action_id'), old_at.label('at'),
            old.receipt_id.label('receipt_id'), state(old.status, old.dispatched_at,
                func.length(func.trim(old.receipt_id)) > 0).label('state')).where(
            old.account_id == self.account_id, old.scope_id == self.account_id,
            old.kind == 'qzone_publish', old.action == 'publish_qzone', ~has_tool,
            old_at >= utc_naive(start), old_at < utc_naive(end))
        publications = union_all(effects, legacy).subquery()
        with session_scope(self.engine) as session:
            counts = dict(session.execute(select(publications.c.state, func.count()).group_by(publications.c.state)).all())
            rows = session.execute(select(publications).order_by(
                publications.c.at.desc(), publications.c.action_id).limit(64)).all()
        confirmed, unknown = counts.get('confirmed', 0), counts.get('unknown', 0)
        return {'kind': 'own_qzone_publication_receipts', 'date': day.isoformat(), 'timezone': 'Asia/Shanghai',
            'as_of': now.isoformat(), 'coverage': 'system_publication_receipts_only',
            'published': True if confirmed else None if unknown else False,
            'confirmed_count': confirmed, 'unknown_count': unknown,
            'failed_count': counts.get('failed', 0), 'pending_count': counts.get('pending', 0),
            'records_truncated': sum(counts.values()) > len(rows),
            'records': [{'action_id': row.action_id, 'status': row.state,
                'at': row.at.replace(tzinfo=UTC).astimezone(ASIA_SHANGHAI).isoformat(),
                'receipt_id': row.receipt_id if row.state == 'confirmed' else None} for row in rows]}

    def _day_start(self) -> datetime:
        return utc_naive(self.clock().astimezone(ASIA_SHANGHAI).replace(hour=0, minute=0, second=0, microsecond=0))

    def count_dispatched(self, *, operation: str | None = None, scope: str | None = None) -> int:
        from app.storage.db import session_scope
        if scope is not None:
            AutonomousRepository._validate_scope(scope)
        with session_scope(self.engine) as session:
            query = select(func.count()).select_from(AgentToolEffect).where(
                AgentToolEffect.account_id == self.account_id, AgentToolEffect.dispatched_at >= self._day_start())
            if operation is not None:
                query = query.where(AgentToolEffect.operation == operation)
            if scope is not None:
                query = query.where(AgentToolEffect.scope == scope)
            return session.scalar(query) or 0

    def mark_dispatching(self, effect_id: str, *, daily_limit: int | None = 2, cooldown_seconds: float = 0,
                         operation_daily_limit=None, shared_cooldown_seconds=0,
                         wake_id="", max_writes_per_wake=None, before_commit=None) -> bool:
        if daily_limit is not None and (type(daily_limit) is not int or not 0 <= daily_limit <= 500):
            raise ValueError("invalid tool effect daily limit")
        if (type(cooldown_seconds) not in {int, float} or not math.isfinite(cooldown_seconds)
                or not 0 <= cooldown_seconds <= 86400):
            raise ValueError("invalid tool operation cooldown")
        self._validate_space_limits(operation_daily_limit, shared_cooldown_seconds, wake_id, max_writes_per_wake)
        with self._transactions._write_scope(before_commit=before_commit) as session:
            row = self._owned(session, effect_id)
            if row is None or row.status != "reserved":
                return False
            now = utc_naive(self.clock())
            # Reservations may precede either dispatch. Recheck while holding
            # the write transaction so two separately reserved wakes cannot
            # race into the same unresolved platform effect.
            unresolved = self._unresolved(session, row.scope, row.operation, row.payload_hash,
                                          exclude=row.action_id)
            if unresolved is not None:
                row.status, row.reason, row.updated_at = "blocked", "prior_unknown_effect", now
                row.receipt = {"prior_effect_id": unresolved.action_id,
                               "prior_status": unresolved.status, "replay_allowed": False}
                return False
            count = session.scalar(select(func.count()).select_from(AgentToolEffect).where(
                AgentToolEffect.account_id == self.account_id, AgentToolEffect.scope == row.scope,
                AgentToolEffect.dispatched_at >= self._day_start())) or 0
            latest = session.scalar(select(func.max(AgentToolEffect.dispatched_at)).where(
                AgentToolEffect.account_id == self.account_id, AgentToolEffect.scope == row.scope,
                AgentToolEffect.operation == row.operation))
            reason = ("daily_scope_budget_exhausted" if daily_limit is not None and count >= daily_limit else
                      "operation_cooldown" if latest and latest + timedelta(seconds=cooldown_seconds) > now else "")
            if reason:
                row.status, row.reason, row.updated_at = "blocked", reason, now
                return False
            reason = self._space_budget_reason(session, row.scope, row.operation,
                operation_daily_limit, shared_cooldown_seconds, wake_id, max_writes_per_wake)
            if reason:
                row.status, row.reason, row.updated_at = "blocked", reason, now
                return False
            row.wake_id = wake_id
            row.status, row.dispatched_at, row.updated_at = "dispatching", now, now
            return True

    @staticmethod
    def _validate_space_limits(limit, cooldown, wake_id, max_writes):
        if limit is not None and (type(limit) is not int or not 0 <= limit <= 500):
            raise ValueError("invalid operation daily limit")
        if type(cooldown) not in {int, float} or not math.isfinite(cooldown) or not 0 <= cooldown <= 86400:
            raise ValueError("invalid shared cooldown")
        if not isinstance(wake_id, str) or len(wake_id) > 160 or any(ord(c) < 32 for c in wake_id):
            raise ValueError("invalid wake identity")
        if max_writes is not None and (type(max_writes) is not int or not 1 <= max_writes <= 32 or not wake_id):
            raise ValueError("invalid per-wake write limit")

    def _space_rows(self):
        return (AgentToolEffect.account_id == self.account_id,
            AgentToolEffect.dispatched_at.is_not(None),
            or_(AgentToolEffect.scope == "public:qzone_publish",
                AgentToolEffect.scope == "public:qzone_comment",
                AgentToolEffect.scope.like("public:qzone_comment:friend:%")),
            AgentToolEffect.operation.in_(("publish_qzone", "comment_qzone")))

    def _space_budget_reason(self, session, scope, operation, daily, cooldown, wake_id="", max_writes=None):
        if not scope.startswith("public:qzone_"):
            return ""
        common = self._space_rows()
        if daily is not None:
            count = session.scalar(select(func.count()).select_from(AgentToolEffect).where(*common,
                AgentToolEffect.operation == operation, AgentToolEffect.dispatched_at >= self._day_start())) or 0
            if count >= daily:
                return "space_operation_daily_budget"
        if cooldown:
            latest = session.scalar(select(func.max(AgentToolEffect.dispatched_at)).where(*common))
            if latest and latest + timedelta(seconds=cooldown) > utc_naive(self.clock()):
                return "space_shared_cooldown"
        if max_writes is not None:
            count = session.scalar(select(func.count()).select_from(AgentToolEffect).where(*common,
                AgentToolEffect.wake_id == wake_id)) or 0
            if count >= max_writes:
                return "space_wake_budget"
        return ""

    def public_budget_reason(self, job, *, friend_id=None) -> str:
        """Read-only preflight; the transaction repeats these checks at dispatch."""
        from app.storage.db import session_scope
        operation = "publish_qzone" if job.kind == "qzone_publish" else "comment_qzone"
        scope = "public:" + job.kind
        if friend_id is not None:
            scope += ":friend:" + str(friend_id)
        with session_scope(self.engine) as session:
            reason = self._space_budget_reason(session, scope, operation, job.max_daily,
                job.shared_cooldown_seconds)
            if reason:
                return reason
            if job.kind == "qzone_publish" or friend_id is not None:
                limit = job.max_daily_per_friend if friend_id is not None else job.max_daily
                count = session.scalar(select(func.count()).select_from(AgentToolEffect).where(
                    AgentToolEffect.account_id == self.account_id, AgentToolEffect.scope == scope,
                    AgentToolEffect.dispatched_at >= self._day_start())) or 0
                if limit and count >= limit:
                    return "daily_scope_budget_exhausted"
            if job.kind == "qzone_publish":
                latest = session.scalar(select(func.max(AgentToolEffect.dispatched_at)).where(
                    AgentToolEffect.account_id == self.account_id, AgentToolEffect.scope == scope,
                    AgentToolEffect.operation == operation))
                if latest and latest + timedelta(seconds=job.cooldown_seconds) > utc_naive(self.clock()):
                    return "operation_cooldown"
        return ""

    def finish(self, effect_id: str, status: str, *, receipt=None, reason: str = "") -> bool:
        if not isinstance(status, str) or status not in self.TERMINAL:
            raise ValueError("invalid terminal tool effect status")
        if not isinstance(reason, str) or len(reason) > 160 or any(ord(c) < 32 for c in reason):
            raise ValueError("invalid tool effect reason")
        copied_receipt = _json_copy(receipt)
        with self._transactions._write_scope() as session:
            row = self._owned(session, effect_id)
            if row is None or row.status in self.TERMINAL:
                return False
            if row.status == "reserved" and status in {"sent", "uncertain"}:
                raise ValueError("tool effect was never dispatched")
            row.status, row.receipt, row.reason = status, copied_receipt, reason
            row.updated_at = utc_naive(self.clock())
            if status == "uncertain":
                self._mark_messages_uncertain(session, row)
            return True

    def _mark_messages_uncertain(self, session, row: AgentToolEffect) -> None:
        message_scope = row.scope.removeprefix("owner:")
        if message_scope.startswith("group:") and message_scope[6:].isdigit():
            for message in session.scalars(select(Message).where(
                Message.group_id == int(message_scope[6:]), Message.user_id == self.account_id,
                func.json_extract(Message.raw_json, "$.delivery_state") == "reserved",
                or_(Message.platform_msg_id == "social:" + row.action_id,
                    func.json_extract(Message.raw_json, "$.social_action_id") == row.action_id),
            )):
                payload = dict(message.raw_json)
                payload.update(delivery_state="uncertain", delivery_attempts=1,
                               failure_kind="delivery_result_unknown", delivery_reason=row.reason)
                message.raw_json = payload

    def recover(self, scope: str | None = None) -> int:
        """Use after acquiring the single account lease, never during a live turn."""
        if scope is not None:
            AutonomousRepository._validate_scope(scope)
        recovered = 0
        with self._transactions._write_scope() as session:
            query = select(AgentToolEffect).where(AgentToolEffect.account_id == self.account_id,
                                                 AgentToolEffect.status == "dispatching")
            if scope is not None:
                query = query.where(AgentToolEffect.scope == scope)
            for row in session.scalars(query):
                row.status, row.reason, row.updated_at = "uncertain", "owner_lost_after_dispatch", utc_naive(self.clock())
                recovered += 1
                self._mark_messages_uncertain(session, row)
        return recovered
