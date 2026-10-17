"""Durable cost reservations and scoped working intentions, not chat facts."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4
import math
import re

from sqlalchemy import String, Integer, DateTime, JSON, Index, or_, select, func
from sqlalchemy.orm import Mapped, mapped_column, Session

from app.storage.models import Base
from app.core.time_utils import ASIA_SHANGHAI
from app.storage.social_agent import utc_naive


_UNSET = object()


class AutonomousUsage(Base):
    __tablename__ = "autonomous_usage"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    kind: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime)


class AutonomousIntention(Base):
    __tablename__ = "autonomous_intentions"
    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    account_id: Mapped[int] = mapped_column(Integer)
    scope: Mapped[str] = mapped_column(String(160))
    note: Mapped[dict] = mapped_column(JSON, default=dict)
    source_message_ids: Mapped[list] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class AutonomousIntentionWakeup(Base):
    """Additive schedule table: legacy intention rows remain untouched."""
    __tablename__ = "autonomous_intention_wakeups"
    __table_args__ = (Index("ix_autonomous_intention_due", "account_id", "next_due"),)
    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    account_id: Mapped[int] = mapped_column(Integer)
    scope: Mapped[str] = mapped_column(String(160))
    status: Mapped[str] = mapped_column(String(24), default="active")
    next_due: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    lease_owner: Mapped[str] = mapped_column(String(80), default="")
    lease_token: Mapped[str] = mapped_column(String(64), default="")
    lease_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


@dataclass(frozen=True, slots=True)
class DueIntentionClaim:
    account_id: int
    scope: str
    working_intention: dict
    source_message_ids: tuple[int, ...]
    due_at: datetime
    owner: str
    token: str
    schedule_revision: int


class AutonomousRepository:
    def __init__(self, engine, *, account_id: int, clock=None):
        if type(account_id) is not int or not 1 <= account_id <= 2**63 - 1:
            raise ValueError("invalid autonomous account")
        self.engine, self.account_id = engine, account_id
        self.clock = clock or (lambda: datetime.now(UTC))

    @contextmanager
    def _write_scope(self, before_commit=None):
        from app.storage.db import session_scope
        with session_scope(self.engine) as session:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            yield session
            if before_commit is not None:
                session.flush()
                before_commit()

    @staticmethod
    def _validate_cost(kind: str, limit: int | None) -> None:
        budget_kind = kind.removeprefix('owner:').removeprefix('private:') if isinstance(kind, str) else kind
        model_usage = budget_kind == 'model' or isinstance(budget_kind,str) and budget_kind.startswith('model:')
        if (not isinstance(kind, str) or len(kind) > 200 or
                not (limit is None and model_usage or type(limit) is int and 0 <= limit <= 500)
                or budget_kind not in {"model", "image", "profile:nickname", "profile:avatar"}
                and not re.fullmatch(r"profile:card:[1-9][0-9]{0,18}", budget_kind)
                and not re.fullmatch(r"(?:model|image):(?:group_interactive|group_autonomous|qzone_publish|qzone_comment|profile)", budget_kind)):
            raise ValueError("invalid cost budget")
        if budget_kind.startswith("profile:card:") and int(kind.rsplit(":", 1)[1]) > 2**63 - 1:
            raise ValueError("invalid profile group")
    def cost_available(self, kind: str, limit: int | None) -> bool:
        self._validate_cost(kind, limit)
        if limit is None:return True
        start = utc_naive(self.clock().astimezone(ASIA_SHANGHAI).replace(hour=0, minute=0, second=0, microsecond=0))
        with Session(self.engine) as session:
            count = session.scalar(select(func.count()).select_from(AutonomousUsage).where(
                AutonomousUsage.account_id == self.account_id, AutonomousUsage.kind == kind,
                AutonomousUsage.created_at >= start)) or 0
        return count < limit

    def reserve_cost(self, kind: str, limit: int | None) -> bool:
        self._validate_cost(kind, limit)
        instant = self.clock()
        now = instant.astimezone(UTC).replace(tzinfo=None)
        start = instant.astimezone(ASIA_SHANGHAI).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC).replace(tzinfo=None)
        with self.engine.connect() as conn:
            conn.exec_driver_sql("BEGIN IMMEDIATE")
            with Session(bind=conn) as session:
                count = session.scalar(select(func.count()).select_from(AutonomousUsage).where(
                    AutonomousUsage.account_id == self.account_id, AutonomousUsage.kind == kind, AutonomousUsage.created_at >= start)) or 0
                if limit is not None and count >= limit:
                    conn.rollback()
                    return False
                session.add(AutonomousUsage(id=uuid4().hex, account_id=self.account_id, kind=kind, created_at=now))
                session.flush()
                conn.commit()
        return True

    def read_intention(self, scope: str) -> dict:
        self._validate_scope(scope)
        with Session(self.engine) as session:
            row = session.get(AutonomousIntention, f"{self.account_id}:{scope}")
            return {"working_intention": row.note, "source_message_ids": row.source_message_ids} if row else {}

    def write_intention(self, scope: str, note: dict, sources=(), *, next_due=_UNSET, status=_UNSET,
                        before_commit=None) -> None:
        self._validate_scope(scope)
        if (not isinstance(note, dict) or len(str(note)) > 2000 or not isinstance(sources, (list, tuple))
                or len(sources) > 200 or any(type(item) is not int or not 1 <= item <= 2**63 - 1 for item in sources)
                or len(set(sources)) != len(sources)):
            raise ValueError("working intention exceeds budget")
        schedule_requested = next_due is not _UNSET or status is not _UNSET
        if schedule_requested:
            due, state = self._validate_schedule(None if next_due is _UNSET else next_due,
                                                 "active" if status is _UNSET else status)
        with self._write_scope(before_commit=before_commit) as session:
            key = f"{self.account_id}:{scope}"
            row = session.get(AutonomousIntention, key)
            if row is None:
                row = AutonomousIntention(key=key, account_id=self.account_id, scope=scope)
                session.add(row)
            row.note, row.source_message_ids = note, list(sources)
            row.updated_at = utc_naive(self.clock())
            if schedule_requested:
                self._store_schedule(session, scope, due, state)

    @staticmethod
    def _validate_schedule(next_due, status) -> tuple[datetime | None, str]:
        if not isinstance(status, str) or status not in {"active", "waiting", "completed", "cancelled"}:
            raise ValueError("invalid goal schedule status")
        if next_due is not None and not isinstance(next_due, datetime):
            raise ValueError("goal due time must be a datetime")
        if status in {"completed", "cancelled"} and next_due is not None:
            raise ValueError("terminal goal cannot have a due time")
        return utc_naive(next_due) if next_due is not None else None, status

    @staticmethod
    def _validate_lease(owner: str, lease_seconds: float) -> None:
        if not isinstance(owner, str) or not owner.strip() or len(owner) > 80 or any(ord(c) < 32 for c in owner):
            raise ValueError("invalid intention lease owner")
        if (type(lease_seconds) not in {int, float} or not math.isfinite(lease_seconds)
                or not 1 <= lease_seconds <= 3600):
            raise ValueError("invalid intention lease duration")

    def _store_schedule(self, session, scope: str, due: datetime | None, status: str) -> None:
        key = f"{self.account_id}:{scope}"
        row = session.get(AutonomousIntentionWakeup, key)
        if row is None:
            row = AutonomousIntentionWakeup(key=key, account_id=self.account_id, scope=scope, revision=1)
            session.add(row)
        else:
            row.revision += 1
        row.next_due, row.status, row.updated_at = due, status, utc_naive(self.clock())

    def schedule_intention(self, scope: str, *, next_due: datetime | None, status: str = "active",
                           before_commit=None) -> None:
        """Only the Agent chooses a future due time; None means dormant."""
        self._validate_scope(scope)
        due, state = self._validate_schedule(next_due, status)
        with self._write_scope(before_commit=before_commit) as session:
            if session.get(AutonomousIntention, f"{self.account_id}:{scope}") is None:
                raise ValueError("cannot schedule a missing intention")
            self._store_schedule(session, scope, due, state)

    def read_intention_schedule(self, scope: str) -> dict:
        self._validate_scope(scope)
        with Session(self.engine) as session:
            row = session.get(AutonomousIntentionWakeup, f"{self.account_id}:{scope}")
            return {"status": row.status, "next_due": row.next_due.replace(tzinfo=UTC) if row.next_due else None,
                    "revision": row.revision} if row else {}

    def list_intentions(self, *, scopes: tuple[str, ...] | None = None, limit: int = 64) -> tuple[dict, ...]:
        """Read only, JSON-ready goals; unscheduled legacy intentions stay dormant."""
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("invalid intention list limit")
        if scopes is not None:
            if not isinstance(scopes, tuple) or len(scopes) > 256:
                raise ValueError("invalid intention scopes")
            for scope in scopes:
                self._validate_scope(scope)
        with Session(self.engine) as session:
            query = select(AutonomousIntention, AutonomousIntentionWakeup).outerjoin(
                AutonomousIntentionWakeup, AutonomousIntention.key == AutonomousIntentionWakeup.key).where(
                AutonomousIntention.account_id == self.account_id)
            if scopes is not None:
                query = query.where(AutonomousIntention.scope.in_(scopes))
            return tuple({"scope": intention.scope, "working_intention": intention.note,
                "source_message_ids": intention.source_message_ids, "status": wakeup.status if wakeup else "active",
                "next_due": wakeup.next_due.replace(tzinfo=UTC).isoformat() if wakeup and wakeup.next_due else None,
                "revision": wakeup.revision if wakeup else 0}
                for intention, wakeup in session.execute(query.order_by(AutonomousIntention.scope).limit(limit)))

    def claim_due_intentions(self, owner: str, *, limit: int = 16, lease_seconds: float = 300,
                             scopes: tuple[str, ...] | None = None) -> tuple[DueIntentionClaim, ...]:
        self._validate_lease(owner, lease_seconds)
        if type(limit) is not int or not 1 <= limit <= 64:
            raise ValueError("invalid due intention limit")
        if scopes is not None:
            if not isinstance(scopes, tuple) or len(scopes) > 256:
                raise ValueError("invalid permitted intention scopes")
            for scope in scopes:
                self._validate_scope(scope)
        now = utc_naive(self.clock())
        claims = []
        with self._write_scope() as session:
            statement = select(AutonomousIntentionWakeup).join(
                AutonomousIntention, AutonomousIntentionWakeup.key == AutonomousIntention.key).where(
                AutonomousIntentionWakeup.account_id == self.account_id,
                AutonomousIntention.account_id == self.account_id,
                AutonomousIntentionWakeup.status.in_(("active", "waiting")),
                AutonomousIntentionWakeup.next_due <= now,
                or_(AutonomousIntentionWakeup.lease_until.is_(None), AutonomousIntentionWakeup.lease_until <= now),
            )
            if scopes is not None:
                statement = statement.where(AutonomousIntentionWakeup.scope.in_(scopes))
            for row in session.scalars(statement.order_by(AutonomousIntentionWakeup.next_due,
                                                         AutonomousIntentionWakeup.key).limit(limit)):
                intention = session.get(AutonomousIntention, row.key)
                row.lease_owner, row.lease_token = owner, uuid4().hex
                row.lease_until = now + timedelta(seconds=lease_seconds)
                row.updated_at = now
                claims.append(DueIntentionClaim(self.account_id, row.scope, intention.note,
                    tuple(intention.source_message_ids), row.next_due.replace(tzinfo=UTC),
                    owner, row.lease_token, row.revision))
        return tuple(claims)

    def _claimed_row(self, session, claim: DueIntentionClaim):
        if not isinstance(claim, DueIntentionClaim) or claim.account_id != self.account_id:
            return None
        row = session.get(AutonomousIntentionWakeup, f"{self.account_id}:{claim.scope}")
        if (row is None or row.lease_token != claim.token or row.lease_owner != claim.owner
                or row.lease_until is None or row.lease_until <= utc_naive(self.clock())):
            return None
        return row

    def renew_intention_claim(self, claim: DueIntentionClaim, *, lease_seconds: float = 300) -> bool:
        self._validate_lease(claim.owner, lease_seconds)
        with self._write_scope() as session:
            row = self._claimed_row(session, claim)
            if row is None:
                return False
            row.lease_until = utc_naive(self.clock()) + timedelta(seconds=lease_seconds)
            return True

    def finish_intention_claim(self, claim: DueIntentionClaim, *, consume: bool = True,
                               retry_at: datetime | None = None) -> bool:
        """Consume this due event, preserving a new Agent-selected schedule."""
        if type(consume) is not bool:
            raise ValueError("invalid intention consumption flag")
        if retry_at is not None and not isinstance(retry_at, datetime):
            raise ValueError("invalid intention retry time")
        if consume and retry_at is not None:
            raise ValueError("consumed intention cannot have a retry time")
        retry = utc_naive(retry_at) if retry_at is not None else None
        with self._write_scope() as session:
            row = self._claimed_row(session, claim)
            if row is None:
                return False
            if row.revision == claim.schedule_revision:
                if consume or retry is not None:
                    row.next_due = None if consume else retry
                    row.revision += 1
            row.lease_owner, row.lease_token, row.lease_until = "", "", None
            row.updated_at = utc_naive(self.clock())
            return True

    def invalidate_intention_source(self, scope: str, message_id: int) -> bool:
        self._validate_scope(scope)
        if type(message_id) is not int or not 1 <= message_id <= 2**63 - 1:
            raise ValueError("invalid intention source")
        with self._write_scope() as session:
            key = f"{self.account_id}:{scope}"
            row = session.get(AutonomousIntention, key)
            if row is None or message_id not in row.source_message_ids:
                return False
            wakeup = session.get(AutonomousIntentionWakeup, key)
            if wakeup:
                session.delete(wakeup)
            session.delete(row)
            return True

    @staticmethod
    def _validate_scope(scope: str) -> None:
        if not isinstance(scope, str) or not scope.strip() or len(scope) > 160 or any(ord(char) < 32 for char in scope):
            raise ValueError("invalid working intention scope")
