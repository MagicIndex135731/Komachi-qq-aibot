"""Durable scheduled participation, exclusive account leases and write ledger."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from random import SystemRandom
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint, and_, cast, delete, func, literal, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.elements import ColumnElement

from app.core.memory_source_guard import is_usable_memory_source
from app.storage.models import Base, Message

if TYPE_CHECKING:
    from app.agent.social_contracts import SocialDecision, SocialJob, SocialPost


def utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("social runtime timestamps must include a timezone")
    return value.astimezone(UTC).replace(tzinfo=None)


def social_provisional_output_clause(account_id: int) -> ColumnElement[bool]:
    """Only our internal, unconfirmed output rows can be skipped as arrivals.

    Recalled inbound messages and actual QQ echoes remain visible to the source
    guard. An orphaned output is not a new chat event and cannot ground memory.
    """
    action_id = func.coalesce(func.json_extract(Message.raw_json, "$.social_action_id"), "")
    state = func.coalesce(func.json_extract(Message.raw_json, "$.delivery_state"), "")
    return and_(
        Message.user_id == account_id,
        func.coalesce(func.json_type(Message.raw_json, "$.social_action_id"), "") == "text",
        action_id != "",
        Message.platform_msg_id == literal("social:") + cast(action_id, String),
        state.in_(("reserved", "deleted", "uncertain")),
    )


class SocialAccountLease(Base):
    __tablename__ = "social_account_leases"

    account_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner: Mapped[str] = mapped_column(String(80), default="")
    epoch: Mapped[int] = mapped_column(Integer, default=0)
    lease_until: Mapped[datetime] = mapped_column(DateTime)


class SocialScheduledJob(Base):
    __tablename__ = "social_scheduled_jobs"
    __table_args__ = (
        UniqueConstraint("account_id", "job_key", name="ux_social_scheduled_job"),
        Index("ix_social_scheduled_job_due", "account_id", "next_due"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("social_account_leases.account_id"))
    job_key: Mapped[str] = mapped_column(String(160))
    kind: Mapped[str] = mapped_column(String(24))
    scope_id: Mapped[int] = mapped_column(Integer)
    next_due: Mapped[datetime] = mapped_column(DateTime)
    last_status: Mapped[str] = mapped_column(String(24), default="scheduled")
    last_reason: Mapped[str] = mapped_column(String(160), default="")


class SocialActionRecord(Base):
    __tablename__ = "social_action_records"
    __table_args__ = (
        UniqueConstraint("account_id", "dedupe_key", name="ux_social_action_dedupe"),
        Index("ix_social_action_budget", "account_id", "kind", "scope_id", "created_at"),
    )

    action_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("social_account_leases.account_id"))
    owner: Mapped[str] = mapped_column(String(80))
    epoch: Mapped[int] = mapped_column(Integer)
    job_key: Mapped[str] = mapped_column(String(160))
    kind: Mapped[str] = mapped_column(String(24))
    scope_id: Mapped[int] = mapped_column(Integer)
    dedupe_key: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(24))
    text: Mapped[str] = mapped_column(Text)
    post_id: Mapped[str] = mapped_column(String(256), default="")
    post_owner_id: Mapped[int] = mapped_column(Integer, default=0)
    source_message_ids: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(24), default="reserved")
    receipt_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    reason: Mapped[str] = mapped_column(String(160), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime)
    dispatched_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class SocialPatrolReview(Base):
    """A short-lived observation cache; never a chat or memory fact."""
    __tablename__ = "social_patrol_reviews"
    account_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), primary_key=True)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime)


class SocialPatrolCandidate(Base):
    """Version-scoped attention, independent of the durable send ledger."""
    __tablename__ = 'social_patrol_candidates'
    account_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(24))
    revisit_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


@dataclass(frozen=True, slots=True)
class SocialLease:
    owner: str
    epoch: int


@dataclass(frozen=True, slots=True)
class DueSocialJob:
    job: SocialJob
    due_at: datetime


class SocialAgentRepository:
    TERMINAL = frozenset({"sent", "uncertain", "failed", "blocked", "silent"})
    MAX_JOBS_PER_TICK = 16
    MAX_SOURCE_IDS = 200

    def __init__(self, engine: Engine, *, random_seconds=None) -> None:
        self.engine = engine
        self.random_seconds = random_seconds or SystemRandom().uniform

    @contextmanager
    def _write_scope(self):
        from app.storage.db import session_scope

        with session_scope(self.engine) as session:
            # Serialize lease/limits/idempotency reads with their updates. No
            # model, network or policy callback executes while this lock is held.
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            yield session

    @staticmethod
    def _owns(session, *, account_id, owner, epoch, now) -> bool:
        row = session.get(SocialAccountLease, account_id)
        return bool(row is not None and row.owner == owner and row.epoch == epoch and row.lease_until > now)

    @staticmethod
    def _recover_actions(session, account_id: int, now: datetime) -> None:
        for row in session.scalars(select(SocialActionRecord).where(
            SocialActionRecord.account_id == account_id,
            SocialActionRecord.status.in_(["reserved", "dispatching"]),
        )):
            row.status = "uncertain" if row.status == "dispatching" else "failed"
            row.reason = "owner_lost_after_dispatch" if row.status == "uncertain" else "owner_lost_before_dispatch"
            row.updated_at = now
            if row.kind == "group" and row.action == "send_group":
                message = session.scalar(select(Message).where(
                    Message.group_id == row.scope_id,
                    Message.platform_msg_id == "social:" + row.action_id,
                    social_provisional_output_clause(account_id),
                    func.json_extract(Message.raw_json, "$.social_action_id") == row.action_id,
                    func.json_extract(Message.raw_json, "$.delivery_state") == "reserved",
                ))
                if message is not None:
                    payload = dict(message.raw_json)
                    payload.update(delivery_state="uncertain" if row.status == "uncertain" else "deleted",
                                   delivery_attempts=1 if row.status == "uncertain" else 0,
                                   delivery_reason=row.reason)
                    if row.status == "uncertain":
                        payload["failure_kind"] = "delivery_result_unknown"
                    message.raw_json = payload

    def acquire(self, *, account_id: int, owner: str, now: datetime, lease_seconds: float) -> SocialLease | None:
        current = utc_naive(now)
        with self._write_scope() as session:
            row = session.get(SocialAccountLease, account_id)
            if row is not None and row.owner != owner and row.owner and row.lease_until > current:
                return None
            if row is None:
                row = SocialAccountLease(account_id=account_id, owner=owner, epoch=1,
                                         lease_until=current + timedelta(seconds=lease_seconds))
                session.add(row)
            else:
                if row.owner != owner or row.lease_until <= current:
                    row.epoch += 1
                    self._recover_actions(session, account_id, current)
                row.owner = owner
                row.lease_until = current + timedelta(seconds=lease_seconds)
            session.flush()
            return SocialLease(owner, row.epoch)

    def owns(self, *, account_id: int, owner: str, epoch: int, now: datetime) -> bool:
        from app.storage.db import session_scope

        with session_scope(self.engine) as session:
            return self._owns(session, account_id=account_id, owner=owner, epoch=epoch, now=utc_naive(now))

    def release(self, *, account_id: int, owner: str, epoch: int, now: datetime) -> None:
        with self._write_scope() as session:
            row = session.get(SocialAccountLease, account_id)
            if row is None or row.owner != owner or row.epoch != epoch:
                return
            self._recover_actions(session, account_id, utc_naive(now))
            row.owner = ""
            row.epoch += 1
            row.lease_until = utc_naive(now)

    def claim_due_jobs(
        self, *, account_id: int, owner: str, epoch: int, jobs: tuple[SocialJob, ...], now: datetime,
    ) -> tuple[DueSocialJob, ...]:
        current = utc_naive(now)
        by_key = {job.key: job for job in jobs}
        if len(by_key) != len(jobs) or len(jobs) > 256:
            raise ValueError("social jobs must have unique keys and at most 256 entries")
        with self._write_scope() as session:
            if not self._owns(session, account_id=account_id, owner=owner, epoch=epoch, now=current):
                return ()
            persisted = {row.job_key: row for row in session.scalars(select(SocialScheduledJob).where(
                SocialScheduledJob.account_id == account_id,
                SocialScheduledJob.job_key.in_((*by_key, "qzone_patrol")),
            ))}
            for job in jobs:
                row = persisted.get(job.key)
                if row is None:
                    row = SocialScheduledJob(account_id=account_id, job_key=job.key, kind=job.kind,
                                             scope_id=job.scope_id, next_due=current + timedelta(seconds=job.interval_seconds))
                    session.add(row)
                    persisted[job.key] = row
                elif row.kind != job.kind or row.scope_id != job.scope_id:
                    row.kind, row.scope_id = job.kind, job.scope_id
                    row.next_due = current + timedelta(seconds=job.interval_seconds)
            patrol = tuple(job for job in jobs if job.patrol_interval_seconds is not None)
            patrol_keys = {job.key for job in patrol}
            patrol_due = ()
            if patrol:
                from app.agent.social_cadence import next_patrol_due
                from app.agent.social_policy import in_quiet_hours
                from app.core.time_utils import ASIA_SHANGHAI
                policy = lambda job: (job.patrol_interval_seconds, job.morning_delay_seconds, job.quiet_hours)
                if any(policy(job) != policy(patrol[0]) for job in patrol):
                    raise ValueError("space patrol policies must agree")
                template = patrol[0]
                fingerprint = "patrol-v1:" + sha256(json.dumps(policy(template)).encode()).hexdigest()[:24]
                anchor = persisted.get("qzone_patrol")
                if anchor is None:
                    anchor = SocialScheduledJob(account_id=account_id, job_key="qzone_patrol", kind="qzone_patrol",
                        scope_id=account_id, next_due=utc_naive(next_patrol_due(template, now, self.random_seconds, initial=True)),
                        last_reason=fingerprint)
                    session.add(anchor)
                elif anchor.last_reason != fingerprint:
                    anchor.next_due = utc_naive(next_patrol_due(template, now, self.random_seconds, initial=True))
                    anchor.last_reason = fingerprint
                elif anchor.next_due <= current:
                    local = now.astimezone(ASIA_SHANGHAI)
                    previous = anchor.next_due.replace(tzinfo=UTC).astimezone(ASIA_SHANGHAI)
                    missed_night = previous.date() < local.date() or in_quiet_hours(template.quiet_hours, previous)
                    if in_quiet_hours(template.quiet_hours, now) or missed_night:
                        morning = False
                        if missed_night and template.quiet_hours and not in_quiet_hours(template.quiet_hours, now):
                            from datetime import time
                            end = time.fromisoformat(template.quiet_hours[1])
                            morning_end = local.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
                            morning = local <= morning_end + timedelta(seconds=template.morning_delay_seconds[1])
                        anchor.next_due = utc_naive(next_patrol_due(template, now, self.random_seconds, morning=morning))
                    else:
                        patrol_due = tuple(DueSocialJob(job, anchor.next_due.replace(tzinfo=UTC)) for job in patrol)
                        anchor.next_due = utc_naive(next_patrol_due(template, now, self.random_seconds))
                for key in patrol_keys:
                    persisted[key].next_due = anchor.next_due
            due = sorted((row for key, row in persisted.items() if key in by_key and key not in patrol_keys and row.next_due <= current),
                         key=lambda row: (row.next_due, row.job_key))[:self.MAX_JOBS_PER_TICK]
            result = []
            for row in due:
                job = by_key[row.job_key]
                result.append(DueSocialJob(job, row.next_due.replace(tzinfo=UTC)))
                row.next_due = current + timedelta(seconds=job.interval_seconds)
                row.last_status, row.last_reason = "observing", ""
            return (*result, *patrol_due)

    def unreviewed_posts(self, *, account_id: int, posts: tuple[SocialPost, ...], now=None) -> tuple[SocialPost, ...]:
        from app.agent.social_cadence import post_fingerprint
        from app.storage.db import session_scope
        if len(posts) > 100:
            raise ValueError("patrol observation exceeds budget")
        fingerprints = {post_fingerprint(post) for post in posts}
        current = utc_naive(now or datetime.now(UTC))
        with session_scope(self.engine) as session:
            seen = set(session.scalars(select(SocialPatrolCandidate.fingerprint).where(
                SocialPatrolCandidate.account_id == account_id,
                SocialPatrolCandidate.fingerprint.in_(fingerprints), SocialPatrolCandidate.revisit_at > current)))
        return tuple(post for post in posts if post_fingerprint(post) not in seen)

    def assess_post(self, *, account_id, post, status, now, minutes=30, before_commit=None):
        if status not in {'deferred', 'no_response'} or type(minutes) is not int or not 5 <= minutes <= 180:
            raise ValueError('invalid_patrol_assessment')
        self._candidate_state(account_id=account_id, posts=(post,), status=status,
            now=now, minutes=minutes, before_commit=before_commit)

    def note_posts_offered(self, *, account_id, posts, now):
        # Availability is not an evaluation. A short local admission pause
        # prevents unchanged packets being polled repeatedly, without teaching
        # the personality that these posts were rejected.
        self._candidate_state(account_id=account_id, posts=posts, status='offered', now=now, minutes=180)

    def _candidate_state(self, *, account_id, posts, status, now, minutes, before_commit=None):
        from app.agent.social_cadence import post_fingerprint
        if len(posts) > 100:
            raise ValueError('patrol observation exceeds budget')
        current = utc_naive(now)
        with self._write_scope() as session:
            session.execute(delete(SocialPatrolCandidate).where(SocialPatrolCandidate.account_id == account_id,
                SocialPatrolCandidate.updated_at < current - timedelta(days=2)))
            for post in posts:
                fingerprint = post_fingerprint(post)
                row = session.get(SocialPatrolCandidate, (account_id, fingerprint))
                if row is None:
                    row = SocialPatrolCandidate(account_id=account_id, fingerprint=fingerprint)
                    session.add(row)
                elif status == 'offered' and row.revisit_at > current:
                    continue  # Do not overwrite an explicit assessment.
                row.status, row.updated_at = status, current
                row.revisit_at = current + timedelta(minutes=minutes)
            if before_commit:
                before_commit()

    def mark_posts_reviewed(self, *, account_id: int, posts: tuple[SocialPost, ...], now: datetime) -> None:
        from app.agent.social_cadence import post_fingerprint
        if len(posts) > 100:
            raise ValueError("patrol observation exceeds budget")
        current = utc_naive(now)
        with self._write_scope() as session:
            session.execute(delete(SocialPatrolReview).where(SocialPatrolReview.account_id == account_id,
                SocialPatrolReview.reviewed_at < current - timedelta(days=2)))
            for post in posts:
                fingerprint = post_fingerprint(post)
                row = session.get(SocialPatrolReview, (account_id, fingerprint))
                if row is None:
                    session.add(SocialPatrolReview(account_id=account_id, fingerprint=fingerprint, reviewed_at=current))
                else:
                    row.reviewed_at = current

    def note_job(self, *, account_id: int, owner: str, epoch: int, job_key: str, status: str, reason: str,
                 now: datetime) -> None:
        with self._write_scope() as session:
            if not self._owns(session, account_id=account_id, owner=owner, epoch=epoch, now=utc_naive(now)):
                return
            row = session.scalar(select(SocialScheduledJob).where(SocialScheduledJob.account_id == account_id,
                                                                SocialScheduledJob.job_key == job_key))
            if row is not None:
                row.last_status, row.last_reason = status[:24], reason[:160]

    def budget_available(self, *, account_id: int, owner: str, epoch: int, job: SocialJob, now: datetime) -> bool:
        from app.storage.db import session_scope

        with session_scope(self.engine) as session:
            return self._owns(session, account_id=account_id, owner=owner, epoch=epoch, now=utc_naive(now)) and (job.autonomous_tools or self._budget_allowed(
                session, account_id=account_id, job=job, now=now,
            ))

    def sources_usable(self, job: SocialJob, source_ids: tuple[int, ...]) -> bool:
        from app.storage.db import session_scope

        with session_scope(self.engine) as session:
            return self._sources_usable(session, job, source_ids)

    def action_budget_available(self, *, account_id: int, job: SocialJob, now: datetime, own_action_id=None) -> bool:
        from app.storage.db import session_scope
        with session_scope(self.engine) as session:
            return self._budget_allowed(session, account_id=account_id, job=job, now=now, own_action_id=own_action_id)

    def unhandled_posts(self, *, account_id: int, posts: tuple[SocialPost, ...]) -> tuple[SocialPost, ...]:
        from app.storage.db import session_scope

        keys = {(post.owner_id, post.post_id) for post in posts}
        if not keys:
            return ()
        with session_scope(self.engine) as session:
            candidates = session.execute(select(SocialActionRecord.post_owner_id, SocialActionRecord.post_id,
                SocialActionRecord.action, SocialActionRecord.dispatched_at).where(
                SocialActionRecord.account_id == account_id, SocialActionRecord.kind == "qzone_comment",
                SocialActionRecord.post_owner_id.in_({key[0] for key in keys}),
                SocialActionRecord.post_id.in_({key[1] for key in keys}),
            ))
            handled = {(owner, post_id) for owner, post_id, action, dispatched in candidates
                       if action != "run_tools" or dispatched is not None}
        return tuple(post for post in posts if (post.owner_id, post.post_id) not in handled)

    @classmethod
    def _sources_usable(cls, session, job: SocialJob, source_ids) -> bool:
        if not source_ids:
            return True
        if job.kind != "group" or len(source_ids) > cls.MAX_SOURCE_IDS or any(type(value) is not int or value <= 0 for value in source_ids):
            return False
        if len(set(source_ids)) != len(source_ids):
            return False
        rows = list(session.scalars(select(Message).where(Message.group_id == job.scope_id, Message.id.in_(source_ids))))
        return len(rows) == len(source_ids) and all(is_usable_memory_source(row, group_id=job.scope_id) for row in rows)

    @staticmethod
    def _budget_allowed(session, *, account_id: int, job: SocialJob, now: datetime, own_action_id=None) -> bool:
        if job.kind == "qzone_comment" and job.autonomous_tools and job.max_daily_per_friend:
            # The wake has no chosen friend. Each real comment is atomically
            # charged to its friend in the tool ledger at dispatch time.
            return True
        day_start = now.astimezone(ZoneInfo("Asia/Shanghai")).replace(hour=0, minute=0, second=0, microsecond=0)
        common = (SocialActionRecord.account_id == account_id, SocialActionRecord.kind == job.kind,
                  SocialActionRecord.scope_id == job.scope_id)
        if job.autonomous_tools:
            # Thinking, searching and saving an intention are not external
            # writes. They have their own durable model/image budgets.
            common = (*common, SocialActionRecord.dispatched_at.is_not(None))
        charged_at = func.coalesce(SocialActionRecord.dispatched_at, SocialActionRecord.created_at)
        quota_query = select(func.count()).select_from(SocialActionRecord).where(*common, charged_at >= utc_naive(day_start))
        if own_action_id is not None:
            quota_query = quota_query.where(SocialActionRecord.action_id != own_action_id)
        count = session.scalar(quota_query) or 0
        if count >= job.max_daily:
            return False
        cooldown_query = select(func.max(charged_at)).where(*common)
        if own_action_id is not None:
            cooldown_query = cooldown_query.where(SocialActionRecord.action_id != own_action_id)
        last = session.scalar(cooldown_query)
        # Harness group sends use the canonical chat cadence at dispatch.
        # This wake ledger must not impose an additional social cooldown.
        cooldown = 0 if job.kind == "group" and job.chat_cadence_only else job.cooldown_seconds
        return last is None or last <= utc_naive(now) - timedelta(seconds=cooldown)

    def reserve_action(
        self, *, account_id: int, owner: str, epoch: int, job: SocialJob, decision: SocialDecision,
        dedupe_key: str, now: datetime,
    ) -> str | None:
        expected = {"group": "send_group", "qzone_publish": "publish_qzone", "qzone_comment": "comment_qzone"}
        allowed = decision.action == "run_tools" if job.autonomous_tools else decision.action == expected.get(job.kind)
        if not allowed or not decision.text.strip() or len(decision.text) > 2000:
            return None
        if job.kind != "group" and (job.scope_id != account_id or decision.source_message_ids):
            return None
        if job.kind == "qzone_comment" and not job.autonomous_tools:
            if not decision.post_id or len(decision.post_id) > 256 or decision.owner_id not in job.friend_ids:
                return None
            dedupe_key = f"comment:{decision.owner_id}:{decision.post_id}"
        digest = sha256(dedupe_key.encode()).hexdigest()
        action_id = "social-" + sha256(f"{account_id}:{digest}".encode()).hexdigest()
        current = utc_naive(now)
        with self._write_scope() as session:
            if not self._owns(session, account_id=account_id, owner=owner, epoch=epoch, now=current):
                return None
            if session.scalar(select(SocialActionRecord.action_id).where(SocialActionRecord.account_id == account_id,
                                                                        SocialActionRecord.dedupe_key == digest)):
                return None
            if not job.autonomous_tools and not self._budget_allowed(session, account_id=account_id, job=job, now=now):
                return None
            if not self._sources_usable(session, job, decision.source_message_ids):
                return None
            session.add(SocialActionRecord(action_id=action_id, account_id=account_id, owner=owner, epoch=epoch,
                                            job_key=job.key, kind=job.kind, scope_id=job.scope_id, dedupe_key=digest,
                                            action=decision.action, text=decision.text, post_id=decision.post_id,
                                            post_owner_id=decision.owner_id, source_message_ids=list(decision.source_message_ids),
                                            status="reserved", created_at=current, updated_at=current))
            return action_id

    def mark_dispatching(self, *, account_id: int, owner: str, epoch: int, action_id: str, job: SocialJob,
                         now: datetime) -> bool:
        with self._write_scope() as session:
            if not self._owns(session, account_id=account_id, owner=owner, epoch=epoch, now=utc_naive(now)):
                return False
            row = session.get(SocialActionRecord, action_id)
            if row is None or row.account_id != account_id or row.owner != owner or row.epoch != epoch or row.status != "reserved":
                return False
            if (row.job_key != job.key or row.kind != job.kind or row.scope_id != job.scope_id
                    or job.kind == "qzone_comment" and row.post_owner_id not in job.friend_ids
                    or not self._sources_usable(session, job, row.source_message_ids)
                    or not self._budget_allowed(session, account_id=account_id, job=job, now=now, own_action_id=action_id)):
                return False
            if job.kind == "qzone_comment" and job.autonomous_tools:
                # Observing/saving an intention does not consume a post. Once
                # a write may have reached it, never replay across later wakes.
                prior = session.execute(select(SocialActionRecord.action, SocialActionRecord.dispatched_at).where(
                    SocialActionRecord.account_id == account_id, SocialActionRecord.kind == "qzone_comment",
                    SocialActionRecord.action_id != action_id,
                    SocialActionRecord.post_owner_id == row.post_owner_id, SocialActionRecord.post_id == row.post_id,
                ))
                if any(action != "run_tools" or dispatched is not None for action, dispatched in prior):
                    return False
            row.status, row.updated_at = "dispatching", utc_naive(now)
            row.dispatched_at = utc_naive(now)
            return True

    def finish(self, *, account_id: int, owner: str, epoch: int, action_id: str, status: str, now: datetime,
               receipt_id: str | None = None, reason: str = "") -> None:
        if status not in self.TERMINAL:
            raise ValueError("invalid social action status")
        with self._write_scope() as session:
            row = session.get(SocialActionRecord, action_id)
            if row is None or row.account_id != account_id or row.owner != owner or row.epoch != epoch or row.status in self.TERMINAL:
                return
            row.status, row.updated_at, row.reason = status, utc_naive(now), reason[:160]
            row.receipt_id = str(receipt_id)[:256] if receipt_id else None
