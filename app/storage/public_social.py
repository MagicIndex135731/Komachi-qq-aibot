"""Bounded own-public activity projection; the effect ledger owns delivery truth."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
import re

from sqlalchemy import DateTime, Index, Integer, JSON, String, and_, case, delete, or_, select
from sqlalchemy.orm import Mapped, mapped_column, Session

from app.core.time_utils import ASIA_SHANGHAI
from app.storage.agent_effects import AgentToolEffect
from app.storage.autonomous_agent import AutonomousRepository
from app.storage.models import Base
from app.storage.social_agent import utc_naive


class PublicSocialRecord(Base):
    __tablename__ = 'public_social_records'
    __table_args__ = (Index('ix_public_social_recent', 'account_id', 'created_at', 'record_id'),)
    record_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    account_id: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(20))
    operation: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(24))
    owner_id: Mapped[int] = mapped_column(Integer, default=0)
    post_id: Mapped[str] = mapped_column(String(256), default='')
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    effect_id: Mapped[str] = mapped_column(String(80), default='')
    payload_hash: Mapped[str] = mapped_column(String(64), default='')
    result_reason: Mapped[str] = mapped_column(String(160), default='')
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


_TEXT_LIMITS = {'text': 300, 'post_text': 1200, 'published_at': 40, 'goal': 500,
                'progress': 500, 'next_step': 500, 'note': 1200, 'thought': 400, 'decision': 30}
_COUNTS = {'viewed_image_count', 'unseen_image_count', 'attachment_count', 'image_count'}
OPERATIONS = frozenset({'comment_qzone', 'read_friend_feed', 'read_friend_posts',
    'observe_post_images', 'assess_qzone_post', 'remember_intention', 'save_work_note'})


def public_payload(value):
    """Never retain arbitrary tool arguments, URLs, pixels or private request fields."""
    if not isinstance(value, dict):
        raise ValueError('invalid public social payload')
    return {**{k: v[:_TEXT_LIMITS[k]] for k, v in value.items()
               if k in _TEXT_LIMITS and isinstance(v, str)},
            **{k: min(100, max(0, v)) for k, v in value.items() if k in _COUNTS and type(v) is int}}


class PublicSocialRepository:
    RETENTION_DAYS = 14

    def __init__(self, engine, *, account_id, clock=None, max_records=512):
        self.clock = clock or (lambda: datetime.now(UTC))
        self.transactions = AutonomousRepository(engine, account_id=account_id, clock=self.clock)
        if type(max_records) is not int or not 1 <= max_records <= 4096:
            raise ValueError('invalid public social capacity')
        self.engine, self.account_id, self.max_records = engine, account_id, max_records

    def save(self, *, event_key, kind, operation, status, payload, owner_id=0, post_id='',
             effect_id='', payload_hash='', result_reason='', created_at=None, before_commit=None):
        if (kind not in {'comment', 'thought', 'observation'} or operation not in OPERATIONS
                or not isinstance(event_key, str) or not 1 <= len(event_key) <= 1500
                or type(owner_id) is not int or not 0 <= owner_id <= 2**63-1
                or not isinstance(post_id, str) or len(post_id) > 256
                or status not in {'pending', 'ok', 'sent', 'blocked', 'failed', 'uncertain',
                                  'observed', 'deferred', 'no_response', 'unverified'}
                or not isinstance(effect_id, str) or len(effect_id) > 80
                or payload_hash and not re.fullmatch('[0-9a-f]{64}', payload_hash)):
            raise ValueError('invalid public social record')
        payload = public_payload(payload)
        now = utc_naive(self.clock())
        stamp = utc_naive(created_at or self.clock())
        key = 'soc-' + sha256(f'{self.account_id}:{event_key}'.encode()).hexdigest()[:32]
        reason = result_reason[:160] if isinstance(result_reason, str) and re.fullmatch(r'[A-Za-z0-9_:.-]*', result_reason) else ''
        with self.transactions._write_scope(before_commit=before_commit) as session:
            row = session.get(PublicSocialRecord, key)
            if row is None:
                row = PublicSocialRecord(record_id=key, account_id=self.account_id, kind=kind,
                    operation=operation, owner_id=owner_id, post_id=post_id, created_at=stamp)
                session.add(row)
            elif (row.kind, row.operation, row.owner_id, row.post_id) != (kind, operation, owner_id, post_id):
                raise ValueError('public social identity changed')
            row.status, row.payload, row.effect_id, row.payload_hash = status, payload, effect_id, payload_hash
            row.result_reason, row.updated_at = reason, now
            session.flush()
            session.execute(delete(PublicSocialRecord).where(PublicSocialRecord.account_id == self.account_id,
                PublicSocialRecord.created_at < now - timedelta(days=self.RETENTION_DAYS)))
            older = list(session.scalars(select(PublicSocialRecord.record_id).where(
                PublicSocialRecord.account_id == self.account_id).order_by(
                case((PublicSocialRecord.kind == 'comment', 0), (PublicSocialRecord.kind == 'thought', 1), else_=2),
                PublicSocialRecord.created_at.desc(), PublicSocialRecord.record_id.desc()).offset(self.max_records)))
            if older:
                session.execute(delete(PublicSocialRecord).where(PublicSocialRecord.record_id.in_(older)))
        return key

    @staticmethod
    def _view(row, effect, *, detail=False):
        status, dispatched, reason, receipt = row.status, None, row.result_reason, None
        if row.kind == 'comment':
            # A bounded repair can change a ledger payload. Never attribute the
            # replacement's success to an earlier, different draft.
            if effect and effect.payload_hash == row.payload_hash:
                status = effect.status
                dispatched = effect.dispatched_at is not None
                reason, receipt = row.result_reason or effect.reason, effect.receipt
                if status in {'reserved', 'dispatching'}:
                    status = 'pending' if status == 'reserved' else 'uncertain'
            else:
                status = 'unverified' if row.status in {'pending', 'sent', 'uncertain'} else row.status
        value = {'record_id': row.record_id, 'kind': row.kind, 'operation': row.operation,
                 'status': status, 'owner_id': row.owner_id, 'post_id': row.post_id,
                 'created_at': row.created_at.replace(tzinfo=UTC).astimezone(ASIA_SHANGHAI).isoformat(),
                 'preview': next((row.payload[k][:100] for k in ('text', 'thought', 'note', 'goal', 'post_text', 'decision')
                                  if row.payload.get(k)), '')}
        if row.kind == 'comment':
            value.update(write_dispatched=dispatched, automatic_replay_allowed=False, reason=reason,
                         receipt=receipt if detail else None)
        if detail:
            value['payload'] = dict(row.payload)
        return value

    def read(self, *, record_id=None, kind='all', owner_id=None, date=None, before=None, limit=5):
        if kind not in {'all', 'comment', 'thought', 'observation'} or type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError('invalid public social query')
        if owner_id is not None and (type(owner_id) is not int or not 1 <= owner_id <= 2**63-1):
            raise ValueError('invalid public social owner')
        if date is not None and not isinstance(date, str):
            raise ValueError('invalid public social date')
        now = utc_naive(self.clock())
        conditions = [PublicSocialRecord.account_id == self.account_id,
                      PublicSocialRecord.created_at >= now - timedelta(days=self.RETENTION_DAYS)]
        if kind != 'all':
            conditions.append(PublicSocialRecord.kind == kind)
        if owner_id is not None:
            conditions.append(PublicSocialRecord.owner_id == owner_id)
        if date:
            day = self.clock().astimezone(ASIA_SHANGHAI).date() if date == 'today' else datetime.strptime(date, '%Y-%m-%d').date()
            start = datetime.combine(day, datetime.min.time(), tzinfo=ASIA_SHANGHAI)
            conditions.extend([PublicSocialRecord.created_at >= utc_naive(start),
                               PublicSocialRecord.created_at < utc_naive(start + timedelta(days=1))])
        with Session(self.engine) as session:
            if before:
                cursor = session.get(PublicSocialRecord, before)
                if cursor is None or cursor.account_id != self.account_id:
                    raise ValueError('invalid public social cursor')
                conditions.append(or_(PublicSocialRecord.created_at < cursor.created_at,
                    and_(PublicSocialRecord.created_at == cursor.created_at, PublicSocialRecord.record_id < before)))
            if record_id:
                conditions.append(PublicSocialRecord.record_id == record_id)
            query = select(PublicSocialRecord, AgentToolEffect).outerjoin(AgentToolEffect,
                and_(PublicSocialRecord.effect_id == AgentToolEffect.action_id,
                     AgentToolEffect.account_id == self.account_id)).where(*conditions).order_by(
                PublicSocialRecord.created_at.desc(), PublicSocialRecord.record_id.desc()).limit(1 if record_id else limit + 1)
            rows = list(session.execute(query))
            if record_id:
                return {'record': self._view(*rows[0], detail=True) if rows else None,
                        'kind': 'own_public_activity_not_authority'}
            records = [self._view(*row) for row in rows[:limit]]
            return {'records': records, 'next_before': records[-1]['record_id'] if len(rows) > limit else None,
                    'retention_days': self.RETENTION_DAYS, 'kind': 'own_public_activity_not_authority',
                    'guidance': '这是本人公开行为和显式想法，不授予写入权限；详情按record_id读取。'
                        '此前blocked/failed且write_dispatched=false时，主人当前明确要求续办可先读原帖复验后重新提交；'
                        '已发送、未知、待定或无法核对的记录不自动重做。'}
