"""Atomic, restart-safe proactive admission. No canonical chat bodies live here."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import DateTime, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.storage.models import Base
from app.storage.social_agent import utc_naive


@dataclass(frozen=True)
class ProactiveLimits:
    minimum_messages: int = 8
    interval_seconds: int = 300
    max_backoff_seconds: int = 1800
    max_wait_seconds: int = 900
    hourly_models: int = 6
    hourly_speeches: int = 3
    speech_cooldown_seconds: int = 600


class ProactiveGroupState(Base):
    __tablename__ = 'proactive_group_states'
    account_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    group_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    evaluated_source_id: Mapped[int] = mapped_column(Integer, default=0)
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    silent_count: Mapped[int] = mapped_column(Integer, default=0)
    lease_token: Mapped[str] = mapped_column(String(64), default='')
    lease_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    window: Mapped[dict] = mapped_column(JSON, default=dict)


class ProactiveRepository:
    def __init__(self, engine, *, account_id: int, clock=None):
        self.engine, self.account_id = engine, account_id
        self.clock = clock or (lambda: datetime.now(UTC))

    @contextmanager
    def _write(self):
        from app.storage.db import session_scope
        with session_scope(self.engine) as session:
            session.connection().exec_driver_sql('BEGIN IMMEDIATE')
            yield session

    def _window(self, row):
        cutoff = self.clock().timestamp() - 3600
        value = row.window or {}
        return {'pending_since': value.get('pending_since'),
                'models': [v for v in value.get('models', []) if v > cutoff],
                'speeches': [v for v in value.get('speeches', []) if v['at'] > cutoff]}

    def claim(self, group_id, *, candidates, bootstrap_cursor, max_source_id,
              limits=ProactiveLimits(), due_goal=False):
        """Only captured sources advance on finish; later arrivals remain new."""
        now = utc_naive(self.clock())
        with self._write() as session:
            row = session.get(ProactiveGroupState, (self.account_id, group_id))
            if row is None:
                row = ProactiveGroupState(account_id=self.account_id, group_id=group_id,
                    evaluated_source_id=bootstrap_cursor, silent_count=0, window={}, lease_token='')
                session.add(row)
            window = self._window(row)
            count = len({value for source_id, value in candidates if source_id > row.evaluated_source_id})
            # Silence is not a preference or a penalty on the next topic.
            interval = limits.interval_seconds
            if count and window['pending_since'] is None:
                window['pending_since'] = self.clock().timestamp()
            if not count:
                window['pending_since'] = None
            waited = (self.clock().timestamp() - window['pending_since']
                if window['pending_since'] is not None else 0)
            row.window = window
            metrics = {'observation_new_messages': count, 'observation_cursor': row.evaluated_source_id,
                'observation_interval_seconds': interval, 'proactive_model_requests_hour': len(window['models']),
                'proactive_speeches_hour': len(window['speeches']), 'proactive_model_limit': limits.hourly_models,
                'proactive_speech_limit': limits.hourly_speeches}
            reason = ('observation_in_flight' if row.lease_until and row.lease_until > now else
                'proactive_model_limit' if len(window['models']) >= limits.hourly_models else
                'proactive_speech_limit' if len(window['speeches']) >= limits.hourly_speeches else
                'proactive_speech_cooldown' if window['speeches'] and
                    self.clock().timestamp() - window['speeches'][-1]['at'] < limits.speech_cooldown_seconds else
                'observation_backoff' if not due_goal and row.last_attempt_at and
                    (now - row.last_attempt_at).total_seconds() < interval else
                'insufficient_new_messages' if not due_goal and (not count or
                    count < limits.minimum_messages and waited < limits.max_wait_seconds) else '')
            if reason:
                return None, reason, metrics
            token = uuid4().hex
            row.last_attempt_at, row.lease_token = now, token
            row.lease_until = now + timedelta(minutes=15)
            row.window = window
            return {'token': token, 'group_id': group_id, 'max_source_id': max_source_id}, '', metrics

    def reserve(self, permit, kind, *, limits=ProactiveLimits()):
        if kind not in {'model', 'speech'}:
            raise ValueError('invalid proactive reservation')
        with self._write() as session:
            row = session.get(ProactiveGroupState, (self.account_id, permit['group_id']))
            if (row is None or row.lease_token != permit['token'] or row.lease_until is None
                    or row.lease_until <= utc_naive(self.clock())):
                return False
            window = self._window(row)
            if kind == 'model':
                if len(window['models']) >= limits.hourly_models:
                    return False
                window['models'].append(self.clock().timestamp())
            else:
                # A multi-chunk reply is one intervention, fenced by this permit.
                if any(v['run'] == permit['token'] for v in window['speeches']):
                    return True
                if (len(window['speeches']) >= limits.hourly_speeches or window['speeches'] and
                        self.clock().timestamp() - window['speeches'][-1]['at'] < limits.speech_cooldown_seconds):
                    return False
                window['speeches'].append({'run': permit['token'], 'at': self.clock().timestamp()})
            row.window = window
            return True

    def finish(self, permit, *, status):
        with self._write() as session:
            row = session.get(ProactiveGroupState, (self.account_id, permit['group_id']))
            if row is None or row.lease_token != permit['token']:
                return
            spoken = any(v['run'] == permit['token'] for v in self._window(row)['speeches'])
            if status in {'silent', 'sent', 'uncertain'} or spoken:
                row.evaluated_source_id = max(row.evaluated_source_id, permit['max_source_id'])
                row.silent_count = 0  # Legacy column retained for safe rollback.
                row.window = {**self._window(row), 'pending_since': None}
            row.lease_token, row.lease_until = '', None
