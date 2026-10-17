"""Local admission before persona/recall/model work; shared by both group paths."""
import logging
import re

from sqlalchemy import select

from app.agent.participation_feedback import normalized
from app.core.memory_source_guard import is_usable_memory_source
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.proactive import ProactiveLimits, ProactiveRepository

logger = logging.getLogger(__name__)


def limits_for(router, group_id):
    values = {}
    defaults = ProactiveLimits()
    for name, maximum in [('minimum_messages', 30), ('interval_seconds', 3600),
            ('max_backoff_seconds', 86400), ('hourly_models', 100), ('hourly_speeches', 30),
            ('speech_cooldown_seconds', 86400), ('max_wait_seconds', 86400)]:
        value = router._group_policy_int(group_id=group_id,
            key='proactive_observation_' + name, default=getattr(defaults, name))
        values[name] = max(1, min(maximum, value))
    return ProactiveLimits(**values)


class ProactiveTurn:
    def __init__(self, parent, repository, permit, limits, metrics):
        self.parent, self.repository, self.permit, self.limits = parent, repository, permit, limits
        self.trace_trigger = {**getattr(parent, 'trace_trigger', {}), **metrics}
        self.reserved_models = 0

    def __getattr__(self, key):
        return getattr(self.parent, key)

    def reserve_model(self):
        reserved = self.repository.reserve(self.permit, 'model', limits=self.limits)
        self.reserved_models += int(reserved)
        return reserved

    def admission_metrics(self):
        return {key: value for key, value in self.trace_trigger.items()
                if key.startswith(('observation_', 'proactive_'))} | {
            'proactive_model_requests_hour': self.trace_trigger['proactive_model_requests_hour'] + self.reserved_models}

    def reserve_speech(self):
        return self.repository.reserve(self.permit, 'speech', limits=self.limits)


def admit_proactive(router, observation, guard, *, interactive, clock):
    group_id = observation.job.scope_id
    limits = limits_for(router, group_id)
    with session_scope(router.engine) as session:
        rows = list(session.scalars(select(Message).where(Message.group_id == group_id,
            Message.id.in_(observation.source_message_ids))))
        rows = [r for r in rows if r.user_id != router.runtime.settings.bot_qq
                and is_usable_memory_source(r, group_id=group_id)]
        candidates = []
        for row in rows:
            value = normalized(row.plain_text)
            segments = (row.raw_json or {}).get('message', [])
            has_image = isinstance(segments, list) and any(isinstance(s, dict) and
                s.get('type') == 'image' for s in segments)
            if has_image or value and not re.fullmatch(r'哈+|嗯+|哦+|好+|[?？!！]+|ok|wx', value):
                candidates.append((row.id, value or 'image:' + str(row.id)))
    triggers = getattr(guard, 'trace_trigger', {}).get('trigger_evidence', {}).get('source_ids', []) if interactive else []
    seen = max((r.id for r in rows), default=0)
    bootstrap = min(triggers) - 1 if triggers else min((i for i, _ in candidates), default=seen+1) - 1
    repository = ProactiveRepository(router.engine, account_id=router.runtime.settings.bot_qq, clock=clock)
    due = bool(getattr(guard, 'trace_trigger', {}).get('trigger_reason_code') == 'intention_due')
    permit, reason, metrics = repository.claim(group_id, candidates=candidates,
        bootstrap_cursor=max(0, bootstrap), max_source_id=seen, limits=limits, due_goal=due)
    logger.info('proactive_observation group_id=%s reason=%s new_messages=%s cursor=%s model_requests_hour=%s',
        group_id, reason or 'admitted', metrics['observation_new_messages'], metrics['observation_cursor'],
        metrics['proactive_model_requests_hour'])
    return (ProactiveTurn(guard, repository, permit, limits, metrics) if permit else None), reason
