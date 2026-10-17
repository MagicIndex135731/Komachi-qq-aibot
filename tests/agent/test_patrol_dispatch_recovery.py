"""A local policy stop must not poison cooldown or look like a user reset."""
from datetime import UTC, datetime
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from app.adapters.sender import Sender
from app.agent.observation import GroupObservationService
from app.agent.social_contracts import SocialDecision, SocialJob, SocialObservation
from app.agent.social_executor import SocialExecutor
from app.agent.tool_agent import AgentTool, ToolOutcome
from app.agent.tool_dispatch import ToolDispatchBoundary
from app.core.context_reset import ContextResetGuard
from app.core.reply_policy import ReplyPolicy
from app.storage.context_resets import ContextResetRepository
from app.storage.db import session_scope
from app.storage.repositories import MessageRepository


NOW = datetime(2026, 10, 17, 12, 39, 48, tzinfo=UTC)
GROUP, BOT = 900000001, 900000200


def setup_boundary(engine, *, cadence_only=False, allow_send=lambda: True, speech=lambda: True):
    router = SimpleNamespace(runtime=SimpleNamespace(settings=SimpleNamespace(
        bot_qq=BOT, owner_qq=900000100, agent_runtime='harness'), persona={'name': 'synthetic bot'}),
        engine=engine, reply_policy=ReplyPolicy(), _normalize_timestamp=lambda value: value,
        _group_policy_bool=lambda **kwargs: False)
    observer = GroupObservationService(router, clock=lambda: NOW)

    def suppression():
        with session_scope(engine) as session:
            return observer._suppression(group_id=GROUP, event_id='synthetic-incoming',
                messages=MessageRepository(session), policy=(True, True, True, (60, 60), None, False))

    class Parent:
        action_id = 'synthetic-patrol'
        source_message_ids = ()
        dispatched = False

        def valid(self):
            return not suppression()

        def require_valid(self):
            if not self.valid():
                raise RuntimeError('interactive authority revoked')

        def on_dispatch(self):
            self.require_valid()
            self.dispatched = True

    context = ContextResetGuard(Parent(), ContextResetRepository(engine, account_id=BOT), f'group:{GROUP}')
    boundary = ToolDispatchBoundary(engine=engine, account_id=BOT, scope=f'group:{GROUP}', guard=context,
        clock=lambda: NOW, max_daily=50, group_send_allowed=allow_send,
        group_interjection_reserve=speech, **({'chat_cadence_only': True} if cadence_only else {}))
    # Direct responses remain logged in the same durable scope, without consuming patrol capacity.
    for index in range(50):
        record = boundary.repository.reserve(f'group:{GROUP}', run_id=f'synthetic-direct-{index}',
            tool_call_id='send', operation='send_group', payload_hash=f'{index:064x}')
        assert boundary.repository.mark_dispatching(record['action_id'], daily_limit=None)
        boundary.repository.finish(record['action_id'], 'sent', receipt=f'synthetic-{index}')

    class Gateway:
        def __init__(self):
            self.calls = []

        async def call_api(self, action, params):
            self.calls.append(action)
            return {'status': 'ok', 'data': {'message_id': 'synthetic-receipt'}}

    gateway = Gateway()
    executor = SocialExecutor(router=router, sender=Sender(gateway), qzone_adapter=None, clock=lambda: NOW)
    executor.policy_check = lambda *args, **kwargs: boundary.valid()
    observation = SocialObservation(SocialJob(f'group:{GROUP}', 'group', GROUP), NOW, 'synthetic persona')

    async def send(arguments):
        result = await executor._send_group(observation, SocialDecision('send_group', arguments['text']), boundary)
        return ToolOutcome({'result': result.status, 'error': result.reason}, result.status, result.receipt_id)

    return boundary, AgentTool('send_group', 'synthetic send', {}, send, 'write'), gateway, context, suppression


async def test_daily_limit_stop_keeps_real_reason_and_guard_usable(sqlite_engine):
    boundary, tool, gateway, context, suppression = setup_boundary(sqlite_engine)
    assert suppression() == ''
    result = await boundary.invoke(tool, {'text': 'synthetic patrol output'}, 'synthetic-call')
    assert result.status == 'blocked'
    assert result.data['error'] == 'daily_scope_budget_exhausted'
    assert gateway.calls == []
    assert suppression() == ''
    assert context.repository.latest_source_id(context.scope) == context.generation
    context.require_valid()  # Checkpoint completion can now run through its original authority.
    recorded = boundary.repository.get(boundary.receipts[-1]['action_id'])
    assert recorded['status'] == 'blocked' and recorded['dispatched_at'] is None
    assert recorded['reason'] == 'daily_scope_budget_exhausted'


async def test_chat_cadence_ignores_direct_scope_count(sqlite_engine):
    boundary, tool, gateway, context, suppression = setup_boundary(sqlite_engine, cadence_only=True)
    result = await boundary.invoke(tool, {'text': 'synthetic patrol output'}, 'synthetic-call')
    assert result.status == 'sent' and result.receipt_id == 'synthetic-receipt'
    assert gateway.calls == ['send_group_msg']


@pytest.mark.parametrize('restriction', ['cadence', 'speech'])
async def test_chat_cadence_mode_keeps_final_policy_limits(sqlite_engine, restriction):
    boundary, tool, gateway, context, suppression = setup_boundary(sqlite_engine, cadence_only=True,
        allow_send=lambda: restriction != 'cadence', speech=lambda: restriction != 'speech')
    result = await boundary.invoke(tool, {'text': 'synthetic patrol output'}, 'synthetic-call')
    assert result.status == 'blocked' and gateway.calls == []
    assert result.data['error'] == ('group_chat_cadence_or_policy_revoked' if restriction == 'cadence'
        else 'proactive_speech_budget')
    assert suppression() == ''
    context.require_valid()


async def test_explicit_tool_daily_limit_is_preserved_in_chat_cadence_mode(sqlite_engine):
    boundary, tool, gateway, context, suppression = setup_boundary(sqlite_engine, cadence_only=True)
    result = await boundary.invoke(replace(tool, daily_limit=1), {'text': 'synthetic output'}, 'limited-tool')
    assert result.status == 'blocked' and result.data['error'] == 'daily_scope_budget_exhausted'
    assert gateway.calls == [] and suppression() == ''
    context.require_valid()


async def test_chat_cadence_flag_does_not_exempt_non_chat_write(sqlite_engine):
    boundary, _, gateway, context, _ = setup_boundary(sqlite_engine, cadence_only=True)

    async def write(arguments):
        boundary.on_dispatch()
        raise AssertionError('Daily-limited non-chat write must never dispatch')

    tool = AgentTool('set_nickname', 'synthetic non-chat write', {}, write, 'write')
    result = await boundary.invoke(tool, {'text': 'synthetic name'}, 'non-chat')
    assert result.status == 'blocked' and result.data['error'] == 'daily_scope_budget_exhausted'
    assert gateway.calls == []
    context.require_valid()


async def test_chat_cadence_mode_still_fences_prior_unknown_send(sqlite_engine):
    boundary, tool, gateway, context, suppression = setup_boundary(sqlite_engine, cadence_only=True)
    arguments = {'text': 'synthetic patrol output'}
    digest = sha256(json.dumps(arguments, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    prior = boundary.repository.reserve(f'group:{GROUP}', run_id='synthetic-prior-unknown',
        tool_call_id='send', operation='send_group', payload_hash=digest)
    assert boundary.repository.mark_dispatching(prior['action_id'], daily_limit=None)
    boundary.repository.finish(prior['action_id'], 'uncertain')
    result = await boundary.invoke(tool, arguments, 'new-call-id')
    assert result.status == 'uncertain' and gateway.calls == []
    assert boundary.repository.get(prior['action_id'])['status'] == 'uncertain'


def test_parent_revocation_keeps_its_original_reason():
    class Parent:
        def valid(self):
            return False

        def require_valid(self):
            raise RuntimeError('source_recalled')

    repository = SimpleNamespace(latest_source_id=lambda scope: 7)
    guard = ContextResetGuard(Parent(), repository, 'group:900000001')
    with pytest.raises(RuntimeError, match='^source_recalled$'):
        guard.require_valid()


def test_real_reset_still_revokes_even_when_parent_is_valid():
    generation = [7]
    parent = SimpleNamespace(valid=lambda: True, require_valid=lambda: None)
    guard = ContextResetGuard(parent, SimpleNamespace(latest_source_id=lambda scope: generation[0]), 'group:900000001')
    generation[0] = 8
    assert not guard.valid()
    with pytest.raises(RuntimeError, match='^context_reset_authority_revoked$'):
        guard.require_valid()
