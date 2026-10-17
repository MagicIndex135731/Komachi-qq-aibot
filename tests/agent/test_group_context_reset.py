"""Group reset followed by a fresh real Harness turn and a context-tool read."""
from datetime import UTC, datetime
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.adapters.onebot_models import parse_group_message_event
from app.agent.contracts import AgentBatch, AgentDecision
from app.agent.social_policy import configured_social_jobs
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.context_resets import ContextResetRepository
from test_autonomous_social import BOT, FRIEND, GROUP, OTHER_GROUP, composed
from test_harness_runtime import model_client, responses_sse
from test_mature_composition import feedback


def event(identifier, text, *, user=FRIEND, group=GROUP, mention=True):
    return parse_group_message_event({'post_type': 'message', 'message_type': 'group', 'group_id': group,
        'message_id': identifier, 'user_id': user, 'time': datetime.now(UTC).timestamp(),
        'sender': {'nickname': '合成'}, 'message': [
            *([{'type': 'at', 'data': {'qq': str(BOT)}}] if mention else []),
            {'type': 'text', 'data': {'text': text}}]}, bot_qq=BOT, bot_name='小町')


def guard(identifier):
    return SimpleNamespace(turn_id=identifier, dispatched=False, valid=lambda: True,
        require_valid=lambda: None, on_dispatch=lambda: None)


async def turn(fixture, message):
    fixture.router.ingest_live_group_message(message)
    batch = AgentBatch(GROUP, (message,), message.platform_msg_id)
    return await fixture.executor.execute_group_turn(message, batch,
        AgentDecision('reply', message.platform_msg_id, 'direct'), guard(message.platform_msg_id))


async def test_group_new_message_does_not_reintroduce_masked_latest_messages(sqlite_engine, tmp_path):
    phase = {'reset': False}
    def model(body, n):
        previous = feedback(body)
        if phase['reset'] and not any(item['name'] == 'read_context' for item in previous):
            return responses_sse(name='read_context', arguments={}, number=n, call_id=f'group-reset-{n}')
        if previous and previous[-1]['name'] == 'send_group':
            return responses_sse(number=n)
        return responses_sse(name='send_group', arguments={'text':
            'NEW_GROUP_REPLY_100' if phase['reset'] else 'OLD_GROUP_BOT_POISON_100', 'artifact_ids': []},
            number=n, call_id=f'group-reset-{n}')
    client, requests, _ = model_client(model)
    async with composed(sqlite_engine, tmp_path, client, runtime='harness') as fixture:
        fixture.router.runtime.settings = fixture.router.runtime.settings.model_copy(update={'agent_monitor_enabled':True})
        await fixture.agent.stop()
        fixture.clock[0] = datetime.now(UTC)
        old = event('group-old', 'OLD_GROUP_USER_POISON_100')
        assert (await turn(fixture, old)).status == 'sent'
        state = fixture.executor.sessions.read(f'group:{GROUP}')
        before = len(requests)
        command = event('group-reset', '重置上下文', user=fixture.router.runtime.settings.owner_qq)
        fixture.router.ingest_live_group_message(command)
        cancelled = []
        async def cancel(group):
            cancelled.append(group)
        assert await fixture.router.try_handle_context_reset(command, cancel_group=cancel)
        from app.agent.monitor_reset import read_reset
        boundary = read_reset(fixture.router.runtime.settings.data_dir/'agent_monitor', 'group', GROUP)
        assert boundary and boundary['source_id'] > 0
        assert cancelled == [GROUP] and len(requests) == before
        phase['reset'] = True
        assert (await turn(fixture, event('group-new', 'NEW_GROUP_REQUEST_100'))).status == 'sent'
        inputs = json.dumps(requests[before:], ensure_ascii=False)
        assert 'NEW_GROUP_REQUEST_100' in inputs and 'read_context' in inputs
        assert 'OLD_GROUP_USER_POISON_100' not in inputs
        assert 'OLD_GROUP_BOT_POISON_100' not in inputs
        assert fixture.executor.sessions.read(f'group:{GROUP}')['checkpoint']['session_id'] != state['checkpoint']['session_id']
        # Group/private and other-group scope markers cannot be changed by this reset.
        resets = ContextResetRepository(sqlite_engine, account_id=BOT)
        assert resets.latest_source_id(f'group:{OTHER_GROUP}') == 0
        assert resets.latest_source_id(f'private:{FRIEND}') == 0


@pytest.mark.parametrize('owner,mention,quoted', [(False, True, False), (True, False, False), (True, True, True)])
async def test_group_reset_cannot_be_issued_by_member_discussion_or_quote(sqlite_engine, tmp_path, owner, mention, quoted):
    client, requests, _ = model_client(lambda body, n: responses_sse(number=n))
    async with composed(sqlite_engine, tmp_path, client, runtime='harness') as fixture:
        user = fixture.router.runtime.settings.owner_qq if owner else FRIEND
        message = event('unauthorized-reset', '重置上下文', user=user, mention=mention)
        if quoted:
            message.raw_payload['message'].append({'type': 'reply', 'data': {'id': 'old-quote'}})
        fixture.router.ingest_live_group_message(message)
        await fixture.router.try_handle_context_reset(message)
        assert ContextResetRepository(sqlite_engine, account_id=BOT).latest_source_id(f'group:{GROUP}') == 0
