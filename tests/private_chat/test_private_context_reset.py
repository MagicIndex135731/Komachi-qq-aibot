"""Inspect real native Harness HTTP input after reset followed by a new message."""
import asyncio
import json

import pytest
from sqlalchemy import select

from app.storage.db import session_scope
from app.storage.models import DevSession, Job
from app.storage.repositories import DevSessionRepository
from tests.private_chat.test_agent_service import ACCOUNT, OWNER, OTHER, event, setup, sse


@pytest.mark.parametrize('user', [OWNER, OTHER])
async def test_new_message_and_context_tool_cannot_restore_reset_content(sqlite_engine, tmp_path, user):
    phase = {'reset': False}
    def model(body, n):
        if phase['reset'] and not any(item.get('type') == 'function_call_output' for item in body['input']):
            return sse('read_private_context', {}, number=n)
        return sse('send_private', {'text': '正常新回复' if phase['reset'] else 'OLD_BOT_POISON_100'}, number=n)
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path, model)
    await service.start()
    try:
        await dispatcher.handle_private_message(event('before-reset', user_id=user, text='OLD_USER_POISON_100'))
        scope = f'private:{user}'
        old_native = agent.sessions.read(scope)['checkpoint']['runtime_id']
        with session_scope(sqlite_engine) as session:
            daily = DevSessionRepository(session).get_or_create_owner_session(owner_qq=user, session_mode='daily')
            daily.summary = 'OLD_SUMMARY_POISON_100'
        before = len(requests)
        await dispatcher.handle_private_message(event('reset', user_id=user, text='重置上下文'))
        assert len(requests) == before  # deterministic control, no model call
        assert '已重置' in sender.texts[-1][1] and '2 条' in sender.texts[-1][1]
        phase['reset'] = True
        await dispatcher.handle_private_message(event('after-reset', user_id=user, text='NEW_REQUEST_100'))
        new_inputs = json.dumps(requests[before:], ensure_ascii=False)
        assert 'NEW_REQUEST_100' in new_inputs and 'read_private_context' in new_inputs
        assert 'OLD_USER_POISON_100' not in new_inputs
        assert 'OLD_BOT_POISON_100' not in new_inputs
        assert 'OLD_SUMMARY_POISON_100' not in new_inputs
        assert agent.sessions.read(scope)['checkpoint']['runtime_id'] != old_native
        assert sender.texts[-1] == (user, '正常新回复')
    finally:
        await service.stop()

    # Construct a fresh service against the same DB and native files.
    service2, agent2, dispatcher2, _, requests2 = setup(sqlite_engine, tmp_path,
        lambda body, n: sse('send_private', {'text': '重启后的正常回复'}, number=n))
    await service2.start()
    try:
        await dispatcher2.handle_private_message(event('after-restart', user_id=user, text='RESTART_REQUEST_100'))
        assert 'POISON_100' not in json.dumps(requests2, ensure_ascii=False)
    finally:
        await service2.stop()


@pytest.mark.parametrize('late_return', [True, False])
async def test_reset_revokes_a_running_turn_before_late_tool_delivery(sqlite_engine, tmp_path, late_return):
    started, release = asyncio.Event(), asyncio.Event()
    class HeldRuntime:
        def __init__(self, **kwargs):
            pass
        async def run(self, prompt, tools, guard, **kwargs):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                # Simulate an uninterruptible provider/tool worker returning late.
                if not late_return:
                    raise
            assert not guard.valid()
            with pytest.raises(RuntimeError):
                guard.require_valid()
            from app.agent.tool_agent import ToolTurnResult
            return ToolTurnResult('blocked', reason='context_reset')
        def close(self):
            pass
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse('send_private', {'text': '重置后的新请求正常执行'}, number=n))
    agent.runtime_factory = HeldRuntime
    await service.start()
    pending = asyncio.create_task(dispatcher.handle_private_message(event('held', text='OLD_RUNNING_POISON_100')))
    try:
        await asyncio.wait_for(started.wait(), 5)
        await dispatcher.handle_private_message(event('reset-held', text='重置上下文 1'))
        release.set()
        await asyncio.wait_for(pending, 5)
        assert len(sender.texts) == 1 and '已重置' in sender.texts[0][1]
        assert agent.sessions.read(f'private:{OWNER}')['items'] == []
        with session_scope(sqlite_engine) as session:
            jobs = list(session.scalars(select(Job).where(Job.job_type == agent.job_type)))
            assert jobs and all(job.status == 'cancelled' for job in jobs)
        assert agent._loop is not None and not agent._loop.done()
        from app.agent.harness_runtime import HarnessToolAgent
        agent.runtime_factory = HarnessToolAgent
        await dispatcher.handle_private_message(event('after-held-reset', text='NEW_AFTER_CANCEL_100'))
        assert sender.texts[-1][1] == '重置后的新请求正常执行'
        assert 'OLD_RUNNING_POISON_100' not in json.dumps(requests, ensure_ascii=False)
    finally:
        release.set()
        await service.stop()


async def test_reset_requires_exact_authored_text_and_allowed_private_user(sqlite_engine, tmp_path):
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse('send_private', {'text': '普通聊天'}, number=n))
    await service.start()
    try:
        quoted = event('quoted-reset', text='重置上下文')
        quoted.raw_payload['message'].insert(0, {'type': 'reply', 'data': {'id': 'prior'}})
        await dispatcher.handle_private_message(quoted)
        assert requests and sender.texts[-1][1] == '普通聊天'
        count = len(requests)
        await dispatcher.handle_private_message(event('bad-count', text='重置上下文 0'))
        assert len(requests) == count and '用法' in sender.texts[-1][1]
        await dispatcher.handle_private_message(event('stranger-reset', user_id=900000103, text='重置上下文'))
        assert len(sender.texts) == 2
    finally:
        await service.stop()
