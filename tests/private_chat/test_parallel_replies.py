import asyncio
import json
import threading
import pytest

from app.agent.runtime_trace import TraceReader
from tests.private_chat.test_agent_service import setup, event, sse, OWNER


@pytest.mark.parametrize('reset', [False, True])
async def test_native_private_overlap_replies_fast_first_and_merges_owner(sqlite_engine, tmp_path, reset):
    started, release = threading.Event(), threading.Event()
    def script(body, number):
        if any(item.get('type') == 'function_call_output' for item in body['input']):
            return sse(number=number)
        content = '\n'.join(str(item.get('content','')) for item in body['input'])
        slow = 'slow-question' in content
        if slow:
            started.set()
            assert release.wait(15)
        return sse('send_private', {'text':'slow-answer' if slow else 'fast-answer'}, number=number)
    service, agent, dispatcher, sender, _ = setup(sqlite_engine, tmp_path, script)
    agent.settings.agent_monitor_enabled = True
    # Queue constructors capture the monitor setting separately; main native
    # requests read it at execution time.
    await service.start()
    slow = asyncio.create_task(dispatcher.handle_private_message(event('slow-private',text='slow-question')))
    fast = None
    try:
        async with asyncio.timeout(5):
            while not started.is_set():await asyncio.sleep(.01)
        fast = asyncio.create_task(dispatcher.handle_private_message(event('fast-private',text='fast-question')))
        await asyncio.wait_for(asyncio.shield(fast), 8)
        assert sender.texts == [(OWNER,'fast-answer')]
        if reset:
            await dispatcher.handle_private_message(event('parallel-reset',text='重置上下文'))
            release.set()
            await asyncio.wait_for(slow, 8)
            assert all(text != 'slow-answer' for _,text in sender.texts)
            assert agent.sessions.read(f'private:{OWNER}')['items'] == []
            return
        release.set()
        await asyncio.wait_for(slow, 8)
        assert sender.texts == [(OWNER,'fast-answer'),(OWNER,'slow-answer')]
        state = agent.sessions.read(f'private:{OWNER}')
        assert [item['content'] for item in state['items'] if item['role']=='assistant'] == ['fast-answer','slow-answer']
        assert not state['checkpoint'].get('in_flight')
        assert state['checkpoint']['shared_native_dirty']
        assert len(state['context_source_message_ids']) == 2
        groups = TraceReader(tmp_path/'agent_monitor').list_conversations()
        assert len(groups) == 1 and groups[0]['turn_count'] == 2 and groups[0]['native_session_count'] == 2
        assert {t['context_reason'] for t in groups[0]['turns']} == {'first_turn','parallel_snapshot'}
    finally:
        release.set()
        await asyncio.gather(*(t for t in (slow,fast) if t), return_exceptions=True)
        await service.stop()


async def test_private_does_not_claim_more_leases_than_model_capacity(sqlite_engine, tmp_path):
    from sqlalchemy import select
    from app.storage.db import session_scope
    from app.storage.models import Job, Message
    from tests.private_chat.test_agent_service import OTHER
    service, agent, dispatcher, _, _ = setup(sqlite_engine, tmp_path, lambda body,n:sse(number=n))
    release = asyncio.Event()
    async def execute(job_id, payload):
        await release.wait()
        agent._finish(job_id, 'completed', 'synthetic')
    agent._execute = execute
    agent.running = True
    try:
        for user in (OWNER,OTHER):
            for n in range(3):
                identifier=f'capacity-{user}-{n}'
                dispatcher._persist_inbound(event(identifier,user_id=user))
                with session_scope(sqlite_engine) as session:
                    source=session.scalar(select(Message).where(Message.platform_msg_id==f'private-inbound-{user}-{identifier}'))
                    source_id=source.id
                agent._enqueue(user_id=user,source_id=source_id,private_scope='daily',key=identifier)
            await agent._drain_user(user,wait=False)
        assert len(agent._active) == 4
        with session_scope(sqlite_engine) as session:
            states=[r.status for r in session.scalars(select(Job).where(Job.job_type==agent.job_type))]
        assert states.count('running') == 4 and states.count('queued') == 2
    finally:
        release.set()
        await asyncio.gather(*tuple(agent._active.values()),return_exceptions=True)
        await service.stop()
