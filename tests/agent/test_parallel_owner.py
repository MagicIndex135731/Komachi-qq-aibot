"""Concurrent completions retain one owner, provenance and exact reply targets."""
import asyncio
import json
import threading
import pytest
from types import SimpleNamespace

from app.agent.contracts import AgentActionResult, AgentDecision
from app.agent.conversation_context import select_native_context
from app.agent.patrol_context import OwnerTurnGate
from tests.agent.test_group_agent_runtime import make_runtime, persist, eventually
from tests.storage.test_agent_sessions import repository, turn, SCOPE


def test_parallel_result_append_does_not_overwrite_running_owner_or_other_results(sqlite_engine, tmp_path):
    repo = repository(sqlite_engine, tmp_path)
    beginning = repo.begin_turn(SCOPE, {'session_id':'primary'}, (1,), expected_revision=0)
    fork = repo.complete_turn(SCOPE, turn(2), {}, (2,), native_revision=beginning['revision'], parallel=True)
    assert fork['checkpoint']['in_flight'] and fork['checkpoint']['session_id'] == 'primary'
    final = repo.complete_turn(SCOPE, turn(1), {'session_id':'primary','last_status':'sent'}, (1,),
        native_revision=beginning['revision'])
    assert final['items'] == [*turn(2), *turn(1)]
    assert final['context_source_message_ids'] == [1,2]
    assert not final['checkpoint'].get('in_flight') and final['checkpoint']['shared_native_dirty']
    selected = select_native_context(SCOPE, final['checkpoint'], version=1, history_present=True)
    assert selected.metadata['context_reason'] == 'parallel_completion' and selected.restore


def test_completed_receipt_cache_is_bounded_without_forgetting_unknown_effect(sqlite_engine, tmp_path):
    repo = repository(sqlite_engine, tmp_path)
    repo.save(SCOPE, [], {'receipts':[{'call_id':'unknown','status':'uncertain'}]}, (), expected_revision=0)
    for number in range(60):
        state = repo.read(SCOPE)
        repo.complete_turn(SCOPE, turn(number), {'last_status':'sent', 'receipts':[
            {'call_id':str(number), 'status':'sent', 'receipt_id':'x'*280}]}, (number+1,), native_revision=state['revision'])
    state = repo.read(SCOPE)
    assert len(json.dumps(state['checkpoint'])) < 16000
    assert any(r['call_id']=='unknown' and r['replay_allowed'] is False for r in state['checkpoint']['receipts'])
    assert any(r['call_id']=='59' for r in state['checkpoint']['receipts'])


@pytest.mark.parametrize('new_status', ['uncertain','sent'])
def test_parallel_receipts_with_equal_call_ids_keep_separate_request_sources(sqlite_engine, tmp_path, new_status):
    repo = repository(sqlite_engine,tmp_path)
    repo.save(SCOPE, [], {'receipts':[{'call_id':'reused','reply_source_id':1,'status':'uncertain'}]}, (1,),expected_revision=0)
    repo.complete_turn(SCOPE, turn(2), {'receipts':[{'call_id':'reused','reply_source_id':2,
        'status':new_status,'receipt_id':'synthetic' if new_status=='sent' else None}]}, (2,),native_revision=1,parallel=True)
    receipts=repo.read(SCOPE)['checkpoint']['receipts']
    assert {r['reply_source_id'] for r in receipts} == {1,2}
    assert next(r for r in receipts if r['reply_source_id']==1)['status'] == 'uncertain'


async def test_owner_gate_allows_direct_overlap_and_holds_patrol_until_all_finish():
    gate = OwnerTurnGate()
    parent = SimpleNamespace(valid=lambda:True, require_valid=lambda:None, on_dispatch=lambda:None)
    async with gate.turn(parent, direct=True) as primary:
        assert not primary.parallel_context
        async with gate.turn(parent, direct=True) as parallel:
            assert parallel.parallel_context
            assert gate.direct_active == 2
        assert gate.busy and gate.direct_active == 1
    assert not gate.busy and gate.direct_active == 0


async def test_owner_defers_same_request_callback_without_blocking_other_requests():
    gate = OwnerTurnGate()
    parent = SimpleNamespace(valid=lambda:True, require_valid=lambda:None, on_dispatch=lambda:None)
    entered = asyncio.Event()
    async def callback():
        async with gate.turn(parent,direct=True,request_key='original'):
            entered.set()
    async with gate.turn(parent,direct=True,request_key='original'):
        pending = asyncio.create_task(callback())
        await asyncio.sleep(0)
        assert not entered.is_set()
        async with gate.turn(parent,direct=True,request_key='unrelated'):
            assert not entered.is_set()
    await asyncio.wait_for(pending,1)
    assert entered.is_set() and not gate.active_requests


async def test_slow_group_request_does_not_hold_later_reply(sqlite_engine):
    slow_started, release = asyncio.Event(), asyncio.Event()
    sent = []
    async def decide(batch):
        return AgentDecision('reply', batch.events[-1].platform_msg_id, 'direct_trigger')
    async def execute(event, batch, decision, guard):
        if event.platform_msg_id == 'slow':
            slow_started.set()
            await release.wait()
        guard.on_dispatch()
        sent.append(event.platform_msg_id)
        return AgentActionResult('sent', ('synthetic-'+event.platform_msg_id,))
    runtime = make_runtime(sqlite_engine, decide, execute, quiet_seconds=0)
    runtime.set_connected(True)
    await runtime.start()
    try:
        await runtime.observe(persist(sqlite_engine, 'slow', mentioned=True))
        await asyncio.wait_for(slow_started.wait(), 2)
        await runtime.observe(persist(sqlite_engine, 'fast', mentioned=True))
        await eventually(lambda:sent == ['fast'])
        release.set()
        await eventually(lambda:sent == ['fast','slow'])
    finally:
        release.set()
        await runtime.stop()


@pytest.mark.parametrize('mode',[None,'auto'])
async def test_media_workers_deliver_independent_fast_request_before_slow_request(sqlite_engine,mode):
    from app.agent.media_tasks import MediaTaskQueue
    from tests.agent.test_research_queue import fixture_queue
    release, slow_started = asyncio.Event(), asyncio.Event()
    delivered = []
    async def execute(payload, guard):
        if payload['tag'] == 'slow':
            slow_started.set()
            await release.wait()
        guard.require_valid()
        return {'status':'ok','data':{}}
    async def deliver(payload, guard):
        guard.on_dispatch()
        delivered.append(payload['tag'])
        return 'completed'
    template, payload, now = fixture_queue(sqlite_engine)
    queue = MediaTaskQueue(sqlite_engine, account_id=template.account_id, surface='private',
        execute=execute, deliver=deliver, authorized=lambda _:True, clock=lambda:now[0])
    for tag in ('slow','fast'):
        # Independent requests have independent canonical sources. Same-source
        # review siblings intentionally wait for one joint return instead.
        from app.storage.db import session_scope
        from app.storage.models import Message
        with session_scope(sqlite_engine) as session:
            row=Message(platform_msg_id='media-request-'+tag,group_id=None,user_id=payload['actor_id'],
                timestamp=now[0],plain_text='synthetic '+tag+' request',raw_json={})
            session.add(row);session.flush();source=row.id
        data = {**payload,'source_id':source,'sources':[source],'tag':tag,
            'operation':'download_web_images','arguments':{},
            'source_fingerprints':queue.source_fingerprints([source]),
            **({'delivery_mode':mode} if mode else {})}
        queue.enqueue(data, parent_turn=tag, before_commit=lambda:None)
    await queue.start()
    try:
        await asyncio.wait_for(slow_started.wait(), 2)
        await eventually(lambda:delivered == ['fast'], timeout=4)
        release.set()
        await eventually(lambda:delivered == ['fast','slow'], timeout=4)
    finally:
        release.set()
        await queue.stop()


async def test_native_group_parallel_requests_keep_one_owner_and_both_replies(sqlite_engine, tmp_path):
    from app.agent.social_contracts import SocialObservation
    from app.agent.social_policy import configured_social_jobs
    from app.agent.runtime_trace import TraceReader
    from tests.agent.test_autonomous_social import composed, add_source, NOW, GROUP, FRIEND
    from tests.agent.test_harness_runtime import model_client, responses_sse
    release, started = threading.Event(), threading.Event()
    def script(body, number):
        if any(item.get('type') == 'function_call_output' for item in body['input']):
            return responses_sse(number=number)
        for item in reversed(body['input']):
            content = item.get('content', '')
            if isinstance(content, list):content = '\n'.join(p.get('text','') for p in content)
            if 'Current observation JSON:\n' in content:
                value = json.JSONDecoder().raw_decode(content.split('Current observation JSON:\n',1)[1])[0]
                break
        slow = value['current_target']['message_id'] == 'slow'
        if slow:
            started.set()
            assert release.wait(15)
        return responses_sse(name='send_group', arguments={'text':'slow-answer' if slow else 'fast-answer','artifact_ids':[]}, number=number)
    client, _, _ = model_client(script)
    async with composed(sqlite_engine, tmp_path, client, runtime='harness') as fixture:
        fixture.router.runtime.settings.agent_monitor_enabled = True
        job = next(j for j in configured_social_jobs(fixture.router.runtime) if j.kind == 'group')
        async def request(tag):
            source = add_source(sqlite_engine, platform_id=tag, text=tag+'-question')
            guard = SimpleNamespace(action_id='parallel-'+tag, valid=lambda:True, require_valid=lambda:None, on_dispatch=lambda:None)
            return await fixture.executor.run_tools(SocialObservation(job,NOW,'',source_message_ids=(source,)),guard,
                interactive=True,target_user_id=FRIEND,target_message_id=tag,direct_group_reply=True)
        slow = asyncio.create_task(request('slow'))
        try:
            async with asyncio.timeout(5):
                while not started.is_set():
                    assert not slow.done(), slow.result()
                    await asyncio.sleep(.01)
            fast = await asyncio.wait_for(request('fast'), 8)
            assert fast.status == 'sent', fast
            release.set()
            assert (await slow).status == 'sent'
            state = fixture.executor.sessions.read(f'group:{GROUP}')
            assert len(state['items']) == 4 and len(state['context_source_message_ids']) == 2
            assert state['checkpoint']['shared_native_dirty']
            rows = TraceReader(fixture.router.runtime.settings.data_dir/'agent_monitor').list_conversations()
            assert len(rows) == 1 and rows[0]['turn_count'] == 2
            assert sum(t.get('context_lifetime') == 'snapshot' for t in rows[0]['turns']) == 1
        finally:
            release.set()
            await asyncio.gather(slow, return_exceptions=True)
