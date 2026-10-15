"""Observed execution is private telemetry, never task authority/model input."""
import pytest

from app.agent.runtime_trace import AgentTrace, TraceReader, current_trace_context, observe_with
from app.storage.db import session_scope
from app.storage.models import Job
from test_media_tasks import setup_queue
from test_research_queue import fixture_queue


@pytest.mark.parametrize('media', [False, True])
async def test_queued_job_records_real_phases_and_keeps_parent_across_restart(sqlite_engine, tmp_path, media):
    queue, payload, now = setup_queue(sqlite_engine) if media else fixture_queue(sqlite_engine)
    queue.trace_root = tmp_path / 'traces'
    parent = AgentTrace(queue.trace_root, context={'surface':'private', 'scope_id':payload['scope_id'],
        'source_id':payload['source_id'], 'context_kind':'private'}, session_key='main', conversation_key='main')
    with observe_with(parent, call_id='original-call'):
        outcome = (queue.submit(payload, parent_turn=42, before_commit=lambda:None).data if media
            else queue.enqueue(payload, parent_turn=42, before_commit=lambda:None))
    parent.finish('silent', 'queued'); parent.close()
    assert await queue.tick()
    with session_scope(sqlite_engine) as session:
        saved = dict(session.get(Job, outcome['job_id']).payload_json)
    assert saved['_trace_parent']['parent_trace_id'] == parent.trace_id
    assert saved['result'].get('_trace_parent') is None  # return value is not the telemetry envelope
    returned = []
    async def deliver(payload, guard):
        guard.require_valid(); returned.append(current_trace_context())
        return 'completed'
    resumed = type(queue)(sqlite_engine, account_id=queue.account_id, surface='private',
        execute=queue.execute, deliver=deliver, authorized=lambda _:True,
        clock=lambda:now[0], trace_root=queue.trace_root)
    resumed.running = True
    assert await resumed.tick() and not await resumed.tick()
    reader = TraceReader(queue.trace_root)
    main = next(c for c in reader.list_conversations() if c['context_kind'] == 'private')
    assert main['task_count'] == 1 and main['turn_count'] == 1
    stages = main['task_traces']
    assert {m['job_phase'] for m in stages} == {'running','delivering'}
    assert all(m['parent_trace_id'] == parent.trace_id and m['model_requests'] == 0 for m in stages)
    assert returned[0]['workflow_id'] == parent.trace_id and returned[0]['job_id'] == outcome['job_id']
    events = [e for meta in stages for e in reader.read_trace(meta['trace_id'])['events']]
    assert {e['data']['status'] for e in events if e['kind'] == 'job.phase.completed'} == {'result_ready','completed'}
    if media:
        assert any(e['kind'] == 'tool.started' and e['data']['name'] == 'download_web_video' for e in events)
    assert current_trace_context() == {}


@pytest.mark.parametrize('enabled', [False, True])
async def test_unavailable_observation_cannot_change_execution_or_retry(sqlite_engine, tmp_path, enabled):
    calls = []
    async def execute(payload, guard):
        guard.require_valid(); calls.append('execute')
        return {'status':'ok', 'data':{}}
    async def deliver(payload, guard):
        guard.require_valid(); calls.append('deliver')
        return 'completed'
    queue, payload, _ = setup_queue(sqlite_engine, execute=execute, deliver=deliver)
    occupied = tmp_path/'occupied'; occupied.write_text('not a directory')
    queue.trace_root = occupied if enabled else None
    job = queue.submit(payload, parent_turn=42, before_commit=lambda:None).data['job_id']
    assert await queue.tick() and await queue.tick() and not await queue.tick()
    with session_scope(sqlite_engine) as session:assert session.get(Job, job).status == 'completed'
    assert calls == ['execute','deliver'] and current_trace_context() == {}
