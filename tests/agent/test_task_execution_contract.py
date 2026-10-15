"""Outcome contracts and durable composition, without paid models or QQ sends."""
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from app.agent.media_tasks import build_media_batch_tool
from app.agent.research_tasks import ResearchTaskWorkspace
from app.agent.tool_agent import AgentTool, ToolOutcome, validate_arguments
from test_media_tasks import setup_queue


def task(root, *, source=123, actor=900000101, scope='group:900000001', valid=lambda _: True):
    return ResearchTaskWorkspace(root, scope=scope, actor_id=actor, source_id=source,
        request='find resources and deliver verified media', valid_source=valid)


def test_receipt_is_not_goal_completion_and_unknown_remains_a_fence(tmp_path):
    workspace = task(tmp_path)
    workspace.begin(delivery='media', remaining=['discover resources', 'download and deliver'])
    workspace.observe_tool('send_group', {'text': 'queued'}, ToolOutcome({}, 'sent', 'ack'), 'send1')
    result = workspace.update(progress='ack sent', next_step='download', findings=[], status='completed')
    assert result['status'] == 'partial'
    assert {'remaining_work', 'media_delivery_unconfirmed'} <= set(result['completion_gaps'])
    workspace.observe_tool('download_web_video', {}, ToolOutcome({'job_id': 1,
        'execution_status': 'pending'}, 'ok'), 'job1')
    workspace.update(progress='queued', next_step='wait', findings=[], status='waiting', remaining=[])
    assert workspace.execution_status()['status'] == 'waiting'
    assert 'background_pending' in workspace.completion_gaps()
    workspace.record_job_result(1, 'completed')
    workspace.observe_tool('send_group', {'media_artifact_ids': ['art-' + 'a'*32]},
        ToolOutcome({}, 'sent', 'actual-receipt'), 'send2')
    assert workspace.update(progress='delivered', next_step='', findings=[], status='completed')['status'] == 'completed'
    workspace.observe_tool('send_group', {}, ToolOutcome({}, 'uncertain'), 'unknown')
    for number in range(135):
        workspace.observe_tool('read_context', {}, ToolOutcome({}), 'read' + str(number))
    assert workspace.update(progress='', next_step='check receipt', findings=[], status='completed')['status'] == 'partial'
    assert 'unknown_external_effect' in workspace.completion_gaps()


def test_resource_manifest_keeps_every_discovered_url_and_checks_scope_and_revocation(tmp_path):
    allowed = [True]
    workspace = task(tmp_path, valid=lambda _: allowed[0])
    workspace.begin()
    for page in range(4):
        workspace.save_evidence({'content': f'page {page}', 'sources': [
            {'source': f'https://example.com/resource/{page*20+index}#section'} for index in range(20)]})
    all_resources, offset = [], 0
    while offset is not None:
        result = workspace.resource_page(offset=offset)
        all_resources.extend(result['resources'])
        offset = result['next_offset']
    assert len(all_resources) == 80 and result['total'] == 80
    assert len({item['url'] for item in all_resources}) == 80
    assert all(item['evidence_ids'] and '#' not in item['url'] for item in all_resources)
    other = task(tmp_path, actor=900000102)
    with pytest.raises(ValueError):
        other.resource_page(workspace.task_id)
    allowed[0] = False
    with pytest.raises(ValueError):
        workspace.resource_page()


def test_worker_and_model_checkpoint_updates_do_not_lose_pending_work(tmp_path):
    workspace = task(tmp_path)
    workspace.begin(remaining=['verify coverage'])
    def record(number):
        other = task(tmp_path)
        other.resume(workspace.task_id)
        if number % 2:
            other.observe_tool('download_web_video', {}, ToolOutcome({'job_id': number,
                'execution_status': 'pending'}), str(number))
        else:
            other.record_job_result(number, 'completed')
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(record, range(24)))
    assert len(workspace.summary()['background_jobs']) == 24
    assert workspace.summary()['remaining'] == ['verify coverage']


@pytest.mark.asyncio
async def test_batch_exceeds_concurrency_capacity_then_restart_drains_in_order_without_model(sqlite_engine):
    delivered = []
    async def deliver(payload, guard):
        guard.require_valid()
        delivered.append(payload['arguments']['ref_id'])
        return 'completed'
    queue, payload, now = setup_queue(sqlite_engine, deliver=deliver)
    async def submit(args):
        return queue.submit({**payload, 'arguments': args}, parent_turn='same-request', before_commit=lambda: None)
    offered = AgentTool('download_web_video', 'download', {'type': 'object', 'properties': {
        'ref_id': {'type': 'string', 'maxLength': 2000}}, 'required': ['ref_id'], 'additionalProperties': False}, submit)
    batch, = build_media_batch_tool((offered,), SimpleNamespace(require_valid=lambda: None))
    args = {'operations': [{'operation': 'download_web_video', 'arguments': {
        'ref_id': 'https://example.com/video/' + str(index)}} for index in range(9)]}
    assert validate_arguments(args, batch.parameters)
    result = await batch.handler(args)
    assert result.data['all_accepted'] and result.data['accepted_count'] == 9
    assert [item['execution_status'] for item in result.data['jobs']] == ['queued'] * 4 + ['pending'] * 5
    duplicate = await batch.handler(args)
    assert [item['job_id'] for item in duplicate.data['jobs']] == [item['job_id'] for item in result.data['jobs']]
    # Recreate only the worker, keeping its SQLite ledger and source authority.
    queue.owner = 'restarted-worker'
    for _ in range(18):
        assert await queue.tick()
    assert not await queue.tick()
    assert delivered == ['https://example.com/video/' + str(index) for index in range(9)]
    assert len(json.dumps(result.data)) < 12000


@pytest.mark.asyncio
async def test_batch_validates_all_arguments_before_queueing_and_preserves_permissions():
    calls = []
    async def handle(args):
        calls.append(args)
        return ToolOutcome({'job_id': len(calls), 'execution_status': 'queued'})
    allowed = AgentTool('download_web_video', '', {'type': 'object', 'properties': {
        'ref_id': {'type': 'string'}}, 'required': ['ref_id'], 'additionalProperties': False}, handle)
    batch, = build_media_batch_tool((allowed,), SimpleNamespace(require_valid=lambda: None))
    result = await batch.handler({'operations': [
        {'operation': 'download_web_video', 'arguments': {'ref_id': 'https://example.com'}},
        {'operation': 'download_web_video', 'arguments': {'image_ids': ['forged']}}]})
    assert result.status == 'blocked' and not calls
    from dataclasses import replace
    batch, = build_media_batch_tool((replace(allowed, authorization=lambda: False),),
        SimpleNamespace(require_valid=lambda: None))
    result = await batch.handler({'operations': [{'operation': 'download_web_video',
        'arguments': {'ref_id': 'https://example.com'}}]})
    assert result.status == 'blocked' and not calls


@pytest.mark.asyncio
async def test_pending_work_cancellation_and_withdrawal_stop_remaining_batch(sqlite_engine):
    from app.storage.db import session_scope
    from app.storage.models import Job, Message
    queue, payload, _ = setup_queue(sqlite_engine)
    jobs = [queue.submit({**payload, 'arguments': {'ref_id': f'https://example.com/{i}'}},
        parent_turn='request', before_commit=lambda: None).data['job_id'] for i in range(6)]
    with session_scope(sqlite_engine) as session:
        assert session.get(Job, jobs[-1]).status == 'pending'
        session.get(Message, payload['source_id']).raw_json = {'delivery_state': 'deleted'}
    for _ in range(12):
        if not await queue.tick():
            break
    queue.cancel_scope(payload['scope_id'])
    with session_scope(sqlite_engine) as session:
        assert all(session.get(Job, identifier).status in {'blocked', 'cancelled'} for identifier in jobs)


@pytest.mark.asyncio
async def test_task_cancel_does_not_cancel_other_requesters_or_requests(sqlite_engine, tmp_path):
    from app.storage.db import session_scope
    from app.storage.models import Job
    queue, payload, _ = setup_queue(sqlite_engine)
    workspace = task(tmp_path)
    workspace.begin()
    workspace.cancel_background = lambda identifier: queue.cancel_task(payload['scope_id'], payload['actor_id'],
        identifier, before_commit=lambda: None)
    own = queue.submit({**payload, 'parent_task_id': workspace.task_id},
        parent_turn='own', before_commit=lambda: None).data['job_id']
    other = queue.submit({**payload, 'parent_task_id': 'b'*24},
        parent_turn='other', before_commit=lambda: None).data['job_id']
    workspace.update(progress='cancelled', next_step='', findings=[], status='cancelled')
    with session_scope(sqlite_engine) as session:
        assert session.get(Job, own).status == 'cancelled'
        assert session.get(Job, other).status == 'queued'


@pytest.mark.asyncio
async def test_real_native_loop_does_not_close_after_progress_receipt(tmp_path):
    from app.agent.harness_runtime import HarnessToolAgent
    from test_harness_runtime import Guard, model_client, responses_sse, tool
    workspace = task(tmp_path/'task')
    workspace.begin(remaining=['read final resource'])
    def response(body, number):
        if number == 1:
            return responses_sse(name='send_group', arguments={'text': 'progress'}, number=number)
        if number == 2:
            return responses_sse(name='read_context', arguments={}, call_id='read-resource', number=number)
        return responses_sse(text='phase complete', number=number)
    client, requests, _ = model_client(response)
    async def send(args):
        return ToolOutcome({}, 'sent', 'actual-progress-receipt')
    async def read(args):
        workspace.update(progress='resource read', next_step='', findings=['verified'], status='completed', remaining=[])
        return ToolOutcome({'content': 'real fixture resource'})
    agent = HarnessToolAgent(decision_client=client, task_workspace=workspace,
        session_home=tmp_path/'native', finish_after_delivery=lambda: True)
    try:
        result = await agent.run(['synthetic task'], (
            tool('send_group', send, effect='write', properties={'text': {'type': 'string'}}, required=('text',)),
            tool('read_context', read)), Guard())
        assert result.status == 'sent' and result.task_status['status'] == 'completed', result
        assert len(requests) == 2
        assert any(item.get('tool') == 'read_context' for item in result.trace)
    finally:
        agent.close()
        client.http_client.close()
