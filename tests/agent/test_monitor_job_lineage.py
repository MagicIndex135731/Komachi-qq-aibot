"""Recover task ownership from saved queue receipts, never proximity."""
from app.agent.runtime_trace import AgentTrace, TraceReader, current_trace_context, observe_with


def trace(root, key, **extra):
    return AgentTrace(root, context={'surface': 'group', 'scope_id': 900000001,
        'actor_id': 900000101, 'source_id': 51, **extra}, session_key=key, conversation_key=key)


def test_saved_job_result_recovers_execution_delivery_and_origin_without_rewriting(tmp_path):
    parent = trace(tmp_path, 'main', context_kind='group')
    reader = TraceReader(tmp_path)
    worker = trace(tmp_path, 'job', context_kind='media_task', execution_kind='background_tool', job_id=12)
    with observe_with(worker):
        delivery = trace(tmp_path, 'job', **current_trace_context(), context_kind='media_task',
            execution_kind='tool_delivery')
    for item in (worker, delivery):
        item.finish('ok', 'synthetic'); item.close()
    before = (worker.path / 'meta.json').read_bytes()
    assert len(reader.list_conversations()) == 2
    parent.event('tool.completed', {'name': 'download_web_video', 'call_id': 'local-download',
        'status': 'ok', 'data': {'job_id': 12}})
    parent.finish('ok', 'synthetic'); parent.close()
    groups = reader.list_conversations()
    assert len(groups) == 1 and groups[0]['task_count'] == 1
    assert groups[0]['turn_count'] == 1
    assert {m['trace_id'] for m in groups[0]['task_traces']} == {worker.trace_id, delivery.trace_id}
    for meta in groups[0]['task_traces']:
        assert meta['workflow_id'] == parent.trace_id and meta['parent_tool_call_id'] == 'local-download'
    assert (worker.path / 'meta.json').read_bytes() == before


def test_recovery_requires_unique_saved_job_receipt_and_same_scope_source_actor(tmp_path):
    parent = trace(tmp_path, 'main')
    parent.event('job.queued', {'job_id': 12})
    parent.finish('ok', 'synthetic'); parent.close()
    for key, fields in [('foreign', {'scope_id': 900000002}), ('other-source', {'source_id': 52}),
                        ('other-actor', {'actor_id': 900000102}), ('no-receipt', {'job_id': 13})]:
        worker = trace(tmp_path, key, **{'context_kind': 'media_task', 'execution_kind': 'background_tool',
            'job_id': 12, **fields})
        worker.finish('ok', 'synthetic'); worker.close()
    reader = TraceReader(tmp_path)
    assert len(reader.list_conversations()) == 5
    ambiguous = trace(tmp_path, 'second-main')
    ambiguous.event('tool.completed', {'status': 'ok', 'data': {'jobs': [{'job_id': 12}]}})
    ambiguous.finish('ok', 'synthetic'); ambiguous.close()
    worker = trace(tmp_path, 'ambiguous-job', context_kind='media_task', execution_kind='background_tool', job_id=12)
    worker.finish('ok', 'synthetic'); worker.close()
    assert all(c['task_count'] == 0 for c in reader.list_conversations())
