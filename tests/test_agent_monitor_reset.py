"""Reset the monitor projection without deleting execution or chat records."""
import time
from types import SimpleNamespace

import pytest

from app.agent.monitor_reset import read_reset, record_context_reset
from app.agent.runtime_trace import AgentTrace, TraceReader, current_trace_context, observe_with
from app.monitor.metrics import ConversationMetrics


def trace(root, surface='group', scope=900000001, source=1, **extra):
    return AgentTrace(root, context={'surface':surface, 'scope_id':scope,
        'source_id':source, 'context_kind':surface, **extra}, session_key=str(source),
        conversation_key=f'{surface}:{scope}' if not extra.get('execution_kind') else f'job:{source}')


def close(t, tokens=100):
    t.request('m01-a01', {'input':[]}, {}, purpose='agent')
    t.event('model.finished', {'request_id':'m01-a01', 'input_tokens':tokens,
        'output_tokens':2, 'cached_input_tokens':tokens//2})
    t.finish('sent', 'done'); t.close()


def reset(root, surface='group', scope=900000001, source=10, at=None):
    settings=SimpleNamespace(agent_monitor_enabled=True, data_dir=root.parent)
    record_context_reset(settings, surface=surface, scope_id=scope, source_id=source,
        reset_at=at or time.time())


@pytest.mark.parametrize('surface', ['group', 'private'])
def test_reset_keeps_empty_owner_and_restarts_usage_after_restart(tmp_path, surface):
    root=tmp_path/'agent_monitor'
    old=trace(root, surface); close(old)
    foreign=trace(root, surface, scope=900000002); close(foreign)
    other_surface=trace(root, 'private' if surface=='group' else 'group'); close(other_surface)
    reader=TraceReader(root); groups=reader.list_conversations()
    owner=next(c for c in groups if c['surface']==surface and c['scope_id']==900000001)
    metrics=ConversationMetrics(reader)
    assert metrics.read(owner['conversation_id'])['usage']['input_tokens']==100
    reset(root, surface)
    empty=reader.read_conversation(owner['conversation_id'])
    assert empty['turns']==[] and empty['task_count']==empty['native_session_count']==0
    assert empty['reset_source_id']==10 and empty['status']=='reset'
    stats=metrics.read(owner['conversation_id'])
    assert stats['usage']['request_count']==stats['usage']['real_total_tokens']==0
    assert stats['context']['input_tokens'] is None and stats['usage']['cache_ratio'] is None
    assert len(reader.list_conversations())==3
    assert (root/old.trace_id/'meta.json').exists()  # Projection reset retains diagnostics.
    new=trace(root, surface, source=11); close(new, tokens=40)
    restarted=TraceReader(root).read_conversation(owner['conversation_id'])
    assert [m['trace_id'] for m in restarted['turns']]==[new.trace_id]
    usage=metrics.read(owner['conversation_id'])['usage']
    assert usage['request_count']==1 and usage['input_tokens']==40
    assert usage['cached_input_tokens']==20 and usage['real_total_tokens']==42


def test_late_old_sources_and_descendants_stay_hidden(tmp_path):
    root=tmp_path/'agent_monitor'; old=trace(root)
    with observe_with(old): lineage=current_trace_context()
    reset(root)
    close(old)  # Old writer can finish after the reset marker.
    late=trace(root, source=1, execution_kind='background_tool', job_id=12, **lineage); close(late)
    no_source=trace(root, source=None, execution_kind='background_tool', job_id=13, **lineage); close(no_source)
    close(trace(root, source=2, execution_kind='background_tool', job_id=14))
    groups=TraceReader(root).list_conversations()
    assert len(groups)==1 and groups[0]['turns']==[] and not groups[0].get('task_traces')
    fresh=trace(root, source=11)
    with observe_with(fresh): new_lineage=current_trace_context()
    child=trace(root, source=11, execution_kind='background_tool', job_id=15, **new_lineage); close(child)
    close(fresh)
    group=TraceReader(root).list_conversations()[0]
    assert [t['trace_id'] for t in group['turns']]==[fresh.trace_id]
    assert [t['trace_id'] for t in group['task_traces']]==[child.trace_id]


def test_marker_is_monotonic_optional_and_scope_validated(tmp_path):
    root=tmp_path/'agent_monitor'; reset(root, source=20)
    reset(root, source=10)
    assert read_reset(root, 'group', 900000001)['source_id']==20
    assert read_reset(root, 'private', 900000001) is None
    settings=SimpleNamespace(agent_monitor_enabled=False, data_dir=tmp_path)
    record_context_reset(settings, surface='group', scope_id=900000001, source_id=30, reset_at=time.time())
    assert read_reset(root, 'group', 900000001)['source_id']==20
    bad=tmp_path/'blocked'; bad.write_text('file')
    settings=SimpleNamespace(agent_monitor_enabled=True, data_dir=bad)
    record_context_reset(settings, surface='group', scope_id=900000001, source_id=30, reset_at=time.time())


def test_api_returns_fresh_epoch_and_zero_metrics_without_restarting_server(tmp_path):
    import json
    from threading import Thread
    from urllib.request import urlopen
    from app.monitor.server import create_server
    root=tmp_path/'agent_monitor'; old=trace(root); close(old)
    server=create_server(root, port=0)
    thread=Thread(target=server.serve_forever, daemon=True); thread.start()
    def get(path):
        with urlopen(f'http://127.0.0.1:{server.server_port}'+path, timeout=3) as response:
            return json.load(response)
    try:
        owner=get('/api/conversations')['conversations'][0]['conversation_id']
        assert get(f'/api/conversations/{owner}/metrics')['usage']['request_count']==1
        reset(root)
        assert get(f'/api/conversations/{owner}')['turns']==[]
        stats=get(f'/api/conversations/{owner}/metrics')
        assert stats['reset_source_id']==10 and stats['usage']['request_count']==0
    finally:
        server.shutdown(); server.server_close(); thread.join(3)


def test_deleting_reset_owner_includes_hidden_monitor_records(tmp_path):
    from app.monitor.records import MonitorRecords
    root=tmp_path/'agent_monitor'; old=trace(root); close(old)
    reader=TraceReader(root); owner=reader.list_conversations()[0]['conversation_id']
    reset(root)
    assert reader.read_conversation(owner)['turns']==[]
    assert MonitorRecords(reader).delete_conversation(owner)['deleted_turns']==1
    assert reader.list_conversations()==[]


def test_post_commit_turn_survives_a_delayed_marker_write(tmp_path):
    root=tmp_path/'agent_monitor'; old=trace(root); close(old)
    committed_at=time.time()
    new=trace(root, source=11); close(new)
    reset(root, at=committed_at)
    assert [t['trace_id'] for t in TraceReader(root).list_conversations()[0]['turns']]==[new.trace_id]
