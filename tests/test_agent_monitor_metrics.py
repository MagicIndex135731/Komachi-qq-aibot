from pathlib import Path
import json

import pytest

from app.agent.runtime_trace import AgentTrace, TraceReader
from app.monitor.metrics import ConversationMetrics


def make_trace(root, session='native', **context):
    return AgentTrace(root, context=context, session_key=session, conversation_key='synthetic-group')


def request(trace, rid, estimate=90, window=1000, purpose='agent'):
    trace.request(rid, {'model':'synthetic','input':[]}, {'context_estimated_tokens':estimate,
        'context_window_tokens':window, 'context_input_ceiling':900}, purpose=purpose)


def finish(trace, rid, input_tokens, cached, output=2, **other):
    trace.event('model.finished', {'request_id':rid,'input_tokens':input_tokens,
        'cached_input_tokens':cached,'output_tokens':output, **other})


def read(root):
    reader=TraceReader(root)
    return ConversationMetrics(reader).read(reader.list_conversations()[0]['conversation_id'])


def test_window_uses_latest_input_and_cache_is_weighted_without_double_count(tmp_path):
    trace=make_trace(tmp_path)
    request(trace,'m01-a01');finish(trace,'m01-a01',100,20)
    request(trace,'m02-a01',estimate=250);finish(trace,'m02-a01',300,100,5)
    finish(trace,'m02-a01',300,100,5)  # Repeated final SSE usage is one request.
    trace.finish('sent','done');trace.close()
    stats=read(tmp_path)
    assert stats['context']['input_tokens']==300 and stats['context']['window_tokens']==1000
    assert stats['context']['source']=='reported' and stats['context']['ratio']==0.3
    assert stats['context']['cache_ratio']==pytest.approx(1/3)
    assert stats['usage']['input_tokens']==400 and stats['usage']['output_tokens']==7
    assert stats['usage']['cached_input_tokens']==120 and stats['usage']['cache_ratio']==0.3
    assert stats['usage']['uncached_input_tokens']==280
    assert stats['usage']['uncached_known_requests']==2
    assert stats['usage']['request_count']==stats['usage']['cache_known_requests']==2


def test_pending_main_request_uses_estimate_and_compaction_only_affects_cost(tmp_path):
    trace=make_trace(tmp_path)
    request(trace,'m01-a01');finish(trace,'m01-a01',100,20)
    request(trace,'c01-a01',estimate=600,purpose='compaction');finish(trace,'c01-a01',600,500)
    request(trace,'m02-a01',estimate=50)
    trace.finish('silent','done');trace.close()
    stats=read(tmp_path)
    assert stats['context']['input_tokens']==50 and stats['context']['source']=='estimated'
    assert stats['context']['cache_ratio'] is None
    assert stats['usage']['request_count']==3 and stats['usage']['input_tokens']==700
    assert stats['usage']['cache_known_requests']==2


def test_missing_and_invalid_usage_are_unknown_and_zero_cache_is_real_zero(tmp_path):
    trace=make_trace(tmp_path)
    request(trace,'m01-a01');finish(trace,'m01-a01',100,None)
    request(trace,'m01-a02');finish(trace,'m01-a02',True,5,usage_status='invalid')
    request(trace,'m02-a01');finish(trace,'m02-a01',200,0)
    trace.finish('ok','done');trace.close()
    stats=read(tmp_path)
    assert stats['usage']['input_tokens']==300 and stats['usage']['input_known_requests']==2
    assert stats['usage']['cache_ratio']==0 and stats['usage']['cache_known_requests']==1
    assert stats['context']['cache_ratio']==0


def test_context_restart_does_not_reuse_old_window_occupancy(tmp_path):
    trace=make_trace(tmp_path)
    request(trace,'m01-a01');finish(trace,'m01-a01',800,100)
    trace.finish('ok','done');trace.close()
    trace=make_trace(tmp_path,session='new-native',context_window_tokens=2000)
    trace.finish('silent','done');trace.close()
    stats=read(tmp_path)
    assert stats['context']['input_tokens'] is None and stats['context']['window_tokens']==2000
    assert stats['context']['source']=='unknown' and stats['usage']['input_tokens']==800


def test_incremental_cache_does_not_reopen_unchanged_history_or_read_snapshots(tmp_path,monkeypatch):
    trace=make_trace(tmp_path)
    request(trace,'m01-a01');trace._queue.join()
    reader=TraceReader(tmp_path);metrics=ConversationMetrics(reader);cid=reader.list_conversations()[0]['conversation_id']
    original=Path.open;opened=[]
    def counted(path,*args,**kwargs):
        opened.append(path.name);return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',counted)
    try:
        assert metrics.read(cid)['context']['source']=='estimated'
        first=opened.count('events.jsonl');assert first==1
        metrics.read(cid);assert opened.count('events.jsonl')==first
        finish(trace,'m01-a01',123,20);trace._queue.join()
        assert metrics.read(cid)['context']['input_tokens']==123
        assert opened.count('events.jsonl')==first+1
        assert not any(name.startswith(('request-','stream-')) for name in opened)
    finally:trace.close()


def test_metrics_blocks_symlink_event_escape(tmp_path):
    root=tmp_path/'records';trace=make_trace(root);trace.finish('ok','done');trace.close()
    outside=tmp_path/'outside';outside.write_text('private-data')
    events=trace.path/'events.jsonl';events.unlink();events.symlink_to(outside)
    with pytest.raises(ValueError):read(root)


def test_legacy_missing_window_and_usage_stay_unknown(tmp_path):
    trace=make_trace(tmp_path)
    trace.request('m01-a01',{'input':[]},{},purpose='agent')
    trace.finish('ok','done');trace.close()
    stats=read(tmp_path)
    assert stats['context']['window_tokens'] is None and stats['context']['input_tokens'] is None
    assert stats['usage']['input_tokens'] is None and stats['usage']['cache_ratio'] is None


def test_old_window_can_use_explicit_current_configuration_reference(tmp_path):
    import time
    trace=make_trace(tmp_path)
    trace.request('m01-a01',{'input':[]},{'context_estimated_tokens':100},purpose='agent')
    trace.finish('ok','done');trace.close()
    config=tmp_path/'window-config.json'
    config.write_text(json.dumps({'version':1,'window_tokens':2000,'captured_at':time.time()}))
    stats=read(tmp_path)
    assert stats['context']['window_tokens']==2000 and stats['context']['ratio']==0.05
    assert stats['context']['window_source']=='configured_reference'
    config.write_text(json.dumps({'version':1,'window_tokens':2000,'captured_at':time.time()-90000}))
    assert read(tmp_path)['context']['window_tokens'] is None


def test_recorded_window_wins_over_configuration_reference(tmp_path):
    import time
    trace=make_trace(tmp_path);request(trace,'m01-a01',window=1000)
    trace.finish('ok','done');trace.close()
    (tmp_path/'window-config.json').write_text(json.dumps({'version':1,'window_tokens':2000,'captured_at':time.time()}))
    stats=read(tmp_path)
    assert stats['context']['window_tokens']==1000 and stats['context']['window_source']=='captured'


def test_global_scan_budget_prioritizes_current_context_and_fills_history(tmp_path,monkeypatch):
    old=make_trace(tmp_path,session='old')
    request(old,'m01-a01')
    for _ in range(30):
        old.event('synthetic.padding',{'text':'x'*300});old._queue.join()
    finish(old,'m01-a01',100,20);old.finish('ok','done');old.close()
    latest=make_trace(tmp_path,session='current')
    request(latest,'m01-a01');finish(latest,'m01-a01',300,100)
    latest.finish('ok','done');latest.close()
    reader=TraceReader(tmp_path);metrics=ConversationMetrics(reader,max_scan_bytes=1024)
    cid=reader.list_conversations()[0]['conversation_id'];opened=[];original=Path.open
    class Counted:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):return self
        def __exit__(self,*args):return self.stream.__exit__(*args)
        def __getattr__(self,key):return getattr(self.stream,key)
        def readline(self,size):
            raw=self.stream.readline(size);opened.append(len(raw));return raw
    def wrapper(path,*args,**kwargs):
        value=original(path,*args,**kwargs)
        return Counted(value) if path.name=='events.jsonl' else value
    monkeypatch.setattr(Path,'open',wrapper)
    for _ in range(50):
        opened.clear();stats=metrics.read(cid)
        assert sum(opened)<=1024 and stats['context']['input_tokens']==300
        if stats['usage']['events_complete']:break
    assert stats['usage']['events_complete'] and stats['usage']['input_tokens']==400
    opened.clear();metrics.read(cid);assert not opened
