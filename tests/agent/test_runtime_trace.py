import json
from pathlib import Path
import time

import pytest

from app.agent.runtime_trace import AgentTrace, TraceReader, sanitize


def test_trace_redacts_credentials_images_and_hidden_reasoning():
    result = sanitize({'authorization':'Bearer sensitive', 'api_key':'sensitive',
        'messages':[{'role':'assistant','reasoning_content':'hidden private analysis','content':'API_KEY=secret-value'},
          {'role':'user','content':[{'type':'input_image','image_url':'data:image/png;base64,secret-pixels'}]}],
        'input':[{'type':'reasoning','encrypted_content':'private-cipher','content':[{'type':'reasoning_text','text':'hidden'}],
                  'summary':[{'type':'summary_text','text':'公开摘要'}]}]})
    encoded=json.dumps(result,ensure_ascii=False)
    assert all(s not in encoded for s in ('sensitive','secret-value','secret-pixels','private-cipher','hidden private','"hidden"'))
    assert '公开摘要' in encoded and '[已省略' in encoded


def test_request_snapshots_streams_receipts_and_prefix(tmp_path):
    root=tmp_path/'traces'
    trace=AgentTrace(root,context={'surface':'group','scope_id':900000001,'source_id':1},session_key='synthetic')
    one={'model':'synthetic','instructions':'固定规则','tools':[], 'input':[{'role':'user','content':'消息一'}]}
    trace.request('m01-a01',one,{'context_estimated_tokens':100},purpose='agent')
    trace.stream('m01-a01',{'type':'response.reasoning_summary_text.delta','delta':'先核对资料'})
    trace.stream('m01-a01',{'type':'response.output_text.delta','delta':'完整回答'})
    trace.event('tool.started',{'name':'send_group','arguments':{'text':'完整回答'}})
    trace.event('tool.completed',{'name':'send_group','status':'sent','receipt_id':'synthetic-receipt'})
    trace.request('m02-a01',{**one,'input':[*one['input'],{'role':'assistant','content':'回答一'},{'role':'user','content':'消息二'}]}, {},purpose='agent')
    trace.finish('sent','model_finished');trace.close()
    reader=TraceReader(root)
    summary=reader.list_traces()[0]
    assert summary['status']=='sent' and summary['surface']=='group'
    events=reader.read_trace(trace.trace_id)['events']
    assert any(e['kind']=='tool.completed' and e['data']['receipt_id']=='synthetic-receipt' for e in events)
    request=reader.read_request(trace.trace_id,'m02-a01')
    assert request['meta']['shared_prefix_chars']>0 and request['meta']['previous_request_id']=='m01-a01'
    streams=reader.read_trace(trace.trace_id)['streams']
    assert any(s['output']=='完整回答' and s['summary']=='先核对资料' for s in streams.values())
    assert not reader.read_trace(trace.trace_id,after=events[-1]['seq'])['events']


def test_hidden_reasoning_is_status_only_even_in_replay(tmp_path):
    trace=AgentTrace(tmp_path,context={'surface':'private'},session_key='synthetic')
    trace.stream('m01-a01',{'choices':[{'delta':{'reasoning_content':'hidden reasoning', 'content':'visible'}}]})
    trace.stream('m01-a01',{'type':'response.reasoning_text.delta','delta':'also hidden'})
    trace.finish('ok','done');trace.close()
    raw=''.join(p.read_text() for p in tmp_path.rglob('*.json*'))
    assert 'hidden reasoning' not in raw and 'also hidden' not in raw
    assert 'visible' in raw and 'reasoning_chars' in raw


def test_trace_disabled_or_io_failure_does_not_raise(tmp_path):
    trace=AgentTrace(None,context={},session_key='none')
    trace.request('m01-a01',{'input':'anything'}, {},purpose='agent');trace.finish('ok','done');trace.close()
    bad=tmp_path/'file';bad.write_text('occupied')
    trace=AgentTrace(bad,context={},session_key='none')
    trace.event('any',{'value':'test'});trace.close()


def test_trace_reader_rejects_paths_and_symlinks(tmp_path):
    reader=TraceReader(tmp_path)
    with pytest.raises(ValueError):reader.read_trace('../outside')
    with pytest.raises(ValueError):reader.read_request('a'*32,'../outside')
    linked=tmp_path/('a'*32)
    try:linked.symlink_to(tmp_path.parent,target_is_directory=True)
    except OSError:pytest.skip('symlink unsupported')
    with pytest.raises(ValueError):reader.read_trace('a'*32)


def test_snapshot_budget_is_explicit(tmp_path):
    trace=AgentTrace(tmp_path,context={},session_key='budget',snapshot_bytes=1024)
    trace.request('m01-a01',{'input':'x'*4000}, {},purpose='agent');trace.finish('ok','done');trace.close()
    request=TraceReader(tmp_path).read_request(trace.trace_id,'m01-a01')
    assert request['meta']['truncated'] and request['body']['_truncated']


def test_structural_and_string_truncation_are_explicit(tmp_path):
    trace=AgentTrace(tmp_path,context={},session_key='budget')
    trace.request('m01-a01',{'input':'x'*(1024*1024+1)}, {},purpose='agent')
    trace.finish('ok','done');trace.close()
    request=TraceReader(tmp_path).read_request(trace.trace_id,'m01-a01')
    assert request['meta']['truncated'] and '[内容截断]' in request['body']['input']
    result=sanitize({str(i):i for i in range(4097)})
    assert result['_truncated'] and result['omitted_keys']==1


def test_stream_files_count_toward_turn_budget(tmp_path,monkeypatch):
    monkeypatch.setattr('app.agent.runtime_trace.MAX_TURN_BYTES',2048)
    trace=AgentTrace(tmp_path,context={},session_key='small')
    trace.stream('m01-a01',{'type':'response.output_text.delta','delta':'x'*4000},force=True)
    trace.finish('ok','done');trace.close()
    meta=TraceReader(tmp_path).list_traces()[0]
    assert meta['dropped_events']>0 and meta['status']=='ok'
    assert not (trace.path/'stream-m01-a01.json').exists()


def test_retention_keeps_active_turns_and_only_owned_files(tmp_path):
    for i in range(3):
        trace=AgentTrace(tmp_path,context={},session_key=str(i),max_turns=2)
        trace.finish('ok','done');trace.close()
    active=AgentTrace(tmp_path,context={},session_key='active',max_turns=2)
    assert len(TraceReader(tmp_path).list_traces())<=3
    assert any(t['trace_id']==active.trace_id for t in TraceReader(tmp_path).list_traces())
    active.close()


def test_conversations_join_turns_but_keep_native_restarts_and_tasks_distinct(tmp_path):
    for session,scope,state in [('native-one','group:900000001','new'),('native-one','group:900000001','continued'),
                                ('native-two','group:900000001','restored'),('research','research:900000001:task','new')]:
        trace=AgentTrace(tmp_path,context={'surface':'group','scope_id':900000001,'context_state':state,
            'trigger_kind':'group_direct','trigger_text':'合成问题'},session_key=session,conversation_key=scope)
        trace.finish('ok','done');trace.close()
    reader=TraceReader(tmp_path);conversations=reader.list_conversations()
    assert len(conversations)==2
    group=next(c for c in conversations if c['turn_count']==3)
    assert group['native_session_count']==2 and group['identity_quality']=='logical'
    assert [t['context_state'] for t in group['turns']]==['new','continued','restored']
    assert reader.read_conversation(group['conversation_id'])['turn_count']==3
    with pytest.raises(ValueError):reader.read_conversation('../outside')


def test_legacy_records_never_guess_logical_group_identity(tmp_path):
    for session in ('same','same','different'):
        trace=AgentTrace(tmp_path,context={'surface':'group','scope_id':900000001},session_key=session)
        trace.finish('ok','done');trace.close()
    conversations=TraceReader(tmp_path).list_conversations()
    assert len(conversations)==2 and all(c['identity_quality']=='native_only' for c in conversations)
    assert sorted(c['turn_count'] for c in conversations)==[1,2]


def test_observed_context_transition_does_not_change_checkpoint():
    from app.agent.runtime_trace import context_observation
    checkpoint={'last_status':'failed','context_format_version':5}
    before=dict(checkpoint)
    result=context_observation('group:900000001',previous_id='one',current_id='two',
        checkpoint=checkpoint,history_present=True,restored_count=4)
    assert result['context_state']=='restored' and result['context_reason']=='previous_failed'
    assert result['restored_context_items']==4 and checkpoint==before
    result=context_observation('research:group:900000001:task',previous_id='one',current_id='one',
        checkpoint={},history_present=True,restored_count=0)
    assert result['context_state']=='continued' and result['context_kind']=='research'
