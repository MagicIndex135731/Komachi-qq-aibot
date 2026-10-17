import json
from pathlib import Path
import time

import pytest

from app.agent.runtime_trace import AgentTrace, TraceReader, sanitize


def test_owner_groups_rotations_but_keeps_ephemeral_judgments_distinct(tmp_path):
    for native, lifetime in [('a','persistent'), ('a','persistent'), ('b','persistent'), ('c','ephemeral'), ('d','ephemeral')]:
        trace = AgentTrace(tmp_path, context={'surface':'group','scope_id':900000001,
            'context_session_key':native*24, 'context_lifetime':lifetime},
            session_key=native, conversation_key='same-group')
        trace.finish('silent','model_finished'); trace.close()
    groups = TraceReader(tmp_path).list_conversations()
    assert len(groups) == 2
    ephemeral = next(g for g in groups if g['identity_quality'] == 'ephemeral')
    assert ephemeral['turn_count'] == 2 and ephemeral['native_session_count'] == 0
    assert ephemeral['ephemeral_execution_count'] == 2
    owner = next(g for g in groups if g['identity_quality'] == 'logical')
    assert owner['turn_count'] == 3 and owner['native_session_count'] == 2
    assert len(owner['conversation_aliases']) == 2
    assert TraceReader(tmp_path).read_conversation('c-'+'a'*24)['conversation_id'] == owner['conversation_id']


def test_legacy_judge_route_does_not_count_each_temporary_execution_as_persistent_context(tmp_path):
    for number in range(3):
        trace=AgentTrace(tmp_path,context={'surface':'group','scope_id':900000001,
            'context_kind':'background_group','trigger_kind':'participation_judge'},
            session_key='temporary-'+str(number),conversation_key='same-judgment-entry')
        trace.finish('silent','model_finished');trace.close()
    group=TraceReader(tmp_path).list_conversations()[0]
    assert group['identity_quality']=='ephemeral' and group['turn_count']==3
    assert group['native_session_count']==0 and group['ephemeral_execution_count']==3
    assert all(t['context_identity_legacy'] for t in group['turns'])


def test_real_parent_trace_links_after_native_grouping_and_provider_metadata(tmp_path):
    from app.agent.runtime_trace import current_trace_context, observe_with
    parent = AgentTrace(tmp_path, context={'surface':'private','scope_id':900000101,
        'context_session_key':'e'*24, 'context_lifetime':'persistent'},
        session_key='native', conversation_key='logical')
    with observe_with(parent):
        lineage = current_trace_context()
    parent.event('model.finished', {'request_id':'m01-a01','provider_response_id':'resp_two',
        'provider_previous_response_id':'resp_one','provider_chain_id':'resp_one',
        'prompt_cache_key_id':'f'*24, 'continuation_mode':'incremental'})
    parent.event('context.retention', {'retain_history':True,'reason':'useful_work_retained'})
    parent.finish('silent','model_finished'); parent.close()
    assert lineage['workflow_conversation_id'] == 'c-'+parent.meta['conversation_key']
    worker = AgentTrace(tmp_path, context={**lineage,'surface':'private','scope_id':900000101,
        'trigger_kind':'media_delivery','job_id':12}, session_key='tool-only', conversation_key='worker')
    worker.finish('sent','delivered'); worker.close()
    groups = TraceReader(tmp_path).list_conversations()
    assert len(groups) == 1 and groups[0]['task_count'] == 1
    assert groups[0]['native_session_count'] == 1
    assert groups[0]['provider_chain_ids'] == ['resp_one']
    assert groups[0]['turns'][0]['context_retention']['retain_history']


def test_old_native_parent_link_resolves_after_owner_rotation_and_expiry(tmp_path):
    from app.monitor.metrics import ConversationMetrics
    from app.monitor.records import MonitorRecords
    old = AgentTrace(tmp_path, context={'surface':'group','scope_id':900000001,
        'context_session_key':'a'*24}, session_key='old', conversation_key='owner-one')
    old.finish('ok','done');old.close()
    current = AgentTrace(tmp_path, context={'surface':'group','scope_id':900000001,
        'context_session_key':'b'*24}, session_key='new', conversation_key='owner-one')
    current.finish('ok','done');current.close()
    worker = AgentTrace(tmp_path, context={'surface':'group','scope_id':900000001,
        'execution_kind':'background_tool','job_id':12,
        'parent_trace_id':'f'*32, 'workflow_conversation_id':'c-'+'a'*24},
        session_key='worker', conversation_key='worker')
    worker.finish('ok','done');worker.close()
    foreign = AgentTrace(tmp_path, context={'surface':'group','scope_id':900000002,
        'context_session_key':'c'*24}, session_key='foreign', conversation_key='owner-two')
    foreign.finish('ok','done');foreign.close()
    reader = TraceReader(tmp_path)
    assert len(reader.list_conversations()) == 2
    owner = reader.read_conversation('c-'+'a'*24)
    assert owner['turn_count'] == 2 and owner['task_count'] == 1
    assert ConversationMetrics(reader).read(owner['conversation_id'])['usage']['request_count'] == 0
    assert MonitorRecords(reader).delete_conversation(owner['conversation_id'])['deleted_turns'] == 3
    assert reader.list_conversations()[0]['turns'][0]['trace_id'] == foreign.trace_id


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


def test_workflow_links_workers_and_child_model_without_changing_main_context_or_usage(tmp_path):
    from app.agent.runtime_trace import current_trace_context, observe_with
    from app.monitor.metrics import ConversationMetrics
    parent = AgentTrace(tmp_path, context={'surface':'private', 'scope_id':900000101,
        'context_kind':'private'}, session_key='main-native', conversation_key='main-conversation')
    assert current_trace_context() == {}
    with observe_with(parent, call_id='synthetic-call'):
        lineage = current_trace_context()
    assert current_trace_context() == {} and lineage['parent_trace_id'] == parent.trace_id
    worker = AgentTrace(tmp_path, context={**lineage, 'surface':'private', 'scope_id':900000101,
        'job_id':12, 'execution_kind':'delegation_job', 'context_kind':'research_job'},
        session_key='job', conversation_key='job:12')
    with observe_with(worker):
        child = AgentTrace(tmp_path, context={**current_trace_context(), 'surface':'private',
            'scope_id':900000101, 'context_kind':'research'}, session_key='child-native', conversation_key='child')
    child.request('m01-a01', {'input':['child-only']}, {}, purpose='agent')
    child.event('model.finished', {'request_id':'m01-a01', 'input_tokens':100,
        'cached_input_tokens':0, 'output_tokens':20})
    for trace in (parent, worker, child):
        trace.finish('ok', 'synthetic'); trace.close()
    reader = TraceReader(tmp_path)
    main = reader.read_conversation(lineage['workflow_conversation_id'])
    assert main['turn_count'] == 1 and main['native_session_count'] == 1 and main['task_count'] == 1
    assert {t['trace_id'] for t in main['task_traces']} == {worker.trace_id, child.trace_id}
    assert len(reader.list_conversations()) == 2  # main and real research, no fake worker chat
    assert ConversationMetrics(reader).read(main['conversation_id'])['usage']['request_count'] == 0
    assert reader.read_trace(parent.trace_id)['events'][-1]['kind'] == 'turn.completed'
    assert not list(parent.path.glob('request-*.json'))
    assert reader.read_request(child.trace_id, 'm01-a01')['body']['input'] == ['child-only']
    from app.monitor.records import MonitorRecords
    deleted = MonitorRecords(reader).delete_conversation(main['conversation_id'])
    assert deleted['deleted_turns'] == 3 and reader.list_traces() == []


def test_worker_ancestry_never_guesses_scope_and_survives_expired_parent(tmp_path):
    from app.agent.runtime_trace import current_trace_context, observe_with
    parent = AgentTrace(tmp_path, context={'surface':'group', 'scope_id':900000001},
        session_key='main', conversation_key='main')
    with observe_with(parent):lineage = current_trace_context()
    foreign = AgentTrace(tmp_path, context={**lineage, 'surface':'group', 'scope_id':900000002,
        'execution_kind':'background_tool', 'context_kind':'media_task', 'job_id':1},
        session_key='foreign', conversation_key='foreign-job')
    for trace in (parent, foreign):trace.finish('ok', 'done');trace.close()
    reader = TraceReader(tmp_path)
    assert reader.read_conversation(lineage['workflow_conversation_id'])['task_count'] == 0
    assert len(reader.list_conversations()) == 2
    from app.monitor.records import MonitorRecords
    MonitorRecords(reader).delete_traces([parent.trace_id])
    assert reader.list_conversations()[0]['turns'][0]['trace_id'] == foreign.trace_id


def test_malformed_observation_ancestry_does_not_break_legacy_listing(tmp_path):
    trace = AgentTrace(tmp_path, context={'workflow_conversation_id':{},'execution_kind':[], 'job_id':{}},
        session_key='legacy')
    trace.finish('ok','synthetic');trace.close()
    assert TraceReader(tmp_path).list_conversations()[0]['task_count'] == 0
