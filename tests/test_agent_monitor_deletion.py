"""Monitor deletion cannot affect execution or reach other application data."""
import json
from threading import Thread
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from app.agent.runtime_trace import AgentTrace, TraceReader
from app.monitor.metrics import ConversationMetrics
from app.monitor.records import MonitorRecords
from app.monitor.server import create_server


def trace(root, *, key='synthetic-group', active=False):
    value=AgentTrace(root,context={'surface':'group','context_kind':'group','scope_id':900000001},
        conversation_key=key,session_key=key)
    value.request('m01-a01',{'model':'synthetic','input':[{'role':'user','content':'isolated input'}]},
        {'context_estimated_tokens':100,'context_window_tokens':1000},purpose='agent')
    if not active:
        value.finish('silent','synthetic');value.close()
    return value


@pytest.fixture
def endpoint(tmp_path):
    value=create_server(tmp_path/'monitor',port=0)
    worker=Thread(target=value.serve_forever,daemon=True);worker.start()
    yield value,tmp_path/'monitor'
    value.shutdown();value.server_close();worker.join()


def call(server,path,*,method='GET',headers=None):
    with urlopen(Request(f'http://127.0.0.1:{server.server_port}'+path,
            headers=headers or {},method=method),timeout=3) as response:
        return json.load(response)


def test_deleting_conversation_removes_all_read_views_and_new_turn_reappears(endpoint,tmp_path):
    server,root=endpoint
    first,second=trace(root),trace(root)
    unrelated=trace(root,key='synthetic-other')
    database=tmp_path/'bot.db';database.write_bytes(b'synthetic-chat-memory-and-jobs')
    listing=call(server,'/api/conversations')['conversations']
    c=next(c for c in listing if c['turn_count']==2)
    route='/api/conversations/'+c['conversation_id']
    assert call(server,route+'/metrics')['usage']['request_count']==2
    token=call(server,'/api/controls')['deletion_token']
    result=call(server,route,method='DELETE',headers={'X-Monitor-Token':token})
    assert result['deleted_turns']==2 and result['scope']=='monitor_records_only'
    assert result['execution_cancelled'] is False
    assert database.read_bytes()==b'synthetic-chat-memory-and-jobs'
    assert (root/unrelated.trace_id).is_dir()
    for path in (route,route+'/metrics','/api/traces/'+first.trace_id,
            '/api/traces/'+second.trace_id+'/requests/m01-a01'):
        with pytest.raises(HTTPError) as exc:call(server,path)
        assert exc.value.code==404
    fresh=trace(root)
    assert call(server,route)['turns'][0]['trace_id']==fresh.trace_id
    assert call(server,route+'/metrics')['usage']['request_count']==1


@pytest.mark.parametrize('headers',[{}, {'X-Monitor-Token':'wrong'}, {'Origin':'https://evil.example'},
    {'Host':'evil.example'},{'Sec-Fetch-Site':'cross-site'}])
def test_delete_requires_same_origin_and_nonce(endpoint,headers):
    server,root=endpoint;value=trace(root)
    c=TraceReader(root).list_conversations()[0]
    token=call(server,'/api/controls')['deletion_token']
    supplied=headers if not headers or 'X-Monitor-Token' in headers else {'X-Monitor-Token':token,**headers}
    with pytest.raises(HTTPError) as exc:
        call(server,'/api/conversations/'+c['conversation_id'],method='DELETE',headers=supplied)
    assert exc.value.code==403 and (root/value.trace_id).is_dir()


@pytest.mark.parametrize('path',['/api/conversations/..%2foutside','/api/conversations/c-invalid',
    '/api/traces/'+('a'*32),'/api/conversations/c-'+('a'*24)+'/metrics',
    '/api/conversations/c-'+('a'*24)+'?all=true'])
def test_invalid_deletion_routes_never_touch_other_records(endpoint,path):
    server,root=endpoint;value=trace(root)
    token=call(server,'/api/controls')['deletion_token']
    with pytest.raises(HTTPError) as exc:call(server,path,method='DELETE',headers={'X-Monitor-Token':token})
    assert exc.value.code==400 and (root/value.trace_id).is_dir()


def test_cutoff_keeps_new_turns_of_mixed_conversation_and_statistics(tmp_path):
    root=tmp_path/'monitor';old,new=trace(root),trace(root)
    cutoff=time.time()-10
    for value,started in ((old,cutoff-10),(new,cutoff+1)):
        path=root/value.trace_id/'meta.json';meta=json.loads(path.read_text());meta['started_at']=started
        path.write_text(json.dumps(meta))
    reader=TraceReader(root);c=reader.list_conversations()[0]
    assert c['turn_count']==2
    assert MonitorRecords(reader).delete_before(cutoff)['trace_ids']==[old.trace_id]
    kept=reader.read_conversation(c['conversation_id'])
    assert kept['turn_count']==1 and kept['turns'][0]['trace_id']==new.trace_id
    assert ConversationMetrics(reader).read(c['conversation_id'])['usage']['request_count']==1


def test_deleted_active_trace_does_not_recreate_itself_or_interrupt_execution(tmp_path):
    value=trace(tmp_path,active=True)
    result=MonitorRecords(TraceReader(tmp_path)).delete_traces([value.trace_id])
    # The trace writer is optional; callers can continue model/tool execution.
    value.event('tool.completed',{'name':'synthetic-read','status':'ok'})
    value.finish('ok','execution-completed');value.close()
    assert value.finished and value.closed and result['execution_cancelled'] is False
    assert not (tmp_path/value.trace_id).exists() and not TraceReader(tmp_path).list_conversations()


@pytest.mark.parametrize('linked',['trace','record'])
def test_linked_paths_cannot_delete_outside_data(tmp_path,linked):
    root=tmp_path/'monitor';value=trace(root)
    outside=tmp_path/'outside';outside.mkdir();sentinel=outside/'meta.json';sentinel.write_text('keep')
    if linked=='trace':
        target=root/('a'*32);destination=outside
    else:
        target=root/value.trace_id/'request-m02-a01.json';destination=sentinel
    try:target.symlink_to(destination,target_is_directory=linked=='trace')
    except OSError:pytest.skip('symlink unavailable')
    ids=[value.trace_id,'a'*32] if linked=='trace' else [value.trace_id]
    with pytest.raises(ValueError):MonitorRecords(TraceReader(root)).delete_traces(ids)
    assert sentinel.read_text()=='keep' and (root/value.trace_id).is_dir()


def test_windows_busy_handle_is_hidden_then_purged_on_retry(tmp_path,monkeypatch):
    value=trace(tmp_path);records=MonitorRecords(TraceReader(tmp_path));purge=records._purge
    monkeypatch.setattr(records,'_purge',lambda _:(_ for _ in ()).throw(PermissionError('busy handle')))
    assert records.delete_traces([value.trace_id])['cleanup_pending']
    assert records.reader.list_conversations()==[] and list(tmp_path.glob('.deleted-*'))
    monkeypatch.setattr(records,'_purge',purge);records.purge_pending()
    assert list(tmp_path.iterdir())==[]
