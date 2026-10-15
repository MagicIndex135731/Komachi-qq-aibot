import json
from pathlib import Path
from threading import Thread
import urllib.error
import urllib.request

import pytest

from app.monitor.server import create_server


@pytest.fixture
def server(tmp_path):
    value=create_server(tmp_path,port=0);thread=Thread(target=value.serve_forever,daemon=True);thread.start()
    yield value
    value.shutdown();value.server_close();thread.join()


def request(server,path='/',headers=None,method='GET'):
    req=urllib.request.Request(f'http://127.0.0.1:{server.server_port}{path}',headers=headers or {},method=method)
    return urllib.request.urlopen(req,timeout=3)


def test_local_ui_and_api_are_read_only(server):
    with request(server) as r:
        assert r.status==200 and '小町 · Agent 现场'.encode() in r.read()
        assert "default-src 'self'" in r.headers['Content-Security-Policy']
    with request(server,'/api/traces') as r:assert json.load(r)['traces']==[]
    with pytest.raises(urllib.error.HTTPError) as exc:request(server,'/api/traces',method='POST')
    assert exc.value.code==405


@pytest.mark.parametrize('headers',[{'Host':'evil.example'},{'Origin':'https://evil.example'},{'Sec-Fetch-Site':'cross-site'}])
def test_cross_origin_and_rebinding_are_blocked(server,headers):
    with pytest.raises(urllib.error.HTTPError) as exc:request(server,'/api/traces',headers=headers)
    assert exc.value.code==403


def test_reader_paths_and_no_cache(server):
    with pytest.raises(urllib.error.HTTPError) as exc:request(server,'/api/traces/..%2foutside')
    assert exc.value.code==400
    with request(server,'/health') as r:assert r.headers['Cache-Control']=='no-store'


def test_conversation_api_preserves_context_boundaries(tmp_path):
    from app.agent.runtime_trace import AgentTrace
    for native in ('one', 'one', 'two'):
        trace = AgentTrace(tmp_path, context={'surface': 'group', 'scope_id': 900000001},
            conversation_key='synthetic-group', session_key=native)
        trace.finish('silent', 'synthetic');trace.close()
    value = create_server(tmp_path, port=0)
    thread = Thread(target=value.serve_forever, daemon=True);thread.start()
    try:
        with request(value, '/api/conversations') as r:listing = json.load(r)
        assert listing['read_only'] and len(listing['conversations']) == 1
        c = listing['conversations'][0]
        assert c['identity_quality'] == 'logical' and c['turn_count'] == 3 and c['native_session_count'] == 2
        assert [t['started_at'] for t in c['turns']] == sorted(t['started_at'] for t in c['turns'])
        with request(value, '/api/conversations/' + c['conversation_id']) as r:assert json.load(r) == c
        for path in ('/api/conversations/..%2foutside', '/api/conversations/s-invalid'):
            with pytest.raises(urllib.error.HTTPError) as exc:request(value, path)
            assert exc.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as exc:request(value, '/api/conversations', method='POST')
        assert exc.value.code == 405
        with pytest.raises(urllib.error.HTTPError) as exc:request(value, '/api/conversations', headers={'Host':'evil.example'})
        assert exc.value.code == 403
    finally:
        value.shutdown();value.server_close();thread.join()


def test_deployment_boundary_and_packaged_assets():
    import tomllib
    import yaml
    root=Path(__file__).resolve().parents[1]
    manifest=yaml.safe_load((root/'infra/wsl/docker-compose.monitor.yml').read_text())['services']['xiaomachi-monitor']
    assert 'env_file' not in manifest and manifest['read_only'] is True
    assert manifest['network_mode']=='host'
    assert manifest['volumes']==[{'type':'volume','source':'xiaomachi_data',
        'target':'/workspace/data/agent_monitor','volume':{'subpath':'agent_monitor'}}]
    assert '8765' in manifest['command'] and 'app.monitor.server' in manifest['command']
    packages=tomllib.loads((root/'pyproject.toml').read_text())['tool']['setuptools']['package-data']['app.monitor']
    assert 'static/*.js' in packages
    for name in ('start.sh','install_linux_runtime.sh'):
        script=(root/'infra/wsl/scripts'/name).read_text()
        assert 'AGENT_MONITOR_ENABLED' in script and 'docker-compose.monitor.yml' in script
        assert 'Agent monitor unavailable; chat remains running.' in script
    assert 'docker rm -f xiaomachi-monitor' in (root/'infra/wsl/scripts/stop.sh').read_text()


def test_usage_footer_api_is_local_read_only_and_keeps_unknowns(server,tmp_path):
    from app.agent.runtime_trace import AgentTrace
    trace=AgentTrace(tmp_path,context={'context_window_tokens':1000},session_key='synthetic',conversation_key='synthetic')
    trace.request('m01-a01',{'input':[]},{'context_estimated_tokens':100,'context_window_tokens':1000},purpose='agent')
    trace.finish('silent','done');trace.close()
    with request(server,'/api/conversations') as r:cid=json.load(r)['conversations'][0]['conversation_id']
    route='/api/conversations/'+cid+'/metrics'
    with request(server,route) as r:stats=json.load(r)
    assert stats['read_only'] and stats['context']['source']=='estimated'
    assert stats['context']['ratio']==0.1 and stats['usage']['cache_ratio'] is None
    for headers in ({'Host':'evil.example'},{'Origin':'https://evil.example'}):
        with pytest.raises(urllib.error.HTTPError) as exc:request(server,route,headers=headers)
        assert exc.value.code==403
    with pytest.raises(urllib.error.HTTPError) as exc:request(server,route,method='POST')
    assert exc.value.code==405
    with pytest.raises(urllib.error.HTTPError) as exc:request(server,'/api/conversations/invalid/metrics')
    assert exc.value.code==400


def test_api_links_worker_without_a_fake_model_conversation_or_usage(server,tmp_path):
    from app.agent.runtime_trace import AgentTrace, current_trace_context, observe_with
    parent=AgentTrace(tmp_path,context={'surface':'private','scope_id':900000101,'context_kind':'private'},
        session_key='synthetic-main',conversation_key='synthetic-main')
    with observe_with(parent):lineage=current_trace_context()
    child=AgentTrace(tmp_path,context={**lineage,'surface':'private','scope_id':900000101,
        'context_kind':'media_task','execution_kind':'background_tool','job_id':1,'operation':'download_web_video'},
        session_key='synthetic-worker',conversation_key='synthetic-worker')
    child.event('job.phase.started',{'phase':'running','job_id':1})
    child.finish('ok','synthetic');child.close();parent.finish('silent','synthetic');parent.close()
    with request(server,'/api/conversations') as r:conversations=json.load(r)['conversations']
    assert len(conversations)==1 and conversations[0]['task_count']==1
    c=conversations[0]
    assert len(c['turns'])==1 and c['task_traces'][0]['trace_id']==child.trace_id
    with request(server,'/api/conversations/'+c['conversation_id']+'/metrics') as r:
        assert json.load(r)['usage']['request_count']==0
    with request(server,'/api/traces/'+child.trace_id) as r:
        assert any(e['kind']=='job.phase.started' for e in json.load(r)['events'])
    with request(server,'/api/controls') as r:token=json.load(r)['deletion_token']
    with request(server,'/api/conversations/'+c['conversation_id'],headers={'X-Monitor-Token':token},method='DELETE') as r:
        assert json.load(r)['deleted_turns']==2
    with pytest.raises(urllib.error.HTTPError) as exc:request(server,'/api/traces/'+child.trace_id)
    assert exc.value.code==404
