"""Real native Harness loop, scripted upstream; no live model/media/QQ calls."""
from types import SimpleNamespace

import pytest

from app.agent.harness_runtime import HarnessToolAgent
from app.agent.research_tasks import ResearchTaskWorkspace
from app.agent.tool_agent import ToolOutcome
from app.agent.web_tools import build_web_tools
from app.providers.web_search import WebSearchReport, WebSearchResult
from test_harness_runtime import Guard, model_client, responses_sse, tool


@pytest.mark.parametrize('model', ['gpt-synthetic', 'gemini-synthetic'])
async def test_native_search_batch_inspection_download_delivery_has_no_extra_model_step(tmp_path, monkeypatch, model):
    url = 'https://x.com/synthetic/status/900000000000000001'
    workspace = ResearchTaskWorkspace(tmp_path / 'evidence', scope='private:900000101',
        actor_id=900000101, source_id=10, request='搜索这个作者的cos图片发给我', valid_source=lambda x: x == 10)
    calls, sent = [], []
    class Web:
        supports_page_tools = True
        def search_report(self, query, max_results):
            calls.append('search')
            return WebSearchReport([WebSearchResult('作品', '作者原帖', url, '')], '来源摘要')
    class Runner:
        async def run(self, fn, *args, **kwargs):return fn(*args, **kwargs)
    def inspect(self, source, **kwargs):
        calls.append('inspect');kwargs['check']()
        return {'source_url': source, 'post_id': '900000000000000001', 'title': '作品',
            'author': 'synthetic', 'published_at': '2026-01-01', 'identity_status': 'matched',
            'media_count': 1, 'text': '公开cos作品',
            'media': [{'kind': 'image', 'role': 'post_image', 'url': 'https://pbs.twimg.com/media/synthetic.jpg'}]}
    monkeypatch.setattr('app.agent.resource_tools.PublicResourceClient.inspect', inspect)
    def script(body, n):
        if n == 1:return responses_sse(name='search_web', arguments={'query': 'synthetic'}, number=n, call_id='c1')
        if n == 2:return responses_sse(name='inspect_web_resources', arguments={'ref_ids': [url]}, number=n, call_id='c2')
        if n == 3:
            assert 'post_image' in str(body)
            return responses_sse(name='download_twitter_media', arguments={'ref_id': url}, number=n, call_id='c3')
        return responses_sse(name='send_private', arguments={'text': '原帖图片', 'artifact_ids': ['art-synthetic']}, number=n, call_id='c4')
    client, requests, _ = model_client(script)
    client.responses_model = model
    async def download(args):
        calls.append('download');return ToolOutcome({'artifact_ids': ['art-synthetic'], 'all_requested_downloaded': True})
    async def send(args):
        sent.append(args);return ToolOutcome({}, 'sent', 'synthetic-receipt')
    tools = (*build_web_tools(Web(), settings=SimpleNamespace(), runner=Runner(), guard=Guard(), workspace=workspace),
        tool('download_twitter_media', download, properties={'ref_id': {'type': 'string'}}),
        tool('send_private', send, effect='write', properties={'text': {'type': 'string'},
            'artifact_ids': {'type': 'array', 'items': {'type': 'string'}}}, required=('text',)))
    agent = HarnessToolAgent(decision_client=client, task_workspace=workspace, require_reply=True,
        reply_source_id=10, max_external_writes=1, session_home=tmp_path/'native')
    try:
        result = await agent.run(['Synthetic resource task'], tools, Guard())
        assert result.status == 'sent' and len(sent) == 1
        assert calls == ['search', 'inspect', 'download'] and len(requests) == 4
        assert all(not item.get('images') for item in requests)
    finally:client.http_client.close()


@pytest.mark.parametrize('model',['gpt-synthetic','gemini-synthetic'])
async def test_native_account_to_auto_queue_ends_without_progress_model_call(tmp_path,monkeypatch,model):
    from datetime import UTC,datetime
    now=datetime(2026,10,16,tzinfo=UTC)
    url='https://x.com/synthetic/status/900000000000000001'
    workspace=ResearchTaskWorkspace(tmp_path/'evidence',scope='private:900000101',
        actor_id=900000101,source_id=10,request='把synthetic最新的cos图片下载发我',
        valid_source=lambda i:i==10,clock=lambda:now.timestamp())
    def discover(self,account,**kwargs):
        return {'account':account,'posts':[{'source_url':url,'post_id':'900000000000000001',
            'author':'synthetic','published_at':'2026-10-15T08:00:00Z','identity_status':'matched',
            'date_status':'source_reported','media_count':1,'media_roles':['post_image'],'text':'作品'}],
            'status':'ok','source_url':'https://syndication.twitter.com/srv/timeline-profile/screen-name/synthetic',
            'complete':False,'latest_proven':False,'coverage':'public_visible_only'}
    monkeypatch.setattr('app.agent.twitter_accounts.TwitterTimelineClient.list_posts',discover)
    def script(body,n):
        assert n<=2,'Auto handoff must not ask the model to generate a stale queue notice'
        return (responses_sse(name='list_x_posts',arguments={'account':'synthetic'},number=n,call_id='list') if n==1
            else responses_sse(name='download_twitter_media',arguments={'ref_id':url,'delivery_mode':'auto'},number=n,call_id='queue'))
    client,requests,_=model_client(script);client.responses_model=model
    queued,sent=[],[]
    async def download(args):
        queued.append(args)
        return ToolOutcome({'status':'queued','execution_status':'pending','job_id':42,
            'delivery':'automatic_delivery_to_original_request'})
    async def send(args):sent.append(args);return ToolOutcome({},'sent','unexpected')
    tools=(*build_web_tools(SimpleNamespace(supports_page_tools=True),settings=SimpleNamespace(),
        runner=RunnerForAccount(),guard=Guard(),workspace=workspace),
        tool('download_twitter_media',download,properties={'ref_id':{'type':'string'},'delivery_mode':{'type':'string'}}),
        tool('send_private',send,effect='write',properties={'text':{'type':'string'}}))
    agent=HarnessToolAgent(decision_client=client,task_workspace=workspace,require_reply=True,
        reply_source_id=10,session_home=tmp_path/'native')
    try:
        result=await agent.run(['Synthetic current account request'],tools,Guard())
        assert result.status=='silent' and result.reason=='background_auto_handoff'
        assert len(requests)==2 and len(queued)==1 and not sent
        assert result.task_status['status']=='waiting'
    finally:client.http_client.close()


class RunnerForAccount:
    async def run(self,fn,*args,**kwargs):return fn(*args,**kwargs)
