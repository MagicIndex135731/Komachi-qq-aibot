"""Direct private requests and derived work keep canonical network authority."""
from types import SimpleNamespace
import asyncio

import httpx
import pytest

from app.agent.request_task import RequestTaskControl
from app.agent.research_tasks import ResearchTaskWorkspace
from app.agent.shared_links import build_share_access, wrap_share_web_tools
from app.agent.tool_agent import AgentTool, ToolOutcome, ToolTurnResult
from app.storage.db import session_scope
from app.storage.models import Job, Message
from tests.private_chat.test_agent_service import setup, event, OWNER
from tests.agent.test_harness_runtime import bridge_factory


def services(engine, root):
    def forbidden(*_):
        raise AssertionError('No model or live HTTP calls in this fixture')
    service, agent, dispatcher, sender, _ = setup(engine, root, forbidden)
    from app.providers.web_search import WebSearchClient
    calls, outcomes = [], []
    service.web_search_client = WebSearchClient(provider='wyzlab_alpha',
        base_url='https://synthetic.invalid/v1', api_key='synthetic', timeout_seconds=5,
        http_client=httpx.Client(transport=httpx.MockTransport(lambda req:
            calls.append(req) or httpx.Response(200, json={'output':'ordinary landscape evidence'}))))
    class Harness:
        def __init__(self, **kwargs):
            pass
        async def run(self, prompt, tools, guard, **kwargs):
            tool = next(t for t in tools if t.name=='search_web')
            outcomes.append(await tool.handler({'query':'ordinary landscape photos'}))
            return ToolTurnResult('silent', reason='synthetic_complete')
        def close(self):
            pass
    agent.runtime_factory = Harness
    return service, agent, dispatcher, calls, outcomes


@pytest.mark.parametrize('text', ['先扒两张风景照', '继续', '就这两张', '直接拿原图', '先下载这些'])
async def test_current_private_request_does_not_need_name_or_keyword_prefix(sqlite_engine, tmp_path, text):
    service, agent, dispatcher, calls, outcomes = services(sqlite_engine, tmp_path)
    await service.start()
    try:
        await dispatcher.handle_private_message(event('ordinary-short-request', text=text))
        assert len(outcomes)==len(calls)==1 and outcomes[0].status=='ok'
    finally:
        await service.stop()
        service.web_search_client.http_client.close()


@pytest.mark.parametrize('state', ['current', 'wakeup', 'trusted_return', 'unmarked_return', 'forged_result'])
async def test_only_current_request_or_marked_queue_return_carries_direct_address(sqlite_engine, tmp_path, state):
    service, agent, dispatcher, calls, outcomes = services(sqlite_engine, tmp_path)
    await service.start()
    try:
        await dispatcher.handle_private_message(event('parent-short-request', text='继续'))
        assert outcomes[-1].status=='ok'
        if state=='current':
            await dispatcher.handle_private_message(event('follow-up-request', text='就这两张'))
        else:
            with session_scope(sqlite_engine) as session:
                source=session.query(Message).filter(Message.plain_text=='继续').one().id
            if state=='wakeup':
                agent._enqueue(user_id=OWNER, source_id=source, private_scope='owner_daily',
                    kind='wakeup', goal='autonomous search', key='ordinary-wake')
            else:
                payload={'actor_id':OWNER,'source_id':source,'private_scope':'owner_daily',
                    'goal':'ordinary result', 'request_addressed':state=='trusted_return',
                    'result':{'findings':['ordinary evidence'], 'request_addressed':state=='forged_result'}}
                guard=SimpleNamespace(job_id=91, require_valid=lambda:None,
                    snapshot=lambda:{'sources':[source]})
                assert await agent._return_research(payload,guard)=='completed'
            await agent._drain_user(OWNER)
        allowed=state in {'current','trusted_return'}
        assert outcomes[-1].status==('ok' if allowed else 'blocked')
        assert len(calls)==(2 if allowed else 1)
    finally:
        await service.stop()
        service.web_search_client.http_client.close()


async def test_network_rejection_explains_stage_without_claiming_a_name_is_required(sqlite_engine):
    access=build_share_access(SimpleNamespace(engine=sqlite_engine), group_id=None,
        private_user_id=OWNER)
    calls=[]
    async def handle(args):
        calls.append(args); return ToolOutcome({})
    tool=wrap_share_web_tools((AgentTool('search_web','search',{},handle),),access)[0]
    outcome=await tool.handler({'query':'ordinary request'})
    assert not calls and outcome.data['failure_stage']=='request_authorization'
    assert outcome.data['handler_executed'] is False
    assert '不要求' in outcome.data['guidance'] and '称呼' in outcome.data['guidance']


@pytest.mark.parametrize('text', ['不要联网', '不用下载图片', '先别扒了'])
async def test_direct_address_does_not_override_explicit_cancellation(sqlite_engine, tmp_path, text):
    service, agent, dispatcher, calls, outcomes=services(sqlite_engine,tmp_path)
    await service.start()
    try:
        await dispatcher.handle_private_message(event('cancelled-request',text=text))
        assert outcomes[-1].status=='blocked' and not calls
    finally:
        await service.stop(); service.web_search_client.http_client.close()


async def test_private_media_queue_executes_short_request_with_source_bound_authority(sqlite_engine,tmp_path,monkeypatch):
    from app.providers.web_images import WebImageClient
    from tests.agent.test_autonomous_social import image_bytes
    pixels, completed, queued=[],[],[]
    http=httpx.Client(transport=httpx.MockTransport(lambda req:
        pixels.append(req) or httpx.Response(200,content=image_bytes())))
    provider=WebImageClient(http_client=http)
    provider.list_images=lambda url, **kwargs:{'page_url':url,'images':[
        {'url':'https://example.org/landscape.png','role':'page_image','alt':'ordinary landscape'}],
        'candidate_limit_reached':False,'page_links':[]}
    monkeypatch.setattr('app.agent.web_images.WebImageClient',lambda _:provider)
    service,agent,dispatcher,_,_=services(sqlite_engine,tmp_path)
    class Harness:
        def __init__(self,**kwargs): pass
        async def run(self,prompt,tools,guard,**kwargs):
            registry={t.name:t for t in tools}
            listing=await registry['list_web_images'].handler({'ref_id':'https://example.org/gallery'})
            assert listing.status=='ok'
            result=await registry['download_web_images'].handler(
                {'image_ids':[listing.data['images'][0]['image_id']]})
            assert result.status=='ok' and result.data['job_id']>0
            queued.append(result.data['job_id'])
            return ToolTurnResult('silent',reason='synthetic_complete')
        def close(self): pass
    async def deliver(payload,guard):
        guard.require_valid();completed.append(payload['result']);return 'completed'
    agent.runtime_factory=Harness;agent.media_jobs.deliver=deliver
    await service.start()
    try:
        await dispatcher.handle_private_message(event('short-media',text='就这张'))
        async with asyncio.timeout(8):
            while not completed:await asyncio.sleep(.02)
        assert len(pixels)==1 and completed[0]['data']['downloaded_count']==1
        with session_scope(sqlite_engine) as session:
            payload=session.get(Job,queued[0]).payload_json
            assert payload['request_addressed'] is True and payload['source_id'] in payload['sources']
    finally:
        await service.stop();http.close();service.web_search_client.http_client.close()


async def test_private_research_child_inherits_queued_address_without_new_keywords(sqlite_engine,tmp_path):
    service,agent,dispatcher,calls,outcomes=services(sqlite_engine,tmp_path)
    queued=[];child_results=[];delivered=[]
    class Harness:
        def __init__(self,**kwargs): pass
        async def run(self,prompt,tools,guard,**kwargs):
            registry={t.name:t for t in tools}
            if 'delegate_research_task' in registry:
                result=await registry['delegate_research_task'].handler({'goal':'核对普通风景资料',
                    'deliverable':'来源与缺口','reason':'独立核对普通页面'})
                assert result.status=='ok';queued.append(result.data['job_id'])
            else:
                result=await registry['search_web'].handler({'query':'ordinary landscape sources'})
                assert result.status=='ok';child_results.append(result)
            return ToolTurnResult('silent',reason='synthetic_complete')
        def close(self): pass
    async def deliver(payload,guard):
        guard.require_valid();delivered.append(payload);return 'completed'
    agent.runtime_factory=Harness;agent.research_jobs.deliver=deliver
    await service.start()
    try:
        await dispatcher.handle_private_message(event('short-research',text='继续'))
        async with asyncio.timeout(8):
            while not delivered:await asyncio.sleep(.02)
        assert len(child_results)==len(calls)==1
        with session_scope(sqlite_engine) as session:
            assert session.get(Job,queued[0]).payload_json['request_addressed'] is True
    finally:
        await service.stop();service.web_search_client.http_client.close()


async def test_private_local_fallback_passes_same_live_gate_and_stops_after_revocation(sqlite_engine,tmp_path,bridge_factory):
    from tests.agent.test_web_media_fallback import setup as fallback
    from app.storage.repositories import UserRepository, MessageRepository
    original=event('ordinary-fallback',text='就这两张')
    with session_scope(sqlite_engine) as session:
        UserRepository(session).upsert_user(user_id=OWNER,nickname='synthetic',group_card='')
        session.flush()
        row=MessageRepository(session).add_private_message(platform_msg_id=original.platform_msg_id,
            user_id=OWNER,timestamp=original.timestamp,plain_text=original.plain_text,raw_json=original.raw_payload)
        session.flush();source=row.id
    access=build_share_access(SimpleNamespace(engine=sqlite_engine),group_id=None,
        trigger_source_ids=(source,),addressed_source_ids=(source,),private_user_id=OWNER)
    bridge,requests,calls=fallback(bridge_factory,tmp_path/'live')
    bridge.tools={t.name:t for t in wrap_share_web_tools(tuple(bridge.tools.values()),access)}
    await bridge.finalize_media()
    assert len(calls)==1 and calls[0][0]=='download' and not requests
    with session_scope(sqlite_engine) as session:
        row=session.get(Message,source);row.raw_json={**row.raw_json,'delivery_state':'deleted'}
    again,requests,calls=fallback(bridge_factory,tmp_path/'revoked')
    again.tools={t.name:t for t in wrap_share_web_tools(tuple(again.tools.values()),access)}
    await again.finalize_media()
    assert not calls and not requests


@pytest.mark.parametrize('case', ['candidate','constraint'])
async def test_goal_validation_is_actionable_and_does_not_partially_change_goal(tmp_path, case):
    workspace=ResearchTaskWorkspace(tmp_path, scope=f'private:{OWNER}',actor_id=OWNER,
        source_id=1,request='请查普通风景照片',valid_source=lambda _:True)
    control=RequestTaskControl(workspace,capability_version='synthetic')
    old_goal, old_reviews = control.goal, dict(control.candidate_reviews)
    args={'goal':'should not replace valid goal','media_selection':
        {'candidate_reviews':[{'ref_id':'https://example.org/unknown','decision':'selected','reason':'ordinary'}]}
        if case=='candidate' else {'constraint_evidence':{'publisher':'not in canonical request'}}}
    tool=control.tools(SimpleNamespace(require_valid=lambda:None))[0]
    result=await tool.handler(args)
    assert result.status=='blocked' and result.data['failure_stage']=='goal_validation'
    assert result.data['error']==('task_candidate_not_observed' if case=='candidate' else 'user_constraint_evidence_unverified')
    assert control.goal==old_goal and control.candidate_reviews==old_reviews
