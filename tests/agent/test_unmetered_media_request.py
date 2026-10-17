"""Cross the former quotas and keep partial delivery independent of the model."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from app.agent.media_request_policy import await_request_work, bind_request_workspace
from app.agent.request_budget import RequestBudget
from app.agent.request_task import RequestTaskControl
from app.agent.tool_agent import ToolOutcome
from tests.agent.test_harness_runtime import bridge_factory, ModelHandler, request_body, responses_sse, tool
from tests.agent.test_request_budget import workspace


def test_default_usage_survives_old_caps_deadline_and_restart(tmp_path):
    now=[1000]
    w=workspace(tmp_path,clock=lambda:now[0]);budget=RequestBudget(w)
    for key,amount in {'model_calls':100,'estimated_tokens':2_560_000,'tool_calls':1000,
            'discovery_calls':100,'transfer_requests':1000,'writes':100}.items():budget.spend(key,amount)
    budget.charge(1024**3)
    now[0]+=3600
    resumed=RequestBudget(w)
    resumed.check();resumed.spend('writes',finalization=True)
    state=resumed.snapshot()
    assert state['used']['writes']==101 and state['used']['media_received']==1024**3
    assert state['expires_at'] is None and all(v is None for v in state['limits'].values())
    assert resumed.remaining==float('inf') and resumed.deadline==float('inf')
    assert 'Infinity' not in resumed.path.read_text()
    w.valid_source=lambda _:False
    with pytest.raises(ValueError,match='task_source_unavailable'):resumed.check()


def test_legacy_implicit_quota_upgrades_without_resetting_usage(tmp_path):
    w=workspace(tmp_path);b=RequestBudget(w)
    state=b.snapshot();state.pop('policy');state.update(expires_at=900,limits=b.LIMITS,used={'writes':8})
    w._write(b.path,state)
    resumed=RequestBudget(w,create=False)
    resumed.spend('writes')
    assert resumed.unmetered and resumed.snapshot()['used']['writes']==9


def test_large_candidate_checkpoint_keeps_every_source_across_wakes(tmp_path):
    w=workspace(tmp_path);control=RequestTaskControl(w,capability_version='synthetic')
    urls={f'https://x.com/synthetic/status/{900000000000000000+n}' for n in range(600)}
    control.resource_urls.update(urls)
    control.candidate_reviews={control._resource_key(url):{'decision':'selected','reason':'合成来源说明'*100} for url in urls}
    control._persist()
    assert w._file('media-goal.json').stat().st_size>256*1024
    resumed=RequestTaskControl(workspace(tmp_path),capability_version='synthetic',initial_result={'kind':'media_result','status':'failed','data':{}})
    assert set(resumed.fallback_sources())==urls
    assert len(resumed.candidate_reviews)==600


def test_media_model_usage_counts_without_daily_cutoff_but_images_stay_finite(sqlite_engine):
    from app.storage.autonomous_agent import AutonomousRepository
    repository=AutonomousRepository(sqlite_engine,account_id=900000001)
    for _ in range(502):assert repository.reserve_cost('model:group_interactive',None)
    assert not repository.cost_available('model:group_interactive',500)
    assert repository.cost_available('model:group_interactive',None)
    with pytest.raises(ValueError):repository.reserve_cost('image',None)


async def test_late_media_intent_passes_both_outer_timers_but_cancel_still_stops(tmp_path):
    w=workspace(tmp_path);leaf=SimpleNamespace();root=SimpleNamespace()
    leaf.dispatch_parent=root;bind_request_workspace(leaf,w)
    async def work():
        w.media_budget=RequestBudget(w)
        await asyncio.sleep(.04)
        return 'done'
    assert await await_request_work(await_request_work(work(),timeout=.01,guard=leaf),
        timeout=.02,guard=root)=='done'
    stopped=asyncio.Event()
    async def endless():
        try:await asyncio.Event().wait()
        finally:stopped.set()
    task=asyncio.create_task(await_request_work(endless(),timeout=.01,workspace=w))
    await asyncio.sleep(.02);task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    assert stopped.is_set()
    with pytest.raises(TimeoutError):await await_request_work(asyncio.sleep(.1),timeout=.001)


def test_media_model_can_pass_32_rounds_and_compactions(bridge_factory,tmp_path):
    bridge,requests,_=bridge_factory(lambda *_:responses_sse(),require_reply=True)
    bridge.owner.task_workspace=w=workspace(tmp_path)
    bridge.control=RequestTaskControl(w,capability_version='synthetic')
    bridge.owner.max_steps=1
    for _ in range(35):
        handler=ModelHandler();bridge.forward(handler,request_body())
        assert not handler.replies,handler.replies
    for _ in range(3):bridge.forward(ModelHandler(),request_body(),compaction=True)
    assert len(requests)==38 and bridge.model_calls==35 and not bridge.stop_reason
    assert w.media_budget.snapshot()['used']['model_calls']==38


@pytest.mark.parametrize('stop',['model_provider_failed','model_request_timeout'])
async def test_finalizer_passes_eight_writes_and_sends_all_source_packets(bridge_factory,tmp_path,stop):
    bridge,requests,_=bridge_factory(lambda *_:responses_sse(),require_reply=True)
    bridge.owner.task_workspace=w=workspace(tmp_path)
    bridge.control=RequestTaskControl(w,capability_version='synthetic')
    for n in range(181):
        url=f'https://x.com/synthetic/status/{900000000000000000+n}'
        bridge.control.observe('download_web_image',{'image_id':f'img-{n}'},
            ToolOutcome({'artifact_id':f'art-{n}','page_url':url,'media':[
                {'artifact_id':f'art-{n}','source_ref':url,'kind':'photo','bytes':100}]}),f'get-{n}')
    sent=[]
    async def send(args):sent.append(args);return ToolOutcome({},'sent',f'receipt-{len(sent)}')
    bridge.tools={'send_private':tool('send_private',send,effect='write',properties={
        'text':{'type':'string','maxLength':300},'artifact_ids':{'type':'array','items':{'type':'string'}},
        'media_artifact_ids':{'type':'array','items':{'type':'string'},'maxItems':20},
        'task_status':{'type':'string'},'as_forward':{'type':'boolean'}})}
    bridge.stop_reason=stop;bridge.owner.max_external_writes=1
    await bridge.finalize_media()
    assert [a for packet in sent for a in packet.get('media_artifact_ids',[]) ]==[f'art-{n}' for n in range(181)]
    assert len(sent)>10 and not requests
    text='\n'.join(packet['text'] for packet in sent)
    assert all(url in text for url in bridge.control.fallback_sources())
    assert all(packet['task_status']=='partial' and len(packet['text'])<=300 for packet in sent)


async def test_web_media_continues_after_four_pages_twenty_downloads_and_32mib(sqlite_engine,tmp_path,monkeypatch):
    from tests.agent.test_web_image_tools import tools_for
    from test_autonomous_social import image_bytes
    body=image_bytes()+b'\0'*(2*1024*1024)
    def respond(request):
        return httpx.Response(200,content=body) if request.url.path.endswith('.png') else httpx.Response(
            200,headers={'Content-Type':'text/html'},text=''.join(f'<img src="/{n}.png">' for n in range(25)))
    tools,store,guard,client=tools_for(monkeypatch,sqlite_engine,tmp_path,respond)
    w=workspace(tmp_path/'request');w.media_budget=RequestBudget(w);bind_request_workspace(guard,w)
    try:
        for n in range(5):
            result=await tools['list_web_images'].handler({'ref_id':f'https://example.com/page-{n}'})
            assert result.status=='ok'
        first=result.data['images']
        rest=await tools['list_web_images'].handler({'ref_id':'https://example.com/page-4','offset':20})
        artifacts=[]
        for image in first+rest.data['images'][:1]:
            result=await tools['download_web_image'].handler({'image_id':image['image_id']})
            assert result.status=='ok',result
            artifacts.append(result.data['artifact_id'])
        assert len(set(artifacts))==21 and all(store.get(i) is not None for i in artifacts)
    finally:client.close()
