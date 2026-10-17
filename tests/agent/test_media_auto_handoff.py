from app.agent.tool_agent import ToolOutcome
from tests.agent.test_harness_runtime import ModelHandler, bridge_factory, responses_sse, tool


async def test_auto_download_handoff_skips_stale_progress_and_next_upstream_request(bridge_factory):
    bridge,requests,_=bridge_factory(lambda *_:responses_sse(),require_reply=True)
    sent=[]
    async def queue(args):
        return ToolOutcome({'job_id':42,'status':'queued','execution_status':'queued',
            'delivery':'automatic_delivery_to_original_request'})
    async def send(args):sent.append(args);return ToolOutcome({},'sent','unexpected')
    bridge.tools={
        'download_twitter_media':tool('download_twitter_media',queue,properties={'delivery_mode':{'type':'string'}}),
        'send_private':tool('send_private',send,effect='write',properties={'text':{'type':'string'},'task_status':{'type':'string'}}),
    }
    accepted=await bridge.call_tool({'name':'download_twitter_media','arguments':{'delivery_mode':'auto'},'call_id':'download'})
    assert accepted['status']=='ok' and bridge.auto_handoff
    assert not bridge.can_recover_delivery() and not bridge.can_recover_task()
    await bridge.call_tool({'name':'send_private','arguments':{'text':'等下发给你','task_status':'progress'},'call_id':'late-progress'})
    handler=ModelHandler();bridge.forward(handler,{'model':bridge.model,'input':[]})
    assert not requests and not sent
    assert b'handed off' in handler.wfile.getvalue()
    result=bridge.result('model_finished')
    assert result.status=='silent' and result.reason=='background_auto_handoff'
    assert result.task_status['status']=='waiting'


async def test_verified_review_return_hands_new_download_to_its_next_result(bridge_factory):
    bridge,requests,_=bridge_factory(lambda *_:responses_sse(),require_reply=True)
    bridge.owner.initial_evidence=({'kind':'media_result','status':'failed','data':{'error':'synthetic_404'}},)
    async def queue(args):
        return ToolOutcome({'job_id':43,'status':'queued','execution_status':'queued',
            'delivery':'background_media_result_returns_to_main'})
    bridge.tools={'download_twitter_media':tool('download_twitter_media',queue)}
    result=await bridge.call_tool({'name':'download_twitter_media','arguments':{},'call_id':'replacement'})
    assert result['status']=='ok' and bridge.auto_handoff
    handler=ModelHandler();bridge.forward(handler,{'model':bridge.model,'input':[]})
    assert not requests


async def test_review_download_keeps_main_conversation_in_charge(bridge_factory):
    bridge,_,_=bridge_factory(lambda *_:responses_sse(),require_reply=True)
    async def queue(args):return ToolOutcome({'job_id':42,'status':'queued',
        'delivery':'background_media_result_returns_to_main'})
    bridge.tools={'download_twitter_media':tool('download_twitter_media',queue)}
    await bridge.call_tool({'name':'download_twitter_media','arguments':{},'call_id':'review'})
    assert not bridge.auto_handoff


async def test_partial_batch_does_not_abandon_unaccepted_work(bridge_factory):
    bridge,_,_=bridge_factory(lambda *_:responses_sse(),require_reply=True)
    async def queue(args):return ToolOutcome({'jobs':[{'job_id':42,'status':'queued',
        'delivery':'automatic_delivery_to_original_request'}],
        'all_accepted':False,'failures':[{'index':1,'error':'capacity'}]})
    bridge.tools={'queue_media_batch':tool('queue_media_batch',queue)}
    await bridge.call_tool({'name':'queue_media_batch','arguments':{},'call_id':'batch'})
    assert not bridge.auto_handoff


async def test_native_parallel_auto_downloads_are_all_accepted_before_local_close(bridge_factory):
    bridge,_,_=bridge_factory(lambda *_:responses_sse(),require_reply=True)
    calls=[]
    async def queue(args):
        calls.append(args)
        return ToolOutcome({'job_id':40+len(calls),'status':'queued',
            'delivery':'automatic_delivery_to_original_request'})
    bridge.tools={'download_twitter_media':tool('download_twitter_media',queue)}
    for n in range(2):
        accepted=await bridge.call_tool({'name':'download_twitter_media','arguments':{},'call_id':str(n)})
        assert accepted['status']=='ok'
    assert len(calls)==2 and bridge.auto_handoff and len(bridge.pending_jobs)==2


async def test_parallel_auto_failure_or_review_preserves_main_followup(bridge_factory):
    bridge,_,_=bridge_factory(lambda *_:responses_sse(),require_reply=True)
    calls=[]
    async def queue(args):
        calls.append(args)
        return (ToolOutcome({'job_id':41,'status':'queued',
            'delivery':'automatic_delivery_to_original_request'}) if len(calls)==1
            else ToolOutcome({'error':'capacity'},'blocked'))
    bridge.tools={'download_twitter_media':tool('download_twitter_media',queue)}
    for n in range(2):await bridge.call_tool({'name':'download_twitter_media','arguments':{},'call_id':str(n)})
    assert len(calls)==2 and not bridge.auto_handoff


async def test_empty_media_discoveries_trigger_one_canonical_intent_review(bridge_factory,tmp_path):
    from app.agent.research_tasks import ResearchTaskWorkspace
    from app.agent.request_task import RequestTaskControl
    bridge,_,_=bridge_factory(lambda *_:responses_sse(),require_reply=True)
    request='帮我扒一些关于某人的cos图片，之前那张不要重复'
    workspace=ResearchTaskWorkspace(tmp_path,scope='private:900000101',actor_id=900000101,
        source_id=10,request=request,valid_source=lambda _:True)
    bridge.control=RequestTaskControl(workspace,capability_version='synthetic')
    reviews=[]
    async def review(control,proposed):
        reviews.append(control.workspace.request)
        return {'gap':'应排除同一素材，不是整个造型','next_tool':'search_x_posts'}
    bridge.review_task=review
    async def empty(args):return ToolOutcome({'resources':[],'content_kind':'search_posts'})
    bridge.tools={'search_x_posts':tool('search_x_posts',empty)}
    for n in range(3):await bridge.call_tool({'name':'search_x_posts','arguments':{},'call_id':str(n)})
    assert reviews==[request] and bridge.control.review_used
    assert bridge.control.last_delivery is None and not bridge.auto_handoff
