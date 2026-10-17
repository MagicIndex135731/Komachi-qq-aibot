"""Budget stops keep useful source-bound results without inventing selection."""
import pytest

from app.agent.request_budget import source_receipt_scope, source_receipt_urls_allowed
from app.agent.request_task import RequestTaskControl
from app.agent.tool_agent import ToolOutcome
from tests.agent.test_harness_runtime import bridge_factory, tool, responses_sse
from tests.agent.test_request_budget import workspace


URL='https://example.com/ordinary-post'


def prepare(factory,tmp_path,surface):
    bridge,requests,_=factory(lambda *_:responses_sse(),require_reply=True)
    w=workspace(tmp_path,scope=surface+':900000101');bridge.owner.task_workspace=w
    bridge.control=RequestTaskControl(w,capability_version='synthetic')
    bridge.control.observe('inspect_web_resources',{'ref_ids':[URL]},ToolOutcome({'resources':[
        {'source_url':URL,'input_index':1,'post_id':'synthetic-post','identity_status':'matched',
         'media_count':2,'media_roles':['post_image']}]}),'inspect')
    bridge.stop_reason='request_budget_exhausted'
    sent=[]
    async def send(args):
        sent.append(args)
        assert source_receipt_urls_allowed(w.scope,w.source_id,args['text'])
        return ToolOutcome({},'sent','synthetic-receipt')
    bridge.tools={'send_'+surface:tool('send_'+surface,send,effect='write',properties={
        'text':{'type':'string','maxLength':300},'artifact_ids':{'type':'array','items':{'type':'string'}},
        'as_forward':{'type':'boolean'},'media_artifact_ids':{'type':'array','maxItems':20,'items':{'type':'string'}},
        'task_status':{'type':'string'}},required=('text','artifact_ids'))}
    return bridge,requests,sent


@pytest.mark.parametrize('surface',['group','private'])
@pytest.mark.parametrize('ready',[False,True])
async def test_budget_stop_sends_existing_unreviewed_files_or_observed_posts(bridge_factory,tmp_path,surface,ready):
    bridge,requests,sent=prepare(bridge_factory,tmp_path,surface)
    if ready:
        for i in range(2):
            bridge.control.observe('download_web_image',{'image_id':f'webimg-{i}'},ToolOutcome({
                'artifact_id':f'art-{i}','page_url':URL,'media':[{'artifact_id':f'art-{i}',
                    'kind':'image','source_ref':URL,'bytes':100}]}),'download-'+str(i))
    await bridge.finalize_media()
    assert len(sent)==1 and not requests
    assert sent[0]['task_status']=='partial' and '额度' in sent[0]['text'] and URL in sent[0]['text']
    if ready:
        assert '尚未逐项核实' in sent[0]['text'] and len(sent[0]['media_artifact_ids'])==2
    else:assert sent[0]['artifact_ids']==[] and '媒体尚未完成' in sent[0]['text']
    assert bridge.control.stage=='partial'
    await bridge.finalize_media();assert len(sent)==1


@pytest.mark.parametrize('blocked',['revoked','unknown','cancelled','excluded'])
async def test_partial_fallback_never_bypasses_revocation_unknown_or_exclusion(bridge_factory,tmp_path,blocked):
    bridge,_,sent=prepare(bridge_factory,tmp_path,'private')
    bridge.control.observe('download_web_image',{'image_id':'webimg-a'},ToolOutcome({
        'artifact_id':'art-a','page_url':URL}),'download')
    if blocked=='revoked':bridge.guard.require_valid=lambda:(_ for _ in ()).throw(RuntimeError('revoked'))
    elif blocked=='unknown':bridge.control.unknown=True
    elif blocked=='cancelled':bridge.cancelled.set()
    else:bridge.control.candidate_reviews[bridge.control._resource_key(URL)]={'decision':'excluded'}
    await bridge.finalize_media();assert not sent


def test_source_receipt_capability_is_scoped_and_cannot_be_forged_by_text(tmp_path):
    w=workspace(tmp_path)
    assert not source_receipt_urls_allowed(w.scope,w.source_id,URL)
    with source_receipt_scope(w,[URL]):
        assert source_receipt_urls_allowed(w.scope,w.source_id,URL)
        assert not source_receipt_urls_allowed('group:900000002',w.source_id,URL)
        assert not source_receipt_urls_allowed(w.scope,w.source_id+1,URL)
        assert not source_receipt_urls_allowed(w.scope,w.source_id,URL+' https://other.example.com')
    assert not source_receipt_urls_allowed(w.scope,w.source_id,URL)


def test_parallel_requests_keep_their_own_pending_media_selections(tmp_path):
    first=workspace(tmp_path);a=RequestTaskControl(first,capability_version='synthetic')
    a.describe({'goal':'第一组旅行图片','delivery':'media','media_selection':{'subject':'甲'}})
    second=workspace(tmp_path);second.source_id=11;second.task_id='synthetic-second'
    b=RequestTaskControl(second,capability_version='synthetic')
    b.describe({'goal':'第二组旅行图片','delivery':'media','media_selection':{'subject':'乙'}})
    assert first.media_selection_for_source()['media_selection']['subject']=='甲'
    assert second.media_selection_for_source()['media_selection']['subject']=='乙'


@pytest.mark.parametrize('mixed',[False,True])
async def test_budget_message_keeps_approved_and_unreviewed_ready_together(bridge_factory,tmp_path,mixed):
    bridge,_,sent=prepare(bridge_factory,tmp_path,'private')
    bridge.control.candidate_reviews[bridge.control._resource_key(URL)]={'decision':'selected'}
    for i in range(3 if mixed else 1):
        source=URL if i==0 else URL+'-'+str(i)
        bridge.control.observe('download_web_image',{'image_id':'webimg-'+str(i)},ToolOutcome({
            'artifact_id':'art-'+str(i),'page_url':source,'media':[
                {'artifact_id':'art-'+str(i),'kind':'image','source_ref':source,'bytes':100}]}),'ready-'+str(i))
        bridge.control.resource_urls.add(source)
    await bridge.finalize_media()
    assert len(sent)==1 and '额度' in sent[0]['text'],(bridge.trace,bridge.control.approved_artifacts(),
        bridge.control.fallback_ready_artifacts(),bridge.control.state())
    assert len(sent[0]['media_artifact_ids'])==(3 if mixed else 1)
    assert ('尚未逐项核实' in sent[0]['text'])==mixed


async def test_packet_sources_match_files_instead_of_an_old_inspected_post(bridge_factory,tmp_path):
    bridge,_,sent=prepare(bridge_factory,tmp_path,'private')
    actual=URL+'-actual'
    bridge.control.observe('inspect_web_resources',{'ref_ids':[actual]},ToolOutcome({'resources':[
        {'source_url':actual,'input_index':1,'post_id':'synthetic-actual','identity_status':'matched',
         'media_count':1,'media_roles':['post_image']}]}),'inspect-actual')
    bridge.control.observe('download_web_image',{'image_id':'webimg-a'},ToolOutcome({
        'artifact_id':'art-a','page_url':actual}),'ready')
    await bridge.finalize_media()
    assert len(sent)==2 and '\n来源：'+actual in sent[0]['text']
    assert '\n来源：'+URL+'\n' not in sent[0]['text']
    assert bridge.control.fallback_sources(['art-a'])==[actual]


async def test_search_only_budget_stop_preserves_candidate_source_receipts(bridge_factory,tmp_path):
    bridge,_,sent=prepare(bridge_factory,tmp_path,'private')
    bridge.control.post_metadata.clear();bridge.control.resource_urls.clear()
    candidate='https://x.com/synthetic/status/900000000000000009'
    bridge.control.observe('search_web',{},ToolOutcome({'sources':[{'source':candidate}]}),'search')
    await bridge.finalize_media()
    assert len(sent)==1 and candidate in sent[0]['text'] and '候选' in sent[0]['text']


async def test_last_model_capacity_is_reserved_for_delivery_and_further_search_is_blocked(bridge_factory,tmp_path):
    from tests.agent.test_harness_runtime import ModelHandler,request_body
    bridge,requests,sent=prepare(bridge_factory,tmp_path,'private')
    from app.agent.request_budget import RequestBudget
    bridge.control.workspace.media_budget.path.unlink()
    bridge.control.workspace.media_budget=RequestBudget(bridge.control.workspace,limits={})
    bridge.stop_reason=None
    bridge.control.observe('download_web_image',{'image_id':'webimg-a'},ToolOutcome({
        'artifact_id':'art-a','page_url':URL}),'ready')
    budget=bridge.request_budget()
    budget.spend('estimated_tokens',budget.snapshot()['limits']['estimated_tokens']-2000)
    async def search(args):raise AssertionError('search must not execute')
    bridge.tools['search_web']=tool('search_web',search,properties={'query':{'type':'string'}},required=('query',))
    body=request_body();body['input'][0]['content']=body['input'][0]['content'][:1]
    body['tools']=[{'type':'function','name':t.name,'description':t.description,
        'parameters':t.parameters} for t in bridge.tools.values()]
    # One call still fits; two full calls would exhaust the request.
    handler=ModelHandler();bridge.forward(handler,body)
    assert len(requests)==1 and bridge.budget_delivery_only,(handler.replies,budget.snapshot())
    assert requests[0]['tool_choice']=={'type':'function','name':'send_private'}
    result=await bridge.call_tool({'name':'search_web','arguments':{'query':'more'},'call_id':'too-late'})
    assert result['data']['error']=='request_budget_delivery_only' and not sent


async def test_completed_to_partial_keeps_budget_explanation_and_source(bridge_factory,tmp_path):
    bridge,_,sent=prepare(bridge_factory,tmp_path,'private')
    bridge.control.candidate_reviews[bridge.control._resource_key(URL)]={'decision':'selected'}
    for i in range(2):
        bridge.control.observe('download_web_image',{'image_id':'webimg-'+str(i)},ToolOutcome({
            'artifact_id':'art-'+str(i),'page_url':URL}),'ready-'+str(i))
    bridge.control.expected_artifacts=3
    await bridge.finalize_media()
    assert len(sent)==1 and sent[0]['task_status']=='partial'
    assert '额度' in sent[0]['text'] and URL in sent[0]['text']


@pytest.mark.parametrize('first_status',['sent','failed','uncertain'])
async def test_finalizer_continues_all_packets_but_stops_at_failed_or_unknown_receipt(
        bridge_factory,tmp_path,first_status):
    bridge,_,_=prepare(bridge_factory,tmp_path,'private')
    bridge.control.minimum_media_count=21
    bridge.control.candidate_reviews[bridge.control._resource_key(URL)]={'decision':'selected'}
    for i in range(21):
        bridge.control.observe('download_web_image',{'image_id':'webimg-'+str(i)},ToolOutcome({
            'artifact_id':'art-'+str(i),'page_url':URL,'bytes':100}),'ready-'+str(i))
    sent=[]
    async def send(args):
        sent.append(args)
        return ToolOutcome({},first_status,'synthetic-receipt' if first_status=='sent' else None)
    from dataclasses import replace
    bridge.tools['send_private']=replace(bridge.tools['send_private'],handler=send)
    await bridge.finalize_media()
    assert [len(args['media_artifact_ids']) for args in sent]==([20,1] if first_status=='sent' else [20])
    assert sent[0]['task_status']=='partial'
    if first_status=='sent':assert sent[1]['task_status']=='completed'
