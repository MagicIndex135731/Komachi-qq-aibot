"""Current-request completion and recovery, independent of formal research mode."""
from types import SimpleNamespace
import json

import pytest

from app.agent.research_tasks import ResearchTaskWorkspace
from app.agent.tool_agent import ToolOutcome
from app.agent.request_task import RequestTaskControl
from tests.agent.test_harness_runtime import bridge_factory


def control(tmp_path, request='把这个账号最近的图片下载发给我', **kwargs):
    workspace=ResearchTaskWorkspace(tmp_path, scope='private:900000101',actor_id=900000101,
        source_id=10,request=request,valid_source=lambda value:value==10)
    return RequestTaskControl(workspace, capability_version=kwargs.pop('capability_version','test-v1'), **kwargs)


def show_images(goal,identifiers):
    import base64,io
    from PIL import Image
    for identifier,color in zip(identifiers,['crimson','seagreen','royalblue','gold']):
        body=io.BytesIO();Image.new('RGB',(20,30),color).save(body,format='PNG')
        goal.observe('read_artifact',{'artifact_id':identifier},ToolOutcome({'artifact_id':identifier},
            images=({'data':base64.b64encode(body.getvalue()).decode(),'mediaType':'image/png'},)),identifier)


def test_media_goal_exists_without_formal_task_and_text_is_not_media_delivery(tmp_path):
    goal=control(tmp_path)
    assert not goal.workspace.formal
    result=goal.preflight('send_private',{'text':'看完了吧'})
    assert result.status=='blocked'
    assert 'media_delivery_unconfirmed' in result.data['missing']
    assert goal.state()['status']=='active'


def test_verified_metadata_counts_as_primary_but_access_failure_does_not_force_repeat_download(tmp_path):
    goal = control(tmp_path)
    url = 'https://www.douyin.com/video/900000000000000001'
    goal.observe('search_web', {}, ToolOutcome({'sources': [{'source': url}]}), 'search')
    goal.observe('inspect_web_resources', {'ref_ids': [url]},
        ToolOutcome({'error': 'web_resource_session_required'}, 'failed'), 'inspect')
    assert goal.preflight('send_private', {'text': '原站需要新鲜会话，尚未取得媒体', 'task_status': 'blocked'}) is None
    other = control(tmp_path / 'other')
    other.observe('inspect_web_resources', {'ref_ids': [url]}, ToolOutcome({'resources': [
        {'source_url': url, 'status': 'ok', 'post_id': '900000000000000001', 'identity_status': 'matched'}]}), 'inspect')
    assert 'primary_source_unverified' not in other._missing()
    assert 'media_delivery_unconfirmed' in other._missing()


@pytest.mark.parametrize('query',['我要看某人最近发的cos图','把网页里的照片搬过来',
    '想看这个作者刚上传的视频','Please download the photos and send them here',
    '继续帮我去X找一些关于某人的cos照片'])
def test_equivalent_media_goals_do_not_depend_on_site_keywords(tmp_path,query):
    goal=control(tmp_path,query)
    assert goal.requires_media


def test_chat_and_image_analysis_do_not_require_download_or_review(tmp_path):
    for text in ('早上好','看看这张图里写了什么','这段视频说的是什么','帮我写一首关于照片的小诗'):
        goal=control(tmp_path,text)
        assert not goal.requires_media
        assert goal.preflight('send_private',{'text':'正常回答'}) is None
        assert not goal.needs_review


def test_summary_only_search_cannot_satisfy_original_resource_goal(tmp_path):
    goal=control(tmp_path)
    for i in range(2):
        delta=goal.observe('search_web',{'query':str(i)},ToolOutcome({'content_kind':'search_summary',
            'sources':[{'source':'https://example.com/news'}],'content':'a generated summary'}),f'search-{i}')
    assert delta['missing']
    assert delta['no_progress']
    assert goal.needs_review
    assert goal.state()['evidence_kinds']==['search_summary']
    assert goal.observations[-1]['absence_proven'] is False


def test_blocked_completion_requires_real_attempt_and_never_proves_absence(tmp_path):
    goal=control(tmp_path)
    assert goal.preflight('send_private',{'text':'拿不到','task_status':'blocked'}).status=='blocked'
    goal.observe('download_web_video',{},ToolOutcome({'error':'web_http_403'},'failed'),'read1')
    assert goal.preflight('send_private',{'text':'下载接口返回403，尚未取得视频','task_status':'blocked'}) is None
    goal.observe('send_private',{'text':'下载失败','task_status':'blocked'},ToolOutcome({},'sent','receipt'),'send1')
    assert goal.state()['status']=='blocked'
    assert goal.can_stop


def test_found_media_locator_requires_a_real_attempt_before_linkless_refusal(tmp_path):
    goal = control(tmp_path)
    goal.observe('search_web', {}, ToolOutcome({'content_kind': 'grounded_answer',
        'sources': [{'source': 'https://x.com/synthetic/status/900000000000000001'}]}), 'search')
    goal.observe('open_web_page', {}, ToolOutcome({'error': 'web_http_404'}, 'failed'), 'wrong-page')
    args = {'text': '没有链接无法下载', 'task_status': 'blocked'}
    result = goal.preflight('send_private', args)
    assert 'discovered_media_not_attempted' in result.data['missing']
    goal.observe('download_twitter_media', {}, ToolOutcome({'error': 'twitter_media_unavailable'}, 'failed'), 'download')
    assert goal.preflight('send_private', args) is None


def test_media_locator_correction_is_bounded_and_does_not_override_authority(tmp_path):
    goal = control(tmp_path)
    goal.observe('search_web', {}, ToolOutcome({'content_kind': 'grounded_answer',
        'sources': [{'source': 'https://media.example/photo.jpg'}]}), 'search')
    args = {'text': '尚未取得媒体', 'task_status': 'partial'}
    assert goal.preflight('send_private', args) is not None
    assert goal.preflight('send_private', args) is not None
    assert goal.preflight('send_private', args) is None
    goal.corrections = 0
    goal.observe('download_web_image', {}, ToolOutcome({'error': 'owner_required'}, 'blocked'), 'no-auth')
    assert goal.preflight('send_private', args) is None


@pytest.mark.parametrize('stage', ['blocked', 'partial'])
def test_missing_archived_evidence_is_recoverable_before_any_external_attempt(tmp_path, stage):
    goal = control(tmp_path, '那把这个十月十三号的图片发给我')
    delta = goal.observe('read_research_evidence', {'evidence_id': 'a' * 24},
        ToolOutcome({'error': 'task_evidence_unavailable'}, 'blocked'), 'old-ref')
    assert delta['failure']['category'] == 'local_lookup'
    result = goal.preflight('send_private', {'text': '没有单帖链接，所以拿不到', 'task_status': stage})
    assert result is not None and 'recoverable_lookup_not_task_blocker' in result.data['missing']
    goal.observe('search_web', {'query': '真实目标'},
        ToolOutcome({'error': 'web_http_503'}, 'failed'), 'actual-network-attempt')
    assert goal.preflight('send_private', {'text': '本次实际搜索返回503，暂时受阻',
                                         'task_status': 'blocked'}) is None


def test_only_observed_artifacts_and_actual_receipts_finish_media_goal(tmp_path):
    goal=control(tmp_path)
    forged={'text':'图','artifact_ids':['art-forged']}
    assert goal.preflight('send_private',forged).status=='blocked'
    goal.observe('download_web_image',{'image_id':'img'},ToolOutcome({'artifact_id':'art-real'}),'d1')
    args={'text':'图','artifact_ids':['art-real']}
    assert goal.preflight('send_private',args) is None
    goal.observe('send_private',args,ToolOutcome({},'sent','real-receipt'),'s1')
    assert goal.state()['status']=='completed' and goal.can_stop


def test_pending_progress_and_uncertain_send_cannot_be_marked_completed(tmp_path):
    goal=control(tmp_path)
    goal.observe('download_web_images',{},ToolOutcome({'job_id':12,'execution_status':'queued'}),'q1')
    assert goal.preflight('send_private',{'text':'已经排队','task_status':'progress'}) is None
    goal.observe('send_private',{'text':'已经排队','task_status':'progress'},ToolOutcome({},'sent','ack'),'ack1')
    assert goal.state()['status']=='waiting' and goal.can_stop
    goal.observe('send_private',{'artifact_ids':['art-real']},ToolOutcome({},'uncertain'),'s2')
    assert not goal.can_stop and 'unknown_external_effect' in goal.state()['completion_gaps']


def test_goal_declaration_cannot_remove_explicit_media_requirement(tmp_path):
    goal=control(tmp_path)
    result=goal.describe({'goal':'回复文字','delivery':'text','requirements':[]})
    assert result['delivery']=='media'


def test_validated_background_result_is_usable_without_redownload(tmp_path):
    goal=control(tmp_path,initial_result={'kind':'media_result','status':'ok','data':{'artifact_ids':['art-result']}})
    assert goal.preflight('send_private',{'text':'图','artifact_ids':['art-result']}) is None


def test_versioned_experience_is_only_saved_after_verified_completion(tmp_path):
    goal=control(tmp_path)
    goal.observe('search_web',{},ToolOutcome({'content_kind':'search_summary'}),'a')
    goal.observe('download_web_image',{},ToolOutcome({'artifact_id':'art-real'}),'b')
    assert not goal.experiences()
    goal.observe('send_private',{'artifact_ids':['art-real']},ToolOutcome({},'sent','receipt'),'c')
    again=control(tmp_path)
    assert again.experiences()[0]['tools']==['search_web','download_web_image','send_private']
    different=control(tmp_path,capability_version='different')
    assert different.experiences()==[]
    again.workspace.valid_source=lambda _:False
    assert again.experiences()==[]


def test_recent_smalltalk_and_partial_media_have_distinct_completion(tmp_path):
    goal=control(tmp_path,'最近过得怎么样')
    assert not goal.active
    goal=control(tmp_path)
    goal.observe('download_web_images',{},ToolOutcome({'requested_count':3,'downloaded_count':2,
        'artifact_ids':['art-one','art-two'],'failures':[{'image_id':'third'}]}),'d')
    assert goal.preflight('send_private',{'text':'都好了','artifact_ids':['art-one','art-two']}).status=='blocked'
    args={'text':'仅2张成功，第三张超时','artifact_ids':['art-one','art-two'],'task_status':'partial'}
    assert goal.preflight('send_private',args) is None
    goal.observe('send_private',args,ToolOutcome({},'sent','partial-receipt'),'s')
    assert goal.can_stop and goal.state()['status']=='partial'


def test_actual_work_activates_goal_and_discovered_resources_advance_it(tmp_path):
    goal=control(tmp_path,'把这件事处理一下')
    assert not goal.active
    delta=goal.observe('read_research_resources',{},ToolOutcome({'resources':[
        {'url':'https://example.com/photo.png'}]}),'index')
    assert goal.active and delta['new_evidence']==1 and not delta['no_progress']
    assert goal.observations[-1]['kind']=='resource_index'
    assert (goal.workspace.root/'request-goal.json').is_file()
    assert not goal.workspace._file('request-goal.json').exists()


def test_scoped_observed_forward_pixels_are_valid_media_but_failure_is_not_a_verified_method(tmp_path):
    goal=control(tmp_path,'把转发里的图片发给我')
    goal.observe('observe_forward_images',{},ToolOutcome({'images':[{'artifact_id':'art-forward'}]}),'pixels')
    assert goal.preflight('send_private',{'text':'图','artifact_ids':['art-forward']}) is None
    goal.observe('download_web_video',{},ToolOutcome({'error':'web_http_403'},'failed'),'failure')
    goal.observe('send_private',{'text':'图','artifact_ids':['art-forward']},ToolOutcome({},'sent','actual'),'sent')
    assert goal.can_stop and not goal.experiences()


@pytest.mark.asyncio
@pytest.mark.parametrize('surface',['group','private'])
async def test_native_default_goal_rejects_premature_text_then_delivers_actual_media_once(tmp_path,surface):
    from app.agent.harness_runtime import HarnessToolAgent
    from app.agent.request_task import delivery_properties
    from test_harness_runtime import Guard,model_client,responses_sse,tool
    name='send_'+surface
    workspace=ResearchTaskWorkspace(tmp_path/'task',scope=surface+':900000101',actor_id=900000101,
        source_id=10,request='把网页里的照片搬过来',valid_source=lambda value:value==10)
    seen=[]
    def script(body,n):
        if n==1:return responses_sse(name=name,arguments={'text':'图拿不到'},number=n,call_id=f'c{n}')
        if n==2:
            assert 'task_completion_unverified' in str(body)
            return responses_sse(name='download_web_image',arguments={},number=n,call_id=f'c{n}')
        return responses_sse(name=name,arguments={'text':'实际原图','artifact_ids':['art-real']},number=n,call_id=f'c{n}')
    client,requests,_=model_client(script)
    async def send(args):
        seen.append(args)
        return ToolOutcome({},'sent','actual-receipt')
    async def download(args):return ToolOutcome({'artifact_id':'art-real'})
    agent=HarnessToolAgent(decision_client=client,task_workspace=workspace,require_reply=True,
        reply_source_id=10,max_external_writes=1,session_home=tmp_path/'native')
    tools=(tool(name,send,effect='write',properties={'text':{'type':'string'},
        'artifact_ids':{'type':'array','items':{'type':'string'}},**delivery_properties()},required=('text',)),
        tool('download_web_image',download))
    try:
        result=await agent.run(['Synthetic current request'],tools,Guard())
        assert result.status=='sent' and result.task_status['status']=='completed',result
        from app.agent.context_recovery import context_reusable
        assert context_reusable(result),result.trace
        assert len(requests)==3 and len(seen)==1
        assert not workspace.formal
        assert {'set_task_goal','review_task_progress'}<= {t['name'] for t in requests[0]['tools']}
    finally:agent.close();client.http_client.close()


@pytest.mark.asyncio
async def test_short_review_is_opt_in_bounded_and_shares_actual_model_budget(tmp_path):
    from app.agent.harness_runtime import HarnessToolAgent,_Bridge
    from test_harness_runtime import Guard,model_client,responses_sse
    goal=control(tmp_path)
    goal.observe('search_web',{},ToolOutcome({'content_kind':'grounded_answer'}),'r1')
    goal.observe('search_web',{},ToolOutcome({'content_kind':'grounded_answer'}),'r2')
    client,requests,_=model_client(lambda *_:responses_sse(text='{"gap":"缺少原帖","next_tool":"open_web_page","next_step":"读取来源"}'))
    reserved=[]
    owner=HarnessToolAgent(decision_client=client,before_call=lambda kind:reserved.append(kind) or True)
    bridge=_Bridge(owner,{},Guard(),None,None,None)
    offered=goal.tools(Guard(),bridge.review_task)
    try:
        for _ in range(3):
            result=await offered[1].handler({'model_review':True})
            assert result.status=='ok'
        assert len(requests)==1 and bridge.model_calls==1 and reserved==['model']
        assert requests[0].get('tools',[])==[]
        assert requests[0]['max_output_tokens']==512
        assert 'canonical_request' in str(requests[0])
        assert not any(x.get('purpose')=='agent' for x in bridge.trace)
    finally:bridge.server.server_close();owner.close();client.http_client.close()


def test_failed_auxiliary_review_does_not_rotate_a_successful_native_conversation():
    from app.agent.context_recovery import context_reusable
    from app.agent.tool_agent import ToolTurnResult
    result=ToolTurnResult('sent','receipt','tool:send_private',trace=(
        {'status':'model_request','purpose':'agent','model_call_index':1,'outcome':'completed'},
        {'status':'model_request','purpose':'task_review','model_call_index':2,'outcome':'TimeoutError'}))
    assert context_reusable(result)


def test_plural_media_cannot_complete_or_auto_handoff_with_one_item(tmp_path):
    goal=control(tmp_path,'帮我扒一些关于某人的cos视频或图片')
    assert goal.state()['minimum_media_count']==2
    rejected=goal.preflight('download_twitter_media',{'ref_id':'https://x.com/sample/status/900000000000000001',
        'delivery_mode':'auto'})
    assert rejected.data['error']=='media_selection_quantity_unverified'
    goal.observe('download_web_image',{},ToolOutcome({'artifact_id':'art-one'}),'one')
    assert goal.preflight('send_private',{'text':'好了','artifact_ids':['art-one']}).status=='blocked'
    args={'text':'仅找到一张，仍缺多份','artifact_ids':['art-one'],'task_status':'partial'}
    assert goal.preflight('send_private',args) is None


def test_semantic_goal_keeps_subject_distinct_from_publishers_and_quantity_floor(tmp_path):
    goal=control(tmp_path,'帮我扒一些关于某人的cos视频或图片')
    goal.describe({'goal':'关于某人的近期cos媒体','delivery':'media','requirements':[],
        'media_selection':{'subject':'某人','publisher_scope':'any','minimum_count':3,
            'criteria':'内容是本人cos；非必须由本人发布；图或视频均可'}})
    goal.describe({'goal':'单图就好','delivery':'media','requirements':[],
        'media_selection':{'minimum_count':1}})
    assert goal.state()['minimum_media_count']==2
    assert goal.state()['media_selection']['publisher_scope']=='any'
    goal.observe('download_web_images',{},ToolOutcome({'artifact_ids':['art-a','art-b','art-c']}),'download')
    assert goal.preflight('send_private',{'text':'全部','artifact_ids':['art-a']}).status=='blocked'
    args={'text':'三份','artifact_ids':['art-a','art-b','art-c']}
    show_images(goal,args['artifact_ids'])
    assert goal.preflight('send_private',args) is None
    goal.observe('send_private',args,ToolOutcome({},'sent','receipt'),'send')
    assert goal.can_stop and goal.state()['delivered_media_count']==3


def test_plural_auto_accepts_verified_multi_media_post_but_not_repeated_single_post(tmp_path):
    goal=control(tmp_path,'下载一些照片给我')
    url='https://x.com/sample/status/900000000000000001'
    row={'source_url':url,'post_id':'900000000000000001','identity_status':'matched','media_count':2}
    goal.observe('search_x_posts',{},ToolOutcome({'resources':[row]}),'search')
    assert goal.preflight('download_twitter_media',{'ref_id':url,'delivery_mode':'auto'}) is None
    row['media_count']=1
    goal.observe('search_x_posts',{},ToolOutcome({'resources':[row]}),'update')
    op={'operation':'download_twitter_media','arguments':{'ref_id':url,'delivery_mode':'auto'}}
    assert goal.preflight('queue_media_batch',{'operations':[op,op]}).status=='blocked'


def test_media_callback_keeps_original_semantic_minimum_but_not_old_artifact_authority(tmp_path):
    goal=control(tmp_path,'下载一些关于某人的照片')
    goal.describe({'goal':'多份','delivery':'media','requirements':[],
        'media_selection':{'minimum_count':4,'publisher_scope':'any','subject':'某人'}})
    goal.observe('download_web_image',{},ToolOutcome({'artifact_id':'art-previous'}),'old')
    again=control(tmp_path,'下载一些关于某人的照片',initial_result={'kind':'media_result',
        'status':'ok','data':{'artifact_ids':['art-current']}})
    assert again.state()['minimum_media_count']==2
    assert again.artifacts=={'art-current'}
    assert again.media_selection['publisher_scope']=='any'
    again.workspace.source_id=11
    assert again.workspace.media_selection_for_source()=={}


def test_cropped_and_resized_photo_does_not_count_as_another_requested_item(tmp_path):
    import base64,io,random
    from PIL import Image
    rng=random.Random(23)
    photo=Image.new('RGB',(96,144));photo.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(96*144)])
    crop=photo.crop((0,30,96,90)).resize((192,120))
    goal=control(tmp_path,'下载一些照片给我')
    for identifier,pixels in [('art-photo',photo),('art-crop',crop)]:
        buffer=io.BytesIO();pixels.save(buffer,format='PNG')
        goal.observe('download_web_image',{},ToolOutcome({'artifact_id':identifier},
            images=({'data':base64.b64encode(buffer.getvalue()).decode(),'mediaType':'image/png'},)),identifier)
    rejected=goal.preflight('send_private',{'text':'两张','artifact_ids':['art-photo','art-crop']})
    assert rejected is not None and 'media_delivery_quantity_incomplete' in rejected.data['missing']
    assert goal.state()['distinct_media_count']==1


def test_semantically_excluded_candidates_do_not_force_irrelevant_downloads(tmp_path):
    goal=control(tmp_path,'扒一些关于某人的cos图片')
    url='https://x.com/sample/status/900000000000000001'
    goal.observe('search_x_posts',{},ToolOutcome({'resources':[{'source_url':url,
        'post_id':'900000000000000001','identity_status':'matched','media_count':2}]}),'search')
    assert 'discovered_media_not_attempted' in goal.preflight('send_private',{
        'text':'当前候选均不符','task_status':'partial'}).data['missing']
    goal.describe({'goal':'同一真实目标','delivery':'media','requirements':[],
        'media_selection':{'candidate_reviews':[{'ref_id':'900000000000000001',
            'decision':'excluded','reason':'原帖正文和预览是日常餐点，不是该人物cos'}]}})
    assert goal.preflight('send_private',{'text':'发现的餐点图不符cos要求，仍缺媒体','task_status':'partial'}) is None
    with pytest.raises(ValueError,match='task_candidate_not_observed'):
        goal.describe({'media_selection':{'candidate_reviews':[{'ref_id':'https://example.com/unseen',
            'decision':'excluded','reason':'未观察来源不得用于绕过校验'}]}})


def test_explicitly_requested_image_variants_can_be_delivered_as_two_files(tmp_path):
    goal=control(tmp_path,'把原图和裁切版这两张照片都发给我')
    goal.describe({'goal':'发送用户明确要的两个版本','delivery':'media','requirements':[],
        'media_selection':{'minimum_count':2,'variants_requested':True}})
    goal.observe('download_web_images',{},ToolOutcome({'artifact_ids':['art-original','art-crop']}),'download')
    goal.content_groups={'art-crop':'art-original'}
    assert goal.preflight('send_private',{'text':'两个版本','artifact_ids':['art-original','art-crop']}) is None


def test_topic_delivery_uses_post_text_without_mandatory_image_preview(tmp_path):
    goal=control(tmp_path,'找一些关于某人的cos照片')
    goal.describe({'goal':'只发目标主体','delivery':'media','requirements':[],
        'media_selection':{'subject':'某人','minimum_count':2}})
    rejected=goal.preflight('download_twitter_media',{'ref_id':'https://x.com/sample/status/900000000000000001',
        'delivery_mode':'auto'})
    assert rejected.data['error']=='media_topic_review_required'
    assert rejected.data['handler_executed'] is False and rejected.data['suggested_delivery_mode']=='review'
    goal.observe('media_result',{},ToolOutcome({'artifact_ids':['art-a','art-b','art-other']}),'download')
    assert not goal.state()['visual_review_required']
    assert goal.state()['selection_basis']=='post_text_primary'
    assert goal.reviewed_images==set()
    assert goal.preflight('send_private',{'text':'按原帖正文挑选两张','artifact_ids':['art-a','art-b']}) is None


def test_topic_mixed_media_delivery_does_not_require_video_or_image_pixels(tmp_path):
    goal=control(tmp_path,'找一些关于某人的cos图片和视频')
    goal.describe({'goal':'依据帖子说明挑选媒体','delivery':'media','requirements':[],
        'media_selection':{'subject':'某人','minimum_count':2}})
    goal.observe('media_result',{},ToolOutcome({'artifact_ids':['art-photo'],
        'video_artifact_ids':['art-video'],'media_artifact_ids':['art-photo','art-video']}),'download')
    args={'text':'根据原帖描述挑选，未观看视频','media_artifact_ids':['art-photo','art-video']}
    assert goal.preflight('send_private',args) is None
    assert 'art-video' not in goal.image_artifacts
    assert goal.reviewed_images==set()
    assert goal.state()['distinct_media_count']==2


@pytest.mark.asyncio
async def test_native_topic_auto_is_corrected_to_review_before_single_delivery(tmp_path):
    from app.agent.harness_runtime import HarnessToolAgent
    from test_harness_runtime import Guard,model_client,responses_sse,tool
    downloads=[];deliveries=[]
    workspace=ResearchTaskWorkspace(tmp_path/'task',scope='private:900000101',actor_id=900000101,
        source_id=10,request='帮我找一些某主题的视频',valid_source=lambda value:value==10)
    url='https://x.com/sample/status/900000000000000001'
    def script(body,n):
        if n==1:return responses_sse(name='set_task_goal',arguments={'goal':'按帖子文本筛选多份视频',
            'delivery':'media','requirements':[], 'media_selection':{'subject':'某主题','minimum_count':2}},number=n,call_id=f'c{n}')
        if n==2:
            return responses_sse(name='download_twitter_media',arguments={'ref_id':url,
                'delivery_mode':'auto'},number=n,call_id=f'c{n}')
        assert n==3 and 'delivery_mode_adjusted' in json.dumps(body)
        return responses_sse(name='send_private',arguments={'text':'依据原帖正文挑选，未观看视频',
            'media_artifact_ids':['art-video-a','art-video-b'],'as_forward':True},number=n,call_id=f'c{n}')
    client,requests,_=model_client(script)
    async def download(args):
        downloads.append(args)
        return ToolOutcome({'media_artifact_ids':['art-video-a','art-video-b'],
            'video_artifact_ids':['art-video-a','art-video-b']})
    async def send(args):deliveries.append(args);return ToolOutcome({},'sent','actual-receipt')
    tools=(tool('download_twitter_media',download,properties={'ref_id':{'type':'string'},
        'delivery_mode':{'type':'string','enum':['auto','review']}}),
        tool('send_private',send,effect='write',properties={'text':{'type':'string'},'as_forward':{'type':'boolean'},
            'media_artifact_ids':{'type':'array','maxItems':20,'items':{'type':'string'}}},required=('text',)))
    agent=HarnessToolAgent(decision_client=client,task_workspace=workspace,require_reply=True,
        reply_source_id=10,session_home=tmp_path/'native',max_steps=6)
    try:
        result=await agent.run(['Synthetic authorized current request'],tools,Guard())
        assert result.status=='sent' and result.task_status['status']=='completed',result
        assert len(requests)==3 and downloads==[{'ref_id':url,'delivery_mode':'review'}] and len(deliveries)==1
        assert all('input_image' not in json.dumps(body) for body in requests)
    finally:agent.close();client.http_client.close()


def test_topic_mode_normalization_only_copies_auto_download_arguments(tmp_path):
    goal=control(tmp_path,'找几张某主题照片发给我')
    goal.describe({'goal':'按原帖文本筛选','delivery':'media','media_selection':{'subject':'sample'}})
    original={'operations':[{'operation':'download_twitter_media','arguments':{'ref_id':'source-a','delivery_mode':'auto','delivery_text':'caption'}},
        {'operation':'download_web_video','arguments':{'ref_id':'source-b','delivery_mode':'review'}}]}
    before=json.dumps(original,sort_keys=True)
    prepared=goal.prepare_tool_arguments('queue_media_batch',original)
    assert prepared is not original and json.dumps(original,sort_keys=True)==before
    assert prepared['operations'][0]['arguments']=={'ref_id':'source-a','delivery_mode':'review','delivery_text':'caption'}
    assert prepared['operations'][1]==original['operations'][1]
    assert goal.preflight('queue_media_batch',prepared) is None
    assert goal.prepare_tool_arguments('search_web',{'query':'sample'})=={'query':'sample'}
    goal.describe({'media_selection':{'selection_scope':'all_from_explicit_source'}})
    assert goal.prepare_tool_arguments('queue_media_batch',original) is original


async def test_topic_mode_correction_is_deduplicated_and_keeps_review_in_main(bridge_factory,tmp_path):
    from test_harness_runtime import tool,responses_sse
    bridge,requests,_=bridge_factory(lambda *_:responses_sse(),require_reply=True)
    bridge.control=control(tmp_path,'找几张某主题照片发给我')
    bridge.control.describe({'delivery':'media','media_selection':{'subject':'sample'}})
    calls=[]
    async def queue(args):
        calls.append(args)
        return ToolOutcome({'job_id':42,'status':'queued','delivery':'background_media_result_returns_to_main'})
    bridge.tools={'download_twitter_media':tool('download_twitter_media',queue,
        properties={'ref_id':{'type':'string'},'delivery_mode':{'type':'string','enum':['auto','review']}})}
    args={'ref_id':'source-a','delivery_mode':'auto'};body={'name':'download_twitter_media','arguments':args,'call_id':'same'}
    first=await bridge.call_tool(body);again=await bridge.call_tool(body)
    assert first['status']==again['status']=='ok' and first['data']['delivery_mode_adjusted']=='review'
    assert calls==[{'ref_id':'source-a','delivery_mode':'review'}] and args['delivery_mode']=='auto'
    assert not bridge.auto_handoff and not requests
    changed=await bridge.call_tool({**body,'arguments':{**args,'ref_id':'source-b'}})
    assert changed['data']['error']=='call_id_reused' and len(calls)==1


def test_pending_completion_exposes_exact_status_refresh_without_redownload(tmp_path):
    goal=control(tmp_path,'找一些图片发我')
    goal.observe('download_twitter_media',{},ToolOutcome({'job_id':9001,'status':'queued'}),'download')
    rejected=goal.preflight('send_private',{'text':'图片已完成','artifact_ids':[]})
    assert rejected.data['task']['pending_jobs']==[{'tool':'media_task_status','arguments':{'job_id':9001},'observed_status':'queued'}]
    assert '不要重新下载' in rejected.data['guidance']
    goal.observe('media_task_status',{'job_id':9001},ToolOutcome({'job_id':9001,'status':'completed'}),'refresh')
    assert goal.state()['pending_jobs']==[]


def recent_candidates(goal):
    from datetime import UTC,datetime
    rows=[{'source_url':f'https://x.com/sample/status/90000000000000000{i}',
        'post_id':f'90000000000000000{i}','author':'sample','identity_status':'matched',
        'published_at':datetime.fromtimestamp(goal.workspace.clock(),UTC).isoformat(),
        'media_count':1,'media_roles':['post_video']} for i in range(1,4)]
    goal.observe('search_x_posts',{},ToolOutcome({'resources':rows}),'search')
    return rows


def test_media_resume_restores_candidate_dates_aliases_and_attempts_without_old_artifacts(tmp_path):
    goal=control(tmp_path,'下载几个近7天的视频给我')
    rows=recent_candidates(goal)
    goal.observe('download_twitter_media',{'ref_id':rows[0]['source_url']},
        ToolOutcome({'job_id':41,'status':'queued'}),'first')
    goal.observe('download_web_image',{},ToolOutcome({'artifact_id':'art-old'}),'old')
    again=control(tmp_path,'下载几个近7天的视频给我',initial_result={'kind':'media_result',
        'status':'ok','data':{'video_artifact_ids':['art-first']}})
    args={'ref_id':rows[1]['post_id'],'delivery_mode':'review'}
    assert again.preflight('download_twitter_media',args) is None
    assert again.prepare_tool_arguments('download_twitter_media',args)['ref_id']==rows[1]['source_url']
    assert again.artifacts=={'art-first'}
    assert rows[0]['source_url'] not in again.state()['remaining_media_candidates']
    assert rows[1]['source_url'] in again.state()['remaining_media_candidates']


def test_plural_partial_must_recover_local_date_check_and_try_remaining_candidates(tmp_path):
    goal=control(tmp_path,'下载几个近7天的视频给我')
    rows=recent_candidates(goal)
    goal.observe('download_twitter_media',{'ref_id':rows[0]['source_url']},
        ToolOutcome({'video_artifact_ids':['art-one'],'source_url':rows[0]['source_url']}),'one')
    blocked=goal.preflight('download_twitter_media',{'ref_id':'https://x.com/sample/status/900000000000000009'})
    goal.observe('download_twitter_media',{'ref_id':'https://x.com/sample/status/900000000000000009'},blocked,'date')
    args={'text':'只有一份，仍缺数量','video_artifact_id':'art-one','task_status':'partial'}
    rejected=goal.preflight('send_private',args)
    assert rejected is None
    assert goal._remaining_candidates()
    assert goal.failures[-1]['category']=='local_lookup'
    assert blocked.data['handler_executed'] is False and blocked.data['background_queued'] is False
    for row in rows[1:]:
        goal.observe('download_twitter_media',{'ref_id':row['source_url']},
            ToolOutcome({'error':'synthetic_download_failed'},'failed'),row['post_id'])
    assert goal.preflight('send_private',args) is None


def test_background_promise_requires_a_real_pending_job_even_on_partial_delivery(tmp_path):
    goal=control(tmp_path,'下载几个视频给我')
    goal.observe('download_twitter_media',{},ToolOutcome({'video_artifact_ids':['art-one']}),'one')
    args={'text':'先给你第一份，第二份还在后台给你下着呢',
        'video_artifact_id':'art-one','task_status':'partial'}
    rejected=goal.preflight('send_private',args)
    assert rejected and 'background_claim_unverified' in rejected.data['missing']
    assert goal.preflight('send_private',{**args,'text':'只取得一份，另一份未启动下载'}) is None
    goal.observe('download_twitter_media',{},ToolOutcome({'job_id':42,'status':'queued'}),'queued')
    assert goal.preflight('send_private',args) is None
    assert goal.preflight('send_private',{**args,'task_status':'progress'}) is None


def test_remaining_candidates_are_bounded_and_do_not_override_hard_access_failure(tmp_path):
    goal=control(tmp_path,'下载几个视频给我')
    recent_candidates(goal)
    args={'text':'当前来源需要登录，未取得素材','task_status':'blocked'}
    goal.observe('download_twitter_media',{},ToolOutcome({'error':'web_resource_session_required'},'blocked'),'auth')
    assert goal.preflight('send_private',args) is None


@pytest.mark.parametrize('changed',['source','scope','actor','task','revoked','dependencies','expired'])
def test_candidate_resume_is_bound_to_original_valid_source(tmp_path,changed):
    from datetime import UTC,datetime
    now=datetime.now(UTC).timestamp()
    original=ResearchTaskWorkspace(tmp_path,scope='group:900000001',actor_id=900000101,
        source_id=10,request='下载几个近7天的视频',valid_source=lambda _:True,clock=lambda:now,
        dependency_snapshot=lambda:{'sources':[10]},valid_dependencies=lambda _:True)
    goal=RequestTaskControl(original,capability_version='test')
    rows=recent_candidates(goal)
    kwargs={'scope':original.scope,'actor_id':original.actor_id,'source_id':10,
        'request':original.request,'valid_source':lambda _:True,'clock':lambda:now,
        'valid_dependencies':lambda _:True}
    if changed=='source':kwargs['source_id']=11
    if changed=='scope':kwargs['scope']='group:900000002'
    if changed=='actor':kwargs['actor_id']=900000102
    if changed=='task':kwargs['task_key']='another'
    if changed=='revoked':kwargs['valid_source']=lambda _:False
    if changed=='dependencies':kwargs['valid_dependencies']=lambda _:False
    if changed=='expired':kwargs['clock']=lambda:now+8*86400
    resumed=ResearchTaskWorkspace(tmp_path,**kwargs)
    if changed in {'revoked','dependencies','expired'}:
        with pytest.raises(ValueError,match='unavailable'):resumed.media_selection_for_source()
    else:
        again=RequestTaskControl(resumed,capability_version='test',initial_result={
            'kind':'media_result','status':'ok','data':{'artifact_ids':['art-current']}})
        assert rows[1]['post_id'] not in again.post_metadata
        assert again.preflight('download_twitter_media',{'ref_id':rows[1]['source_url']}).status=='blocked'


def test_exhausted_topic_candidates_request_one_supplement_then_allow_honest_partial(tmp_path):
    goal=control(tmp_path,'帮我找几个最近的旅行视频')
    goal.describe({'delivery':'media','media_selection':{'subject':'旅行','selection_scope':'topic'}})
    rows=recent_candidates(goal)
    for row in rows:
        goal.observe('download_twitter_media',{'ref_id':row['source_url']},
            ToolOutcome({'error':'synthetic_download_failed'},'failed'),row['post_id'])
    args={'text':'本次未取得视频','task_status':'partial'}
    assert 'media_discovery_incomplete' in goal.preflight('send_private',args).data['missing']
    goal.observe('search_x_posts',{'subject_terms':['旅行'],'topic_match':'review'},
        ToolOutcome({'resources':[]}),'supplement')
    assert goal.preflight('send_private',args) is None


def test_date_preflight_without_external_attempt_requires_local_evidence_recovery(tmp_path):
    goal=control(tmp_path,'下载几个近7天的视频给我')
    args={'ref_id':'https://x.com/sample/status/900000000000000001','delivery_mode':'review'}
    blocked=goal.preflight('download_twitter_media',args)
    goal.observe('download_twitter_media',args,blocked,'local-check')
    rejected=goal.preflight('send_private',{'text':'无法下载','task_status':'blocked'})
    assert 'recoverable_lookup_not_task_blocker' in rejected.data['missing']


def test_partial_batch_only_marks_accepted_candidates_as_attempted(tmp_path):
    goal=control(tmp_path,'下载几个视频给我')
    rows=recent_candidates(goal)
    operations=[{'operation':'download_twitter_media','arguments':{'ref_id':r['source_url']}} for r in rows]
    goal.observe('queue_media_batch',{'operations':operations},ToolOutcome({
        'jobs':[{'index':0,'job_id':41,'status':'queued'}],
        'failures':[{'index':1,'error':'synthetic_capacity'},{'index':2,'error':'synthetic_capacity'}]}),'batch')
    assert goal._remaining_candidates()==[r['source_url'] for r in rows[1:]]


def test_partial_cannot_omit_already_available_media_without_bounded_feedback(tmp_path):
    goal=control(tmp_path,'下载几个视频给我')
    goal.observe('download_twitter_media',{},ToolOutcome({'video_artifact_ids':['art-a','art-b']}),'download')
    args={'text':'只有一份','video_artifact_id':'art-a','task_status':'partial'}
    assert goal.preflight('send_private',args) is None
    assert goal.preflight('send_private',{'text':'两份','media_artifact_ids':['art-a','art-b']}) is None


async def test_reference_resolution_does_not_falsely_report_a_delivery_mode_change(bridge_factory,tmp_path):
    from tests.agent.test_harness_runtime import tool,responses_sse
    bridge,_,_=bridge_factory(lambda *_:responses_sse(),require_reply=True)
    bridge.control=control(tmp_path,'下载这个帖子的视频给我')
    rows=recent_candidates(bridge.control)
    calls=[]
    async def download(args):calls.append(args);return ToolOutcome({'video_artifact_ids':['art-a']})
    bridge.tools={'download_twitter_media':tool('download_twitter_media',download,
        properties={'ref_id':{'type':'string'},'delivery_mode':{'type':'string','enum':['auto','review']}})}
    output=await bridge.call_tool({'name':'download_twitter_media','call_id':'resolve',
        'arguments':{'ref_id':rows[0]['post_id'],'delivery_mode':'auto'}})
    assert calls==[{'ref_id':rows[0]['source_url'],'delivery_mode':'auto'}]
    assert 'delivery_mode_adjusted' not in output['data']


async def test_native_media_resume_replaces_failed_second_candidate_and_delivers_two(tmp_path):
    from app.agent.harness_runtime import HarnessToolAgent
    from tests.agent.test_harness_runtime import Guard,model_client,responses_sse,tool
    original=control(tmp_path/'task','下载几个最近的旅行视频给我')
    rows=recent_candidates(original)
    original.observe('download_twitter_media',{'ref_id':rows[0]['source_url']},
        ToolOutcome({'job_id':41,'status':'queued'}),'first')
    effects=[];sent=[]
    def script(body,n):
        if n==1:return responses_sse(name='download_twitter_media',arguments={
            'ref_id':rows[1]['post_id'],'delivery_mode':'review'},number=n,call_id=f'c{n}')
        if n==2:return responses_sse(name='send_private',arguments={'text':'只找到一份',
            'task_status':'partial','media_artifact_ids':['art-first']},number=n,call_id=f'c{n}')
        if n==3:
            assert 'partial' in json.dumps(body)
            return responses_sse(name='download_twitter_media',arguments={
                'ref_id':rows[2]['post_id'],'delivery_mode':'review'},number=n,call_id=f'c{n}')
        assert n==4
        return responses_sse(name='send_private',arguments={'text':'根据原帖选出两份，另一候选下载失败',
            'task_status':'completed','media_artifact_ids':['art-first','art-third']},number=n,call_id=f'c{n}')
    async def download(args):
        effects.append(args['ref_id'])
        return (ToolOutcome({'error':'synthetic_download_failed'},'failed') if len(effects)==1 else
            ToolOutcome({'video_artifact_ids':['art-third']}))
    async def send(args):sent.append(args);return ToolOutcome({},'sent','actual-synthetic-receipt')
    tools=(tool('download_twitter_media',download,properties={'ref_id':{'type':'string'},
        'delivery_mode':{'type':'string','enum':['review','auto']}}),
        tool('send_private',send,effect='write',properties={'text':{'type':'string'},
            'task_status':{'type':'string','enum':['completed','partial','progress','blocked']},
            'media_artifact_ids':{'type':'array','items':{'type':'string'}}},required=('text',)))
    client,requests,_=model_client(script)
    agent=HarnessToolAgent(decision_client=client,task_workspace=original.workspace,
        initial_evidence=({'kind':'media_result','status':'ok','data':{'video_artifact_ids':['art-first']}},),
        require_reply=True,reply_source_id=10,session_home=tmp_path/'native',max_steps=6)
    try:
        result=await agent.run(['Synthetic current request'],tools,Guard())
        assert result.status=='sent' and result.task_status['status']=='completed',result
        assert effects==[rows[1]['source_url'],rows[2]['source_url']]
        assert len(requests)==4 and len(sent)==2 and result.task_status['delivered_media_count']==2
        assert [item['media_artifact_ids'] for item in sent]==[['art-first'],['art-third']]
    finally:agent.close();client.http_client.close()


@pytest.mark.parametrize('query_text',['去X扒一些某人的图和视频','不要只找近7天，给我某人的图',
    '之前回复说“近30天没图”，现在帮我找图片','找某人最近的图'])
def test_model_window_does_not_harden_unbounded_request(tmp_path,query_text):
    goal=control(tmp_path,query_text)
    goal.describe({'media_selection':{'lookback_days':30}})
    assert goal.published_after is None
    assert goal.state()['publication_filter']['provenance']=='unbounded'
    for operation in ['download_twitter_media','download_web_video']:
        assert goal.preflight(operation,{'ref_id':'https://x.com/synthetic/status/900000000000000001'}) is None


def test_explicit_date_preserved_and_no_implicit_author_restriction(tmp_path):
    goal=control(tmp_path,'去小红书找某人近7天的图')
    cutoff=goal.published_after
    goal.describe({'media_selection':{'lookback_days':30,'subject':'某人','publisher_scope':'subject_only','minimum_count':5}})
    assert goal.published_after==cutoff and goal.media_selection['publisher_scope']=='any'
    assert goal.minimum_media_count==1 and goal.media_selection['target_count']==5
    assert goal.preflight('download_web_video',{'ref_id':'https://www.xiaohongshu.com/explore/0123456789abcdef01234567'}).data['error']=='resource_freshness_unverified'


async def test_three_failed_inspects_allow_first_actual_websearch(bridge_factory,tmp_path):
    from tests.agent.test_harness_runtime import tool,responses_sse
    bridge,_,_=bridge_factory(lambda *_:responses_sse())
    bridge.control=control(tmp_path,'去X扒某人的图片')
    for i in range(3):
        bridge.control.observe('inspect_web_resources',{},ToolOutcome({'error':'web_http_404'},'failed'),str(i))
    calls=[]
    async def search(args):calls.append(args);return ToolOutcome({'sources':[]})
    bridge.tools={'search_web':tool('search_web',search,properties={'query':{'type':'string'}})}
    result=await bridge.call_tool({'name':'search_web','call_id':'fresh-method','arguments':{'query':'site:x.com synthetic'}})
    assert len(calls)==1 and result['status']=='ok'


def test_partial_delivery_does_not_complete_with_remaining_candidates(tmp_path):
    goal=control(tmp_path,'下载3个视频')
    recent_candidates(goal)
    goal.observe('download_twitter_media',{},ToolOutcome({'video_artifact_ids':['art-a']}),'ready')
    args={'text':'先交付一份，仍缺两份','video_artifact_id':'art-a','task_status':'partial'}
    assert goal.preflight('send_private',args) is None
    goal.observe('send_private',args,ToolOutcome({},'sent','receipt-a'),'send')
    assert goal.stage=='partial' and not goal.can_stop and goal.minimum_media_count==3


@pytest.mark.parametrize('query,expected',[('下载20张图片',20),('给我十二张照片',12),('帮我找100份图片',100)])
def test_user_quantity_not_clamped_to_per_tool_batch(tmp_path,query,expected):
    goal=control(tmp_path,query)
    goal.describe({'media_selection':{'minimum_count':1}})
    assert goal.minimum_media_count==expected


def test_images_do_not_complete_explicit_image_and_video_request(tmp_path):
    goal=control(tmp_path,'帮我找图片和视频')
    goal.observe('download_web_images',{},ToolOutcome({'artifact_ids':['art-a','art-b']}),'images')
    result=goal.preflight('send_private',{'text':'全好了','artifact_ids':['art-a','art-b']})
    assert result and 'media_delivery_types_incomplete' in result.data['missing']
    assert goal.preflight('send_private',{'text':'先给两张图，视频仍缺','artifact_ids':['art-a','art-b'],'task_status':'partial'}) is None


def test_self_photos_to_send_do_not_invent_a_publisher_restriction(tmp_path):
    goal=control(tmp_path,'找某人本人的cos照片发给我')
    goal.describe({'media_selection':{'publisher_scope':'subject_only'}})
    assert goal.publisher_proof is None and goal.media_selection['publisher_scope']=='any'


def test_explicit_publisher_handle_cannot_be_changed_by_model(tmp_path):
    goal=control(tmp_path,'从X找 @synthetic 本人发布的近7天图片')
    goal.describe({'media_selection':{'publisher_scope':'any','publisher_account':'different'}})
    assert goal.media_selection['publisher_scope']=='subject_only'
    assert goal.media_selection['publisher_account']=='synthetic'


@pytest.mark.parametrize('query,days',[('下载近七天图片',7),('找过去两周的图',14),('最近十二天发的图片',12)])
def test_chinese_explicit_date_is_a_user_bound(tmp_path,query,days):
    goal=control(tmp_path,query)
    assert goal.explicit_recent_days==days
    assert goal.publication_filter['provenance']=='explicit_user'
