"""Current-request completion and recovery, independent of formal research mode."""
from types import SimpleNamespace

import pytest

from app.agent.research_tasks import ResearchTaskWorkspace
from app.agent.tool_agent import ToolOutcome
from app.agent.request_task import RequestTaskControl


def control(tmp_path, request='把这个账号最近的图片下载发给我', **kwargs):
    workspace=ResearchTaskWorkspace(tmp_path, scope='private:900000101',actor_id=900000101,
        source_id=10,request=request,valid_source=lambda value:value==10)
    return RequestTaskControl(workspace, capability_version=kwargs.pop('capability_version','test-v1'), **kwargs)


def test_media_goal_exists_without_formal_task_and_text_is_not_media_delivery(tmp_path):
    goal=control(tmp_path)
    assert not goal.workspace.formal
    result=goal.preflight('send_private',{'text':'看完了吧'})
    assert result.status=='blocked'
    assert 'media_delivery_unconfirmed' in result.data['missing']
    assert goal.state()['status']=='active'


@pytest.mark.parametrize('query',['我要看某人最近发的cos图','把网页里的照片搬过来',
    '想看这个作者刚上传的视频','Please download the photos and send them here'])
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
        assert not any(x.get('purpose')=='agent' for x in bridge.trace)
    finally:bridge.server.server_close();owner.close();client.http_client.close()


def test_failed_auxiliary_review_does_not_rotate_a_successful_native_conversation():
    from app.agent.context_recovery import context_reusable
    from app.agent.tool_agent import ToolTurnResult
    result=ToolTurnResult('sent','receipt','tool:send_private',trace=(
        {'status':'model_request','purpose':'agent','model_call_index':1,'outcome':'completed'},
        {'status':'model_request','purpose':'task_review','model_call_index':2,'outcome':'TimeoutError'}))
    assert context_reusable(result)
