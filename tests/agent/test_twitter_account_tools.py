from datetime import UTC, datetime
import json
from types import SimpleNamespace

import pytest

from app.agent.research_tasks import ResearchTaskWorkspace
from app.agent.shared_links import wrap_share_web_tools
from app.agent.web_tools import build_web_tools
from app.agent.request_task import RequestTaskControl
from app.agent.tool_agent import ToolOutcome
from tests.agent.test_web_tools import Client, Guard, Runner

URL='https://x.com/synthetic/status/900000000000000001'
NOW=datetime(2026,10,16,tzinfo=UTC)


def workspace(tmp_path,request='下载synthetic最新的cos图片发我'):
    return ResearchTaskWorkspace(tmp_path,scope='private:900000101',actor_id=900000101,
        source_id=10,request=request,valid_source=lambda i:i==10,clock=lambda:NOW.timestamp())


def result(date='2026-10-15T08:00:00+00:00'):
    return {'account':'synthetic','status':'ok','posts':[{'source_url':URL,'post_id':'900000000000000001',
        'author':'synthetic','published_at':date,'text':'作品'*500,'identity_status':'matched',
        'date_status':'source_reported','author_status':'source_reported','media_count':2,'media_roles':['post_image']}],
        'source_url':'https://syndication.twitter.com/srv/timeline-profile/screen-name/synthetic',
        'observed_at':NOW.isoformat(),'latest_proven':False,'complete':False,
        'coverage':'public_embedded_account_entries_only; latest_not_proven'}


async def test_account_discovery_is_compact_archived_and_cached(tmp_path,monkeypatch):
    calls=[]
    def fetch(self,account,**kwargs):calls.append(account);return result()
    monkeypatch.setattr('app.agent.twitter_accounts.TwitterTimelineClient.list_posts',fetch)
    w=workspace(tmp_path)
    tools={t.name:t for t in build_web_tools(Client(),settings=SimpleNamespace(),runner=Runner(),guard=Guard(),workspace=w)}
    first=await tools['list_x_posts'].handler({'account':'@synthetic'})
    second=await tools['list_x_posts'].handler({'account':'synthetic'})
    assert calls==['synthetic']
    assert first.data['resources'][0]['published_at'].startswith('2026-10-15')
    assert len(json.dumps(first.data,ensure_ascii=False))<3000
    assert '作品'*500 not in json.dumps(first.data,ensure_ascii=False)
    assert '作品'*500 in w.read_evidence(first.data['evidence_id'])['content']
    assert second.data['cached_in_turn'] is True and second.data['evidence_id']==first.data['evidence_id']


async def test_account_discovery_needs_current_web_grant(monkeypatch):
    calls=[]
    monkeypatch.setattr('app.agent.twitter_accounts.TwitterTimelineClient.list_posts',lambda *a,**k:calls.append(1))
    access=SimpleNamespace(web_allowed=lambda:False,known_card_target_allowed=lambda *a:True)
    tools=wrap_share_web_tools(build_web_tools(Client(),settings=SimpleNamespace(),runner=Runner(),guard=Guard()),access)
    outcome=await next(t for t in tools if t.name=='list_x_posts').handler({'account':'synthetic'})
    assert outcome.data['error']=='web_requires_explicit_request' and not calls


async def test_semantic_query_reordered_case_and_duplicate_terms_reuse_evidence(tmp_path,monkeypatch):
    calls=[]
    def fetch(self,query,**kwargs):calls.append((query,kwargs));return result()
    monkeypatch.setattr('app.providers.x_crawler_client.XCrawlerClient.search_posts',fetch)
    settings=SimpleNamespace(x_crawler_session_file='/run/synthetic-session.json',x_crawler_proxy_url='http://127.0.0.1:18797')
    tools=build_web_tools(Client(),settings=settings,runner=Runner(),guard=Guard(),workspace=workspace(tmp_path))
    search=next(t for t in tools if t.name=='search_x_posts')
    args={'subject_terms':['Synthetic','別名','synthetic'],'topic_terms':['写真','Photo','photo'],
        'media_types':['video','image']}
    first=await search.handler(args)
    second=await search.handler({**args,'subject_terms':['別名','synthetic'],
        'topic_terms':['photo','写真'],'media_types':['image','video']})
    assert first.status==second.status=='ok' and len(calls)==1
    assert second.data['cached_in_turn'] and second.data['evidence_id']==first.data['evidence_id']
    assert calls[0][0].casefold().count('synthetic')==1
    for changed in ({'lookback_days':30},{'sort':'top'},{'min_likes':5},
            {'topic_terms':['different']}):
        await search.handler({**args,**changed})
    assert len(calls)==4  # Four actual queries; the duplicate consumed no budget.


async def test_profile_shared_without_quote_cannot_bypass_card_grant(monkeypatch):
    access=SimpleNamespace(web_allowed=lambda:True,known_card_target_allowed=lambda target,sources:False)
    tools=wrap_share_web_tools(build_web_tools(Client(),settings=SimpleNamespace(),runner=Runner(),guard=Guard()),access)
    outcome=await next(t for t in tools if t.name=='list_x_posts').handler({'account':'synthetic'})
    assert outcome.data['error']=='share_requires_quoted_request'


@pytest.mark.parametrize('date',['2025-08-26T08:41:21Z','','not-a-date','2026-10-23T01:00:00Z'])
def test_latest_download_cannot_use_old_unknown_or_future_post(tmp_path,date):
    control=RequestTaskControl(workspace(tmp_path,'下载近7天的图片'),capability_version='synthetic')
    data=result(date)
    data['resources']=data.pop('posts');data['content_kind']='account_posts'
    control.observe('list_x_posts',{'account':'synthetic'},ToolOutcome(data),'list')
    rejected=control.preflight('download_twitter_media',{'ref_id':URL,'delivery_mode':'auto'})
    assert rejected is not None and rejected.data['error']=='resource_freshness_unverified'
    assert '不得' in rejected.data['guidance']


def test_latest_download_accepts_verified_recent_metadata_and_batch_is_atomic(tmp_path):
    control=RequestTaskControl(workspace(tmp_path,'下载近7天的图片'),capability_version='synthetic')
    data=result();data['resources']=data.pop('posts');data['content_kind']='account_posts'
    control.observe('list_x_posts',{'account':'synthetic'},ToolOutcome(data),'list')
    assert control.preflight('download_twitter_media',{'ref_id':URL}) is None
    invalid=control.preflight('queue_media_batch',{'operations':[
        {'operation':'download_twitter_media','arguments':{'ref_id':URL}},
        {'operation':'download_twitter_media','arguments':{'ref_id':'https://x.com/synthetic/status/900000000000000002'}}]})
    assert invalid is not None and invalid.data['error']=='resource_freshness_unverified'


def test_explicit_old_post_request_does_not_require_recent_date(tmp_path):
    control=RequestTaskControl(workspace(tmp_path,'把这个2025年的帖子图片下载发给我'),capability_version='synthetic')
    assert control.published_after.year==2025
    row=result('2025-08-26T08:00:00Z');row['resources']=row.pop('posts')
    control.observe('list_x_posts',{},ToolOutcome(row),'year-evidence')
    assert control.preflight('download_twitter_media',{'ref_id':URL}) is None


@pytest.mark.parametrize('describe_media', [False, True])
def test_latest_constraint_is_preserved_when_media_delivery_is_selected_later(tmp_path, describe_media):
    control=RequestTaskControl(workspace(tmp_path,'帮我扒synthetic近7天的内容'),capability_version='synthetic')
    assert not control.requires_media
    if describe_media:
        control.describe({'delivery':'media','goal':'下载资源'})
    rejected=control.preflight('download_twitter_media',{'ref_id':URL})
    assert rejected is not None and rejected.data['error']=='resource_freshness_unverified'


async def test_topic_search_keeps_subject_independent_of_publisher_and_is_source_gated(tmp_path,monkeypatch):
    calls=[]
    def fetch(self,query,**kwargs):
        calls.append(query)
        data=result();data['posts'][0]['author']='photographer'
        data['posts'][0]['source_url']=URL.replace('synthetic/','photographer/')
        data['source_url']='https://x.com/search?q=subject&f=live'
        return data
    monkeypatch.setattr('app.providers.x_crawler_client.XCrawlerClient.search_posts',fetch)
    settings=SimpleNamespace(x_crawler_session_file='/run/synthetic-session.json',x_crawler_proxy_url='http://127.0.0.1:18797')
    w=workspace(tmp_path,'原样X查询 subject cosplay filter:media，下载一些关于某人的近期cos图片')
    access=SimpleNamespace(web_allowed=lambda:True,known_card_target_allowed=lambda *a:True)
    tools=wrap_share_web_tools(build_web_tools(Client(),settings=settings,runner=Runner(),guard=Guard(),workspace=w),access)
    search=next(t for t in tools if t.name=='search_x_posts')
    output=await search.handler({'mode':'raw','query':'subject cosplay filter:media'})
    assert output.status=='ok' and output.data['resources'][0]['author']=='photographer'
    goal=RequestTaskControl(w,capability_version='synthetic')
    goal.observe('search_x_posts',{},output,'search')
    assert goal.preflight('download_twitter_media',{'ref_id':output.data['resources'][0]['source_url'],
        'delivery_mode':'auto'}) is None
    access.web_allowed=lambda:False
    assert (await search.handler({'mode':'raw','query':'second'})).data['error']=='web_requires_explicit_request'
    assert len(calls)==1


async def test_observed_post_id_resolves_for_preflight_and_real_download_handler(tmp_path,monkeypatch):
    from app.agent.tool_agent import AgentTool
    seen=[]
    monkeypatch.setattr('app.agent.twitter_accounts.TwitterTimelineClient.list_posts',lambda *a,**k:result())
    async def download(args):seen.append(args['ref_id']);return ToolOutcome({})
    extra=AgentTool('download_twitter_media','synthetic',{'type':'object'},download)
    access=SimpleNamespace(web_allowed=lambda:True,known_card_target_allowed=lambda *a:True)
    w=workspace(tmp_path)
    tools=wrap_share_web_tools((*build_web_tools(Client(),settings=SimpleNamespace(),runner=Runner(),guard=Guard(),workspace=w),extra),access)
    data=await next(t for t in tools if t.name=='list_x_posts').handler({'account':'synthetic'})
    goal=RequestTaskControl(w,capability_version='synthetic')
    goal.observe('list_x_posts',{},data,'list')
    identifier=data.data['resources'][0]['post_id']
    assert goal.preflight('download_twitter_media',{'ref_id':identifier}) is None
    await next(t for t in tools if t.name=='download_twitter_media').handler({'ref_id':identifier})
    assert seen==[URL]


def test_unspecified_recent_window_can_be_interpreted_but_explicit_window_is_locked(tmp_path):
    row=result('2026-10-01T08:00:00Z');row['resources']=row.pop('posts')
    loose=RequestTaskControl(workspace(tmp_path/'loose'),capability_version='synthetic')
    loose.observe('list_x_posts',{},ToolOutcome(row),'list')
    assert loose.preflight('download_twitter_media',{'ref_id':URL}) is None
    loose.describe({'goal':'近30天相关媒体，说明实际日期','delivery':'media','requirements':[],
        'media_selection':{'lookback_days':30}})
    assert loose.preflight('download_twitter_media',{'ref_id':URL}) is None
    strict=RequestTaskControl(workspace(tmp_path/'strict','下载近3天的图片'),capability_version='synthetic')
    strict.describe({'goal':'试图放宽','delivery':'media','requirements':[],
        'media_selection':{'lookback_days':30}})
    strict.observe('list_x_posts',{},ToolOutcome(row),'list')
    assert strict.media_selection['lookback_days']==3
    assert strict.preflight('download_twitter_media',{'ref_id':URL}).status=='blocked'


@pytest.mark.parametrize('excluded',[0,3])
async def test_empty_search_requeries_when_server_window_changes(tmp_path,monkeypatch,excluded):
    calls=[]
    def fetch(self,query,**kwargs):
        calls.append((query,kwargs['lookback_days']))
        return {**result(),'posts':[],'excluded_by_window':excluded,'query':query}
    monkeypatch.setattr('app.providers.x_crawler_client.XCrawlerClient.search_posts',fetch)
    settings=SimpleNamespace(x_crawler_session_file='/run/synthetic-session.json',x_crawler_proxy_url='http://127.0.0.1:18797')
    w=workspace(tmp_path,'原样查询 ambiguous name');w.request_control={'media_selection':{'lookback_days':30}}
    tools=build_web_tools(Client(),settings=settings,runner=Runner(),guard=Guard(),workspace=w)
    search=next(t for t in tools if t.name=='search_x_posts')
    first=await search.handler({'mode':'raw','query':'ambiguous name'})
    again=await search.handler({'mode':'raw','query':'ambiguous name','lookback_days':60})
    assert first.data['empty_reason']==('outside_selected_window' if excluded else 'no_visible_candidates')
    assert not again.data.get('reused_empty_visible_query')
    assert calls==[('ambiguous name',30),('ambiguous name',60)]


async def test_structured_conditions_flow_and_cache_include_every_condition(tmp_path,monkeypatch):
    calls=[]
    def fetch(self,query,**kwargs):calls.append(kwargs);return result()
    monkeypatch.setattr('app.providers.x_crawler_client.XCrawlerClient.search_posts',fetch)
    settings=SimpleNamespace(x_crawler_session_file=str(tmp_path/'session.json'),x_crawler_proxy_url='http://127.0.0.1:18797')
    tools=build_web_tools(Client(),settings=settings,runner=Runner(),guard=Guard(),workspace=workspace(tmp_path))
    search=next(t for t in tools if t.name=='search_x_posts')
    base={'subject_terms':['sample'],'topic_terms':['cosplay'],'media_types':['image','video']}
    await search.handler(base)
    assert (await search.handler({**base,'media_types':['video','image']})).data['cached_in_turn']
    await search.handler({**base,'sort':'top'})
    await search.handler({**base,'min_likes':100})
    assert len(calls)==3 and calls[0]['media_types']==['image','video']
    assert calls[0]['min_likes'] is None and calls[1]['sort']=='top' and calls[2]['min_likes']==100
    assert (await search.handler({**base,'limit':1})).status=='blocked'
    assert (await search.handler({'mode':'raw','query':'from:sample'})).data['error']=='raw_x_query_requires_user_source'


def test_declared_discovery_window_updates_unspecified_goal_but_not_explicit_limit(tmp_path):
    loose=RequestTaskControl(workspace(tmp_path/'loose'),capability_version='synthetic')
    loose.observe('search_x_posts',{'lookback_days':30},ToolOutcome({'resources':[]}), 'search')
    assert loose.media_selection['lookback_days']==30
    strict=RequestTaskControl(workspace(tmp_path/'strict','下载近3天图片'),capability_version='synthetic')
    strict.observe('search_x_posts',{'lookback_days':30},ToolOutcome({'resources':[]}), 'search')
    assert strict.media_selection['lookback_days']==3


def test_topic_review_mode_broadens_recall_without_adding_a_publisher_filter():
    from app.agent.twitter_accounts import media_query
    assert media_query(['sample','サンプル'],['cosplay'],'review')=='(sample OR サンプル) filter:media'
    with pytest.raises(ValueError):media_query(['sample'],[],'review')


@pytest.mark.parametrize('topic_match',['keywords','review'])
async def test_four_topic_terms_are_accepted_by_schema_and_reach_search(tmp_path,monkeypatch,topic_match):
    calls=[]
    def fetch(self,query,**kwargs):calls.append((query,kwargs));return result()
    monkeypatch.setattr('app.providers.x_crawler_client.XCrawlerClient.search_posts',fetch)
    settings=SimpleNamespace(x_crawler_session_file=str(tmp_path/'session.json'),x_crawler_proxy_url='http://127.0.0.1:18797')
    search=next(t for t in build_web_tools(Client(),settings=settings,runner=Runner(),guard=Guard(),workspace=workspace(tmp_path)) if t.name=='search_x_posts')
    terms=['space','astronomy','stars','planets']
    args={'subject_terms':['synthetic'],'topic_terms':terms,'media_types':['video'],'topic_match':topic_match}
    from app.agent.tool_agent import validate_arguments
    assert validate_arguments(args,search.parameters)
    output=await search.handler(args)
    assert output.status=='ok' and calls[0][1]['topic_terms']==terms
    assert ('(space OR astronomy OR stars OR planets)' in calls[0][0]) is (topic_match=='keywords')
    validate_limit=search.parameters['properties']['topic_terms']['maxItems']
    assert validate_limit==4 and search.parameters['properties']['subject_terms']['maxItems']==3
    rejected=await search.handler({**args,'topic_terms':terms+['galaxy']})
    assert not validate_arguments({**args,'topic_terms':terms+['galaxy']},search.parameters)
    assert rejected.status=='blocked' and len(calls)==1


async def test_maximum_length_semantic_terms_do_not_hit_raw_query_length_limit(tmp_path,monkeypatch):
    calls=[]
    def fetch(self,query,**kwargs):calls.append(query);return result()
    monkeypatch.setattr('app.providers.x_crawler_client.XCrawlerClient.search_posts',fetch)
    settings=SimpleNamespace(x_crawler_session_file=str(tmp_path/'session.json'),x_crawler_proxy_url='http://127.0.0.1:18797')
    search=next(t for t in build_web_tools(Client(),settings=settings,runner=Runner(),guard=Guard(),workspace=workspace(tmp_path)) if t.name=='search_x_posts')
    output=await search.handler({'subject_terms':[v*40 for v in ('甲','乙','丙')],
        'topic_terms':[v*40 for v in ('丁','戊','己','庚')]})
    assert output.status=='ok' and 300<len(calls[0])<=400


@pytest.mark.parametrize('excluded',[0,2])
async def test_empty_search_distinguishes_unverified_visibility_from_date_exclusion(tmp_path,monkeypatch,excluded):
    calls=[]
    def fetch(self,query,**kwargs):
        calls.append(query)
        return {**result(),'posts':[],'excluded_by_window':excluded}
    monkeypatch.setattr('app.providers.x_crawler_client.XCrawlerClient.search_posts',fetch)
    settings=SimpleNamespace(x_crawler_session_file=str(tmp_path/'session.json'),x_crawler_proxy_url='http://127.0.0.1:18797')
    search=next(t for t in build_web_tools(Client(),settings=settings,runner=Runner(),guard=Guard(),workspace=workspace(tmp_path)) if t.name=='search_x_posts')
    args={'subject_terms':['synthetic'],'media_types':['video']}
    first=await search.handler(args)
    cached=await search.handler(args)
    assert first.data['returned_count']==0 and len(calls)==1
    if not excluded:
        assert first.data['search_visibility']==cached.data['search_visibility']=='not_verified'
        assert '搜索过滤' in first.data['guidance'] and '未读取账号设置' in first.data['guidance']
        assert '尚未启动下载' in first.data['guidance'] and '不证明名字错误' in first.data['guidance']
    else:
        assert first.data['empty_reason']==cached.data['empty_reason']=='outside_selected_window'
        assert 'search_visibility' not in first.data and 'search_visibility' not in cached.data


@pytest.mark.parametrize('mode',['keywords','review'])
async def test_subject_and_video_search_does_not_require_invented_topic_terms(tmp_path,monkeypatch,mode):
    calls=[]
    def fetch(self,query,**kwargs):calls.append((query,kwargs));return result()
    monkeypatch.setattr('app.providers.x_crawler_client.XCrawlerClient.search_posts',fetch)
    settings=SimpleNamespace(x_crawler_session_file=str(tmp_path/'session.json'),x_crawler_proxy_url='http://127.0.0.1:18797')
    search=next(t for t in build_web_tools(Client(),settings=settings,runner=Runner(),guard=Guard(),workspace=workspace(tmp_path)) if t.name=='search_x_posts')
    output=await search.handler({'subject_terms':['sample'],'media_types':['video'],'topic_match':mode})
    assert output.status=='ok' and calls[0][0]=='(sample) filter:media'
    assert calls[0][1]['media_types']==['video'] and calls[0][1]['topic_terms'] is None


async def test_compact_result_preserves_only_video_when_last_by_date(tmp_path,monkeypatch):
    def fetch(self,query,**kwargs):
        posts=[{**result()['posts'][0],'post_id':str(900000000000000001+i),
            'source_url':f'https://x.com/synthetic/status/{900000000000000001+i}',
            'extra':'x'*450} for i in range(12)]
        posts[-1]['media_roles']=['post_video']
        return {**result(),'posts':posts,'missing_media_types':[]}
    monkeypatch.setattr('app.providers.x_crawler_client.XCrawlerClient.search_posts',fetch)
    settings=SimpleNamespace(x_crawler_session_file=str(tmp_path/'session.json'),x_crawler_proxy_url='http://127.0.0.1:18797')
    search=next(t for t in build_web_tools(Client(),settings=settings,runner=Runner(),guard=Guard(),workspace=workspace(tmp_path))
        if t.name=='search_x_posts')
    output=await search.handler({'subject_terms':['sample'],'topic_terms':['cosplay'],'media_types':['image','video']})
    assert output.data['truncated'] and not output.data['missing_media_types']
    assert output.data['media_type_counts']['video']==1


async def test_rate_limit_stops_different_queries_in_same_turn(tmp_path,monkeypatch):
    from app.providers.twitter_media import TwitterMediaError
    calls=[]
    def fetch(self,query,**kwargs):calls.append(query);raise TwitterMediaError('x_crawler_rate_limited')
    monkeypatch.setattr('app.providers.x_crawler_client.XCrawlerClient.search_posts',fetch)
    settings=SimpleNamespace(x_crawler_session_file=str(tmp_path/'session.json'),x_crawler_proxy_url='http://127.0.0.1:18797')
    search=next(t for t in build_web_tools(Client(),settings=settings,runner=Runner(),guard=Guard(),workspace=workspace(tmp_path))
        if t.name=='search_x_posts')
    args={'subject_terms':['sample'],'topic_terms':['cosplay']}
    assert (await search.handler(args)).data['error']=='x_crawler_rate_limited'
    assert (await search.handler({**args,'sort':'top','lookback_days':30})).data['stopped_in_turn']
    assert len(calls)==1


async def test_stale_keyword_results_recommend_same_window_subject_review(tmp_path,monkeypatch):
    def fetch(self,query,**kwargs):return {**result('2026-10-05T09:00:00+00:00'),'newest_visible_at':'2026-10-05T09:00:00+00:00'}
    monkeypatch.setattr('app.providers.x_crawler_client.XCrawlerClient.search_posts',fetch)
    settings=SimpleNamespace(x_crawler_session_file=str(tmp_path/'session.json'),x_crawler_proxy_url='http://127.0.0.1:18797')
    search=next(t for t in build_web_tools(Client(),settings=settings,runner=Runner(),guard=Guard(),workspace=workspace(tmp_path))
        if t.name=='search_x_posts')
    args={'subject_terms':['sample'],'topic_terms':['cosplay'],'lookback_days':30}
    output=await search.handler(args)
    assert output.data['freshness_gap']=='keyword_results_older_than_7_days'
    assert '同一时间窗口' in output.data['guidance'] and 'topic_match=review' in output.data['guidance']
    assert 'freshness_gap' not in (await search.handler({**args,'sort':'top'})).data


async def test_semantic_search_combines_subject_aliases_and_topic_synonyms(tmp_path,monkeypatch):
    calls=[]
    def fetch(self,query,**kwargs):calls.append(query);return result()
    monkeypatch.setattr('app.providers.x_crawler_client.XCrawlerClient.search_posts',fetch)
    settings=SimpleNamespace(x_crawler_session_file='/run/synthetic-session.json',x_crawler_proxy_url='http://127.0.0.1:18797')
    tools=build_web_tools(Client(),settings=settings,runner=Runner(),guard=Guard(),workspace=workspace(tmp_path))
    search=next(t for t in tools if t.name=='search_x_posts')
    wrong=await search.handler({'query':'name only'})
    assert wrong.data['error']=='structured_media_query_required' and not calls
    output=await search.handler({'subject_terms':['sample','サンプル','sample_cos'],
        'topic_terms':['cosplay','コスプレ']})
    assert output.status=='ok'
    assert calls==['(sample OR サンプル OR sample_cos) (cosplay OR コスプレ) filter:media']
