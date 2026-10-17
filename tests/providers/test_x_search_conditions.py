from datetime import timedelta

import pytest

from app.providers.x_posts import XSearchOptions, compile_search_targets, search_target
from app.providers.x_crawler import CrawlerError, network_budget, project_gallery
from tests.providers.test_x_crawler import NOW, row, crawler


def test_query_pushes_window_types_and_explicit_thresholds_without_publisher_limit():
    options = XSearchOptions.parse(media_types=['video', 'image'], sort='top', min_likes=30)
    targets = compile_search_targets('name OR alias', options, NOW, 7)
    assert len(targets) == 2
    assert all('(name OR alias)' in t.identifier and 'since:2026-10-09' in t.identifier
        and 'until:2026-10-17' in t.identifier and 'min_faves:30' in t.identifier
        and 'from:' not in t.identifier and t.sort == 'top' for _, t in targets)
    assert 'filter:images' in targets[0][1].identifier
    assert 'filter:native_video' in targets[1][1].identifier


@pytest.mark.parametrize('values', [dict(sort='invalid'), dict(min_likes=True),
    dict(min_replies=-1), dict(media_types=['gif']), dict(media_types=[]),
    dict(media_types=['image', 'image'])])
def test_invalid_conditions_are_rejected(values):
    with pytest.raises(ValueError): XSearchOptions.parse(**values)


def test_both_types_reserve_candidates_and_share_one_request_budget():
    calls=[]
    def fetch(target, cookies, **kwargs):
        calls.append((target, kwargs))
        if 'filter:native_video' in target.identifier:
            return [row(post_id='900000000000000002', source_url='https://x.com/sample/status/900000000000000002',
                published_at=(NOW-timedelta(days=1)).isoformat(),
                media=[{'kind':'video','url':'https://video.twimg.com/synthetic.mp4'}])]
        return [row(), row(post_id='900000000000000003',
            source_url='https://x.com/sample/status/900000000000000003')]
    result=crawler(fetch).acquire('search',search_target('subject').url,
        limit=2,media_types=['image','video'])
    assert [r['post_id'] for r in result['posts']]==['900000000000000001','900000000000000002']
    assert calls[0][1]['budget'] is calls[1][1]['budget']
    assert result['media_type_counts']=={'image':1,'video':1}


def test_top_keeps_native_order_and_rechecks_dates_types_and_thresholds():
    older=row(published_at=(NOW-timedelta(days=1)).isoformat(), engagement={'likes':50})
    newer=row(post_id='900000000000000002', source_url='https://x.com/sample/status/900000000000000002', engagement={'likes':40})
    unknown=row(post_id='900000000000000003', source_url='https://x.com/sample/status/900000000000000003')
    result=crawler(lambda *a,**k:[older,newer,unknown,row(published_at=(NOW-timedelta(days=20)).isoformat())]).acquire(
        'search', search_target('subject').url,sort='top',min_likes=30,media_types=['image'],lookback_days=7)
    assert [r['post_id'] for r in result['posts']]==[older['post_id'],newer['post_id']]
    assert result['excluded_by_conditions']==1 and result['excluded_by_window']==1
    assert result['search_conditions']['sort']=='top' and not result['latest_proven']


def test_mixed_deduplicates_root_posts_and_reports_missing_type():
    result=crawler(lambda *a,**k:[row()]).acquire('search',search_target('subject').url,
        media_types=['image','video'])
    assert len(result['posts'])==1
    assert result['missing_media_types']==['video'] and result['status']=='partial'


def test_terminal_failure_in_second_branch_discards_partial_results_without_fallback():
    def fetch(target,*a,**kw):
        if 'filter:native_video' in target.identifier:raise CrawlerError('rate_limited')
        return [row()]
    def forbidden(*a,**kw):raise AssertionError('must not fall back')
    with pytest.raises(CrawlerError,match='rate_limited'):
        crawler(fetch,forbidden).acquire('search',search_target('subject').url,media_types=['image','video'])


def test_shared_budget_is_not_reset_between_type_queries(monkeypatch):
    import time
    import requests
    def send(client,request,**kwargs):
        response=requests.Response();response.status_code=200;response._content=b'{}';response._content_consumed=True
        return response
    monkeypatch.setattr(requests.Session,'send',send)
    state={'count':0,'status':0}
    for _ in range(2):
        with network_budget(time.monotonic()+30,'http://127.0.0.1:18797',state):
            client=requests.Session()
            for _ in range(9):client.get('https://x.com/')
    with network_budget(time.monotonic()+30,'http://127.0.0.1:18797',state):
        with pytest.raises(CrawlerError,match='request_budget'):requests.get('https://x.com/')


def test_both_metadata_adapters_project_counts_and_keep_unknown_values():
    from tests.providers.test_x_posts import tweet
    from app.providers.x_posts import parse_api_posts
    raw=tweet();raw['legacy'].update(favorite_count=35,reply_count=True,retweet_count=-1)
    assert parse_api_posts(raw)[0]['engagement']=={'likes':35,'replies':None,'reposts':None}
    data={'tweet_id':row()['post_id'],'author':{'name':'sample','protected':False},'date':NOW,
        'count':0,'favorite_count':35,'retweet_count':10}
    assert project_gallery(data,[])['engagement']=={'likes':35,'replies':None,'reposts':10}


def test_platform_top_unverified_subject_is_flagged_and_ranked_after_matching_candidates():
    unrelated=row(text='enoshima cosplay',author='other',post_id='900000000000000002',
        source_url='https://x.com/other/status/900000000000000002')
    subject=row(text='sample at TGS stage')
    search=crawler(lambda *a,**k:[unrelated,subject])
    strict=search.acquire('search',search_target('sample cosplay').url,
        sort='top',subject_terms=['sample'],topic_terms=['cosplay'])
    assert strict['posts'][0]['text']==subject['text']
    assert strict['posts'][0]['lexical_match']=={'subject':True,'topic':False}
    assert strict['posts'][1]['lexical_match']=={'subject':False,'topic':True}
    assert strict['subject_unverified_count']==1
    review=search.acquire('search',search_target('sample').url,
        subject_terms=['sample'],topic_terms=['cosplay'],topic_match='review')
    assert len(review['posts'])==2 and review['posts'][0]['lexical_match']['subject'] is True


def test_alias_word_boundaries_and_unicode_are_conservative():
    from app.providers.x_posts import matches_terms
    assert not matches_terms('fake_enako_sample',('enako',))
    assert matches_terms('えなこさんの写真',('えなこ',))
    assert matches_terms('Photo by @sample',('@sample',))


def test_top_url_is_respected_when_no_separate_sort_override_is_given():
    seen=[]
    def fetch(target,*a,**kw):seen.append(target.sort);return [row()]
    result=crawler(fetch).acquire('search',search_target('subject','top').url)
    assert seen==['top'] and result['search_conditions']['sort']=='top'


def test_four_topic_terms_preserve_query_facets_and_five_are_rejected():
    options=XSearchOptions.parse(subject_terms=['sample'],topic_terms=['space','stars','galaxy','planets'])
    assert options.topic_terms==('space','stars','galaxy','planets')
    with pytest.raises(ValueError):XSearchOptions.parse(topic_terms=['a','b','c','d','e'])
    with pytest.raises(ValueError):XSearchOptions.parse(subject_terms=['a','b','c','d'])
