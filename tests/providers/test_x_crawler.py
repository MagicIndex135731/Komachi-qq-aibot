from datetime import UTC, datetime
from types import SimpleNamespace
import json
import time

import pytest

from app.providers.x_crawler import XCrawler, CrawlerError, project_gallery, load_session, network_budget, _classify
from app.providers.x_posts import parse_target, search_target

PID='900000000000000001'
URL=f'https://x.com/sample/status/{PID}'
NOW=datetime(2026,10,16,tzinfo=UTC)

def row(**changes):
    return {'post_id':PID,'source_url':URL,'author':'sample','published_at':NOW.isoformat(),
        'text':'synthetic cosplay','protected':False,'identity_status':'matched','is_reply':False,
        'media_count':1,'media_error':'','media':[{'kind':'photo','url':'https://pbs.twimg.com/media/synthetic.jpg'}],**changes}

def crawler(primary, fallback=lambda *a,**k: [], **kwargs):
    return XCrawler({'auth_token':'synthetic-session-token-value','ct0':'synthetic-csrf-token-value'},
        primary=primary,fallback=fallback,clock=lambda:NOW,**kwargs)

def test_primary_snapshot_keeps_identity_media_and_backend():
    result=crawler(lambda *a,**k:[row()]).acquire('inspect',URL)
    assert result['provider']=='x_crawler' and result['backend']=='gallery-dl'
    assert result['posts'][0]['evidence_ref'].startswith('x-posts-')
    assert result['posts'][0]['media_count']==1 and result['authenticated'] is True

def test_fallback_is_once_for_parse_failure_and_keeps_reason():
    calls=[]
    def primary(*a,**kw):raise CrawlerError('parse_failed')
    def fallback(*a,**kw):calls.append(1);return [row()]
    result=crawler(primary,fallback).acquire('inspect',URL)
    assert calls==[1] and result['backend']=='twscrape'
    assert result['fallback_reason']=='parse_failed'

@pytest.mark.parametrize('code',['login_required','challenge','rate_limited','access_denied','identity_mismatch'])
def test_authority_or_identity_failure_does_not_try_another_backend(code):
    def primary(*a,**kw):raise CrawlerError(code)
    def forbidden(*a,**kw):raise AssertionError('must not fall back')
    with pytest.raises(CrawlerError,match=code):crawler(primary,forbidden).acquire('inspect',URL)

@pytest.mark.parametrize('changes',[{'post_id':'900000000000000002'},
    {'source_url':'https://x.com/sample/status/900000000000000002'},
    {'media':[{'kind':'photo','url':'https://127.0.0.1/private'}]}])
def test_untrusted_wrong_identity_or_media_never_enters_snapshot(changes):
    with pytest.raises(CrawlerError):crawler(lambda *a,**k:[row(**changes)]).acquire('inspect',URL)

def test_protected_and_old_posts_are_excluded_without_losing_window_evidence():
    result=crawler(lambda *a,**k:[row(protected=True),row(published_at='2026-09-01T00:00:00+00:00')]).acquire('list','sample',lookback_days=7)
    assert not result['posts'] and result['excluded_by_window']==1
    assert result['complete'] is False and result['latest_proven'] is False

def test_gallery_projection_rejects_unknown_public_status_and_media_count_gaps():
    data={'tweet_id':PID,'author':{'name':'sample'},'date':NOW,'count':1,'content':'synthetic'}
    with pytest.raises(CrawlerError):project_gallery(data,[{'kind':'photo','url':'https://pbs.twimg.com/media/synthetic.jpg'}])
    data['author']['protected']=False
    with pytest.raises(CrawlerError):project_gallery(data,[])

def test_target_accepts_only_bounded_search_or_public_post_account():
    assert parse_target(URL).identifier==PID
    assert parse_target('@sample').kind=='account'
    target=search_target('subject cosplay')
    assert parse_target(target.url).identifier=='subject cosplay'
    assert parse_target('https://x.com/search?q=a&f=top').sort=='top'
    for url in ('https://x.com/messages','https://x.com/search?q=a&f=unknown','https://evil.example/sample'):
        with pytest.raises(ValueError):parse_target(url)


def test_session_only_loads_two_fields_and_requires_private_regular_file(tmp_path):
    path=tmp_path/'session.json'
    path.write_text(json.dumps({'auth_token':'synthetic-token-value-for-test','ct0':'synthetic-csrf-value-for-test','irrelevant':'discard'}))
    path.chmod(0o600)
    assert set(load_session(path))=={'auth_token','ct0'}
    path.chmod(0o644)
    with pytest.raises(CrawlerError,match='credentials_invalid'):load_session(path)
    path.chmod(0o600)
    link=tmp_path/'linked.json';link.symlink_to(path)
    with pytest.raises(CrawlerError,match='credentials_invalid'):load_session(link)


def test_library_challenge_is_terminal_and_does_not_become_parse_failure():
    error=type('ChallengeError',(Exception,),{})('synthetic')
    assert _classify(error).code=='challenge'


def test_stock_requests_are_read_only_bounded_and_forced_to_x_proxy(monkeypatch):
    import requests
    calls=[]
    def fake_send(client,request,**kwargs):
        calls.append(kwargs)
        response=requests.Response();response.status_code=200
        response._content=b'{}';response._content_consumed=True
        return response
    monkeypatch.setattr(requests.Session,'send',fake_send)
    with network_budget(time.monotonic()+30,'http://127.0.0.1:18797'):
        client=requests.Session()
        for url,method in [('https://x.com/i/api/graphql/hash/CreateTweet','GET'),
            ('https://x.com/i/api/graphql/hash/SearchTimeline','POST'),('https://evil.example/','GET')]:
            with pytest.raises(CrawlerError,match='read_only_boundary'):
                client.send(requests.Request(method,url).prepare())
        for _ in range(18):client.send(requests.Request('GET','https://x.com/i/api/graphql/hash/SearchTimeline').prepare())
        with pytest.raises(CrawlerError,match='request_budget'):
            client.send(requests.Request('GET','https://x.com/').prepare())
    assert len(calls)==18 and all(c['timeout']<=8 and c['proxies']['https']=='http://127.0.0.1:18797' for c in calls)


async def test_twscrape_httpx_reads_are_bounded_before_parser(monkeypatch):
    import httpx
    async def fake_send(client,request):return httpx.Response(200,content=b'x'*(8*1024*1024+1),request=request)
    monkeypatch.setattr(httpx.AsyncClient,'_send_single_request',fake_send)
    with network_budget(time.monotonic()+30,'http://127.0.0.1:18797'):
        async with httpx.AsyncClient(trust_env=False) as client:
            with pytest.raises(CrawlerError,match='read_only_boundary'):
                await client.get('https://evil.example/')
            with pytest.raises(CrawlerError,match='response_too_large'):
                await client.get('https://x.com/i/api/graphql/hash/SearchTimeline')
