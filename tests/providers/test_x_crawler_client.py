import json
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from app.providers.x_crawler_client import XCrawlerClient
from app.providers.twitter_media import TwitterMediaDownloader, TwitterMediaError
from test_x_posts import PID, URL, tweet
from app.providers.x_posts import make_snapshot, parse_target, parse_api_posts


def snapshot():
    return make_snapshot(parse_target(URL), parse_api_posts(tweet()), status='ok', backend='gallery-dl')


def client_with(data=None, error=None, observe=lambda *a: None):
    def run(args, **kw):
        observe(args, kw)
        return SimpleNamespace(returncode=0, stdout=json.dumps(
            {'ok': False, 'error': error} if error else {'ok': True, 'data': data or snapshot()}).encode())
    return XCrawlerClient('/run/synthetic-session.json', 'http://127.0.0.1:18797', runner=run)


def test_worker_uses_fixed_python_script_without_model_secrets(monkeypatch):
    monkeypatch.setenv('LLM_API_KEY', 'synthetic-do-not-inherit')
    calls = []
    def observe(args, kw):
        calls.append(args)
        assert args[-1].endswith('scripts/x_crawler_worker.py')
        assert kw['timeout'] == 45 and kw['capture_output'] is True
        assert 'LLM_API_KEY' not in kw['env'] and 'HTTP_PROXY' not in kw['env']
        assert kw['env']['X_CRAWLER_SESSION_FILE'] == '/run/synthetic-session.json'
        assert json.loads(kw['input'])['target'] == parse_target(URL).url
        assert 'synthetic-session' not in kw['input'].decode()
    row = client_with(observe=observe).inspect(URL)
    assert row['post_id'] == PID and len(calls) == 1


@pytest.mark.parametrize('proxy', ['https://example.com', 'http://localhost:18797',
    'http://127.0.0.1:18797/commands', 'http://user@127.0.0.1:18797'])
def test_no_remote_or_arbitrary_proxy(proxy):
    with pytest.raises(ValueError): XCrawlerClient('/run/synthetic-session.json', proxy)


def test_wrong_post_is_not_accepted():
    data = snapshot(); data['posts'][0]['post_id'] = '900000000000000009'
    with pytest.raises(TwitterMediaError, match='identity_mismatch'): client_with(data).inspect(URL)


@pytest.mark.parametrize('code,expected', [('login_required','login_required'), ('auth_token=synthetic-secret','unavailable')])
def test_failure_does_not_expose_upstream_details(code, expected):
    with pytest.raises(TwitterMediaError, match='x_crawler_' + expected): client_with(error=code).inspect(URL)


def test_worker_deadline_is_finite_and_slot_released():
    def timeout(*args, **kwargs): raise subprocess.TimeoutExpired('worker', 45)
    client = XCrawlerClient('/run/synthetic.json', 'http://127.0.0.1:18797', runner=timeout)
    with pytest.raises(TwitterMediaError, match='deadline_exceeded'): client.inspect(URL)
    assert client_with().inspect(URL)['post_id'] == PID


def test_guard_rechecked_after_child_completion():
    calls = []
    def check():
        calls.append(1)
        if len(calls) == 3: raise RuntimeError('source_superseded')
    with pytest.raises(RuntimeError, match='source_superseded'): client_with().inspect(URL, check=check)


def test_contending_worker_wait_is_bounded(monkeypatch):
    calls=[]
    monkeypatch.setattr(XCrawlerClient,'_slots',SimpleNamespace(acquire=lambda **kw:calls.append(kw) or False))
    with pytest.raises(TwitterMediaError,match='crawler_busy'):client_with().inspect(URL)
    assert calls==[{'timeout':8}]


async def test_configured_downloader_uses_crawler_without_public_resolver(tmp_path):
    calls = []
    def respond(request):
        calls.append(str(request.url)); assert request.url.host == 'pbs.twimg.com'
        assert 'cookie' not in request.headers
        return httpx.Response(200, content=b'\xff\xd8\xff' + b'synthetic', headers={'content-type': 'image/jpeg'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await TwitterMediaDownloader(client, metadata_client=client_with()).download(URL, tmp_path)
    assert result.manifest['resolver'] == 'x_crawler' and len(calls) == 1


def test_protected_tweets_are_never_projected():
    item = tweet(); item['core']['user_results']['result']['privacy'] = {'protected': True}
    assert parse_api_posts(item) == []


def test_search_carries_query_without_author_restriction():
    def observe(args, kw):
        data = json.loads(kw['input'])
        assert data['action'] == 'search' and data['target'].startswith('https://x.com/search?')
        assert data['media_types']==['image','video'] and data['sort']=='top' and data['min_replies']==30
    data = client_with(observe=observe).search_posts('subject cosplay',media_types=['image','video'],sort='top',min_replies=30)
    assert data['posts'][0]['author'] == 'sample' and 'media' not in data['posts'][0]


@pytest.mark.parametrize('alphabet',[('甲','乙','丙','丁','戊','己','庚'),tuple(chr(0x20000+i) for i in range(7))])
def test_maximum_multilingual_four_topics_roundtrip_through_worker(monkeypatch,alphabet,tmp_path):
    import io, logging, signal, sys
    from app.agent.twitter_accounts import media_query
    from app.providers.x_crawler import XCrawler
    from scripts import x_crawler_worker as worker
    from test_x_posts import tweet
    subjects=[c*40 for c in alphabet[:3]];topics=[c*40 for c in alphabet[3:]]
    calls=[];packets=[]
    def primary(target,cookies,**kwargs):calls.append(target);return parse_api_posts(tweet())
    crawler=XCrawler({'auth_token':'synthetic-session-token-value','ct0':'synthetic-csrf-token-value'},primary=primary)
    monkeypatch.setattr(worker,'load_session',lambda path:{'auth_token':'synthetic-session-token-value','ct0':'synthetic-csrf-token-value'})
    monkeypatch.setattr(worker,'XCrawler',lambda cookies:crawler)
    monkeypatch.setattr(signal,'alarm',lambda seconds:None,raising=False)
    monkeypatch.setattr(logging,'disable',lambda level:None)
    monkeypatch.setattr(worker.os,'umask',lambda mode:0)
    monkeypatch.setenv('X_CRAWLER_SESSION_FILE',str(tmp_path/'synthetic-session.json'))
    def run(args,**kwargs):
        packets.append(kwargs['input']);stdout=io.StringIO()
        with monkeypatch.context() as m:
            m.setattr(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(kwargs['input'])))
            m.setattr(sys,'stdout',stdout)
            worker.main()
        return SimpleNamespace(returncode=0,stdout=stdout.getvalue().encode())
    client=XCrawlerClient(str(tmp_path/'synthetic-session.json'),'http://127.0.0.1:18797',runner=run)
    result=client.search_posts(media_query(subjects,topics),subject_terms=subjects,topic_terms=topics,lookback_days=90)
    assert result['search_conditions']['topic_terms']==topics and len(calls)==1
    assert 4096<len(packets[0])<=8192 and len(calls[0].identifier)<=600


def test_worker_rejects_requests_over_8kib_before_loading_session(monkeypatch,capsys):
    import io, logging, signal, sys
    from scripts import x_crawler_worker as worker
    monkeypatch.setattr(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(b' '*8193)))
    monkeypatch.setattr(signal,'alarm',lambda seconds:None,raising=False)
    monkeypatch.setattr(logging,'disable',lambda level:None)
    monkeypatch.setattr(worker.os,'umask',lambda mode:0)
    monkeypatch.setattr(worker,'load_session',lambda path:pytest.fail('oversized input must fail before session access'))
    worker.main()
    assert json.loads(capsys.readouterr().out)=={'ok':False,'error':'parse_failed'}
