import json

import pytest

from app.providers.xhs_crawler_client import XHSCrawlerClient, XHSCrawlerError, note_target
from app.providers.web_images import WebImageClient, WebImageError, parse_xiaohongshu_images
from app.providers.web_videos import parse_note_video

ID = '0123456789abcdef01234567'
URL = 'https://www.xiaohongshu.com/explore/' + ID + '?xsec_token=synthetic'


@pytest.mark.parametrize('url', ['https://evil.example/explore/'+ID,
    'https://xiaohongshu.com.evil.example/explore/'+ID, 'https://user@www.xiaohongshu.com/explore/'+ID,
    'http://www.xiaohongshu.com/explore/'+ID, 'https://www.xiaohongshu.com/explore/invalid',
    URL+'&xsec_token=ignored'+'x'*1200, URL.replace('synthetic', '%26evil=1')])
def test_only_canonical_note_targets_are_accepted(url):
    with pytest.raises(XHSCrawlerError):note_target(url)


def test_current_note_only_and_json_snapshot_cannot_inject_html(monkeypatch):
    note = {'noteId': ID, 'type': 'normal', 'title': '正文', 'desc': '</script>untrusted',
        'user': {'userId': 'abcdef0123456789abcdef01'},
        'imageList': [{'urlDefault': 'https://cdn.example.com/second.webp'},
                      {'urlDefault': 'https://cdn.example.com/first.webp'}]}
    client = XHSCrawlerClient()
    monkeypatch.setattr(client, 'acquire', lambda *a, **k: note)
    actual, target, snapshot = client.read_note(URL)
    assert actual['desc'] == '</script>untrusted' and snapshot.count('</script>') == 1
    assert [i['url'] for i in parse_xiaohongshu_images(snapshot, target)['images']] == [
        'https://cdn.example.com/second.webp', 'https://cdn.example.com/first.webp']
    note['noteId'] = 'abcdef0123456789abcdef01'
    with pytest.raises(XHSCrawlerError, match='identity_mismatch'):client.read_note(URL)


def test_discovery_candidates_do_not_claim_verified_dates_or_attachments(monkeypatch):
    rows = [{'id': ID, 'xsecToken': 'synthetic', 'modelType': 'note',
        'noteCard': {'type': 'normal', 'displayTitle': '同名候选', 'user': {'nickname': '发布者'},
            'cover': {'urlDefault': 'https://cdn.example.com/cover.jpg'}}}]
    client = XHSCrawlerClient()
    monkeypatch.setattr(client, 'acquire', lambda *a, **k: rows+rows)
    result = client.search_posts('主体 主题', limit=2)
    assert result['visible_posts'] == 1
    assert result['posts'][0]['identity_status'] == 'candidate'
    assert result['posts'][0]['published_at'] == '' and result['posts'][0]['media_roles'] == []
    assert 'cover.jpg' not in json.dumps(result)


def test_authentication_failure_cannot_fall_back_to_public_requests(monkeypatch):
    monkeypatch.setattr(XHSCrawlerClient, 'configured', lambda *args: True)
    def denied(*a, **k):raise XHSCrawlerError('xhs_challenge_required')
    monkeypatch.setattr(XHSCrawlerClient, 'read_note', denied)
    reader = WebImageClient()
    monkeypatch.setattr(reader, 'fetch', lambda *a, **k: pytest.fail('must not retry another transport'))
    with pytest.raises(WebImageError, match='xhs_challenge_required'):reader.read_note(URL)


def test_model_client_has_no_login_or_write_capability():
    for action in ['login', 'status', 'publish', 'like', 'comment', 'exec']:
        with pytest.raises(XHSCrawlerError, match='invalid_action'):
            XHSCrawlerClient().acquire({'action': action})


def test_worker_environment_has_no_provider_keys_or_ambient_proxy(monkeypatch):
    seen = {}
    class Process:
        returncode = 0
        def __init__(self, command, **kwargs):seen.update(command=command, **kwargs)
        def __enter__(self):return self
        def __exit__(self, *args):pass
        def communicate(self, *args, **kwargs):return b'{"ok":true,"authenticated":true,"data":[]}', b''
    monkeypatch.setattr(XHSCrawlerClient, 'configured', lambda *args: True)
    monkeypatch.setattr('app.providers.xhs_crawler_client.subprocess.Popen', Process)
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-secret')
    monkeypatch.setenv('HTTPS_PROXY', 'http://sensitive-proxy.example')
    assert XHSCrawlerClient().acquire({'action': 'search'}) == []
    assert 'OPENAI_API_KEY' not in seen['env'] and 'HTTPS_PROXY' not in seen['env']
    assert seen['start_new_session'] and seen['command'] == ['/usr/local/bin/xhs-readonly']


def test_new_xhs_avc_bucket_remains_bounded_and_never_uses_other_codecs():
    stream = {'format': 'mp4', 'masterUrl': 'https://cdn.example.com/current.mp4',
        'size': 1000, 'duration': 70000, 'width': 720, 'height': 1280}
    note = {'type': 'video', 'video': {'media': {'video': {'drmType': 0}, 'stream': {
        'EF4': [stream], 'EF5': [{**stream, 'masterUrl': 'https://cdn.example.com/wrong.mp4', 'width': 1080}]}}}}
    assert parse_note_video(note, URL)['url'].endswith('/current.mp4')
    note['video']['media']['video']['drmType'] = 1
    with pytest.raises(WebImageError, match='protected_stream'):parse_note_video(note, URL)


def test_short_link_resolves_without_cookies_then_uses_fixed_note_reader(monkeypatch):
    seen=[]
    monkeypatch.setattr(XHSCrawlerClient,'configured',lambda *a:True)
    def read(self,url,**kwargs):seen.append(url);return {'noteId':ID},url,'snapshot'
    monkeypatch.setattr(XHSCrawlerClient,'read_note',read)
    reader=WebImageClient()
    monkeypatch.setattr(reader,'fetch',lambda *a,**k:(URL,b'public redirect body','utf-8'))
    assert reader.read_note('https://xhslink.com/synthetic')[0]['noteId']==ID
    assert seen==[URL]


def test_short_link_cannot_redirect_reader_to_an_unrelated_site(monkeypatch):
    reader=WebImageClient()
    monkeypatch.setattr(reader,'fetch',lambda *a,**k:('https://other.example/album',b'','utf-8'))
    with pytest.raises(WebImageError,match='invalid_redirect'):reader.read_note('https://xhslink.com/synthetic')
