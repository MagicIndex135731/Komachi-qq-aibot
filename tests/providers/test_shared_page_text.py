"""Only public Bilibili text is read; redirects cannot expand the trust scope."""
import httpx
import pytest

from app.providers.web_search import WebSearchClient, WebSearchProviderError

BVID = 'BV1ab411c7xy'
PAGE = f'https://www.bilibili.com/video/{BVID}'


def client(handler):
    return WebSearchClient(provider='wyzlab_alpha', base_url='https://bridge.example/v1',
        api_key='bridge-secret', timeout_seconds=8,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_bilibili_short_share_falls_back_to_public_text_without_credentials_or_media():
    requests = []
    def handler(request):
        requests.append(request)
        if request.url.host == 'bridge.example':
            return httpx.Response(200, json={'output': 'Internal Error: page unavailable'})
        assert not request.headers.get('authorization') and not request.headers.get('cookie')
        if request.url.host == 'b23.tv':
            return httpx.Response(302, headers={'Location': PAGE + '?share_source=qq'})
        assert request.url.host == 'api.bilibili.com'
        assert request.url.path == '/x/web-interface/view'
        assert request.url.params['bvid'] == BVID
        return httpx.Response(200, json={'code': 0, 'data': {'bvid': BVID,
            'title': 'Example public title', 'desc': 'Detailed public description.',
            'owner': {'name': 'Example author'}, 'pubdate': 100, 'duration': 12}})
    result = client(handler).open_page('https://b23.tv/Example')
    assert result.content_kind == 'bilibili_public_metadata'
    assert result.results[0].source == PAGE
    assert 'Detailed public description.' in result.content and 'Example author' in result.content
    assert '未观看视频' in result.content and 'L0:' in result.content
    assert [r.url.host for r in requests] == ['bridge.example', 'b23.tv', 'api.bilibili.com']
    assert 'share_source' not in result.results[0].source


@pytest.mark.parametrize('location', ['http://127.0.0.1/secret', 'https://evil.example/video/' + BVID,
    'https://user:password@www.bilibili.com/video/' + BVID, 'https://www.bilibili.com:8443/video/' + BVID])
def test_short_link_redirect_outside_fixed_hosts_is_never_followed(location):
    seen=[]
    def handler(request):
        seen.append(request.url.host)
        if request.url.host == 'bridge.example':
            return httpx.Response(200, json={'output': 'Internal Error: unavailable'})
        assert request.url.host == 'b23.tv'
        return httpx.Response(302, headers={'Location': location})
    with pytest.raises(WebSearchProviderError, match='web_page_unavailable'):
        client(handler).open_page('https://b23.tv/Example')
    assert seen == ['bridge.example', 'b23.tv']


def test_no_fallback_after_success_or_on_unrelated_page_and_no_empty_desc_invention():
    def successful(request):
        assert request.url.host == 'bridge.example'
        return httpx.Response(200, json={'output': 'Normal public web content'})
    assert client(successful).open_page(PAGE).content_kind == 'web_page'
    def unavailable(request):
        assert request.url.host == 'bridge.example'
        return httpx.Response(200, json={'output': 'Internal Error: unavailable'})
    with pytest.raises(WebSearchProviderError):client(unavailable).open_page('https://example.com/article')
    def empty_desc(request):
        if request.url.host == 'bridge.example':return httpx.Response(200,json={'output':'Internal Error: unavailable'})
        return httpx.Response(200,json={'code':0,'data':{'bvid':BVID,'title':'Empty description','desc':''}})
    result=client(empty_desc).open_page(PAGE,lineno=2)
    assert '简介为空' in result.content and 'L2:' in result.content


@pytest.mark.parametrize('payload', [{'code': -404}, {'code':0,'data':{'bvid':'BV1zz411c7xy','title':'Wrong item'}},
    {'code':0,'data':{'bvid':BVID,'title':''}}])
def test_failed_or_mismatched_metadata_does_not_become_success(payload):
    def handler(request):
        if request.url.host=='bridge.example':return httpx.Response(200,json={'output':'Internal Error: unavailable'})
        return httpx.Response(200,json=payload)
    with pytest.raises(WebSearchProviderError,match='web_page_unavailable'):client(handler).open_page(PAGE)


def test_short_redirect_loop_and_large_metadata_have_finite_bounds():
    seen=[]
    def loop(request):
        seen.append(request.url.host)
        if request.url.host=='bridge.example':return httpx.Response(200,json={'output':'Internal Error: unavailable'})
        return httpx.Response(302,headers={'Location':'https://b23.tv/Another'})
    with pytest.raises(WebSearchProviderError):client(loop).open_page('https://b23.tv/Example')
    assert seen==['bridge.example','b23.tv','b23.tv']
    def too_large(request):
        if request.url.host=='bridge.example':return httpx.Response(200,json={'output':'Internal Error: unavailable'})
        return httpx.Response(200,content=b'x'*(512*1024+1))
    with pytest.raises(WebSearchProviderError):client(too_large).open_page(PAGE)
