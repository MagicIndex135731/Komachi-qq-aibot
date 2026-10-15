import json

import httpx
import pytest

from app.providers.web_search import WebSearchProviderError


def native_frames(*, grounding=True, finish='STOP'):
    meta={'groundingChunks':[{'web':{'uri':'https://example.com/release','title':'Official release'}},
        {'web':{'uri':'http://127.0.0.1/private','title':'Private'}}],
        'groundingSupports':[{'segment':{'text':'Version 0.13.0, published October 9.'},'groundingChunkIndices':[0]}]}
    return ('data: '+json.dumps({'candidates':[{'content':{'parts':[{'text':'0.13.0, 2026-10-09.'}]}}]})+'\n\n'
        +'data: '+json.dumps({'candidates':[{'finishReason':finish,**({'groundingMetadata':meta} if grounding else {})}]})+'\n\n').encode()


def make(handler):
    from app.providers.gemini_search import GeminiSearchClient
    return GeminiSearchClient(provider='gemini',base_url='https://relay.example/v1',api_key='SYNTHETIC_PRIVATE_KEY',
        timeout_seconds=45,model='gemini-3.8-flash-medium',reasoning_effort='medium',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),address_resolver=lambda *_: ['93.184.216.34'])


def test_open_page_has_its_own_transport_and_accepts_github_tree_json():
    from app.providers.gemini_search import GeminiSearchClient
    direct, pages = [], []
    def main(request):
        direct.append(request)
        if request.method == 'POST':return httpx.Response(200, content=native_frames())
        return httpx.Response(200, text='direct find text', headers={'content-type':'text/plain'})
    def page(request):
        pages.append(request)
        if request.url.host != 'api.github.com':
            return httpx.Response(200, text='page find text', headers={'content-type':'text/plain'})
        assert request.headers['accept']=='application/vnd.github+json'
        return httpx.Response(200,json={'sha':'synthetic', 'tree':[{'path':'src/server.ts','type':'blob','size':50}]})
    client=GeminiSearchClient(provider='gemini',base_url='https://relay.example/v1',api_key='SYNTHETIC_PRIVATE_KEY',
        timeout_seconds=45,model='gemini-3.8-flash-medium',http_client=httpx.Client(transport=httpx.MockTransport(main)),
        page_http_client=httpx.Client(transport=httpx.MockTransport(page)),address_resolver=lambda *_:['93.184.216.34'])
    client.search_report('test')
    assert 'src/server.ts' in client.open_page('https://api.github.com/repos/example/repo/git/trees/main').content
    client.open_page('https://api.github.com/repos/example/repo/git/trees/main',lineno=1)
    assert 'page find text' in client.find_page('https://example.com/other',pattern='page find').content
    assert len(pages)==2 and len(direct)==1
    assert all('x-goog-api-key' not in r.headers for r in pages)


def test_native_search_keeps_grounding_and_never_executes_a_client_function():
    requests=[]
    def handler(r):
        requests.append(r)
        return httpx.Response(200,content=native_frames(),headers={'content-type':'text/event-stream'})
    c=make(handler)
    with c.http_client:
        report=c.search_report('latest release',max_results=8)
    assert len(requests)==1
    body=json.loads(requests[0].content)
    assert body['tools']==[{'googleSearch':{}}]
    assert body['generationConfig']['maxOutputTokens'] == 8192
    assert 'streamGenerateContent' in str(requests[0].url) and '/v1/v1beta/' not in str(requests[0].url)
    assert requests[0].headers['x-goog-api-key']=='SYNTHETIC_PRIVATE_KEY'
    assert report.content_kind=='grounded_answer' and '0.13.0' in report.content
    assert len(report.results)==1 and report.results[0].source=='https://example.com/release'
    assert 'October 9' in report.results[0].snippet
    assert report.results[0].ref_id=='turn0search0'


@pytest.mark.parametrize('grounding,finish',[(False,'STOP'),(True,'MALFORMED_FUNCTION_CALL')])
def test_accepted_but_ungrounded_or_malformed_is_not_search_success(grounding,finish):
    c=make(lambda _:httpx.Response(200,content=native_frames(grounding=grounding,finish=finish)))
    with c.http_client,pytest.raises(WebSearchProviderError):c.search_report('test')


def test_terminal_native_frame_closes_without_waiting_for_eof():
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield native_frames()
            raise AssertionError('terminal frame must end reading')
    c=make(lambda _:httpx.Response(200,stream=Stream()))
    with c.http_client:assert c.search_report('test').results


def test_web_response_is_byte_bounded():
    c=make(lambda _:httpx.Response(200,content=b'x'*(2*1024*1024+1)))
    with c.http_client,pytest.raises(WebSearchProviderError,match='too_large'):c.search_report('test')


def test_direct_page_read_and_find_use_actual_body_without_provider_credentials():
    requests=[]
    def handler(r):
        requests.append(r)
        if r.url.path=='/redirect':return httpx.Response(302,headers={'location':'https://example.com/article'})
        return httpx.Response(200,headers={'content-type':'text/html'},text='<html><title>Actual title</title><main><p>First paragraph</p><p>needle actual body</p><p>Last paragraph</p></main></html>')
    c=make(handler)
    with c.http_client:
        report=c.open_page('https://example.com/redirect')
        found=c.find_page('https://example.com/article',pattern='needle')
    assert c.supports_page_tools and 'needle actual body' in report.content and 'L0:' in report.content
    assert 'needle' in found.content and 'Total lines:' in found.content
    assert all('authorization' not in r.headers and 'x-goog-api-key' not in r.headers for r in requests)


def test_page_reader_keeps_bounded_public_article_links_without_fetching_them():
    calls = []
    html = '''<html><nav><a href="https://noise.example.com">menu</a></nav>
        <article><p>这是一篇有原文依据的新闻，报道时间不一定是原帖时间。</p>
        <blockquote><a href="https://x.com/synthetic/status/900000000000000001">原帖日期</a></blockquote>
        <a href="/original?x=1&amp;y=2">原始页面</a>
        <a href="http://127.0.0.1/private">本地</a><a href="javascript:alert(1)">脚本</a>
        <script>https://hidden.example.com/script</script></article></html>'''
    c = make(lambda r: calls.append(r) or httpx.Response(200,
        headers={'content-type': 'text/html'}, text=html))
    with c.http_client:
        report = c.open_page('https://example.com/news')
        found = c.find_page('https://example.com/news', pattern='x.com/synthetic/status')
    assert 'https://x.com/synthetic/status/900000000000000001' in report.content
    assert 'https://example.com/original?x=1&y=2' in report.content
    assert 'x.com/synthetic/status' in found.content
    assert all(value not in report.content for value in ('127.0.0.1', 'javascript:', 'hidden.example.com', 'noise.example.com'))
    assert len(calls) == 1


def test_raw_github_source_preserves_blank_and_long_line_locations():
    source='export const one = 1;\n\n'+'x'*600+'\n'
    c=make(lambda _:httpx.Response(200,headers={'content-type':'text/plain'},text=source))
    with c.http_client:
        page=c.open_page('https://raw.githubusercontent.com/example/repo/main/server.ts')
        continuation=c.open_page('https://raw.githubusercontent.com/example/repo/main/server.ts',lineno=3)
        found=c.find_page('https://raw.githubusercontent.com/example/repo/main/server.ts',pattern='source_line')
    assert page.content_kind=='raw_source'
    assert 'L1: [source_line=2' in page.content
    assert page.content.count('source_line=3')==3
    assert 'L3: [source_line=3' in continuation.content
    assert 'Pattern not found' in found.content


@pytest.mark.parametrize('target',['http://127.0.0.1/admin','http://localhost/admin'])
def test_public_page_redirect_cannot_target_internal_hosts(target):
    c=make(lambda _:httpx.Response(302,headers={'location':target}))
    with c.http_client,pytest.raises(WebSearchProviderError):c.open_page('https://example.com/redirect')


def test_nonpublic_resolved_address_is_blocked_before_fetch():
    calls=[];c=make(lambda r:calls.append(r) or httpx.Response(200,text='private'))
    c.address_resolver=lambda *_:['127.0.0.1']
    with c.http_client,pytest.raises(WebSearchProviderError):c.open_page('https://example.com/page')
    assert calls==[]


def test_gemini_factory_keeps_search_and_image_reference_credentials_on_gemini():
    from app.config import AppSettings
    from app.runtime.providers import build_web_search_client,build_image_reference_search_client
    settings=AppSettings(_env_file=None,BOT_QQ=900000101,OWNER_QQ=900000102,
        NAPCAT_WS_URL='ws://example.com/ws',LLM_BASE_URL='https://relay.example/v1',LLM_API_KEY='synthetic',
        LLM_MODEL='gemini-3.8-flash-high',SEARCH_PROVIDER='gemini',SEARCH_BASE_URL='https://relay.example/v1',
        SEARCH_API_KEY='synthetic',LLM_WEB_SEARCH_MODEL='gemini-3.8-flash-medium',WEB_SEARCH_REASONING_EFFORT='medium')
    for builder in (build_web_search_client,build_image_reference_search_client):
        c=builder(settings)
        try:
            assert c.provider=='gemini' and c.model=='gemini-3.8-flash-medium' and c.timeout_seconds==45
            assert c.base_url==settings.llm_base_url and c.api_key==settings.llm_api_key
        finally:c.http_client.close()


def test_image_reference_search_returns_attachments_from_actual_public_pages():
    requests=[]
    def handler(r):
        requests.append(r)
        if r.method=='POST':return httpx.Response(200,content=native_frames())
        return httpx.Response(200,headers={'content-type':'text/html'},text='<meta property="og:image" content="/photo.jpg"><img src="http://127.0.0.1/private.jpg">')
    c=make(handler)
    with c.http_client:images=c.image_search('reference',max_results=3)
    assert len(images)==1 and images[0].url=='https://example.com/photo.jpg' and images[0].file_id=='photo.jpg'
    assert [r.method for r in requests]==['POST','GET']


def test_legacy_page_reader_preserves_skim_contract():
    from app.providers.web_search import WebSearchResult
    c=make(lambda r:httpx.Response(200,headers={'content-type':'text/plain'},text='actual text'))
    with c.http_client:pages=c.read_pages([WebSearchResult('title','','https://example.com/page','')],query='actual',skim_limit=6)
    assert pages[0].content=='actual text'


def test_bilibili_metadata_fallback_is_labeled_as_metadata():
    def handler(r):
        if r.url.host!='api.bilibili.com':return httpx.Response(412)
        return httpx.Response(200,json={'code':0,'data':{'bvid':'BV1234567890','title':'Public title','desc':'Public description','owner':{'name':'Public author'}}})
    c=make(handler)
    with c.http_client:report=c.open_page('https://www.bilibili.com/video/BV1234567890')
    assert report.content_kind=='bilibili_public_metadata' and '未观看视频' in report.content


def test_interrupted_stream_is_not_a_grounded_success():
    stream=b'data: '+json.dumps({'candidates':[{'content':{'parts':[{'text':'Guess'}]}}]}).encode()+b'\n\n'
    c=make(lambda _:httpx.Response(200,content=stream))
    with c.http_client,pytest.raises(WebSearchProviderError):c.search_report('latest')


def test_search_prose_cannot_invent_actionable_urls():
    payload = native_frames().replace(b'0.13.0, 2026-10-09.',
        b'[invented](https://example.com/does-not-exist) 0.13.0')
    c = make(lambda _: httpx.Response(200, content=payload))
    with c.http_client:
        result = c.search_report('locate an actual post')
    assert 'does-not-exist' not in result.content
    assert 'https://example.com/release' in result.content


def test_media_discovery_reads_grounded_sources_and_keeps_short_embed_region():
    calls = []
    post = 'https://x.com/synthetic/status/900000000000000001'
    def handler(request):
        calls.append(request)
        if request.method == 'POST':
            return httpx.Response(200, content=native_frames())
        if request.url.path == '/release':
            return httpx.Response(302, headers={'location': '/article'})
        return httpx.Response(200, headers={'content-type': 'text/html'}, text=
            '<article><p>' + 'Long news paragraph. ' * 50 + '</p></article>'
            '<article><a href="' + post + '">Original post</a></article>')
    c = make(handler)
    with c.http_client:
        result = c.search_resources('latest photos')
    assert result.results[0].source == 'https://example.com/article'
    assert any(hit.source == post for hit in result.results)
    assert post in result.content and len(calls) == 3
    assert all('x-goog-api-key' not in r.headers for r in calls[1:])


def test_media_discovery_does_not_turn_one_failed_source_into_global_failure():
    body = native_frames().replace(b'http://127.0.0.1/private', b'https://other.example/article')
    calls = []
    def handler(request):
        calls.append(request)
        if request.method == 'POST':return httpx.Response(200, content=body)
        if request.url.host == 'example.com':return httpx.Response(404)
        return httpx.Response(200, headers={'content-type': 'text/html'},
            text='<article><a href="https://media.example/photo.jpg">Photo</a></article>')
    c = make(handler)
    with c.http_client:result = c.search_resources('photos')
    assert any(hit.source == 'https://media.example/photo.jpg' for hit in result.results)
    assert len(calls) == 3


def test_media_discovery_bounds_source_reads_and_prioritizes_media_over_share_actions():
    sources = [{'web': {'uri': f'https://example.com/article{i}', 'title': f'Source {i}'}} for i in range(5)]
    frame = ('data: ' + json.dumps({'candidates': [{'content': {'parts': [{'text': 'Candidates'}]},
        'finishReason': 'STOP', 'groundingMetadata': {'groundingChunks': sources}}]}) + '\n\n').encode()
    calls = []
    def handler(request):
        calls.append(request)
        if request.method == 'POST':
            assert '保留对象和时间范围' in json.loads(request.content)['contents'][0]['parts'][0]['text']
            return httpx.Response(200, content=frame)
        index = request.url.path[-1]
        return httpx.Response(200, headers={'content-type': 'text/html'}, text=
            '<article><p>Actual body</p>' + ''.join(
                f'<a href="https://share.example/action{i}">Share</a>' for i in range(8)) +
            f'<a href="https://x.com/synthetic/status/90000000000000000{index}">Post</a></article>')
    c = make(handler)
    with c.http_client:result = c.search_resources('profile', goal='最近的图片')
    assert len(calls) == 4  # One search plus three source pages, no recursion.
    assert len(result.results) <= 20
    assert sum('/status/' in hit.source for hit in result.results) == 3


def test_recent_media_discovery_uses_later_recent_source_hint_without_asserting_a_post_date():
    from app.providers.web_search import WebSearchReport, WebSearchResult
    calls = []
    c = make(lambda r: calls.append(str(r.url)) or httpx.Response(200,
        headers={'content-type': 'text/html'}, text='<article><p>Actual source</p></article>'))
    c._search_report = lambda *args: WebSearchReport([
        WebSearchResult('Source', '2019年9月', f'https://example.com/old{i}', '') for i in range(3)] +
        [WebSearchResult('Source', '2020年10月中旬', 'https://example.com/newer', '')], 'summary', 'grounded_answer')
    with c.http_client:result = c.search_resources('account', goal='最新的图片')
    assert calls[0] == 'https://example.com/newer' and len(calls) == 3
    assert all(hit.date == '' for hit in result.results)
