import socket

import httpx
import pytest

from app.providers.web_images import WebImageClient, WebImageError, parse_page_images, parse_xiaohongshu_images
import json


def test_template_lazy_srcset_original_context_and_file_description():
    html = '''<script><img src="/script.png"></script><noscript><img src="/fallback.png"></noscript>
    <template><h2>鹿目圆人设</h2><a href="/File:Madoka.png"><img alt="鹿目圆" width="210"
    data-src="//cdn.example.com/thumb/a/ab/Madoka.png/210px-Madoka.png"
    src="data:image/png;base64,invalid"></a><figcaption>魔法少女与校服三视图</figcaption>
    <img src="/small.jpg" srcset="/medium.jpg 1x, /original.jpg 2x">
    <a href="/full.png"><img src="/preview.png"></a></template>'''
    result = parse_page_images(html, 'https://example.com/wiki')
    assert [item['url'] for item in result['images']] == [
        'https://cdn.example.com/a/ab/Madoka.png', 'https://example.com/original.jpg', 'https://example.com/full.png']
    first = result['images'][0]
    assert first['link_url'] == 'https://example.com/File:Madoka.png'
    assert first['alt'] == '鹿目圆' and '三视图' in first['context'] and first['width'] == '210'


def test_candidates_deduplicate_limit_and_reject_private_urls():
    html = '<img src="http://127.0.0.1/a.png"><img><img src="https://user:pw@example.com/a.png">'
    html += '<img src="/a.png" alt="人物"><img src="/a.png" alt="设定图">'
    html += ''.join(f'<img src="/{n}.png">' for n in range(200))
    result = parse_page_images(html, 'https://example.com/')
    assert len(result['images']) == 160 and result['candidate_limit_reached']
    assert result['images'][0]['url'] == 'https://example.com/a.png'
    assert '设定图' in result['images'][0]['context']


@pytest.mark.parametrize('failure', ['redirect_private', 'redirect_loop', 'too_large', 'chunked_large', 'not_html', 'http_403', 'empty'])
def test_bounded_page_failures(failure):
    calls = []
    def respond(request):
        calls.append(request)
        if failure.startswith('redirect'):
            return httpx.Response(302, headers={'Location': 'http://127.0.0.1/a' if failure == 'redirect_private' else '/loop'})
        if failure == 'too_large':
            return httpx.Response(200, headers={'Content-Length': str(3 * 1024 * 1024)})
        if failure == 'chunked_large':
            return httpx.Response(200, content=b'x' * (2 * 1024 * 1024 + 1))
        if failure == 'not_html':
            return httpx.Response(200, headers={'Content-Type': 'application/json'}, json={})
        return httpx.Response(403 if failure == 'http_403' else 200, text='JavaScript required')
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises((ValueError, httpx.HTTPError)):
            WebImageClient(http_client=http).list_images('https://example.com/page')
    assert len(calls) <= 5
    assert all(request.url.host == 'example.com' for request in calls)


def test_each_redirect_is_bounded_and_download_has_only_origin_referer():
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(302, headers={'Location': '/image.png'}) if len(requests) == 1 else httpx.Response(200, content=b'pixels')
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        target, pixels, _ = WebImageClient(http_client=http).fetch('https://example.com/redirect', image=True,
            referer='https://wiki.example.com/page?private-token=secret')
    assert target == 'https://example.com/image.png' and pixels == b'pixels'
    assert all(request.headers['Referer'] == 'https://wiki.example.com/' for request in requests)
    assert all('authorization' not in request.headers for request in requests)


@pytest.mark.parametrize('address', ['127.0.0.1', '10.0.0.1', '169.254.169.254', '::1', '198.18.0.1'])
def test_actual_transport_rejects_private_dns_before_socket_connect(monkeypatch, address):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: [(2, 1, 6, '', (address, 443))])
    def forbidden(*a, **k):
        raise AssertionError('private address reached connection')
    monkeypatch.setattr('app.providers.web_images.http.client.HTTPConnection', forbidden)
    with pytest.raises(WebImageError, match='nonpublic'):
        WebImageClient().fetch('https://example.com/image.png', image=True)


def test_credential_free_transport_pins_address_and_preserves_host_sni(monkeypatch):
    records = {}
    class Sock:
        def settimeout(self, seconds):
            assert 0 < seconds <= 30
    class Response:
        status = 200
        def getheaders(self):
            return [('Content-Type', 'text/html')]
        def read1(self, size):
            if records.get('read'):
                return b''
            records['read'] = True
            return b'<img src="/a.png">'
        def close(self):
            pass
    class Connection:
        sock = Sock()
        def __init__(self, host, port, **kwargs):
            records['connect'] = (host, port)
        def connect(self):
            pass
        def request(self, method, path, headers):
            records['request'] = (method, path, headers)
        def getresponse(self):
            return Response()
        def close(self):
            pass
    class TLS:
        def wrap_socket(self, sock, server_hostname):
            records['sni'] = server_hostname
            return sock
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: [(2, 1, 6, '', ('8.8.8.8', 443))])
    monkeypatch.setattr('app.providers.web_images.http.client.HTTPConnection', Connection)
    monkeypatch.setattr('app.providers.web_images.ssl.create_default_context', TLS)
    result = WebImageClient().list_images('https://example.com/page')
    assert result['images'][0]['url'] == 'https://example.com/a.png'
    assert records['connect'] == ('8.8.8.8', 443) and records['sni'] == 'example.com'
    assert records['request'][2]['Host'] == 'example.com'


NOTE = '0123456789abcdef01234567'


def note_page(*, note_type='normal', note_id=NOTE):
    state = {'note': {'noteDetailMap': {note_id: {'note': {'noteId': note_id, 'type': note_type,
        'title': '原笔记 undefined 字符串', 'imageList': [
            {'urlDefault': 'https://cdn.example.com/one.webp', 'width': 100, 'height': 200},
            {'urlPre': 'https://cdn.example.com/preview.jpg', 'infoList': [
                {'imageScene': 'WB_DFT', 'url': 'https://cdn.example.com/two.jpg'}]},
        ]}}}}}
    return '<script>window.__INITIAL_STATE__=' + json.dumps(state)[:-1] + ',"unrelated":undefined}</script>'


def test_xiaohongshu_selects_only_requested_album_and_original_order():
    page = parse_xiaohongshu_images(note_page(), f'https://www.xiaohongshu.com/explore/{NOTE}')
    assert page['album'] and page['declared_image_count'] == 2
    assert [item['url'] for item in page['images']] == ['https://cdn.example.com/one.webp', 'https://cdn.example.com/two.jpg']
    assert 'undefined' in page['title']


def test_mobile_album_uses_h5_note_data_and_url_fields():
    state = {'noteData': {'data': {'noteData': {'noteId': NOTE, 'type': 'normal', 'imageList': [
        {'url': 'https://cdn.example.com/mobile.jpg'},
        {'infoList': [{'imageScene': 'H5_DTL', 'url': 'https://cdn.example.com/second.jpg'}]}
    ]}}}}
    html = '<script>window.__INITIAL_STATE__=' + json.dumps(state) + '</script>'
    page = parse_xiaohongshu_images(html, f'https://www.xiaohongshu.com/discovery/item/{NOTE}')
    assert [item['url'] for item in page['images']] == ['https://cdn.example.com/mobile.jpg', 'https://cdn.example.com/second.jpg']


@pytest.mark.parametrize('condition', ['different_note', 'video', 'login', 'script_expression'])
def test_xiaohongshu_never_substitutes_cover_avatar_or_recommendations(condition):
    html = note_page(note_type='video') if condition == 'video' else note_page()
    if condition == 'different_note':
        html = note_page(note_id='1123456789abcdef01234567')
    if condition == 'login':
        html = '<meta property="og:image" content="https://cdn.example.com/cover.jpg"><img src="/avatar.png">登录'
    if condition == 'script_expression':
        html = '<script>window.__INITIAL_STATE__=dangerous()</script>'
    with pytest.raises(WebImageError):
        parse_xiaohongshu_images(html, f'https://www.xiaohongshu.com/explore/{NOTE}')


def test_desktop_login_can_retry_only_same_public_mobile_note():
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, headers={'Content-Type': 'text/html'}, text=(
            note_page() if '/discovery/item/' in request.url.path else '登录'))
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        page = WebImageClient(http_client=http).list_images(f'https://www.xiaohongshu.com/explore/{NOTE}?xsec_token=synthetic')
    assert len(requests) == 2 and page['declared_image_count'] == 2
    assert requests[1].url.path == f'/discovery/item/{NOTE}'
    assert requests[1].url.params['xsec_token'] == 'synthetic'
