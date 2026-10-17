import json

import pytest

from app.providers.resource_inspector import PublicResourceClient
from app.providers.web_images import WebImageError


class Reader:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def fetch(self, url, **kwargs):
        kwargs.get('check', lambda: None)()
        self.calls.append(url)
        value = self.pages[url]
        return url, (json.dumps(value) if isinstance(value, dict) else value).encode(), 'utf-8'


def test_bangumi_identity_cover_role_and_not_a_video_site():
    reader = Reader({'https://api.bgm.tv/v0/subjects/12': {
        'id': 12, 'name_cn': '示例动画', 'date': '2026-01-01',
        'summary': '原站简介', 'images': {'large': 'https://lain.bgm.tv/cover.jpg'}}})
    result = PublicResourceClient(reader=reader).inspect('https://bgm.tv/subject/12')
    assert result['post_id'] == '12' and result['identity_status'] == 'matched'
    assert result['media'][0]['role'] == 'cover'
    assert result['published_at'] == '2026-01-01'
    assert result['media'][0]['url'] == 'https://lain.bgm.tv/cover.jpg'


def test_bangumi_mismatched_id_is_rejected():
    reader = Reader({'https://api.bgm.tv/v0/subjects/12': {'id': 13, 'images': {}}})
    with pytest.raises(WebImageError, match='web_resource_id_mismatch'):
        PublicResourceClient(reader=reader).inspect('https://bgm.tv/subject/12')


def test_generic_error_pages_and_site_icons_do_not_become_media():
    url = 'https://example.com/post/12'
    reader = Reader({url: '<title>Article</title><main>Useful body</main>'
        '<img src="/favicon.png"><img src="/avatar.jpg"><img src="/body.jpg" alt="作品">'})
    result = PublicResourceClient(reader=reader).inspect(url)
    assert [m['url'] for m in result['media']] == ['https://example.com/body.jpg']
    assert result['media'][0]['role'] == 'page_image'
    assert result['identity_status'] == 'unverified'
    reader.pages[url] = '<title>小红书 - 你访问的页面不见了</title><img src="/error.png">'
    with pytest.raises(WebImageError, match='web_resource_unavailable'):
        PublicResourceClient(reader=reader).inspect(url)


def test_fantia_specific_extractor_preserves_cover_and_post_identity():
    reader = Reader({})
    client = PublicResourceClient(reader=reader)
    client._gallery = lambda url, check, deadline: [
        {'url': 'https://c.fantia.jp/cover.jpg', 'post_id': 123, 'post_title': '作品',
         'fanclub_user_name': '作者', 'posted_at': '2026-01-01', 'content_category': 'thumb'},
        {'url': 'https://c.fantia.jp/body.jpg', 'post_id': 123, 'content_category': 'photo'},
    ]
    result = client.inspect('https://fantia.jp/posts/123')
    assert [m['role'] for m in result['media']] == ['cover', 'post_image']
    client._gallery = lambda *a: [{'url': 'https://c.fantia.jp/other.jpg', 'post_id': 124}]
    with pytest.raises(WebImageError, match='web_resource_id_mismatch'):
        client.inspect('https://fantia.jp/posts/123')


def test_bilibili_uses_specific_extractor_and_progressive_fallback_without_mixing_tracks():
    reader = Reader({
      'https://api.bilibili.com/x/web-interface/view?bvid=BV1example': {'code': 0, 'data': {
        'bvid': 'BV1example', 'aid': 123, 'cid': 456, 'title': '作品', 'pubdate': 1767225600,
        'owner': {'name': '作者', 'mid': 1}, 'duration': 15}},
      'https://api.bilibili.com/x/player/playurl?avid=123&cid=456&qn=16&fnval=1&platform=html5': {
        'code': 0, 'data': {'durl': [{'url': 'https://video.bilivideo.com/file.mp4', 'size': 5000}]}}
    })
    client = PublicResourceClient(reader=reader)
    client._yt = lambda *a: {'id': 'BV1example', 'formats': [{'url': 'https://video.bilivideo.com/video.m4s', 'vcodec': 'avc1', 'acodec': 'none'}]}
    result = client.inspect('https://www.bilibili.com/video/BV1example')
    assert result['media'][0]['url'].endswith('.mp4')
    assert result['media'][0]['declared_bytes'] == 5000
    assert result['identity_status'] == 'matched'


def test_invalid_urls_and_revocation_never_dispatch():
    reader = Reader({})
    client = PublicResourceClient(reader=reader)
    with pytest.raises(ValueError):client.inspect('http://127.0.0.1/a')
    def revoked():raise RuntimeError('revoked')
    with pytest.raises(RuntimeError, match='revoked'):
        client.inspect('https://bgm.tv/subject/12', check=revoked)
    assert not reader.calls


def test_bilibili_part_two_does_not_download_part_one():
    root = 'https://api.bilibili.com/x/web-interface/view?bvid=BV1example'
    stream = 'https://api.bilibili.com/x/player/playurl?avid=123&cid=457&qn=16&fnval=1&platform=html5'
    reader = Reader({root: {'code': 0, 'data': {'bvid': 'BV1example', 'aid': 123,
        'pages': [{'cid': 456, 'duration': 30}, {'cid': 457, 'duration': 15}]}},
        stream: {'code': 0, 'data': {'durl': [{'url': 'https://video.bilivideo.com/part2.mp4', 'size': 100}]}}})
    client = PublicResourceClient(reader=reader)
    client._yt = lambda *a: {'id': 'BV1example_p2', 'formats': []}
    result = client.inspect('https://www.bilibili.com/video/BV1example?p=2')
    assert result['media'][0]['url'].endswith('part2.mp4')
    assert result['media'][0]['duration_seconds'] == 15
    client._yt = lambda *a: {'id': 'BV1example_p1', 'formats': []}
    with pytest.raises(WebImageError, match='web_resource_id_mismatch'):
        client.inspect('https://www.bilibili.com/video/BV1example?p=2')


def test_mature_extractors_use_only_guarded_public_gets_and_do_not_fetch_media():
    pytest.importorskip('gallery_dl')
    url = 'https://fantia.jp/posts/123'
    reader = Reader({'https://fantia.jp/': '<meta name="csrf-token" content="public-token">',
      'https://fantia.jp/api/v1/posts/123': {'post': {'id': 123, 'title': '作品', 'comment': '',
        'rating': 'general', 'posted_at': 'Wed, 08 Mar 2023 17:16:51 +0900',
        'fanclub': {'id': 1, 'name': '作者', 'user': {'id': 2, 'name': '作者'}}, 'tags': [],
        'thumb': {'original': 'https://c.fantia.jp/cover.jpg'}, 'post_contents': []}}})
    result = PublicResourceClient(reader=reader).inspect(url)
    assert result['media'][0]['role'] == 'cover'
    assert reader.calls == ['https://fantia.jp/', 'https://fantia.jp/api/v1/posts/123']
