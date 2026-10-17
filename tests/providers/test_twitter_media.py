import asyncio
import hashlib
import json

import httpx
import pytest

from app.providers.twitter_media import (
    TwitterMediaDownloader,
    TwitterMediaError,
    parse_post_url,
)


POST_ID = "900000000000000001"
URL = f"https://x.com/sample/status/{POST_ID}"
PHOTO = b"\xff\xd8\xff\xe0" + b"synthetic-jpeg" * 5
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"synthetic-mp4" * 5


def photo(name="image"):
    return {"type": "photo", "media_url_https": f"https://pbs.twimg.com/media/{name}.jpg"}


def video(kind="video"):
    return {"type": kind, "video_info": {"variants": [
        {"content_type": "application/x-mpegURL", "url": "https://video.twimg.com/hls.m3u8"},
        {"content_type": "video/mp4", "bitrate": 128, "url": "https://video.twimg.com/low.mp4"},
        {"content_type": "video/mp4", "bitrate": 512, "url": "https://video.twimg.com/high.mp4"},
    ]}}


def metadata(items):
    return {"id_str": POST_ID, "mediaDetails": items}


def client_for(payload, handler=None):
    def respond(request):
        if request.url.host == "cdn.syndication.twimg.com":
            return httpx.Response(200, json=payload)
        if handler:
            return handler(request)
        if request.url.host == "pbs.twimg.com":
            return httpx.Response(200, content=PHOTO, headers={"content-type": "image/jpeg"})
        return httpx.Response(200, content=MP4, headers={"content-type": "video/mp4"})
    return httpx.AsyncClient(transport=httpx.MockTransport(respond))


@pytest.mark.parametrize("url", [URL, URL.replace("x.com", "mobile.twitter.com") + "?s=20",
    f"https://twitter.com/i/web/status/{POST_ID}", f"https://x.com/i/status/{POST_ID}", URL + "/photo/2"])
def test_post_url_normalizes_source(url):
    post = parse_post_url(url)
    assert post.post_id == POST_ID
    assert post.source_url == f"https://x.com/i/status/{POST_ID}"


async def test_browser_observation_download_reuses_guards_and_provenance(tmp_path):
    evidence = 'x-posts-' + 'a' * 24 + ':post:0'
    observation = {'post_id': POST_ID, 'identity_status': 'matched', 'protected':False, 'author': 'sample',
        'published_at': '2026-10-16T00:00:00+00:00', 'media_count': 1, 'media_error': '',
        'media': [{'kind': 'photo', 'url': 'https://pbs.twimg.com/media/image.jpg'}]}
    async with client_for({}) as client:
        saved = await TwitterMediaDownloader(client).download_resolved(URL, tmp_path,
            observation, evidence_ref=evidence)
    assert saved.manifest['resolver'] == 'x_crawler'
    assert saved.manifest['post']['evidence_ref'] == evidence
    assert saved.manifest['files'][0]['sha256'] == hashlib.sha256(PHOTO).hexdigest()


@pytest.mark.parametrize('change,code', [
    ({'post_id': '900000000000000002'}, 'post_id_mismatch'),
    ({'media_count': 2}, 'incomplete_resolved_media'),
    ({'media': [{'kind': 'photo', 'url': 'https://127.0.0.1/secret'}]}, 'invalid_media_url'),
    ({'media': [{'kind': 'video', 'url': 'https://video.twimg.com/stream.m3u8'}]}, 'no_mp4_stream'),
])
async def test_browser_observation_rejects_wrong_identity_partial_or_unsafe_media(tmp_path, change, code):
    observation = {'post_id': POST_ID, 'identity_status': 'matched', 'protected':False, 'media_count': 1,
        'media': [{'kind': 'photo', 'url': 'https://pbs.twimg.com/media/image.jpg'}], **change}
    async with client_for({}) as client:
        with pytest.raises(TwitterMediaError, match=code):
            await TwitterMediaDownloader(client).download_resolved(URL, tmp_path,
                observation, evidence_ref='x-posts-' + 'a' * 24 + ':post:0')
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("url", ["https://x.com/sample", "https://x.com/search?q=cat",
    "https://x.com.evil.example/sample/status/123", "https://x.com@evil.example/a/status/123",
    "https://x.com:444/a/status/123", "file:///sample/status/123",
    URL + "/../../status/222", "https://127.0.0.1/a/status/123", URL + "\nX: value"])
def test_rejects_non_post_urls(url):
    with pytest.raises(TwitterMediaError, match="invalid_post_url"):
        parse_post_url(url)


async def test_download_mixed_media_preserves_order_and_provenance(tmp_path):
    requests = []
    def handler(request):
        requests.append(request.url)
        if request.url.host == "pbs.twimg.com":
            assert request.url.params["name"] == "orig"
            return httpx.Response(200, content=PHOTO, headers={"content-type": "image/jpeg"})
        assert request.url.path == "/high.mp4"
        return httpx.Response(200, content=MP4, headers={"content-type": "video/mp4"})
    async with client_for(metadata([photo(), video(), video("animated_gif")]), handler) as client:
        result = await TwitterMediaDownloader(client).download(URL, tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["post_id"] == POST_ID
    assert manifest["source_url"] == f"https://x.com/i/status/{POST_ID}"
    assert manifest["resolver"] == "syndication"
    assert [m["kind"] for m in manifest["files"]] == ["photo", "video", "gif"]
    assert [m["filename"] for m in manifest["files"]] == ["01_photo.jpg", "02_video.mp4", "03_gif.mp4"]
    assert manifest["files"][0]["sha256"] == hashlib.sha256(PHOTO).hexdigest()
    assert manifest['files'][0]['image_size']=='orig' and 'fallback_reason' not in manifest['files'][0]
    assert len(requests) == 3
    for item in manifest["files"]:
        assert (result.directory / item["filename"]).stat().st_size == item["bytes"]
    assert not list(tmp_path.glob(".twitter-*"))


async def test_fx_fallback_and_exclusion_of_quoted_media(tmp_path):
    def respond(request):
        if request.url.host == "cdn.syndication.twimg.com":
            return httpx.Response(200, json={"__typename": "TweetTombstone"})
        if request.url.host == "api.fxtwitter.com":
            return httpx.Response(200, json={"code": 200, "tweet": {"id": POST_ID,
                "media": {"all": [{"type": "gif", "url": "https://video.twimg.com/gif.mp4"}]},
                "quote": {"media": {"all": [{"type": "photo", "url": "https://pbs.twimg.com/other.jpg"}]}}}})
        return httpx.Response(200, content=MP4, headers={"content-type": "video/mp4"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await TwitterMediaDownloader(client).download(URL, tmp_path)
    assert result.manifest["resolver"] == "fxtwitter"
    assert len(result.manifest["files"]) == 1
    assert result.manifest["resolver_failures"] == [{"resolver": "syndication", "code": "post_unavailable"}]


async def test_separate_mixed_arrays_never_invent_original_order(tmp_path):
    calls=[]
    def respond(request):
        calls.append(request)
        return httpx.Response(200,json={'code':200,'tweet':{'id':POST_ID,'media':{
            'photos':[{'type':'photo','url':'https://pbs.twimg.com/sample.jpg'}],
            'videos':[{'type':'video','url':'https://video.twimg.com/sample.mp4'}]}}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(TwitterMediaError,match='media_order_unavailable'):
            await TwitterMediaDownloader(client,resolver='fxtwitter').download(URL,tmp_path)
    assert len(calls)==1 and not list(tmp_path.iterdir())


@pytest.mark.parametrize("payload,code", [
    ({"id_str": "900000000000000002", "mediaDetails": [photo()]}, "post_id_mismatch"),
    (metadata([{"type": "photo", "media_url_https": "https://127.0.0.1/secret.jpg"}]), "invalid_media_url"),
    (metadata([{"type": "photo", "media_url_https": "https://pbs.twimg.com.evil.example/a.jpg"}]), "invalid_media_url"),
    (metadata([{"type": "video", "video_info": {"variants": []}}]), "no_mp4_stream"),
    (metadata([{"type": "unknown"}]), "unsupported_media_type"),
])
async def test_invalid_metadata_fails_without_output(tmp_path, payload, code):
    async with client_for(payload) as client:
        with pytest.raises(TwitterMediaError, match=code):
            await TwitterMediaDownloader(client, resolver="syndication").download(URL, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("response,limit,code", [
    (httpx.Response(200, content=b"<html>challenge</html>", headers={"content-type": "text/html"}), 1000, "unexpected_content_type"),
    (httpx.Response(200, content=b"fake", headers={"content-type": "image/jpeg"}), 1000, "invalid_media_signature"),
    (httpx.Response(200, content=PHOTO, headers={"content-type": "image/jpeg"}), 10, "size_limit"),
    (httpx.Response(200, content=PHOTO, headers={"content-type": "image/jpeg", "content-length": "999"}), 1000, "incomplete_download"),
    (httpx.Response(302, headers={"location": "http://127.0.0.1/secret"}), 1000, "invalid_media_url"),
    (httpx.Response(200, content=b"", headers={"content-type": "image/jpeg"}), 1000, "empty_download"),
])
async def test_bad_media_rolls_back_run(tmp_path, response, limit, code):
    keep = tmp_path / "keep.txt"
    keep.write_text("existing")
    async with client_for(metadata([photo()]), lambda _: response) as client:
        with pytest.raises(TwitterMediaError, match=code):
            await TwitterMediaDownloader(client, max_file_bytes=limit).download(URL, tmp_path)
    assert list(tmp_path.iterdir()) == [keep]


async def test_total_budget_retains_already_downloaded_files(tmp_path):
    async with client_for(metadata([photo(), photo("second")])) as client:
        result = await TwitterMediaDownloader(client, max_total_bytes=len(PHOTO) + 1).download(URL, tmp_path)
    assert len(result.manifest['files']) == 1
    assert result.manifest['all_requested_downloaded'] is False
    assert result.manifest['failures'][0]['index'] == 2
    assert (result.directory / result.manifest['files'][0]['filename']).read_bytes() == PHOTO


async def test_transient_download_retries_are_bounded(tmp_path):
    attempts = 0
    def flaky(_):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503)
        return httpx.Response(200, content=PHOTO, headers={"content-type": "image/jpeg"})
    async with client_for(metadata([photo()]), flaky) as client:
        await TwitterMediaDownloader(client, retry_delay=0).download(URL, tmp_path)
    assert attempts == 3


async def test_exhausted_retries_and_no_retry_on_forbidden(tmp_path):
    for status, expected in [(503, 6), (403, 1), (429, 3)]:
        attempts = 0
        def fail(_):
            nonlocal attempts
            attempts += 1
            return httpx.Response(status)
        async with client_for(metadata([photo()]), fail) as client:
            with pytest.raises(TwitterMediaError, match=f"http_{status}"):
                await TwitterMediaDownloader(client, retry_delay=0).download(URL, tmp_path)
        assert attempts == expected
        assert not list(tmp_path.iterdir())


async def test_original_photo_failure_falls_back_once_to_large_and_records_quality(tmp_path):
    calls=[]
    def respond(request):
        calls.append(request.url)
        if request.url.host=='pbs.twimg.com' and request.url.params['name']=='orig':
            return httpx.Response(503)
        return httpx.Response(200,content=PHOTO if request.url.host=='pbs.twimg.com' else MP4,
            headers={'content-type':'image/jpeg' if request.url.host=='pbs.twimg.com' else 'video/mp4'})
    async with client_for(metadata([photo(),video()]),respond) as client:
        result=await TwitterMediaDownloader(client,retry_delay=0).download(URL,tmp_path)
    first,second=result.manifest['files']
    assert first['image_size']=='large' and first['fallback_reason']=='http_503'
    assert 'name=orig' in first['media_url'] and 'name=large' in first['download_url']
    assert 'image_size' not in second
    assert [u.params.get('name') for u in calls]==['orig','orig','orig','large',None]
    assert (result.directory/first['filename']).read_bytes()==PHOTO


async def test_photo_size_limit_fallback_obeys_same_limit(tmp_path):
    small=b'\xff\xd8\xff\xe0small'
    async with client_for(metadata([photo()]),lambda r:httpx.Response(200,
        content=PHOTO if r.url.params['name']=='orig' else small,headers={'content-type':'image/jpeg'})) as client:
        result=await TwitterMediaDownloader(client,max_file_bytes=10,retry_delay=0).download(URL,tmp_path)
    assert result.manifest['total_bytes']==len(small)<=10
    assert result.manifest['files'][0]['fallback_reason']=='size_limit'


async def test_video_http_failure_does_not_use_photo_size_fallback(tmp_path):
    calls=[]
    async with client_for(metadata([video()]),lambda r:calls.append(r) or httpx.Response(503)) as client:
        with pytest.raises(TwitterMediaError,match='http_503'):
            await TwitterMediaDownloader(client,retry_delay=0).download(URL,tmp_path)
    assert len(calls)==3 and not list(tmp_path.iterdir())


async def test_cancel_cleans_partial_run(tmp_path):
    async def cancelled(request):
        if request.url.host == "cdn.syndication.twimg.com":
            return httpx.Response(200, json=metadata([photo()]))
        raise asyncio.CancelledError
    async with httpx.AsyncClient(transport=httpx.MockTransport(cancelled)) as client:
        with pytest.raises(asyncio.CancelledError):
            await TwitterMediaDownloader(client).download(URL, tmp_path)
    assert not list(tmp_path.iterdir())


async def test_whole_job_deadline_covers_network_and_cleanup(tmp_path):
    async def stalled(request):
        if request.url.host == "cdn.syndication.twimg.com":
            return httpx.Response(200, json=metadata([photo()]))
        await asyncio.Event().wait()
    async with httpx.AsyncClient(transport=httpx.MockTransport(stalled)) as client:
        with pytest.raises(TwitterMediaError, match="download_timeout"):
            await TwitterMediaDownloader(client, timeout_seconds=0.02).download(URL, tmp_path)
    assert not list(tmp_path.iterdir())


async def test_streaming_limit_without_content_length(tmp_path):
    class LargeStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield PHOTO
            yield b"x" * 65536
    def large(_):
        return httpx.Response(200, stream=LargeStream(), headers={"content-type": "image/jpeg"})
    async with client_for(metadata([photo()]), large) as client:
        with pytest.raises(TwitterMediaError, match="size_limit"):
            await TwitterMediaDownloader(client, max_file_bytes=1000).download(URL, tmp_path)
    assert not list(tmp_path.iterdir())


async def test_broken_stream_restarts_file_instead_of_appending(tmp_path):
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield PHOTO + b"x" * 65536
            raise httpx.ReadError("private upstream details")
    attempts = 0
    def respond(_):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, stream=BrokenStream(), headers={"content-type": "image/jpeg"})
        return httpx.Response(200, content=PHOTO, headers={"content-type": "image/jpeg"})
    async with client_for(metadata([photo()]), respond) as client:
        result = await TwitterMediaDownloader(client, retry_delay=0).download(URL, tmp_path)
    assert (result.directory / "01_photo.jpg").read_bytes() == PHOTO
    assert attempts == 2


async def test_redirects_remain_inside_cdn_allowlist(tmp_path):
    def redirect(request):
        if request.url.path.endswith("image.jpg"):
            return httpx.Response(302, headers={"location": "/media/final.jpg"})
        return httpx.Response(200, content=PHOTO, headers={"content-type": "image/jpeg"})
    async with client_for(metadata([photo()]), redirect) as client:
        result = await TwitterMediaDownloader(client).download(URL, tmp_path)
    assert result.manifest["files"][0]["download_url"] == "https://pbs.twimg.com/media/final.jpg"


async def test_repeated_downloads_keep_previous_result(tmp_path):
    async with client_for(metadata([photo()])) as client:
        downloader = TwitterMediaDownloader(client)
        first = await downloader.download(URL, tmp_path)
        second = await downloader.download(URL, tmp_path)
    assert first.directory != second.directory
    assert first.manifest_path.is_file() and second.manifest_path.is_file()


async def test_malformed_or_oversized_metadata_does_not_create_output(tmp_path):
    for response, code in [
        (httpx.Response(200, content=b"not json", headers={"content-type": "application/json"}), "invalid_metadata_json"),
        (httpx.Response(200, content=b"{}", headers={"content-type": "application/json", "content-length": "9999999"}), "size_limit"),
        (httpx.Response(200, json=metadata([photo()] * 17)), "invalid_media_metadata"),
    ]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
            with pytest.raises(TwitterMediaError, match=code):
                await TwitterMediaDownloader(client, resolver="syndication").download(URL, tmp_path)
    assert not list(tmp_path.iterdir())


def test_cli_json_success_and_safe_failure(monkeypatch, capsys, tmp_path):
    from scripts import download_twitter_media as cli
    async def success(args):
        assert args.output_dir == tmp_path
        return {"ok": True, "count": 3}
    monkeypatch.setattr(cli, "run", success)
    assert cli.main([URL, "-o", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "count": 3}
    async def failure(args):
        raise TwitterMediaError("network_error") from ValueError("secret upstream body")
    monkeypatch.setattr(cli, "run", failure)
    assert cli.main([URL]) == 1
    output = capsys.readouterr()
    assert json.loads(output.err)["error"] == "network_error"
    assert "secret" not in output.err and not output.out


async def test_second_item_timeout_keeps_committed_first(tmp_path):
    committed = []
    async def commit(item, path):
        committed.append(path.read_bytes())
    async def respond(request):
        if request.url.host == 'cdn.syndication.twimg.com':
            return httpx.Response(200,json=metadata([photo(),photo('stalled')]))
        if 'stalled' in request.url.path:
            await asyncio.Event().wait()
        return httpx.Response(200,content=PHOTO,headers={'content-type':'image/jpeg'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result=await TwitterMediaDownloader(client,timeout_seconds=.04,on_file=commit).download(URL,tmp_path)
    assert committed==[PHOTO] and result.manifest['failures'][0]['error']=='download_timeout'


async def test_nine_mib_photo_does_not_consume_or_rollback_sibling(tmp_path):
    calls=[]
    def respond(request):
        calls.append(request.url.path)
        if 'large' in request.url.path:
            return httpx.Response(200,headers={'content-type':'image/jpeg','content-length':str(9*1024*1024)})
        return httpx.Response(200,content=PHOTO,headers={'content-type':'image/jpeg'})
    async with client_for(metadata([photo(),photo('large')]),respond) as client:
        result=await TwitterMediaDownloader(client).download(URL,tmp_path)
    assert len(result.manifest['files'])==1 and len(calls)==3
    assert result.manifest['failures'][0]['error']=='size_limit'


async def test_video_oversize_variant_falls_back_with_shared_receive_counter(tmp_path):
    calls=[]
    def respond(request):
        calls.append(request.url.path)
        if request.url.path=='/high.mp4':
            return httpx.Response(200,headers={'content-type':'video/mp4','content-length':str(70*1024*1024)})
        return httpx.Response(200,content=MP4,headers={'content-type':'video/mp4'})
    async with client_for(metadata([video()]),respond) as client:
        result=await TwitterMediaDownloader(client,max_file_bytes=64*1024*1024).download(URL,tmp_path)
    assert calls==['/high.mp4','/low.mp4']
    assert result.manifest['received_bytes']==len(MP4) and result.manifest['files'][0]['quality']=='lower_native'


async def test_strict_original_photo_does_not_fallback(tmp_path):
    calls=[]
    async with client_for(metadata([photo()]),lambda r:calls.append(r) or httpx.Response(404)) as client:
        with pytest.raises(TwitterMediaError):await TwitterMediaDownloader(client,original_only=True).download(URL,tmp_path)
    assert len(calls)==1 and calls[0].url.params['name']=='orig'


@pytest.mark.parametrize('reason', ['duration_limit','estimated_size_limit','unsupported_codec'])
async def test_rejected_video_variants_report_precise_safe_reason(tmp_path,reason):
    variant={'url':'https://video.twimg.com/ordinary.mp4','codec':'h264',
        'width':1280,'height':720,'duration_seconds':20,'bitrate':500_000}
    variant.update({'duration_seconds':1201} if reason=='duration_limit' else
        {'bitrate':80_000_000} if reason=='estimated_size_limit' else {'codec':'hevc'})
    observation={'post_id':POST_ID,'identity_status':'matched','protected':False,'media_count':1,
        'media':[{'kind':'video','url':variant['url'],'variants':[variant]}]}
    calls=[]
    async with client_for({},lambda request:calls.append(request) or httpx.Response(500)) as client:
        with pytest.raises(TwitterMediaError) as caught:
            await TwitterMediaDownloader(client,max_file_bytes=64*1024*1024).download_resolved(URL,tmp_path,
                observation,evidence_ref='x-posts-'+'a'*24+':post:0')
    assert caught.value.code=='no_mp4_stream_within_budget'
    assert caught.value.details['stage']=='video_variant_selection'
    assert caught.value.details['rejected_by_reason'][reason]==1
    assert not calls and not list(tmp_path.iterdir())


async def test_unknown_length_oversize_fallback_counts_discarded_bytes(tmp_path):
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield MP4+b'x'*600
            yield b'x'*600
    def respond(request):
        if request.url.path=='/high.mp4':return httpx.Response(200,stream=Stream(),headers={'content-type':'video/mp4'})
        return httpx.Response(200,content=MP4,headers={'content-type':'video/mp4'})
    async with client_for(metadata([video()]),respond) as client:
        result=await TwitterMediaDownloader(client,max_file_bytes=1024,max_total_bytes=4096).download(URL,tmp_path)
    assert result.manifest['files'][0]['download_url'].endswith('/low.mp4')
    assert result.manifest['received_bytes']==len(MP4)*2+1200
