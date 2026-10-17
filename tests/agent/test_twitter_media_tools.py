import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from app.agent.twitter_media import build_twitter_media_tool
from app.core.downloaded_media import DownloadedMediaCache
from test_web_image_tools import Runner, Guard, tools_for
from test_autonomous_social import image_bytes

POST = '900000000000000001'
URL = 'https://x.com/synthetic/status/' + POST
VIDEO = b'\x00\x00\x00\x18ftypisom' + b'synthetic' * 30


def test_download_client_uses_explicit_mihomo_instead_of_environment_proxy(monkeypatch):
    from app.agent.twitter_media import download_client
    calls=[]
    monkeypatch.setenv('HTTPS_PROXY','http://127.0.0.1:1')
    monkeypatch.setattr('app.agent.twitter_media.httpx.AsyncClient',lambda **kwargs:calls.append(kwargs))
    download_client(SimpleNamespace(web_page_proxy_url='http://127.0.0.1:18797'))
    assert calls==[{'proxy':'http://127.0.0.1:18797','trust_env':False,'follow_redirects':False}]


def provider(monkeypatch, *, video=VIDEO, check=None):
    calls = []
    def respond(request):
        calls.append(str(request.url))
        if check:
            check(request)
        if request.url.host == 'cdn.syndication.twimg.com':
            return httpx.Response(200, json={'id_str': POST, 'mediaDetails': [
                {'type': 'photo', 'media_url_https': 'https://pbs.twimg.com/media/synthetic.png'},
                {'type': 'video', 'video_info': {'variants': [{'content_type': 'video/mp4',
                    'bitrate': 2000, 'url': 'https://video.twimg.com/synthetic.mp4'}]}},
                {'type': 'animated_gif', 'video_info': {'variants': [{'content_type': 'video/mp4',
                    'url': 'https://video.twimg.com/synthetic-gif.mp4'}]}}]})
        pixels = request.url.host == 'pbs.twimg.com'
        return httpx.Response(200, content=image_bytes() if pixels else video,
            headers={'Content-Type': 'image/png' if pixels else 'video/mp4'})
    monkeypatch.setattr('app.agent.twitter_media.download_client',
        lambda _: httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    return calls


async def test_mixed_order_scoped_cache_and_no_standalone_directory(sqlite_engine, tmp_path, monkeypatch):
    _, store, guard, http = tools_for(monkeypatch, sqlite_engine, tmp_path, lambda _: None)
    calls = provider(monkeypatch)
    monkeypatch.setattr('app.agent.twitter_media.validate_video', lambda _: {'codec': 'h264', 'duration_seconds': 1})
    tool = build_twitter_media_tool(runner=Runner(), guard=guard, store=store)
    try:
        outcome = await tool.handler({'ref_id': URL})
        assert outcome.status == 'ok', outcome.data
        data = outcome.data
        assert [item['kind'] for item in data['media']] == ['photo', 'video', 'gif']
        assert data['media_artifact_ids'] == [item['artifact_id'] for item in data['media']]
        cache = DownloadedMediaCache.for_store(store)
        assert store.get(data['artifact_ids'][0]).read_bytes() == image_bytes()
        assert [cache.video(store, item)[0].read_bytes() for item in data['video_artifact_ids']] == [VIDEO, VIDEO]
        assert not list(store.root.glob('.twitter-download-*')) and not (tmp_path/'twitter_media').exists()
        assert (await tool.handler({'ref_id': URL})).data['cached_in_turn']
        assert len(calls) == 4
    finally:
        http.close()


async def test_large_photo_fallback_reaches_scoped_artifacts_and_quality_result(sqlite_engine,tmp_path,monkeypatch):
    _,store,guard,http=tools_for(monkeypatch,sqlite_engine,tmp_path,lambda _:None)
    calls=[]
    def respond(request):
        if request.url.host=='cdn.syndication.twimg.com':
            return httpx.Response(200,json={'id_str':POST,'mediaDetails':[
                {'type':'photo','media_url_https':'https://pbs.twimg.com/media/synthetic.png'}]})
        calls.append(request.url.params['name'])
        return (httpx.Response(404) if request.url.params['name']=='orig' else
            httpx.Response(200,content=image_bytes(),headers={'content-type':'image/png'}))
    monkeypatch.setattr('app.agent.twitter_media.download_client',lambda _:httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    try:
        outcome=await build_twitter_media_tool(runner=Runner(),guard=guard,store=store).handler({'ref_id':URL})
        assert outcome.status=='ok' and calls==['orig','large']
        assert outcome.data['quality_fallbacks']==[{'index':1,'image_size':'large','reason':'http_404'}]
        assert outcome.data['media'][0]['image_size']=='large' and '不能声称全部原图' in outcome.data['guidance']
        assert store.get(outcome.data['artifact_ids'][0]).read_bytes()==image_bytes()
        assert not list(store.root.glob('.twitter-download-*'))
    finally:http.close()


@pytest.mark.parametrize('failure', ['invalid_video', 'revoked', 'cancelled'])
async def test_failed_mixed_download_removes_staging_and_registered_siblings(sqlite_engine, tmp_path, monkeypatch, failure):
    _, store, guard, http = tools_for(monkeypatch, sqlite_engine, tmp_path, lambda _: None)
    provider(monkeypatch)
    def validate(_):
        if failure == 'revoked':
            guard.valid = False
            raise RuntimeError('source revoked')
        if failure == 'cancelled':
            raise asyncio.CancelledError()
        raise ValueError('web_video_invalid')
    monkeypatch.setattr('app.agent.twitter_media.validate_video', validate)
    tool = build_twitter_media_tool(runner=Runner(), guard=guard, store=store)
    try:
        if failure == 'invalid_video':
            result = await tool.handler({'ref_id': URL})
            assert result.status == 'ok' and result.data['downloaded_count'] == 1
            assert result.data['all_requested_downloaded'] is False
            assert store.get(result.data['artifact_ids'][0]).read_bytes() == image_bytes()
        else:
            with pytest.raises(RuntimeError if failure == 'revoked' else asyncio.CancelledError):
                await tool.handler({'ref_id': URL})
        assert len(list(store.root.glob('art-*.meta.json'))) == (1 if failure == 'invalid_video' else 0)
        assert not list(store.root.glob('.twitter-download-*'))
        assert not list(DownloadedMediaCache.for_store(store).root.glob('webvid-*'))
    finally:
        http.close()


async def test_x_tool_only_enqueues_and_invalid_urls_do_not_download(sqlite_engine, tmp_path, monkeypatch):
    _, store, guard, http = tools_for(monkeypatch, sqlite_engine, tmp_path, lambda _: None)
    queued=[]
    async def submit(*args):
        from app.agent.tool_agent import ToolOutcome
        queued.append(args)
        return ToolOutcome({'status':'queued','job_id':1})
    tool = build_twitter_media_tool(runner=Runner(), guard=guard, store=store, submit=submit)
    try:
        assert (await tool.handler({'ref_id':'https://example.com/private'})).status == 'blocked'
        assert (await tool.handler({'ref_id':URL+'/photo/1'})).data['status']=='queued'
        assert queued == [('download_twitter_media', {'ref_id': 'https://x.com/i/status/' + POST}, {})]
    finally:http.close()
