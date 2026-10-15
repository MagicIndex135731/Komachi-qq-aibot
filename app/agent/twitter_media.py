"""X single-post downloads use the shared media queue, artifacts and cache."""
from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
from uuid import uuid4

import httpx

from app.agent.tool_agent import AgentTool, ToolOutcome
from app.core.downloaded_media import DownloadedMediaCache
from app.core.video_file import MAX_VIDEO_BYTES, validate_video
from app.providers.twitter_media import TwitterMediaDownloader, TwitterMediaError, parse_post_url

MAX_TOTAL_BYTES = 128 * 1024 * 1024


def download_client(settings):
    # No search API credentials or browser cookies are sent to media hosts.
    return httpx.AsyncClient(proxy=getattr(settings, 'web_page_proxy_url', '') or None,
                             trust_env=False, follow_redirects=False)


def build_twitter_media_tool(*, runner, guard, store, settings=None, submit=None):
    cache = DownloadedMediaCache.for_store(store, settings)
    results = {}
    lock = asyncio.Lock()

    async def checked(args):
        async with lock:
            return await download(args)

    async def download(args):
        guard.require_valid()
        try:
            url = parse_post_url(args['ref_id']).source_url
        except (KeyError, ValueError, TypeError):
            return ToolOutcome({'error': 'invalid_twitter_post_url'}, 'blocked')
        if submit is not None:
            return await submit('download_twitter_media', {**args, 'ref_id': url}, {})
        if url in results:
            await runner.run(cache.pin, results[url]['media_artifact_ids'])
            return ToolOutcome({**results[url], 'cached_in_turn': True})
        if len(results) >= 2:
            return ToolOutcome({'error': 'twitter_media_turn_budget'}, 'blocked')
        created, images, videos, media = [], [], [], []
        try:
            # Include overhead and reserve before downloading a single byte.
            reservation_bytes = min(cache.max_bytes, MAX_TOTAL_BYTES + 2 * 1024 * 1024)
            transfer_bytes = min(MAX_TOTAL_BYTES, max(1, reservation_bytes - 2 * 1024 * 1024))
            with cache.reserve(reservation_bytes):
                store.root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix='.twitter-download-', dir=store.root) as staging:
                    async with download_client(settings) as client:
                        downloaded = await TwitterMediaDownloader(client, max_file_bytes=min(MAX_VIDEO_BYTES, transfer_bytes),
                            max_total_bytes=transfer_bytes, timeout_seconds=180,
                            check=guard.require_valid).download(url, Path(staging))
                    for item in downloaded.manifest['files']:
                        guard.require_valid()
                        path = downloaded.directory / item['filename']
                        if item['kind'] == 'photo':
                            identifier = await runner.run(cache.register, store, path, 'web_page_image')
                            path.unlink(missing_ok=True)
                            images.append(identifier)
                        else:
                            info = await runner.run(validate_video, path)
                            info.update(bytes=item['bytes'], page_url=url, media_kind=item['kind'])
                            pending = cache.root / ('video-' + uuid4().hex + '.part')
                            try:
                                path.replace(pending)
                                identifier = await runner.run(cache.save_video, store, pending, info)
                            finally:
                                pending.unlink(missing_ok=True)
                            videos.append(identifier)
                        created.append(identifier)
                        media.append({'artifact_id': identifier, 'kind': item['kind'], 'index': item['index']})
                    guard.require_valid()
                    result = {'source_url': url, 'post_id': downloaded.manifest['post_id'],
                        'media_artifact_ids': created[:], 'artifact_ids': images, 'video_artifact_ids': videos,
                        'media': media, 'downloaded_count': len(created), 'total_bytes': downloaded.manifest['total_bytes'],
                        'all_requested_downloaded': True, 'trust': 'untrusted_external_sources',
                        'guidance': '真实原帖媒体，动图为MP4。按顺序将media_artifact_ids原样交给send_group/send_private并设as_forward=true打包；不要拆成连续单条。下载不代表QQ送达，以发送回执为准。'}
                    results[url] = result
                    return ToolOutcome(result)
        except BaseException as exc:
            if created:
                await asyncio.shield(runner.run(cache.discard, tuple(created)))
            if not isinstance(exc, (TwitterMediaError, httpx.HTTPError, ValueError, OSError)):
                raise
            guard.require_valid()
            code = exc.code if isinstance(exc, TwitterMediaError) else str(exc) if isinstance(exc, ValueError) else 'twitter_media_download_failed'
            return ToolOutcome({'error': code, 'guidance': '本次媒体未完成，不能声称已发送；可能是私密/删除/限制、接口验证、文件格式或容量上限。失败半成品已清理。'}, 'failed')

    from app.agent.media_tasks import download_delivery_properties
    return AgentTool('download_twitter_media',
        '后台下载用户指定X/Twitter公开单帖的全部原图、视频和动图；最多16项/128MiB，单视频64MiB/10分钟。立即返回任务号；完成后用media_artifact_ids和as_forward=true按序合并转发，不重复排队，不展开引用帖或读取登录态。',
        {'type': 'object', 'properties': {'ref_id': {'type': 'string', 'minLength': 1, 'maxLength': 2000},
            **download_delivery_properties()},
         'required': ['ref_id'], 'additionalProperties': False}, checked)
