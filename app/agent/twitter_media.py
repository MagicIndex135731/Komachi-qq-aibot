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
    # Authenticated metadata and credential-free CDN transfers share X's egress.
    proxy = (getattr(settings, 'x_crawler_proxy_url', 'http://127.0.0.1:18797')
        if getattr(settings, 'x_crawler_session_file', '') else getattr(settings, 'web_page_proxy_url', ''))
    return httpx.AsyncClient(proxy=proxy or None,
                             trust_env=False, follow_redirects=False)


def build_twitter_media_tool(*, runner, guard, store, settings=None, submit=None, transfer_budget=None):
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
        created, images, videos, media, quality_fallbacks = [], [], [], [], []
        try:
            # Include overhead and reserve before downloading a single byte.
            reservation_bytes = min(cache.max_bytes, MAX_TOTAL_BYTES + 2 * 1024 * 1024)
            transfer_bytes = min(MAX_TOTAL_BYTES, max(1, reservation_bytes - 2 * 1024 * 1024))
            with cache.reserve(reservation_bytes):
                store.root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix='.twitter-download-', dir=store.root) as staging:
                    async def commit(item, path):
                        guard.require_valid()
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
                        logical = await runner.run(store.bind_media_identity, identifier, url + ':' + str(item['index']))
                        created.append(identifier)
                        media.append({'artifact_id': identifier, 'kind': item['kind'], 'index': item['index'], 'status': 'ready', 'bytes': item['bytes'], 'sha256':item['sha256'],
                            'logical_media_id': logical, 'source_ref': url, **{k:item[k] for k in ('quality','fallback_reason') if k in item},
                            **({'image_size':item.get('image_size','orig')} if item['kind']=='photo' else {})})
                        if item.get('fallback_reason'):
                            quality_fallbacks.append({'index':item['index'],'image_size':item['image_size'],'reason':item['fallback_reason']})
                    from app.providers.x_crawler_client import XCrawlerClient
                    async with download_client(settings) as client:
                        downloaded = await TwitterMediaDownloader(client, max_file_bytes=min(MAX_VIDEO_BYTES, transfer_bytes),
                            max_total_bytes=transfer_bytes, timeout_seconds=180, on_file=commit,transfer_budget=transfer_budget, original_only=args.get('quality')=='original_only',
                            check=guard.require_valid, metadata_client=XCrawlerClient.from_settings(settings)).download(url, Path(staging))
                    guard.require_valid()
                    result = {'source_url': url, 'post_id': downloaded.manifest['post_id'],
                        'post': downloaded.manifest.get('post', {}),
                        'media_artifact_ids': created[:], 'artifact_ids': images, 'video_artifact_ids': videos,
                        'media': media, 'quality_fallbacks':quality_fallbacks,
                        'received_bytes':downloaded.manifest.get('received_bytes',downloaded.manifest['total_bytes']),
                        'downloaded_count': len(created), 'total_bytes': downloaded.manifest['total_bytes'],
                        'all_requested_downloaded': downloaded.manifest.get('all_requested_downloaded', True),
                        'requested_count': downloaded.manifest.get('requested_count', len(created)),
                        'failures': downloaded.manifest.get('failures', []), 'trust': 'untrusted_external_sources',
                        'guidance': '真实原帖媒体，动图为MP4。按顺序将media_artifact_ids原样交给send_group/send_private并设as_forward=true打包；不要拆成连续单条。下载不代表QQ送达，以发送回执为准。'}
                    if not result['all_requested_downloaded']:
                        result['guidance'] += '部分项目失败，只交付已成功项并说明缺口。'
                    if quality_fallbacks:
                        result['guidance']+='部分图片原图不可用，已降一档为large；须如实说明，不能声称全部原图。'
                    results[url] = result
                    return ToolOutcome(result)
        except BaseException as exc:
            if not isinstance(exc, (TwitterMediaError, httpx.HTTPError, ValueError, OSError)):
                if created:
                    await asyncio.shield(runner.run(cache.discard, tuple(created)))
                raise
            guard.require_valid()
            if created:
                return ToolOutcome({'source_url':url, 'media_artifact_ids':created, 'artifact_ids':images,
                    'video_artifact_ids':videos, 'media':media, 'downloaded_count':len(created),
                    'all_requested_downloaded':False, 'failures':[{'error':'batch_interrupted'}],
                    'guidance':'批次未全部完成，已保留验证成功的媒体；发送子集并说明缺口。'})
            guard.require_valid()
            code = exc.code if isinstance(exc, TwitterMediaError) else str(exc) if isinstance(exc, ValueError) else 'twitter_media_download_failed'
            return ToolOutcome({'error': code, 'guidance': '本次媒体未完成，不能声称已发送；可能是私密/删除/限制、接口验证、文件格式或容量上限。失败半成品已清理。'}, 'failed')

    from app.agent.media_tasks import download_delivery_properties
    return AgentTool('download_twitter_media',
        '后台下载当前获准任务中发现的X/Twitter公开原帖全部图片、视频和动图；图片优先原图，原图不可用时仅降一档large并在结果注明。已有原帖即可用，主题检索不限本人用search_x_posts，明确限定发布者才list_x_posts。配置X会话时自动使用爬虫读取元数据。最多16项/128MiB，单视频64MiB/10分钟；返回任务号，按原顺序交付，不展开引用帖。',
        {'type': 'object', 'properties': {'ref_id': {'type': 'string', 'minLength': 1, 'maxLength': 2000,
            'description':'本轮发现的原帖source_url、搜索ref_id或已观察到的post_id；优先完整原帖URL'},
            **download_delivery_properties()},
         'required': ['ref_id'], 'additionalProperties': False}, checked)
