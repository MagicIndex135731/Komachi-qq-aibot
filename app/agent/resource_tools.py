"""Batch deterministic post inspection with compact, recoverable observations."""
from __future__ import annotations

import asyncio
import json
import time
from threading import Event
from urllib.parse import urlsplit

import httpx

from app.agent.tool_agent import AgentTool, ToolOutcome
from app.core.url_policy import public_web_url
from app.providers.resource_inspector import PublicResourceClient
from app.providers.web_images import WebImageError
from app.providers.twitter_media import TwitterMediaError


def build_resource_tool(client, *, runner, guard, workspace=None, resolve=public_web_url, settings=None) -> AgentTool:
    from app.providers.x_crawler_client import XCrawlerClient
    crawler = XCrawlerClient.from_settings(settings)
    completed = {}
    lock = asyncio.Lock()

    async def inspect(args):
        async with lock:
            guard.require_valid()
            try:
                refs = args['ref_ids']
                if not isinstance(refs, list) or not 1 <= len(refs) <= 6:
                    raise ValueError()
                resolved = [resolve(value) for value in refs]
                urls = list(dict.fromkeys(resolved))
                positions = {url: resolved.index(url) + 1 for url in urls}
            except (ValueError, KeyError, TypeError, AttributeError):
                return ToolOutcome({'error': 'invalid_resource_targets'}, 'blocked')
            results = []
            deadline, cancelled = time.monotonic() + (120 if crawler else 60), Event()
            def check():
                guard.require_valid()
                if cancelled.is_set():raise WebImageError('web_resource_cancelled')
            try:
                for url in urls:
                    check()
                    if url in completed:
                        old = completed[url]
                        results.append({k: v for k, v in old.items() if k in
                            {'source_url', 'status', 'error', 'task_id', 'evidence_id'}} | {'cached_in_turn': True})
                        continue
                    if len(completed) >= 6 or time.monotonic() >= deadline:
                        results.append({'source_url': url, 'status': 'blocked', 'error': 'web_resource_turn_budget'})
                        continue
                    try:
                        use_crawler = crawler and urlsplit(url).hostname in {'x.com','www.x.com','twitter.com','www.twitter.com'}
                        async with asyncio.timeout(max(0.01, min(57 if use_crawler else 27, deadline - time.monotonic()))):
                            if use_crawler:
                                full = await runner.run(crawler.inspect, url, check=check)
                                full = {**full, 'title': full.get('text','')[:240],
                                    'media': [{**m, 'kind': 'image' if m['kind']=='photo' else 'video',
                                        'role': 'post_image' if m['kind']=='photo' else 'post_video'} for m in full['media']]}
                            else:
                                full = await runner.run(PublicResourceClient(client).inspect, url, check=check, deadline=deadline)
                        check()
                        summary = {k: full[k] for k in ('source_url', 'post_id', 'title', 'author', 'published_at',
                            'identity_status', 'author_status', 'date_status', 'media_count') if k in full}
                        summary.update(status='ok', text=str(full.get('text', ''))[:240],
                            media_roles=list(dict.fromkeys(m['role'] for m in full.get('media', []))))
                        if workspace is not None:
                            content = json.dumps(full, ensure_ascii=False, separators=(',', ':'))
                            summary['task_id'] = workspace.task_id
                            summary['evidence_id'] = workspace.save_evidence({'content': content,
                                'sources': [{'source': full['source_url'], 'title': full['title']}],
                                'resources': [summary.copy()], 'content_kind': 'post_metadata',
                                'trust': 'untrusted_external_sources'})
                    except (httpx.HTTPError, ValueError, OSError, TimeoutError,
                            KeyError, TypeError, AttributeError, IndexError, RecursionError) as exc:
                        check()
                        code = str(exc) if isinstance(exc, (WebImageError,TwitterMediaError)) else (
                            f'web_resource_http_{exc.response.status_code}' if isinstance(exc, httpx.HTTPStatusError)
                            else 'web_resource_invalid_metadata' if isinstance(exc,
                                (KeyError, TypeError, AttributeError, IndexError, RecursionError))
                            else 'web_resource_unavailable')
                        summary = {'source_url': url, 'status': 'failed', 'error': code[:100]}
                    completed[url] = summary
                    results.append(dict(summary))
            finally:
                cancelled.set()
            guard.require_valid()
            ok = any(r.get('status') == 'ok' for r in results)
            codes = {r.get('error') for r in results if r.get('error')}
            for requested, row in zip(urls, results):
                row['input_index'] = positions[requested]
                if len(row.get('source_url', '')) > 512:
                    # The caller already supplied this exact URL. Preserve it
                    # in the archive/index, not twice in the model context.
                    row.pop('source_url')
            return ToolOutcome({'resources': results,
                **({} if ok else {'error': next(iter(codes)) if len(codes) == 1 else 'web_resources_unavailable'}),
                'trust': 'untrusted_external_sources', 'content_kind': 'post_metadata',
                'context_retention': 'compact_v1', 'coverage': 'inspected_candidates_only; latest_not_proven',
                'guidance': '核对作者/日期/正文与资源角色。原文/媒体URL按task_id/evidence_id续读；X用download_twitter_media，视频用download_web_video，图册用list_web_images。cover/page_image不证明目标正文；失败不要同轮反复撞站，不重复尝试被限流来源。'},
                'ok' if ok else 'failed')

    return AgentTool('inspect_web_resources',
        '批量核验最多6个原帖URL/ref_id，返回作者日期、正文摘要、媒体数量/角色及证据句柄；配置X会话时自动使用爬虫读取，否则公开解析，不调用模型或下载像素。找帖后可核验多个候选再下载。',
        {'type': 'object', 'properties': {'ref_ids': {'type': 'array', 'minItems': 1, 'maxItems': 6,
            'items': {'type': 'string', 'minLength': 1, 'maxLength': 2000}}},
         'required': ['ref_ids'], 'additionalProperties': False}, inspect)
