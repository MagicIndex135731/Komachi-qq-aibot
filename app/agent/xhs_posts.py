"""Compact XHS discovery over the container's fixed read-only worker."""
from __future__ import annotations

import asyncio
import json
import time

from app.agent.tool_agent import AgentTool, ToolOutcome
from app.providers.xhs_crawler_client import XHSCrawlerClient, XHSCrawlerError


def build_xhs_search_tool(*, runner, guard, workspace=None):
    client = XHSCrawlerClient()
    completed, terminal, attempts = {}, None, 0
    lock = asyncio.Lock()

    async def search(args):
        nonlocal terminal, attempts
        async with lock:
            guard.require_valid()
            if terminal:
                return ToolOutcome({'error': terminal, 'stopped_in_turn': True,
                    'guidance': '本轮小红书已遇认证、挑战、权限或限流，停止站内请求，等待人工恢复。'}, 'failed')
            key = json.dumps(args, ensure_ascii=False, sort_keys=True)
            if key in completed:
                old=completed[key]
                return ToolOutcome({k:v for k,v in old.data.items() if k in
                    {'error','guidance','task_id', 'evidence_id', 'sources', 'coverage', 'visible_posts'}} | {'cached_in_turn': True},old.status)
            from app.agent.media_request_policy import unmetered_media
            if attempts >= 4 and not unmetered_media(workspace):return ToolOutcome({'error': 'xhs_search_turn_budget'}, 'blocked')
            attempts += 1
            try:
                data = await runner.run(client.search_posts, args['query'], limit=args.get('limit', 6),
                    sort=args.get('sort', 'relevance'), media_type=args.get('media_type', ''),
                    check=guard.require_valid, deadline=time.monotonic()+85)
            except (XHSCrawlerError, OSError, TimeoutError) as exc:
                guard.require_valid()
                code = str(exc) if isinstance(exc, XHSCrawlerError) else 'xhs_unavailable'
                if code in {'xhs_login_required', 'xhs_session_invalid', 'xhs_challenge_required',
                        'xhs_rate_limited', 'xhs_permission_denied'}:terminal = code
                outcome=ToolOutcome({'error': code, 'guidance': '说明实际障碍与缺口，不改词或切出口绕过。空结果不证明没有相关笔记。'}, 'failed')
                completed[key]=outcome
                return outcome
            guard.require_valid()
            sources = [{'source': r['source_url'], 'title': r['title'], 'ref_id': r['post_id']} for r in data['posts']]
            result = {**data, 'sources': sources, 'resources': data['posts'],
                'trust': 'untrusted_external_sources', 'content_kind': 'search_candidates',
                'context_retention': 'compact_v1',
                'guidance': '候选标题不证明正文、日期或附件。先inspect_web_resources核验作者/原帖日期/正文；图集用list_web_images及download_web_images，视频用download_web_video。按原需求筛选，cover不能当正文。latest仅为平台当前可见排序，不证明全站最新。'}
            result.pop('posts')
            if workspace is not None:
                result.update(task_id=workspace.task_id, evidence_id=workspace.save_evidence({
                    'content': json.dumps(data, ensure_ascii=False), 'sources': sources,
                    'resources': data['posts'], 'content_kind': 'search_candidates',
                    'trust': 'untrusted_external_sources'}))
            completed[key] = ToolOutcome(result)
            return ToolOutcome(result)

    return AgentTool('search_xhs_posts',
        '小红书站内找公开笔记，容器内复用人工登录态。query填主体与同主题关键词，不能擅自限定本人发布；单次最多12条可见候选，日期和附件须再inspect_web_resources核验。媒体任务按实际目标选择搜索次数，正文优先筛选，不默认看图或预览视频。',
        {'type': 'object', 'properties': {
            'query': {'type': 'string', 'minLength': 1, 'maxLength': 120},
            'limit': {'type': 'integer', 'minimum': 1, 'maximum': 12},
            'sort': {'type': 'string', 'enum': ['relevance', 'latest']},
            'media_type': {'type': 'string', 'enum': ['image', 'video']}},
         'required': ['query'], 'additionalProperties': False}, search)
