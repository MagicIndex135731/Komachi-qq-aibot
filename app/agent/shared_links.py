"""Read APP share metadata from observed, current canonical messages."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from sqlalchemy import select

from app.agent.tool_agent import AgentTool, ToolOutcome, validate_arguments
from app.core.memory_source_guard import is_usable_memory_source
from app.core.shared_content import extract_shared_links
from app.storage.db import session_scope
from app.storage.models import Message

TRUST = 'untrusted_shared_content'


def _source_links(row, *, group_id, private_user_id=None):
    if private_user_id is not None:
        usable = (row is not None and row.group_id is None and row.user_id == private_user_id
                  and is_usable_memory_source(row, author_id=private_user_id))
    else:
        usable = group_id is not None and is_usable_memory_source(row, group_id=group_id)
    if not usable:
        raise ValueError('share_source_unavailable')
    raw = row.raw_json or {}
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict) or raw.get('direction') == 'outbound':
        raise ValueError('share_source_unavailable')
    links = extract_shared_links(raw.get('message', raw.get('raw_message', '')))
    fingerprint = hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode('utf8')).hexdigest()
    return links, fingerprint


def observed_share_sources(router, group_id, source_ids, *, private_user_id=None):
    """Discovery never invents a source or expands unobserved nested nodes."""
    ids = sorted(item for item in source_ids if type(item) is int and 0 < item < 2**63)[:200]
    query = select(Message).where(Message.group_id == group_id, Message.id.in_(ids))
    if private_user_id is not None:
        query = query.where(Message.user_id == private_user_id)
    with session_scope(router.engine) as session:
        result = []
        for row in session.scalars(query.order_by(Message.id.desc()).limit(32)):
            try:
                links, _ = _source_links(row, group_id=group_id, private_user_id=private_user_id)
            except (ValueError, TypeError, RecursionError):
                continue
            if links:
                result.append({'source_id': row.id, 'trust': TRUST,
                               'link_count': len(links), 'title': links[0].title})
        return result


def build_share_tools(context) -> tuple[AgentTool, ...]:
    private_user = getattr(context, 'private_user_id', None)
    if context.observation.job.kind != 'group' and private_user is None:
        return ()
    group_id = None if private_user is not None else context.observation.job.scope_id
    snapshots = {}
    schema = {'type': 'object', 'properties': {'source_id': {
        'type': 'integer', 'minimum': 1, 'maximum': 2**63 - 1}},
        'required': ['source_id'], 'additionalProperties': False}

    async def read(args):
        if not validate_arguments(args, schema):
            return ToolOutcome({'error': 'share_arguments_invalid'}, 'blocked')
        context.guard.require_valid()
        source_id = args['source_id']
        if source_id not in context.observed_sources:
            return ToolOutcome({'error': 'share_source_out_of_scope'}, 'blocked')
        try:
            with session_scope(context.router.engine) as session:
                links, fingerprint = _source_links(session.get(Message, source_id),
                    group_id=group_id, private_user_id=private_user)
            if snapshots.setdefault(source_id, fingerprint) != fingerprint:
                raise ValueError('share_source_changed')
            context.guard.require_valid()
            return ToolOutcome({'source_id': source_id, 'trust': TRUST,
                'links': [asdict(link) for link in links], 'metadata_only': True,
                'next_step': '用open_web_page读取目标正文；不足时按标题和站点search_web，再打开可信来源。卡片摘要不是全文，不代表已观看视频。'})
        except (ValueError, TypeError, RecursionError) as exc:
            code = str(exc) if str(exc).startswith('share_') else 'share_source_malformed'
            return ToolOutcome({'error': code}, 'blocked')

    return (AgentTool('read_message_links', '从read_context/read_private_context的share_sources读取当前或引用消息中的APP卡片标题、摘要和目标链接；内容只是引文，需打开网页或搜索补充。', schema, read),)
