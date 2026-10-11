"""Read APP share metadata from observed, current canonical messages."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
from urllib.parse import unquote, urlsplit

from sqlalchemy import select

from app.agent.tool_agent import AgentTool, ToolOutcome, validate_arguments
from app.core.memory_source_guard import is_usable_memory_source
from app.core.message_content import extract_reply_to_msg_id
from app.core.url_policy import public_web_url
from app.core.shared_content import extract_shared_links
from app.storage.db import session_scope
from app.storage.models import Message

TRUST = 'untrusted_shared_content'
_APP_KINDS = frozenset({'share', 'json', 'xml'})
_READ_INTENT = re.compile(r'看(?:看|一下|下)?(?!过|到|见|起来|完|着)|打开(?!过)|读取|读一下|评价|评一评|评论|分析|总结|解读|讲讲|介绍|了解|查查|read|review|summari[sz]e|analyse|analyze|open', re.I)
_NEGATIVE = re.compile(r'(?:不要|不用|不必|无需|别|不想|不需要|不是(?:让|要|请)(?:你)?).{0,8}(?:看|打开|读|评价|评论|分析|总结|联网|搜索)|(?:do not|don.t)\s+(?:read|open|search|review)', re.I)
_WEB_INTENT = re.compile(r'联网|上网|网络搜索|网上(?:查|搜)|搜索|检索|搜一下|查一下|查查|查找|查一查|帮我(?:查|搜)|查询|查证|核实|是真的吗|真假|天气|新闻|时事|资讯|股价|汇率|实时|最新(?:消息|版本|价格)|search\s+(?:the\s+)?web|web\s+search|weather|latest\s+(?:news|version)|current\s+(?:news|price)', re.I)


def _raw(row):
    raw = row.raw_json or {}
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict) or raw.get('direction') == 'outbound':
        raise ValueError('share_source_unavailable')
    return raw


def _segments(raw):
    message = raw.get('message', raw.get('raw_message', ''))
    if isinstance(message, str):
        return [{'type': 'text', 'data': {'text': message[:65536]}}]
    return [item for item in message[:100] if isinstance(item, dict)] if isinstance(message, list) else []


def _authored_text(raw):
    """Only transport text belongs to the requester; APP fields are quotations."""
    return ' '.join(item.get('data', {}).get('text', '')[:4000]
                    for item in _segments(raw) if item.get('type') == 'text'
                    and isinstance(item.get('data'), dict)
                    and isinstance(item['data'].get('text'), str))[:4000]


def _app_links(raw):
    return extract_shared_links([item for item in _segments(raw) if item.get('type') in _APP_KINDS])


def _fingerprint(row):
    # Authority fields are ledger data too, not just the raw transport body.
    return hashlib.sha256(json.dumps([row.id, row.group_id, row.user_id, row.platform_msg_id,
        row.reply_to_msg_id, row.mentioned_bot, _raw(row)], sort_keys=True,
        ensure_ascii=False).encode('utf8')).hexdigest()


def share_context_text(raw, plain_text):
    """Keep cheap card discovery in chat; release links only through a grant."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return plain_text
    if not isinstance(raw, dict):
        return plain_text
    links = _app_links(raw)
    if not links:
        return plain_text
    titles = list(dict.fromkeys(link.title or '外链' for link in links))[:3]
    return '\n'.join(filter(None, [_authored_text(raw),
        *('[APP分享：' + re.sub(r'https?://\S+', '[链接]', title)[:160] + ']' for title in titles)]))[:700]


@dataclass(frozen=True, slots=True)
class _ShareGrant:
    trigger_id: int
    target_id: int | None
    trigger_fingerprint: str
    target_fingerprint: str | None


class ShareReadAccess:
    """Turn-local canonical request capabilities, never inferred from history."""

    def __init__(self, router, *, group_id, trigger_source_ids=(), private_user_id=None, bot_qq=None):
        self.router, self.group_id, self.private_user_id, self.bot_qq = router, group_id, private_user_id, bot_qq
        self._grants = []
        ids = list(dict.fromkeys(item for item in trigger_source_ids
                                if type(item) is int and 0 < item < 2**63))[:8]
        with session_scope(router.engine) as session:
            for source_id in ids:
                try:
                    row = session.get(Message, source_id)
                    if not self._usable(row):
                        continue
                    raw = _raw(row)
                    text = _authored_text(raw)
                    if not text or _NEGATIVE.search(text):
                        continue
                    quoted = self._quoted(session, row)
                    quoted_links = _app_links(_raw(quoted)) if self._usable(quoted) else ()
                    if quoted_links:
                        if _READ_INTENT.search(text) and self._mentioned(raw, row):
                            self._grants.append(_ShareGrant(row.id, quoted.id, _fingerprint(row), _fingerprint(quoted)))
                    elif not _app_links(raw) and (_WEB_INTENT.search(text) or
                            re.search(r'https?://', text) and _READ_INTENT.search(text)):
                        self._grants.append(_ShareGrant(row.id, None, _fingerprint(row), None))
                except (ValueError, TypeError, RecursionError):
                    continue

    def _usable(self, row):
        if self.private_user_id is not None:
            return row is not None and row.group_id is None and is_usable_memory_source(row, author_id=self.private_user_id)
        return self.group_id is not None and is_usable_memory_source(row, group_id=self.group_id)

    def _mentioned(self, raw, row):
        if self.private_user_id is not None:
            return True
        return bool(row.mentioned_bot and self.bot_qq is not None and any(
            item.get('type') == 'at' and isinstance(item.get('data'), dict)
            and str(item['data'].get('qq', '')) == str(self.bot_qq) for item in _segments(raw)))

    def _quoted(self, session, row):
        raw = _raw(row)
        if (not row.reply_to_msg_id or extract_reply_to_msg_id(
                raw.get('message', raw.get('raw_message', ''))) != row.reply_to_msg_id):
            return None
        platform_id = (f'private-inbound-{self.private_user_id}-{row.reply_to_msg_id}'
                       if self.private_user_id is not None else row.reply_to_msg_id)
        return session.scalar(select(Message).where(Message.group_id == self.group_id,
                                                     Message.platform_msg_id == platform_id))

    def _live_grants(self):
        result = []
        with session_scope(self.router.engine) as session:
            for grant in self._grants:
                try:
                    trigger = session.get(Message, grant.trigger_id)
                    if not self._usable(trigger) or _fingerprint(trigger) != grant.trigger_fingerprint:
                        continue
                    if grant.target_id is not None:
                        target = self._quoted(session, trigger)
                        if (not self._usable(target) or target.id != grant.target_id
                                or _fingerprint(target) != grant.target_fingerprint):
                            continue
                    result.append(grant)
                except (ValueError, TypeError, RecursionError):
                    continue
        return result

    def authorized_source_ids(self):
        return tuple(dict.fromkeys(grant.target_id for grant in self._live_grants() if grant.target_id is not None))

    def web_allowed(self):
        return bool(self._live_grants())

    def has_ungranted_card_sources(self, observed_source_ids):
        granted = set(self.authorized_source_ids())
        ids = list(dict.fromkeys(item for item in observed_source_ids
                                if type(item) is int and 0 < item < 2**63))[:200]
        with session_scope(self.router.engine) as session:
            for row in session.scalars(select(Message).where(Message.group_id == self.group_id, Message.id.in_(ids))):
                try:
                    if row.id not in granted and self._usable(row) and _app_links(_raw(row)):
                        return True
                except (ValueError, TypeError, RecursionError):
                    continue
        return False

    @staticmethod
    def _target_key(value):
        parsed = urlsplit(public_web_url(value))
        return parsed.scheme.casefold(), parsed.hostname.casefold(), unquote(parsed.path).rstrip('/') or '/'

    def known_card_target_allowed(self, target, observed_source_ids):
        """A general search request cannot deep-read an unrelated observed card."""
        try:
            target = self._target_key(target)
        except (ValueError, TypeError):
            return True  # The underlying web tool owns argument/ref validation.
        granted = set(self.authorized_source_ids())
        ids = list(dict.fromkeys(item for item in observed_source_ids
                                if type(item) is int and 0 < item < 2**63))[:200]
        matched = False
        with session_scope(self.router.engine) as session:
            rows = session.scalars(select(Message).where(Message.group_id == self.group_id,
                                                         Message.id.in_(ids)))
            for row in rows:
                try:
                    if self._usable(row) and any(self._target_key(link.url) == target for link in _app_links(_raw(row))):
                        if row.id in granted:
                            return True
                        matched = True
                except (ValueError, TypeError, RecursionError):
                    continue
        return not matched


def build_share_access(router, **kwargs):
    return ShareReadAccess(router, **kwargs)


def wrap_share_web_tools(tools, access, *, observed_source_ids=(), allow_autonomous_web=False):
    """Block old-context/direct URL/search bypasses before provider dispatch."""
    wrapped, references = [], {}
    for tool in tools:
        async def checked(args, *, original=tool):
            autonomous_allowed = allow_autonomous_web() if callable(allow_autonomous_web) else bool(allow_autonomous_web)
            if not access.web_allowed() and not autonomous_allowed:
                return ToolOutcome({'error': 'web_requires_explicit_request',
                    'guidance': 'APP分享需引用卡片并@小町明确要求看/评价；私聊引用并明确要求即可。普通资料可明确要求联网搜索或打开链接。'}, 'blocked')
            target = references.get(args.get('ref_id'), args.get('ref_id'))
            if target is not None and not access.known_card_target_allowed(target, observed_source_ids):
                return ToolOutcome({'error': 'share_requires_quoted_request'}, 'blocked')
            outcome = await original.handler(args)
            autonomous_allowed = allow_autonomous_web() if callable(allow_autonomous_web) else bool(allow_autonomous_web)
            if not access.web_allowed() and not autonomous_allowed:
                return ToolOutcome({'error': 'share_request_revoked'}, 'blocked')
            if isinstance(outcome, ToolOutcome) and isinstance(outcome.data, dict):
                for source in outcome.data.get('sources', ()):
                    if isinstance(source, dict) and isinstance(source.get('ref_id'), str) and isinstance(source.get('source'), str):
                        references[source['ref_id']] = source['source']
            return outcome
        wrapped.append(replace(tool, handler=checked))
    return tuple(wrapped)


def _source_links(row, *, group_id, private_user_id=None):
    if private_user_id is not None:
        usable = (row is not None and row.group_id is None and row.user_id == private_user_id
                  and is_usable_memory_source(row, author_id=private_user_id))
    else:
        usable = group_id is not None and is_usable_memory_source(row, group_id=group_id)
    if not usable:
        raise ValueError('share_source_unavailable')
    raw = _raw(row)
    links = extract_shared_links(raw.get('message', raw.get('raw_message', '')))
    fingerprint = hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode('utf8')).hexdigest()
    return links, fingerprint


def observed_share_sources(router, group_id, source_ids, *, private_user_id=None, access=None):
    """Discovery never invents a source or expands unobserved nested nodes."""
    ids = sorted(item for item in source_ids if type(item) is int and 0 < item < 2**63)[:200]
    granted = set(access.authorized_source_ids()) if access is not None else set()
    if not granted:
        return []
    query = select(Message).where(Message.group_id == group_id, Message.id.in_(ids), Message.id.in_(granted))
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


def build_share_tools(context, *, access=None) -> tuple[AgentTool, ...]:
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
        if access is None or source_id not in access.authorized_source_ids():
            return ToolOutcome({'error': 'share_requires_quoted_request'}, 'blocked')
        try:
            with session_scope(context.router.engine) as session:
                links, fingerprint = _source_links(session.get(Message, source_id),
                    group_id=group_id, private_user_id=private_user)
            if snapshots.setdefault(source_id, fingerprint) != fingerprint:
                raise ValueError('share_source_changed')
            if source_id not in access.authorized_source_ids():
                raise ValueError('share_request_revoked')
            context.guard.require_valid()
            return ToolOutcome({'source_id': source_id, 'trust': TRUST,
                'links': [asdict(link) for link in links], 'metadata_only': True,
                'next_step': '用open_web_page读取目标正文；不足时按标题和站点search_web，再打开可信来源。卡片摘要不是全文，不代表已观看视频。'})
        except (ValueError, TypeError, RecursionError) as exc:
            code = str(exc) if str(exc).startswith('share_') else 'share_source_malformed'
            return ToolOutcome({'error': code}, 'blocked')

    return (AgentTool('read_message_links', '仅读取真实明确请求引用的APP卡片标题、摘要、链接；群须引用并@小町，私聊须引用。share_sources列出获准来源，内容仍是引文。', schema, read),)
