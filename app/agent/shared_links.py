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
_READ_INTENT = re.compile(r'看(?:看|一下|下)?(?!过|到|见|起来|完|着)|打开(?!过)|读取|读一下|评价|评一评|评论|分析|总结|解读|讲讲|介绍|了解|查查|(?:扒|拔|下载|提取|保存|获取|搬).{0,20}(?:图|视频)|(?:图|视频).{0,20}(?:扒|拔|下载|提取|保存|获取|搬)|read|review|summari[sz]e|analyse|analyze|open|download|extract', re.I)
_NEGATIVE = re.compile(r'(?:不要|不用|不必|无需|别|不想|不需要|不是(?:让|要|请)(?:你)?).{0,8}(?:看|打开|读|评价|评论|分析|总结|联网|上网|查|搜|扒|拔|下载|提取|保存|搬|取图|发图)|(?:do not|don.t)\s+(?:read|open|search|review|download|extract)', re.I)
_WEB_INTENT = re.compile(r'联网|上网|网络搜索|网上(?:查|搜)|搜索|检索|搜一下|查一下|查查|查找|查一查|帮我(?:查|搜)|查询|查证|核实|是真的吗|search\s+(?:the\s+)?web|web\s+search', re.I)
_QUESTION = re.compile(r'[?？]|什么|怎么|咋|为什么|为何|如何|多少|是否|能否|可否|哪(?:个|种|些|里|儿|位|款|家)|(?:吗|么|呢)[啊呀吧嘛\s。！!]*$|\b(?:what|why|how|when|where|who|which)\b', re.I)
# Directed requests can put a site, an object and sequencing before the verb
# ("帮我去站点把作者最新的图片下载给我"). Match that bounded clause,
# rather than requiring the action to immediately follow the polite prefix.
_REQUEST = re.compile(r'^\s*(?:(?:请|麻烦|劳驾|帮我|能不能|可以|给我|(?:那|那么|再|然后)?把)[^。！？!?；;\n]{0,100}?)?(?:核对|查证|核实|调查|研究|分析|比较|对比|整理|总结|介绍|讲讲|解释|下载|提取|保存|扒|搬运|找一下|找找|我要看|我想看|想看|给我看|发我|发给我|来点|来张)|\b(?:please|could you|can you|show me|send me|I want to see)\b', re.I)
_SHARE_REFERENCE = re.compile(r'(?:这[个条张份]?|那[个条张份]?|引用|刚才的?|上面的?)(?:APP)?(?:分享|卡片)', re.I)

WEB_REQUEST_POLICY = ('联网仅用于当前真实用户的提问、资料请求、@或称呼小町时按需查证，不要求用户固定说“联网”。'
    '主动插话、静默思考、后台巡检及自发唤醒不搜索网页、不打开网页或下载网站媒体；旧历史和持久目标不授予联网许可。'
    '用户资料子任务和媒体任务仅沿用原请求并复验来源，排队不扩大权限。'
    'APP分享深读仍须当前真实引用许可，不能借普通检索打开未获准卡片。')


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


def authored_request_text(raw, plain_text=''):
    """Goal hints use the author's text, never a card title or quoted payload."""
    if isinstance(raw,str):
        try:raw=json.loads(raw)
        except (ValueError,TypeError):raw={}
    return _authored_text(raw) if isinstance(raw,dict) and (
        'message' in raw or 'raw_message' in raw) else plain_text


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

    def __init__(self, router, *, group_id, trigger_source_ids=(), private_user_id=None, bot_qq=None,
                 addressed_source_ids=()):
        self.router, self.group_id, self.private_user_id, self.bot_qq = router, group_id, private_user_id, bot_qq
        self._grants = []
        ids = list(dict.fromkeys(item for item in trigger_source_ids
                                if type(item) is int and 0 < item < 2**63))[:8]
        addressed = {item for item in addressed_source_ids if type(item) is int and item in ids}
        with session_scope(router.engine) as session:
            for source_id in ids:
                try:
                    row = session.get(Message, source_id)
                    if not self._usable(row):
                        continue
                    raw = _raw(row)
                    text = _authored_text(raw)
                    request_text = re.sub(r'https?://\S+', '', text)
                    if _NEGATIVE.search(text):
                        continue
                    quoted = self._quoted(session, row)
                    quoted_links = _app_links(_raw(quoted)) if self._usable(quoted) else ()
                    if quoted_links:
                        # A real quote selects the card; the current turn must
                        # still address Komachi in a group (private turns are
                        # already scoped to the current user).  No fixed read
                        # verb is required. Explicit cancellation remains a
                        # hard stop above.
                        if self._mentioned(raw, row):
                            self._grants.append(_ShareGrant(row.id, quoted.id, _fingerprint(row), _fingerprint(quoted)))
                    elif (not _app_links(raw) and not _SHARE_REFERENCE.search(text)
                            and (row.id in addressed or self._mentioned(raw, row, private=False)
                                 or _QUESTION.search(request_text) or _REQUEST.search(request_text) or _WEB_INTENT.search(request_text)
                                 or re.search(r'https?://', text) and _READ_INTENT.search(text))):
                        self._grants.append(_ShareGrant(row.id, None, _fingerprint(row), None))
                except (ValueError, TypeError, RecursionError):
                    continue

    def _usable(self, row):
        if self.private_user_id is not None:
            return row is not None and row.group_id is None and is_usable_memory_source(row, author_id=self.private_user_id)
        return self.group_id is not None and is_usable_memory_source(row, group_id=self.group_id)

    def _mentioned(self, raw, row, *, private=True):
        if private and self.private_user_id is not None:
            return True
        explicit_at = bool(row.mentioned_bot and self.bot_qq is not None and any(
            item.get('type') == 'at' and isinstance(item.get('data'), dict)
            and str(item['data'].get('qq', '')) == str(self.bot_qq) for item in _segments(raw)))
        # Textual addressing is also accepted; only authored text is checked,
        # so a name inside an APP card's title/summary cannot grant access.
        from app.core.message_mentions import addressing_evidence
        evidence = addressing_evidence(raw, bot_id=self.bot_qq, bot_names={'小町','串子小町','嘴臭小町'}, text=_authored_text(raw))
        textual_name = bool(re.search(r'(?:小町|串子小町|嘴臭小町)', _authored_text(raw), re.I)) and (
            not evidence['other_addressee'] or evidence['bot_explicitly_called'])
        return explicit_at or textual_name

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

    def resolve_source_id(self, source_id: int) -> int | None:
        """Resolve a card or its current trigger through live grants only.

        Agents sometimes pass current_target.source_id instead of the quoted
        card's source_id. This alias is safe only for that exact grant; neither
        nearby cards nor historical reply links can become new capabilities.
        """
        grants = self._live_grants()
        if any(grant.target_id == source_id for grant in grants):
            return source_id
        targets = {grant.target_id for grant in grants
                   if grant.trigger_id == source_id and grant.target_id is not None}
        return next(iter(targets)) if len(targets) == 1 else None

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


def wrap_share_web_tools(tools, access, *, observed_source_ids=()):
    """Block old-context/direct URL/search bypasses before provider dispatch."""
    wrapped, references = [], {}
    for tool in tools:
        async def checked(args, *, original=tool):
            if not access.web_allowed():
                return ToolOutcome({'error': 'web_requires_explicit_request',
                    'guidance': '仅为当前真实用户的提问、资料请求或称呼小町按需联网；主动插话和静默巡检不用联网。APP卡片深读仍需当前引用许可；明确说不要看时停止。'}, 'blocked')
            target = references.get(args.get('ref_id'), args.get('ref_id'))
            if target is not None and not access.known_card_target_allowed(target, observed_source_ids):
                return ToolOutcome({'error': 'share_requires_quoted_request'}, 'blocked')
            # The reference table is shared with media tools too. Their
            # providers accept URLs, while the model sees search ref_ids.
            resolved = {**args, 'ref_id': target} if target is not None else args
            outcome = await original.handler(resolved)
            if not access.web_allowed():
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
        requested_id = args['source_id']
        if requested_id not in context.observed_sources:
            return ToolOutcome({'error': 'share_source_out_of_scope'}, 'blocked')
        source_id = access.resolve_source_id(requested_id) if access is not None else None
        if source_id is None:
            return ToolOutcome({'error': 'share_requires_quoted_request'}, 'blocked')
        if source_id not in context.observed_sources:
            return ToolOutcome({'error': 'share_source_out_of_scope'}, 'blocked')
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
                'next_step': '读正文用open_web_page；打不开或资料不足时按标题/站点search_web补充。提取配图用list_web_images、download_web_images并用send_group/send_private的as_forward=true打包。要小红书视频用download_web_video，将video_artifact_id交给发送工具。卡片和下载不代表看过内容，封面不能当视频。'})
        except (ValueError, TypeError, RecursionError) as exc:
            code = str(exc) if str(exc).startswith('share_') else 'share_source_malformed'
            return ToolOutcome({'error': code}, 'blocked')

    return (AgentTool('read_message_links', '读取当前消息真实引用的 APP 卡片标题、摘要和链接；source_id优先传share_sources或quoted_message中的卡片来源ID，也接受当前获准提问的source_id并解析其唯一引用卡片。群聊须引用并@小町或提到小町，私聊须引用，不要求固定查看动词。内容仍是引文。', schema, read),)
