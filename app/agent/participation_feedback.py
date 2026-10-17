"""Cheap contextual feedback detection shared by ingress and dispatch gates."""
from __future__ import annotations

from collections.abc import Collection, Sequence
from datetime import UTC, datetime
import logging
import re
from typing import TYPE_CHECKING

from sqlalchemy import select

from app.core.memory_source_guard import is_usable_memory_source
from app.core.time_utils import shanghai_aware
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.participation_feedback import save_feedback
from app.storage.repositories import GroupRepository, MessageRepository

if TYPE_CHECKING:
    from app.adapters.onebot_models import GroupMessageEvent
    from app.core.router import InboundRouter

logger = logging.getLogger(__name__)

PROACTIVE_PARTICIPATION_RULES = (
    '消息数量和间隔只提供观察机会。依据当前人格与话题决定是否自然想回应，'
    '普通闲聊、具体感受和适度调侃都可以；不强求纠错或知识贡献。'
    '不复述刷屏，不机械追问，不给每件事硬编一句调侃，也不用证明自己活跃。'
    '没有合适接话点时安静结束；过去静默不否决后来出现的新话题。'
    '群友表达嫌烦、闭嘴或少插话时，结合最近谁在说话和引用对象理解，不能接梗反击；'
    '不要把对工作/别人嫌烦误当成针对自己。复读你的话不自动表示邀请继续。'
)

_NEGATIVE = re.compile(r'烦|吵|闭嘴|闭麦|安静|别.{0,5}(?:说|插|接|回)|少.{0,4}(?:说|发言|插)|'
                       r'不要.{0,4}(?:说|插|回)|停.{0,4}(?:说|回)|话.{0,4}(?:太多|好多)|'
                       r'(?:太|真|好).{0,2}(?:烦|吵)|刷屏|叭叭|聒噪')
_SOFT_STOP = re.compile(r'闭嘴|(?:安静|闭麦).{0,3}(?:点|些|模式)|(?:开|启用|切).{0,5}(?:安静|闭嘴|闭麦).{0,2}模式|'
                        r'(?:别|不要|少).{0,5}(?:说话|发言|插嘴|插话|接话|回复)|话.{0,4}(?:太多|好多)|刷屏|叭叭|聒噪')
_OTHER_SUBJECT = re.compile(r'(?:上班|工作|面试官|同事|老板|代码|游戏|这人|那人|学生会).{0,10}(?:烦|吵|话|闭嘴)')


def normalized(text: str) -> str:
    return re.sub(r'\s+', '', text or '').lower()


def repeats_bot(source: Message, recent: Sequence[Message], bot_id: int) -> bool:
    text = normalized(source.plain_text)
    return bool(text and any(normalized(row.plain_text) == text for row in recent
                            if row.user_id == bot_id and row.id < source.id))


def third_party_control(text: str, bot_names: Collection[str]) -> bool:
    match = re.search(r'让(.{1,12}?)闭嘴', normalized(text))
    if not match:
        return False
    target = match[1].strip('这个那个')
    return target not in {'他', '它', '你', '这机器人', '机器人', 'ai'} and not any(name in target for name in bot_names)


def detect_feedback(source: Message, recent: Sequence[Message], *, bot_id: int,
                    bot_names: Collection[str], messages: MessageRepository) -> str:
    text = normalized(source.plain_text)
    if not text or not _NEGATIVE.search(text) or source.msg_type in {'json', 'xml', 'forward'}:
        return ''
    if repeats_bot(source, recent, bot_id) or re.search(r'(?:不|没|并不|一点也不)(?:是|很|太|那么){0,2}(?:烦|吵|话多)|(?:别|不要|不用|不必).{0,2}(?:闭嘴|闭麦|安静)', text):
        return ''
    if third_party_control(text, bot_names):
        return ''
    # Asking the bot about an annoying colleague is still a normal request.
    # A mention or quote alone does not make third-party frustration its fault.
    if _OTHER_SUBJECT.search(text):
        return ''
    named = any(name in text for name in bot_names)
    quoted = messages.get_by_platform_msg_id(source.reply_to_msg_id) if source.reply_to_msg_id else None
    if source.mentioned_bot or named:
        return 'explicit_feedback'
    if quoted is not None:
        return 'reply_feedback' if quoted.user_id == bot_id and is_usable_memory_source(quoted, group_id=source.group_id) else ''
    prior = [row for row in recent if row.id < source.id and
             0 <= (shanghai_aware(source.timestamp)-shanghai_aware(row.timestamp)).total_seconds() <= 600]
    if not any(row.user_id == bot_id for row in prior[-10:]):
        return ''
    # Contextless emotional words like "烦死了" do not qualify. A stop-speaking
    # expression or robot reference plus recent actual bot speech is required.
    immediate_pronoun = prior[-1].user_id == bot_id and re.search(r'(?:你|它).{0,3}(?:烦|吵)', text)
    if _SOFT_STOP.search(text) or immediate_pronoun or re.search(r'(?:机器人|这ai|这个ai|这玩意).{0,6}(?:烦|吵)', text):
        return 'contextual_feedback'
    return ''


def feedback_options(router: InboundRouter, group_id: int) -> dict[str, int]:
    def integer(key, default, maximum):
        return max(1, min(maximum, router._group_policy_int(group_id=group_id, key=key, default=default)))
    quiet = integer('proactive_feedback_quiet_seconds', 1200, 3600)
    return {'quiet_seconds': quiet, 'duration_seconds': max(quiet,
        integer('proactive_feedback_duration_seconds', 7200, 21600))}


def record_feedback(router: InboundRouter, event: GroupMessageEvent, *, now: datetime | None = None) -> bool:
    if not router._group_policy_bool(group_id=event.group_id, key='proactive_feedback_enabled', default=True):
        return False
    now = now or datetime.now(UTC)
    with session_scope(router.engine) as session:
        messages = MessageRepository(session)
        source = messages.get_by_platform_msg_id(event.platform_msg_id)
        if not is_usable_memory_source(source, group_id=event.group_id, author_id=event.user_id) or source.user_id == router.runtime.settings.bot_qq:
            return False
        enabled, speak, *_ = router._resolve_group_policy(groups=GroupRepository(session), group_id=event.group_id)
        if not enabled or not speak:
            return False
        options = feedback_options(router, event.group_id)
        age = (now-shanghai_aware(source.timestamp)).total_seconds()
        if age < -120 or age >= options['duration_seconds']:
            return False
        recent = list(session.scalars(select(Message).where(Message.group_id == source.group_id,
            Message.id < source.id).order_by(Message.id.desc()).limit(24)))[::-1]
        recent = [row for row in recent if is_usable_memory_source(row, group_id=source.group_id)]
        reason = detect_feedback(source, recent, bot_id=router.runtime.settings.bot_qq,
            bot_names=router._build_bot_names(''), messages=messages)
        if not reason:
            return False
        changed = save_feedback(session, source=source, reason=reason, **options)
    if changed:
        logger.info('group_participation_feedback group_id=%s source_id=%s reason=%s', event.group_id, source.id, reason)
    return changed


def meaningful_new_messages(recent: Sequence[Message], *, bot_id: int, since: datetime | None) -> int:
    bot_texts = {normalized(row.plain_text) for row in recent if row.user_id == bot_id}
    seen = set()
    for row in recent:
        text = normalized(row.plain_text)
        if row.user_id == bot_id or not text or text in bot_texts:
            continue
        if since is not None and shanghai_aware(row.timestamp) <= since:
            continue
        if re.fullmatch(r'哈+|嗯+|哦+|好+|[?？!！]+|ok|wx', text):
            continue
        seen.add(text)
    return len(seen)
