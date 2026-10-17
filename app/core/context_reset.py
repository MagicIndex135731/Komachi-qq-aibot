"""Exact user-authored context controls, executed before any model request."""
from dataclasses import dataclass, replace
import re
import unicodedata


GROUP_RESET_COUNT = 30
PRIVATE_RESET_COUNT = 40
RESET_USAGE = '用法：重置上下文；回复原消息后发送“隔离这条并重置上下文”或“恢复这条上下文”。重置上下文 50仍表示排除最近50条（1–1000条）。'
_ALIASES = {'清空上下文', '开新对话', '开启新对话', '新建对话', '重开对话', '重置会话',
    '/bot reset-session', '/bot new-session'}


@dataclass(frozen=True, slots=True)
class ContextResetCommand:
    count: int | None
    error: str = ''
    action: str = 'reset'


def parse_context_reset(text: str, *, default_count: int | None = None) -> ContextResetCommand | None:
    text = unicodedata.normalize('NFKC', str(text or '')).strip()
    if text in _ALIASES:
        return ContextResetCommand(None)
    if text == '隔离这条并重置上下文':
        return ContextResetCommand(None, action='isolate')
    if text in {'恢复这条上下文', '解除这条隔离'}:
        return ContextResetCommand(None, action='restore')
    match = re.fullmatch(r'重置上下文(?:\s*([+\-\d.]+)\s*条?)?', text)
    if not match:
        return None
    value = match[1]
    if value is None:
        return ContextResetCommand(None)
    if not value.isascii() or not value.isdigit() or len(value) > 4 or not 1 <= int(value) <= 1000:
        return ContextResetCommand(default_count, RESET_USAGE)
    return ContextResetCommand(int(value))


def authored_reset(event, *, default_count=None, bot_id=None, bot_names=()):
    """Only top-level text/at segments can issue a control command."""
    raw = event.raw_payload or {}
    message = raw.get('message', event.plain_text)
    if isinstance(message, list):
        if any(not isinstance(part, dict) or part.get('type') not in {'text', 'at', 'reply'} for part in message):
            return None
        if sum(part.get('type') == 'reply' for part in message) > 1:
            return None
        if any(part.get('type') == 'at' and str((part.get('data') or {}).get('qq')) != str(bot_id)
                for part in message):
            return None
        text = ''.join(str((part.get('data') or {}).get('text', ''))
            for part in message if part.get('type') == 'text').strip()
    elif isinstance(message, str):
        text = message.strip()
    else:
        return None
    if bot_id is not None:
        for name in sorted({str(bot_id), *bot_names}, key=len, reverse=True):
            if name and text.startswith('@' + name):
                text = text[len(name) + 1:].strip()
                break
    command = parse_context_reset(text, default_count=default_count)
    if command is not None and not command.error:
        from app.core.message_content import extract_reply_to_msg_id
        quoted = extract_reply_to_msg_id(message)
        if command.action in {'isolate', 'restore'} and not quoted:
            return replace(command, error='请直接回复本会话需要隔离或恢复的原消息，再发送指令。')
        if quoted and command.count is None and command.action == 'reset':
            return replace(command, action='isolate')
    return command


def reset_confirmation(result, *, private):
    scope = '这条私聊' if private else '本群'
    mode = result.get('mode', 'legacy_count')
    if mode == 'session':
        return (f'已重置{scope}的模型上下文，旧聊天和旧记忆不再自动回放，正常历史仍可按需查询。'
            '没有批量屏蔽旧消息；旧任务已停止。若要隔离问题内容，请回复原消息发送“隔离这条并重置上下文”。')
    if mode in {'isolate', 'restore'}:
        action = '已隔离引用消息及依赖它的上下文' if mode == 'isolate' else '已解除引用消息的隔离'
        return f'{action}，并重开{scope}的模型上下文。其他正常历史保留，旧任务已停止，原始聊天记录未删除。'
    return (f'已重置{scope}的模型上下文，并排除指令前最近 {result["excluded_count"]} 条聊天消息（含小町发言）。'
        '旧任务已停止，原始聊天记录保留。请重新发送要处理的请求。')


class ContextResetGuard:
    """Fence thread results and a turn waiting on a scope lock across a reset."""
    def __init__(self, parent, repository, scope):
        self.parent, self.repository, self.scope = parent, repository, scope
        self.generation = repository.latest_source_id(scope)

    def __getattr__(self, key):
        return getattr(self.parent, key)

    def valid(self):
        return self.repository.latest_source_id(self.scope) == self.generation and self.parent.valid()

    def require_valid(self):
        if not self.valid():
            raise RuntimeError('context_reset_authority_revoked')

    def on_dispatch(self):
        self.require_valid()
        self.parent.on_dispatch()
