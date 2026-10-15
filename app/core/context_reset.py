"""Exact user-authored context controls, executed before any model request."""
from dataclasses import dataclass
import re
import unicodedata


GROUP_RESET_COUNT = 30
PRIVATE_RESET_COUNT = 40
RESET_USAGE = '用法：重置上下文 或 重置上下文 50（1–1000条）。'
_ALIASES = {'清空上下文', '开新对话', '开启新对话', '新建对话', '重开对话', '重置会话',
    '/bot reset-session', '/bot new-session'}


@dataclass(frozen=True, slots=True)
class ContextResetCommand:
    count: int
    error: str = ''


def parse_context_reset(text: str, *, default_count: int) -> ContextResetCommand | None:
    text = unicodedata.normalize('NFKC', str(text or '')).strip()
    if text in _ALIASES:
        return ContextResetCommand(default_count)
    match = re.fullmatch(r'重置上下文(?:\s*([+\-\d.]+)\s*条?)?', text)
    if not match:
        return None
    value = match[1]
    if value is None:
        return ContextResetCommand(default_count)
    if not value.isascii() or not value.isdigit() or len(value) > 4 or not 1 <= int(value) <= 1000:
        return ContextResetCommand(default_count, RESET_USAGE)
    return ContextResetCommand(int(value))


def authored_reset(event, *, default_count, bot_id=None, bot_names=()):
    """Only top-level text/at segments can issue a control command."""
    raw = event.raw_payload or {}
    message = raw.get('message', event.plain_text)
    if isinstance(message, list):
        if any(not isinstance(part, dict) or part.get('type') not in {'text', 'at'} for part in message):
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
    return parse_context_reset(text, default_count=default_count)


def reset_confirmation(result, *, private):
    scope = '这条私聊' if private else '本群'
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
