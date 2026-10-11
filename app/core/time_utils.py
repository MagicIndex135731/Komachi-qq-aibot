from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from zoneinfo import ZoneInfo


ASIA_SHANGHAI = ZoneInfo("Asia/Shanghai")

# Replayed conversation and quoted relative dates are evidence, not a clock.
CONVERSATION_TIME_POLICY = (
    "当前时间以本次模型请求的 Current time JSON 为准，时区固定 Asia/Shanghai（北京时间）。"
    "今天、昨天、明天按其中日期判断，不沿用历史回合的当前日期或UTC日期。"
    "历史消息、记忆和转发中的‘今天/昨天’是原作者在来源时间的说法；"
    "先按来源时间确定实际日期，再相对当前日期表述。来源日期不明时不要猜测或称为今天。"
)


def conversation_time_context(value: datetime) -> dict[str, str]:
    """Current conversation clock, independent of host TZ and source-event age."""
    local = shanghai_aware(value)
    today = local.date()
    return {"timezone": "Asia/Shanghai", "now": local.isoformat(timespec="seconds"),
            "today": today.isoformat(), "yesterday": (today - timedelta(days=1)).isoformat(),
            "tomorrow": (today + timedelta(days=1)).isoformat(),
            "weekday": ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")[local.weekday()]}


def conversation_time_prompt(value: datetime) -> str:
    return CONVERSATION_TIME_POLICY + "\nCurrent time JSON:\n" + json.dumps(conversation_time_context(value), ensure_ascii=False)


def shanghai_naive(value: datetime) -> datetime:
    """Convert any datetime to a naive Shanghai clock face for SQLite storage."""
    if value.tzinfo is None:
        return value
    return value.astimezone(ASIA_SHANGHAI).replace(tzinfo=None)


def shanghai_aware(value: datetime) -> datetime:
    """Interpret a stored naive datetime as Shanghai, or convert aware to Shanghai."""
    if value.tzinfo is None:
        return value.replace(tzinfo=ASIA_SHANGHAI)
    return value.astimezone(ASIA_SHANGHAI)


def stored_as_utc(value: datetime) -> datetime:
    """Interpret a stored naive datetime as Shanghai clock face and return aware UTC."""
    return shanghai_aware(value).astimezone(UTC)


def shanghai_now_naive() -> datetime:
    return datetime.now(ASIA_SHANGHAI).replace(tzinfo=None)


def shanghai_now_aware() -> datetime:
    return datetime.now(ASIA_SHANGHAI)
