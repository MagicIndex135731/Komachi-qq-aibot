from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re


PROFANITY_TERMS = (
    "他妈",
    "傻逼",
    "煞笔",
    "妈的",
    "草泥马",
    "垃圾",
    "废物",
    "操",
    "草",
    "sb",
    "SB",
)
MAX_CANDIDATE_MESSAGES = 200
MAX_MESSAGE_TEXT_LENGTH = 80


@dataclass(slots=True)
class WeeklyReportResult:
    ok: bool
    reply_text: str
    error_code: str | None = None


def mask_profane_text(text: str) -> str:
    masked = str(text)
    for term in PROFANITY_TERMS:
        if len(term) <= 1:
            replacement = "*"
        else:
            replacement = term[0] + ("*" * (len(term) - 1))
        masked = masked.replace(term, replacement)
    return masked


def build_group_weekly_report(*, group_id: int, now: datetime, messages: list[object], users_by_id: dict[int, object], llm_client: object) -> WeeklyReportResult:
    candidate_messages = [message for message in messages if str(getattr(message, "plain_text", "")).strip()]
    if len(candidate_messages) < 2:
        return WeeklyReportResult(ok=False, reply_text="", error_code="insufficient_data")

    limited_candidates = candidate_messages[-MAX_CANDIDATE_MESSAGES:]
    prompt_lines = _build_weekly_report_prompt_lines(
        now=now,
        messages=limited_candidates,
        users_by_id=users_by_id,
    )
    try:
        raw_reply = llm_client.generate_text(
            prompt_lines,
            conversation_key=f"group-weekly-report:{group_id}",
        )
    except Exception:
        return WeeklyReportResult(ok=False, reply_text="", error_code="generation_failed")

    parsed_items = _parse_weekly_report_lines(raw_reply)
    if not parsed_items:
        return WeeklyReportResult(ok=False, reply_text="", error_code="generation_failed")
    return WeeklyReportResult(
        ok=True,
        reply_text=_format_weekly_report_reply(now=now, items=parsed_items[:5]),
        error_code=None,
    )


def _build_weekly_report_prompt_lines(*, now: datetime, messages: list[object], users_by_id: dict[int, object]) -> list[str]:
    candidate_lines = []
    for message in messages:
        label = _user_label(user_id=int(getattr(message, "user_id")), users_by_id=users_by_id)
        timestamp = _normalize_timestamp(getattr(message, "timestamp")).strftime("%Y-%m-%d %H:%M")
        masked_text = _normalize_candidate_text(str(getattr(message, "plain_text", "")))
        candidate_lines.append(
            f"- id={getattr(message, 'platform_msg_id')} | user={label} | time={timestamp} | text={masked_text}"
        )
    return [
        "System persona: 你在帮 QQ 群生成近一周高能雷霆发言周报。",
        "Safety rules: 只能根据给定候选消息评选，不要编造不存在的消息；输出严格使用每行 rank|name|quote|reason 的格式；最多输出 5 行。",
        "Task: 从候选消息里选出近一周最有高能雷霆感的发言。高能既包括火药味重、情绪强烈，也包括节目效果强、炸裂、容易引发群聊讨论。",
        f"Report window end: {_normalize_timestamp(now).strftime('%Y-%m-%d %H:%M %Z')}",
        "Candidate messages:\n" + "\n".join(candidate_lines),
    ]


def _parse_weekly_report_lines(text: str) -> list[tuple[int, str, str, str]]:
    items: list[tuple[int, str, str, str]] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|", maxsplit=3)]
        if len(parts) != 4:
            continue
        rank_text, name, quote, reason = parts
        if not rank_text.isdigit():
            continue
        items.append((int(rank_text), name, quote, reason))
    items.sort(key=lambda item: item[0])
    return items


def _format_weekly_report_reply(*, now: datetime, items: list[tuple[int, str, str, str]]) -> str:
    lines = [
        "本群近一周高能雷霆发言周报",
        f"统计截止：{_normalize_timestamp(now).strftime('%Y-%m-%d %H:%M')}",
    ]
    for rank, name, quote, reason in items:
        lines.append(f"第{rank}名 {name}")
        lines.append(f"原话：{quote}")
        lines.append(f"上榜理由：{reason}")
    return "\n".join(lines).strip()


def _normalize_candidate_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", mask_profane_text(text)).strip()
    if len(normalized) <= MAX_MESSAGE_TEXT_LENGTH:
        return normalized
    return normalized[: MAX_MESSAGE_TEXT_LENGTH - 1].rstrip() + "…"


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _user_label(*, user_id: int, users_by_id: dict[int, object]) -> str:
    user = users_by_id.get(user_id)
    if user is None:
        return str(user_id)
    group_card = str(getattr(user, "group_card", "")).strip()
    nickname = str(getattr(user, "nickname", "")).strip()
    if group_card:
        return group_card
    if nickname:
        return nickname
    return str(user_id)
