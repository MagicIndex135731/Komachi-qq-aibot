from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import re
from typing import TypeVar


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
MAX_REASON_TEXT_LENGTH = 80
MAX_USER_LABEL_LENGTH = 40
MAX_SOURCE_MSG_ID_LENGTH = 128
MAX_DOCUMENT_ID_LENGTH = 64
MAX_RANK = 9999

MAX_SUMMARY_DOCUMENTS_PER_PAGE = 20
MAX_SUMMARY_PAGES = 16
MAX_SUMMARY_DOCUMENTS = MAX_SUMMARY_DOCUMENTS_PER_PAGE * MAX_SUMMARY_PAGES
MAX_SUMMARY_CONTENT_LENGTH = 600
MAX_OVERVIEW_LENGTH = 360
MAX_OUTLINE_DOCUMENT_IDS = 7
MAX_OUTLINE_RESPONSE_LENGTH = 4096

MAX_UNCOVERED_MESSAGES_PER_PAGE = 50
MAX_UNCOVERED_PAGES = 16
MAX_UNCOVERED_MESSAGES = (
    MAX_UNCOVERED_MESSAGES_PER_PAGE * MAX_UNCOVERED_PAGES
)
MAX_UNCOVERED_SELECTED_PER_PAGE = 3
MAX_UNCOVERED_RESPONSE_LENGTH = 1024
MAX_UNCOVERED_RESPONSE_LINES = 10

MAX_DOCUMENT_MESSAGE_GROUPS = 16
MAX_DOCUMENT_MESSAGES_PER_GROUP = 800
MAX_SELECTION_RESPONSE_LENGTH = 4096
MAX_SELECTION_RESPONSE_LINES = 20

_T = TypeVar("_T")


@dataclass(slots=True)
class WeeklyReportResult:
    ok: bool
    reply_text: str
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class WeeklyOutlineResult:
    ok: bool
    overview: str = ""
    selected_document_ids: tuple[str, ...] = ()
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class _EpisodeSummary:
    document_id: str
    content: str
    source_msg_ids: tuple[str, ...]
    start_at: datetime
    end_at: datetime


@dataclass(frozen=True, slots=True)
class _Outline:
    overview: str
    document_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SelectedItem:
    rank: int
    source_msg_id: str
    reason: str


def mask_profane_text(text: str) -> str:
    masked = str(text)
    for term in PROFANITY_TERMS:
        replacement = "*" if len(term) <= 1 else term[0] + ("*" * (len(term) - 1))
        masked = masked.replace(term, replacement)
    return masked


def build_group_weekly_outline(
    *,
    group_id: int,
    now: datetime,
    episode_summaries: Sequence[object],
    llm_client: object,
) -> WeeklyOutlineResult:
    """Summarize all supplied V2 documents and select whitelisted document IDs."""
    if not episode_summaries:
        return WeeklyOutlineResult(ok=False, error_code="no_summaries")
    if len(episode_summaries) > MAX_SUMMARY_DOCUMENTS:
        return WeeklyOutlineResult(
            ok=False,
            error_code="summary_limit_exceeded",
        )
    summaries = _normalize_episode_summaries(
        episode_summaries,
        require_source_ids=False,
    )
    if summaries is None:
        return WeeklyOutlineResult(
            ok=False,
            error_code="invalid_summaries",
        )
    outline = _generate_weekly_outline(
        group_id=group_id,
        now=now,
        summaries=summaries,
        llm_client=llm_client,
    )
    if outline is None:
        return WeeklyOutlineResult(
            ok=False,
            error_code="generation_failed",
        )
    return WeeklyOutlineResult(
        ok=True,
        overview=outline.overview,
        selected_document_ids=outline.document_ids,
    )


def build_group_weekly_report_from_evidence(
    *,
    group_id: int,
    now: datetime,
    document_message_groups: Sequence[Sequence[object]],
    uncovered_messages: Sequence[object],
    llm_client: object,
    overview: str = "",
    users_by_id: dict[int, object] | None = None,
) -> WeeklyReportResult:
    """Rank safely reloaded messages and render only data owned by each Message row."""
    del users_by_id  # Kept only for compatibility with the former public call shape.
    if len(document_message_groups) > MAX_DOCUMENT_MESSAGE_GROUPS:
        return WeeklyReportResult(
            ok=False,
            reply_text="",
            error_code="document_group_limit_exceeded",
        )
    if any(
        len(group) > MAX_DOCUMENT_MESSAGES_PER_GROUP
        for group in document_message_groups
    ):
        return WeeklyReportResult(
            ok=False,
            reply_text="",
            error_code="document_evidence_limit_exceeded",
        )
    if len(uncovered_messages) > MAX_UNCOVERED_MESSAGES:
        return WeeklyReportResult(
            ok=False,
            reply_text="",
            error_code="uncovered_limit_exceeded",
        )
    if (
        _count_unique_displayable_messages(
            document_message_groups=document_message_groups,
            uncovered_messages=uncovered_messages,
            stop_at=2,
        )
        < 2
    ):
        return WeeklyReportResult(
            ok=False,
            reply_text="",
            error_code="insufficient_data",
        )

    normalized_uncovered = _normalize_message_sequence(uncovered_messages)
    uncovered_candidates = _nominate_uncovered_messages(
        group_id=group_id,
        now=now,
        messages=normalized_uncovered,
        llm_client=llm_client,
    )
    if uncovered_candidates is None:
        return WeeklyReportResult(
            ok=False,
            reply_text="",
            error_code="generation_failed",
        )

    uncovered_ids = {
        source_id
        for message in uncovered_candidates
        if (source_id := _message_source_id(message)) is not None
    }
    document_budget = MAX_CANDIDATE_MESSAGES - len(uncovered_ids)
    document_candidates = _round_robin_document_messages(
        document_message_groups,
        limit=document_budget,
    )
    final_candidates = _deduplicate_messages(
        [*document_candidates, *uncovered_candidates],
        limit=MAX_CANDIDATE_MESSAGES,
    )
    if not final_candidates:
        return WeeklyReportResult(
            ok=False,
            reply_text="",
            error_code="generation_failed",
        )

    selected_items = _select_source_messages(
        group_id=group_id,
        now=now,
        messages=final_candidates,
        overview=overview,
        llm_client=llm_client,
    )
    if not selected_items:
        return WeeklyReportResult(
            ok=False,
            reply_text="",
            error_code="generation_failed",
        )
    message_by_id = {
        source_id: message
        for message in final_candidates
        if (source_id := _message_source_id(message)) is not None
    }
    rendered_items = [
        (
            item.rank,
            _message_sender_label(message_by_id[item.source_msg_id]),
            _normalize_candidate_text(
                str(getattr(message_by_id[item.source_msg_id], "plain_text", ""))
            ),
            item.reason,
        )
        for item in selected_items[:5]
    ]
    return WeeklyReportResult(
        ok=True,
        reply_text=_format_weekly_report_reply(
            now=now,
            overview=overview,
            items=rendered_items,
        ),
    )


def build_group_weekly_report(
    *,
    group_id: int,
    now: datetime,
    messages: list[object],
    users_by_id: dict[int, object],
    llm_client: object,
    episode_summaries: Sequence[object] | None = None,
) -> WeeklyReportResult:
    """Compatibility wrapper for callers that still provide one filtered raw list."""
    candidate_messages = _normalize_message_sequence(messages)
    if len(candidate_messages) < 2:
        return WeeklyReportResult(
            ok=False,
            reply_text="",
            error_code="insufficient_data",
        )

    def raw_fallback() -> WeeklyReportResult:
        return build_group_weekly_report_from_evidence(
            group_id=group_id,
            now=now,
            document_message_groups=[],
            uncovered_messages=candidate_messages,
            llm_client=llm_client,
            users_by_id=users_by_id,
        )

    summaries_input = episode_summaries or ()
    outline_result = build_group_weekly_outline(
        group_id=group_id,
        now=now,
        episode_summaries=summaries_input,
        llm_client=llm_client,
    )
    if outline_result.error_code == "summary_limit_exceeded":
        return WeeklyReportResult(
            ok=False,
            reply_text="",
            error_code="summary_limit_exceeded",
        )
    normalized_summaries = _normalize_episode_summaries(
        summaries_input,
        require_source_ids=True,
    )
    if not outline_result.ok or normalized_summaries is None:
        return raw_fallback()

    message_by_id = {
        source_id: message
        for message in candidate_messages
        if (source_id := _message_source_id(message)) is not None
    }
    selected_document_ids = set(outline_result.selected_document_ids)
    selected_summaries = [
        summary
        for summary in normalized_summaries
        if summary.document_id in selected_document_ids
    ]
    if any(
        source_id not in message_by_id
        for summary in selected_summaries
        for source_id in summary.source_msg_ids
    ):
        return raw_fallback()
    document_message_groups = [
        [
            message_by_id[source_id]
            for source_id in summary.source_msg_ids
        ]
        for summary in selected_summaries
    ]
    all_covered_source_ids = {
        source_id
        for summary in normalized_summaries
        for source_id in summary.source_msg_ids
    }
    uncovered_messages = [
        message
        for message in candidate_messages
        if _message_source_id(message) not in all_covered_source_ids
    ]
    result = build_group_weekly_report_from_evidence(
        group_id=group_id,
        now=now,
        document_message_groups=document_message_groups,
        uncovered_messages=uncovered_messages,
        llm_client=llm_client,
        overview=outline_result.overview,
        users_by_id=users_by_id,
    )
    if result.ok or result.error_code in {
        "uncovered_limit_exceeded",
        "document_group_limit_exceeded",
        "document_evidence_limit_exceeded",
    }:
        return result
    return raw_fallback()


def _generate_weekly_outline(
    *,
    group_id: int,
    now: datetime,
    summaries: list[_EpisodeSummary],
    llm_client: object,
) -> _Outline | None:
    pages = list(_chunked(summaries, MAX_SUMMARY_DOCUMENTS_PER_PAGE))
    page_outlines: list[_Outline] = []
    for page_number, page in enumerate(pages, start=1):
        try:
            raw_reply = llm_client.generate_text(
                _build_outline_prompt_lines(
                    now=now,
                    summaries=page,
                    page_number=page_number,
                    page_count=len(pages),
                ),
                conversation_key=(
                    f"group-weekly-report:{group_id}:outline:"
                    f"{page_number}-of-{len(pages)}"
                ),
            )
        except Exception:
            return None
        parsed = _parse_outline_json(
            raw_reply,
            allowed_document_ids={summary.document_id for summary in page},
        )
        if parsed is None:
            return None
        page_outlines.append(parsed)

    if len(page_outlines) == 1:
        return page_outlines[0] if page_outlines[0].document_ids else None
    allowed_ids = {
        document_id
        for outline in page_outlines
        for document_id in outline.document_ids
    }
    if not allowed_ids:
        return None
    try:
        raw_reply = llm_client.generate_text(
            _build_outline_merge_prompt_lines(
                now=now,
                page_outlines=page_outlines,
            ),
            conversation_key=f"group-weekly-report:{group_id}:outline:merge",
        )
    except Exception:
        return None
    merged = _parse_outline_json(
        raw_reply,
        allowed_document_ids=allowed_ids,
    )
    return merged if merged is not None and merged.document_ids else None


def _build_outline_prompt_lines(
    *,
    now: datetime,
    summaries: list[_EpisodeSummary],
    page_number: int,
    page_count: int,
) -> list[str]:
    document_lines = [
        (
            f"- document_id={summary.document_id} | "
            f"time={summary.start_at.strftime('%Y-%m-%d %H:%M')}"
            f"..{summary.end_at.strftime('%Y-%m-%d %H:%M')} | "
            f"summary={summary.content}"
        )
        for summary in summaries[:MAX_SUMMARY_DOCUMENTS_PER_PAGE]
    ]
    return [
        "System persona: 你在为 QQ 群周报概括一页 V2 episode summaries。",
        (
            "Output contract: 只输出严格 JSON 对象 "
            '{"overview":"不超过360字","document_ids":[当前页真实ID]}。'
            f"document_ids 最多{MAX_OUTLINE_DOCUMENT_IDS}个；"
            "不得输出 Markdown、原话、姓名或其他字段。"
        ),
        "Task: 概括本页重要话题，并仅提名当前页真实 document_id。",
        (
            f"Page: {page_number}/{page_count}; "
            f"report window end: {_normalize_timestamp(now).strftime('%Y-%m-%d %H:%M %Z')}"
        ),
        "Episode summaries:\n" + "\n".join(document_lines),
    ]


def _build_outline_merge_prompt_lines(
    *,
    now: datetime,
    page_outlines: list[_Outline],
) -> list[str]:
    page_lines = [
        (
            f"- page={index} | overview={outline.overview} | "
            f"document_ids={','.join(outline.document_ids)}"
        )
        for index, outline in enumerate(
            page_outlines[:MAX_SUMMARY_PAGES],
            start=1,
        )
    ]
    return [
        "System persona: 你在合并 QQ 群周报各页的有界提要。",
        (
            "Output contract: 只输出严格 JSON 对象 "
            '{"overview":"不超过360字","document_ids":[下方真实ID]}。'
            f"document_ids 最多{MAX_OUTLINE_DOCUMENT_IDS}个；"
            "不得输出 Markdown、原话、姓名或其他字段。"
        ),
        "Task: 合并全周概况，并仅从各页已提名 ID 中保留候选。",
        f"Report window end: {_normalize_timestamp(now).strftime('%Y-%m-%d %H:%M %Z')}",
        "Page synopses:\n" + "\n".join(page_lines),
    ]


def _parse_outline_json(
    text: object,
    *,
    allowed_document_ids: set[str],
) -> _Outline | None:
    response_text = str(text).strip()
    if not response_text or len(response_text) > MAX_OUTLINE_RESPONSE_LENGTH:
        return None
    try:
        payload = json.loads(response_text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"overview", "document_ids"}:
        return None
    overview = payload.get("overview")
    document_ids = payload.get("document_ids")
    if (
        not isinstance(overview, str)
        or not overview.strip()
        or len(overview.strip()) > MAX_OVERVIEW_LENGTH
        or not isinstance(document_ids, list)
    ):
        return None
    selected_ids: list[str] = []
    for raw_id in document_ids:
        if isinstance(raw_id, bool) or not isinstance(raw_id, (int, str)):
            continue
        document_id = str(raw_id).strip()
        if (
            not document_id
            or document_id not in allowed_document_ids
            or document_id in selected_ids
        ):
            continue
        selected_ids.append(document_id)
        if len(selected_ids) >= MAX_OUTLINE_DOCUMENT_IDS:
            break
    return _Outline(
        overview=_normalize_bounded_text(
            overview,
            limit=MAX_OVERVIEW_LENGTH,
            mask=False,
        ),
        document_ids=tuple(selected_ids),
    )


def _nominate_uncovered_messages(
    *,
    group_id: int,
    now: datetime,
    messages: list[object],
    llm_client: object,
) -> list[object] | None:
    if not messages:
        return []
    pages = list(_chunked(messages, MAX_UNCOVERED_MESSAGES_PER_PAGE))
    if len(pages) > MAX_UNCOVERED_PAGES:
        return None
    nominated_ids: list[str] = []
    for page_number, page in enumerate(pages, start=1):
        try:
            raw_reply = llm_client.generate_text(
                _build_uncovered_prompt_lines(
                    now=now,
                    messages=page,
                    page_number=page_number,
                    page_count=len(pages),
                ),
                conversation_key=(
                    f"group-weekly-report:{group_id}:uncovered:"
                    f"{page_number}-of-{len(pages)}"
                ),
            )
        except Exception:
            return None
        page_ids = _parse_uncovered_source_ids(
            raw_reply,
            allowed_source_msg_ids={
                source_id
                for message in page
                if (source_id := _message_source_id(message)) is not None
            },
        )
        nominated_ids.extend(
            source_id
            for source_id in page_ids
            if source_id not in nominated_ids
        )
    message_by_id = {
        source_id: message
        for message in messages
        if (source_id := _message_source_id(message)) is not None
    }
    return [
        message_by_id[source_id]
        for source_id in nominated_ids
        if source_id in message_by_id
    ]


def _build_uncovered_prompt_lines(
    *,
    now: datetime,
    messages: list[object],
    page_number: int,
    page_count: int,
) -> list[str]:
    return [
        "System persona: 你在为 QQ 群周报筛选一页未被 V2 文档覆盖的原始消息。",
        (
            "Output contract: 每行只能输出一个下方真实 source_msg_id，"
            f"最多{MAX_UNCOVERED_SELECTED_PER_PAGE}行；"
            "不得输出 rank、reason、姓名、原话或其他文本。"
        ),
        "Task: 提名本页最值得进入最终周报排行的高能消息 ID。",
        (
            f"Page: {page_number}/{page_count}; "
            f"report window end: {_normalize_timestamp(now).strftime('%Y-%m-%d %H:%M %Z')}"
        ),
        "Candidate messages:\n" + "\n".join(_candidate_message_lines(messages)),
    ]


def _parse_uncovered_source_ids(
    text: object,
    *,
    allowed_source_msg_ids: set[str],
) -> tuple[str, ...]:
    response_text = str(text)
    if len(response_text) > MAX_UNCOVERED_RESPONSE_LENGTH:
        return ()
    selected_ids: list[str] = []
    for raw_line in response_text.splitlines()[:MAX_UNCOVERED_RESPONSE_LINES]:
        source_id = raw_line.strip()
        if (
            not source_id
            or source_id not in allowed_source_msg_ids
            or source_id in selected_ids
        ):
            continue
        selected_ids.append(source_id)
        if len(selected_ids) >= MAX_UNCOVERED_SELECTED_PER_PAGE:
            break
    return tuple(selected_ids)


def _round_robin_document_messages(
    groups: Sequence[Sequence[object]],
    *,
    limit: int,
) -> list[object]:
    normalized_groups = [
        _normalize_message_sequence(group)
        for group in groups
        if group
    ]
    selected: list[object] = []
    seen_ids: set[str] = set()
    depth = 0
    while len(selected) < limit:
        found_at_depth = False
        for group in normalized_groups:
            if depth >= len(group):
                continue
            found_at_depth = True
            message = group[depth]
            source_id = _message_source_id(message)
            if source_id is None or source_id in seen_ids:
                continue
            seen_ids.add(source_id)
            selected.append(message)
            if len(selected) >= limit:
                break
        if not found_at_depth:
            break
        depth += 1
    return selected


def _select_source_messages(
    *,
    group_id: int,
    now: datetime,
    messages: list[object],
    overview: str,
    llm_client: object,
) -> list[_SelectedItem]:
    allowed_by_id = {
        source_id: message
        for message in messages[:MAX_CANDIDATE_MESSAGES]
        if (source_id := _message_source_id(message)) is not None
    }
    if not allowed_by_id:
        return []
    try:
        raw_reply = llm_client.generate_text(
            _build_selection_prompt_lines(
                now=now,
                messages=list(allowed_by_id.values()),
                overview=overview,
            ),
            conversation_key=f"group-weekly-report:{group_id}",
        )
    except Exception:
        return []
    return _parse_source_selection_lines(
        raw_reply,
        allowed_source_msg_ids=set(allowed_by_id),
    )


def _build_selection_prompt_lines(
    *,
    now: datetime,
    messages: list[object],
    overview: str,
) -> list[str]:
    return [
        "System persona: 你在帮 QQ 群生成近一周高能雷霆发言周报。",
        (
            "Safety rules: 只能从给定白名单选择 source_msg_id；"
            "不得输出或改写姓名、群名片、原话。"
        ),
        (
            "Output contract: 每行严格使用 rank|source_msg_id|reason，"
            "rank 为正整数，reason 简短非空，最多输出 5 行；不要输出其他文本。"
        ),
        "Task: 选出最有火药味、强烈情绪、节目效果或讨论价值的发言。",
        f"Report window end: {_normalize_timestamp(now).strftime('%Y-%m-%d %H:%M %Z')}",
        (
            "Weekly overview: "
            + (
                _normalize_bounded_text(
                    overview,
                    limit=MAX_OVERVIEW_LENGTH,
                    mask=False,
                )
                if overview
                else "(raw fallback)"
            )
        ),
        "Candidate messages:\n"
        + "\n".join(_candidate_message_lines(messages[:MAX_CANDIDATE_MESSAGES])),
    ]


def _candidate_message_lines(messages: Sequence[object]) -> list[str]:
    lines: list[str] = []
    for message in messages:
        source_id = _message_source_id(message)
        if source_id is None:
            continue
        timestamp = _normalize_timestamp(
            getattr(message, "timestamp")
        ).strftime("%Y-%m-%d %H:%M")
        text = _normalize_candidate_text(
            str(getattr(message, "plain_text", ""))
        )
        lines.append(
            f"- source_msg_id={source_id} | time={timestamp} | text={text}"
        )
    return lines


def _parse_source_selection_lines(
    text: object,
    *,
    allowed_source_msg_ids: set[str],
) -> list[_SelectedItem]:
    response_text = str(text)
    if len(response_text) > MAX_SELECTION_RESPONSE_LENGTH:
        return []
    items: list[_SelectedItem] = []
    seen_ranks: set[int] = set()
    seen_source_ids: set[str] = set()
    for raw_line in response_text.splitlines()[:MAX_SELECTION_RESPONSE_LINES]:
        parts = [part.strip() for part in raw_line.strip().split("|")]
        if len(parts) != 3:
            continue
        rank_text, source_msg_id, reason = parts
        if (
            not rank_text.isdigit()
            or len(rank_text) > len(str(MAX_RANK))
            or source_msg_id not in allowed_source_msg_ids
            or not reason
        ):
            continue
        rank = int(rank_text)
        if (
            rank <= 0
            or rank > MAX_RANK
            or rank in seen_ranks
            or source_msg_id in seen_source_ids
        ):
            continue
        seen_ranks.add(rank)
        seen_source_ids.add(source_msg_id)
        items.append(
            _SelectedItem(
                rank=rank,
                source_msg_id=source_msg_id,
                reason=_normalize_bounded_text(
                    reason,
                    limit=MAX_REASON_TEXT_LENGTH,
                    mask=True,
                ),
            )
        )
    items.sort(key=lambda item: item.rank)
    return items[:5]


def _format_weekly_report_reply(
    *,
    now: datetime,
    overview: str,
    items: list[tuple[int, str, str, str]],
) -> str:
    lines = [
        "本群近一周高能雷霆发言周报",
        f"统计截止：{_normalize_timestamp(now).strftime('%Y-%m-%d %H:%M')}",
    ]
    if overview:
        lines.append(
            "本周概况："
            + _normalize_bounded_text(
                overview,
                limit=MAX_OVERVIEW_LENGTH,
                mask=True,
            )
        )
    for rank, name, quote, reason in items:
        lines.append(f"第{rank}名 {name}")
        lines.append(f"原话：{quote}")
        lines.append(f"上榜理由：{reason}")
    return "\n".join(lines).strip()


def _normalize_episode_summaries(
    summaries: Iterable[object],
    *,
    require_source_ids: bool,
) -> list[_EpisodeSummary] | None:
    normalized: list[_EpisodeSummary] = []
    seen_document_ids: set[str] = set()
    for summary in summaries:
        document_id = _normalize_identifier(
            getattr(summary, "document_id", None),
            limit=MAX_DOCUMENT_ID_LENGTH,
        )
        content = getattr(summary, "content", None)
        raw_source_ids = getattr(summary, "source_msg_ids", ())
        start_at = getattr(summary, "start_at", None)
        end_at = getattr(summary, "end_at", None)
        if (
            document_id is None
            or document_id in seen_document_ids
            or not isinstance(content, str)
            or not content.strip()
            or isinstance(raw_source_ids, (str, bytes))
            or not isinstance(raw_source_ids, Iterable)
            or not isinstance(start_at, datetime)
            or not isinstance(end_at, datetime)
            or _normalize_timestamp(start_at) > _normalize_timestamp(end_at)
        ):
            return None
        source_ids: list[str] = []
        for raw_source_id in raw_source_ids:
            source_id = _normalize_identifier(
                raw_source_id,
                limit=MAX_SOURCE_MSG_ID_LENGTH,
            )
            if source_id is None:
                return None
            if source_id not in source_ids:
                source_ids.append(source_id)
        if require_source_ids and not source_ids:
            return None
        seen_document_ids.add(document_id)
        normalized.append(
            _EpisodeSummary(
                document_id=document_id,
                content=_normalize_bounded_text(
                    content,
                    limit=MAX_SUMMARY_CONTENT_LENGTH,
                    mask=False,
                ),
                source_msg_ids=tuple(source_ids),
                start_at=_normalize_timestamp(start_at),
                end_at=_normalize_timestamp(end_at),
            )
        )
    normalized.sort(key=lambda item: (item.start_at, item.end_at))
    return normalized


def _normalize_message_sequence(messages: Iterable[object]) -> list[object]:
    return _deduplicate_messages(
        [
            message
            for message in messages
            if str(getattr(message, "plain_text", "")).strip()
            and _message_source_id(message) is not None
            and isinstance(getattr(message, "timestamp", None), datetime)
        ],
        limit=None,
    )


def _deduplicate_messages(
    messages: Iterable[object],
    *,
    limit: int | None,
) -> list[object]:
    selected: list[object] = []
    seen_ids: set[str] = set()
    for message in messages:
        source_id = _message_source_id(message)
        if source_id is None or source_id in seen_ids:
            continue
        seen_ids.add(source_id)
        selected.append(message)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _count_unique_displayable_messages(
    *,
    document_message_groups: Sequence[Sequence[object]],
    uncovered_messages: Sequence[object],
    stop_at: int,
) -> int:
    seen_ids: set[str] = set()
    for messages in [*document_message_groups, uncovered_messages]:
        for message in messages:
            if not str(getattr(message, "plain_text", "")).strip():
                continue
            source_id = _message_source_id(message)
            if source_id is None:
                continue
            seen_ids.add(source_id)
            if len(seen_ids) >= stop_at:
                return len(seen_ids)
    return len(seen_ids)


def _message_sender_label(message: object) -> str:
    raw_json = getattr(message, "raw_json", None)
    sender: object = {}
    if isinstance(raw_json, dict):
        sender = raw_json.get("sender", {})
    if isinstance(sender, dict):
        card = str(sender.get("card") or "").strip()
        nickname = str(sender.get("nickname") or "").strip()
    else:
        card = str(getattr(sender, "card", "") or "").strip()
        nickname = str(getattr(sender, "nickname", "") or "").strip()
    user_id = str(getattr(message, "user_id", "")).strip() or "unknown"
    return _normalize_bounded_text(
        card or nickname or user_id,
        limit=MAX_USER_LABEL_LENGTH,
        mask=True,
    )


def _normalize_candidate_text(text: str) -> str:
    return _normalize_bounded_text(
        text,
        limit=MAX_MESSAGE_TEXT_LENGTH,
        mask=True,
    )


def _normalize_bounded_text(text: str, *, limit: int, mask: bool) -> str:
    value = mask_profane_text(text) if mask else str(text)
    normalized = re.sub(r"\s+", " ", value).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _message_source_id(message: object) -> str | None:
    return _normalize_identifier(
        getattr(message, "platform_msg_id", None),
        limit=MAX_SOURCE_MSG_ID_LENGTH,
    )


def _normalize_identifier(value: object, *, limit: int) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    identifier = str(value).strip()
    if (
        not identifier
        or len(identifier) > limit
        or "|" in identifier
        or any(character.isspace() for character in identifier)
    ):
        return None
    return identifier


def _chunked(items: Sequence[_T], size: int) -> Iterable[list[_T]]:
    for index in range(0, len(items), size):
        yield list(items[index : index + size])
