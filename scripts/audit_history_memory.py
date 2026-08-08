"""Audit real history messages against the current memory pipeline.

Loads real human messages from a read-only snapshot, runs the live semantic
understanding channel (rewrite) plus retrieval/packing for each message, and
cross-checks the outputs with lightweight labels to surface suspicious or
failing behaviors. Reports per-dimension stats and a ranked list of issues.

Usage:
  python scripts/audit_history_memory.py \
    --database /path/to/snapshot.db --group-id 10001 --bot-user-id 123 \
    --limit 200 --offset 0 --output report.jsonl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any, Sequence

from sqlalchemy import text

from app.config import AppSettings
from app.core.legacy_memory_context import GroupMemoryContextRequest
from app.core.memory_context_packer import (
    MEMORY_GROUNDING_WITH_EVIDENCE,
    EvidenceMessage,
)
from app.core.search_policy import (
    is_explicit_search_request,
    is_general_search_decision_candidate,
    is_time_sensitive_request,
)
from app.main import build_llm_client, build_memory_runtime
from app.storage.db import build_engine


QUESTION_WORDS = (
    "什么",
    "啥",
    "吗",
    "呢",
    "哪",
    "怎么",
    "如何",
    "多少",
    "几",
    "谁",
    "怎么样",
    "咋样",
    "是不是",
    "有没有",
)
HISTORY_WORDS = (
    "以前",
    "曾经",
    "过去",
    "历史",
    "之前",
    "当时",
    "那时",
    "说过",
    "发过",
    "提过",
    "聊过",
    "发生",
    "哪条",
    "哪句",
    "什么时候",
    "哪一次",
    "原话",
)
RELATIVE_TIME_WORDS = ("今天", "昨天", "前天", "上周", "去年", "今年")
WEATHER_WORDS = ("天气", "温度", "气温", "台风", "降雨", "预警")
IMAGE_REQUEST_WORDS = ("画一张", "画个", "生成一张", "生成个", "帮我画", "画图", "图片生成")
HYPOTHETICAL_WORDS = ("如果", "假如", "假设", "要是")
GAME_PLAY_WORDS = ("猜动画", "猜游戏", "你问我", "我回答", "来玩", "游戏规则")


@dataclass(frozen=True, slots=True)
class AuditRow:
    message_id: str
    timestamp: str
    user_id: int
    text: str
    labels: dict[str, bool]
    rewrite_used: bool
    answer_mode: str
    subject_role: str
    subject_ids: tuple[str, ...]
    fact_kinds: tuple[str, ...]
    time_range: bool
    retrieval_query: str
    fact_count: int
    evidence_count: int
    grounding_evidence_backed: bool
    issues: tuple[str, ...]
    latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "text": self.text[:300],
            "labels": self.labels,
            "rewrite_used": self.rewrite_used,
            "answer_mode": self.answer_mode,
            "subject_role": self.subject_role,
            "subject_ids": list(self.subject_ids),
            "fact_kinds": list(self.fact_kinds),
            "time_range": self.time_range,
            "retrieval_query": self.retrieval_query,
            "fact_count": self.fact_count,
            "evidence_count": self.evidence_count,
            "grounding_evidence_backed": self.grounding_evidence_backed,
            "issues": list(self.issues),
            "latency_ms": round(self.latency_ms, 1),
        }


def _load_messages(engine, *, group_id: int, bot_user_id: int, limit: int, offset: int):
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT platform_msg_id, user_id, plain_text, timestamp, "
                "reply_to_msg_id, mentioned_bot FROM messages "
                "WHERE group_id = :g AND user_id != :bot "
                "AND plain_text != '' AND mentioned_bot = 1 "
                "ORDER BY id DESC LIMIT :limit OFFSET :offset"
            ),
            {"g": int(group_id), "bot": int(bot_user_id), "limit": int(limit), "offset": int(offset)},
        ).mappings().all()
    rows.reverse()
    return rows


def _member_names(engine, *, group_id: int, bot_user_id: int) -> set[str]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT DISTINCT u.nickname, u.group_card FROM users u "
                "JOIN messages m ON m.user_id = u.user_id "
                "WHERE m.group_id = :g AND m.user_id != :bot"
            ),
            {"g": int(group_id), "bot": int(bot_user_id)},
        ).all()
    names: set[str] = set()
    for row in rows:
        for value in (row[0], row[1]):
            if isinstance(value, str) and value.strip() and len(value.strip()) >= 2:
                names.add(value.strip())
    return names


def _labels(text: str, member_names: set[str]) -> dict[str, bool]:
    first_person_subject = bool(
        re.match(r"^\s*(?:最近|现在|目前)?我", text)
        or re.search(
            r"[，。！？\s](?:最近|现在|目前)?我"
            r"[^，。！？]{0,10}(?:什么|啥|吗|呢|哪|怎么|如何|多少|几|谁)",
            text,
        )
    )
    return {
        "question": any(word in text for word in QUESTION_WORDS),
        "first_person": first_person_subject,
        "member_name": any(name in text for name in member_names),
        "history_word": any(word in text for word in HISTORY_WORDS),
        "relative_time": any(word in text for word in RELATIVE_TIME_WORDS),
        "weather": any(word in text for word in WEATHER_WORDS),
        "long": len(text) > 500,
        "short": len(text) <= 4,
        "explicit_search": is_explicit_search_request(text),
        "time_sensitive": is_time_sensitive_request(text),
        "image_request": any(word in text for word in IMAGE_REQUEST_WORDS),
        "hypothetical": any(word in text for word in HYPOTHETICAL_WORDS),
        "game_play": any(word in text for word in GAME_PLAY_WORDS),
    }


def _check_issues(row: AuditRow) -> tuple[str, ...]:
    issues: list[str] = []
    labels = row.labels
    should_rewrite = len(row.text) >= 4 and (
        (labels["question"] or labels["member_name"])
        and not labels["explicit_search"]
        and not labels["image_request"]
    )
    if should_rewrite and not row.rewrite_used:
        issues.append("model_channel_unused")
    if (
        labels["weather"]
        and len(row.text) < 200
        and (row.time_range or row.answer_mode == "dated_history")
    ):
        issues.append("realtime_question_treated_as_history")
    low_risk = labels["hypothetical"] or labels["game_play"] or labels["image_request"]
    if row.rewrite_used and not low_risk:
        if labels["first_person"] and labels["question"]:
            bound_to_requester = bool(row.subject_ids) and row.subject_ids == (
                str(row.user_id),
            )
            if bound_to_requester:
                if row.subject_role not in ("requester", "member"):
                    issues.append("role_label_mismatch")
            elif row.subject_role not in ("requester", "member"):
                issues.append("first_person_not_bound")
        if (
            labels["member_name"]
            and labels["question"]
            and row.subject_role == "none"
            and not row.subject_ids
        ):
            issues.append("member_question_not_bound")
        if (
            labels["history_word"]
            and labels["question"]
            and row.subject_role == "none"
            and not row.time_range
            and row.answer_mode not in {"dated_history", "summary", "assessment"}
        ):
            issues.append("history_question_dropped")
        if not row.retrieval_query.strip():
            issues.append("empty_retrieval_query")
    if labels["long"] and labels["explicit_search"]:
        issues.append("long_text_triggers_forced_search")
    if labels["short"] and not labels["question"] and row.fact_count > 25:
        issues.append("short_message_packs_many_facts")
    if row.answer_mode == "general_history" and row.time_range and not labels["history_word"]:
        issues.append("time_range_without_history_intent")
    return tuple(dict.fromkeys(issues))


def _build_request(
    *,
    group_id: int,
    query: str,
    user_id: int,
    now: datetime,
    recent: Sequence[EvidenceMessage],
) -> GroupMemoryContextRequest:
    return GroupMemoryContextRequest(
        group_id=group_id,
        query=query,
        recent_messages=tuple(recent),
        quoted_message=None,
        target_message_id=f"audit-{query[:20]}",
        available_input=12000,
        now=now,
        current_user_id=user_id,
        use_full_history=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--group-id", required=True, type=int)
    parser.add_argument("--bot-user-id", required=True, type=int)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    settings = AppSettings().model_copy(
        update={"memory_retrieval_channel_timeout_seconds": 2.0}
    )
    engine = build_engine(args.database)
    llm = build_llm_client(settings=settings, engine=engine)
    runtime = build_memory_runtime(
        settings=settings,
        engine=engine,
        llm_client=llm,
        bot_display_name="小町",
    )
    messages = _load_messages(
        engine,
        group_id=args.group_id,
        bot_user_id=args.bot_user_id,
        limit=args.limit,
        offset=args.offset,
    )
    member_names = _member_names(
        engine,
        group_id=args.group_id,
        bot_user_id=args.bot_user_id,
    )

    rows: list[AuditRow] = []
    for message in messages:
        text = str(message["plain_text"] or "").strip()
        timestamp = message["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        labels = _labels(text, member_names)
        request = _build_request(
            group_id=args.group_id,
            query=text,
            user_id=int(message["user_id"]),
            now=timestamp,
            recent=(),
        )
        started = __import__("time").monotonic()
        trace = runtime.memory_orchestrator.v2_provider.evaluate(request)
        latency_ms = (__import__("time").monotonic() - started) * 1000
        resolved = trace.resolved_query
        packed = trace.result.packed_context
        row = AuditRow(
            message_id=str(message["platform_msg_id"]),
            timestamp=timestamp.isoformat(),
            user_id=int(message["user_id"]),
            text=text,
            labels=labels,
            rewrite_used=resolved.rewrite_used,
            answer_mode=str(resolved.answer_mode),
            subject_role=str(resolved.subject_role),
            subject_ids=tuple(resolved.subject_ids or ()),
            fact_kinds=tuple(resolved.preferred_fact_kinds),
            time_range=resolved.time_range is not None,
            retrieval_query=str(resolved.retrieval_query or ""),
            fact_count=len(packed.facts),
            evidence_count=len(packed.evidence_segments),
            grounding_evidence_backed=(
                packed.grounding_policy == MEMORY_GROUNDING_WITH_EVIDENCE
            ),
            issues=(),
            latency_ms=latency_ms,
        )
        row = AuditRow(
            message_id=row.message_id,
            timestamp=row.timestamp,
            user_id=row.user_id,
            text=row.text,
            labels=row.labels,
            rewrite_used=row.rewrite_used,
            answer_mode=row.answer_mode,
            subject_role=row.subject_role,
            subject_ids=row.subject_ids,
            fact_kinds=row.fact_kinds,
            time_range=row.time_range,
            retrieval_query=row.retrieval_query,
            fact_count=row.fact_count,
            evidence_count=row.evidence_count,
            grounding_evidence_backed=row.grounding_evidence_backed,
            issues=_check_issues(row),
            latency_ms=row.latency_ms,
        )
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_dict(), ensure_ascii=False) + "\n")

    total = len(rows)
    with_issues = sum(1 for row in rows if row.issues)
    print(
        json.dumps(
            {
                "total": total,
                "with_issues": with_issues,
                "rewrite_used": sum(1 for row in rows if row.rewrite_used),
                "issue_counts": {
                    issue: sum(1 for row in rows if issue in row.issues)
                    for issue in sorted(
                        {issue for row in rows for issue in row.issues}
                    )
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
