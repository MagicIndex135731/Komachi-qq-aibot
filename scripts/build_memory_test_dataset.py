"""Build a large stratified memory test dataset from a read-only snapshot DB.

The dataset is a JSONL stream of ``evaluate_memory_recall.EvaluationCase``
fields plus platform extras (kind, expected_layer, gold_text, target_message_id,
now_iso, tags). It feeds both the offline full-pipeline stage and the
full-chain real-model stage of the memory test platform.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


KIND_TEMPLATES: dict[str, tuple[str, ...]] = {
    "preference": (
        "{alias}喜欢什么",
        "{alias}最喜欢什么",
        "{alias}偏好什么",
        "{alias}爱看什么",
        "{alias}喜欢{obj}吗",
        "{alias}的口味是什么",
    ),
    "taboo": (
        "{alias}讨厌什么",
        "{alias}不喜欢什么",
        "{alias}反感什么",
        "{alias}有什么忌讳",
    ),
    "profile": (
        "{alias}是什么样的人",
        "{alias}的完整个人画像",
        "介绍一下{alias}",
        "{alias}是哪里人",
        "{alias}是做什么的",
    ),
    "plan": (
        "{alias}打算做什么",
        "{alias}的计划是什么",
        "{alias}最近有什么计划",
        "{alias}准备做什么",
    ),
    "decision": (
        "{alias}做了什么决定",
        "{alias}决定做什么",
        "{alias}最近决定了什么",
    ),
    "current": (
        "{alias}最近在做什么",
        "{alias}现在在忙什么",
        "{alias}最近在玩什么",
    ),
    "relationship": (
        "{alias}和谁是什么关系",
        "{alias}的关系",
        "{alias}和{other}是什么关系",
    ),
    "running_joke": (
        "{alias}有什么梗",
        "{alias}的梗是什么",
        "{alias}有什么名场面",
    ),
    "event": (
        "{alias}最近发生了什么",
        "{alias}最近聊过什么",
        "{alias}最近遇到了什么",
    ),
    "fact": (
        "{alias}说过什么关于{obj}",
        "{alias}的{obj}是什么",
        "{alias}怎么说的{obj}",
    ),
}

FIRST_PERSON_TEMPLATES = (
    "我喜欢什么",
    "我的完整个人画像",
    "介绍一下我",
    "我最近在做什么",
    "我的计划是什么",
    "我和{other}谁是你的主人",
    "我的称呼是什么",
)

AMBIGUOUS_TEMPLATES = (
    "{a}和{b}谁更厉害",
    "{a}和{b}是什么关系",
    "{a}、{b}谁是你的主人",
)

ABSTRACTION_QUERIES = (
    "晚上吃什么",
    "今天天气怎么样",
    "你真可爱",
    "推荐一首歌",
    "周末去哪玩",
    "讲个笑话",
)

DISTRACTOR_QUERIES = (
    "{alias}推荐一部动画",
    "{alias}喜欢什么游戏",
    "群里有谁在玩原神",
    "最近有什么好看的电影",
)

_CJK = re.compile(r"[\u4e00-\u9fff]{2,4}")
_ALIAS_STOP = {"小町", "比企谷小町", "机器人", "bot", "Bot"}


def _iter_rows(engine, statement: str, parameters: dict[str, Any] | None = None):
    with engine.connect() as connection:
        yield from connection.execute(text(statement), parameters or {})


def _parse_dt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        parsed = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
        return parsed.astimezone(UTC).isoformat()
    except ValueError:
        return None


def _load_aliases(engine) -> dict[int, list[str]]:
    """Member alias candidates per group from the users table when available."""
    columns = {
        str(row[1])
        for row in _iter_rows(engine, "PRAGMA table_info(users)")
    }
    aliases: dict[int, list[str]] = defaultdict(list)
    if not ({"user_id", "nickname"} <= columns):
        return aliases
    select_columns = ["user_id", "nickname"]
    if "group_card" in columns:
        select_columns.append("group_card")
    for row in _iter_rows(
        engine,
        f"SELECT user_id, nickname, group_card FROM users",
    ):
        user_id, nickname, group_card = row[0], row[1], row[2]
        candidates = [
            value
            for value in (nickname, group_card)
            if isinstance(value, str) and value.strip() and value.strip() not in _ALIAS_STOP
        ]
        if not candidates:
            candidates = [f"用户{user_id}"]
        for candidate in candidates:
            aliases[int(user_id)].append(candidate.strip())
    return aliases


def _load_memory_items(engine) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _iter_rows(
        engine,
        "SELECT id, scope_id, subject_id, memory_kind, predicate, object_text, "
        "content, source_msg_id, source_msg_ids, status "
        "FROM memory_items WHERE status = 'active' AND scope_type = 'group'",
    ):
        source_ids = list(row[8] or []) if isinstance(row[8], list) else []
        if not source_ids and row[7]:
            source_ids = [str(row[7])]
        if not source_ids:
            continue
        rows.append(
            {
                "id": int(row[0]),
                "group_id": int(row[1]),
                "subject_id": str(row[2]) if row[2] is not None else "",
                "kind": str(row[3]),
                "predicate": str(row[4] or ""),
                "object_text": str(row[5] or ""),
                "content": str(row[6] or ""),
                "source_ids": [str(value) for value in source_ids],
            }
        )
    return rows


def _load_summaries(engine) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _iter_rows(
        engine,
        "SELECT id, scope_id, summary_level, start_at, end_at, content, "
        "source_start_msg_id, source_end_msg_id, status FROM summaries "
        "WHERE status = 'active'",
    ):
        sources = [
            str(value)
            for value in (row[6], row[7])
            if value is not None and str(value).strip()
        ]
        if not sources:
            continue
        rows.append(
            {
                "id": int(row[0]),
                "group_id": int(row[1]),
                "level": str(row[2]),
                "start_at": _parse_dt(row[3]),
                "end_at": _parse_dt(row[4]),
                "content": str(row[5] or ""),
                "source_ids": sources,
            }
        )
    return rows


def _load_messages(engine) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _iter_rows(
        engine,
        "SELECT id, group_id, platform_msg_id, user_id, timestamp, plain_text, "
        "reply_to_msg_id, mentioned_bot FROM messages",
    ):
        if row[1] is None or row[2] is None:
            continue
        plain_text = str(row[5] or "")
        if not plain_text.strip():
            continue
        rows.append(
            {
                "id": int(row[0]),
                "group_id": int(row[1]),
                "platform_msg_id": str(row[2]),
                "user_id": int(row[3]) if row[3] is not None else 0,
                "timestamp": _parse_dt(row[4]),
                "plain_text": plain_text,
                "reply_to_msg_id": (
                    str(row[6]) if row[6] is not None else None
                ),
                "mentioned_bot": bool(row[7]),
            }
        )
    return rows


def _recent_window(
    messages: Sequence[dict[str, Any]],
    group_id: int,
    before_id: int,
    *,
    limit: int = 60,
) -> list[int]:
    return [
        row["id"]
        for row in sorted(
            (
                row
                for row in messages
                if row["group_id"] == group_id and row["id"] < before_id
            ),
            key=lambda row: row["id"],
            reverse=True,
        )[:limit]
    ]


def _topic_keywords(text_value: str) -> list[str]:
    return [
        match.group(0)
        for match in _CJK.finditer(text_value)
        if match.group(0) not in _ALIAS_STOP
    ]


def _build_fact_case(
    item: dict[str, Any],
    aliases: Mapping[int, Sequence[str]],
    rng: random.Random,
    index: int,
) -> dict[str, Any]:
    group_id = item["group_id"]
    subject_id = item["subject_id"]
    alias_candidates = aliases.get(int(subject_id), []) if subject_id.isdigit() else []
    alias = alias_candidates[0] if alias_candidates else f"用户{subject_id}"
    templates = KIND_TEMPLATES.get(item["kind"], KIND_TEMPLATES["fact"])
    template = templates[index % len(templates)]
    other_id = None
    other_alias = ""
    if "{other}" in template:
        for candidate_id, candidate_aliases in aliases.items():
            if str(candidate_id) != subject_id and candidate_aliases:
                other_id = candidate_id
                other_alias = candidate_aliases[0]
                break
    query = template.format(
        alias=alias,
        obj=item["object_text"][:12] or item["content"][:12] or "事",
        other=other_alias or "别人",
    )
    sources = item["source_ids"]
    gold = item["content"][:300] or item["object_text"][:300] or query
    tags = [
        "kind=" + item["kind"],
        "layer=fact",
        "subject=" + subject_id,
        "intent=" + item["kind"],
    ]
    if other_id is not None:
        tags.append("multi_subject")
    return {
        "group_id": group_id,
        "query": query,
        "recent_context_message_ids": (),
        "expected_evidence_message_ids": tuple(sources),
        "category": item["kind"],
        "time_range": None,
        "quoted_context_message_id": None,
        "schema_version": 1,
        "requester_uin": subject_id if subject_id.isdigit() else None,
        "allowed_subject_user_ids": (subject_id,) if subject_id.isdigit() else None,
        "allowed_evidence_user_ids": None,
        "expected_answer_mode": "current_fact",
        "expected_coverage_strategy": "relevance",
        "minimum_time_bucket_count": 0,
        "forbidden_evidence_message_ids": (),
        "gate_tags": tuple(tags),
        "contract_fields_complete": True,
        "kind": item["kind"],
        "expected_layer": "fact",
        "gold_text": gold,
        "target_message_id": None,
        "now_iso": None,
        "tags": tuple(tags),
    }


def _build_mention_case(
    row: dict[str, Any],
    messages_by_id: Mapping[str, dict[str, Any]],
    index: int,
) -> dict[str, Any]:
    sources = [row["platform_msg_id"]]
    if row["reply_to_msg_id"]:
        sources.append(str(row["reply_to_msg_id"]))
    return {
        "group_id": row["group_id"],
        "query": row["plain_text"][:200],
        "recent_context_message_ids": (),
        "expected_evidence_message_ids": tuple(dict.fromkeys(sources)),
        "category": "mention",
        "time_range": None,
        "quoted_context_message_id": row["reply_to_msg_id"],
        "schema_version": 1,
        "requester_uin": str(row["user_id"]),
        "allowed_subject_user_ids": (str(row["user_id"]),),
        "allowed_evidence_user_ids": None,
        "expected_answer_mode": "mention",
        "expected_coverage_strategy": "relevance",
        "minimum_time_bucket_count": 0,
        "forbidden_evidence_message_ids": (),
        "gate_tags": ("category=mention", "layer=raw", "real_mention=1"),
        "contract_fields_complete": True,
        "kind": "mention",
        "expected_layer": "raw",
        "gold_text": row["plain_text"][:300],
        "target_message_id": str(row["id"]),
        "now_iso": row["timestamp"],
        "tags": ("category=mention", "layer=raw", "real_mention=1"),
    }


def _build_raw_case(
    row: dict[str, Any],
    keyword: str,
    index: int,
) -> dict[str, Any]:
    query = f"说说{keyword}" if index % 2 else f"{keyword}是什么"
    return {
        "group_id": row["group_id"],
        "query": query,
        "recent_context_message_ids": (),
        "expected_evidence_message_ids": (row["platform_msg_id"],),
        "category": "raw_history",
        "time_range": None,
        "quoted_context_message_id": None,
        "schema_version": 1,
        "requester_uin": str(row["user_id"]),
        "allowed_subject_user_ids": None,
        "allowed_evidence_user_ids": None,
        "expected_answer_mode": "general_history",
        "expected_coverage_strategy": "relevance",
        "minimum_time_bucket_count": 0,
        "forbidden_evidence_message_ids": (),
        "gate_tags": ("category=raw_history", "layer=raw"),
        "contract_fields_complete": True,
        "kind": "raw_history",
        "expected_layer": "raw",
        "gold_text": row["plain_text"][:300],
        "target_message_id": str(row["id"]),
        "now_iso": row["timestamp"],
        "tags": ("category=raw_history", "layer=raw"),
    }


def _build_summary_case(
    summary: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    topic = re.sub(r"\s+", "", summary["content"])[:10] or "最近"
    query = (
        f"昨天{('说了' if index % 2 else '聊了')}什么关于{topic}"
    )
    return {
        "group_id": summary["group_id"],
        "query": query,
        "recent_context_message_ids": (),
        "expected_evidence_message_ids": tuple(summary["source_ids"]),
        "category": "summary",
        "time_range": (
            (summary["start_at"], summary["end_at"])
            if summary["start_at"] and summary["end_at"]
            else None
        ),
        "quoted_context_message_id": None,
        "schema_version": 1,
        "requester_uin": None,
        "allowed_subject_user_ids": None,
        "allowed_evidence_user_ids": None,
        "expected_answer_mode": "summary",
        "expected_coverage_strategy": "time_buckets",
        "minimum_time_bucket_count": 1,
        "forbidden_evidence_message_ids": (),
        "gate_tags": ("category=summary", "layer=summary"),
        "contract_fields_complete": True,
        "kind": "summary",
        "expected_layer": "summary",
        "gold_text": summary["content"][:300],
        "target_message_id": None,
        "now_iso": None,
        "tags": ("category=summary", "layer=summary"),
    }


def _build_abstention_case(
    group_id: int,
    query: str,
    index: int,
) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "query": query,
        "recent_context_message_ids": (),
        "expected_evidence_message_ids": (),
        "category": "abstention",
        "time_range": None,
        "quoted_context_message_id": None,
        "schema_version": 1,
        "requester_uin": None,
        "allowed_subject_user_ids": None,
        "allowed_evidence_user_ids": None,
        "expected_answer_mode": "general",
        "expected_coverage_strategy": "relevance",
        "minimum_time_bucket_count": 0,
        "forbidden_evidence_message_ids": (),
        "gate_tags": ("category=abstention", "layer=none"),
        "contract_fields_complete": True,
        "kind": "abstention",
        "expected_layer": "none",
        "gold_text": "",
        "target_message_id": None,
        "now_iso": None,
        "tags": ("category=abstention", "layer=none"),
    }


def _attach_recent(case: dict[str, Any], messages: Sequence[dict[str, Any]]) -> None:
    group_id = case["group_id"]
    before_id = (
        int(case["target_message_id"])
        if case["target_message_id"] is not None
        else max(
            (row["id"] for row in messages if row["group_id"] == group_id),
            default=0,
        )
    )
    if before_id <= 0:
        return
    window = _recent_window(messages, group_id, before_id)
    case["recent_context_message_ids"] = tuple(str(value) for value in window)


def build_cases(
    engine,
    *,
    count: int = 3000,
    seed: int = 20260811,
    group_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    messages = _load_messages(engine)
    if group_ids:
        allowed = set(int(value) for value in group_ids)
        messages = [row for row in messages if row["group_id"] in allowed]
    aliases = _load_aliases(engine)
    items = _load_memory_items(engine)
    summaries = _load_summaries(engine)
    groups = sorted({row["group_id"] for row in messages})
    if not groups:
        raise ValueError("snapshot has no messages; cannot build a dataset")
    mention_rows = [
        row for row in messages if row["mentioned_bot"] and row["plain_text"].strip()
    ]
    raw_rows = [
        row for row in messages if len(row["plain_text"]) >= 8 and not row["mentioned_bot"]
    ]
    cases: list[dict[str, Any]] = []
    index = 0
    # 1) Structured fact cases (round-robin over kinds to keep coverage).
    if items:
        by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_kind[item["kind"]].append(item)
        while len(cases) < int(count * 0.45):
            made = False
            for kind_items in by_kind.values():
                if not kind_items:
                    continue
                item = kind_items[index % len(kind_items)]
                cases.append(_build_fact_case(item, aliases, rng, index))
                made = True
                index += 1
                if len(cases) >= int(count * 0.45):
                    break
            if not made:
                break
    # 2) Real mention cases.
    for offset, row in enumerate(mention_rows):
        if len(cases) >= int(count * 0.55):
            break
        cases.append(_build_mention_case(row, {}, offset))
    # 3) Raw-history cases from topic keywords.
    raw_pool = raw_rows
    for offset in range(len(raw_pool)):
        if len(cases) >= int(count * 0.75):
            break
        row = raw_pool[offset]
        keywords = _topic_keywords(row["plain_text"])
        if not keywords:
            continue
        cases.append(_build_raw_case(row, keywords[offset % len(keywords)], offset))
    # 4) Summary/dated cases.
    for offset, summary in enumerate(summaries):
        if len(cases) >= int(count * 0.85):
            break
        cases.append(_build_summary_case(summary, offset))
    # 5) Abstention / precision cases to reach the target.
    while len(cases) < count:
        group_id = groups[rng.randrange(len(groups))]
        cases.append(
            _build_abstention_case(
                group_id,
                ABSTRACTION_QUERIES[rng.randrange(len(ABSTRACTION_QUERIES))],
                len(cases),
            )
        )
    for case in cases:
        _attach_recent(case, messages)
    result = cases[:count]
    for index, case in enumerate(result):
        case["case_id"] = f"{case['category']}-{index}"
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a stratified memory test dataset from a snapshot DB."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--group-ids", type=str, default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    group_ids = (
        [int(value) for value in args.group_ids.split(",") if value.strip()]
        if args.group_ids
        else None
    )
    engine = create_engine(
        f"sqlite:///{args.database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )
    cases = build_cases(engine, count=args.count, seed=args.seed, group_ids=group_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    kinds = defaultdict(int)
    for case in cases:
        kinds[str(case["kind"])] += 1
    print(json.dumps({"cases": len(cases), "by_kind": dict(kinds)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
