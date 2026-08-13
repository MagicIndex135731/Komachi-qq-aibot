"""Build a large stratified memory test dataset from a read-only snapshot DB.

The dataset is a JSONL stream of ``evaluate_memory_recall.EvaluationCase``
fields plus platform extras (kind, expected_layer, gold_text, target_message_id,
now_iso, tags). It feeds both the offline full-pipeline stage and the
full-chain real-model stage of the memory test platform.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


KIND_TEMPLATES: dict[str, tuple[str, ...]] = {
    "preference": (
        "{alias}喜欢{obj}吗",
        "{alias}最喜欢{obj}吗",
        "{alias}喜欢什么",
        "{alias}最喜欢什么",
        "{alias}偏好什么",
    ),
    "taboo": (
        "{alias}讨厌{obj}吗",
        "{alias}不喜欢{obj}吗",
        "{alias}讨厌什么",
        "{alias}不喜欢什么",
        "{alias}反感什么",
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
        "{alias}说过{obj}吗",
        "{alias}的{obj}是什么",
        "{alias}说过什么关于{obj}",
        "{alias}怎么说的{obj}",
    ),
}

TEMPORAL_KINDS = frozenset({"plan", "current", "event", "decision"})
RECENCY_WINDOW_DAYS = 45


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
    "帮我算一下2的10次方",
    "现在几点钟了",
    "你好吗",
    "你叫什么名字",
    "推荐一个学习编程的网站",
    "今天有什么新闻",
    "怎么做红烧肉",
    "什么是质数",
    "最近有什么好看的番剧",
    "推荐一首周杰伦的歌",
    "怎么快速入睡",
    "有什么好用的笔记软件",
    "明天会下雨吗",
    "你怎么看人工智能",
)

DISTRACTOR_QUERIES = (
    "{alias}推荐一部动画",
    "{alias}喜欢什么游戏",
    "群里有谁在玩原神",
    "最近有什么好看的电影",
)

_CJK_TRIGRAM = re.compile(r"[\u4e00-\u9fff]{3}")
_ALIAS_STOP = {"小町", "比企谷小町", "机器人", "bot", "Bot"}
_KEYWORD_STOP = {
    "什么", "怎么", "一个", "我们", "你们", "他们", "这个", "那个", "没有",
    "不是", "就是", "知道", "可以", "现在", "今天", "昨天", "晚上", "时候",
    "真的", "感觉", "还是", "已经", "因为", "所以", "如果", "但是", "自己",
    "大家", "东西", "问题", "意思", "这样", "那样", "起来", "出来", "开始",
    "以后", "之前", "然后", "最后", "现在", "觉得", "喜欢",
}
_LEGACY_NOISE = re.compile(
    r"（QQ昵称|\(QQ昵称| likes | dislikes |\bdis\b|（昵称|\(昵称"
)
_SUMMARY_LEVEL_MARKERS = re.compile(
    r"^\s*(?:(?:recent\s*chat|daily|weekly|monthly|rolling|semantic|window|compact)\s*)?"
    r"summary\s*[:：\-]?\s*",
    re.IGNORECASE,
)


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


def _recent_item_pool(
    items: Sequence[dict[str, Any]],
    messages: Sequence[dict[str, Any]],
    *,
    days: int = RECENCY_WINDOW_DAYS,
) -> list[dict[str, Any]]:
    """Keep only memory items whose source messages are recent.

    Temporal queries ("最近有什么计划/现在在做什么") must be paired with
    recent facts; otherwise the gold reference is stale and the judged
    answer is marked as a mismatch even when it is a correct fresh answer.
    """
    if days <= 0 or not items or not messages:
        return list(items)
    ts_by_source: dict[str, datetime] = {}
    latest: datetime | None = None
    for row in messages:
        source_id = str(row.get("platform_msg_id") or "")
        raw_ts = row.get("timestamp")
        if not source_id or not raw_ts:
            continue
        try:
            parsed = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        ts_by_source[source_id] = parsed
        if latest is None or parsed > latest:
            latest = parsed
    if latest is None:
        return list(items)
    cutoff = latest - timedelta(days=int(days))
    recent: list[dict[str, Any]] = []
    for item in items:
        if any(
            ts_by_source.get(str(source_id)) is not None
            and ts_by_source[str(source_id)] >= cutoff
            for source_id in item.get("source_ids") or ()
        ):
            recent.append(item)
    return recent


def _sort_by_source_recency(
    items: Sequence[dict[str, Any]],
    messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort items by their latest source-message timestamp, newest first.

    Temporal queries must pair with the freshest fact for the subject;
    otherwise the gold reference can be stale while the model answers a
    newer supported fact and gets judged as a mismatch.
    """
    ts_by_source: dict[str, datetime] = {}
    for row in messages:
        source_id = str(row.get("platform_msg_id") or "")
        raw_ts = row.get("timestamp")
        if not source_id or not raw_ts:
            continue
        try:
            parsed = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        ts_by_source[source_id] = parsed

    def latest_ts(item: Mapping[str, Any]) -> datetime:
        timestamps = [
            ts_by_source[str(source_id)]
            for source_id in item.get("source_ids") or ()
            if str(source_id) in ts_by_source
        ]
        return max(timestamps) if timestamps else datetime.min.replace(tzinfo=UTC)

    return sorted(items, key=latest_ts, reverse=True)


def _dedupe_subject_newest(
    items: Sequence[dict[str, Any]],
    messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only the freshest memory item per subject.

    Temporal queries ("最近/现在") describe one current state per subject;
    keeping several facts for the same subject makes the gold ambiguous and
    produces false reference_mismatch when the model answers a newer fact.
    """
    newest_by_subject: dict[str, dict[str, Any]] = {}
    for item in _sort_by_source_recency(items, messages):
        subject = str(item.get("subject_id") or "")
        if subject not in newest_by_subject:
            newest_by_subject[subject] = item
    return list(newest_by_subject.values())


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
    trigrams = [match.group(0) for match in _CJK_TRIGRAM.finditer(text_value)]
    return [
        trigram
        for trigram in dict.fromkeys(trigrams)
        if trigram not in _KEYWORD_STOP and trigram not in _ALIAS_STOP
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
    if alias_candidates:
        alias = alias_candidates[0]
    elif subject_id.isdigit():
        alias = f"用户{subject_id}"
    else:
        # Group-scope memory items have no member alias; ask about the group.
        alias = "群里"
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
    object_text = _LEGACY_NOISE.sub("", item["object_text"][:12] or "").strip(" ，。、")
    if not object_text:
        object_text = _LEGACY_NOISE.sub("", item["content"][:12]).strip(" ，。、") or "事"
    query = template.format(
        alias=alias,
        obj=object_text,
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
        "allowed_subject_user_ids": (),
        "allowed_evidence_user_ids": None,
        "expected_answer_mode": "mention",
        "expected_coverage_strategy": "relevance",
        "minimum_time_bucket_count": 0,
        "forbidden_evidence_message_ids": (),
        "gate_tags": (
            "category=mention",
            "layer=raw",
            "real_mention=1",
            "subject_mode=none",
        ),
        "contract_fields_complete": True,
        "kind": "mention",
        "expected_layer": "raw",
        # A real mention is the user's own message, not a factual reference;
        # a natural grounded reply and a genuine abstention are both valid.
        "gold_text": "",
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
        "gold_text": "参考证据：" + row["plain_text"][:300],
        "target_message_id": str(row["id"]),
        "now_iso": row["timestamp"],
        "tags": ("category=raw_history", "layer=raw"),
    }


def _build_summary_case(
    summary: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    # Strip generated summary level headers (e.g. "Recent chat summary: ")
    # before extracting the topic so queries do not echo the header itself.
    content = _SUMMARY_LEVEL_MARKERS.sub("", summary["content"], count=1)
    topic = re.sub(r"[\s:：|,，。;；]+", "", content)[:8]
    topic_suffix = f"关于{topic}" if topic else ""
    query = f"昨天{('说了' if index % 2 else '聊了')}什么{topic_suffix}"
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


def _build_first_person_case(
    item: dict[str, Any],
    aliases: Mapping[int, Sequence[str]],
    index: int,
) -> dict[str, Any]:
    subject_id = item["subject_id"]
    template = FIRST_PERSON_TEMPLATES[index % len(FIRST_PERSON_TEMPLATES)]
    other_alias = ""
    for candidate_id, candidate_aliases in aliases.items():
        if str(candidate_id) != subject_id and candidate_aliases:
            other_alias = candidate_aliases[0]
            break
    query = template.format(other=other_alias or "别人")
    tags = [
        "kind=" + item["kind"],
        "layer=fact",
        "subject=requester",
        "intent=first_person",
    ]
    return {
        "group_id": item["group_id"],
        "query": query,
        "recent_context_message_ids": (),
        "expected_evidence_message_ids": tuple(item["source_ids"]),
        "category": "first_person",
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
        "gold_text": item["content"][:300] or item["object_text"][:300],
        "target_message_id": None,
        "now_iso": None,
        "tags": tuple(tags),
    }


def _build_ambiguous_case(
    group_id: int,
    aliases: Sequence[str],
    index: int,
) -> dict[str, Any]:
    a = aliases[0]
    b = aliases[1] if len(aliases) > 1 else "别人"
    template = AMBIGUOUS_TEMPLATES[index % len(AMBIGUOUS_TEMPLATES)]
    query = template.format(a=a, b=b)
    return {
        "group_id": group_id,
        "query": query,
        "recent_context_message_ids": (),
        "expected_evidence_message_ids": (),
        "category": "ambiguous",
        "time_range": None,
        "quoted_context_message_id": None,
        "schema_version": 1,
        "requester_uin": None,
        "allowed_subject_user_ids": (),
        "allowed_evidence_user_ids": None,
        "expected_answer_mode": "general_history",
        "expected_coverage_strategy": "relevance",
        "minimum_time_bucket_count": 0,
        "forbidden_evidence_message_ids": (),
        "gate_tags": ("category=ambiguous", "layer=none", "subject_mode=ambiguous"),
        "contract_fields_complete": True,
        "kind": "ambiguous",
        "expected_layer": "none",
        "gold_text": "",
        "target_message_id": None,
        "now_iso": None,
        "tags": ("category=ambiguous", "layer=none", "subject_mode=ambiguous"),
    }


def _build_cross_group_case(
    group_id: int,
    alias: str,
    index: int,
) -> dict[str, Any]:
    query = f"{alias}喜欢什么" if index % 2 else f"{alias}是什么样的人"
    return {
        "group_id": group_id,
        "query": query,
        "recent_context_message_ids": (),
        "expected_evidence_message_ids": (),
        "category": "cross_group",
        "time_range": None,
        "quoted_context_message_id": None,
        "schema_version": 1,
        "requester_uin": None,
        "allowed_subject_user_ids": (),
        "allowed_evidence_user_ids": None,
        "expected_answer_mode": "current_fact",
        "expected_coverage_strategy": "relevance",
        "minimum_time_bucket_count": 0,
        "forbidden_evidence_message_ids": (),
        "gate_tags": ("category=cross_group", "layer=none"),
        "contract_fields_complete": True,
        "kind": "cross_group",
        "expected_layer": "none",
        "gold_text": "",
        "target_message_id": None,
        "now_iso": None,
        "tags": ("category=cross_group", "layer=none"),
    }


def _build_distractor_case(
    item: dict[str, Any],
    aliases: Mapping[int, Sequence[str]],
    index: int,
) -> dict[str, Any]:
    subject_id = item["subject_id"]
    alias_candidates = aliases.get(int(subject_id), []) if subject_id.isdigit() else []
    alias = alias_candidates[0] if alias_candidates else f"用户{subject_id}"
    template = DISTRACTOR_QUERIES[index % len(DISTRACTOR_QUERIES)]
    query = template.format(alias=alias)
    return {
        "group_id": item["group_id"],
        "query": query,
        "recent_context_message_ids": (),
        "expected_evidence_message_ids": (),
        "category": "distractor",
        "time_range": None,
        "quoted_context_message_id": None,
        "schema_version": 1,
        "requester_uin": subject_id if subject_id.isdigit() else None,
        "allowed_subject_user_ids": None,
        "allowed_evidence_user_ids": None,
        "expected_answer_mode": "general",
        "expected_coverage_strategy": "relevance",
        "minimum_time_bucket_count": 0,
        "forbidden_evidence_message_ids": (),
        "gate_tags": ("category=distractor", "layer=none", "precision=1"),
        "contract_fields_complete": True,
        "kind": item["kind"],
        "expected_layer": "none",
        "gold_text": "",
        "target_message_id": None,
        "now_iso": None,
        "tags": ("category=distractor", "layer=none", "precision=1"),
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
    recent_items = _recent_item_pool(items, messages)
    recent_item_ids = {item["id"] for item in recent_items}
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
    target_fact = int(count * 0.40)
    target_mention = int(count * 0.15)
    target_raw = int(count * 0.20)
    target_first = int(count * 0.08)
    target_misc = int(count * 0.09)
    target_abstention = int(count * 0.08)
    # 1) Structured fact cases (round-robin over kinds to keep coverage).
    if items:
        by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_kind[item["kind"]].append(item)
        while len(cases) < target_fact:
            made = False
            for kind, kind_items in by_kind.items():
                if not kind_items:
                    continue
                pool = (
                    kind_items
                    if kind not in TEMPORAL_KINDS
                    else _dedupe_subject_newest(
                        [
                            item
                            for item in kind_items
                            if item["id"] in recent_item_ids
                        ],
                        messages,
                    )
                )
                item = pool[index % len(pool)]
                cases.append(_build_fact_case(item, aliases, rng, index))
                made = True
                index += 1
                if len(cases) >= target_fact:
                    break
            if not made:
                break
    # 1b) First-person variants over the same fact pool.
    if items:
        first_index = 0
        while len(cases) < target_fact + target_first:
            template = FIRST_PERSON_TEMPLATES[
                first_index % len(FIRST_PERSON_TEMPLATES)
            ]
            temporal = "最近" in template or "现在" in template
            pool = (
                _dedupe_subject_newest(recent_items, messages)
                if (temporal and recent_items)
                else items
            )
            item = pool[first_index % len(pool)]
            cases.append(_build_first_person_case(item, aliases, first_index))
            first_index += 1
            if first_index >= len(pool) * len(FIRST_PERSON_TEMPLATES):
                break
    # 2) Real mention cases.
    for offset, row in enumerate(mention_rows):
        if len(cases) >= target_fact + target_first + target_mention:
            break
        cases.append(_build_mention_case(row, {}, offset))
    # 3) Raw-history cases from topic keywords.
    raw_pool = raw_rows
    for offset in range(len(raw_pool)):
        if len(cases) >= target_fact + target_first + target_mention + target_raw:
            break
        row = raw_pool[offset]
        keywords = _topic_keywords(row["plain_text"])
        if not keywords:
            continue
        cases.append(_build_raw_case(row, keywords[offset % len(keywords)], offset))
    # 4) Summary/dated cases.
    for offset, summary in enumerate(summaries):
        if len(cases) >= target_fact + target_first + target_mention + target_raw + target_misc:
            break
        cases.append(_build_summary_case(summary, offset))
    # 4b) Ambiguous / cross-group / distractor families.
    misc_index = 0
    if aliases:
        membership = {
            (row["group_id"], row["user_id"]) for row in messages
        }
        for group_id in groups:
            group_aliases = list(
                dict.fromkeys(
                    alias
                    for user_id, candidates in aliases.items()
                    for alias in candidates
                    if (group_id, int(user_id)) in membership
                )
            )
            if len(group_aliases) >= 2:
                cases.append(_build_ambiguous_case(group_id, group_aliases, misc_index))
                misc_index += 1
    for group_id in groups:
        foreign_alias = None
        for user_id, candidates in aliases.items():
            if not any(
                row["group_id"] == group_id and row["user_id"] == user_id
                for row in messages
            ) and candidates:
                foreign_alias = candidates[0]
                break
        if foreign_alias is not None:
            cases.append(_build_cross_group_case(group_id, foreign_alias, misc_index))
            misc_index += 1
    if items:
        for distractor_index, item in enumerate(items):
            if distractor_index >= target_misc:
                break
            cases.append(_build_distractor_case(item, aliases, distractor_index))
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
    seen_queries: set[tuple[str, str, str, int]] = set()
    deduped: list[dict[str, Any]] = []
    for case in result:
        key = (
            str(case["category"]),
            str(case["query"]),
            str(case.get("requester_uin") or ""),
            int(case["group_id"]),
        )
        if key in seen_queries:
            continue
        seen_queries.add(key)
        deduped.append(case)
    result = deduped
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
