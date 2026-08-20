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

_CJK_SPAN = re.compile(r"[\u4e00-\u9fff]{3,}")
_ALIAS_STOP = {"小町", "比企谷小町", "机器人", "bot", "Bot"}
_KEYWORD_STOP = {
    "什么", "怎么", "一个", "我们", "你们", "他们", "这个", "那个", "没有",
    "不是", "就是", "知道", "可以", "现在", "今天", "昨天", "晚上", "时候",
    "真的", "感觉", "还是", "已经", "因为", "所以", "如果", "但是", "自己",
    "大家", "东西", "问题", "意思", "这样", "那样", "起来", "出来", "开始",
    "以后", "之前", "然后", "最后", "现在", "觉得", "喜欢", "印象",
}
_LEGACY_NOISE = re.compile(
    r"（QQ昵称|\(QQ昵称| likes | dislikes |\bdis\b|（昵称|\(昵称"
)
_TOPIC_GENERIC_CHARS = frozenset("我你他她它这那谁啥哪的了呢吗吧呀啊哦哈嘛")
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


def _load_user_aliases(engine) -> dict[int, list[str]]:
    """Global fallback aliases from the users table."""
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
    for row in _iter_rows(engine, f"SELECT {', '.join(select_columns)} FROM users"):
        user_id, nickname = row[0], row[1]
        group_card = row[2] if len(row) > 2 else None
        raw_candidates = [
            value
            for value in (nickname, group_card)
            if isinstance(value, str) and value.strip()
        ]
        candidates = [
            value for value in raw_candidates if value.strip() not in _ALIAS_STOP
        ]
        if not raw_candidates:
            candidates = [f"用户{user_id}"]
        for candidate in candidates:
            aliases[int(user_id)].append(candidate.strip())
    return aliases


def _raw_payload(raw_json: object) -> Mapping[str, Any]:
    if isinstance(raw_json, str):
        try:
            raw_json = json.loads(raw_json)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw_json, Mapping):
        return {}
    return raw_json


def _sender_aliases(raw_json: object) -> tuple[str, ...]:
    raw_json = _raw_payload(raw_json)
    sender = raw_json.get("sender")
    if not isinstance(sender, Mapping):
        return ()
    return tuple(
        dict.fromkeys(
            value.strip()
            for value in (sender.get("card"), sender.get("nickname"))
            if isinstance(value, str)
            and value.strip()
            and value.strip() not in _ALIAS_STOP
        )
    )


def _is_bot_message(row: Mapping[str, Any]) -> bool:
    raw_json = _raw_payload(row.get("raw_json"))
    delivery_state = str(raw_json.get("delivery_state") or "").strip()
    sender = raw_json.get("sender")
    sender_names = (
        tuple(
            str(sender.get(key) or "").strip()
            for key in ("nickname", "card")
        )
        if isinstance(sender, Mapping)
        else ()
    )
    return bool(
        delivery_state
        or str(row.get("platform_msg_id") or "").startswith("bot-reply-")
        or any(name in _ALIAS_STOP for name in sender_names if name)
    )


def _load_group_aliases(
    engine,
    messages: Sequence[Mapping[str, Any]],
) -> dict[int, dict[int, list[str]]]:
    """Resolve aliases from real per-group sender snapshots, newest first.

    The users table is global and its group_card can be overwritten by traffic
    from another group.  It is therefore only a compatibility fallback for
    observed memberships whose message snapshot has no sender metadata.
    """

    aliases: dict[int, dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in sorted(messages, key=lambda value: int(value["id"]), reverse=True):
        group_id = int(row["group_id"])
        user_id = int(row["user_id"])
        for alias in _sender_aliases(row.get("raw_json")):
            if alias not in aliases[group_id][user_id]:
                aliases[group_id][user_id].append(alias)
    fallback = _load_user_aliases(engine)
    for group_id, user_id in {
        (int(row["group_id"]), int(row["user_id"])) for row in messages
    }:
        for alias in fallback.get(user_id, ()):
            if alias not in aliases[group_id][user_id]:
                aliases[group_id][user_id].append(alias)
    return {
        group_id: dict(members) for group_id, members in aliases.items()
    }


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


def _load_messages(engine) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    columns = {str(row[1]) for row in _iter_rows(engine, "PRAGMA table_info(messages)")}
    raw_json_select = ", raw_json" if "raw_json" in columns else ""
    for row in _iter_rows(
        engine,
        "SELECT id, group_id, platform_msg_id, user_id, timestamp, plain_text, "
        f"reply_to_msg_id, mentioned_bot{raw_json_select} FROM messages",
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
                "raw_json": row[8] if len(row) > 8 else None,
            }
        )
    return rows


def _load_retrievable_raw_message_ids(engine) -> set[int] | None:
    """Return active raw-v3 source IDs, or None for legacy snapshots.

    A raw-history gold message absent from every retrieval document is not a
    ranking test: no online channel can retrieve it.  Production snapshots
    therefore sample raw-history cases only from the materialized raw index.
    """

    tables = {
        str(row[0])
        for row in _iter_rows(
            engine,
            "SELECT name FROM sqlite_master WHERE type = 'table'",
        )
    }
    required = {"retrieval_documents", "retrieval_document_messages"}
    if not required <= tables:
        return None
    return {
        int(row[0])
        for row in _iter_rows(
            engine,
            "SELECT DISTINCT rdm.message_id "
            "FROM retrieval_document_messages AS rdm "
            "JOIN retrieval_documents AS rd ON rd.id = rdm.document_id "
            "AND rd.group_id = rdm.group_id "
            "WHERE rd.status = 'active' "
            "AND rd.document_kind = 'raw_message_v3' "
            "AND rd.source_table = 'messages'",
        )
    }


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
    candidates: list[str] = []
    for match in _CJK_SPAN.finditer(text_value):
        span = match.group(0)
        for size in range(min(6, len(span)), 2, -1):
            for start in range(0, len(span) - size + 1):
                candidate = span[start : start + size]
                if len(set(candidate)) < 3:
                    continue
                if any(char in _TOPIC_GENERIC_CHARS for char in candidate):
                    continue
                if any(stop in candidate for stop in _KEYWORD_STOP | _ALIAS_STOP):
                    continue
                candidates.append(candidate)
    return list(dict.fromkeys(candidates))[:8]


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
    return {
        "group_id": row["group_id"],
        "query": row["plain_text"][:200],
        "recent_context_message_ids": (),
        # A live @bot message is the current request, not historical memory
        # evidence.  Using it as gold makes source recall and abstention labels
        # impossible to interpret because the request is deliberately absent
        # from the historical retrieval corpus at evaluation time.
        "expected_evidence_message_ids": (),
        "category": "mention",
        "time_range": None,
        "quoted_context_message_id": row["reply_to_msg_id"],
        "schema_version": 1,
        "requester_uin": str(row["user_id"]),
        # Mention target and personal subject are independent axes.  A plain
        # @bot request names no historical person, so the subject is unbound
        # (None), not ambiguous/blocked (()).
        "allowed_subject_user_ids": None,
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
    query = (
        f"群里以前提到“{keyword}”时说了什么"
        if index % 2
        else f"之前关于“{keyword}”说过什么"
    )
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
    # Keep the summary benchmark subject-neutral.  Deriving a topic from the
    # first summary characters frequently copies a member alias into the
    # question, which correctly binds a personal subject while the benchmark
    # still expects an unbound group summary.  That label drift then filters
    # out the intended summary and masquerades as a retrieval regression.
    query = f"昨天群里{('说了' if index % 2 else '聊了')}什么"
    summary_end = summary.get("end_at")
    summary_clock = None
    if summary_end is not None:
        parsed_summary_end = datetime.fromisoformat(
            str(summary_end).replace("Z", "+00:00")
        )
        if parsed_summary_end.tzinfo is None:
            parsed_summary_end = parsed_summary_end.replace(tzinfo=UTC)
        summary_clock = (
            parsed_summary_end + timedelta(days=1)
        ).isoformat()
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
        # The query deliberately uses the relative word "昨天".  Place the
        # evaluation clock one calendar day after the summary boundary so
        # "昨天" denotes the day covered by the summary, without depending on
        # the wall clock date of the machine running the test.
        "now_iso": summary_clock,
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
        # The query explicitly names this member even though no memory answer
        # should be accepted.  Subject binding and evidence precision are
        # separate contracts; expecting None here marks correct binding as a
        # subject mismatch.
        "allowed_subject_user_ids": (
            (subject_id,) if subject_id.isdigit() else None
        ),
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
    aliases_by_group = _load_group_aliases(engine, messages)
    items = _load_memory_items(engine)
    summaries = _load_summaries(engine)
    if group_ids:
        items = [item for item in items if item["group_id"] in allowed]
        summaries = [summary for summary in summaries if summary["group_id"] in allowed]
    items = [
        item
        for item in items
        if not item["subject_id"].isdigit()
        or bool(
            aliases_by_group.get(item["group_id"], {}).get(
                int(item["subject_id"]), ()
            )
        )
    ]
    recent_items = _recent_item_pool(items, messages)
    recent_item_ids = {item["id"] for item in recent_items}
    groups = sorted({row["group_id"] for row in messages})
    if not groups:
        raise ValueError("snapshot has no messages; cannot build a dataset")
    mention_rows = [
        row for row in messages if row["mentioned_bot"] and row["plain_text"].strip()
    ]
    raw_rows = [
        row
        for row in messages
        if len(row["plain_text"]) >= 8
        and not row["mentioned_bot"]
        and not _is_bot_message(row)
    ]
    retrievable_raw_ids = _load_retrievable_raw_message_ids(engine)
    if retrievable_raw_ids is not None:
        raw_rows = [row for row in raw_rows if row["id"] in retrievable_raw_ids]
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
        kind_offsets: dict[str, int] = defaultdict(int)
        while len(cases) < target_fact:
            made = False
            for kind, kind_items in by_kind.items():
                if not kind_items:
                    continue
                pool = (
                    kind_items
                    if kind not in TEMPORAL_KINDS
                    else _sort_by_source_recency(
                        [
                            item
                            for item in kind_items
                            if item["id"] in recent_item_ids
                        ],
                        messages,
                    )
                )
                kind_offset = kind_offsets[kind]
                item = pool[kind_offset % len(pool)]
                kind_offsets[kind] = kind_offset + 1
                cases.append(
                    _build_fact_case(
                        item,
                        aliases_by_group.get(item["group_id"], {}),
                        rng,
                        index,
                    )
                )
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
                _sort_by_source_recency(recent_items, messages)
                if (temporal and recent_items)
                else items
            )
            item = pool[first_index % len(pool)]
            cases.append(
                _build_first_person_case(
                    item,
                    aliases_by_group.get(item["group_id"], {}),
                    first_index,
                )
            )
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
    if aliases_by_group:
        for group_id in groups:
            group_aliases = list(
                dict.fromkeys(
                    alias
                    for candidates in aliases_by_group.get(group_id, {}).values()
                    for alias in candidates
                )
            )
            if len(group_aliases) >= 2:
                cases.append(_build_ambiguous_case(group_id, group_aliases, misc_index))
                misc_index += 1
    for group_id in groups:
        foreign_alias = None
        local_members = set(aliases_by_group.get(group_id, {}))
        for foreign_group, members in aliases_by_group.items():
            if foreign_group == group_id:
                continue
            for user_id, candidates in members.items():
                if user_id not in local_members and candidates:
                    foreign_alias = candidates[0]
                    break
            if foreign_alias is not None:
                break
        if foreign_alias is not None:
            cases.append(_build_cross_group_case(group_id, foreign_alias, misc_index))
            misc_index += 1
    if items:
        for distractor_index, item in enumerate(items):
            if distractor_index >= target_misc:
                break
            cases.append(
                _build_distractor_case(
                    item,
                    aliases_by_group.get(item["group_id"], {}),
                    distractor_index,
                )
            )
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
    seen_queries: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for case in result:
        key = (
            str(case["category"]),
            str(case["query"]),
            str(case.get("requester_uin") or ""),
            int(case["group_id"]),
            str(case.get("now_iso") or ""),
            tuple(
                str(value)
                for value in (case.get("expected_evidence_message_ids") or ())
            ),
            str(case.get("expected_layer") or ""),
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
