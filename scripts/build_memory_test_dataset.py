"""Build a large stratified memory test dataset from a read-only snapshot DB.

The dataset is a JSONL stream of ``evaluate_memory_recall.EvaluationCase``
fields plus platform extras (kind, expected_layer, gold_text, target_message_id,
now_iso, tags). It feeds both the offline full-pipeline stage and the
full-chain real-model stage of the memory test platform.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime, time, timedelta
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from app.core.member_identity import normalize_member_alias
from app.core.time_utils import ASIA_SHANGHAI, stored_as_utc

from scripts.memory_test_metrics import dataset_coverage


KIND_TEMPLATES: dict[str, tuple[str, ...]] = {
    "preference": (
        "{alias}喜欢{obj}吗",
        "{alias}喜欢什么",
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
SUMMARY_GOLD_LIMIT = 3
SUMMARY_LEVELS = frozenset(
    {"episode", "semantic_window", "semantic_daily", "window", "daily"}
)
ANSWER_EXPECTATIONS = frozenset({"must_answer", "must_abstain", "either"})
IDENTITY_AUDIT_MEMORY_KINDS = frozenset(
    {"profile", "preference", "taboo", "relationship", "fact"}
)
IDENTITY_QUERY_PATTERN = re.compile(
    r"^(?:小町[，,:： ]*)?(?:我是谁|介绍一下我|你(?:还)?认识我吗|"
    r"你知道我是谁吗|我的(?:完整)?个人画像)[？?!！。 ]*$"
)
RELATIONSHIP_EVIDENCE_MARKERS = (
    "同事",
    "同学",
    "朋友",
    "好友",
    "亲戚",
    "亲属",
    "家人",
    "父亲",
    "母亲",
    "爸爸",
    "妈妈",
    "兄弟",
    "姐妹",
    "老师",
    "导师",
    "老板",
    "上司",
    "下属",
    "室友",
    "队友",
    "男友",
    "女友",
    "对象",
    "夫妻",
    "丈夫",
    "妻子",
    "主仆",
    "主人",
    "师徒",
    "搭档",
    "合作伙伴",
    "colleague",
    "coworker",
    "classmate",
    "friend",
    "roommate",
    "teacher",
    "mentor",
    "boss",
    "partner",
)


# A first-person question is requester-bound, so every template must describe
# exactly the kind of the gold memory item.  Do not make this a shared rotating
# list: doing so silently pairs e.g. a plan item with a preference question.
# Relationship/title questions deliberately have no entry here.  A single
# member fact cannot support claims about the bot's relationship with another
# member (such as "我和 X 谁是你的主人").
FIRST_PERSON_TEMPLATES_BY_KIND: dict[str, tuple[str, ...]] = {
    "preference": ("我喜欢什么", "我偏好什么"),
    "taboo": ("我讨厌什么", "我不喜欢什么", "我反感什么"),
    "profile": ("我是什么样的人", "我的完整个人画像", "介绍一下我"),
    "plan": ("我的计划是什么", "我最近有什么计划", "我准备做什么"),
    "decision": ("我做了什么决定", "我最近决定了什么"),
    "current": ("我最近在做什么",),
    "event": ("我最近发生了什么",),
    "running_joke": ("我有什么梗", "我的梗是什么", "我有什么名场面"),
}
_NUMERIC_MEMBER_ID = re.compile(r"^[0-9]+$")

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

DISTRACTOR_QUERY_SPECS = (
    ("{alias}的血型是什么", ("血型",)),
    ("{alias}的护照号码是多少", ("护照",)),
    ("{alias}的鞋码是多少", ("鞋码",)),
    ("{alias}的车牌号是多少", ("车牌",)),
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


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return stored_as_utc(value)
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        parsed = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
        return stored_as_utc(parsed)
    except ValueError:
        return None


def _parse_dt(value: Any) -> str | None:
    parsed = _parse_datetime(value)
    return parsed.isoformat() if parsed is not None else None


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


def _is_bot_author_message(row: Mapping[str, Any]) -> bool:
    """Identify bot authors without treating deleted member messages as bot output."""

    raw_json = _raw_payload(row.get("raw_json"))
    delivery_state = str(raw_json.get("delivery_state") or "").strip().lower()
    sender = raw_json.get("sender")
    sender_names = (
        tuple(str(sender.get(key) or "").strip() for key in ("nickname", "card"))
        if isinstance(sender, Mapping)
        else ()
    )
    return bool(
        str(row.get("platform_msg_id") or "").startswith("bot-reply-")
        or any(name in _ALIAS_STOP for name in sender_names if name)
        or delivery_state in {"sent", "blocked", "uncertain", "reserved"}
    )


def _load_group_aliases(
    engine,
    messages: Sequence[Mapping[str, Any]],
) -> dict[int, dict[int, list[str]]]:
    """Resolve aliases from the latest eligible sender snapshot per membership.

    This deliberately matches the production member loader. The global users
    table can contain a card overwritten by traffic from another group and is
    therefore not an authoritative group alias source.
    """

    del messages
    columns = {str(row[1]) for row in _iter_rows(engine, "PRAGMA table_info(messages)")}
    raw_json_select = ", raw_json" if "raw_json" in columns else ""
    snapshots = [
        {
            "id": int(row[0]),
            "group_id": int(row[1]),
            "user_id": int(row[2]) if row[2] is not None else 0,
            "timestamp": _parse_dt(row[3]),
            "raw_json": row[4] if len(row) > 4 else None,
        }
        for row in _iter_rows(
            engine,
            "SELECT id, group_id, user_id, timestamp"
            f"{raw_json_select} FROM messages WHERE group_id IS NOT NULL",
        )
    ]
    aliases: dict[int, dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen_memberships: set[tuple[int, int]] = set()
    ineligible_delivery_states = {"reserved", "blocked", "uncertain", "deleted"}
    for row in sorted(
        snapshots,
        key=lambda value: (
            _parse_datetime(value.get("timestamp")) or datetime.min.replace(tzinfo=UTC),
            int(value["id"]),
        ),
        reverse=True,
    ):
        group_id = int(row["group_id"])
        user_id = int(row["user_id"])
        membership = (group_id, user_id)
        delivery_state = str(
            _raw_payload(row.get("raw_json")).get("delivery_state") or ""
        ).strip()
        if delivery_state in ineligible_delivery_states:
            continue
        if membership in seen_memberships:
            continue
        seen_memberships.add(membership)
        for alias in _sender_aliases(row.get("raw_json")):
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
        if str(row[2]) not in SUMMARY_LEVELS:
            continue
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


def _select_resolvable_member_alias(
    subject_id: str,
    candidates: Sequence[str],
) -> str:
    """Choose an alias the runtime resolver can bind in a generated query."""

    for candidate in candidates:
        if len(normalize_member_alias(candidate)) >= 2:
            return candidate
    return f"QQ号{subject_id}"


def _build_fact_case(
    item: dict[str, Any],
    aliases: Mapping[int, Sequence[str]],
    rng: random.Random,
    index: int,
    gold_items: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    group_id = item["group_id"]
    subject_id = item["subject_id"]
    alias_candidates = aliases.get(int(subject_id), []) if subject_id.isdigit() else []
    if alias_candidates:
        alias = _select_resolvable_member_alias(subject_id, alias_candidates)
    elif subject_id.isdigit():
        alias = f"QQ号{subject_id}"
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
    supporting_items = (
        (item,)
        if "{obj}" in template
        else tuple(gold_items or (item,))
    )
    sources = tuple(
        dict.fromkeys(
            str(source_id)
            for supporting_item in supporting_items
            for source_id in supporting_item.get("source_ids", ())
            if str(source_id)
        )
    )
    gold = "\n".join(
        str(supporting_item.get("content") or supporting_item.get("object_text") or "")[:300]
        for supporting_item in supporting_items
    )[:900] or query
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
        "answer_expectation": "must_answer",
        "target_message_id": None,
        "now_iso": None,
        "tags": tuple(tags),
    }


def _is_supported_relationship_item(item: Mapping[str, Any]) -> bool:
    """Keep only member-bound facts that state a concrete relation predicate.

    The compaction model can occasionally label group commentary or a shared
    activity as ``relationship``.  A benchmark query such as "X 和谁是什么关系"
    is answerable only when the stored fact names a conventional relationship;
    co-occurrence, teasing, or an intention to meet is not sufficient.
    """

    if str(item.get("kind") or "") != "relationship":
        return True
    subject_id = str(item.get("subject_id") or "")
    if not _NUMERIC_MEMBER_ID.fullmatch(subject_id):
        return False
    support_text = " ".join(
        str(item.get(field) or "")
        for field in ("predicate", "object_text", "content")
    ).casefold()
    return any(marker.casefold() in support_text for marker in RELATIONSHIP_EVIDENCE_MARKERS)


def _build_identity_audit_cases(
    messages: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    *,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    """Build requester-bound cases from real first-person identity questions."""

    if start >= end:
        raise ValueError("identity audit start must be before end")
    message_by_platform_id = {
        str(row.get("platform_msg_id") or ""): row for row in messages
    }
    source_timestamp = {
        str(row.get("platform_msg_id") or ""): _parse_datetime(row.get("timestamp"))
        for row in messages
    }
    cases: list[dict[str, Any]] = []
    for row in messages:
        timestamp = _parse_datetime(row.get("timestamp"))
        query = str(row.get("plain_text") or "").strip()
        if (
            timestamp is None
            or not (start <= timestamp < end)
            or _is_bot_message(row)
            or not bool(row.get("mentioned_bot"))
            or IDENTITY_QUERY_PATTERN.fullmatch(query) is None
        ):
            continue
        subject_id = str(row.get("user_id") or "")
        if not _NUMERIC_MEMBER_ID.fullmatch(subject_id):
            continue
        supporting_items: list[Mapping[str, Any]] = []
        supporting_sources: list[str] = []
        for item in items:
            if (
                int(item.get("group_id") or 0) != int(row["group_id"])
                or str(item.get("subject_id") or "") != subject_id
                or str(item.get("kind") or "") not in IDENTITY_AUDIT_MEMORY_KINDS
                or not _is_supported_relationship_item(item)
            ):
                continue
            eligible_sources = [
                str(source_id)
                for source_id in item.get("source_ids") or ()
                if source_timestamp.get(str(source_id)) is not None
                and source_timestamp[str(source_id)] <= timestamp
            ]
            if not eligible_sources:
                continue
            supporting_items.append(item)
            supporting_sources.extend(eligible_sources)
        expected_sources = tuple(dict.fromkeys(supporting_sources))
        gold_text = "\n".join(
            str(item.get("content") or item.get("object_text") or "")[:300]
            for item in supporting_items
            if str(item.get("content") or item.get("object_text") or "").strip()
        )[:1200]
        must_answer = bool(expected_sources and gold_text)
        answer_message = message_by_platform_id.get(
            "bot-reply-" + str(row.get("platform_msg_id") or "")
        )
        tags = (
            "category=identity_audit",
            "intent=first_person_identity",
            "subject=requester",
            "real_mention=1",
            "historical_bot_answer=" + ("1" if answer_message is not None else "0"),
        )
        cases.append(
            {
                "group_id": int(row["group_id"]),
                "query": query,
                "recent_context_message_ids": (),
                "expected_evidence_message_ids": expected_sources,
                "category": "identity_audit",
                "time_range": None,
                "quoted_context_message_id": None,
                "schema_version": 1,
                "requester_uin": subject_id,
                "allowed_subject_user_ids": (subject_id,),
                "allowed_evidence_user_ids": (subject_id,),
                "expected_answer_mode": "current_fact",
                "expected_coverage_strategy": "relevance",
                "minimum_time_bucket_count": 0,
                "forbidden_evidence_message_ids": (),
                "gate_tags": tags,
                "contract_fields_complete": True,
                "kind": "profile",
                "expected_layer": "fact" if must_answer else "none",
                "gold_text": gold_text if must_answer else "",
                "answer_expectation": "must_answer" if must_answer else "must_abstain",
                # Internal numeric ID anchors the real recent-message window.
                "target_message_id": str(row["id"]),
                "now_iso": timestamp.isoformat(),
                "tags": tags,
                # Private audit fields. Public report projection never includes cases.
                "observed_answer": (
                    str(answer_message.get("plain_text") or "")
                    if answer_message is not None
                    else None
                ),
                "observed_answer_message_id": (
                    str(answer_message.get("platform_msg_id") or "")
                    if answer_message is not None
                    else None
                ),
            }
        )
    return cases


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
        "answer_expectation": "either",
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
        "answer_expectation": "must_answer",
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
    summary_clock = summary.get("now_iso")
    summary_end = summary.get("end_at")
    if summary_clock is None and summary_end is not None:
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
        "answer_expectation": "must_answer",
        "target_message_id": None,
        # The query deliberately uses the relative word "昨天".  Place the
        # evaluation clock one calendar day after the summary boundary so
        # "昨天" denotes the day covered by the summary, without depending on
        # the wall clock date of the machine running the test.
        "now_iso": summary_clock,
        "tags": ("category=summary", "layer=summary"),
    }


def _build_summary_day_cases(
    summaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one semantic summary case per group and Shanghai calendar day.

    The resolver normalizes every ``昨天`` query to a calendar-day range. A
    case per stored summary would duplicate the same question dozens of times
    while assigning mutually exclusive gold sources. Mirror the runtime
    summary loader's bounded ordering and use one union gold for that day.
    """

    by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    evaluation_days: set[tuple[int, object]] = set()
    for summary in summaries:
        group_id = int(summary["group_id"])
        start_at = _parse_datetime(summary.get("start_at"))
        end_at = _parse_datetime(summary.get("end_at"))
        if start_at is None or end_at is None or end_at <= start_at:
            continue
        by_group[group_id].append(summary)
        local_day = start_at.astimezone(ASIA_SHANGHAI).date()
        final_day = (end_at - timedelta(microseconds=1)).astimezone(
            ASIA_SHANGHAI
        ).date()
        while local_day <= final_day:
            evaluation_days.add((group_id, local_day))
            local_day += timedelta(days=1)

    cases: list[dict[str, Any]] = []
    for index, (group_id, local_day) in enumerate(sorted(evaluation_days)):
        day_start = datetime.combine(
            local_day,
            time.min,
            tzinfo=ASIA_SHANGHAI,
        ).astimezone(UTC)
        day_end = day_start + timedelta(days=1)
        overlapping: list[dict[str, Any]] = []
        for summary in by_group[group_id]:
            start_at = _parse_datetime(summary.get("start_at"))
            end_at = _parse_datetime(summary.get("end_at"))
            if (
                start_at is not None
                and end_at is not None
                and end_at > day_start
                and start_at < day_end
            ):
                overlapping.append(summary)
        # SummaryRepository orders DESC, limits to 3x the context limit, then
        # reverses; app.main applies the final context limit after validation.
        repository_window = sorted(
            overlapping,
            key=lambda item: (
                _parse_datetime(item.get("end_at")) or day_start,
                int(item.get("id") or 0),
            ),
            reverse=True,
        )[: SUMMARY_GOLD_LIMIT * 3]
        selected = list(reversed(repository_window))[:SUMMARY_GOLD_LIMIT]
        if not selected:
            continue
        # Any summary overlapping the requested day is valid evidence for an
        # open-ended day recap. The runtime packer may keep a different subset
        # than the loader's reference-text subset as raw/fact budgets vary.
        source_ids = tuple(
            dict.fromkeys(
                str(source_id)
                for item in overlapping
                for source_id in item.get("source_ids", ())
                if str(source_id)
            )
        )
        gold_text = "\n".join(
            str(item.get("content") or "")[:300] for item in selected
        )[:900]
        cases.append(
            _build_summary_case(
                {
                    "group_id": group_id,
                    "start_at": day_start.isoformat(),
                    "end_at": day_end.isoformat(),
                    "now_iso": day_end.isoformat(),
                    "source_ids": source_ids,
                    "content": gold_text,
                },
                index,
            )
        )
    return cases


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
        "answer_expectation": "must_abstain",
        "target_message_id": None,
        "now_iso": None,
        "tags": ("category=abstention", "layer=none"),
    }


def _build_first_person_case(
    item: dict[str, Any],
    aliases: Mapping[int, Sequence[str]],
    index: int,
    gold_items: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    del aliases  # First-person questions never interpolate another member.
    subject_id = str(item["subject_id"])
    kind = str(item["kind"])
    templates = FIRST_PERSON_TEMPLATES_BY_KIND.get(kind)
    if not _NUMERIC_MEMBER_ID.fullmatch(subject_id):
        raise ValueError("first-person case requires a numeric member subject")
    if not templates:
        raise ValueError(f"unsupported first-person memory kind: {kind!r}")
    query = templates[index % len(templates)]
    tags = [
        "kind=" + kind,
        "layer=fact",
        "subject=requester",
        "intent=first_person",
        "first_person_kind=" + kind,
    ]
    if kind in TEMPORAL_KINDS:
        tags.append("temporal_recent=1")
    supporting_items = tuple(gold_items or (item,))
    sources = tuple(
        dict.fromkeys(
            str(source_id)
            for supporting_item in supporting_items
            for source_id in supporting_item.get("source_ids", ())
            if str(source_id)
        )
    )
    gold = "\n".join(
        str(supporting_item.get("content") or supporting_item.get("object_text") or "")[:300]
        for supporting_item in supporting_items
    )[:900]
    return {
        "group_id": item["group_id"],
        "query": query,
        "recent_context_message_ids": (),
        "expected_evidence_message_ids": sources,
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
        "kind": kind,
        "expected_layer": "fact",
        "gold_text": gold,
        "answer_expectation": "must_answer",
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
        "answer_expectation": "must_abstain",
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
        "answer_expectation": "must_abstain",
        "target_message_id": None,
        "now_iso": None,
        "tags": ("category=cross_group", "layer=none"),
    }


def _build_distractor_case(
    item: dict[str, Any],
    aliases: Mapping[int, Sequence[str]],
    index: int,
    support_text: str,
) -> dict[str, Any] | None:
    subject_id = item["subject_id"]
    if not subject_id.isdigit():
        return None
    alias_candidates = aliases.get(int(subject_id), []) if subject_id.isdigit() else []
    alias = _select_resolvable_member_alias(subject_id, alias_candidates)
    template = ""
    for offset in range(len(DISTRACTOR_QUERY_SPECS)):
        candidate, support_markers = DISTRACTOR_QUERY_SPECS[
            (index + offset) % len(DISTRACTOR_QUERY_SPECS)
        ]
        if not any(marker in support_text for marker in support_markers):
            template = candidate
            break
    if not template:
        return None
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
        "kind": "distractor",
        "expected_layer": "none",
        "gold_text": "",
        "answer_expectation": "must_abstain",
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


def _validate_answer_contract(case: Mapping[str, Any]) -> None:
    """Reject contradictory answer labels before a case is emitted."""

    expectation = str(case.get("answer_expectation") or "")
    if expectation not in ANSWER_EXPECTATIONS:
        raise ValueError(f"invalid answer_expectation: {expectation!r}")
    has_gold = bool(str(case.get("gold_text") or "").strip())
    has_expected_evidence = bool(case.get("expected_evidence_message_ids"))
    if expectation == "must_answer" and not (has_gold and has_expected_evidence):
        raise ValueError("must_answer case requires gold text and expected evidence")
    if expectation in {"must_abstain", "either"} and (
        has_gold or has_expected_evidence
    ):
        raise ValueError(f"{expectation} case cannot carry gold evidence")
    tags = {str(value) for value in (case.get("tags") or ())}
    expected_subject = case.get("allowed_subject_user_ids")
    if (
        "multi_subject" in tags
        and expected_subject not in (None, (), [])
        and (has_gold or has_expected_evidence)
    ):
        raise ValueError(
            "multi-subject case cannot require an exact subject and nonempty gold"
        )
    if str(case.get("category") or "") == "first_person":
        kind = str(case.get("kind") or "")
        templates = FIRST_PERSON_TEMPLATES_BY_KIND.get(kind)
        query = str(case.get("query") or "")
        requester = str(case.get("requester_uin") or "")
        actual_subject = tuple(
            str(value) for value in (case.get("allowed_subject_user_ids") or ())
        )
        if not templates or query not in templates:
            raise ValueError("first-person query-kind mismatch")
        if not _NUMERIC_MEMBER_ID.fullmatch(requester):
            raise ValueError("first-person requester must be a numeric member")
        if actual_subject != (requester,):
            raise ValueError("first-person subject/requester mismatch")
        required_tags = {
            "kind=" + kind,
            "first_person_kind=" + kind,
            "subject=requester",
            "intent=first_person",
        }
        if not required_tags <= tags:
            raise ValueError("first-person contract tags are incomplete")
        if kind in TEMPORAL_KINDS and "temporal_recent=1" not in tags:
            raise ValueError("temporal first-person case must use recent evidence")


def build_cases(
    engine,
    *,
    count: int = 3000,
    seed: int = 20260811,
    group_ids: Sequence[int] | None = None,
    identity_audit_start: str | None = None,
    identity_audit_end: str | None = None,
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
    bot_user_ids = {
        str(row["user_id"])
        for row in messages
        if _is_bot_author_message(row)
    }
    observed_memberships = {
        (int(row["group_id"]), str(row["user_id"]))
        for row in messages
        if str(row["user_id"]) not in bot_user_ids
    }
    items = [
        item
        for item in items
        if not item["subject_id"].isdigit()
        or (
            int(item["group_id"]), str(item["subject_id"])
        ) in observed_memberships
    ]
    recent_items = _recent_item_pool(items, messages)
    recent_item_ids = {item["id"] for item in recent_items}
    groups = sorted({row["group_id"] for row in messages})
    if not groups:
        raise ValueError("snapshot has no messages; cannot build a dataset")
    mention_rows = [
        row
        for row in messages
        if row["mentioned_bot"]
        and row["plain_text"].strip()
        and not _is_bot_message(row)
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
    if bool(identity_audit_start) != bool(identity_audit_end):
        raise ValueError("identity audit start and end must be provided together")
    if identity_audit_start and identity_audit_end:
        start = _parse_datetime(identity_audit_start)
        end = _parse_datetime(identity_audit_end)
        if start is None or end is None:
            raise ValueError("identity audit bounds must be ISO datetimes")
        cases.extend(
            _build_identity_audit_cases(messages, items, start=start, end=end)
        )
    index = 0
    target_fact = int(count * 0.40)
    target_mention = int(count * 0.15)
    target_raw = int(count * 0.20)
    target_first = int(count * 0.08)
    target_misc = int(count * 0.09)
    target_abstention = int(count * 0.08)
    # 1) Structured fact cases (round-robin over kinds to keep coverage).
    fact_items = [item for item in items if _is_supported_relationship_item(item)]
    if fact_items:
        by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in fact_items:
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
                supporting_items = [
                    candidate
                    for candidate in pool
                    if candidate["group_id"] == item["group_id"]
                    and candidate["subject_id"] == item["subject_id"]
                ]
                cases.append(
                    _build_fact_case(
                        item,
                        aliases_by_group.get(item["group_id"], {}),
                        rng,
                        index,
                        supporting_items,
                    )
                )
                made = True
                index += 1
                if len(cases) >= target_fact:
                    break
            if not made:
                break
    # 1b) First-person variants use an explicit kind -> template mapping.
    # Group-scoped/non-numeric subjects cannot be requesters.  Temporal kinds
    # are restricted to the recent pool before case construction, never
    # relabelled as abstention when their gold is unsuitable.
    first_person_by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        kind = str(item["kind"])
        subject_id = str(item["subject_id"])
        if not _NUMERIC_MEMBER_ID.fullmatch(subject_id):
            continue
        if kind not in FIRST_PERSON_TEMPLATES_BY_KIND:
            continue
        if kind in TEMPORAL_KINDS and item["id"] not in recent_item_ids:
            continue
        first_person_by_kind[kind].append(item)
    first_person_capacity = sum(
        len(kind_items) * len(FIRST_PERSON_TEMPLATES_BY_KIND[kind])
        for kind, kind_items in first_person_by_kind.items()
    )
    if first_person_capacity:
        first_offsets: dict[str, int] = defaultdict(int)
        first_kind_capacities = {
            kind: len(kind_items) * len(FIRST_PERSON_TEMPLATES_BY_KIND[kind])
            for kind, kind_items in first_person_by_kind.items()
        }
        first_emitted = 0
        while (
            first_emitted < target_first
            and first_emitted < first_person_capacity
        ):
            made = False
            for kind in sorted(first_person_by_kind):
                if first_emitted >= target_first:
                    break
                pool = first_person_by_kind[kind]
                offset = first_offsets[kind]
                if offset >= first_kind_capacities[kind]:
                    continue
                item = pool[offset % len(pool)]
                supporting_items = [
                    candidate
                    for candidate in pool
                    if candidate["group_id"] == item["group_id"]
                    and candidate["subject_id"] == item["subject_id"]
                ]
                # Enumerate each item/template pair before repeating, keeping
                # both the bucket size and the generated cases deterministic.
                template_index = offset // len(pool)
                cases.append(
                    _build_first_person_case(
                        item,
                        aliases_by_group.get(item["group_id"], {}),
                        template_index,
                        supporting_items,
                    )
                )
                first_offsets[kind] = offset + 1
                first_emitted += 1
                made = True
            if not made:
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
    # 4) Summary/dated cases. ``昨天`` is one calendar-day intent, not one
    # independent question per stored window/episode summary.
    for summary_case in _build_summary_day_cases(summaries):
        if len(cases) >= target_fact + target_first + target_mention + target_raw + target_misc:
            break
        cases.append(summary_case)
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
        support_parts: dict[tuple[int, str], list[str]] = defaultdict(list)
        for message in messages:
            support_parts[
                (int(message["group_id"]), str(message["user_id"]))
            ].append(str(message.get("plain_text") or ""))
        for candidate in items:
            support_parts[
                (int(candidate["group_id"]), str(candidate["subject_id"]))
            ].extend(
                (
                    str(candidate.get("predicate") or ""),
                    str(candidate.get("object_text") or ""),
                    str(candidate.get("content") or ""),
                )
            )
        distractor_seen: set[tuple[int, str, str]] = set()
        distractor_count = 0
        for distractor_index, item in enumerate(items):
            if distractor_count >= target_misc:
                break
            case = _build_distractor_case(
                item,
                aliases_by_group.get(item["group_id"], {}),
                distractor_index,
                "\n".join(
                    support_parts.get(
                        (int(item["group_id"]), str(item["subject_id"])), ()
                    )
                ),
            )
            if case is None:
                continue
            identity = (
                int(case["group_id"]),
                str(item["subject_id"]),
                str(case["query"]),
            )
            if identity in distractor_seen:
                continue
            distractor_seen.add(identity)
            cases.append(case)
            distractor_count += 1
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
        # Validate every constructed candidate, including cases later removed
        # by the requested output limit or deterministic de-duplication.
        _validate_answer_contract(case)
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
    parser.add_argument("--identity-audit-start", default="")
    parser.add_argument("--identity-audit-end", default="")
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=None,
        help="Write a stratification coverage JSON report alongside the dataset.",
    )
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
    cases = build_cases(
        engine,
        count=args.count,
        seed=args.seed,
        group_ids=group_ids,
        identity_audit_start=args.identity_audit_start or None,
        identity_audit_end=args.identity_audit_end or None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    kinds = defaultdict(int)
    for case in cases:
        kinds[str(case["kind"])] += 1
    summary = {"cases": len(cases), "by_kind": dict(kinds)}
    if args.coverage_report is not None:
        coverage = dataset_coverage(cases)
        args.coverage_report.parent.mkdir(parents=True, exist_ok=True)
        with args.coverage_report.open("w", encoding="utf-8") as handle:
            json.dump(coverage, handle, ensure_ascii=False, indent=2)
        summary["coverage_report"] = str(args.coverage_report)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
