"""Real-history memory stress evaluation.

Generates realistic question cases from the actual database (member facts,
raw-history topics, cross-group isolation, first person and pronoun
references), runs the full resolver -> retrieval -> packer pipeline offline,
and reports per-category recall. Run against a read-only backup copy.

Commands:
  plan       print case inventory without evaluating
  run        evaluate every case and write/print a JSON report
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import re
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import bindparam, create_engine, event as sa_event, text
from sqlalchemy.pool import NullPool

from app.config import AppSettings
from app.core.legacy_memory_context import GroupMemoryContextRequest
from app.core.member_identity import (
    group_member_identities_from_messages,
    normalize_member_alias,
)
from app.core.memory_context_packer import EvidenceMessage
from app.main import build_llm_client, build_memory_runtime
from app.storage.db import session_scope
from app.storage.repositories import MessageRepository


STOPWORDS = frozenset(
    {
        "什么",
        "怎么",
        "怎么样",
        "这个",
        "那个",
        "我们",
        "你们",
        "他们",
        "她们",
        "今天",
        "昨天",
        "明天",
        "可以",
        "没有",
        "一个",
        "自己",
        "知道",
        "觉得",
        "喜欢",
        "但是",
        "因为",
        "所以",
        "如果",
        "还是",
        "就是",
        "真的",
        "现在",
        "之前",
        "以后",
    }
)

KIND_QUERY_TEMPLATES = {
    "preference": ("{alias}喜欢{object}吗？", "{alias}的{object}偏好？"),
    "taboo": ("{alias}讨厌什么？", "{alias}不喜欢什么？"),
    "profile": ("介绍一下{alias}", "{alias}是什么样的人？"),
    "plan": ("{alias}打算做什么？", "{alias}的计划是什么？"),
    "decision": ("{alias}决定了什么？",),
    "relationship": ("{alias}和谁是什么关系？",),
    "event": ("{alias}最近发生了什么？",),
    "current": ("{alias}最近在做什么？",),
    "running_joke": ("{alias}有什么梗？",),
}

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{3,6}")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
_CLEAN_ALIAS = re.compile(
    r"^(?=.*[\u4e00-\u9fffA-Za-z0-9])[\u4e00-\u9fffA-Za-z0-9_\-]{2,16}$"
)
_NOISY_KEYWORD = re.compile(r"[吗呢了吧的你我他她这那是不有和与]")


def _runtime_in_scope_aliases(engine, group_id: int) -> set[str]:
    """Aliases the runtime member loader would treat as in-scope."""
    with session_scope(engine) as session:
        rows = MessageRepository(session).list_recent_group_member_messages(
            group_id=None,
            limit=None,
        )
    members = group_member_identities_from_messages(
        rows,
        target_group_id=int(group_id),
    )
    return {
        normalize_member_alias(alias)
        for member in members
        if member.in_scope
        for alias in (member.nickname, member.group_card)
        if normalize_member_alias(alias)
    }


def _iter_rows(engine, statement: str, parameters: dict | None = None):
    with engine.connect() as connection:
        yield from connection.execute(text(statement), parameters or {})


def _member_alias(engine, user_id: int) -> str:
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT nickname, group_card FROM users "
                "WHERE user_id = :user_id"
            ),
            {"user_id": int(user_id)},
        ).one_or_none()
    candidates = (
        str(row.nickname or "").strip(),
        str(row.group_card or "").strip(),
    ) if row is not None else ()
    for candidate in candidates:
        if _CLEAN_ALIAS.fullmatch(candidate):
            return candidate
    return f"成员{user_id}"


def _alias_is_unique(engine, group_id: int, alias: str, user_id: int) -> bool:
    if not alias or alias.startswith("成员"):
        return False
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT DISTINCT u.user_id FROM users u "
                "JOIN messages m ON m.user_id = u.user_id "
                "WHERE m.group_id = :g AND (u.nickname = :a OR u.group_card = :a)"
            ),
            {"g": int(group_id), "a": alias},
        )
        ids = [int(row[0]) for row in rows]
    return ids == [int(user_id)]


def _clean_object(value: str, *, max_length: int = 12) -> str | None:
    cleaned = str(value or "").strip()
    if not 2 <= len(cleaned) <= max_length:
        return None
    if not _CLEAN_ALIAS.fullmatch(cleaned):
        return None
    return cleaned


def _parse_source_ids(value) -> list[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return []
    else:
        parsed = value
    if not isinstance(parsed, list):
        return []
    return [
        str(item)
        for item in parsed
        if isinstance(item, str) and item.strip()
    ]


def _clean_keyword(value: str) -> str | None:
    keyword = str(value or "").strip()
    if not 3 <= len(keyword) <= 6:
        return None
    if _NOISY_KEYWORD.search(keyword):
        return None
    if not _CLEAN_ALIAS.fullmatch(keyword):
        return None
    return keyword


def _topic_keywords(
    engine,
    group_id: int,
    user_id: int,
    *,
    min_occurrences: int = 2,
) -> list[str]:
    texts = [
        str(row[0])
        for row in _iter_rows(
            engine,
            "SELECT plain_text FROM messages "
            "WHERE group_id = :g AND user_id = :u AND plain_text != ''",
            {"g": int(group_id), "u": int(user_id)},
        )
    ]
    counts: dict[str, int] = {}
    for value in texts:
        seen: set[str] = set()
        for piece in (*_CJK_RUN.findall(value), *_LATIN_WORD.findall(value)):
            cleaned = _clean_keyword(piece)
            if cleaned is None:
                continue
            normalized = cleaned.lower()
            if (
                normalized not in STOPWORDS
                and normalized not in seen
            ):
                seen.add(normalized)
                counts[normalized] = counts.get(normalized, 0) + 1
    return sorted(
        (keyword for keyword, count in counts.items() if count >= min_occurrences),
        key=lambda keyword: (-counts[keyword], keyword),
    )


def _group_ids(engine) -> list[int]:
    return [
        int(row[0])
        for row in _iter_rows(
            engine,
            "SELECT DISTINCT group_id FROM messages "
            "WHERE group_id IS NOT NULL ORDER BY group_id",
        )
    ]


def _group_user_ids(engine, group_id: int) -> list[int]:
    return [
        int(row[0])
        for row in _iter_rows(
            engine,
            "SELECT DISTINCT user_id FROM messages "
            "WHERE group_id = :g ORDER BY user_id",
            {"g": int(group_id)},
        )
    ]


def _message_ids_for_keyword(engine, group_id: int, user_id: int, keyword: str) -> list[str]:
    return [
        str(row[0])
        for row in _iter_rows(
            engine,
            "SELECT platform_msg_id FROM messages "
            "WHERE group_id = :g AND user_id = :u "
            "AND plain_text LIKE :pattern ORDER BY id",
            {
                "g": int(group_id),
                "u": int(user_id),
                "pattern": f"%{keyword}%",
            },
        )
    ]


def _build_cases(
    engine,
    *,
    limit_cases: int | None,
    excluded_user_ids: set[int] = frozenset(),
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    group_ids = _group_ids(engine)
    in_scope_aliases_by_group: dict[int, set[str]] = {
        int(group_id): _runtime_in_scope_aliases(engine, int(group_id))
        for group_id in group_ids
    }
    fact_rows = list(
        _iter_rows(
            engine,
            "SELECT scope_id, subject_id, memory_kind, content, predicate, "
            "object_text, source_msg_id, source_msg_ids FROM memory_items "
            "WHERE status='active' AND subject_type='user' "
            "ORDER BY importance DESC, id DESC LIMIT 4000",
        )
    )
    for scope_id, subject_id, kind, content, predicate, object_text, source_id, source_ids in fact_rows:
        if kind not in KIND_QUERY_TEMPLATES or int(subject_id) in excluded_user_ids:
            continue
        alias = _member_alias(engine, int(subject_id))
        if (
            normalize_member_alias(alias)
            not in in_scope_aliases_by_group.get(int(scope_id), set())
        ):
            continue
        if not _alias_is_unique(engine, int(scope_id), alias, int(subject_id)):
            continue
        expected = list(
            dict.fromkeys(
                [
                    *_parse_source_ids(source_ids),
                    *([str(source_id)] if source_id else []),
                ]
            )
        )
        if not expected:
            continue
        for template in KIND_QUERY_TEMPLATES[kind]:
            if "{object}" in template:
                clean_object = _clean_object(str(object_text or ""))
                if clean_object is None:
                    if kind == "preference":
                        query_text = f"{alias}喜欢什么？"
                    elif kind == "taboo":
                        query_text = f"{alias}讨厌什么？"
                    else:
                        continue
                else:
                    query_text = template.format(alias=alias, object=clean_object)
            else:
                query_text = template.format(alias=alias)
            if len(query_text) < 4:
                continue
            cases.append(
                {
                    "category": f"fact:{kind}",
                    "group_id": int(scope_id),
                    "query": query_text,
                    "expected": expected,
                    "content": str(content)[:80],
                }
            )

    for group_id in group_ids:
        for user_id in [
            value
            for value in _group_user_ids(engine, group_id)[:120]
            if value not in excluded_user_ids
        ]:
            alias = _member_alias(engine, user_id)
            if (
                normalize_member_alias(alias)
                not in in_scope_aliases_by_group.get(int(group_id), set())
            ):
                continue
            if not _alias_is_unique(engine, group_id, alias, user_id):
                continue
            for keyword in _topic_keywords(engine, group_id, user_id)[:4]:
                message_ids = _message_ids_for_keyword(engine, group_id, user_id, keyword)
                if len(message_ids) < 2:
                    continue
                for template in (
                    "{alias}以前说过{keyword}吗？",
                    "{alias}怎么看{keyword}？",
                    "{alias}聊过{keyword}吗？",
                ):
                    cases.append(
                        {
                            "category": "raw_history",
                            "group_id": group_id,
                            "query": template.format(alias=alias, keyword=keyword),
                            "expected": message_ids,
                            "content": keyword,
                        }
                    )

    for group_id in group_ids:
        for user_id in [
            value
            for value in _group_user_ids(engine, group_id)[:80]
            if value not in excluded_user_ids
        ]:
            for keyword in _topic_keywords(engine, group_id, user_id)[:2]:
                alias = _member_alias(engine, user_id)
                if (
                    normalize_member_alias(alias)
                    not in in_scope_aliases_by_group.get(int(group_id), set())
                ):
                    continue
                if not _alias_is_unique(engine, group_id, alias, user_id):
                    continue
                other_groups = [
                    int(row[0])
                    for row in _iter_rows(
                        engine,
                        "SELECT DISTINCT group_id FROM messages "
                        "WHERE user_id = :u AND group_id != :g "
                        "AND plain_text LIKE :pattern ORDER BY group_id",
                        {
                            "u": int(user_id),
                            "g": int(group_id),
                            "pattern": f"%{keyword}%",
                        },
                    )
                ]
                if not other_groups:
                    continue
                cases.append(
                    {
                        "category": "cross_group",
                        "group_id": group_id,
                        "query": f"{alias}聊过{keyword}吗？",
                        "expected": _message_ids_for_keyword(
                            engine,
                            group_id,
                            user_id,
                            keyword,
                        ),
                        "forbidden_groups": other_groups,
                        "content": keyword,
                    }
                )

    first_person = list(
        _iter_rows(
            engine,
            "SELECT scope_id, subject_id, content, source_msg_id, source_msg_ids "
            "FROM memory_items WHERE status='active' AND subject_type='user' "
            "AND memory_kind IN ('preference','taboo') LIMIT 60",
        )
    )
    for scope_id, subject_id, content, source_id, source_ids in first_person:
        if int(subject_id) in excluded_user_ids:
            continue
        alias = _member_alias(engine, int(subject_id))
        if (
            normalize_member_alias(alias)
            not in in_scope_aliases_by_group.get(int(scope_id), set())
        ):
            continue
        expected = list(
            dict.fromkeys(
                [
                    *_parse_source_ids(source_ids),
                    *([str(source_id)] if source_id else []),
                ]
            )
        )
        if not expected:
            continue
        content_text = str(content or "")
        if any(token in content_text for token in ("讨厌", "不喜欢", "反感")):
            query_text = "我讨厌什么？"
        elif any(token in content_text for token in ("喜欢", "偏好", "最爱")):
            query_text = "我最喜欢什么？"
        else:
            continue
        cases.append(
            {
                "category": "first_person",
                "group_id": int(scope_id),
                "query": query_text,
                "expected": expected,
                "requester_id": int(subject_id),
                "content": str(content)[:80],
            }
        )

    if limit_cases is not None and limit_cases > 0:
        by_category: dict[str, list[dict[str, Any]]] = {}
        for case in cases:
            by_category.setdefault(case["category"], []).append(case)
        sampled: list[dict[str, Any]] = []
        buckets = [items for items in by_category.values() if items]
        index = 0
        while buckets and len(sampled) < limit_cases:
            for bucket in buckets:
                if index < len(bucket):
                    sampled.append(bucket[index])
                    if len(sampled) >= limit_cases:
                        break
            index += 1
        cases = sampled
    return cases


def _build_recent(engine, group_id: int, limit: int = 40) -> tuple[EvidenceMessage, ...]:
    with session_scope(engine) as session:
        rows = MessageRepository(session).list_recent_group_messages(
            group_id=group_id,
            limit=limit,
        )
    return tuple(
        EvidenceMessage(
            source_msg_id=row.platform_msg_id,
            speaker=str(row.user_id),
            content=str(row.plain_text or ""),
            sent_at=row.timestamp,
            blocked=False,
            group_id=group_id,
            is_bot=False,
            user_id=row.user_id,
        )
        for row in rows
    )


def _evaluate_case(runtime, settings, engine, case: dict[str, Any]) -> dict[str, Any]:
    group_id = int(case["group_id"])
    recent = _build_recent(engine, group_id)
    request = GroupMemoryContextRequest(
        group_id=group_id,
        query=case["query"],
        recent_messages=recent,
        quoted_message=None,
        target_message_id="stress-query",
        available_input=34000,
        now=datetime.now(UTC),
        current_user_id=int(case.get("requester_id") or settings.owner_qq),
        use_full_history=True,
    )
    try:
        trace = runtime.v2_provider.evaluate(request)
    except Exception as exc:
        return {"error": type(exc).__name__, "ok": False}
    packed = trace.result.packed_context
    selected = set(trace.result.selected_source_msg_ids)
    fact_sources = {
        source_id
        for fact in packed.facts
        for source_id in fact.source_msg_ids
    }
    expected = set(case.get("expected") or ())
    hit = bool(expected & (selected | fact_sources))
    violation = False
    forbidden = set(case.get("forbidden_groups") or ())
    if forbidden and selected:
        with engine.connect() as connection:
            leaked = [
                str(row[0])
                for row in connection.execute(
                    text(
                        "SELECT platform_msg_id FROM messages "
                        "WHERE group_id IN :groups AND platform_msg_id IN :ids"
                    ).bindparams(
                        bindparam("groups", expanding=True),
                        bindparam("ids", expanding=True),
                    ),
                    {"groups": tuple(forbidden), "ids": tuple(selected)},
                )
            ]
        violation = bool(leaked)
    return {
        "ok": hit and not violation,
        "hit": hit,
        "violation": violation,
        "subject": trace.resolved_query.subject_ids,
        "facts": len(packed.facts),
        "summaries": len(packed.summaries),
        "selected": len(selected),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-history memory stress evaluation."
    )
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    settings = AppSettings().model_copy(
        update={
            "memory_query_rewrite_enabled": False,
            "memory_retrieval_channel_timeout_seconds": 0.5,
        }
    )
    engine = create_engine(
        f"sqlite:///{args.database}",
        connect_args={"timeout": 60},
        poolclass=NullPool,
        future=True,
    )

    @sa_event.listens_for(engine, "connect")
    def _set_busy_timeout(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA busy_timeout=60000")
        cursor.close()

    with engine.begin() as connection:
        connection.execute(text("PRAGMA journal_mode=WAL;"))
    try:
        cases = _build_cases(
            engine,
            limit_cases=args.limit_cases,
            excluded_user_ids={int(settings.bot_qq)},
        )
        if args.command == "plan":
            counts: dict[str, int] = {}
            for case in cases:
                counts[case["category"]] = counts.get(case["category"], 0) + 1
            print(
                json.dumps(
                    {"total": len(cases), "categories": counts},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0

        llm_client = build_llm_client(settings=settings, engine=engine)
        runtime = build_memory_runtime(
            settings=settings,
            engine=engine,
            llm_client=llm_client,
            bot_display_name="小町",
        )
        results: list[dict[str, Any]] = []
        for case in cases:
            outcome = _evaluate_case(runtime, settings, engine, case)
            results.append(
                {
                    "category": case["category"],
                    "group_id": case["group_id"],
                    "query": case["query"],
                    "expected_count": len(case.get("expected") or ()),
                    "expected": case.get("expected"),
                    "content": case.get("content", ""),
                    **outcome,
                }
            )
        by_category: dict[str, dict[str, Any]] = {}
        for result in results:
            category = result["category"]
            bucket = by_category.setdefault(
                category,
                {
                    "cases": 0,
                    "hits": 0,
                    "violations": 0,
                    "errors": 0,
                    "failures": [],
                },
            )
            bucket["cases"] += 1
            if result.get("error"):
                bucket["errors"] += 1
            elif result["ok"]:
                bucket["hits"] += 1
            else:
                if result.get("violation"):
                    bucket["violations"] += 1
                if len(bucket["failures"]) < 4:
                    bucket["failures"].append(
                        {
                            "query": result["query"],
                            "subject": result["subject"],
                            "facts": result["facts"],
                            "selected": result["selected"],
                            "expected": result["expected"][:3],
                        }
                    )
        for bucket in by_category.values():
            bucket["hit_rate"] = (
                round(bucket["hits"] / bucket["cases"], 3) if bucket["cases"] else 0.0
            )
        report = {
            "total_cases": len(results),
            "total_ok": sum(1 for result in results if result.get("ok")),
            "categories": by_category,
        }
        rendered = json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
