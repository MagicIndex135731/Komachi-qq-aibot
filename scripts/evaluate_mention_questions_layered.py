"""Layered-memory evaluation over real @bot mention questions.

Builds a test set from real group messages that mentioned the bot, runs the
offline resolver -> retrieval -> packer pipeline for each question, and
classifies issues per memory layer:

- raw layer: original message evidence (evidence segments)
- fact layer: structured memory_items (preference/profile/taboo/...)
- summary layer: episode/semantic summaries
- subject binding: requester/member resolution
- grounding: evidence present when the question needs memory

Usage (inside the container, on a read-only DB copy):
    python -m scripts.evaluate_mention_questions_layered \
        --database /tmp/eval.db --group-id <gid> --limit 150 \
        --output /tmp/mention-results.jsonl --report /tmp/mention-report.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import create_engine, event as sa_event, text
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


FACT_EXPECTED_MODES = {"current_fact", "assessment"}
FACT_EXPECTED_KINDS = {"preference", "taboo", "profile", "relationship", "decision", "plan", "current"}
SUMMARY_EXPECTED_MODES = {"dated_history", "summary"}
RAW_EXPECTED_MODES = {"exact", "mention", "general_history", "assessment", "dated_history"}


def _iter_rows(engine, statement: str, parameters: dict | None = None):
    with engine.connect() as connection:
        yield from connection.execute(text(statement), parameters or {})


def _message_columns(engine) -> set[str]:
    with engine.connect() as connection:
        rows = connection.execute(text("PRAGMA table_info(messages)"))
        return {str(row[1]) for row in rows}


def _mention_messages(engine, group_id: int, bot_user_id: int, limit: int):
    rows = list(
        _iter_rows(
            engine,
            "SELECT id, platform_msg_id, user_id, timestamp, plain_text, "
            "reply_to_msg_id, mentioned_bot FROM messages "
            "WHERE group_id = :g AND user_id != :bot AND mentioned_bot = 1 "
            "AND plain_text != '' ORDER BY id DESC LIMIT :limit",
            {"g": int(group_id), "bot": int(bot_user_id), "limit": int(limit)},
        )
    )
    return list(reversed(rows))


def _recent_before(engine, group_id: int, before_id: int, limit: int):
    rows = list(
        _iter_rows(
            engine,
            "SELECT id, platform_msg_id, user_id, timestamp, plain_text, "
            "reply_to_msg_id, mentioned_bot FROM messages "
            "WHERE group_id = :g AND id < :before ORDER BY id DESC LIMIT :limit",
            {"g": int(group_id), "before": int(before_id), "limit": int(limit)},
        )
    )
    return list(reversed(rows))


def _quoted_message(engine, group_id: int, platform_msg_id: str):
    if not platform_msg_id:
        return None
    rows = list(
        _iter_rows(
            engine,
            "SELECT id, platform_msg_id, user_id, timestamp, plain_text, "
            "reply_to_msg_id, mentioned_bot FROM messages "
            "WHERE group_id = :g AND platform_msg_id = :pid LIMIT 1",
            {"g": int(group_id), "pid": str(platform_msg_id)},
        )
    )
    return rows[0] if rows else None


def _evidence(row, group_id: int, bot_user_id: int) -> EvidenceMessage:
    return EvidenceMessage(
        source_msg_id=str(row[1]),
        speaker=str(row[2]),
        content=str(row[4] or ""),
        sent_at=_parse_dt(row[3]),
        blocked=False,
        group_id=int(group_id),
        reply_to_msg_id=str(row[5]) if row[5] else None,
        is_bot=int(row[2]) == int(bot_user_id),
        user_id=int(row[2]),
    )


def _parse_dt(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _in_scope_aliases(engine, group_id: int) -> set[str]:
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


def _fact_kinds_for_texts(engine, group_id: int, texts: Sequence[str]) -> list[str]:
    if not texts:
        return []
    kinds: list[str] = []
    for value in dict.fromkeys(texts):
        rows = list(
            _iter_rows(
                engine,
                "SELECT memory_kind FROM memory_items "
                "WHERE scope_id = :g AND content = :c AND status = 'active' LIMIT 4",
                {"g": str(group_id), "c": str(value)},
            )
        )
        kinds.extend(str(row[0]) for row in rows)
    return list(dict.fromkeys(kinds))


def _topic_overlap(topic_terms: Sequence[str], texts: Sequence[str]) -> list[str]:
    terms = [term for term in topic_terms if len(str(term).strip()) >= 2]
    if not terms:
        return []
    hits: list[str] = []
    for term in terms:
        if any(term in text for text in texts):
            hits.append(term)
    return hits


def _evaluate_case(runtime, settings, engine, case: dict[str, Any], aliases: set[str]):
    group_id = int(case["group_id"])
    target = case["target"]
    recent_rows = _recent_before(engine, group_id, int(target[0]), settings.memory_adaptive_max_recent_messages)
    recent = tuple(_evidence(row, group_id, settings.bot_qq) for row in recent_rows)
    quoted_row = _quoted_message(engine, group_id, str(target[5]) if target[5] else "")
    quoted = _evidence(quoted_row, group_id, settings.bot_qq) if quoted_row is not None else None
    request = GroupMemoryContextRequest(
        group_id=group_id,
        query=str(case["query"]),
        recent_messages=recent,
        quoted_message=quoted,
        target_message_id=str(target[1]),
        available_input=34000,
        now=_parse_dt(target[3]),
        current_user_id=int(target[2]),
        use_full_history=True,
    )
    try:
        trace = runtime.v2_provider.evaluate(request)
    except Exception as exc:
        return {"error": type(exc).__name__}
    packed = trace.result.packed_context
    resolved = trace.resolved_query
    selected = set(trace.result.selected_source_msg_ids)
    fact_texts = [fact.text for fact in packed.facts]
    fact_kinds = _fact_kinds_for_texts(engine, group_id, fact_texts)
    summary_texts = [summary.text for summary in packed.summaries]
    segment_texts = [
        message.content
        for segment in packed.evidence_segments
        for message in segment.messages
    ]
    all_evidence_texts = [*fact_texts, *summary_texts, *segment_texts]
    query_terms = list(getattr(resolved, "topic_terms", ()) or ())
    entities = list(getattr(resolved, "entities", ()) or ())
    overlap = _topic_overlap([*query_terms, *entities], all_evidence_texts)
    answer_mode = getattr(resolved, "answer_mode", "")
    subject_ids = getattr(resolved, "subject_ids", None)
    subject_bound = bool(subject_ids)

    mention_alias = any(
        alias and alias in str(case["query"])
        for alias in aliases
        if len(str(alias).strip()) >= 2
    )
    issues: list[str] = []
    evidence_present = bool(all_evidence_texts)
    if answer_mode in RAW_EXPECTED_MODES and not segment_texts and evidence_present is False:
        issues.append("raw_layer_miss")
    if answer_mode in FACT_EXPECTED_MODES and not fact_texts and mention_alias:
        issues.append("fact_layer_miss")
    if answer_mode in SUMMARY_EXPECTED_MODES and not summary_texts and not segment_texts:
        issues.append("summary_layer_miss")
    if (
        answer_mode not in {"general", "general_history"}
        and not evidence_present
        and not packed.recent_messages
    ):
        issues.append("no_evidence_no_recent")
    if mention_alias and not subject_bound:
        issues.append("subject_unbound")
    if answer_mode in {"current_fact", "assessment", "dated_history", "summary", "mention"}:
        if not overlap and not evidence_present:
            issues.append("no_relevant_evidence")
    if not issues and answer_mode in {"current_fact", "assessment"} and not fact_texts:
        issues.append("fact_layer_miss")
    return {
        "answer_mode": answer_mode,
        "coverage": getattr(resolved, "coverage_mode", ""),
        "subject_ids": list(subject_ids) if subject_ids else None,
        "subject_binding": getattr(resolved, "subject_binding", ""),
        "time_range": getattr(resolved, "time_range", None) is not None,
        "topic_terms": list(query_terms),
        "preferred_fact_kinds": list(getattr(resolved, "preferred_fact_kinds", ()) or ()),
        "rewrite_used": bool(getattr(resolved, "rewrite_used", False)),
        "recent_packed": len(packed.recent_messages),
        "evidence_messages": sum(len(segment.messages) for segment in packed.evidence_segments),
        "segments": len(packed.evidence_segments),
        "facts": len(packed.facts),
        "fact_kinds": fact_kinds,
        "summaries": len(packed.summaries),
        "selected_sources": len(selected),
        "grounding_policy": packed.grounding_policy,
        "overlap_terms": overlap,
        "subject_bound": subject_bound,
        "evidence_present": evidence_present,
        "issues": issues,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layered @bot mention evaluation.")
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--group-id", required=True, type=int)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
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
    columns = _message_columns(engine)
    if "mentioned_bot" not in columns:
        raise SystemExit("messages table has no mentioned_bot column")

    aliases = _in_scope_aliases(engine, int(args.group_id))
    targets = _mention_messages(engine, int(args.group_id), int(settings.bot_qq), args.limit)
    llm_client = build_llm_client(settings=settings, engine=engine)
    runtime = build_memory_runtime(
        settings=settings,
        engine=engine,
        llm_client=llm_client,
        bot_display_name="小町",
    )
    results: list[dict[str, Any]] = []
    for target in targets:
        query = str(target[4] or "").strip()
        if not query:
            continue
        outcome = _evaluate_case(
            runtime,
            settings,
            engine,
            {
                "group_id": int(args.group_id),
                "target": target,
                "query": query,
            },
            aliases,
        )
        results.append(
            {
                "message_id": str(target[1]),
                "user_id": int(target[2]),
                "timestamp": (
                    _parse_dt(target[3]).isoformat()
                    if _parse_dt(target[3]) is not None
                    else None
                ),
                "query": query,
                "reply_to": str(target[5]) if target[5] else None,
                **outcome,
            }
        )

    by_mode: dict[str, Counter] = defaultdict(Counter)
    issue_examples: dict[str, list[str]] = defaultdict(list)
    mode_counts: Counter = Counter()
    for item in results:
        mode = item.get("answer_mode", "unknown")
        mode_counts[mode] += 1
        for issue in item.get("issues", []):
            by_mode[mode][issue] += 1
            if len(issue_examples[issue]) < 5:
                issue_examples[issue].append(
                    f"[{mode}] {item['query'][:60]} (facts={item['facts']}, "
                    f"segments={item['segments']}, summaries={item['summaries']}, "
                    f"subject={item['subject_ids']})"
                )

    report = {
        "group_id": int(args.group_id),
        "limit": args.limit,
        "evaluated": len(results),
        "mode_counts": dict(mode_counts),
        "issue_counts_by_mode": {k: dict(v) for k, v in by_mode.items()},
        "total_issues": int(sum(sum(v.values()) for v in by_mode.values())),
        "issue_examples": issue_examples,
        "note": (
            "offline deterministic pipeline (rewrite disabled, 0.5s channel "
            "timeout); layered hit rates and issue taxonomy for @bot mentions"
        ),
    }
    if args.output:
        with args.output.open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    if args.report:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
