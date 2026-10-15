"""Real-model memory semantic regression.

Runs the live LLM semantic-understanding channel over representative queries
against a read-only database snapshot and verifies positive recall and
negative correctness. No real group/user ids are embedded in this file; pass
them at evaluation time.

Usage:
  python scripts/run_memory_semantic_regression.py \
    --database /path/to/snapshot.db \
    --group-id 10001 --user-id 20001 --member-user-id 20002 \
    --member-alias 阿渣 \
    --output report.json
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Sequence

from app.config import AppSettings
from app.core.legacy_memory_context import GroupMemoryContextRequest
from app.main import build_llm_client, build_memory_runtime
from app.storage.db import build_engine


CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "positive_viewing_first_person",
        "query": "我最近在看什么动画",
        "kind": "positive",
        "expect_rewrite": True,
        "expect_modes": {"current_fact"},
        "expect_subject": "requester",
        "expect_kinds_any": ("current",),
        "expect_facts": True,
    },
    {
        "name": "positive_preference_first_person",
        "query": "我喜欢喝什么",
        "kind": "positive",
        "expect_rewrite": None,
        "expect_modes": {"current_fact"},
        "expect_subject": "requester",
        "expect_kinds_any": ("preference", "current"),
        "expect_facts": True,
    },
    {
        "name": "positive_member_viewing",
        "query": "{alias}最近在看什么动画",
        "kind": "positive",
        "expect_rewrite": None,
        "expect_modes": {"current_fact"},
        "expect_subject": "member",
        "expect_kinds_any": ("current",),
        "expect_facts": True,
        "needs_alias": True,
    },
    {
        "name": "negative_chitchat",
        "query": "在吗",
        "kind": "negative",
        "expect_subject": "unbound",
        "max_facts": 12,
    },
    {
        "name": "negative_weather",
        "query": "今天天气怎么样",
        "kind": "negative",
        "expect_subject": "unbound",
        "max_facts": 12,
    },
    {
        "name": "negative_mood",
        "query": "真恶心",
        "kind": "negative",
        "expect_subject": "unbound",
        "max_facts": 12,
    },
)


def _build_request(
    *,
    group_id: int,
    query: str,
    user_id: int,
) -> GroupMemoryContextRequest:
    return GroupMemoryContextRequest(
        group_id=group_id,
        query=query,
        recent_messages=(),
        quoted_message=None,
        target_message_id=f"semantic-regression-{query[:20]}",
        available_input=12000,
        now=datetime.now(UTC),
        current_user_id=user_id,
        use_full_history=False,
    )


def _subject_ok(
    *,
    expectation: str | None,
    subject_ids: Sequence[str] | None,
    requester_id: str | None,
    member_user_id: str | None,
) -> bool:
    if expectation is None:
        return True
    if expectation == "unbound":
        return subject_ids in (None, ())
    if expectation == "requester":
        return requester_id is not None and subject_ids == (requester_id,)
    if expectation == "member":
        return bool(subject_ids) and (
            member_user_id is None or subject_ids == (member_user_id,)
        )
    return False


def _evaluate_case(
    runtime,
    *,
    case: dict[str, Any],
    group_id: int,
    user_id: int,
    member_user_id: int | None,
    alias: str | None,
) -> dict[str, Any]:
    query = str(case["query"])
    if case.get("needs_alias"):
        query = query.format(alias=alias or "成员")
    trace = runtime.memory_orchestrator.v2_provider.evaluate(
        _build_request(
            group_id=group_id,
            query=query,
            user_id=user_id,
        )
    )
    resolved = trace.resolved_query
    packed = trace.result.packed_context
    failures: list[str] = []
    expect_rewrite = case.get("expect_rewrite")
    if expect_rewrite is True and not resolved.rewrite_used:
        failures.append("rewrite_used=False")
    expect_modes = case.get("expect_modes")
    if expect_modes and resolved.answer_mode not in expect_modes:
        failures.append(f"answer_mode={resolved.answer_mode}")
    if not _subject_ok(
        expectation=case.get("expect_subject"),
        subject_ids=resolved.subject_ids,
        requester_id=str(user_id),
        member_user_id=str(member_user_id) if member_user_id is not None else None,
    ):
        failures.append(f"subject={resolved.subject_ids}")
    if case.get("expect_kinds_any") and resolved.rewrite_used:
        if not set(case["expect_kinds_any"]) & set(resolved.preferred_fact_kinds):
            failures.append(f"fact_kinds={resolved.preferred_fact_kinds}")
    fact_count = len(packed.facts)
    if case.get("expect_facts") and fact_count == 0:
        failures.append("no facts packed")
    if case.get("max_facts") is not None and fact_count > case["max_facts"]:
        failures.append(f"facts={fact_count} > {case['max_facts']}")
    return {
        "name": case["name"],
        "kind": case["kind"],
        "query": query,
        "passed": not failures,
        "failures": failures,
        "rewrite_used": resolved.rewrite_used,
        "answer_mode": resolved.answer_mode,
        "subject_ids": list(resolved.subject_ids or ()),
        "fact_kinds": list(resolved.preferred_fact_kinds),
        "fact_count": fact_count,
        "retrieval_query": resolved.retrieval_query,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--group-id", required=True, type=int)
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--member-user-id", type=int, default=None)
    parser.add_argument("--member-alias", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
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
    results = [
        _evaluate_case(
            runtime,
            case=case,
            group_id=args.group_id,
            user_id=args.user_id,
            member_user_id=args.member_user_id,
            alias=args.member_alias,
        )
        for case in CASES
    ]
    report = {
        "total": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "cases": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
