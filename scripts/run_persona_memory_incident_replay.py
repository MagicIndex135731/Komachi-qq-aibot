"""Run the versioned, anonymized persona-memory incident contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo

from app.core.memory_fact_ranking import fact_intent_policy
from app.core.memory_clarification_threads import resolve_clarification_threads
from app.core.memory_context_packer import EvidenceMessage
from app.core.memory_query_resolver import MemoryQueryResolver
from app.core.member_identity import GroupMemberIdentity


DEFAULT_FIXTURE = Path("tests/fixtures/persona_memory_incidents.json")


_FORBIDDEN_ANSWER_PATTERNS: dict[str, re.Pattern[str]] = {
    "numeric_probability": re.compile(r"(?:概率.{0,8})?\d+(?:\.\d+)?\s*%"),
    "final_offer": re.compile(
        r"(?:已经|确认|确定)(?:拿到|获得|收到)(?:了)?.{0,6}offer|"
        r"offer.{0,4}(?:已经|确认|确定)(?:拿到|获得|收到)",
        re.I,
    ),
    "employment_outcome": re.compile(
        r"(?:已经|确认将|确定要)(?:入职|加入|去(?:了|往))"
    ),
}


def run_incident_contract(fixture_path: Path) -> dict[str, Any]:
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version") or 0) != 1:
        raise ValueError("unsupported incident fixture schema")
    ids = payload["synthetic_members"]
    persona_id = int(ids["persona"])
    requester_id = int(ids["requester"])
    bot_id = int(ids["bot"])
    members = (
        GroupMemberIdentity(user_id=persona_id, nickname="示例人格", in_scope=True),
        GroupMemberIdentity(user_id=requester_id, nickname="示例请求者", in_scope=True),
        GroupMemberIdentity(user_id=int(ids["other"]), nickname="示例成员", in_scope=True),
        GroupMemberIdentity(user_id=bot_id, nickname="示例机器人", in_scope=False),
    )
    resolver = MemoryQueryResolver()
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    now = datetime(2026, 9, 27, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    for case in payload.get("cases") or []:
        case_id = str(case["id"])
        resolved = resolver.resolve(
            str(case["query"]),
            recent_messages=(),
            now=now,
            group_members=members,
            excluded_member_ids={bot_id},
            group_id=900000001,
            requester_id=requester_id,
            impersonated_subject_id=persona_id,
            impersonated_subject_addressed=bool(case.get("addressed")),
            addressed_bot_user_id=bot_id,
        )
        policy = fact_intent_policy(
            query=resolved.original_query,
            answer_mode=resolved.answer_mode,
        )
        case_failures: list[str] = []
        if resolved.subject_binding != case["expected_binding"]:
            case_failures.append(
                f"binding={resolved.subject_binding!r} expected={case['expected_binding']!r}"
            )
        if "expected_subject_ids" in case:
            raw_expected_subjects = case.get("expected_subject_ids")
            expected_subjects = (
                None
                if raw_expected_subjects is None
                else tuple(str(value) for value in raw_expected_subjects)
            )
            if resolved.subject_ids != expected_subjects:
                case_failures.append(
                    f"subject_ids={resolved.subject_ids!r} expected={expected_subjects!r}"
                )
        expected_role = case.get("expected_subject_role")
        if expected_role and resolved.subject_role != expected_role:
            case_failures.append(
                f"subject_role={resolved.subject_role!r} expected={expected_role!r}"
            )
        expected_mode = case.get("expected_answer_mode")
        if expected_mode and resolved.answer_mode != expected_mode:
            case_failures.append(
                f"answer_mode={resolved.answer_mode!r} expected={expected_mode!r}"
            )
        topic = str(resolved.topic_query or "")
        for term in case.get("topic_contains") or []:
            if str(term) not in topic:
                case_failures.append(f"topic_missing={term!r}")
        expected_coverage = case.get("expected_fact_coverage")
        if expected_coverage and policy.coverage != expected_coverage:
            case_failures.append(
                f"coverage={policy.coverage!r} expected={expected_coverage!r}"
            )
        expected_kinds = tuple(case.get("expected_allowed_kinds") or ())
        if expected_kinds and policy.allowed_kinds != expected_kinds:
            case_failures.append(
                f"allowed_kinds={policy.allowed_kinds!r} expected={expected_kinds!r}"
            )
        must_include = tuple(str(value) for value in case.get("must_include_source_ids") or ())
        must_exclude = tuple(str(value) for value in case.get("must_exclude_source_ids") or ())
        if set(must_include).intersection(must_exclude):
            case_failures.append("source_contract_overlap")
        failures.extend(f"{case_id}: {value}" for value in case_failures)
        results.append(
            {
                "id": case_id,
                "case_type": "query",
                "passed": not case_failures,
                "subject_binding": resolved.subject_binding,
                "subject_reason": resolved.subject_decision_reason,
                "answer_mode": resolved.answer_mode,
                "topic_term_count": len(resolved.topic_terms),
                "fact_coverage": policy.coverage,
                "allowed_fact_kinds": list(policy.allowed_kinds),
                "must_include_source_ids": list(must_include),
                "must_exclude_source_ids": list(must_exclude),
                "answer_must_not_claim": list(case.get("answer_must_not_claim") or ()),
            }
        )

    clarification_results: list[dict[str, Any]] = []
    for case in payload.get("clarification_cases") or []:
        case_id = str(case["id"])
        messages = tuple(
            EvidenceMessage(
                source_msg_id=str(item["source_msg_id"]),
                speaker=str(item["user_id"]),
                content=str(item["content"]),
                sent_at=now + timedelta(seconds=int(item.get("offset_seconds") or 0)),
                blocked=bool(item.get("blocked")),
                group_id=900000001,
                reply_to_msg_id=item.get("reply_to_msg_id"),
                is_bot=str(item["user_id"]) == str(bot_id),
                user_id=str(item["user_id"]),
                delivery_state=str(item.get("delivery_state") or ""),
            )
            for item in case.get("messages") or ()
        )
        threads = resolve_clarification_threads(
            messages,
            anchor_source_ids=tuple(case.get("anchor_source_ids") or ()),
        )
        atomic_groups = [list(thread.source_msg_ids) for thread in threads]
        expected_groups = [list(group) for group in case.get("expected_atomic_groups") or ()]
        case_failures = []
        if atomic_groups != expected_groups:
            case_failures.append(
                f"atomic_groups={atomic_groups!r} expected={expected_groups!r}"
            )
        failures.extend(f"{case_id}: {value}" for value in case_failures)
        clarification_results.append(
            {
                "id": case_id,
                "case_type": "clarification",
                "passed": not case_failures,
                "atomic_source_groups": atomic_groups,
                "clarification_link_count": len(threads),
            }
        )
    results.extend(clarification_results)
    return {
        "schema_version": 1,
        "fixture": fixture_path.as_posix(),
        "passed": not failures,
        "case_count": len(results),
        "query_case_count": len(payload.get("cases") or ()),
        "clarification_case_count": len(clarification_results),
        "failures": failures,
        "results": results,
    }


def validate_fullchain_report(
    fixture_path: Path,
    fullchain_report_path: Path,
) -> dict[str, Any]:
    """Validate an actual production-chain replay against logical fixture IDs.

    The replay producer owns database access, retrieval, packing, and provider
    generation.  This validator maps local real source IDs to the committed
    anonymized labels and fails closed on missing evidence or unsupported final
    claims; it never treats fixture declarations themselves as observed hits.
    """

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    report = json.loads(fullchain_report_path.read_text(encoding="utf-8"))
    if int(report.get("schema_version") or 0) != 1:
        raise ValueError("unsupported fullchain report schema")
    source_id_map = {
        str(key): str(value)
        for key, value in (report.get("source_id_map") or {}).items()
        if str(key) and str(value)
    }
    actual_by_id = {
        str(item.get("id")): item
        for item in (report.get("cases") or ())
        if isinstance(item, dict) and str(item.get("id") or "")
    }
    failures: list[str] = []
    results: list[dict[str, Any]] = []
    for expected in fixture.get("cases") or ():
        case_id = str(expected["id"])
        if not (
            expected.get("must_include_source_ids")
            or expected.get("must_exclude_source_ids")
            or expected.get("answer_must_not_claim")
        ):
            continue
        actual = actual_by_id.get(case_id)
        case_failures: list[str] = []
        if actual is None:
            case_failures.append("missing_fullchain_case")
        else:
            selected = {
                str(value)
                for value in (actual.get("selected_source_ids") or ())
                if str(value)
            }
            must_include = {
                source_id_map.get(str(value), str(value))
                for value in (expected.get("must_include_source_ids") or ())
            }
            must_exclude = {
                source_id_map.get(str(value), str(value))
                for value in (expected.get("must_exclude_source_ids") or ())
            }
            missing = sorted(must_include - selected)
            leaked = sorted(must_exclude & selected)
            if missing:
                case_failures.append(f"missing_sources={missing!r}")
            if leaked:
                case_failures.append(f"excluded_sources={leaked!r}")
            if actual.get("subject_binding") != expected.get("expected_binding"):
                case_failures.append("subject_binding_mismatch")
            expected_mode = expected.get("expected_answer_mode")
            if expected_mode and actual.get("answer_mode") != expected_mode:
                case_failures.append("answer_mode_mismatch")
            actual_topics = " ".join(
                str(value) for value in (actual.get("topic_terms") or ())
            )
            for term in expected.get("topic_contains") or ():
                if str(term) not in actual_topics:
                    case_failures.append(f"topic_missing={term!r}")
            answer_text = str(actual.get("answer_text") or "").strip()
            if expected.get("answer_must_not_claim") and not answer_text:
                case_failures.append("missing_final_answer")
            for claim in expected.get("answer_must_not_claim") or ():
                pattern = _FORBIDDEN_ANSWER_PATTERNS.get(str(claim))
                if pattern is None:
                    case_failures.append(f"unknown_answer_claim={claim!r}")
                elif pattern.search(answer_text):
                    case_failures.append(f"forbidden_answer_claim={claim!r}")
        failures.extend(f"{case_id}: {value}" for value in case_failures)
        results.append(
            {
                "id": case_id,
                "passed": not case_failures,
                "failures": case_failures,
            }
        )
    return {
        "schema_version": 1,
        "fixture": fixture_path.as_posix(),
        "fullchain_report": fullchain_report_path.as_posix(),
        "passed": not failures,
        "case_count": len(results),
        "failures": failures,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--fullchain-report", type=Path)
    parser.add_argument("--require-fullchain", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_incident_contract(args.fixture)
    fullchain = None
    if args.fullchain_report is not None:
        fullchain = validate_fullchain_report(args.fixture, args.fullchain_report)
        report["fullchain"] = fullchain
        report["passed"] = bool(report["passed"] and fullchain["passed"])
    elif args.require_fullchain:
        report["passed"] = False
        report["failures"].append("fullchain report is required")
    report["fullchain_checked"] = fullchain is not None
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
