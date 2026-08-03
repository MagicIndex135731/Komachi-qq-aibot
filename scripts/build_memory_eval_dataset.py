from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Sequence

from app.core.memory_backfill import (
    message_ledger_manifest_sha256,
    verify_message_ledger_manifest,
)
from app.core.memory_backfill_runner import group_watermarks_from_manifest
from app.core.memory_query_resolver import MemoryQueryResolver

try:
    from .evaluate_memory_recall import validate_real_dataset, load_evaluation_cases
    from .evaluate_memory_v3 import (
        load_message_metadata,
        validate_v3_dataset_contract,
        validate_v3_dataset_sources,
    )
except ImportError:  # Direct script execution.
    from evaluate_memory_recall import validate_real_dataset, load_evaluation_cases
    from evaluate_memory_v3 import (
        load_message_metadata,
        validate_v3_dataset_contract,
        validate_v3_dataset_sources,
    )


CATEGORY_COUNTS = {
    "exact": 10,
    "paraphrase": 10,
    "vague_reference": 10,
    "temporal": 10,
    "multi_hop": 8,
    "update": 8,
    "abstention": 8,
}

_INELIGIBLE_DELIVERY_STATES = frozenset(
    {"reserved", "blocked", "uncertain", "deleted"}
)
_SHANGHAI = timezone(timedelta(hours=8))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a 64-case, real-history memory evaluation candidate set."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-output", required=True, type=Path)
    parser.add_argument("--paraphrase-overrides", required=True, type=Path)
    return parser


def _load_paraphrase_overrides(
    path: Path,
    *,
    snapshot_manifest_sha256: str,
) -> dict[str, str]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant is forbidden: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("paraphrase overrides schema_version must be 1")
    if payload.get("snapshot_manifest_sha256") != snapshot_manifest_sha256:
        raise ValueError("paraphrase overrides snapshot manifest mismatch")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("paraphrase overrides items must be a list")
    overrides: dict[str, str] = {}
    queries: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"paraphrase override {index} must be an object")
        source_id = item.get("source_message_id")
        query = item.get("query")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"paraphrase override {index} has invalid source ID")
        if not isinstance(query, str) or len(query.strip()) < 8:
            raise ValueError(f"paraphrase override {index} has invalid query")
        if item.get("semantic_equivalence_approved") is not True:
            raise ValueError(f"paraphrase override {index} lacks semantic approval")
        normalized_query = " ".join(query.split()).casefold()
        if source_id in overrides or normalized_query in queries:
            raise ValueError("paraphrase overrides contain duplicate source or query")
        overrides[source_id] = query.strip()
        queries.add(normalized_query)
    if len(overrides) < CATEGORY_COUNTS["paraphrase"]:
        raise ValueError("paraphrase overrides contain fewer than 10 approved items")
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not verify_message_ledger_manifest(args.database, manifest).matches:
        raise ValueError("database does not match the evaluation snapshot manifest")
    manifest_sha256 = message_ledger_manifest_sha256(manifest)
    rows = _load_safe_group_messages(
        args.database,
        group_watermarks=group_watermarks_from_manifest(manifest),
    )
    snapshot_rows = _load_snapshot_group_messages(
        args.database,
        group_watermarks=group_watermarks_from_manifest(manifest),
    )
    paraphrase_overrides = _load_paraphrase_overrides(
        args.paraphrase_overrides,
        snapshot_manifest_sha256=manifest_sha256,
    )
    cases, review = build_cases(
        rows,
        distractor_rows=snapshot_rows,
        paraphrase_overrides=paraphrase_overrides,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(
                json.dumps(
                    case,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    loaded, digest = load_evaluation_cases(args.output)
    validate_real_dataset(loaded)
    validate_v3_dataset_contract(loaded)
    validate_v3_dataset_sources(
        loaded,
        metadata=load_message_metadata(args.database),
        snapshot_watermarks=group_watermarks_from_manifest(manifest),
    )
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    args.review_output.write_text(
        json.dumps(
            {
                "review_version": 1,
                "dataset_sha256": digest,
                "snapshot_manifest_sha256": manifest_sha256,
                "case_count": len(cases),
                "category_counts": CATEGORY_COUNTS,
                "cases": [
                    {
                        "case_index": row["case_index"],
                        "group_id": row["group_id"],
                        "category": row["category"],
                        "expected_evidence_message_ids": row["expected_ids"],
                        "evidence_sha256": row["source_sha256"],
                        "approved": False,
                        "reviewer": "",
                        "reviewed_at": "",
                    }
                    for row in review["case_evidence"]
                ],
                "structural_review": review,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "dataset_sha256": digest,
                "snapshot_manifest_sha256": manifest_sha256,
                "case_count": len(cases),
                "category_counts": CATEGORY_COUNTS,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _load_snapshot_group_messages(
    database: Path,
    *,
    group_watermarks: dict[int, int],
) -> list[dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = []
        for group_id, watermark in sorted(group_watermarks.items()):
            group_rows = connection.execute(
                "SELECT id, platform_msg_id, group_id, user_id, timestamp, "
                "plain_text, reply_to_msg_id, raw_json FROM messages "
                "WHERE group_id = ? AND id <= ? "
                "ORDER BY id",
                (int(group_id), int(watermark)),
            )
            for row in group_rows:
                raw = row["raw_json"]
                try:
                    payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except json.JSONDecodeError:
                    payload = {}
                rows.append(
                    {
                        "id": int(row["id"]),
                        "platform_msg_id": str(row["platform_msg_id"]),
                        "group_id": int(row["group_id"]),
                        "user_id": int(row["user_id"]),
                        "timestamp": str(row["timestamp"]),
                        "plain_text": str(row["plain_text"]).strip(),
                        "delivery_state": (
                            str(payload.get("delivery_state", ""))
                            .strip()
                            .casefold()
                            if isinstance(payload, dict)
                            else ""
                        ),
                        "reply_to_msg_id": (
                            str(row["reply_to_msg_id"])
                            if row["reply_to_msg_id"]
                            else None
                        ),
                    }
                )
        return rows
    finally:
        connection.close()


def _load_safe_group_messages(
    database: Path,
    *,
    group_watermarks: dict[int, int],
) -> list[dict[str, Any]]:
    rows = _load_snapshot_group_messages(
        database,
        group_watermarks=group_watermarks,
    )
    safe_rows = [
        row
        for row in rows
        if 8 <= len(row["plain_text"]) <= 400
        and row["delivery_state"] not in _INELIGIBLE_DELIVERY_STATES
    ]
    if len(safe_rows) < 80:
        raise ValueError("not enough safe real group messages for evaluation")
    return safe_rows


def build_cases(
    rows: list[dict[str, Any]],
    *,
    distractor_rows: list[dict[str, Any]] | None = None,
    paraphrase_overrides: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict]:
    by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_platform_id = {row["platform_msg_id"]: row for row in rows}
    for row in rows:
        by_group[row["group_id"]].append(row)
    positions = {
        (row["group_id"], row["platform_msg_id"]): index
        for group_id, group_rows in by_group.items()
        for index, row in enumerate(group_rows)
    }

    chosen: set[str] = set()
    cases: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    def recent(row: dict[str, Any], *, width: int = 6) -> list[str]:
        group_rows = by_group[row["group_id"]]
        index = positions[(row["group_id"], row["platform_msg_id"])]
        return [
            item["platform_msg_id"]
            for item in group_rows[max(0, index - width + 1) : index + 1]
        ]

    def following(row: dict[str, Any], *, width: int = 6) -> list[str]:
        group_rows = by_group[row["group_id"]]
        index = positions[(row["group_id"], row["platform_msg_id"])]
        later_rows = group_rows[index + 1 :]
        target_index = next(
            (
                offset
                for offset, item in enumerate(later_rows[width - 1 :], width - 1)
                if not item.get("reply_to_msg_id")
            ),
            next(
                (
                    offset
                    for offset, item in enumerate(later_rows)
                    if not item.get("reply_to_msg_id")
                ),
                None,
            ),
        )
        if target_index is None:
            raise ValueError("evaluation evidence has no later non-reply context")
        return [
            item["platform_msg_id"]
            for item in later_rows[max(0, target_index - width + 1) : target_index + 1]
        ]

    resolver = MemoryQueryResolver()

    def has_general_unbound_plan(query: str, recent_ids: list[str]) -> bool:
        target = by_platform_id[recent_ids[-1]]
        plan = resolver.resolve(
            query,
            recent_messages=(),
            now=_parse_timestamp(target["timestamp"]),
            group_id=int(target["group_id"]),
            requester_id=str(target["user_id"]),
        )
        return (
            plan.subject_ids is None
            and plan.time_range is None
            and plan.answer_mode == "general_history"
            and plan.coverage_strategy == "relevance"
        )

    def add(
        *,
        category: str,
        row: dict[str, Any],
        query: str,
        expected: list[str],
        recent_ids: list[str] | None = None,
        time_range: dict[str, str] | None = None,
        quoted_context_message_id: str | None = None,
    ) -> None:
        case = {
            "group_id": row["group_id"],
            "query": query,
            "recent_context_message_ids": recent_ids or recent(row),
            "expected_evidence_message_ids": list(dict.fromkeys(expected)),
            "category": category,
        }
        if time_range is not None:
            case["time_range"] = time_range
        if quoted_context_message_id is not None:
            case["quoted_context_message_id"] = quoted_context_message_id
        cases.append(case)
        for source_id in expected:
            chosen.add(source_id)
        review_rows.append(
            {
                "case_index": len(cases) - 1,
                "category": category,
                "group_id": row["group_id"],
                "expected_ids": list(dict.fromkeys(expected)),
                "source_sha256": hashlib.sha256(
                    "\n".join(
                        by_platform_id[source_id]["plain_text"]
                        for source_id in expected
                    ).encode("utf-8")
                ).hexdigest(),
                "scope_verified": all(
                    by_platform_id[source_id]["group_id"] == row["group_id"]
                    for source_id in expected
                ),
            }
        )

    text_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        text_counts[" ".join(row["plain_text"].split()).casefold()] += 1
    candidates = []
    seen_candidate_texts: set[str] = set()
    for row in rows:
        normalized_text = " ".join(row["plain_text"].split()).casefold()
        group_rows = by_group[row["group_id"]]
        row_position = positions[(row["group_id"], row["platform_msg_id"])]
        following_texts = {
            " ".join(item["plain_text"].split()).casefold()
            for item in group_rows[row_position + 1 : row_position + 7]
        }
        if (
            len(_excerpt(row["plain_text"])) < 8
            or row_position >= len(group_rows) - 6
            or normalized_text in seen_candidate_texts
            or normalized_text in following_texts
            or text_counts[normalized_text] != 1
        ):
            continue
        seen_candidate_texts.add(normalized_text)
        candidates.append(row)
    cursor = 0
    rejected_exact: set[str] = set()
    reserved_paraphrase_sources = set(paraphrase_overrides or ())
    for _ in range(CATEGORY_COUNTS["exact"]):
        while True:
            row, cursor = _next_unused(
                candidates,
                chosen | rejected_exact | reserved_paraphrase_sources,
                cursor,
            )
            recent_ids = following(row)
            query = f"查找历史原话：“{_excerpt(row['plain_text'])}”"
            if has_general_unbound_plan(query, recent_ids):
                break
            rejected_exact.add(row["platform_msg_id"])
        excerpt = _excerpt(row["plain_text"])
        add(
            category="exact",
            row=row,
            query=query,
            expected=[row["platform_msg_id"]],
            recent_ids=recent_ids,
        )

    if paraphrase_overrides is None:
        for _ in range(CATEGORY_COUNTS["paraphrase"]):
            row, cursor = _next_unused(candidates, chosen, cursor)
            excerpt = _excerpt(row["plain_text"])
            add(
                category="paraphrase",
                row=row,
                query=f"查找历史中表达过这个意思的消息：{_paraphrase(excerpt)}",
                expected=[row["platform_msg_id"]],
                recent_ids=following(row),
            )
    else:
        curated_rows = [
            row
            for row in candidates
            if row["platform_msg_id"] in paraphrase_overrides
            and row["platform_msg_id"] not in chosen
            and has_general_unbound_plan(
                paraphrase_overrides[row["platform_msg_id"]],
                following(row),
            )
        ]
        if len(curated_rows) < CATEGORY_COUNTS["paraphrase"]:
            raise ValueError("not enough eligible curated paraphrase sources")
        for row in curated_rows[: CATEGORY_COUNTS["paraphrase"]]:
            add(
                category="paraphrase",
                row=row,
                query=paraphrase_overrides[row["platform_msg_id"]],
                expected=[row["platform_msg_id"]],
                recent_ids=following(row),
            )

    reply_rows = [
        row
        for row in rows
        if row["reply_to_msg_id"] in by_platform_id
        and by_platform_id[row["reply_to_msg_id"]]["group_id"] == row["group_id"]
    ]
    complete_reply_rows = [
        row
        for row in (distractor_rows if distractor_rows is not None else rows)
        if row.get("reply_to_msg_id")
        and row["delivery_state"] not in _INELIGIBLE_DELIVERY_STATES
    ]
    replies_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in complete_reply_rows:
        replies_by_parent[str(row["reply_to_msg_id"])].append(row)
    unique_direct_reply_ids = {
        children[0]["platform_msg_id"]
        for children in replies_by_parent.values()
        if len(children) == 1
    }
    single_reply_rows = [
        row for row in reply_rows if row["platform_msg_id"] in unique_direct_reply_ids
    ]
    for row in single_reply_rows[: CATEGORY_COUNTS["vague_reference"]]:
        quoted = by_platform_id[row["reply_to_msg_id"]]
        add(
            category="vague_reference",
            row=row,
            query="这条引用消息在群里的直接回复是什么？",
            expected=[row["platform_msg_id"]],
            recent_ids=following(row),
            quoted_context_message_id=quoted["platform_msg_id"],
        )
    if sum(case["category"] == "vague_reference" for case in cases) < 10:
        raise ValueError("not enough single-reply chains for vague-reference cases")

    rejected_summary_days: set[str] = set()
    for temporal_index in range(CATEGORY_COUNTS["temporal"]):
        while True:
            row, cursor = _next_unused(
                candidates,
                chosen | rejected_summary_days,
                cursor,
            )
            temporal_recent_ids = following(row)
            if temporal_index != 0 or _day_has_two_halfday_buckets(
                row,
                rows=rows,
                excluded_message_ids=set(temporal_recent_ids),
            ):
                break
            rejected_summary_days.add(row["platform_msg_id"])
        day = (
            _parse_timestamp(row["timestamp"])
            .astimezone(_SHANGHAI)
            .date()
            .isoformat()
        )
        start = f"{day}T00:00:00"
        end = f"{day}T23:59:59.999999"
        add(
            category="temporal",
            row=row,
            query=f"{day} 群里围绕“{_excerpt(row['plain_text'], 24)}”说过什么？",
            expected=[row["platform_msg_id"]],
            recent_ids=temporal_recent_ids,
            time_range={"start_at": start, "end_at": end},
        )

    reply_pairs = [
        (by_platform_id[row["reply_to_msg_id"]], row)
        for row in reply_rows
        if row["platform_msg_id"] in unique_direct_reply_ids
        and row["platform_msg_id"] not in chosen
        and row["reply_to_msg_id"] not in chosen
        and positions[(row["group_id"], row["platform_msg_id"])]
        < len(by_group[row["group_id"]]) - 6
    ]
    selected_multi_hop = 0
    for first, second in reply_pairs:
        recent_ids = following(second)
        query = (
            f"“{_excerpt(first['plain_text'], 18)}”这条消息和它的直接回复"
            "分别说了什么？"
        )
        if not has_general_unbound_plan(query, recent_ids):
            continue
        add(
            category="multi_hop",
            row=second,
            query=query,
            expected=[first["platform_msg_id"], second["platform_msg_id"]],
            recent_ids=recent_ids,
        )
        selected_multi_hop += 1
        if selected_multi_hop >= CATEGORY_COUNTS["multi_hop"]:
            break

    by_group_user: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group_user[(row["group_id"], row["user_id"])].append(row)
    complete_by_group_user: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in (distractor_rows if distractor_rows is not None else rows):
        if row["delivery_state"] in _INELIGIBLE_DELIVERY_STATES:
            continue
        complete_by_group_user[(row["group_id"], row["user_id"])].append(row)
    complete_adjacent_pairs = {
        (first["platform_msg_id"], second["platform_msg_id"])
        for values in complete_by_group_user.values()
        for first, second in zip(values, values[1:])
    }
    update_pairs = []
    for values in by_group_user.values():
        for first, second in zip(values, values[1:]):
            group_rows = by_group[second["group_id"]]
            second_position = positions[
                (second["group_id"], second["platform_msg_id"])
            ]
            if (
                second_position < len(group_rows) - 6
                and not first["platform_msg_id"].startswith("bot-reply-")
                and not second["platform_msg_id"].startswith("bot-reply-")
                and first["platform_msg_id"] not in chosen
                and second["platform_msg_id"] not in chosen
                and (
                    first["platform_msg_id"],
                    second["platform_msg_id"],
                ) in complete_adjacent_pairs
                and _is_related_update(first, second)
            ):
                update_pairs.append((first, second))
    update_pairs.sort(key=lambda pair: (pair[1]["id"], pair[0]["id"]))
    for first, second in update_pairs[: CATEGORY_COUNTS["update"]]:
        add(
            category="update",
            row=second,
            query=(
                f"梳理前后信息：“{_excerpt(first['plain_text'], 18)}”与"
                f"“{_excerpt(second['plain_text'], 18)}”之间有什么变化？"
            ),
            expected=[first["platform_msg_id"], second["platform_msg_id"]],
            recent_ids=following(second),
        )

    if sum(case["category"] == "multi_hop" for case in cases) < 8:
        raise ValueError("not enough safe adjacent pairs for multi-hop cases")
    if sum(case["category"] == "update" for case in cases) < 8:
        raise ValueError("not enough same-member pairs for update cases")

    for index in range(CATEGORY_COUNTS["abstention"]):
        row = candidates[(cursor + index) % len(candidates)]
        nonce = hashlib.sha256(f"{row['id']}:{index}".encode()).hexdigest()[:16]
        add(
            category="abstention",
            row=row,
            query=f"群里有人讨论过不存在的代号 ZX-{nonce} 吗？",
            expected=[],
            recent_ids=following(row),
        )

    _attach_v3_contracts(
        cases,
        source_rows=distractor_rows if distractor_rows is not None else rows,
    )
    evidence_rows = distractor_rows if distractor_rows is not None else rows
    evidence_by_platform_id = {
        row["platform_msg_id"]: row
        for row in evidence_rows
    }
    if len(cases) != 64:
        raise RuntimeError("evaluation builder did not create exactly 64 cases")
    gate_tag_counts: dict[str, int] = defaultdict(int)
    forbidden_ids = set()
    for case in cases:
        for tag in case["gate_tags"]:
            gate_tag_counts[tag] += 1
        forbidden_ids.update(case["forbidden_evidence_message_ids"])
    return cases, {
        "expected_ids_exist": all(
            source_id in by_platform_id
            for case in cases
            for source_id in case["expected_evidence_message_ids"]
        ),
        "all_expected_ids_group_scoped": all(
            evidence_by_platform_id[source_id]["group_id"] == case["group_id"]
            for case in cases
            for source_id in case["expected_evidence_message_ids"]
        ),
        "evaluation_schema_version": 3,
        "gate_tag_counts": dict(sorted(gate_tag_counts.items())),
        "forbidden_source_count": len(forbidden_ids),
        "case_evidence": [
            {
                "case_index": index,
                "category": case["category"],
                "group_id": case["group_id"],
                "expected_ids": list(case["expected_evidence_message_ids"]),
                "source_sha256": hashlib.sha256(
                    "\n".join(
                        evidence_by_platform_id[source_id]["plain_text"]
                        for source_id in case["expected_evidence_message_ids"]
                    ).encode("utf-8")
                ).hexdigest(),
                "scope_verified": all(
                    evidence_by_platform_id[source_id]["group_id"] == case["group_id"]
                    for source_id in case["expected_evidence_message_ids"]
                ),
            }
            for index, case in enumerate(cases)
        ],
    }


def _attach_v3_contracts(
    cases: list[dict[str, Any]],
    *,
    source_rows: list[dict[str, Any]],
) -> None:
    """Freeze V3 query scopes and derive real provenance-only distractors."""

    by_id = {row["platform_msg_id"]: row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise ValueError("snapshot ledger has duplicate platform message IDs")
    for case in cases:
        anchor_id = next(
            iter(case["expected_evidence_message_ids"]),
            case["recent_context_message_ids"][0],
        )
        anchor = by_id.get(anchor_id)
        if anchor is None:
            raise ValueError("evaluation case anchor is absent from snapshot ledger")
        target = by_id.get(case["recent_context_message_ids"][-1])
        if target is None:
            raise ValueError("evaluation requester target is absent from snapshot ledger")
        requester_uin = str(target["user_id"])
        category = case["category"]
        time_range = _shanghai_day_range(anchor["timestamp"]) if category == "temporal" else None
        case.update(
            {
                "schema_version": 3,
                "requester_uin": requester_uin,
                "allowed_subject_user_ids": None,
                "allowed_evidence_user_ids": None,
                "time_range": time_range,
                "expected_answer_mode": (
                    "exact"
                    if category == "vague_reference"
                    else "dated_history"
                    if category == "temporal" else "general_history"
                ),
                "expected_coverage_strategy": "chronological"
                if category == "temporal"
                else "relevance",
                "minimum_time_bucket_count": 0,
                "forbidden_evidence_message_ids": [],
                "gate_tags": [],
            }
        )

    first_exact = next(case for case in cases if case["category"] == "exact")
    exact_source_id = first_exact["expected_evidence_message_ids"][0]
    exact_source = by_id[exact_source_id]
    requester_target = next(
        (
            row
            for row in source_rows
            if row["platform_msg_id"] != exact_source_id
            and int(row["group_id"]) == int(exact_source["group_id"])
            and int(row["user_id"]) == int(exact_source["user_id"])
            and not row.get("reply_to_msg_id")
            and _parse_timestamp(row["timestamp"])
            > _parse_timestamp(exact_source["timestamp"])
            and row["delivery_state"] not in _INELIGIBLE_DELIVERY_STATES
        ),
        None,
    )
    if requester_target is None:
        raise ValueError("snapshot lacks a same-author requester target")
    first_exact["recent_context_message_ids"] = [
        requester_target["platform_msg_id"]
    ]
    first_exact["requester_uin"] = str(requester_target["user_id"])
    first_exact["query"] = f"我在群里说过\u201c{_excerpt(by_id[first_exact['expected_evidence_message_ids'][0]]['plain_text'])}\u201d吗？"
    first_exact["allowed_subject_user_ids"] = [first_exact["requester_uin"]]
    first_exact["allowed_evidence_user_ids"] = [first_exact["requester_uin"]]
    first_exact["gate_tags"].extend(("first_person", "source_resolution", "subject"))
    first_exact["forbidden_evidence_message_ids"] = _real_v3_distractors(
        first_exact,
        source_rows=source_rows,
    )

    first_abstention = next(
        case for case in cases if case["category"] == "abstention"
    )
    abstention_nonce = re.search(r"ZX-[0-9a-f]+", first_abstention["query"])
    if abstention_nonce is None:
        raise ValueError("mention abstention case has no frozen nonce")
    first_abstention["query"] = (
        f"关于不存在的代号 {abstention_nonce.group(0)}，他们最近有没有@我？"
    )
    first_abstention["expected_answer_mode"] = "mention"
    first_abstention["allowed_subject_user_ids"] = [
        first_abstention["requester_uin"]
    ]
    first_abstention["gate_tags"].append("mention")

    first_temporal = next(case for case in cases if case["category"] == "temporal")
    temporal_source = by_id[first_temporal["expected_evidence_message_ids"][0]]
    local_day = (
        _parse_timestamp(temporal_source["timestamp"])
        .astimezone(_SHANGHAI)
        .date()
        .isoformat()
    )
    first_temporal["query"] = (
        f"总结{local_day}上半天和下半天各一条有内容的聊天记录"
    )
    first_temporal["expected_answer_mode"] = "summary"
    first_temporal["expected_coverage_strategy"] = "time_buckets"
    first_temporal["minimum_time_bucket_count"] = 2
    temporal_range = first_temporal["time_range"]
    if temporal_range is None:
        raise ValueError("summary time-bucket case has no frozen range")
    temporal_start, temporal_end = (
        _parse_timestamp(value) for value in temporal_range.values()
    )
    temporal_expected = [
        row
        for row in source_rows
        if int(row["group_id"]) == int(first_temporal["group_id"])
        and row["delivery_state"] not in _INELIGIBLE_DELIVERY_STATES
        and 12 <= len(str(row["plain_text"]).strip()) <= 400
        and row["platform_msg_id"]
        not in set(first_temporal["recent_context_message_ids"])
        and temporal_start <= _parse_timestamp(row["timestamp"]) < temporal_end
    ]
    midpoint = temporal_start + (temporal_end - temporal_start) / 2
    early_row = max(
        (
            row
            for row in temporal_expected
            if _parse_timestamp(row["timestamp"]) < midpoint
        ),
        key=lambda row: len(str(row["plain_text"]).strip()),
        default=None,
    )
    late_row = max(
        (
            row
            for row in temporal_expected
            if _parse_timestamp(row["timestamp"]) >= midpoint
        ),
        key=lambda row: len(str(row["plain_text"]).strip()),
        default=None,
    )
    early = early_row["platform_msg_id"] if early_row is not None else None
    late = late_row["platform_msg_id"] if late_row is not None else None
    if early is None or late is None:
        raise ValueError("summary time-bucket case lacks two real time buckets")
    first_temporal["expected_evidence_message_ids"] = [early, late]
    first_temporal["gate_tags"].extend(
        ("time_range", "time_bucket", "cross_group", "blocked_reserved")
    )
    first_temporal["forbidden_evidence_message_ids"] = _real_v3_distractors(
        first_temporal,
        source_rows=source_rows,
    )

    for case in cases:
        if case["category"] == "abstention":
            case["gate_tags"].append("abstention")


def _real_v3_distractors(
    case: dict[str, Any],
    *,
    source_rows: list[dict[str, Any]],
) -> list[str]:
    group_id = int(case["group_id"])
    tags = set(case["gate_tags"])
    allowed_users = set(case["allowed_evidence_user_ids"] or ())
    time_range = case["time_range"]
    if time_range is not None:
        start_at, end_at = (_parse_timestamp(value) for value in time_range.values())

    def choose(label: str, predicate) -> str:
        for row in source_rows:
            if predicate(row):
                return row["platform_msg_id"]
        raise ValueError(f"snapshot lacks a real {label} V3 distractor")

    selected: list[str] = []
    if "cross_group" in tags:
        selected.append(
            choose("cross-group", lambda row: int(row["group_id"]) != group_id)
        )
    if "blocked_reserved" in tags:
        selected.append(
            choose(
                "blocked/reserved",
                lambda row: row["delivery_state"] in _INELIGIBLE_DELIVERY_STATES,
            )
        )
    if "subject" in tags:
        selected.append(
            choose(
                "wrong-subject",
                lambda row: int(row["group_id"]) == group_id
                and str(row["user_id"]) not in allowed_users,
            )
        )
    if "time_range" in tags:
        if time_range is None:
            raise ValueError("time-range gate has no frozen range")
        selected.append(
            choose(
                "out-of-range",
                lambda row: int(row["group_id"]) == group_id
                and not start_at <= _parse_timestamp(row["timestamp"]) < end_at,
            )
        )
    return list(dict.fromkeys(selected))


def _shanghai_day_range(timestamp: str) -> dict[str, str]:
    local = _parse_timestamp(timestamp).astimezone(_SHANGHAI)
    start_at = local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_at = start_at + timedelta(days=1)
    return {
        "start_at": start_at.isoformat(),
        "end_at": end_at.isoformat(),
    }


def _day_has_two_halfday_buckets(
    anchor: dict[str, Any],
    *,
    rows: Sequence[dict[str, Any]],
    excluded_message_ids: set[str] | frozenset[str] = frozenset(),
) -> bool:
    time_range = _shanghai_day_range(anchor["timestamp"])
    start_at, end_at = (_parse_timestamp(value) for value in time_range.values())
    midpoint = start_at + (end_at - start_at) / 2
    matching = [
        _parse_timestamp(row["timestamp"])
        for row in rows
        if int(row["group_id"]) == int(anchor["group_id"])
        and row["platform_msg_id"] not in excluded_message_ids
        and row["delivery_state"] not in _INELIGIBLE_DELIVERY_STATES
        and 12 <= len(str(row["plain_text"]).strip()) <= 400
        and start_at <= _parse_timestamp(row["timestamp"]) < end_at
    ]
    return any(value < midpoint for value in matching) and any(
        value >= midpoint for value in matching
    )


def _parse_timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _is_related_update(first: dict[str, Any], second: dict[str, Any]) -> bool:
    try:
        gap_seconds = (
            datetime.fromisoformat(second["timestamp"])
            - datetime.fromisoformat(first["timestamp"])
        ).total_seconds()
    except (TypeError, ValueError):
        return False
    if gap_seconds < 0 or gap_seconds > 30 * 60:
        return False
    if second.get("reply_to_msg_id") == first.get("platform_msg_id"):
        return True
    first_ngrams = _character_bigrams(first["plain_text"])
    second_ngrams = _character_bigrams(second["plain_text"])
    if not first_ngrams or not second_ngrams:
        return False
    overlap = len(first_ngrams & second_ngrams)
    similarity = overlap / min(len(first_ngrams), len(second_ngrams))
    return 0.12 <= similarity < 0.85


def _character_bigrams(value: str) -> set[str]:
    normalized = "".join(character for character in str(value) if character.isalnum())
    return {
        normalized[index : index + 2]
        for index in range(max(0, len(normalized) - 1))
    }


def _next_unused(candidates, chosen: set[str], cursor: int):
    for offset in range(len(candidates)):
        index = (cursor + offset) % len(candidates)
        row = candidates[index]
        if row["platform_msg_id"] not in chosen:
            return row, index + 1
    raise ValueError("not enough unique evaluation candidates")


def _excerpt(value: str, limit: int = 40) -> str:
    return " ".join(str(value).split())[:limit].strip("，。！？,.!? ")


def _paraphrase(value: str) -> str:
    rewritten = str(value)
    replacements = (
        ("因为", "由于"),
        ("但是", "不过"),
        ("觉得", "认为"),
        ("可以", "能够"),
        ("需要", "得"),
        ("已经", "早已"),
        ("不会", "并不会"),
        ("我", "这位成员"),
    )
    for source, target in replacements:
        rewritten = rewritten.replace(source, target)
    return rewritten


if __name__ == "__main__":
    raise SystemExit(main())
