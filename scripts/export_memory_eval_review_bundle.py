from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Sequence

from app.core.memory_backfill import (
    message_ledger_manifest_sha256,
    verify_message_ledger_manifest,
)

try:
    from .evaluate_memory_recall import load_evaluation_cases
except ImportError:
    from evaluate_memory_recall import load_evaluation_cases


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a local, content-bearing V3 dataset review bundle."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("snapshot manifest must be an object")
    if not verify_message_ledger_manifest(args.database, manifest).matches:
        raise ValueError("database does not match the snapshot manifest")
    manifest_sha256 = message_ledger_manifest_sha256(manifest)
    cases, dataset_sha256 = load_evaluation_cases(args.dataset)
    source_ids = tuple(
        dict.fromkeys(
            source_id
            for case in cases
            for source_id in (
                *case.expected_evidence_message_ids,
                *case.forbidden_evidence_message_ids,
                *case.recent_context_message_ids,
                *(
                    (case.quoted_context_message_id,)
                    if case.quoted_context_message_id
                    else ()
                ),
            )
        )
    )
    rows = _load_messages(args.database, source_ids)
    bundle = {
        "review_bundle_version": 1,
        "dataset_sha256": dataset_sha256,
        "snapshot_manifest_sha256": manifest_sha256,
        "case_count": len(cases),
        "contains_private_chat_content": True,
        "cases": [
            {
                "case_index": index,
                "category": case.category,
                "group_id": case.group_id,
                "query": case.query,
                "requester_uin": case.requester_uin,
                "expected_answer_mode": case.expected_answer_mode,
                "expected_coverage_strategy": case.expected_coverage_strategy,
                "time_range": list(case.time_range) if case.time_range else None,
                "expected_evidence": [
                    rows[source_id]
                    for source_id in case.expected_evidence_message_ids
                ],
                "forbidden_evidence": [
                    rows[source_id]
                    for source_id in case.forbidden_evidence_message_ids
                ],
                "recent_context": [
                    rows[source_id]
                    for source_id in case.recent_context_message_ids
                ],
                "quoted_context": (
                    rows[case.quoted_context_message_id]
                    if case.quoted_context_message_id
                    else None
                ),
                "gate_tags": list(case.gate_tags),
            }
            for index, case in enumerate(cases)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "dataset_sha256": dataset_sha256,
                "snapshot_manifest_sha256": manifest_sha256,
                "case_count": len(cases),
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _load_messages(
    database: Path,
    source_ids: Sequence[str],
) -> dict[str, dict]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows: dict[str, dict] = {}
        batch_size = 500
        for offset in range(0, len(source_ids), batch_size):
            batch = source_ids[offset : offset + batch_size]
            placeholders = ",".join("?" for _ in batch)
            for row in connection.execute(
                "SELECT platform_msg_id, group_id, user_id, timestamp, "
                "plain_text, raw_json FROM messages "
                f"WHERE platform_msg_id IN ({placeholders})",
                tuple(batch),
            ):
                payload = _raw_payload(row["raw_json"])
                source_id = str(row["platform_msg_id"])
                rows[source_id] = {
                    "source_message_id": source_id,
                    "group_id": int(row["group_id"]),
                    "user_id": str(row["user_id"]),
                    "timestamp": str(row["timestamp"]),
                    "delivery_state": str(
                        payload.get("delivery_state", "")
                    ).strip(),
                    "plain_text": str(row["plain_text"] or ""),
                }
        missing = [source_id for source_id in source_ids if source_id not in rows]
        if missing:
            raise ValueError("review bundle contains unresolved source message IDs")
        return rows
    finally:
        connection.close()


def _raw_payload(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
