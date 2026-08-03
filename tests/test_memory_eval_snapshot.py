from __future__ import annotations

from datetime import UTC, datetime
import json
import sqlite3
from types import SimpleNamespace

import pytest

from app.core.memory_query_resolver import MemoryQueryResolver
from scripts import build_memory_eval_dataset as builder
from scripts import export_memory_eval_review_bundle as review_exporter
from scripts.build_memory_eval_dataset import (
    _load_paraphrase_overrides,
    _load_safe_group_messages,
)
from scripts.evaluate_memory_recall import (
    EvaluationCase,
    load_evaluation_cases,
    validate_cases_within_snapshot,
)
from scripts.evaluate_memory_v3 import (
    V3_REQUIRED_GATE_TAGS,
    load_message_metadata,
    validate_v3_dataset_contract,
    validate_v3_dataset_sources,
)


def test_eval_candidates_are_bounded_by_manifest_group_watermarks(tmp_path) -> None:
    database = tmp_path / "bot.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE messages ("
        "id INTEGER PRIMARY KEY, platform_msg_id TEXT, group_id INTEGER, "
        "user_id INTEGER, timestamp TEXT, plain_text TEXT, "
        "reply_to_msg_id TEXT, raw_json TEXT)"
    )
    within_rows = [
        (
            index,
            f"within-{index}",
            100,
            f"safe message within watermark {index}",
            json.dumps({}),
        )
        for index in range(1, 81)
    ]
    connection.executemany(
        "INSERT INTO messages VALUES (?, ?, ?, 1, '2026-07-23', ?, NULL, ?)",
        within_rows
        + [
            (81, "above", 100, "new message above watermark", json.dumps({})),
            (82, "other", 200, "message from another group", json.dumps({})),
        ],
    )
    connection.commit()
    connection.close()

    rows = _load_safe_group_messages(database, group_watermarks={100: 80})

    assert len(rows) == 80
    assert rows[-1]["platform_msg_id"] == "within-80"
    assert all(row["group_id"] == 100 for row in rows)

    valid_case = EvaluationCase(
        group_id=100,
        query="snapshot query",
        recent_context_message_ids=("within-79",),
        expected_evidence_message_ids=("within-80",),
        category="exact",
    )
    validate_cases_within_snapshot(
        database,
        cases=(valid_case,),
        group_watermarks={100: 80},
    )
    quoted_case = EvaluationCase(
        group_id=100,
        query="quoted query",
        recent_context_message_ids=("within-79",),
        expected_evidence_message_ids=("within-80",),
        category="vague_reference",
        quoted_context_message_id="within-78",
    )
    validate_cases_within_snapshot(
        database,
        cases=(quoted_case,),
        group_watermarks={100: 80},
    )
    outside_case = EvaluationCase(
        group_id=100,
        query="outside query",
        recent_context_message_ids=("within-79",),
        expected_evidence_message_ids=("above",),
        category="exact",
    )
    with pytest.raises(ValueError, match="outside snapshot"):
        validate_cases_within_snapshot(
            database,
            cases=(outside_case,),
            group_watermarks={100: 80},
        )


def test_paraphrase_overrides_require_strict_bound_human_approval(tmp_path) -> None:
    path = tmp_path / "paraphrase-overrides.json"
    path.write_text('{"schema_version":1,"snapshot_manifest_sha256":NaN,"items":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        _load_paraphrase_overrides(path, snapshot_manifest_sha256="a" * 64)

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_manifest_sha256": "b" * 64,
                "items": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="snapshot manifest mismatch"):
        _load_paraphrase_overrides(path, snapshot_manifest_sha256="a" * 64)


def test_builder_emits_v3_contracts_and_unapproved_real_snapshot_review(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "bot.db"
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE TABLE messages ("
            "id INTEGER PRIMARY KEY, platform_msg_id TEXT, group_id INTEGER, "
            "user_id INTEGER, timestamp TEXT, plain_text TEXT, "
            "reply_to_msg_id TEXT, raw_json TEXT)"
        )
        rows = []
        for index in range(1, 151):
            day = "2026-07-20" if index <= 90 else "2026-07-22"
            hour = "14" if 30 <= index <= 90 else "00"
            user_id = 1 if index <= 120 else 2
            reply_to = f"safe-{index - 1}" if index % 2 == 0 and index <= 70 else None
            delivery_state = "blocked" if index == 145 else ""
            rows.append(
                (
                    index,
                    f"safe-{index}",
                    100,
                    user_id,
                    f"{day}T{hour}:{index % 60:02d}:00+00:00",
                    f"detailed source message {index} with unique evaluation context",
                    reply_to,
                    json.dumps({"delivery_state": delivery_state}),
                )
            )
        for index in range(151, 156):
            rows.append(
                (
                    index,
                    f"other-{index}",
                    200,
                    9,
                    "2026-07-20T01:00:00+00:00",
                    f"other group distractor {index} with enough text",
                    None,
                    json.dumps({}),
                )
            )
        connection.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        connection.commit()
    finally:
        connection.close()

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    dataset = tmp_path / "dataset.jsonl"
    review = tmp_path / "review.json"
    paraphrase_overrides = tmp_path / "paraphrase-overrides.json"
    paraphrase_overrides.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_manifest_sha256": "a" * 64,
                "items": [
                    {
                        "source_message_id": f"safe-{index}",
                        "query": f"curated equivalent evaluation query {index}",
                        "reviewer": "fixture",
                        "semantic_equivalence_approved": True,
                    }
                    for index in range(11, 21)
                ],
            }
        ),
        encoding="utf-8",
    )
    watermarks = {100: 150, 200: 155}
    monkeypatch.setattr(
        builder,
        "verify_message_ledger_manifest",
        lambda *_args: SimpleNamespace(matches=True),
    )
    monkeypatch.setattr(builder, "message_ledger_manifest_sha256", lambda _value: "a" * 64)
    monkeypatch.setattr(builder, "group_watermarks_from_manifest", lambda _value: watermarks)

    assert builder.main(
        [
            "--database",
            str(database),
            "--manifest",
            str(manifest),
            "--output",
            str(dataset),
            "--review-output",
            str(review),
            "--paraphrase-overrides",
            str(paraphrase_overrides),
        ]
    ) == 0

    cases, _ = load_evaluation_cases(dataset)
    tag_counts = validate_v3_dataset_contract(cases)
    validate_v3_dataset_sources(
        cases,
        metadata=load_message_metadata(database),
        snapshot_watermarks=watermarks,
    )
    metadata = load_message_metadata(database)
    assert all(
        case.requester_uin
        == metadata[case.recent_context_message_ids[-1]].user_id
        for case in cases
    )
    assert all(
        metadata[case.recent_context_message_ids[-1]].reply_to_message_id is None
        for case in cases
    )
    first_person = next(case for case in cases if "first_person" in case.gate_tags)
    assert (
        metadata[first_person.recent_context_message_ids[-1]].timestamp
        > metadata[first_person.expected_evidence_message_ids[0]].timestamp
    )
    assert V3_REQUIRED_GATE_TAGS <= set(tag_counts)
    temporal = next(case for case in cases if "time_range" in case.gate_tags)
    assert {"cross_group", "blocked_reserved", "time_range"} <= set(
        temporal.gate_tags
    )

    draft = json.loads(review.read_text(encoding="utf-8"))
    assert draft["dataset_sha256"]
    assert draft["snapshot_manifest_sha256"] == "a" * 64
    assert all(row["approved"] is False for row in draft["cases"])
    assert draft["structural_review"]["forbidden_source_count"] >= 4

    monkeypatch.setattr(
        review_exporter,
        "verify_message_ledger_manifest",
        lambda *_args: SimpleNamespace(matches=True),
    )
    monkeypatch.setattr(
        review_exporter,
        "message_ledger_manifest_sha256",
        lambda _value: "a" * 64,
    )
    bundle_path = tmp_path / "review-bundle.json"
    assert review_exporter.main(
        [
            "--database",
            str(database),
            "--manifest",
            str(manifest),
            "--dataset",
            str(dataset),
            "--output",
            str(bundle_path),
        ]
    ) == 0
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["dataset_sha256"] == draft["dataset_sha256"]
    assert bundle["snapshot_manifest_sha256"] == "a" * 64
    assert bundle["case_count"] == 64
    assert bundle["contains_private_chat_content"] is True

    resolver = MemoryQueryResolver()
    for case_index, case in enumerate(cases):
        target = metadata[case.recent_context_message_ids[-1]]
        quoted = (
            SimpleNamespace(
                message_id=case.quoted_context_message_id,
                speaker="quoted",
                content="quoted source",
                sent_at=target.timestamp,
                blocked=False,
                user_id=case.requester_uin,
                reply_to_msg_id=None,
                is_bot=False,
            )
            if case.quoted_context_message_id
            else None
        )
        plan = resolver.resolve(
            case.query,
            recent_messages=(),
            quoted_message=quoted,
            now=target.timestamp,
            group_id=case.group_id,
            requester_id=case.requester_uin,
        )
        assert plan.answer_mode == case.expected_answer_mode, case_index
        assert plan.coverage_strategy == case.expected_coverage_strategy, case_index
        assert plan.subject_ids == case.allowed_subject_user_ids, case_index
        actual_range = (
            None
            if plan.time_range is None
            else (
                plan.time_range.start.astimezone(UTC),
                plan.time_range.end.astimezone(UTC),
            )
        )
        expected_range = (
            None
            if case.time_range is None
            else tuple(
                datetime.fromisoformat(value).astimezone(UTC)
                for value in case.time_range
            )
        )
        assert actual_range == expected_range, case_index
