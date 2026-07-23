from __future__ import annotations

import json
import sqlite3

import pytest

from scripts.build_memory_eval_dataset import _load_safe_group_messages
from scripts.evaluate_memory_recall import (
    EvaluationCase,
    validate_cases_within_snapshot,
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
