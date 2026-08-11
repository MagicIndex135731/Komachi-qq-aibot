import json
import sqlite3

from scripts.run_memory_test_suite import (
    stage_dataset,
    stage_prepare,
    stage_report,
)


def _minimal_db(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, group_id INTEGER, platform_msg_id TEXT,
            user_id INTEGER, timestamp TEXT, plain_text TEXT,
            reply_to_msg_id TEXT, mentioned_bot INTEGER
        );
        CREATE TABLE users (user_id INTEGER PRIMARY KEY, nickname TEXT, group_card TEXT);
        CREATE TABLE memory_items (
            id INTEGER PRIMARY KEY, scope_type TEXT, scope_id TEXT, subject_id TEXT,
            memory_kind TEXT, predicate TEXT, object_text TEXT, content TEXT,
            source_msg_id TEXT, source_msg_ids TEXT, status TEXT, expires_at TEXT
        );
        CREATE TABLE summaries (
            id INTEGER PRIMARY KEY, scope_type TEXT, scope_id TEXT,
            summary_level TEXT, start_at TEXT, end_at TEXT, content TEXT,
            source_start_msg_id TEXT, source_end_msg_id TEXT, status TEXT
        );
        INSERT INTO messages VALUES
          (1, 1001, 'p1', 11, '2026-08-01T10:00:00+08:00', '阿渣喜欢看动画', NULL, 0),
          (2, 1001, 'p2', 12, '2026-08-01T10:01:00+08:00', '小町 昨天说的计划', NULL, 1);
        INSERT INTO users VALUES (11, '阿渣', '阿渣'), (12, '逆蝶蝶', '逆蝶蝶');
        INSERT INTO memory_items VALUES
          (1, 'group', '1001', '11', 'preference', 'likes', '动画',
           '阿渣喜欢看动画', 'p1', '["p1"]', 'active', NULL);
        """
    )
    connection.commit()
    connection.close()


def test_prepare_and_dataset(tmp_path):
    database = tmp_path / "source.db"
    _minimal_db(database)
    workdir = tmp_path / "run"
    meta = stage_prepare(database, workdir)
    assert meta["integrity_check"] == "ok"
    assert (workdir / "snapshot.db").exists()
    result = stage_dataset(database, workdir, count=60, seed=1, group_ids=[])
    assert 0 < result["cases"] <= 60
    assert (workdir / "cases.jsonl").exists()


def test_report_aggregation(tmp_path):
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / "offline-results.jsonl").write_text(
        json.dumps(
            {
                "case_id": "c1",
                "category": "preference",
                "kind": "preference",
                "expected_layer": "fact",
                "expected_evidence_message_ids": ["a"],
                "packed_source_ids": ["a"],
                "subject_expected": ["11"],
                "subject_actual": ["11"],
                "raw_hit": False,
                "fact_hit": True,
                "summary_hit": False,
                "latency_ms": 5.0,
                "cross_group_violation": False,
                "query": "q",
            }
        ),
        encoding="utf-8",
    )
    (workdir / "fullchain-results.jsonl").write_text(
        json.dumps(
            {
                "case_id": "c1",
                "category": "preference",
                "kind": "preference",
                "expected_layer": "fact",
                "answer_grounded": True,
                "answer_correct": True,
                "abstained": False,
                "expected_abstention": False,
                "citation_precision": 1.0,
                "citation_recall": 1.0,
                "protocol_failure_codes": [],
                "input_tokens": 10,
                "output_tokens": 5,
                "ttft_ms": 1.0,
                "total_ms": 2.0,
                "cached": False,
            }
        ),
        encoding="utf-8",
    )
    report = stage_report(workdir, baseline_dir=None, gate_grounded=0.5, gate_recall=None, gate_protocol_failures=1, gate_p95_ms=100)
    assert report["offline"]["cases"] == 1
    assert report["fullchain"]["answer_accuracy"] == 1.0
    assert report["gate"]["grounded_answer_accuracy"]["passed"] is True
    assert (workdir / "report.json").exists()
    assert (workdir / "report.md").exists()
