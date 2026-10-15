import json
import sqlite3
from types import SimpleNamespace

import pytest

from scripts import run_memory_test_suite as suite
from scripts.run_memory_test_suite import (
    build_argument_parser,
    stage_fullchain,
    stage_dataset,
    stage_prepare,
    stage_report,
)


def test_suite_parser_separates_offline_and_fullchain_channel_timeouts():
    args = build_argument_parser().parse_args(["--database", "snapshot.db"])
    assert args.channel_timeout == 0.5
    assert args.fullchain_channel_timeout == 4.0
    assert args.offline_limit == 300
    assert args.judge_packet_mode == "full"
    assert args.identity_audit_start == ""
    assert args.identity_audit_end == ""


def test_suite_parser_accepts_identity_audit_window():
    args = build_argument_parser().parse_args(
        [
            "--database",
            "snapshot.db",
            "--identity-audit-start",
            "2026-08-24T18:00:00+08:00",
            "--identity-audit-end",
            "2026-08-25T00:00:00+08:00",
        ]
    )

    assert args.identity_audit_start == "2026-08-24T18:00:00+08:00"
    assert args.identity_audit_end == "2026-08-25T00:00:00+08:00"


def test_suite_main_routes_separate_channel_timeouts(monkeypatch, tmp_path):
    captured: dict[str, float] = {}

    def fake_offline(database, workdir, **kwargs):
        del database, workdir
        captured["offline"] = kwargs["channel_timeout"]
        return {}

    def fake_fullchain(database, workdir, **kwargs):
        del database, workdir
        captured["fullchain"] = kwargs["channel_timeout"]
        captured["judge_packet_mode"] = kwargs["judge_packet_mode"]
        return {}

    monkeypatch.setattr(suite, "stage_offline", fake_offline)
    monkeypatch.setattr(suite, "stage_fullchain", fake_fullchain)
    database = tmp_path / "snapshot.db"

    assert (
        suite.main(
            [
                "--database",
                str(database),
                "--workdir",
                str(tmp_path / "offline"),
                "--stage",
                "offline",
            ]
        )
        == 0
    )
    assert (
        suite.main(
            [
                "--database",
                str(database),
                "--workdir",
                str(tmp_path / "fullchain"),
                "--stage",
                "fullchain",
            ]
        )
        == 0
    )
    assert captured == {
        "offline": 0.5,
        "fullchain": 4.0,
        "judge_packet_mode": "full",
    }


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


def test_stage_fullchain_recovers_interrupted_detail_rows(tmp_path, monkeypatch):
    from scripts import memory_test_fullchain as fullchain_module

    database = tmp_path / "source.db"
    _minimal_db(database)
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / "cases.jsonl").write_text(
        json.dumps({"category": "fact", "query": "q", "case_id": "case-a"}) + "\n",
        encoding="utf-8",
    )
    detail = workdir / "fullchain-results.detail.jsonl"
    detail.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {
                    "case_id": "case-a",
                    "answer": "stale",
                    "case_input_signature": "0" * 64,
                    "resume_base_signature": "1" * 64,
                },
                {
                    "case_id": "case-a",
                    "answer": "ok",
                    "case_input_signature": "a" * 64,
                    "resume_base_signature": "b" * 64,
                    "protocol_failure_codes": [],
                },
                {
                    "case_id": "outside-selection",
                    "answer": "stale",
                    "case_input_signature": "a" * 64,
                    "resume_base_signature": "b" * 64,
                    "protocol_failure_codes": [],
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (workdir / "progress-fullchain.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"case_id": "case-a", "ok": True, "case_input_signature": "a" * 64},
                {
                    "case_id": "outside-selection",
                    "ok": True,
                    "case_input_signature": "a" * 64,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = workdir / "fullchain-results.jsonl"
    output.write_text(
        json.dumps({"case_id": "old", "answer": "stale"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    calls: dict[str, object] = {}

    def fake_run_cases(engine, cases, **kwargs):
        calls["resume"] = kwargs.get("resume")
        calls["detail_path"] = kwargs.get("detail_path")
        calls["prewarm_embedding"] = kwargs.get("prewarm_embedding")
        calls["judge_packet_mode"] = kwargs.get("judge_packet_mode")
        return [], {"requested": 1, "executed": 0, "skipped_resumed": 1}

    monkeypatch.setattr(fullchain_module, "run_cases", fake_run_cases)
    stage_fullchain(
        database,
        workdir,
        limit=5,
        seed=1,
        model="m",
        judge_model="m",
        dry_run=False,
        resume=True,
        rewrite_enabled=True,
        channel_timeout=0.5,
        input_price_mtok=1.0,
        output_price_mtok=5.0,
        provider_attempts=3,
        provider_backoff=1.0,
        answer_model="m",
        answer_effort="low",
        aux_model="m",
        aux_effort="low",
        judge_packet_mode="citation-focused",
    )
    merged = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    merged_ids = {row["case_id"] for row in merged}
    assert merged_ids == {"case-a"}
    assert merged[0]["answer"] == "ok"
    assert calls["resume"] is True
    assert calls["detail_path"] == detail
    assert calls["prewarm_embedding"] is True
    assert calls["judge_packet_mode"] == "citation-focused"


def test_offline_case_records_only_answerable_citation_sources():
    packed = SimpleNamespace(
        source_msg_ids=("gold", "recent-only"),
        evidence_segments=(),
        facts=(SimpleNamespace(source_msg_ids=("gold",)),),
        summaries=(),
    )
    trace = SimpleNamespace(
        result=SimpleNamespace(packed_context=packed),
        resolved_query=SimpleNamespace(subject_ids=None, rewrite_used=False),
        phase_timings_ms=(),
        attempted_channels=(),
        failed_channels=(),
    )
    runtime = SimpleNamespace(
        v2_provider=SimpleNamespace(evaluate=lambda request: trace)
    )

    row = suite._offline_case(
        engine=None,
        runtime=runtime,
        case={
            "case_id": "case-a",
            "group_id": 900000001,
            "query": "test question",
            "expected_evidence_message_ids": ("gold",),
            "expected_layer": "fact",
            "recent_context_message_ids": (),
            "tags": (),
        },
        settings=SimpleNamespace(bot_qq=900000101),
    )

    assert row["packed_source_ids"] == ("gold", "recent-only")
    assert row["allowed_citation_ids"] == ("gold",)


def test_materialize_fullchain_rows_fails_closed_on_mismatched_detail_signature(tmp_path):
    detail = tmp_path / "detail.jsonl"
    detail.write_text(
        json.dumps(
            {
                "case_id": "case-a",
                "case_input_signature": "old-signature",
                "resume_base_signature": "base-signature",
                "protocol_failure_codes": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        json.dumps(
            {
                "case_id": "case-a",
                "ok": True,
                "case_input_signature": "current-signature",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="matching signed fullchain detail"):
        suite._materialize_fullchain_rows(
            [{"case_id": "case-a", "query": "q"}],
            fresh_rows=(),
            detail_path=detail,
            progress_path=progress,
            resume=True,
        )


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
    report = stage_report(
        workdir,
        baseline_dir=None,
        gate_grounded=0.5,
        gate_recall=None,
        gate_protocol_failures=1,
        gate_p95_ms=100,
    )
    assert report["offline"]["cases"] == 1
    assert report["fullchain"]["answer_accuracy"] == 1.0
    assert report["gate"]["grounded_answer_accuracy"]["passed"] is True
    assert (workdir / "report.json").exists()
    assert (workdir / "report.md").exists()


def test_stage_report_uses_answerable_recall_and_keeps_legacy_projection(tmp_path):
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / "offline-results.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "case_id": "answerable",
                    "category": "profile",
                    "kind": "profile",
                    "expected_layer": "fact",
                    "expected_evidence_message_ids": ["gold-a"],
                    "packed_source_ids": ["gold-a"],
                    "allowed_citation_ids": ["gold-a"],
                    "fact_hit": True,
                },
                {
                    "case_id": "recent-only",
                    "category": "profile",
                    "kind": "profile",
                    "expected_layer": "fact",
                    "expected_evidence_message_ids": ["gold-b"],
                    "packed_source_ids": ["gold-b", "recent-only"],
                    "allowed_citation_ids": ["recent-only"],
                    "fact_hit": True,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = stage_report(
        workdir,
        baseline_dir=None,
        gate_grounded=None,
        gate_recall=0.75,
        gate_protocol_failures=None,
        gate_p95_ms=None,
    )

    assert report["offline"]["kind_recall"]["profile"] == 0.5
    assert report["offline"]["legacy_kind_recall"]["profile"] == 1.0
    assert report["gate"]["average_kind_recall"]["actual"] == 0.5
    assert report["gate"]["average_kind_recall"]["passed"] is False
    assert "legacy kind recall (packed-ID)" in (
        workdir / "report.md"
    ).read_text(encoding="utf-8")


def test_stage_report_projects_private_stress_failures_to_aggregates(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "stress.json").write_text(
        json.dumps(
            {
                "total_cases": 2,
                "total_ok": 1,
                "categories": {
                    "raw_history": {
                        "cases": 2,
                        "hits": 1,
                        "hit_rate": 0.5,
                        "violations": 0,
                        "errors": 0,
                        "failures": [
                            {"query": "private query", "expected": ["private-id"]}
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = stage_report(
        workdir,
        baseline_dir=None,
        gate_grounded=None,
        gate_recall=None,
        gate_protocol_failures=None,
        gate_p95_ms=None,
    )

    assert report["stress"] == {
        "total_cases": 2,
        "total_ok": 1,
        "categories": {
            "raw_history": {
                "cases": 2,
                "hits": 1,
                "hit_rate": 0.5,
                "violations": 0,
                "errors": 0,
            }
        },
    }
    rendered = (workdir / "report.json").read_text(encoding="utf-8")
    assert "private query" not in rendered
    assert "private-id" not in rendered


def _write_baseline_compatibility_inputs(workdir):
    (workdir / "offline-results.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "case_id": "answerable",
                    "kind": "profile",
                    "expected_evidence_message_ids": ["gold-a"],
                    "packed_source_ids": ["gold-a"],
                    "allowed_citation_ids": ["gold-a"],
                    "expected_layer": "fact",
                    "fact_hit": True,
                },
                {
                    "case_id": "packed-only",
                    "kind": "profile",
                    "expected_evidence_message_ids": ["gold-b"],
                    "packed_source_ids": ["gold-b", "recent"],
                    "allowed_citation_ids": ["recent"],
                    "expected_layer": "fact",
                    "fact_hit": True,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (workdir / "fullchain-results.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "case_id": "must-abstain",
                    "answer_expectation": "must_abstain",
                    "expected_abstention": True,
                    "abstained": True,
                },
                {
                    "case_id": "must-answer",
                    "answer_expectation": "must_answer",
                    "expected_abstention": False,
                    "abstained": True,
                },
                {
                    "case_id": "either",
                    "answer_expectation": "either",
                    "expected_abstention": False,
                    "abstained": False,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_stage_report_legacy_baseline_only_diffs_legacy_metric_views(tmp_path):
    workdir = tmp_path / "run"
    workdir.mkdir()
    _write_baseline_compatibility_inputs(workdir)
    baseline_dir = tmp_path / "legacy-baseline"
    baseline_dir.mkdir()
    (baseline_dir / "report.json").write_text(
        json.dumps(
            {
                "offline": {"kind_recall": {"profile": 0.25}},
                "fullchain": {
                    "abstention_precision": 0.25,
                    "abstention_recall": 0.4,
                    "abstention_f1": 0.3,
                },
            }
        ),
        encoding="utf-8",
    )

    report = stage_report(
        workdir,
        baseline_dir=baseline_dir,
        gate_grounded=None,
        gate_recall=None,
        gate_protocol_failures=None,
        gate_p95_ms=None,
    )

    offline_diff = report["baseline"]["offline_diff"]
    assert offline_diff["current"]["allowed_citation_kind_recall"][
        "status"
    ] == "not_comparable"
    assert offline_diff["legacy"]["packed_ids_kind_recall"]["profile"] == {
        "before": 0.25,
        "after": 1.0,
        "delta": 0.75,
    }
    fullchain_diff = report["baseline"]["fullchain_diff"]
    assert fullchain_diff["current"]["expectation_aware_abstention"][
        "status"
    ] == "not_comparable"
    assert fullchain_diff["legacy"]["empty_gold_abstention"][
        "legacy_abstention_precision"
    ] == {"before": 0.25, "after": 0.5, "delta": 0.25}
    markdown = (workdir / "report.md").read_text(encoding="utf-8")
    assert "current (allowed-citation)" in markdown
    assert "legacy (packed-ID)" in markdown
    assert "current (expectation-aware)" in markdown
    assert "legacy (empty-gold)" in markdown


def test_stage_report_dual_field_baseline_diffs_current_and_legacy_separately(
    tmp_path,
):
    workdir = tmp_path / "run"
    workdir.mkdir()
    _write_baseline_compatibility_inputs(workdir)
    current = stage_report(
        workdir,
        baseline_dir=None,
        gate_grounded=None,
        gate_recall=None,
        gate_protocol_failures=None,
        gate_p95_ms=None,
    )
    baseline_dir = tmp_path / "dual-baseline"
    baseline_dir.mkdir()
    (baseline_dir / "report.json").write_text(
        json.dumps({"offline": current["offline"], "fullchain": current["fullchain"]}),
        encoding="utf-8",
    )

    report = stage_report(
        workdir,
        baseline_dir=baseline_dir,
        gate_grounded=None,
        gate_recall=None,
        gate_protocol_failures=None,
        gate_p95_ms=None,
    )

    offline_diff = report["baseline"]["offline_diff"]
    assert offline_diff["current"]["allowed_citation_kind_recall"][
        "profile"
    ]["delta"] == 0.0
    assert offline_diff["legacy"]["packed_ids_kind_recall"]["profile"][
        "delta"
    ] == 0.0
    fullchain_diff = report["baseline"]["fullchain_diff"]
    assert fullchain_diff["current"]["expectation_aware_abstention"][
        "abstention_precision"
    ]["delta"] == 0.0
    assert fullchain_diff["legacy"]["empty_gold_abstention"][
        "legacy_abstention_precision"
    ]["delta"] == 0.0
