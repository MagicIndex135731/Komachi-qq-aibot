from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.backfill_memory_v2 as backfill_cli
import scripts.run_memory_recall_eval as eval_cli


def test_real_eval_separates_functional_rewrite_from_local_benchmark() -> None:
    source = (Path(__file__).parents[1] / "scripts/run_memory_recall_eval.py").read_text(
        encoding="utf-8"
    )

    assert '"memory_query_rewrite_enabled": True' in source
    assert '"memory_raw_v3_enabled": True' in source
    assert 'update={"memory_query_rewrite_enabled": False}' in source
    assert "bot_display_name=str(functional_settings.bot_qq)" in source
    assert "AC_VECTOR_NOT_EXERCISED" in source
    assert "AC_VECTOR_QUERY_FAILED" in source
    assert 'embed_query("memory-v3-vector-readiness-probe")' in source
    assert "AC_RECALL_AT_150" in source
    assert "AC_RECALL_WITHIN_24K" in source
    assert "raw_message_embedding_generation_override=prepared_generation" in source
    assert 'report["vector_generation"] = prepared_generation' in source


def test_real_eval_requires_database_bound_ready_prepared_report(tmp_path: Path) -> None:
    database = tmp_path / "bot.db"
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(
        json.dumps(
            {
                "phase": "prepared",
                "database_path": str(database.resolve()),
                "vector_status": "ready",
                "vector_generation": 17,
                "vector_identity": {
                    "provider": "local",
                    "model": "test",
                    "dimensions": 2,
                    "version": "v1",
                },
            }
        ),
        encoding="utf-8",
    )

    prepared = eval_cli._load_prepared_report(
        prepared_path,
        database=database,
    )

    assert prepared["vector_generation"] == 17
    prepared["database_path"] = str((tmp_path / "other.db").resolve())
    prepared_path.write_text(json.dumps(prepared), encoding="utf-8")
    with pytest.raises(eval_cli.AcceptanceGateError) as exc_info:
        eval_cli._load_prepared_report(prepared_path, database=database)
    assert exc_info.value.codes == ("AC_PREPARED_REPORT_INVALID",)


def test_real_eval_rejects_nonstandard_json_constants(tmp_path: Path) -> None:
    prepared_path = tmp_path / "prepared.json"
    manifest_path = tmp_path / "manifest.json"
    for invalid_constant in ("NaN", "Infinity", "-Infinity"):
        prepared_path.write_text(
            f'{{"phase": {invalid_constant}}}',
            encoding="utf-8",
        )
        with pytest.raises(eval_cli.AcceptanceGateError) as exc_info:
            eval_cli._load_prepared_report(prepared_path, database=tmp_path / "bot.db")
        assert exc_info.value.codes == ("AC_PREPARED_REPORT_INVALID",)

        manifest_path.write_text(
            f'{{"watermark": {invalid_constant}}}',
            encoding="utf-8",
        )
        with pytest.raises(eval_cli.AcceptanceGateError) as exc_info:
            eval_cli._validate_v3_rollout_state(
                engine=object(),
                runtime=object(),
                database=tmp_path / "bot.db",
                manifest_path=manifest_path,
                prepared_report={},
            )
        assert exc_info.value.codes == ("AC_SNAPSHOT_MANIFEST_INVALID",)


def test_final_ledger_mismatch_marks_run_failed(monkeypatch, tmp_path: Path) -> None:
    marked: list[tuple[int, str]] = []
    monkeypatch.setattr(
        backfill_cli,
        "verify_message_ledger_manifest",
        lambda *_args: SimpleNamespace(matches=False, buckets={}),
    )
    monkeypatch.setattr(
        backfill_cli,
        "_mark_backfill_failed",
        lambda _engine, *, run_id, error_code: marked.append((run_id, error_code)),
    )

    with pytest.raises(RuntimeError, match="snapshot watermark"):
        backfill_cli._verify_final_ledger(
            database=tmp_path / "db.sqlite",
            manifest={},
            engine=object(),
            run_id=17,
        )

    assert marked == [(17, "LedgerMismatch")]


def test_final_ledger_allows_and_reports_rows_above_watermark(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        backfill_cli,
        "verify_message_ledger_manifest",
        lambda *_args: SimpleNamespace(
            matches=True,
            buckets={
                "group:1": SimpleNamespace(rows_above_watermark=3),
                "group:2": SimpleNamespace(rows_above_watermark=0),
                "group:9": SimpleNamespace(rows_above_watermark=2),
            },
        ),
    )

    result = backfill_cli._verify_final_ledger(
        database=tmp_path / "db.sqlite",
        manifest={},
        engine=object(),
        run_id=18,
    )

    assert result == {
        "matches": True,
        "rows_above_watermark": {"group:1": 3, "group:9": 2},
        "rows_above_watermark_total": 5,
    }
