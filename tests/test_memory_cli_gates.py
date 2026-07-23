from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.backfill_memory_v2 as backfill_cli


def test_real_eval_separates_functional_rewrite_from_local_benchmark() -> None:
    source = (Path(__file__).parents[1] / "scripts/run_memory_recall_eval.py").read_text(
        encoding="utf-8"
    )

    assert '"memory_query_rewrite_enabled": True' in source
    assert 'update={"memory_query_rewrite_enabled": False}' in source
    assert "bot_display_name=str(functional_settings.bot_qq)" in source
    assert "AC_VECTOR_NOT_EXERCISED" in source
    assert "AC_VECTOR_QUERY_FAILED" in source
    assert 'embed_query("memory-v2-vector-readiness-probe")' in source


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
