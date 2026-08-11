import sqlite3

from scripts.repair_daily_summary_days import repair_daily_summary_days


def _build_db(path) -> None:
    con = sqlite3.connect(str(path))
    con.executescript(
        """
        CREATE TABLE summaries (
            id INTEGER PRIMARY KEY,
            summary_key TEXT,
            start_at TEXT,
            end_at TEXT,
            summary_level TEXT
        );
        """
    )
    con.executemany(
        "INSERT INTO summaries (id, summary_key, start_at, end_at, summary_level) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "semantic-daily:2026-08-10", "2026-08-09 23:36:52.000000", "2026-08-10 16:02:58.000000", "semantic_daily"),
            (2, "semantic-daily:2026-08-04", "2026-08-04 18:40:21.000000", "2026-08-05 01:40:23.000000", "semantic_daily"),
            (3, "daily:2026-07-31", "2026-07-30 22:08:21.000000", "2026-07-31 20:33:00.000000", "daily"),
            (4, "semantic-daily:garbage", "2026-08-01 00:00:00", "2026-08-01 12:00:00", "semantic_daily"),
        ],
    )
    con.commit()
    con.close()


def test_repair_daily_summary_days_clamps_windows_to_own_day(tmp_path) -> None:
    db_path = tmp_path / "bot.db"
    _build_db(db_path)

    result = repair_daily_summary_days(db_path)

    assert result["scanned"] == 4
    assert result["updated"] == 3
    assert result["skipped"] == 1
    con = sqlite3.connect(str(db_path))
    rows = dict(
        (row_id, (start_at, end_at))
        for row_id, start_at, end_at in con.execute("SELECT id, start_at, end_at FROM summaries")
    )
    con.close()
    assert rows[1] == ("2026-08-10 00:00:00.000000", "2026-08-10 16:02:58.000000")
    assert rows[2] == ("2026-08-04 18:40:21.000000", "2026-08-05 00:00:00.000000")
    assert rows[3] == ("2026-07-31 00:00:00.000000", "2026-07-31 20:33:00.000000")
    assert rows[4] == ("2026-08-01 00:00:00", "2026-08-01 12:00:00")


def test_repair_daily_summary_days_dry_run_does_not_write(tmp_path) -> None:
    db_path = tmp_path / "bot.db"
    _build_db(db_path)

    result = repair_daily_summary_days(db_path, dry_run=True)

    assert result["updated"] == 3
    con = sqlite3.connect(str(db_path))
    start_at = con.execute("SELECT start_at FROM summaries WHERE id = 1").fetchone()[0]
    con.close()
    assert start_at == "2026-08-09 23:36:52.000000"
