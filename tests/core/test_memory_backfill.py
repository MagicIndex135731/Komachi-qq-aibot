from __future__ import annotations

import json
import sqlite3

from scripts.backup_memory_v2 import main as backup_main
from app.core.memory_backfill import (
    build_message_ledger_manifest,
    create_verified_sqlite_backup,
    message_ledger_manifest_sha256,
    verify_message_ledger_manifest,
)


def create_message_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform_msg_id TEXT NOT NULL UNIQUE,
            group_id INTEGER,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            plain_text TEXT NOT NULL,
            msg_type TEXT NOT NULL,
            reply_to_msg_id TEXT,
            mentioned_bot INTEGER NOT NULL
        )
        """
    )


def insert_message(
    connection: sqlite3.Connection,
    *,
    platform_msg_id: str,
    group_id: int | None,
    text: str,
) -> None:
    connection.execute(
        """
        INSERT INTO messages (
            platform_msg_id, group_id, user_id, timestamp, raw_json,
            plain_text, msg_type, reply_to_msg_id, mentioned_bot
        ) VALUES (?, ?, 42, '2026-07-23 12:00:00.000000', ?, ?, 'text', NULL, 0)
        """,
        (platform_msg_id, group_id, json.dumps({"message": text}), text),
    )


def test_online_backup_captures_committed_wal_and_writes_reproducible_manifest(tmp_path) -> None:
    source_path = tmp_path / "bot.db"
    writer = sqlite3.connect(source_path)
    writer.execute("PRAGMA journal_mode=WAL")
    create_message_table(writer)
    insert_message(writer, platform_msg_id="g1", group_id=100, text="group one")
    insert_message(writer, platform_msg_id="g2", group_id=200, text="group two")
    insert_message(writer, platform_msg_id="p1", group_id=None, text="private")
    writer.commit()

    result = create_verified_sqlite_backup(
        source_path,
        tmp_path / "backups",
        backup_tag="pre-v2-test",
    )
    writer.close()

    assert result.integrity_check == "ok"
    assert result.path.name == "bot-pre-v2-test.db"
    assert result.manifest_path.is_file()
    assert result.manifest["format_version"] == 1
    assert result.manifest["buckets"]["group:100"]["count"] == 1
    assert result.manifest["buckets"]["group:200"]["count"] == 1
    assert result.manifest["buckets"]["private"]["count"] == 1
    assert build_message_ledger_manifest(
        result.path,
        backup_name=result.path.name,
        generated_at=result.manifest["generated_at"],
    ) == result.manifest


def test_manifest_verification_ignores_new_rows_above_snapshot_watermark(tmp_path) -> None:
    database_path = tmp_path / "bot.db"
    connection = sqlite3.connect(database_path)
    create_message_table(connection)
    insert_message(connection, platform_msg_id="m1", group_id=100, text="before")
    connection.commit()
    manifest = build_message_ledger_manifest(
        database_path,
        backup_name="snapshot.db",
        generated_at="2026-07-23T00:00:00Z",
    )

    insert_message(connection, platform_msg_id="m2", group_id=100, text="after")
    connection.commit()
    connection.close()

    comparison = verify_message_ledger_manifest(database_path, manifest)

    assert comparison.matches is True
    assert comparison.buckets["group:100"].watermark_count == 1
    assert comparison.buckets["group:100"].rows_above_watermark == 1


def test_manifest_digest_is_canonical_and_changes_with_snapshot_contract(tmp_path) -> None:
    database_path = tmp_path / "bot.db"
    connection = sqlite3.connect(database_path)
    create_message_table(connection)
    insert_message(connection, platform_msg_id="m1", group_id=100, text="before")
    connection.commit()
    connection.close()
    manifest = build_message_ledger_manifest(
        database_path,
        backup_name="snapshot.db",
        generated_at="2026-07-23T00:00:00Z",
    )

    digest = message_ledger_manifest_sha256(manifest)
    reordered = json.loads(json.dumps(manifest, sort_keys=False))
    changed = {**manifest, "backup_name": "different.db"}

    assert len(digest) == 64
    assert message_ledger_manifest_sha256(reordered) == digest
    assert message_ledger_manifest_sha256(changed) != digest


def test_manifest_verification_detects_any_change_to_a_watermarked_raw_message(tmp_path) -> None:
    database_path = tmp_path / "bot.db"
    connection = sqlite3.connect(database_path)
    create_message_table(connection)
    insert_message(connection, platform_msg_id="m1", group_id=100, text="original")
    connection.commit()
    manifest = build_message_ledger_manifest(
        database_path,
        backup_name="snapshot.db",
        generated_at="2026-07-23T00:00:00Z",
    )

    connection.execute("UPDATE messages SET plain_text = 'changed' WHERE platform_msg_id = 'm1'")
    connection.commit()
    connection.close()

    comparison = verify_message_ledger_manifest(database_path, manifest)

    assert comparison.matches is False
    assert comparison.buckets["group:100"].digest_matches is False


def test_manifest_verification_reports_new_group_rows_above_snapshot_boundary(tmp_path) -> None:
    database_path = tmp_path / "bot.db"
    connection = sqlite3.connect(database_path)
    create_message_table(connection)
    insert_message(connection, platform_msg_id="m1", group_id=100, text="before")
    connection.commit()
    manifest = build_message_ledger_manifest(
        database_path,
        backup_name="snapshot.db",
        generated_at="2026-07-23T00:00:00Z",
    )

    insert_message(connection, platform_msg_id="m2", group_id=200, text="new group")
    connection.commit()
    connection.close()

    comparison = verify_message_ledger_manifest(database_path, manifest)

    assert comparison.matches is True
    assert comparison.buckets["group:200"].watermark == 0
    assert comparison.buckets["group:200"].rows_above_watermark == 1


def test_backup_cli_emits_only_safe_paths_counts_and_integrity(tmp_path, capsys) -> None:
    database_path = tmp_path / "bot.db"
    connection = sqlite3.connect(database_path)
    create_message_table(connection)
    insert_message(connection, platform_msg_id="m1", group_id=100, text="private chat text")
    connection.commit()
    connection.close()

    assert backup_main(
        [
            "--database",
            str(database_path),
            "--backup-dir",
            str(tmp_path / "backups"),
            "--tag",
            "safe-test",
        ]
    ) == 0
    output = capsys.readouterr().out

    assert '"integrity_check": "ok"' in output
    assert '"message_count": 1' in output
    assert "private chat text" not in output
