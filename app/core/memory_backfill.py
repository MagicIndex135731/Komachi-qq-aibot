from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


LEDGER_FORMAT_VERSION = 1
LEDGER_COLUMNS = (
    "id",
    "platform_msg_id",
    "group_id",
    "user_id",
    "timestamp_raw_text",
    "raw_json_raw_text",
    "plain_text",
    "msg_type",
    "reply_to_msg_id",
    "mentioned_bot",
)
LEDGER_HASH_ALGORITHM = "sha256-length-prefixed-utf8-json-array"


def message_ledger_manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class BackupResult:
    path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    integrity_check: str


@dataclass(frozen=True, slots=True)
class LedgerBucketComparison:
    watermark: int
    expected_count: int
    watermark_count: int
    rows_above_watermark: int
    digest_matches: bool


@dataclass(frozen=True, slots=True)
class LedgerComparison:
    matches: bool
    buckets: dict[str, LedgerBucketComparison]


def create_verified_sqlite_backup(
    source_path: Path,
    backup_dir: Path,
    *,
    backup_tag: str | None = None,
) -> BackupResult:
    """Create a WAL-consistent backup and its immutable-message manifest."""
    source_path = Path(source_path)
    backup_dir = Path(backup_dir)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    tag = backup_tag or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if not tag or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in tag):
        raise ValueError("backup_tag contains unsupported characters")

    backup_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    target_path = backup_dir / f"bot-{tag}.db"
    temporary_path = backup_dir / f".bot-{tag}.tmp"
    manifest_path = backup_dir / f"bot-{tag}.manifest.json"
    for path in (target_path, temporary_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)

    source = sqlite3.connect(source_path, timeout=30)
    destination = sqlite3.connect(temporary_path)
    try:
        source.execute("PRAGMA query_only=ON")
        with destination:
            source.backup(destination, pages=256, sleep=0.05)
    finally:
        destination.close()
        source.close()

    verification = sqlite3.connect(temporary_path)
    try:
        integrity_rows = [
            str(row[0])
            for row in verification.execute("PRAGMA integrity_check")
        ]
        foreign_key_rows = list(verification.execute("PRAGMA foreign_key_check"))
    finally:
        verification.close()
    if integrity_rows != ["ok"]:
        raise RuntimeError("backup integrity_check failed")
    if foreign_key_rows:
        raise RuntimeError("backup foreign_key_check failed")

    temporary_path.replace(target_path)
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    manifest = build_message_ledger_manifest(
        target_path,
        backup_name=target_path.name,
        generated_at=generated_at,
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return BackupResult(
        path=target_path,
        manifest_path=manifest_path,
        manifest=manifest,
        integrity_check="ok",
    )


def build_message_ledger_manifest(
    database_path: Path,
    *,
    backup_name: str,
    generated_at: str,
) -> dict[str, Any]:
    connection = sqlite3.connect(Path(database_path))
    try:
        connection.execute("PRAGMA query_only=ON")
        buckets: dict[str, dict[str, Any]] = {}
        group_ids = [
            int(row[0])
            for row in connection.execute(
                "SELECT DISTINCT group_id FROM messages WHERE group_id IS NOT NULL ORDER BY group_id"
            )
        ]
        private_exists = bool(
            connection.execute(
                "SELECT 1 FROM messages WHERE group_id IS NULL LIMIT 1"
            ).fetchone()
        )
        bucket_specs: list[tuple[str, int | None]] = [
            *( (f"group:{group_id}", group_id) for group_id in group_ids ),
            *((("private", None),) if private_exists else ()),
        ]
        for bucket_key, group_id in bucket_specs:
            watermark = _bucket_watermark(connection, group_id)
            count, digest = _bucket_digest(connection, group_id, watermark)
            buckets[bucket_key] = {
                "group_id": group_id,
                "watermark": watermark,
                "count": count,
                "sha256": digest,
            }
        return {
            "format_version": LEDGER_FORMAT_VERSION,
            "algorithm": LEDGER_HASH_ALGORITHM,
            "columns": list(LEDGER_COLUMNS),
            "backup_name": str(backup_name),
            "generated_at": str(generated_at),
            "total_count": sum(int(bucket["count"]) for bucket in buckets.values()),
            "buckets": buckets,
        }
    finally:
        connection.close()


def verify_message_ledger_manifest(
    database_path: Path,
    manifest: dict[str, Any],
) -> LedgerComparison:
    _validate_manifest_contract(manifest)
    connection = sqlite3.connect(Path(database_path))
    comparisons: dict[str, LedgerBucketComparison] = {}
    try:
        connection.execute("PRAGMA query_only=ON")
        for bucket_key, expected in dict(manifest["buckets"]).items():
            group_id = expected.get("group_id")
            normalized_group_id = None if group_id is None else int(group_id)
            watermark = int(expected["watermark"])
            count, digest = _bucket_digest(connection, normalized_group_id, watermark)
            extras = _bucket_rows_above_watermark(
                connection,
                normalized_group_id,
                watermark,
            )
            comparisons[str(bucket_key)] = LedgerBucketComparison(
                watermark=watermark,
                expected_count=int(expected["count"]),
                watermark_count=count,
                rows_above_watermark=extras,
                digest_matches=digest == str(expected["sha256"]),
            )
        expected_bucket_keys = set(str(key) for key in manifest["buckets"])
        live_bucket_specs: list[tuple[str, int | None]] = [
            *(
                (f"group:{int(row[0])}", int(row[0]))
                for row in connection.execute(
                    "SELECT DISTINCT group_id FROM messages "
                    "WHERE group_id IS NOT NULL ORDER BY group_id"
                )
            ),
            *(
                (("private", None),)
                if connection.execute(
                    "SELECT 1 FROM messages WHERE group_id IS NULL LIMIT 1"
                ).fetchone()
                else ()
            ),
        ]
        for bucket_key, group_id in live_bucket_specs:
            if bucket_key in expected_bucket_keys:
                continue
            comparisons[bucket_key] = LedgerBucketComparison(
                watermark=0,
                expected_count=0,
                watermark_count=0,
                rows_above_watermark=_bucket_rows_above_watermark(
                    connection,
                    group_id,
                    0,
                ),
                digest_matches=True,
            )
    finally:
        connection.close()
    matches = all(
        item.watermark_count == item.expected_count and item.digest_matches
        for item in comparisons.values()
    )
    return LedgerComparison(matches=matches, buckets=comparisons)


def _validate_manifest_contract(manifest: dict[str, Any]) -> None:
    if int(manifest.get("format_version", 0)) != LEDGER_FORMAT_VERSION:
        raise ValueError("unsupported ledger manifest format")
    if manifest.get("algorithm") != LEDGER_HASH_ALGORITHM:
        raise ValueError("unsupported ledger digest algorithm")
    if tuple(manifest.get("columns", ())) != LEDGER_COLUMNS:
        raise ValueError("ledger manifest columns do not match the v1 contract")
    if not isinstance(manifest.get("buckets"), dict):
        raise ValueError("ledger manifest buckets must be an object")


def _bucket_watermark(connection: sqlite3.Connection, group_id: int | None) -> int:
    predicate, parameters = _bucket_predicate(group_id)
    row = connection.execute(
        f"SELECT COALESCE(MAX(id), 0) FROM messages WHERE {predicate}",
        parameters,
    ).fetchone()
    return int(row[0] if row else 0)


def _bucket_digest(
    connection: sqlite3.Connection,
    group_id: int | None,
    watermark: int,
) -> tuple[int, str]:
    predicate, parameters = _bucket_predicate(group_id)
    rows = connection.execute(
        f"""
        SELECT id, platform_msg_id, group_id, user_id, CAST(timestamp AS TEXT),
               CAST(raw_json AS TEXT), plain_text, msg_type, reply_to_msg_id, mentioned_bot
        FROM messages
        WHERE {predicate} AND id <= ?
        ORDER BY id
        """,
        (*parameters, int(watermark)),
    )
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        payload = json.dumps(
            list(row),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        count += 1
    return count, digest.hexdigest()


def _bucket_rows_above_watermark(
    connection: sqlite3.Connection,
    group_id: int | None,
    watermark: int,
) -> int:
    predicate, parameters = _bucket_predicate(group_id)
    row = connection.execute(
        f"SELECT COUNT(*) FROM messages WHERE {predicate} AND id > ?",
        (*parameters, int(watermark)),
    ).fetchone()
    return int(row[0] if row else 0)


def _bucket_predicate(group_id: int | None) -> tuple[str, tuple[Any, ...]]:
    if group_id is None:
        return "group_id IS NULL", ()
    return "group_id = ?", (int(group_id),)
