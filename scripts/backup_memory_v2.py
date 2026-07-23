from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.core.memory_backfill import create_verified_sqlite_backup


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a verified SQLite online backup and V2 message ledger."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = create_verified_sqlite_backup(
        args.database,
        args.backup_dir,
        backup_tag=args.tag,
    )
    print(
        json.dumps(
            {
                "backup_path": str(result.path),
                "manifest_path": str(result.manifest_path),
                "integrity_check": result.integrity_check,
                "message_count": int(result.manifest["total_count"]),
                "bucket_count": len(result.manifest["buckets"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
