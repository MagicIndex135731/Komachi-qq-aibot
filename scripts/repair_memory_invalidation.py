from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC
import hashlib
import json
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.storage.db import build_engine, session_scope
from app.storage.models import MemoryItem, Message
from app.storage.repositories import MemoryRepository


@dataclass(frozen=True, slots=True)
class InvalidationRepairResult:
    target_memory_id: int
    target_sha256: str
    reason: str
    source_count: int
    applied: bool
    receipt_memory_id: int | None


def repair_memory_invalidation(
    *,
    database: Path,
    group_id: int,
    target_memory_id: int,
    expected_target_sha256: str,
    source_msg_ids: Sequence[str],
    reason: str,
    apply: bool,
) -> InvalidationRepairResult:
    """Preflight and optionally retire one exact, independently reviewed fact."""

    sources = tuple(dict.fromkeys(str(item).strip() for item in source_msg_ids if str(item).strip()))
    if not sources:
        raise ValueError("at least one source_msg_id is required")
    if reason not in {"explicit_denial", "manual_review_rejected"}:
        raise ValueError("unsupported invalidation reason")
    engine = build_engine(database)
    try:
        with session_scope(engine) as session:
            target = session.get(MemoryItem, int(target_memory_id))
            if target is None:
                raise ValueError("target memory does not exist")
            if target.scope_type != "group" or target.scope_id != str(int(group_id)):
                raise ValueError("target memory is outside the requested group")
            if target.memory_kind not in {"profile", "preference"}:
                raise ValueError("only profile or preference memories may be invalidated")
            target_key = str(target.canonical_key or "").strip()
            target_hash = hashlib.sha256(target_key.encode("utf-8")).hexdigest()
            if target_hash != str(expected_target_sha256).strip().casefold():
                raise ValueError("target canonical hash mismatch")
            source_rows = list(
                session.scalars(
                    select(Message).where(
                        Message.group_id == int(group_id),
                        Message.platform_msg_id.in_(sources),
                    )
                )
            )
            if len(source_rows) != len(sources):
                raise ValueError("one or more audit sources are missing from the group")
            if reason == "explicit_denial" and any(
                str(row.user_id) != str(target.subject_id) for row in source_rows
            ):
                raise ValueError("explicit denial must be authored by the target subject")
            receipt = None
            if apply:
                local_until = max(row.timestamp for row in source_rows).replace(
                    tzinfo=ZoneInfo("Asia/Shanghai")
                )
                receipt = MemoryRepository(session).invalidate_canonical_memory(
                    scope_id=str(group_id),
                    target_canonical_key=target_key,
                    source_msg_ids=list(sources),
                    valid_until=local_until.astimezone(UTC),
                    reason=reason,
                    expected_target_sha256=target_hash,
                )
                if receipt is None:
                    raise ValueError("repository rejected the exact invalidation")
            return InvalidationRepairResult(
                target_memory_id=int(target_memory_id),
                target_sha256=target_hash,
                reason=reason,
                source_count=len(sources),
                applied=bool(apply),
                receipt_memory_id=int(receipt.id) if receipt is not None else None,
            )
    finally:
        engine.dispose()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run by default; use --apply only after exact hash review."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--group-id", required=True, type=int)
    parser.add_argument("--target-memory-id", required=True, type=int)
    parser.add_argument("--expected-target-sha256", required=True)
    parser.add_argument("--source-msg-id", action="append", required=True)
    parser.add_argument(
        "--reason",
        choices=("explicit_denial", "manual_review_rejected"),
        required=True,
    )
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if not args.database.is_file():
        raise FileNotFoundError(args.database)
    result = repair_memory_invalidation(
        database=args.database,
        group_id=args.group_id,
        target_memory_id=args.target_memory_id,
        expected_target_sha256=args.expected_target_sha256,
        source_msg_ids=args.source_msg_id,
        reason=args.reason,
        apply=args.apply,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
