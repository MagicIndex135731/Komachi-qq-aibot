from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

from sqlalchemy.engine import Engine

from app.core.member_identity import (
    group_member_identities_from_messages,
    resolve_group_member_reference,
)
from app.core.memory_compaction import canonical_key
from app.core.memory_engine import parse_personal_claim
from app.storage.db import build_engine, create_all, session_scope
from app.storage.repositories import MemoryRepository, MessageRepository, UserRepository


@dataclass(frozen=True, slots=True)
class MemoryCorrectionRepairResult:
    group_id: int
    subject_id: int
    replacement_memory_id: int
    erroneous_memory_ids: tuple[int, ...]
    superseded_count: int
    already_superseded_count: int
    supporting_source_ids: tuple[str, ...]
    erroneous_source_ids: tuple[str, ...]


def _source_ids(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    if not normalized:
        raise ValueError(f"{field} must contain at least one source message ID")
    return normalized


def _memory_has_source(memory: object, source_id: str) -> bool:
    return source_id == str(getattr(memory, "source_msg_id", "")) or source_id in {
        str(item) for item in (getattr(memory, "source_msg_ids", None) or [])
    }


def repair_memory_correction(
    engine: Engine,
    *,
    group_id: int,
    predicate: str,
    object_text: str,
    supporting_source_ids: Sequence[str],
    erroneous_source_ids: Sequence[str],
    subject_id: int | None = None,
    subject_alias: str = "",
) -> MemoryCorrectionRepairResult:
    """Repair one proven attribution without deleting raw or derived evidence."""

    if int(group_id) <= 0:
        raise ValueError("group_id must be positive")
    normalized_predicate = str(predicate).strip().casefold()
    if normalized_predicate not in {"likes", "dislikes"}:
        raise ValueError("predicate must be likes or dislikes")
    normalized_object = " ".join(str(object_text or "").split())
    if not normalized_object:
        raise ValueError("object_text is required")
    supporting = _source_ids(supporting_source_ids, field="supporting_source_ids")
    erroneous = _source_ids(erroneous_source_ids, field="erroneous_source_ids")
    with session_scope(engine) as session:
        messages = MessageRepository(session)
        users = UserRepository(session)
        memories = MemoryRepository(session)
        all_source_ids = [*supporting, *erroneous]
        source_messages = messages.get_group_messages_by_platform_msg_ids(
            group_id=group_id,
            platform_msg_ids=all_source_ids,
        )
        missing_messages = [source_id for source_id in all_source_ids if source_id not in source_messages]
        if missing_messages:
            raise ValueError(
                "source messages not found in requested group: " + ",".join(missing_messages)
            )

        member_user_ids = messages.list_recent_group_user_ids(group_id=group_id, limit=10_000)
        member_users = users.get_users_by_ids(member_user_ids)
        members = group_member_identities_from_messages(
            messages.list_recent_group_member_messages(group_id=group_id, limit=10_000)
        )
        resolved_subject_id = int(subject_id) if subject_id is not None else None
        matched_alias = ""
        if subject_alias.strip():
            resolved_alias = resolve_group_member_reference(
                subject_alias,
                members,
                match_mode="exact",
            )
            if resolved_alias is None:
                raise ValueError("subject alias is not unique in requested group")
            if resolved_subject_id is not None and resolved_alias.user_id != resolved_subject_id:
                raise ValueError("subject alias and subject_id identify different users")
            resolved_subject_id = resolved_alias.user_id
            matched_alias = resolved_alias.matched_alias
        if resolved_subject_id is None:
            raise ValueError("subject_id or subject_alias is required")
        if resolved_subject_id not in member_users:
            raise ValueError("subject_id has not spoken in requested group")

        for source_id in supporting:
            source_message = source_messages[source_id]
            claim = parse_personal_claim(str(source_message.plain_text or ""))
            if claim is None:
                raise ValueError(f"supporting source {source_id} is not an explicit personal claim")
            if claim.predicate != normalized_predicate or not memories.correction_objects_are_related(
                normalized_object,
                claim.object_text,
            ):
                raise ValueError(f"supporting source {source_id} does not support the requested fact")
            if claim.subject_mode == "sender":
                claim_subject_id = int(source_message.user_id)
            else:
                resolved_claim_subject = resolve_group_member_reference(
                    str(claim.subject_alias or ""),
                    members,
                    match_mode="exact",
                )
                claim_subject_id = (
                    resolved_claim_subject.user_id
                    if resolved_claim_subject is not None
                    else None
                )
            if claim_subject_id != resolved_subject_id:
                raise ValueError(f"supporting source {source_id} identifies a different or ambiguous subject")

        memory_kind = "preference" if normalized_predicate == "likes" else "taboo"
        source_bound_memories = memories.list_group_memories_by_source_msg_ids(
            scope_id=str(group_id),
            source_msg_ids=list(erroneous),
        )
        erroneous_memories_by_id: dict[int, object] = {}
        for source_id in erroneous:
            matches = [
                memory
                for memory in source_bound_memories
                if _memory_has_source(memory, source_id)
                and memory.subject_type == "user"
                and str(memory.subject_id) != str(resolved_subject_id)
                and memory.memory_kind == memory_kind
                and str(memory.predicate or "") in {"", normalized_predicate}
                and memories.correction_objects_are_related(
                    normalized_object,
                    memory.object_text
                    if str(memory.object_text or "").strip()
                    else memory.content,
                )
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"erroneous source {source_id} must identify exactly one matching wrong-subject fact"
                )
            erroneous_memories_by_id[int(matches[0].id)] = matches[0]
        erroneous_memories = list(erroneous_memories_by_id.values())

        observed_at = max(source_messages[source_id].timestamp for source_id in supporting)
        subject_user = member_users[resolved_subject_id]
        display_subject = (
            matched_alias
            or str(subject_user.group_card or "").strip()
            or str(subject_user.nickname or "").strip()
            or str(resolved_subject_id)
        )
        replacement = memories.upsert_canonical_memory(
            scope_type="group",
            scope_id=str(group_id),
            subject_type="user",
            subject_id=str(resolved_subject_id),
            memory_kind=memory_kind,
            canonical_key=canonical_key(
                memory_kind,
                str(resolved_subject_id),
                normalized_predicate,
                normalized_object,
            ),
            predicate=normalized_predicate,
            object_text=normalized_object,
            content=f"{display_subject} {normalized_predicate} {normalized_object}.",
            importance=4,
            confidence=0.99,
            source_msg_ids=list(supporting),
            valid_from=observed_at,
        )

        for memory in erroneous_memories:
            if memory.status == "active":
                continue
            if memory.status == "superseded" and memory.superseded_by_id == replacement.id:
                continue
            raise ValueError(f"erroneous memory {memory.id} already has a different lifecycle")

        superseded_count = 0
        already_superseded_count = 0
        for memory in erroneous_memories:
            if memory.status == "superseded":
                already_superseded_count += 1
                continue
            memories.mark_superseded(
                memory_id=memory.id,
                superseded_by_id=replacement.id,
                valid_until=observed_at,
            )
            superseded_count += 1
        if erroneous_memories and replacement.supersedes_id is None:
            replacement.supersedes_id = erroneous_memories[0].id
            session.add(replacement)

        return MemoryCorrectionRepairResult(
            group_id=int(group_id),
            subject_id=resolved_subject_id,
            replacement_memory_id=replacement.id,
            erroneous_memory_ids=tuple(memory.id for memory in erroneous_memories),
            superseded_count=superseded_count,
            already_superseded_count=already_superseded_count,
            supporting_source_ids=supporting,
            erroneous_source_ids=erroneous,
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Idempotently repair one group-memory attribution after a verified online backup."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--group-id", required=True, type=int)
    parser.add_argument("--subject-id", type=int)
    parser.add_argument("--subject-alias", default="")
    parser.add_argument("--predicate", required=True, choices=("likes", "dislikes"))
    parser.add_argument("--object", dest="object_text", required=True)
    parser.add_argument("--supporting-source-id", action="append", required=True)
    parser.add_argument("--erroneous-source-id", action="append", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if not args.database.is_file():
        raise FileNotFoundError(args.database)
    engine = build_engine(args.database)
    try:
        create_all(engine)
        result = repair_memory_correction(
            engine,
            group_id=args.group_id,
            subject_id=args.subject_id,
            subject_alias=args.subject_alias,
            predicate=args.predicate,
            object_text=args.object_text,
            supporting_source_ids=args.supporting_source_id,
            erroneous_source_ids=args.erroneous_source_id,
        )
    finally:
        engine.dispose()
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
