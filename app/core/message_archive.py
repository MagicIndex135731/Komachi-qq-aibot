from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from app.core.message_content import extract_images_from_raw_payload
from app.storage.db import session_scope
from app.storage.models import Message, User


def append_group_message_archive(
    *,
    history_dir: Path,
    group_id: int,
    timestamp: datetime,
    platform_msg_id: str,
    user_id: int,
    nickname: str,
    group_card: str,
    plain_text: str,
    msg_type: str,
    mentioned_bot: bool,
    reply_to_msg_id: str | None,
    direction: str,
    image_local_paths: list[str],
) -> Path:
    archive_path = history_dir / f"group-{group_id}" / f"{timestamp.date().isoformat()}.jsonl"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": timestamp.isoformat(),
        "group_id": group_id,
        "platform_msg_id": platform_msg_id,
        "user_id": user_id,
        "nickname": nickname,
        "group_card": group_card,
        "plain_text": plain_text,
        "msg_type": msg_type,
        "mentioned_bot": mentioned_bot,
        "reply_to_msg_id": reply_to_msg_id,
        "direction": direction,
        "image_local_paths": image_local_paths,
    }
    if _archive_contains_platform_msg_id(archive_path, platform_msg_id):
        return archive_path
    with archive_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return archive_path


def sync_group_message_archives_from_db(
    *,
    engine,
    history_dir: Path,
    allowed_group_ids: set[int],
) -> dict[int, int]:
    if not allowed_group_ids:
        return {}

    grouped_records: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    counts: dict[int, int] = defaultdict(int)

    with session_scope(engine) as session:
        stmt = (
            select(Message, User)
            .join(User, Message.user_id == User.user_id)
            .where(Message.group_id.in_(allowed_group_ids))
            .order_by(Message.group_id.asc(), Message.timestamp.asc(), Message.id.asc())
        )
        rows = session.execute(stmt).all()

    for message, user in rows:
        if message.group_id is None or _is_reserved_outbound(message):
            continue
        record = _record_from_db_message(message=message, user=user)
        archive_day = record["timestamp"][:10]
        grouped_records[int(message.group_id)][archive_day].append(record)
        counts[int(message.group_id)] += 1

    for group_id in allowed_group_ids:
        group_dir = history_dir / f"group-{group_id}"
        if group_dir.exists():
            for existing in group_dir.glob("*.jsonl"):
                existing.unlink()
        else:
            group_dir.mkdir(parents=True, exist_ok=True)

    for group_id, day_records in grouped_records.items():
        group_dir = history_dir / f"group-{group_id}"
        for archive_day, records in day_records.items():
            archive_path = group_dir / f"{archive_day}.jsonl"
            deduped_records = _dedupe_records(records)
            with archive_path.open("w", encoding="utf-8") as handle:
                for record in deduped_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return dict(counts)


def _archive_contains_platform_msg_id(archive_path: Path, platform_msg_id: str) -> bool:
    if not archive_path.exists():
        return False
    try:
        with archive_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("platform_msg_id") == platform_msg_id:
                    return True
    except FileNotFoundError:
        return False
    return False


def _dedupe_records(records: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for record in records:
        platform_msg_id = str(record.get("platform_msg_id", "")).strip()
        if not platform_msg_id or platform_msg_id in seen:
            continue
        seen.add(platform_msg_id)
        deduped.append(record)
    deduped.sort(key=lambda record: (record.get("timestamp", ""), record.get("platform_msg_id", "")))
    return deduped


def _record_from_db_message(*, message: Message, user: User) -> dict:
    raw_json = message.raw_json if isinstance(message.raw_json, dict) else {}
    sender = raw_json.get("sender", {}) if isinstance(raw_json, dict) else {}
    timestamp = _ensure_utc(message.timestamp)
    direction = str(raw_json.get("direction", "")).strip() or (
        "outbound" if str(message.platform_msg_id).startswith("bot-reply-") else "inbound"
    )
    images = extract_images_from_raw_payload(raw_json)
    return {
        "timestamp": timestamp.isoformat(),
        "group_id": int(message.group_id),
        "platform_msg_id": str(message.platform_msg_id),
        "user_id": int(message.user_id),
        "nickname": str(sender.get("nickname", "")).strip() or str(user.nickname or "").strip(),
        "group_card": str(sender.get("card", "")).strip() or str(user.group_card or "").strip(),
        "plain_text": str(message.plain_text or ""),
        "msg_type": str(message.msg_type or "text"),
        "mentioned_bot": bool(message.mentioned_bot),
        "reply_to_msg_id": str(message.reply_to_msg_id).strip() if message.reply_to_msg_id else None,
        "direction": direction,
        "image_local_paths": [image.local_path for image in images if image.local_path],
    }


def _ensure_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _is_reserved_outbound(message: Message) -> bool:
    raw_json = message.raw_json
    return isinstance(raw_json, dict) and raw_json.get("delivery_state") == "reserved"
