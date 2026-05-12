from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml

from app.adapters.sender import OutboundPrivateMessage

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PrivateReminder:
    reminder_id: str
    user_id: int
    text: str
    run_at: datetime
    catch_up_if_missed: bool = False


def load_private_reminders(*, config_dir: Path) -> list[PrivateReminder]:
    path = config_dir / "private_reminders.yaml"
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")

    reminders_payload = payload.get("reminders", [])
    if not isinstance(reminders_payload, list):
        raise ValueError(f"expected reminders list in {path}")

    reminders: list[PrivateReminder] = []
    for item in reminders_payload:
        if not isinstance(item, dict):
            raise ValueError(f"expected reminder mapping in {path}")
        run_at = datetime.fromisoformat(str(item["run_at"]))
        if run_at.tzinfo is None:
            raise ValueError(f"reminder run_at must include timezone offset in {path}")
        reminders.append(
            PrivateReminder(
                reminder_id=str(item["id"]).strip(),
                user_id=int(item["user_id"]),
                text=str(item["text"]),
                run_at=run_at,
                catch_up_if_missed=bool(item.get("catch_up_if_missed", False)),
            )
        )
    return reminders


class PrivateReminderScheduler:
    def __init__(
        self,
        *,
        sender,
        data_dir: Path,
        reminders: list[PrivateReminder],
        allowed_user_ids: set[int],
        now_provider: Callable[[], datetime] | None = None,
        poll_interval_seconds: float = 30.0,
    ) -> None:
        self.sender = sender
        self.data_dir = data_dir.resolve()
        self.reminders = list(reminders)
        self.allowed_user_ids = set(allowed_user_ids)
        self.now_provider = now_provider or (lambda: datetime.now().astimezone())
        self.poll_interval_seconds = poll_interval_seconds
        self._stop_event = asyncio.Event()
        self._worker_task: asyncio.Task | None = None

    @property
    def state_path(self) -> Path:
        return self.data_dir / "private_reminders_state.json"

    async def start(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        await self._run_due_reminders_once()
        if not self.reminders:
            return
        self._stop_event.clear()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._stop_event.set()
        await self._worker_task
        self._worker_task = None

    async def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval_seconds)
            except asyncio.TimeoutError:
                await self._run_due_reminders_once()

    async def _run_due_reminders_once(self) -> None:
        state = self._load_state()
        now = self.now_provider()
        dirty = False
        for reminder in self.reminders:
            if reminder.reminder_id in state:
                continue
            if reminder.user_id not in self.allowed_user_ids:
                logger.warning(
                    "private_reminder_skipped_not_allowed reminder_id=%s user_id=%s",
                    reminder.reminder_id,
                    reminder.user_id,
                )
                state[reminder.reminder_id] = "not-allowed"
                dirty = True
                continue
            if now < reminder.run_at:
                continue
            if now > reminder.run_at and not reminder.catch_up_if_missed:
                state[reminder.reminder_id] = "missed"
                dirty = True
                continue

            await self.sender.send_private_text(
                OutboundPrivateMessage(user_id=reminder.user_id, text=reminder.text)
            )
            state[reminder.reminder_id] = "sent"
            dirty = True
            logger.info(
                "private_reminder_sent reminder_id=%s user_id=%s run_at=%s",
                reminder.reminder_id,
                reminder.user_id,
                reminder.run_at.isoformat(),
            )
        if dirty:
            self._write_state(state)

    def _load_state(self) -> dict[str, str]:
        if not self.state_path.exists():
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("private_reminder_state_read_failed path=%s", self.state_path)
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): str(value) for key, value in payload.items() if str(key).strip()}

    def _write_state(self, state: dict[str, str]) -> None:
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
