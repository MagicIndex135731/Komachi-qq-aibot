from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path


class RuntimeHeartbeat:
    def __init__(self, *, heartbeat_file: Path, interval_seconds: float = 5.0) -> None:
        self.heartbeat_file = heartbeat_file
        self.interval_seconds = max(1.0, float(interval_seconds))
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
        self._write(state="starting")
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._write(state="stopped")

    async def _run(self) -> None:
        try:
            while True:
                self._write(state="alive")
                await asyncio.sleep(self.interval_seconds)
        except asyncio.CancelledError:
            raise

    def _write(self, *, state: str) -> None:
        payload = {
            "pid": os.getpid(),
            "state": state,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self.heartbeat_file.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
