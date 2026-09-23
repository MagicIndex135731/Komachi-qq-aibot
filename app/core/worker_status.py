"""Small best-effort heartbeat for periodic background workers."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)


def write_worker_status(log_dir: Path, name: str, state: str, interval_seconds: float) -> None:
    """Record scheduler progress without writing message text or identifiers."""

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{name}.worker.json"
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "state": state,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "interval_seconds": float(interval_seconds),
                }
            ),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    except OSError:
        logger.exception("worker_status_write_failed name=%s", name)
