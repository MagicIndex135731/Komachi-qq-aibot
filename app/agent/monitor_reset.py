"""Optional, durable reset boundaries for the private monitor projection."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import threading
from uuid import uuid4


logger = logging.getLogger(__name__)
_LOCK = threading.Lock()


def _path(root: Path, surface: str, scope_id: int) -> Path:
    if surface not in {'group', 'private'} or type(scope_id) is not int or scope_id <= 0:
        raise ValueError('monitor_reset_scope_invalid')
    if root.is_symlink():
        raise ValueError('monitor_reset_path_invalid')
    key = hashlib.sha256(f'{surface}:{scope_id}'.encode()).hexdigest()[:24]
    path = root / f'reset-{key}.json'
    if path.is_symlink():
        raise ValueError('monitor_reset_path_invalid')
    return path


def read_reset(root: Path, surface: str, scope_id: int) -> dict | None:
    try:
        path = _path(root, surface, scope_id)
        if path.stat().st_size > 4096:
            return None
        value = json.loads(path.read_bytes())
        if (isinstance(value, dict) and value.get('version') == 1
                and value.get('surface') == surface and value.get('scope_id') == scope_id
                and type(value.get('source_id')) is int and value['source_id'] > 0
                and type(value.get('reset_at')) in (int, float)
                and math.isfinite(value['reset_at']) and value['reset_at'] > 0):
            return value
    except (OSError, ValueError, TypeError):
        pass
    return None


def record_context_reset(settings, *, surface: str, scope_id: int,
                         source_id: int, reset_at: float) -> None:
    """Called after SQL commit, before awaiting cancellation or confirmation.

    Observation failure never changes the successful execution reset. Group and
    private writers own distinct scopes; atomic replace is visible across the
    bot processes, the read-only monitor container and Windows fallback viewer.
    """
    if not getattr(settings, 'agent_monitor_enabled', False):
        return
    temporary = None
    try:
        if type(source_id) is not int or source_id <= 0 or not math.isfinite(reset_at) or reset_at <= 0:
            raise ValueError('monitor_reset_boundary_invalid')
        root = Path(settings.data_dir) / 'agent_monitor'
        with _LOCK:
            path = _path(root, surface, scope_id)
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            previous = read_reset(root, surface, scope_id)
            if previous and previous['source_id'] >= source_id:
                return
            value = {'version':1, 'surface':surface, 'scope_id':scope_id,
                'source_id':source_id, 'reset_at':reset_at}
            temporary = root / f'.reset-{uuid4().hex}.tmp'
            with temporary.open('x', encoding='utf8') as handle:
                json.dump(value, handle)
            os.replace(temporary, path)
    except (OSError, ValueError, TypeError):
        logger.warning('agent_monitor_reset_unavailable')
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
