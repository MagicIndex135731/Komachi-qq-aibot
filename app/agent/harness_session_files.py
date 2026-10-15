"""Native session file I/O without changing Harness session identities.

Node can create a session JSONL whose full path exceeds Windows MAX_PATH.
Python must use the extended path syntax at the I/O boundary as well.  Keep
ordinary absolute paths in checkpoints so storage scope checks, SDK inputs
and deployments on other platforms continue to use the same identities.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def native_file_path(path: str | Path) -> Path:
    """Return an I/O path; callers must not persist its Windows prefix."""
    value = os.fspath(path)
    if os.name != "nt":
        return Path(value)
    value = os.path.abspath(value)
    if value.startswith("\\\\?\\"):
        return Path(value)
    if value.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + value[2:])
    return Path("\\\\?\\" + value)


def _logical_path(path: Path) -> Path:
    value = os.fspath(path)
    if os.name == "nt":
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
    return Path(value)


def native_session_logs(home: str | Path, session_id: str) -> tuple[Path, ...]:
    """Discover one session's JSONL and return ordinary logical paths."""
    if (not isinstance(session_id, str) or not session_id or session_id in {".", ".."}
            or any(character in session_id for character in "/\\*?[]")):
        raise ValueError("native session ID must be a single path component")
    pattern = f"sessions/*/{session_id}/session.v3.jsonl"
    return tuple(sorted((_logical_path(path) for path in native_file_path(home).glob(pattern)),
                        key=os.fspath))


def native_session_size(paths: Iterable[Path]) -> int:
    """Read real log sizes; tolerate only a file removed after discovery."""
    size = 0
    for path in paths:
        try:
            size += native_file_path(path).stat().st_size
        except FileNotFoundError:
            continue
    return size


def snapshot_native_session(home, session_id):
    return {str(path): (native_file_path(path).read_bytes()
            if native_file_path(path).stat().st_size <= 16 * 1024 * 1024 else None)
            for path in native_session_logs(home, session_id)}


def discard_silent_native_turn(home, session_id, previous):
    """Only after the SDK has stopped, and only its exact append-only suffix."""
    root = native_file_path(home).resolve()
    for path in native_session_logs(home, session_id):
        target = native_file_path(path)
        if not target.resolve().is_relative_to(root):
            raise ValueError('native session escaped its home')
        before = previous.get(str(path))
        if str(path) not in previous:
            target.unlink()
        elif before is not None and target.read_bytes().startswith(before):
            with target.open('r+b') as stream:
                stream.truncate(len(before))
