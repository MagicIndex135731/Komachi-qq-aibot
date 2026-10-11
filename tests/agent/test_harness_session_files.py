"""Session I/O covers paths written by Node beyond Windows MAX_PATH."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.agent.harness_session_files import native_file_path, native_session_logs, native_session_size


def long_session_home(root: Path) -> Path:
    # Every individual component remains below the filesystem's name limit.
    # The full native log path, including cwd-key and session ID, exceeds 260.
    home = root.resolve()
    while len(str(home)) < 205:
        home /= "隔离会话-" + "x" * min(48, 206 - len(str(home)))
    return home


def test_native_logs_beyond_windows_max_path_keep_logical_identity_and_real_size(tmp_path):
    home = long_session_home(tmp_path)
    session_id = "a" * 32
    path = home / "sessions" / "encoded-cwd-key" / session_id / "session.v3.jsonl"
    assert len(str(path)) > 260
    native_file_path(path.parent).mkdir(parents=True)
    payload = '{"type":"message","text":"跨醒回执"}\n'
    native_file_path(path).write_bytes(payload.encode("utf-8"))

    discovered = native_session_logs(home, session_id)
    assert discovered == (path,)
    assert not str(discovered[0]).startswith("\\\\?\\")
    assert native_session_size(discovered) == len(payload.encode("utf-8"))
    assert native_file_path(discovered[0]).read_text(encoding="utf-8") == payload
    assert native_session_logs(home, "different-session") == ()

    # A removed file is a race rather than a reason to drop the prior turn.
    native_file_path(path).unlink()
    assert native_session_size(discovered) == 0


@pytest.mark.parametrize("session_id", ["", ".", "..", "../other", "other\\child", "*", "[other]"])
def test_native_session_discovery_rejects_path_or_glob_injection(tmp_path, session_id):
    with pytest.raises(ValueError, match="single path component"):
        native_session_logs(tmp_path, session_id)


def test_native_session_size_does_not_swallow_access_failures(tmp_path, monkeypatch):
    path = tmp_path / "session.v3.jsonl"
    def deny(_):
        raise PermissionError("synthetic access denied")
    monkeypatch.setattr("app.agent.harness_session_files.native_file_path", deny)
    with pytest.raises(PermissionError, match="access denied"):
        native_session_size((path,))


@pytest.mark.skipif(os.name != "nt", reason="Windows extended path syntax")
def test_windows_native_io_handles_unc_and_is_idempotent():
    unc = native_file_path(r"\\synthetic-server\share\会话\session.v3.jsonl")
    assert str(unc) == r"\\?\UNC\synthetic-server\share\会话\session.v3.jsonl"
    assert native_file_path(unc) == unc
