from pathlib import Path

import pytest

from app.dev_control.checkpoints import create_repo_checkpoint, restore_repo_checkpoint
from app.dev_control.codex_bridge import CodexBridge, _extract_thread_id, _parse_last_message


def test_codex_bridge_parses_json_last_message(tmp_path) -> None:
    output_file = tmp_path / "last.json"
    output_file.write_text('{"summary":"done","reply_text":"ok","restart_required":false}', encoding="utf-8")

    result = _parse_last_message(output_file)

    assert result.summary == "done"
    assert result.reply_text == "ok"
    assert result.restart_required is False
    assert result.thread_id == ""


def test_codex_bridge_extracts_thread_id_from_jsonl_output() -> None:
    stdout_text = '\n'.join(
        [
            "warn line",
            '{"type":"thread.started","thread_id":"thread-123"}',
            '{"type":"turn.started"}',
        ]
    )

    assert _extract_thread_id(stdout_text) == "thread-123"


def test_checkpoint_restore_removes_new_files_and_restores_old_content(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "file.txt").write_text("before", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoint"

    manifest = create_repo_checkpoint(repo_root=repo_root, checkpoint_dir=checkpoint_dir)
    (repo_root / "file.txt").write_text("after", encoding="utf-8")
    (repo_root / "new.txt").write_text("new", encoding="utf-8")

    restore_repo_checkpoint(repo_root=repo_root, checkpoint_dir=checkpoint_dir, manifest=manifest)

    assert (repo_root / "file.txt").read_text(encoding="utf-8") == "before"
    assert not (repo_root / "new.txt").exists()


def test_checkpoint_ignores_runtime_data_directory(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "file.txt").write_text("before", encoding="utf-8")
    (repo_root / "data").mkdir()
    (repo_root / "data" / "bot.db").write_text("live-db", encoding="utf-8")

    manifest = create_repo_checkpoint(repo_root=repo_root, checkpoint_dir=tmp_path / "checkpoint")

    assert manifest["files"] == ["file.txt"]


def test_codex_bridge_prefers_explicit_env_path(monkeypatch, tmp_path) -> None:
    fake_codex = tmp_path / "codex.exe"
    fake_codex.write_text("", encoding="utf-8")
    monkeypatch.setenv("CODEX_EXECUTABLE", str(fake_codex))
    monkeypatch.setattr("app.dev_control.codex_bridge.shutil.which", lambda name: None)

    bridge = CodexBridge()

    assert Path(bridge.codex_executable) == fake_codex


def test_codex_bridge_uses_vscode_extension_fallback_when_path_missing(monkeypatch, tmp_path) -> None:
    vscode_root = tmp_path / ".vscode" / "extensions" / "openai.chatgpt-test" / "bin" / "windows-x86_64"
    vscode_root.mkdir(parents=True)
    fake_codex = vscode_root / "codex.exe"
    fake_codex.write_text("", encoding="utf-8")
    monkeypatch.delenv("CODEX_EXECUTABLE", raising=False)
    monkeypatch.setattr("app.dev_control.codex_bridge.shutil.which", lambda name: None)
    monkeypatch.setattr("app.dev_control.codex_bridge.Path.home", lambda: tmp_path)

    bridge = CodexBridge()

    assert Path(bridge.codex_executable) == fake_codex


def test_codex_bridge_raises_when_no_candidate_exists(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CODEX_EXECUTABLE", raising=False)
    monkeypatch.setattr("app.dev_control.codex_bridge.shutil.which", lambda name: None)
    monkeypatch.setattr("app.dev_control.codex_bridge.Path.home", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="codex executable not found"):
        CodexBridge()


def test_codex_bridge_writes_utf8_prompt_to_subprocess(monkeypatch, tmp_path) -> None:
    fake_codex = tmp_path / "codex.exe"
    fake_codex.write_text("", encoding="utf-8")
    bridge = CodexBridge.__new__(CodexBridge)
    bridge.timeout_seconds = 30
    bridge.codex_executable = str(fake_codex)
    calls: dict[str, object] = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text('{"summary":"done","reply_text":"ready","restart_required":false}', encoding="utf-8")
        return type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": '{"type":"thread.started","thread_id":"thread-123"}\n',
                "stderr": "",
            },
        )()

    monkeypatch.setattr("app.dev_control.codex_bridge.subprocess.run", fake_run)

    result = bridge.run_task(
        prompt="hello xiaomachi",
        repo_root=tmp_path,
        artifact_dir=tmp_path / "artifacts",
    )

    assert result.reply_text == "ready"
    assert result.thread_id == "thread-123"
    assert calls["kwargs"]["encoding"] == "utf-8"
    assert calls["kwargs"]["input"] == "hello xiaomachi"


def test_codex_bridge_uses_exec_resume_when_resume_thread_id_provided(monkeypatch, tmp_path) -> None:
    fake_codex = tmp_path / "codex.exe"
    fake_codex.write_text("", encoding="utf-8")
    bridge = CodexBridge.__new__(CodexBridge)
    bridge.timeout_seconds = 30
    bridge.codex_executable = str(fake_codex)
    calls: dict[str, object] = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text('{"summary":"done","reply_text":"resumed","restart_required":false}', encoding="utf-8")
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("app.dev_control.codex_bridge.subprocess.run", fake_run)

    result = bridge.run_task(
        prompt="continue",
        repo_root=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        resume_thread_id="thread-42",
    )

    assert result.reply_text == "resumed"
    assert result.thread_id == "thread-42"
    assert calls["command"][:3] == [str(fake_codex), "exec", "resume"]
    assert "thread-42" in calls["command"]
