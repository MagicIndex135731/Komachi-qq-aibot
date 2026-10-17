from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import check_github_sensitive as guard


def _git(root: Path, *args: str) -> None:
    process = subprocess.run(["git", *args], cwd=root, capture_output=True)
    assert process.returncode == 0, process.stderr.decode(errors="replace")


def _write(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _repo(root: Path) -> None:
    _git(root, "init", "-q")


def test_clean_tracked_and_staged_content_passes(tmp_path: Path) -> None:
    _repo(tmp_path)
    _write(tmp_path, "app/含 空格.py", "print('safe')\n")
    _git(tmp_path, "add", "--", "app/含 空格.py")
    result = guard.audit_repository(tmp_path)
    assert result["status"] == "PASS", result
    assert result["stats"]["staged_blobs_checked"] == 1


def test_staged_secret_is_found_even_after_worktree_is_scrubbed(tmp_path: Path) -> None:
    _repo(tmp_path)
    secret = "fixture-private-value-42"
    _write(tmp_path, "data/project-tools/.secrets-blocklist", f"# Fixture\n{secret}\n")
    file = _write(tmp_path, "app/config.py", secret)
    _git(tmp_path, "add", "--", "app/config.py")
    file.write_text("safe worktree", encoding="utf-8")
    result = guard.audit_repository(tmp_path)
    assert {item["source"] for item in result["violations"]} == {"staged-blob"}
    assert secret not in json.dumps(result)


def test_worktree_secret_is_found_after_clean_content_is_staged(tmp_path: Path) -> None:
    _repo(tmp_path)
    secret = "fixture-worktree-value-43"
    _write(tmp_path, "data/project-tools/.secrets-blocklist", secret)
    file = _write(tmp_path, "app/config.py", "safe staged content")
    _git(tmp_path, "add", "--", "app/config.py")
    file.write_text(secret, encoding="utf-8")
    result = guard.audit_repository(tmp_path)
    assert {item["source"] for item in result["violations"]} == {"worktree"}
    assert secret not in json.dumps(result)


@pytest.mark.parametrize("relative", [".env", "infra/wsl/.env", "configs/groups.local.yaml",
                                     "configs/social.local.yaml",
                                     "data/nested/file.txt", "cache.db", "backup.bundle", "AGENTS.md",
                                     "backlog/tasks/task.md", "docs/project/当前状态.md",
                                     ".agents/skills/helper/SKILL.md", ".codex/config.toml", ".trellis/config.json"])
def test_private_paths_cannot_be_published(tmp_path: Path, relative: str) -> None:
    _repo(tmp_path)
    _write(tmp_path, relative, "safe content")
    _git(tmp_path, "add", "--", relative)
    result = guard.audit_repository(tmp_path)
    assert result["status"] == "FAIL"
    assert any(item["rule"] == "private-path" for item in result["violations"])


def test_generic_secrets_are_detected_without_printing_values(tmp_path: Path) -> None:
    _repo(tmp_path)
    api_key = "sk-" + "A" * 24
    password = "fixture-password-44"
    _write(tmp_path, "app/settings.py", api_key + "\nNAPCAT_QUICK_PASSWORD" + "=" + password)
    _git(tmp_path, "add", "--", "app/settings.py")
    result = guard.audit_repository(tmp_path)
    assert {item["rule"] for item in result["violations"]} == {"api-key", "gateway-password"}
    serialized = json.dumps(result)
    assert api_key not in serialized
    assert password not in serialized


def test_nul_path_parser_does_not_split_newlines_or_quotes() -> None:
    assert guard._split_paths(b'line\nbreak.py\0quoted" name.py\0') == ["line\nbreak.py", 'quoted" name.py']


@pytest.mark.parametrize("relative", ["configs/personas/member.yaml", "infra/wsl/.env.backup",
    "infra/wsl/runtime/login.json", "infra/wsl/private-secrets/x-session.json", ".test-tmp-run/report.txt", "cache.sqlite3", "cache.db-wal"])
def test_additional_private_runtime_paths_are_blocked(tmp_path: Path, relative: str) -> None:
    _repo(tmp_path)
    _write(tmp_path, relative, "private fixture")
    _git(tmp_path, "add", "--", relative)
    assert guard.audit_repository(tmp_path)["status"] == "FAIL"


def test_untracked_public_candidate_is_checked_before_staging(tmp_path: Path) -> None:
    _repo(tmp_path)
    _write(tmp_path, "app/new.py", "sk-" + "A" * 24)
    assert guard.audit_repository(tmp_path, include_untracked=True)["status"] == "FAIL"


def test_env_example_remains_publishable(tmp_path: Path) -> None:
    _repo(tmp_path)
    _write(tmp_path, "infra/wsl/.env.example", "LLM_API_KEY=\n")
    _git(tmp_path, "add", "--", "infra/wsl/.env.example")
    assert guard.audit_repository(tmp_path)["status"] == "PASS"


def test_git_failure_is_a_failed_audit(tmp_path: Path) -> None:
    result = guard.audit_repository(tmp_path)
    assert result["status"] == "FAIL"
    assert any("Git" in error for error in result["errors"])


def test_staged_blob_git_failure_is_not_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _repo(tmp_path)
    _write(tmp_path, "app/code.py", "safe")
    _git(tmp_path, "add", "--", "app/code.py")
    original = guard._git

    def failing_show(root: Path, args: list[str]) -> bytes:
        if args[0] == "show":
            raise guard.GitReadError("Git show failed (exit 128)")
        return original(root, args)

    monkeypatch.setattr(guard, "_git", failing_show)
    result = guard.audit_repository(tmp_path)
    assert result["status"] == "FAIL"
    assert any("staged blob unavailable" in error for error in result["errors"])


def test_main_outputs_json_and_nonzero_for_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(guard, "BLOCKLIST", tmp_path / "data/project-tools/.secrets-blocklist")
    assert guard.main([]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"
