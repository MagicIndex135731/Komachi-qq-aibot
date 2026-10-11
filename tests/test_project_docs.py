from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts import check_project_docs as guard


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _entry(path: str, role: str = "reference", **extra: object) -> dict:
    return {"path": path, "role": role, "max_chars": guard.ROLE_BUDGETS[role],
            "purpose": "Fixture documentation", "maintainer": "Fixture owner", **extra}


def _save_manifest(root: Path, manifest: dict) -> None:
    _write(root / guard.MANIFEST_PATH, json.dumps(manifest, ensure_ascii=False))


def _fixture(root: Path) -> dict:
    archive_path = "data/project-docs-history/PROJECT_DOCS_HISTORY_20261009.zip"
    archive = root / archive_path
    archive.parent.mkdir(parents=True)
    members = {".trellis/tasks/old.md": b"original historical bytes\r\n"}
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, content in members.items():
            zipped.writestr(name, content)
    metadata_path = "data/project-docs-history/manifest.json"
    _write(root / metadata_path, json.dumps({
        "zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "entries": [{"path": name, "sha256": hashlib.sha256(content).hexdigest()}
                    for name, content in members.items()],
    }))
    _write(root / "README.md", "[Rules](AGENTS.md)\n")
    _write(root / "AGENTS.md", "Current project rules\n")
    _write(root / "docs/project/当前状态.md", "Current state\n")
    manifest = {
        "schema_version": 1,
        "entries": [_entry("README.md", "entry"), _entry("AGENTS.md", "rules"),
                    _entry("docs/project/当前状态.md", "current")],
        "archive": {"path": archive_path, "manifest": metadata_path},
    }
    _save_manifest(root, manifest)
    return manifest


def _audit(root: Path, tasks: list[dict] | None = None) -> dict:
    return guard.audit_project(root, task_reader=lambda _: tasks or [])


def _task(root: Path, manifest: dict, *, status: str = "To Do", labels: str = "[]",
          documentation: str = "docs/project/当前状态.md") -> dict:
    path = "backlog/tasks/qq-1 - Fixture.md"
    _write(root / path, f"---\nid: QQ-1\nstatus: {status}\nlabels: {labels}\n"
           f"references:\n  - README.md\ndocumentation:\n  - '{documentation}'\n---\nTask\n")
    manifest["entries"].append(_entry(path, "task"))
    _save_manifest(root, manifest)
    return {"id": "QQ-1", "status": status, "labels": json.loads(labels), "references": ["README.md"]}


def test_valid_documents_tasks_and_archive_pass(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    task = _task(tmp_path, manifest)
    result = _audit(tmp_path, [task])
    assert result["status"] == "PASS", result
    assert result["stats"]["backlog_documentation_checked"] == 1
    assert result["stats"]["archived_originals_verified"] == 1


def test_links_handle_spaces_percent_encoding_parentheses_and_ignore_code(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    name = "docs/project/图 (测试(一)).md"
    _write(tmp_path / name, "Reference\n")
    manifest["entries"].append(_entry(name))
    _save_manifest(tmp_path, manifest)
    _write(tmp_path / "README.md", "[a](<docs/project/图 (测试(一)).md>)\n"
           "[b](docs/project/图%20(测试(一)).md#anchor)\n"
           "[c](<docs/project/图 (测试(一)).md> \"title\")\n"
           "[d]: <docs/project/图 (测试(一)).md>\n"
           "[external](https://example.invalid/anything) [anchor](#missing)\n"
           "```md\n[ignored](missing.md)\n```\n~~~\n[ignored](also-missing.md)\n~~~\n"
           "    [ignored](indented.md)\n`[ignored](inline.md)`\n")
    result = _audit(tmp_path)
    assert result["status"] == "PASS", result
    assert result["stats"]["links_checked"] == 4


def test_broken_link_and_unregistered_document_fail(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _write(tmp_path / "README.md", "[broken](docs/project/missing.md)\n")
    _write(tmp_path / "docs/project/loose.md", "Unregistered\n")
    result = _audit(tmp_path)
    assert result["status"] == "FAIL"
    assert any("broken link" in error for error in result["errors"])
    assert any("unregistered" in error for error in result["errors"])


@pytest.mark.parametrize("bad_path", ["../escape.md", "C:/outside.md", "/outside.md"])
def test_escaping_registry_paths_fail(tmp_path: Path, bad_path: str) -> None:
    manifest = _fixture(tmp_path)
    manifest["entries"].append(_entry(bad_path))
    _save_manifest(tmp_path, manifest)
    assert any("invalid documentation path" in error for error in _audit(tmp_path)["errors"])


def test_duplicate_paths_and_budget_errors_are_reported(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    manifest["entries"].append(_entry("./README.md", "entry", max_chars=5000))
    manifest["entries"][0]["max_chars"] = 2
    _save_manifest(tmp_path, manifest)
    errors = _audit(tmp_path)["errors"]
    assert any("duplicate" in error for error in errors)
    assert any("exception reason" in error for error in errors)
    assert any("budget exceeded" in error for error in errors)


def test_budget_exception_requires_reason_but_never_waives_declared_budget(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    manifest["entries"][0].update(max_chars=5000, exception_reason="Reviewed expanded entry")
    _save_manifest(tmp_path, manifest)
    assert _audit(tmp_path)["status"] == "PASS"
    _write(tmp_path / "README.md", "x" * 5001)
    assert any("budget exceeded" in error for error in _audit(tmp_path)["errors"])


def test_backlog_cli_failure_is_not_silently_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fixture(tmp_path)
    monkeypatch.setattr(guard.shutil, "which", lambda _: "backlog.cmd")
    monkeypatch.setattr(guard.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 7, "", "failure"))
    result = guard.audit_project(tmp_path)
    assert result["status"] == "FAIL"
    assert any("CLI failed (exit 7)" in error for error in result["errors"])


@pytest.mark.parametrize("stdout", ["not JSON", '{}', '{"tasks": [1]}'])
def test_invalid_backlog_json_is_a_failed_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              stdout: str) -> None:
    _fixture(tmp_path)
    monkeypatch.setattr(guard.shutil, "which", lambda _: "backlog.cmd")
    monkeypatch.setattr(guard.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout, ""))
    assert any("invalid task JSON" in error for error in guard.audit_project(tmp_path)["errors"])


def test_missing_backlog_cli_is_a_failed_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fixture(tmp_path)
    monkeypatch.setattr(guard.shutil, "which", lambda _: None)
    assert any("CLI is unavailable" in error for error in guard.audit_project(tmp_path)["errors"])


@pytest.mark.parametrize("labels, expected", [('["paused"]', "paused label/status mismatch"),
                                            ('["historical"]', "historical task cannot")])
def test_invalid_paused_and_historical_states_fail(tmp_path: Path, labels: str, expected: str) -> None:
    manifest = _fixture(tmp_path)
    task = _task(tmp_path, manifest, status="In Progress", labels=labels)
    assert any(expected in error for error in _audit(tmp_path, [task])["errors"])


def test_backlog_reference_and_front_matter_documentation_must_exist(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    task = _task(tmp_path, manifest, documentation="docs/missing.md")
    task["references"] = ["missing-reference.py"]
    errors = _audit(tmp_path, [task])["errors"]
    assert any("missing Backlog references" in error for error in errors)
    assert any("missing Backlog documentation" in error for error in errors)


@pytest.mark.parametrize("relative", [".trellis/returned.txt", ".agents/skills/trellis-start/SKILL.md",
                                     ".codex/agents/trellis-check.md", ".codex/hooks/context.py",
                                     ".codex/hooks.json"])
def test_retired_framework_paths_cannot_return(tmp_path: Path, relative: str) -> None:
    _fixture(tmp_path)
    _write(tmp_path / relative, "Retired\n")
    assert any("retired framework path returned" in error for error in _audit(tmp_path)["errors"])


def test_current_documents_cannot_reference_old_framework(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _write(tmp_path / "AGENTS.md", "Run the .trellis/scripts/old.py tool\n")
    assert any("obsolete framework reference" in error for error in _audit(tmp_path)["errors"])


@pytest.mark.parametrize("corruption", ["zip-hash", "member-hash", "member-set", "bad-zip", "crc"])
def test_archive_corruption_fails(tmp_path: Path, corruption: str) -> None:
    manifest = _fixture(tmp_path)
    archive = tmp_path / manifest["archive"]["path"]
    meta_path = tmp_path / manifest["archive"]["manifest"]
    metadata = json.loads(meta_path.read_text())
    if corruption == "zip-hash":
        metadata["zip_sha256"] = "0" * 64
    elif corruption == "member-hash":
        metadata["entries"][0]["sha256"] = "0" * 64
    elif corruption == "member-set":
        with zipfile.ZipFile(archive, "a") as zipped:
            zipped.writestr("extra.md", "Extra")
        metadata["zip_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    elif corruption == "bad-zip":
        archive.write_bytes(b"not a ZIP")
    else:
        raw = archive.read_bytes().replace(b"original historical bytes", b"changed! historical bytes")
        archive.write_bytes(raw)
        metadata["zip_sha256"] = hashlib.sha256(raw).hexdigest()
    _write(meta_path, json.dumps(metadata))
    errors = _audit(tmp_path)["errors"]
    assert errors
    assert any("archive" in error or "ZIP" in error for error in errors)


def test_report_stays_inside_workspace_and_main_is_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture) -> None:
    _fixture(tmp_path)
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    monkeypatch.setattr(guard, "read_backlog_tasks", lambda _: [])
    assert guard.main(["--report", "data/report.json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    assert json.loads((tmp_path / "data/report.json").read_text())["status"] == "PASS"
    assert guard.main(["--report", str(tmp_path / "data/absolute-report.json")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    assert guard.main(["--report", "../escaped-report.json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"
    assert not (tmp_path.parent / "escaped-report.json").exists()
