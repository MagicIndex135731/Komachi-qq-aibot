"""Read-only guard for current project documentation, Backlog and archived bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = "docs/project/文档清单.json"
ROLE_BUDGETS = {
    "rules": 4500,
    "current": 6500,
    "entry": 4000,
    "task": 9000,
    "parent-task": 5000,
    "active-plan": 9000,
    "reference": 9000,
    "governance": 6000,
}


def relative_path(root: Path, value: Any) -> Path:
    """Resolve a project-relative path, rejecting traversal and symlink escape."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a nonempty string")
    normalized = value.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or re.match(r"^[A-Za-z]:", normalized) or ".." in relative.parts:
        raise ValueError("path must be relative and stay inside the workspace")
    candidate = (root / normalized).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError("path must stay inside the workspace")
    return candidate


def _without_code(text: str) -> str:
    lines = []
    fence: tuple[str, int] | None = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1]:
                fence = None
            continue
        if marker:
            fence = (marker[1][0], len(marker[1]))
        elif not line.startswith(("    ", "\t")):
            lines.append(line)
    return re.sub(r"(`+)[^\n]*?\1", "", "".join(lines))


def _destination(text: str, start: int) -> tuple[str, int] | None:
    """Read an inline destination while preserving balanced/escaped parentheses."""
    cursor = start
    depth = 0
    escaped = False
    angle = False
    while cursor < len(text) and text[cursor] != "\n":
        char = text[cursor]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "<" and cursor == start:
            angle = True
        elif char == ">" and angle:
            angle = False
        elif not angle and char == "(":
            depth += 1
        elif not angle and char == ")":
            if depth == 0:
                target = text[start:cursor].strip()
                if target.startswith("<"):
                    closing = target.find(">")
                    if closing < 0:
                        return None
                    target = target[1:closing]
                else:
                    target = re.sub(r"\s+[\"'][^\n]*[\"']\s*$", "", target)
                    target = re.sub(r"\s+\([^()]*\)\s*$", "", target)
                return re.sub(r"\\([()\[\]\\ ])", r"\1", target), cursor + 1
            depth -= 1
        cursor += 1
    return None


def link_targets(text: str) -> Iterator[str]:
    text = _without_code(text)
    for match in re.finditer(r"!?\[[^\]\n]*\]\(\s*", text):
        destination = _destination(text, match.end())
        if destination:
            yield destination[0]
    # Reference definitions also describe local links, including image references.
    for match in re.finditer(r"(?m)^\s{0,3}\[[^\]\n]+\]:\s*(.+)$", text):
        target = match[1].strip()
        if target.startswith("<"):
            closing = target.find(">")
            if closing >= 0:
                yield target[1:closing]
        else:
            yield re.sub(r"\s+[\"'][^\n]*[\"']\s*$", "", target)


def is_external(target: str) -> bool:
    return target.startswith("//") or bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target))


def resolve_link(root: Path, document: Path, target: str) -> Path | None:
    if not target or target.startswith("#") or is_external(target):
        return None
    decoded = unquote(target.split("#", 1)[0].split("?", 1)[0])
    decoded = re.sub(r":\d+$", "", decoded).replace("\\", "/")
    if re.match(r"^/?[A-Za-z]:", decoded):
        raise ValueError("absolute workspace link is not project-relative")
    candidate = (document.parent / decoded).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("link escapes the workspace")
    return candidate


def read_backlog_tasks(root: Path) -> list[dict[str, Any]]:
    command = shutil.which("backlog.cmd") or shutil.which("backlog")
    if command is None:
        raise RuntimeError("Backlog CLI is unavailable")
    try:
        process = subprocess.run(
            [command, "task", "list", "--json"], cwd=root, capture_output=True,
            text=True, encoding="utf-8", check=False, timeout=60,
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Backlog CLI could not be executed") from exc
    if process.returncode:
        raise RuntimeError(f"Backlog CLI failed (exit {process.returncode})")
    try:
        payload = json.loads(process.stdout)
        tasks = payload.get("tasks") if isinstance(payload, dict) else payload
        if not isinstance(tasks, list) or not all(isinstance(task, dict) for task in tasks):
            raise ValueError("tasks must be a list of objects")
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("Backlog CLI returned invalid task JSON") from exc
    return tasks


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _front_matter(text: str) -> dict[str, Any]:
    match = re.match(r"\A---\s*\r?\n(.*?)\r?\n---(?:\r?\n|\Z)", text, re.S)
    if not match:
        raise ValueError("Backlog task front matter is missing")
    payload = yaml.safe_load(match[1])
    if not isinstance(payload, dict):
        raise ValueError("Backlog task front matter must be an object")
    return payload


def _check_archive(root: Path, specification: Any, errors: list[str], stats: dict[str, Any]) -> None:
    try:
        if not isinstance(specification, dict):
            raise ValueError("archive metadata must be an object")
        archive = relative_path(root, specification.get("path"))
        metadata = relative_path(root, specification.get("manifest"))
        manifest = _load_json(metadata)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("entries"), list):
            raise ValueError("archive manifest entries must be a list")
        expected = {}
        for entry in manifest["entries"]:
            if not isinstance(entry, dict):
                raise ValueError("archive entries must be objects")
            member = entry.get("path")
            relative_path(root, member)
            if member in expected:
                raise ValueError(f"duplicate archive manifest member: {member}")
            digest = entry.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"invalid archive member checksum: {member}")
            expected[member] = digest
        if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest.get("zip_sha256"):
            errors.append("history ZIP checksum changed")
        with zipfile.ZipFile(archive) as zipped:
            members = [entry.filename for entry in zipped.infolist() if not entry.is_dir()]
            if len(members) != len(set(members)):
                errors.append("history ZIP has duplicate members")
            if set(members) != set(expected):
                errors.append("history ZIP member set differs from archive manifest")
            for member in members:
                relative_path(root, member)
            corrupt = zipped.testzip()
            if corrupt is not None:
                errors.append(f"history ZIP CRC failed: {corrupt}")
            for member, digest in expected.items():
                if member in members and hashlib.sha256(zipped.read(member)).hexdigest() != digest:
                    errors.append(f"archived original bytes differ: {member}")
        stats["archived_originals_verified"] = len(expected)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError, zlib.error) as exc:
        errors.append(f"history archive verification failed: {exc}")


def audit_project(
    root: Path, *, task_reader: Callable[[Path], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    stats: dict[str, Any] = {
        "documents": 0, "links_checked": 0, "tasks": 0,
        "backlog_refs_checked": 0, "backlog_documentation_checked": 0,
    }
    registered: set[str] = set()
    task_documents: dict[str, dict[str, Any]] = {}
    try:
        manifest = _load_json(root / MANIFEST_PATH)
        if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            raise ValueError("documentation manifest schema_version must be 1")
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise ValueError("documentation manifest entries must be a list")
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"documentation manifest read failed: {exc}")
        manifest, entries = {}, []
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("documentation manifest entry must be an object")
            continue
        relative = entry.get("path")
        try:
            path = relative_path(root, relative)
        except ValueError as exc:
            errors.append(f"invalid documentation path: {relative!r}: {exc}")
            continue
        identity = path.relative_to(root).as_posix().casefold()
        if identity in registered:
            errors.append(f"duplicate documentation registry path: {relative}")
        registered.add(identity)
        role, budget = entry.get("role"), entry.get("max_chars")
        if not isinstance(role, str) or role not in ROLE_BUDGETS:
            errors.append(f"invalid documentation role: {relative}")
        if not all(isinstance(entry.get(field), str) and entry[field].strip() for field in ("purpose", "maintainer")):
            errors.append(f"documentation purpose/maintainer incomplete: {relative}")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            errors.append(f"invalid character budget: {relative}")
            budget = None
        elif budget > ROLE_BUDGETS.get(role if isinstance(role, str) else "", 9000):
            reason = entry.get("exception_reason")
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"budget exception reason missing: {relative}")
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            errors.append(f"registered document unreadable: {relative}: {exc}")
            continue
        stats["documents"] += 1
        if budget is not None and len(text) > budget:
            errors.append(f"character budget exceeded: {relative} = {len(text)} > {budget}")
        if re.search(r"\.trellis[/\\]", text, re.I):
            errors.append(f"obsolete framework reference in current document: {relative}")
        for target in link_targets(text):
            try:
                actual = resolve_link(root, path, target)
                if actual is not None:
                    stats["links_checked"] += 1
                    if not actual.exists():
                        errors.append(f"broken link: {relative} -> {target}")
            except (OSError, ValueError) as exc:
                errors.append(f"invalid local link: {relative} -> {target}: {exc}")
        if role in ("task", "parent-task") or identity.startswith("backlog/tasks/"):
            try:
                metadata = _front_matter(text)
                task_id = metadata.get("id")
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError("task id is missing")
                if task_id in task_documents:
                    errors.append(f"duplicate Backlog task document id: {task_id}")
                task_documents[task_id] = metadata
            except (ValueError, yaml.YAMLError) as exc:
                errors.append(f"invalid Backlog task document: {relative}: {exc}")
    candidates = list(root.glob("*.md"))
    for directory in ("docs", "backlog/tasks", "backlog/docs"):
        candidates.extend((root / directory).rglob("*"))
    for path in candidates:
        if path.is_file() and path.suffix.lower() in (".md", ".markdown", ".mdown", ".mdx"):
            relative = path.relative_to(root).as_posix()
            if relative.casefold() not in registered:
                errors.append(f"unregistered project Markdown: {relative}")
    forbidden = [".trellis", ".codex/hooks", ".codex/hooks.json", "hooks.json"]
    forbidden.extend(path.relative_to(root).as_posix() for path in (root / ".agents/skills").glob("trellis-*"))
    forbidden.extend(path.relative_to(root).as_posix() for path in (root / ".codex/agents").glob("trellis-*"))
    for relative in forbidden:
        if (root / relative).exists():
            errors.append(f"retired framework path returned: {relative}")
    try:
        tasks = (task_reader or read_backlog_tasks)(root)
        if not isinstance(tasks, list) or not all(isinstance(task, dict) for task in tasks):
            raise ValueError("Backlog tasks must be a list of objects")
    except (RuntimeError, OSError, ValueError) as exc:
        errors.append(f"Backlog JSON read failed: {exc}")
        tasks = []
    stats["tasks"] = len(tasks)
    for task in tasks:
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            errors.append("Backlog task id is missing")
            continue
        metadata = task_documents.get(task_id)
        if metadata is None:
            errors.append(f"Backlog task document missing: {task_id}")
            metadata = {}
        status = task.get("status", metadata.get("status"))
        labels = task.get("labels", metadata.get("labels", []))
        if not isinstance(labels, list):
            errors.append(f"Backlog labels must be a list: {task_id}")
            labels = []
        if "paused" in labels and status != "To Do":
            errors.append(f"paused label/status mismatch: {task_id}")
        if "historical" in labels and status == "In Progress":
            errors.append(f"historical task cannot be In Progress: {task_id}")
        for field, counter in (("references", "backlog_refs_checked"), ("documentation", "backlog_documentation_checked")):
            values = task.get(field, [])
            recorded = metadata.get(field, [])
            if not isinstance(values, list) or not isinstance(recorded, list):
                errors.append(f"Backlog {field} must be a list: {task_id}")
                continue
            for value in set(values + recorded) if all(isinstance(value, str) for value in values + recorded) else []:
                if is_external(value) or value.startswith("#"):
                    continue
                stats[counter] += 1
                try:
                    candidate = relative_path(root, unquote(value.split("#", 1)[0]))
                    if not candidate.exists():
                        errors.append(f"missing Backlog {field}: {task_id} -> {value}")
                except ValueError as exc:
                    errors.append(f"invalid Backlog {field}: {task_id} -> {value}: {exc}")
            if not all(isinstance(value, str) for value in values + recorded):
                errors.append(f"Backlog {field} values must be strings: {task_id}")
    _check_archive(root, manifest.get("archive"), errors, stats)
    return {"status": "FAIL" if errors else "PASS", "stats": stats, "errors": sorted(set(errors))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", help="Optional JSON report path inside this workspace")
    args = parser.parse_args(argv)
    result = audit_project(ROOT)
    if args.report:
        try:
            target = (ROOT / args.report).resolve()
            if target == ROOT.resolve() or not target.is_relative_to(ROOT.resolve()):
                raise ValueError("report path must stay inside the workspace")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except (OSError, ValueError) as exc:
            result["errors"].append(f"report write failed: {exc}")
            result["status"] = "FAIL"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
