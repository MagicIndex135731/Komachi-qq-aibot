"""Fail before GitHub publication if tracked or staged files contain private material."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCKLIST = REPO_ROOT / "data/project-tools/.secrets-blocklist"
PRIVATE_DIRECTORIES = (
    ".trellis/", "data/", "backlog/", "docs/project/", ".agents/", ".codex/",
    "configs/personas/", "infra/wsl/runtime/", "infra/wsl/.cache/", "infra/wsl/mihomo/",
    ".worktrees/", ".artifacts/", ".venv/", ".venv-wsl/", "release/",
)
GENERIC_PATTERNS = (
    ("api-key", re.compile(rb"sk-[A-Za-z0-9]{16,}")),
    ("gateway-password", re.compile(b"NAPCAT_QUICK_PASSWORD" + rb"=[^$\s\"'][^\s]*")),
    ("github-token", re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("private-key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
)


class GitReadError(RuntimeError):
    """A failed Git read must never become a successful empty audit."""


def _git(root: Path, args: list[str]) -> bytes:
    try:
        process = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    except OSError as exc:
        raise GitReadError(f"Git {args[0]} could not be executed") from exc
    if process.returncode:
        raise GitReadError(f"Git {args[0]} failed (exit {process.returncode})")
    return process.stdout


def _split_paths(raw: bytes) -> list[str]:
    return [os.fsdecode(path) for path in raw.split(b"\0") if path]


def banned_path(path: str) -> bool:
    lowered = path.replace("\\", "/").lower()
    parts = lowered.split("/")
    return (
        lowered == "agents.md"
        or lowered in tuple(directory.rstrip("/") for directory in PRIVATE_DIRECTORIES)
        or lowered.startswith(PRIVATE_DIRECTORIES)
        or any(part == ".env" or (part.startswith(".env.") and part != ".env.example") for part in parts)
        or "groups.local.yaml" in parts
        or "social.local.yaml" in parts
        or lowered.endswith((".local.yaml", ".db", ".db-wal", ".db-shm", ".sqlite", ".sqlite3", ".bundle"))
        or any(part == "__pycache__" or part.endswith(".egg-info") for part in parts)
        or any(part.startswith((".test-tmp", ".pytest", ".tmp", ".codex_tmp")) for part in parts)
    )


def _secrets(blocklist: Path) -> list[bytes]:
    if not blocklist.exists():
        return []
    return [
        line.strip().encode("utf-8")
        for line in blocklist.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _redact(value: str, secrets: list[bytes]) -> str:
    raw = value.encode("utf-8", errors="surrogateescape")
    for secret in sorted(secrets, key=len, reverse=True):
        raw = raw.replace(secret, b"[REDACTED]")
    for _, pattern in GENERIC_PATTERNS:
        raw = pattern.sub(b"[REDACTED]", raw)
    return raw.decode("utf-8", errors="replace")


def audit_repository(root: Path, *, blocklist: Path | None = None, include_untracked: bool = False) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    violations: list[dict[str, str]] = []
    stats = {"tracked_files": 0, "staged_files": 0, "worktree_files_checked": 0, "staged_blobs_checked": 0}
    secrets: list[bytes] = []
    try:
        secrets = _secrets(blocklist or root / "data/project-tools/.secrets-blocklist")
    except (OSError, UnicodeError):
        errors.append("Local secrets blocklist could not be read")
    try:
        top_level = Path(os.fsdecode(_git(root, ["rev-parse", "--show-toplevel"]).rstrip(b"\r\n"))).resolve()
        if top_level != root:
            raise GitReadError("Git audit root is not the repository top level")
        tracked = set(_split_paths(_git(root, ["ls-files", "-z"])))
        staged = set(_split_paths(_git(root, ["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMRTU"])))
        untracked = set(_split_paths(_git(root, ["ls-files", "--others", "--exclude-standard", "-z"]))) if include_untracked else set()
    except GitReadError as exc:
        errors.append(str(exc))
        tracked, staged, untracked = set(), set(), set()
    stats["tracked_files"], stats["staged_files"] = len(tracked), len(staged)

    def inspect(path: str, source: str, content: bytes) -> None:
        if any(secret in content for secret in secrets):
            violations.append({"path": path, "source": source, "rule": "local-blocklist"})
        for name, pattern in GENERIC_PATTERNS:
            if pattern.search(content):
                violations.append({"path": path, "source": source, "rule": name})

    for path in sorted(tracked | staged | untracked):
        if banned_path(path):
            violations.append({"path": path, "source": "git-path", "rule": "private-path"})
        actual = root / path
        try:
            if actual.is_symlink():
                content = os.readlink(actual).encode("utf-8", errors="surrogateescape")
            elif actual.is_file():
                content = actual.read_bytes()
            elif not actual.exists() or actual.is_dir():
                continue
            else:
                errors.append(f"Tracked worktree file could not be read: {path}")
                continue
            inspect(path, "worktree", content)
            stats["worktree_files_checked"] += 1
        except OSError:
            errors.append(f"Tracked worktree file could not be read: {path}")
    for path in sorted(staged):
        try:
            inspect(path, "staged-blob", _git(root, ["show", f":{path}"]))
            stats["staged_blobs_checked"] += 1
        except GitReadError as exc:
            errors.append(f"{exc}: staged blob unavailable for {path}")
    for violation in violations:
        violation["path"] = _redact(violation["path"], secrets)
    return {
        "status": "FAIL" if errors or violations else "PASS", "stats": stats,
        "errors": [_redact(error, secrets) for error in errors], "violations": violations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--blocklist", type=Path, default=None)
    parser.add_argument("--include-untracked", action="store_true")
    args = parser.parse_args(argv)
    blocklist = args.blocklist or (BLOCKLIST if args.root == REPO_ROOT else args.root / "data/project-tools/.secrets-blocklist")
    result = audit_repository(args.root, blocklist=blocklist, include_untracked=args.include_untracked)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
