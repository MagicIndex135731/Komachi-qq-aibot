from __future__ import annotations

import argparse
from pathlib import Path


ALLOWED_KEYS = frozenset(
    {
        "MEMORY_ORCHESTRATION_V2_ENABLED",
        "MEMORY_ORCHESTRATION_SHADOW_MODE",
        "MEMORY_EMBEDDING_DEVICE",
        "MEMORY_EMBEDDING_LOCAL_FILES_ONLY",
    }
)


def update_env_keys(path: Path, assignments: list[str]) -> None:
    if path.name != ".env":
        raise ValueError("refusing to edit a file other than .env")
    updates: dict[str, str] = {}
    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        allowed_values = {"auto", "cuda", "cpu"} if key == "MEMORY_EMBEDDING_DEVICE" else {"true", "false"}
        if separator != "=" or key not in ALLOWED_KEYS or value not in allowed_values:
            raise ValueError("only approved memory rollout keys may be changed")
        updates[key] = value
    if not updates:
        raise ValueError("at least one approved key assignment is required")

    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    seen: set[str] = set()
    rendered: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#") or "=" not in stripped:
            rendered.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            rendered.append(f"{key}={updates[key]}{newline}")
            seen.add(key)
        else:
            rendered.append(line)
    newline = "\r\n" if "\r\n" in original else "\n"
    for key, value in updates.items():
        if key not in seen:
            rendered.append(f"{key}={value}{newline}")
    path.write_text("".join(rendered), encoding="utf-8", newline="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--set", dest="assignments", action="append", required=True)
    args = parser.parse_args()
    update_env_keys(args.path, args.assignments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
