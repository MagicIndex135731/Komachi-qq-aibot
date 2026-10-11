"""Validate the public documentation without private project metadata or network calls."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    "README.md", "docs/ARCHITECTURE.md", "docs/CONFIGURATION.md", "docs/MEMORY.md",
    "docs/DEVELOPMENT.md", "docs/SECURITY.md", "infra/wsl/README.md",
)
LINKS = re.compile(r"!?\[[^\]\n]*\]\((<[^>]+>|[^)\n]+)\)")
PRIVATE = ("data/", "backlog/", "docs/project/", ".codex/", ".agents/")


def _anchors(content: str) -> set[str]:
    anchors, counts = set(), {}
    for heading in re.findall(r"^#{1,6}\s+(.+)$", content, re.MULTILINE):
        base = re.sub(r"[^\w\- ]", "", heading.replace("`", "").lower()).replace(" ", "-")
        count = counts.get(base, 0)
        anchors.add(base if count == 0 else f"{base}-{count}")
        counts[base] = count + 1
    return anchors


def audit_public_docs(root: Path, documents=DOCUMENTS) -> dict:
    root = root.resolve()
    errors, checked = [], 0
    for relative in documents:
        document = root / relative
        if not document.is_file():
            errors.append(f"Missing public document: {relative}")
            continue
        content = document.read_text(encoding="utf-8-sig")
        for match in LINKS.finditer(content):
            target = match.group(1).strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            actual = (document.parent / unquote(parsed.path)).resolve() if parsed.path else document.resolve()
            if not actual.is_relative_to(root):
                errors.append(f"Link outside public tree: {relative}")
                continue
            path = actual.relative_to(root).as_posix()
            if path == "AGENTS.md" or path.startswith(PRIVATE):
                errors.append(f"Private dependency in public document: {relative} -> {path}")
                continue
            if not actual.exists():
                errors.append(f"Broken public link: {relative} -> {path}")
                continue
            if parsed.fragment and actual.suffix.lower() == ".md":
                if unquote(parsed.fragment) not in _anchors(actual.read_text(encoding="utf-8-sig")):
                    errors.append(f"Missing heading: {relative} -> {path}#{parsed.fragment}")
                    continue
            checked += 1
        if re.search(r"\]\([^)]*(?:project-refactor|backlog/tasks|docs/project)/", content):
            errors.append(f"Private history link in public document: {relative}")
    return {"status": "FAIL" if errors else "PASS", "documents": len(documents),
            "local_links_checked": checked, "errors": errors}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = audit_public_docs(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
