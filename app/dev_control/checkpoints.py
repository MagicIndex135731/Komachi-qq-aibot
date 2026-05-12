from __future__ import annotations

from pathlib import Path
import shutil


def _iter_repo_files(*, repo_root: Path, checkpoint_dir: Path):
    ignored_dir_names = {"data", ".git", ".venv", "__pycache__", ".pytest_cache"}
    resolved_checkpoint_dir = checkpoint_dir.resolve()
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if resolved_checkpoint_dir in path.resolve().parents:
            continue
        if any(part in ignored_dir_names for part in path.relative_to(repo_root).parts):
            continue
        yield path


def create_repo_checkpoint(*, repo_root: Path, checkpoint_dir: Path) -> dict[str, list[str]]:
    repo_root = repo_root.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = checkpoint_dir / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    files: list[str] = []
    for source_path in _iter_repo_files(repo_root=repo_root, checkpoint_dir=checkpoint_dir):
        relative_path = source_path.relative_to(repo_root)
        target_path = snapshot_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        files.append(relative_path.as_posix())
    return {"files": files}


def restore_repo_checkpoint(*, repo_root: Path, checkpoint_dir: Path, manifest: dict[str, list[str]]) -> None:
    repo_root = repo_root.resolve()
    snapshot_dir = checkpoint_dir / "snapshot"
    known_files = {Path(relative_path) for relative_path in manifest.get("files", [])}

    for relative_path in known_files:
        source_path = snapshot_dir / relative_path
        target_path = repo_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    for current_path in list(_iter_repo_files(repo_root=repo_root, checkpoint_dir=checkpoint_dir)):
        relative_path = current_path.relative_to(repo_root)
        if relative_path in known_files:
            continue
        current_path.unlink(missing_ok=True)
