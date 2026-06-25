from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil


@dataclass
class SyncSummary:
    copied: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def merge(self, other: "SyncSummary") -> None:
        self.copied.extend(other.copied)
        self.deleted.extend(other.deleted)
        self.skipped.extend(other.skipped)


def default_release_root_for_source_root(source_root: Path) -> Path:
    resolved_source_root = source_root.resolve()
    if resolved_source_root.parent.name == ".worktrees":
        return resolved_source_root.parent.parent / "release" / "github-public"
    return resolved_source_root / "release" / "github-public"


class PublicReleaseSync:
    def __init__(self, *, source_root: Path, release_root: Path, asset_root: Path) -> None:
        self.source_root = source_root.resolve()
        self.release_root = release_root.resolve()
        self.asset_root = asset_root.resolve()

    def reconcile(self) -> SyncSummary:
        summary = SyncSummary()
        desired_files: set[Path] = set()

        for relative_path in self._iter_asset_relative_files():
            summary.merge(self._copy_asset(relative_path))
            desired_files.add(relative_path)

        for relative_path in self._iter_publishable_source_relative_files():
            summary.merge(self._copy_source(relative_path))
            desired_files.add(relative_path)

        for target_file in self._iter_release_files():
            relative_path = target_file.relative_to(self.release_root)
            if relative_path not in desired_files:
                target_file.unlink()
                summary.deleted.append(self._display_path(relative_path))

        self._prune_empty_directories()
        return summary

    def sync_relative_path(self, relative_path: str) -> SyncSummary:
        normalized = self._normalize_relative_path(relative_path)
        source_path = self.source_root / normalized

        if self._is_asset_relative_path(normalized):
            if source_path.is_dir():
                return self._sync_asset_directory(normalized)
            if source_path.is_file():
                return self._copy_asset(self._target_relative_path_for_asset(normalized))
            return self.delete_relative_path(relative_path)

        if self._should_ignore_source_relative_path(normalized):
            return SyncSummary(skipped=[self._display_path(normalized)])

        if normalized in self._asset_target_paths():
            return SyncSummary(skipped=[self._display_path(normalized)])

        if source_path.is_dir():
            summary = SyncSummary()
            for child in source_path.rglob("*"):
                if not child.is_file():
                    continue
                child_relative = child.relative_to(self.source_root)
                if self._should_ignore_source_relative_path(child_relative):
                    summary.skipped.append(self._display_path(child_relative))
                    continue
                if child_relative in self._asset_target_paths():
                    summary.skipped.append(self._display_path(child_relative))
                    continue
                summary.merge(self._copy_source(child_relative))
            return summary

        if source_path.is_file():
            return self._copy_source(normalized)

        return self.delete_relative_path(relative_path)

    def delete_relative_path(self, relative_path: str) -> SyncSummary:
        normalized = self._normalize_relative_path(relative_path)

        if self._is_asset_relative_path(normalized):
            target_relative = self._target_relative_path_for_asset(normalized)
            return self._delete_target_relative_path(target_relative)

        if self._should_ignore_source_relative_path(normalized):
            return SyncSummary(skipped=[self._display_path(normalized)])

        if normalized in self._asset_target_paths():
            return SyncSummary(skipped=[self._display_path(normalized)])

        return self._delete_target_relative_path(normalized)

    def _iter_publishable_source_relative_files(self) -> list[Path]:
        result: list[Path] = []
        for source_file in self.source_root.rglob("*"):
            if not source_file.is_file():
                continue
            relative_path = source_file.relative_to(self.source_root)
            if self._should_ignore_source_relative_path(relative_path):
                continue
            if relative_path in self._asset_target_paths():
                continue
            result.append(relative_path)
        result.sort()
        return result

    def _iter_asset_relative_files(self) -> list[Path]:
        if not self.asset_root.exists():
            return []
        result = [path.relative_to(self.asset_root) for path in self.asset_root.rglob("*") if path.is_file()]
        result.sort()
        return result

    def _iter_release_files(self) -> list[Path]:
        result: list[Path] = []
        for target_file in self.release_root.rglob("*"):
            if not target_file.is_file():
                continue
            if self._is_protected_release_path(target_file):
                continue
            result.append(target_file)
        result.sort()
        return result

    def _copy_source(self, relative_path: Path) -> SyncSummary:
        source_path = self.source_root / relative_path
        target_path = self.release_root / relative_path
        self._copy_file(source_path, target_path)
        return SyncSummary(copied=[self._display_path(relative_path)])

    def _copy_asset(self, relative_path: Path) -> SyncSummary:
        source_path = self.asset_root / relative_path
        target_path = self.release_root / relative_path
        self._copy_file(source_path, target_path)
        return SyncSummary(copied=[self._display_path(relative_path)])

    def _sync_asset_directory(self, relative_path: Path) -> SyncSummary:
        summary = SyncSummary()
        asset_directory = self.source_root / relative_path
        if not asset_directory.exists():
            return summary
        for child in asset_directory.rglob("*"):
            if child.is_file():
                child_relative = child.relative_to(self.asset_root)
                summary.merge(self._copy_asset(child_relative))
        return summary

    def _delete_target_relative_path(self, relative_path: Path) -> SyncSummary:
        target_path = self.release_root / relative_path
        if not self._is_within_release_root(target_path):
            raise ValueError(f"Refusing to delete outside release root: {target_path}")

        summary = SyncSummary()
        if target_path.is_file():
            target_path.unlink()
            summary.deleted.append(self._display_path(relative_path))
        elif target_path.is_dir():
            shutil.rmtree(target_path)
            summary.deleted.append(self._display_path(relative_path))
        else:
            summary.skipped.append(self._display_path(relative_path))
        self._prune_empty_directories()
        return summary

    def _copy_file(self, source_path: Path, target_path: Path) -> None:
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    def _prune_empty_directories(self) -> None:
        directories = sorted(
            [path for path in self.release_root.rglob("*") if path.is_dir()],
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            if self._is_protected_release_path(directory):
                continue
            try:
                directory.relative_to(self.release_root)
            except ValueError:
                continue
            if any(directory.iterdir()):
                continue
            directory.rmdir()

    def _asset_target_paths(self) -> set[Path]:
        return {path for path in self._iter_asset_relative_files()}

    def _normalize_relative_path(self, relative_path: str) -> Path:
        normalized = Path(relative_path.replace("\\", "/"))
        parts = [part for part in normalized.parts if part not in ("", ".")]
        if any(part == ".." for part in parts):
            raise ValueError(f"Relative path cannot escape the source root: {relative_path}")
        return Path(*parts) if parts else Path(".")

    def _should_ignore_source_relative_path(self, relative_path: Path) -> bool:
        if relative_path == Path("."):
            return True
        parts = relative_path.parts
        if not parts:
            return True

        top_level = parts[0]
        ignored_top_levels = {
            ".git",
            ".venv",
            ".pytest_cache",
            ".pytest_tmp",
            ".worktrees",
            "__pycache__",
            "docs",
            "qq_ai_bot.egg-info",
            "release",
        }
        if top_level in ignored_top_levels:
            return True
        if top_level.startswith(".tmp") or top_level.startswith(".codex_tmp"):
            return True
        if top_level.startswith("dbg_service_"):
            return True

        if top_level == "scripts" and len(parts) > 1 and parts[1] == self.asset_root.name:
            return True

        if relative_path == Path(".env"):
            return True

        if "__pycache__" in parts:
            return True

        if top_level == "data":
            if len(parts) == 1:
                return True
            if parts[1] in {
                "history",
                "dev_control",
                "image_cache",
                "generated_images",
                "generated_private_images",
                "logs",
                "napcat",
            }:
                return True
            if relative_path.name in {
                "bot.db",
                "private_reminders_state.json",
            }:
                return True
            if relative_path.name.endswith(".db-wal") or relative_path.name.endswith(".db-shm"):
                return True
            if ".usage-backup-" in relative_path.name:
                return True

        return False

    def _is_asset_relative_path(self, relative_path: Path) -> bool:
        parts = relative_path.parts
        if len(parts) < 2:
            return False
        return parts[0] == "scripts" and parts[1] == self.asset_root.name

    def _target_relative_path_for_asset(self, relative_path: Path) -> Path:
        return Path(*relative_path.parts[2:])

    def _is_within_release_root(self, target_path: Path) -> bool:
        try:
            target_path.resolve().relative_to(self.release_root)
            return True
        except ValueError:
            return False

    def _is_protected_release_path(self, target_path: Path) -> bool:
        try:
            relative_path = target_path.relative_to(self.release_root)
        except ValueError:
            return True
        return relative_path == Path(".git") or ".git" in relative_path.parts

    @staticmethod
    def _display_path(relative_path: Path) -> str:
        return relative_path.as_posix()
