from pathlib import Path

from app.public_release_sync import PublicReleaseSync, default_release_root_for_source_root


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_sync(tmp_path: Path) -> tuple[Path, Path, Path, PublicReleaseSync]:
    source_root = tmp_path / "repo"
    release_root = source_root / "release" / "github-public"
    asset_root = source_root / "scripts" / "public_release_assets"
    (release_root / ".git").mkdir(parents=True, exist_ok=True)
    sync = PublicReleaseSync(
        source_root=source_root,
        release_root=release_root,
        asset_root=asset_root,
    )
    return source_root, release_root, asset_root, sync


def test_reconcile_copies_publishable_source_file(tmp_path: Path) -> None:
    source_root, release_root, asset_root, sync = _build_sync(tmp_path)
    _write(source_root / "app" / "main.py", "print('hi')\n")
    _write(asset_root / "README.md", "public readme\n")

    sync.reconcile()

    assert (release_root / "app" / "main.py").read_text(encoding="utf-8") == "print('hi')\n"


def test_reconcile_uses_public_asset_mapping_for_root_readme(tmp_path: Path) -> None:
    source_root, release_root, asset_root, sync = _build_sync(tmp_path)
    _write(source_root / "README.md", "private readme\n")
    _write(asset_root / "README.md", "public readme\n")

    sync.reconcile()

    assert (release_root / "README.md").read_text(encoding="utf-8") == "public readme\n"


def test_sync_deleted_path_removes_public_file_but_keeps_dot_git(tmp_path: Path) -> None:
    source_root, release_root, asset_root, sync = _build_sync(tmp_path)
    source_file = source_root / "app" / "main.py"
    _write(source_file, "print('hi')\n")
    _write(asset_root / "README.md", "public readme\n")
    sync.reconcile()
    source_file.unlink()

    sync.delete_relative_path("app/main.py")

    assert not (release_root / "app" / "main.py").exists()
    assert (release_root / ".git").is_dir()


def test_reconcile_skips_ignored_runtime_files(tmp_path: Path) -> None:
    source_root, release_root, asset_root, sync = _build_sync(tmp_path)
    _write(source_root / ".env", "LLM_API_KEY=secret\n")
    _write(source_root / ".codex_tmp" / "pytest" / "secret.txt", "private\n")
    _write(source_root / ".tmp_policy_tests_red" / "bot.db", "sqlite\n")
    _write(source_root / "data" / "history" / "chat.json", "{}\n")
    _write(source_root / "data" / "napcat" / "v4.18.7" / "config.json", "{}\n")
    _write(source_root / "data" / "generated_images" / "private.png", "png-bytes\n")
    _write(source_root / "data" / "generated_private_images" / "private-dm.png", "png-bytes\n")
    _write(source_root / "docs" / "notes.md", "internal\n")
    _write(asset_root / "README.md", "public readme\n")

    sync.reconcile()

    assert not (release_root / ".env").exists()
    assert not (release_root / ".codex_tmp" / "pytest" / "secret.txt").exists()
    assert not (release_root / ".tmp_policy_tests_red" / "bot.db").exists()
    assert not (release_root / "data" / "history" / "chat.json").exists()
    assert not (release_root / "data" / "napcat" / "v4.18.7" / "config.json").exists()
    assert not (release_root / "data" / "generated_images" / "private.png").exists()
    assert not (release_root / "data" / "generated_private_images" / "private-dm.png").exists()
    assert not (release_root / "docs" / "notes.md").exists()


def test_reconcile_removes_stale_public_file_while_preserving_git_directory(tmp_path: Path) -> None:
    source_root, release_root, asset_root, sync = _build_sync(tmp_path)
    _write(asset_root / "README.md", "public readme\n")
    _write(release_root / "app" / "stale.py", "old\n")

    sync.reconcile()

    assert not (release_root / "app" / "stale.py").exists()
    assert (release_root / ".git").is_dir()


def test_default_release_root_uses_common_repo_root_for_worktrees(tmp_path: Path) -> None:
    source_root = tmp_path / ".worktrees" / "task-01-bootstrap"
    source_root.mkdir(parents=True)

    release_root = default_release_root_for_source_root(source_root)

    assert release_root == tmp_path / "release" / "github-public"
