from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.public_release_sync import PublicReleaseSync, default_release_root_for_source_root


def _build_sync(args: argparse.Namespace) -> PublicReleaseSync:
    source_root = (Path(args.source_root) if args.source_root else REPO_ROOT).resolve()
    release_root = (Path(args.release_root) if args.release_root else default_release_root_for_source_root(source_root)).resolve()
    asset_root = (
        Path(args.asset_root) if args.asset_root else source_root / "scripts" / "public_release_assets"
    ).resolve()
    return PublicReleaseSync(
        source_root=source_root,
        release_root=release_root,
        asset_root=asset_root,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync the sanitized public release mirror.")
    parser.add_argument("--source-root")
    parser.add_argument("--release-root")
    parser.add_argument("--asset-root")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("reconcile")

    sync_parser = subparsers.add_parser("sync")
    sync_parser.add_argument("relative_path")

    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("relative_path")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    sync = _build_sync(args)

    if args.command == "reconcile":
        summary = sync.reconcile()
    elif args.command == "sync":
        summary = sync.sync_relative_path(args.relative_path)
    else:
        summary = sync.delete_relative_path(args.relative_path)

    print(
        f"copied={len(summary.copied)} deleted={len(summary.deleted)} skipped={len(summary.skipped)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
