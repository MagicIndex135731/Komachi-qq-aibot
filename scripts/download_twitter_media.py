"""Download one public X/Twitter post locally, without starting Komachi."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.providers.twitter_media import MIB, TwitterMediaDownloader, TwitterMediaError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下载一个公开X/Twitter帖子的图片、视频和动图；不启动或接入小町。")
    parser.add_argument("url", help="x.com 或 twitter.com 的 status 帖子链接")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/twitter_media")
    parser.add_argument("--resolver", choices=["auto", "syndication", "fxtwitter"], default="auto")
    parser.add_argument("--max-file-mib", type=int, default=100, help="单文件上限，默认100MiB")
    parser.add_argument("--max-total-mib", type=int, default=300, help="单次下载总上限，默认300MiB")
    parser.add_argument("--timeout", type=float, default=180, help="包含重试的总时限，秒；最大1800")
    parser.add_argument("--proxy", help="可选HTTP/SOCKS代理URL；未指定时遵循HTTP_PROXY/HTTPS_PROXY环境变量")
    return parser


async def run(args: argparse.Namespace) -> dict:
    async with httpx.AsyncClient(proxy=args.proxy, follow_redirects=False) as client:
        downloader = TwitterMediaDownloader(client, resolver=args.resolver,
            max_file_bytes=args.max_file_mib * MIB, max_total_bytes=args.max_total_mib * MIB,
            timeout_seconds=args.timeout)
        result = await downloader.download(args.url, args.output_dir)
    return {"ok": True, "directory": str(result.directory), "manifest": str(result.manifest_path),
            "count": len(result.manifest["files"]), "total_bytes": result.manifest["total_bytes"],
            "files": result.manifest["files"]}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = asyncio.run(run(args))
    except TwitterMediaError as exc:
        hint = ("确认是公开且含媒体的帖子；删除、私密、年龄或地区限制及上游验证可能导致无法下载。"
                "网络问题可配置 --proxy 或 HTTPS_PROXY；超限可调整 --max-file-mib/--max-total-mib。")
        print(json.dumps({"ok": False, "error": exc.code, "hint": hint}, ensure_ascii=False), file=sys.stderr)
        return 1
    except (ValueError, ImportError):
        print(json.dumps({"ok": False, "error": "invalid_proxy_configuration",
                          "hint": "使用有效的HTTP代理URL；SOCKS代理需要额外安装httpx[socks]。"}, ensure_ascii=False), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"ok": False, "error": "cancelled"}), file=sys.stderr)
        return 130
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
