"""Select the optional WSL proxy from actual provider proxy settings."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from urllib.parse import urlsplit


def uses_mihomo(content: str) -> bool:
    values: dict[str, str] = {}
    keys = {"XIAOMACHI_HTTP_PROXY", "XIAOMACHI_HTTPS_PROXY", "DOCKER_HTTP_PROXY", "DOCKER_HTTPS_PROXY", 'WEB_PAGE_PROXY_URL'}
    for line in content.splitlines():
        match = re.match(r"\s*(?:export\s+)?([A-Z_]+)\s*=\s*(.*?)\s*$", line)
        if match and match[1] in keys:
            values[match[1]] = match[2].strip().strip("\"'")
    for scheme in ("HTTP", "HTTPS"):
        value = values.get(f"XIAOMACHI_{scheme}_PROXY") or values.get(f"DOCKER_{scheme}_PROXY", "")
        if not value:
            continue
        try:
            parsed = urlsplit(value)
            if parsed.scheme in {"http", "https", "socks5", "socks5h"} and parsed.hostname in {"127.0.0.1", "localhost", "::1"} and parsed.port == 18797:
                return True
        except ValueError:
            pass
    page_proxy = values.get('WEB_PAGE_PROXY_URL', '')
    if page_proxy:
        try:
            parsed = urlsplit(page_proxy)
            if parsed.scheme in {'http', 'https'} and parsed.hostname in {'127.0.0.1', 'localhost', '::1'} and parsed.port == 18797:
                return True
        except ValueError:
            pass
    return False


def configure_unit(content: str, *, required: bool) -> str:
    lines = []
    for line in content.splitlines(keepends=True):
        if line.startswith(("Wants=", "After=")):
            key, values = line.strip().split("=", 1)
            words = [word for word in values.split() if word != "xiaomachi-mihomo.service"]
            if required:
                words.append("xiaomachi-mihomo.service")
            line = key + "=" + " ".join(words) + "\n"
        lines.append(line)
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--unit-file", type=Path, required=True)
    args = parser.parse_args()
    required = uses_mihomo(args.env_file.read_text(encoding="utf-8-sig"))
    args.unit_file.write_text(configure_unit(args.unit_file.read_text(encoding="utf8"), required=required), encoding="utf8")
    print("true" if required else "false")


if __name__ == "__main__":
    main()
