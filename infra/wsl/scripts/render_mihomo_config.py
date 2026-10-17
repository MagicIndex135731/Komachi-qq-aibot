#!/usr/bin/env python3
"""Render a private Clash Verge profile into Xiaomachi's WSL Mihomo config."""

from __future__ import annotations

import argparse
import re
import secrets
from pathlib import Path
from typing import Any

import yaml


DEFAULT_JP_PATTERN = r"(?:日本|Japan|(?:^|\W)JP(?:\W|$)|🇯🇵)"
RATE_ONE_PATTERN = r'(?:[x×✕]\s*1(?:\.0+)?(?![\d.])|(?<![\d.])1(?:\.0+)?\s*[x×✕]|倍率\s*[:：]?\s*1(?:\.0+)?(?![\d.])|(?<![\d.])1(?:\.0+)?\s*倍(?!率))'
PAGE_GROUP = 'XIAOMACHI-WEB-JP-X1'
NOVA_GROUP = "XIAOMACHI-NOVA-JP"
NOVA_HOST = "ai.novacode.top"

DIRECT_RULES = (
    # Pin commonly used mainland pages and their CDN/short-link domains before
    # the general geographic rules; GeoSite updates must not regress these.
    "DOMAIN-SUFFIX,xiaohongshu.com,DIRECT",
    "DOMAIN-SUFFIX,xhslink.com,DIRECT",
    "DOMAIN-SUFFIX,xhscdn.com,DIRECT",
    "DOMAIN-SUFFIX,xhsimg.com,DIRECT",
    "DOMAIN-SUFFIX,xhsstatic.com,DIRECT",
    "DOMAIN-SUFFIX,qq.com,DIRECT",
    "DOMAIN-SUFFIX,qpic.cn,DIRECT",
    "DOMAIN-SUFFIX,gtimg.cn,DIRECT",
    "DOMAIN-SUFFIX,tenpay.com,DIRECT",
    # DeepSeek's official API is reachable from mainland China and carries the
    # bot's chat/search traffic; routing it through a proxy node only adds
    # latency and puts model calls on a residential-ish egress.
    "DOMAIN-SUFFIX,deepseek.com,DIRECT",
    "IP-CIDR,127.0.0.0/8,DIRECT,no-resolve",
    "IP-CIDR,10.0.0.0/8,DIRECT,no-resolve",
    "IP-CIDR,172.16.0.0/12,DIRECT,no-resolve",
    "IP-CIDR,192.168.0.0/16,DIRECT,no-resolve",
    "IP-CIDR6,::1/128,DIRECT,no-resolve",
    "IP-CIDR6,fc00::/7,DIRECT,no-resolve",
    "DOMAIN-SUFFIX,cn,DIRECT",
    "GEOSITE,cn,DIRECT",
    "GEOIP,CN,DIRECT",
)


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source profile must contain a YAML mapping")
    return payload


def render_config(
    source: dict[str, Any],
    *,
    region_pattern: str = DEFAULT_JP_PATTERN,
    nova_host: str = NOVA_HOST,
) -> dict[str, Any]:
    proxies = source.get("proxies")
    groups = source.get("proxy-groups")
    rules = source.get("rules")
    if not isinstance(proxies, list) or not isinstance(groups, list):
        raise ValueError("source profile must contain proxies and proxy-groups lists")
    if not isinstance(rules, list):
        raise ValueError("source profile must contain a rules list")

    matcher = re.compile(region_pattern, re.IGNORECASE)
    selected = [
        dict(item)
        for item in proxies
        if isinstance(item, dict)
        and isinstance((name := item.get("name")), str)
        and matcher.search(name)
    ]
    if not selected:
        raise ValueError("source profile contains no Japan proxy nodes")
    _validate_dependencies(selected)

    # Do not inherit source listeners, providers or country selectors: these
    # could retain a non-Japanese fallback or bind Windows' reserved ports.
    rendered = {
            "mixed-port": 18797,
            "mode": "rule",
            "allow-lan": False,
            "bind-address": "127.0.0.1",
            "external-controller": "127.0.0.1:19090",
            "secret": secrets.token_urlsafe(24),
            "log-level": "info",
        }
    rendered["tun"] = {"enable": False}
    rendered["proxies"] = selected

    nova_group = {
        "name": NOVA_GROUP,
        "type": "url-test",
        "proxies": [item['name'] for item in selected],
        "url": "https://cp.cloudflare.com/generate_204",
        "interval": 300,
        "tolerance": 80,
        "lazy": True,
    }
    rendered["proxy-groups"] = [nova_group]
    rendered["rules"] = [
        f"DOMAIN,{nova_host},{NOVA_GROUP}",
        *DIRECT_RULES,
        f"MATCH,{NOVA_GROUP}",
    ]
    return rendered


def _validate_dependencies(selected: list[dict[str, Any]]) -> None:
    names = {item['name'] for item in selected}
    if any(item.get('dialer-proxy') not in (None, 'DIRECT')
           and item['dialer-proxy'] not in names for item in selected):
        raise ValueError('selected Japan node depends on an excluded proxy')


def resolve_ikuuu_source(directory: Path) -> Path:
    metadata = _load_mapping(directory / 'profiles.yaml')
    matches = [item for item in metadata.get('items', []) if isinstance(item, dict)
               and re.search('ikuuu', str(item.get('name', '')), re.IGNORECASE)]
    if len(matches) != 1:
        raise ValueError('expected exactly one named ikuuu subscription')
    root = (directory / 'profiles').resolve()
    path = (root / str(matches[0].get('file', ''))).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('ikuuu source must be a local file inside profiles')
    return path


def render_page_config(source: dict[str, Any], *, unmarked_rate_one: bool = False) -> dict[str, Any]:
    """A dedicated listener: only explicitly proxied page clients reach it."""
    proxies = source.get('proxies')
    if not isinstance(proxies, list):
        raise ValueError('subscription has no proxy list')
    japan = re.compile(DEFAULT_JP_PATTERN, re.IGNORECASE)
    rate_one = re.compile(RATE_ONE_PATTERN, re.IGNORECASE)
    any_rate = re.compile(r'(?:[x×✕]\s*\d|\d(?:\.\d+)?\s*[x×✕]|倍率|\d\s*倍)', re.IGNORECASE)
    selected = [dict(item) for item in proxies if isinstance(item, dict)
                and isinstance(item.get('name'), str) and japan.search(item['name'])
                and (rate_one.search(item['name']) or unmarked_rate_one and not any_rate.search(item['name']))]
    if not selected:
        raise ValueError('subscription has no Japan rate-x1 nodes')
    _validate_dependencies(selected)
    return {'mixed-port': 18797, 'mode': 'rule', 'allow-lan': False,
            'bind-address': '127.0.0.1', 'external-controller': '127.0.0.1:19090',
            'secret': secrets.token_urlsafe(24), 'log-level': 'warning', 'tun': {'enable': False},
            'proxies': selected, 'proxy-groups': [{'name': PAGE_GROUP, 'type': 'url-test',
                'proxies': [item['name'] for item in selected], 'url': 'https://cp.cloudflare.com/generate_204',
                'interval': 300, 'tolerance': 0, 'lazy': False, 'timeout': 5000}],
            'rules': [f'DOMAIN,{NOVA_HOST},{PAGE_GROUP}', *DIRECT_RULES, 'MATCH,' + PAGE_GROUP]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument('--clash-directory', type=Path)
    parser.add_argument('--print-ikuuu-source', action='store_true')
    parser.add_argument("--region-pattern", default=DEFAULT_JP_PATTERN)
    parser.add_argument("--nova-host", default=NOVA_HOST)
    parser.add_argument('--page-only', action='store_true')
    parser.add_argument('--unmarked-rate-one', action='store_true',
                        help='Explicit operator convention: unlabeled nodes have rate x1')
    args = parser.parse_args()

    if args.print_ikuuu_source:
        if args.clash_directory is None:
            parser.error('--clash-directory is required for subscription discovery')
        print(resolve_ikuuu_source(args.clash_directory))
        return 0
    if args.source is None or args.output is None:
        parser.error('--source and --output are required for rendering')

    rendered = render_page_config(_load_mapping(args.source), unmarked_rate_one=args.unmarked_rate_one) if args.page_only else render_config(
        _load_mapping(args.source),
        region_pattern=args.region_pattern,
        nova_host=args.nova_host,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(rendered, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(
        f"mihomo_config=rendered proxies={len(rendered['proxies'])} "
        f"japan_nodes={len(rendered['proxy-groups'][0]['proxies'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
