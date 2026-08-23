#!/usr/bin/env python3
"""Render a private Clash Verge profile into Xiaomachi's WSL Mihomo config."""

from __future__ import annotations

import argparse
import re
import secrets
from pathlib import Path
from typing import Any

import yaml


DEFAULT_HK_PATTERN = r"(?:香港|Hong\s*Kong|(?:^|\W)HK(?:\W|$)|🇭🇰)"
NOVA_GROUP = "XIAOMACHI-NOVA-HK"
NOVA_HOST = "ai.novacode.top"

DIRECT_RULES = (
    "DOMAIN-SUFFIX,qq.com,DIRECT",
    "DOMAIN-SUFFIX,qpic.cn,DIRECT",
    "DOMAIN-SUFFIX,gtimg.cn,DIRECT",
    "DOMAIN-SUFFIX,tenpay.com,DIRECT",
    "IP-CIDR,127.0.0.0/8,DIRECT,no-resolve",
    "IP-CIDR,10.0.0.0/8,DIRECT,no-resolve",
    "IP-CIDR,172.16.0.0/12,DIRECT,no-resolve",
    "IP-CIDR,192.168.0.0/16,DIRECT,no-resolve",
)


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source profile must contain a YAML mapping")
    return payload


def render_config(
    source: dict[str, Any],
    *,
    hk_pattern: str = DEFAULT_HK_PATTERN,
    nova_host: str = NOVA_HOST,
) -> dict[str, Any]:
    proxies = source.get("proxies")
    groups = source.get("proxy-groups")
    rules = source.get("rules")
    if not isinstance(proxies, list) or not isinstance(groups, list):
        raise ValueError("source profile must contain proxies and proxy-groups lists")
    if not isinstance(rules, list):
        raise ValueError("source profile must contain a rules list")

    matcher = re.compile(hk_pattern, re.IGNORECASE)
    hk_nodes = [
        name
        for item in proxies
        if isinstance(item, dict)
        and isinstance((name := item.get("name")), str)
        and matcher.search(name)
    ]
    if not hk_nodes:
        raise ValueError("source profile contains no Hong Kong proxy nodes")

    rendered = dict(source)
    rendered.update(
        {
            "mixed-port": 7897,
            "mode": "rule",
            "allow-lan": False,
            "bind-address": "127.0.0.1",
            "external-controller": "127.0.0.1:19090",
            "secret": secrets.token_urlsafe(24),
            "log-level": "info",
        }
    )
    tun = dict(rendered.get("tun") or {})
    tun["enable"] = False
    rendered["tun"] = tun
    rendered.pop("external-controller-pipe", None)
    rendered.pop("external-controller-cors", None)

    nova_group = {
        "name": NOVA_GROUP,
        "type": "url-test",
        "proxies": hk_nodes,
        "url": "https://cp.cloudflare.com/generate_204",
        "interval": 300,
        "tolerance": 80,
        "lazy": True,
    }
    rendered["proxy-groups"] = [
        nova_group,
        *[
            item
            for item in groups
            if not (isinstance(item, dict) and item.get("name") == NOVA_GROUP)
        ],
    ]

    managed_prefixes = (
        f"DOMAIN,{nova_host},",
        *(f"{rule.rsplit(',', 1)[0]}," for rule in DIRECT_RULES),
    )
    retained_rules = [
        rule
        for rule in rules
        if isinstance(rule, str)
        and not any(rule.startswith(prefix) for prefix in managed_prefixes)
    ]
    rendered["rules"] = [
        f"DOMAIN,{nova_host},{NOVA_GROUP}",
        *DIRECT_RULES,
        *retained_rules,
    ]
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hk-pattern", default=DEFAULT_HK_PATTERN)
    parser.add_argument("--nova-host", default=NOVA_HOST)
    args = parser.parse_args()

    rendered = render_config(
        _load_mapping(args.source),
        hk_pattern=args.hk_pattern,
        nova_host=args.nova_host,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(rendered, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(
        f"mihomo_config=rendered proxies={len(rendered['proxies'])} "
        f"hk_nodes={len(rendered['proxy-groups'][0]['proxies'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
