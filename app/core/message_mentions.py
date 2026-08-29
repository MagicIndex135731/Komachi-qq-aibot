"""Detect whether a raw group message explicitly addresses the bot."""

from __future__ import annotations

import json
from typing import Any, Iterable


def message_mentions_bot(
    raw_json: dict[str, Any] | str | None,
    *,
    bot_qq: int,
    bot_names: Iterable[str] = (),
) -> bool:
    """True when the message @-mentions or names the bot.

    Messages addressed to the bot are human-to-AI turns: their wording and
    content are shaped by the fact that a bot is being addressed, so they
    must never be used as style samples, distillation corpus, or fact
    evidence for the member.
    """

    if isinstance(raw_json, str):
        try:
            raw_json = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            return False
    if not isinstance(raw_json, dict):
        return False
    message = raw_json.get("message", raw_json.get("raw_message", ""))
    at_target = str(bot_qq)
    if isinstance(message, list):
        for item in message:
            if (
                isinstance(item, dict)
                and item.get("type") == "at"
                and str((item.get("data") or {}).get("qq", "")) == at_target
            ):
                return True
        text = "".join(
            str(item.get("data", {}).get("text", ""))
            for item in message
            if isinstance(item, dict)
        )
    else:
        text = str(message or "")
    normalized = str(text or "").lstrip()
    if normalized.startswith(f"@{at_target}"):
        return True
    if f"[CQ:at,qq={at_target}]" in normalized:
        return True
    for bot_label in {str(name).strip() for name in bot_names if str(name).strip()}:
        if bot_label and normalized.startswith(f"@{bot_label}"):
            return True
    return False


def collect_bot_display_names(
    raw_json_values: Iterable[dict[str, Any] | str | None],
) -> set[str]:
    """All sender card/nickname values the bot has ever used."""

    names: set[str] = set()
    for raw in raw_json_values:
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            parsed = raw
        if not isinstance(parsed, dict):
            continue
        sender = parsed.get("sender")
        if not isinstance(sender, dict):
            continue
        for value in (sender.get("card"), sender.get("nickname")):
            cleaned = str(value or "").strip()
            if cleaned:
                names.add(cleaned)
    return names


def bot_mention_names(
    *,
    bot_qq: int,
    default_name: str = "",
    display_names: Iterable[str] = (),
) -> set[str]:
    """Full mention-name set: QQ, default persona name, historical cards,
    plus the short form when any known name contains 小町."""

    names: set[str] = {str(bot_qq)}
    for value in (*display_names, default_name):
        cleaned = str(value or "").strip()
        if cleaned:
            names.add(cleaned)
    if any("小町" in name for name in names):
        names.add("小町")
    return names
