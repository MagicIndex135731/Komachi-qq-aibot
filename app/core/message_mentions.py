"""Detect whether a raw group message explicitly addresses the bot."""

from __future__ import annotations

import json
from typing import Any


def message_mentions_bot(
    raw_json: dict[str, Any] | str | None,
    *,
    bot_qq: int,
    bot_name: str = "",
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
    bot_label = str(bot_name or "").strip()
    if bot_label and normalized.startswith(f"@{bot_label}"):
        return True
    return False
