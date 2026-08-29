"""Detect whether a raw group message explicitly addresses the bot."""

from __future__ import annotations

import json
from typing import Any, Iterable


def message_mentions_bot(
    raw_json: dict[str, Any] | str | None,
    *,
    bot_qqs: Iterable[int],
    bot_text_names: Iterable[str] = (),
) -> bool:
    """True when the message @-mentions or names one of the bot accounts.

    Messages addressed to the bot are human-to-AI turns: their wording and
    content are shaped by the fact that a bot is being addressed, so they
    must never be used as style samples, distillation corpus, or fact
    evidence for the member. The QQ at-type match is exact; the text
    ``@name`` match only fires for names that are bot-exclusive (no real
    member shares the name), so ``@阿渣`` aimed at the real member is kept.
    """

    if isinstance(raw_json, str):
        try:
            raw_json = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            return False
    if not isinstance(raw_json, dict):
        return False
    message = raw_json.get("message", raw_json.get("raw_message", ""))
    at_targets = {str(int(qq)) for qq in bot_qqs}
    if isinstance(message, list):
        for item in message:
            if (
                isinstance(item, dict)
                and item.get("type") == "at"
                and str((item.get("data") or {}).get("qq", "")) in at_targets
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
    if any(normalized.startswith(f"@{at_target}") for at_target in at_targets):
        return True
    if any(f"[CQ:at,qq={at_target}]" in normalized for at_target in at_targets):
        return True
    for bot_label in {
        str(name).strip() for name in bot_text_names if str(name).strip()
    }:
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


def bot_text_mention_names(
    *,
    bot_qqs: Iterable[int],
    default_name: str = "",
    bot_display_names: Iterable[str] = (),
    member_display_names: Iterable[str] = (),
) -> set[str]:
    """Text mention names that unambiguously address the bot.

    Every bot card (比企谷小町/阿渣/逆蝶蝶) may collide with a real member's
    name; only names absent from the real-member set are safe to filter on
    text form. The short form 小町 is added only when no real member uses it.
    """

    member_names = {str(name).strip() for name in member_display_names if str(name).strip()}
    names: set[str] = set()
    for value in (*bot_display_names, default_name):
        cleaned = str(value or "").strip()
        if cleaned and cleaned not in member_names:
            names.add(cleaned)
    if (
        any("小町" in name for name in (*bot_display_names, default_name))
        and "小町" not in member_names
    ):
        names.add("小町")
    return names
