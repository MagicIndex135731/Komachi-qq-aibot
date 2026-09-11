"""Quoted-message prompt helpers shared by the group and private chat paths.

The group router owned these helpers first; the private chat service needs the
same "Quoted message: ..." line and pronoun-referent note, so the pure
functions live here and both surfaces delegate to them. Nothing in this module
touches the database, the gateway, or the LLM.
"""

from __future__ import annotations

import re

from app.core.legacy_memory_context import format_member_label


QUOTED_PRONOUN_PATTERN = re.compile(r"他|她|那位|这位|这个人|那家伙")
QUOTED_REFERENT_ASK_PATTERN = re.compile(
    r"谁|什么|什么意思|在说谁|说的是谁|指谁|是谁|说什么|在说什么|指的谁"
)


def flatten_raw_message_text(raw_payload: dict | None) -> str:
    """Return the plain text of a raw OneBot payload, or an empty string."""

    if not isinstance(raw_payload, dict):
        return ""
    message = raw_payload.get("message", raw_payload.get("raw_message", ""))
    if isinstance(message, str):
        return message.strip()
    if not isinstance(message, list):
        return ""
    parts: list[str] = []
    for item in message:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        text = str(item.get("data", {}).get("text", ""))
        if text:
            parts.append(text)
    return "".join(parts).strip()


def quoted_message_line_for_prompt(*, quoted_raw_payload: dict | None) -> str | None:
    """Render ``<sender label>: <quoted text>`` for the prompt, when there is one."""

    quoted_text = flatten_raw_message_text(quoted_raw_payload)
    if not quoted_text:
        return None
    payload = quoted_raw_payload if isinstance(quoted_raw_payload, dict) else {}
    sender = payload.get("sender", {})
    if not isinstance(sender, dict):
        sender = {}
    label = format_member_label(
        nickname=str(sender.get("nickname", "")),
        group_card=str(sender.get("card", "")),
        fallback=str(payload.get("user_id", "quoted-user")),
    )
    return f"{label}: {quoted_text}"


def quoted_pronoun_referent_note(
    *,
    query_text: str,
    quoted_raw_payload: dict | None,
) -> str | None:
    """Explain "他/她" in a question about a quoted message, when it applies."""

    if not isinstance(quoted_raw_payload, dict):
        return None
    if not flatten_raw_message_text(quoted_raw_payload):
        return None
    if not QUOTED_PRONOUN_PATTERN.search(query_text):
        return None
    if not QUOTED_REFERENT_ASK_PATTERN.search(query_text):
        return None
    return (
        "Note: “他/她” in this question refers to the sender of the quoted "
        "message above. Use the recent chat to determine who or what that "
        "sender is talking about and quote the original lines. If the "
        "quoted text explicitly names another person, follow the quoted "
        "text; if no clear referent exists, say the evidence is insufficient."
    )
