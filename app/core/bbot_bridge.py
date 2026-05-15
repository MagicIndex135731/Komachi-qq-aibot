from __future__ import annotations

from dataclasses import dataclass

from app.core.bbot_commands import (
    BBOT_ADMIN_DENIED_TEXT,
    BBOT_TARGET_GROUP_ID,
    BBOT_TARGET_QQ,
    BbotCommandResolver,
)


@dataclass(slots=True)
class BbotCommandMatch:
    command_text: str | None = None
    denied_reason: str | None = None


_RESOLVER = BbotCommandResolver()


def resolve_bbot_command(*, group_id: int, mentioned_bot: bool, plain_text: str) -> BbotCommandMatch | None:
    parsed = _RESOLVER.resolve(group_id=group_id, mentioned_bot=mentioned_bot, plain_text=plain_text)
    if parsed is None:
        return None
    return BbotCommandMatch(command_text=parsed.command_text, denied_reason=parsed.denied_reason)


def build_bbot_outbound_message(command_text: str) -> str:
    return _RESOLVER.build_outbound_message(command_text)


__all__ = [
    "BBOT_ADMIN_DENIED_TEXT",
    "BBOT_TARGET_GROUP_ID",
    "BBOT_TARGET_QQ",
    "BbotCommandMatch",
    "build_bbot_outbound_message",
    "resolve_bbot_command",
]
