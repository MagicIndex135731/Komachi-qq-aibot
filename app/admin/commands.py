from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(slots=True)
class CommandContext:
    sender_qq: int
    is_private_chat: bool
    group_id: int | None


@dataclass(slots=True)
class ParsedCommand:
    name: str
    arguments: dict[str, int | str]


class AdminCommandParser:
    def __init__(self, *, admin_whitelist: set[int]) -> None:
        self.admin_whitelist = admin_whitelist
        self._group_allow = re.compile(r"^/bot group allow (?P<group_id>\d+)$")
        self._group_deny = re.compile(r"^/bot group deny (?P<group_id>\d+)$")
        self._status = re.compile(r"^/bot status$")
        self._off = re.compile(r"^/bot off$")
        self._on = re.compile(r"^/bot on$")

    def parse(self, raw_text: str, context: CommandContext) -> ParsedCommand | None:
        if context.sender_qq not in self.admin_whitelist:
            return None
        for pattern, name in (
            (self._group_allow, "group_allow"),
            (self._group_deny, "group_deny"),
            (self._status, "status"),
            (self._off, "off"),
            (self._on, "on"),
        ):
            match = pattern.fullmatch(raw_text.strip())
            if match is None:
                continue
            if name in {"group_allow", "group_deny"} and not context.is_private_chat:
                return None
            values = {
                key: int(value) if value.isdigit() else value
                for key, value in match.groupdict().items()
            }
            return ParsedCommand(name=name, arguments=values)
        return None
