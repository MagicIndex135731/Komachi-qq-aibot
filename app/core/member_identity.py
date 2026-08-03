from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Protocol, Sequence
import unicodedata


_LOOKUP_NORMALIZER = re.compile(
    r"[\s\u3000`~!@#$%^&*()_+\-=\[\]{}\\|;:'\",<.>/?，。！？：；、‘’“”（）《》【】]"
)


@dataclass(frozen=True, slots=True)
class GroupMemberIdentity:
    user_id: int
    nickname: str = ""
    group_card: str = ""
    in_scope: bool = True


@dataclass(frozen=True, slots=True)
class ResolvedGroupMember:
    user_id: int
    matched_alias: str


@dataclass(frozen=True, slots=True)
class GroupMemberReferenceResolution:
    status: Literal["unbound", "resolved", "ambiguous"]
    member: ResolvedGroupMember | None = None


class GroupMemberMessage(Protocol):
    user_id: int
    group_id: int | None
    raw_json: object


def normalize_member_alias(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return _LOOKUP_NORMALIZER.sub("", normalized).casefold()


def group_member_identities_from_messages(
    messages: Sequence[GroupMemberMessage],
    *,
    target_group_id: int | None = None,
) -> tuple[GroupMemberIdentity, ...]:
    """Build scoped identities from latest sender snapshots across groups."""

    identities: list[GroupMemberIdentity] = []
    seen: set[tuple[int, str, str, bool]] = set()
    for message in messages:
        raw_json = message.raw_json if isinstance(message.raw_json, dict) else {}
        sender = raw_json.get("sender")
        sender = sender if isinstance(sender, dict) else {}
        nickname = str(sender.get("nickname", "") or "").strip()
        group_card = str(sender.get("card", "") or "").strip()
        if not nickname and not group_card:
            continue
        in_scope = (
            target_group_id is None
            or int(getattr(message, "group_id", 0) or 0)
            == int(target_group_id)
        )
        identity = (int(message.user_id), nickname, group_card, in_scope)
        if identity in seen:
            continue
        seen.add(identity)
        identities.append(
            GroupMemberIdentity(
                user_id=identity[0],
                nickname=nickname,
                group_card=group_card,
                in_scope=identity[3],
            )
        )
    return tuple(identities)


def classify_group_member_reference(
    text: str,
    members: Sequence[GroupMemberIdentity],
    *,
    match_mode: Literal["exact", "contained"],
    exclude_user_ids: set[int] | frozenset[int] = frozenset(),
) -> GroupMemberReferenceResolution:
    """Classify no match separately from an unsafe ambiguous alias match."""

    normalized_text = normalize_member_alias(text)
    if not normalized_text:
        return GroupMemberReferenceResolution("unbound")
    matches: dict[int, list[str]] = {}
    alias_matched = False
    for member in members:
        user_id = int(member.user_id)
        for alias in dict.fromkeys((member.group_card.strip(), member.nickname.strip())):
            normalized_alias = normalize_member_alias(alias)
            if len(normalized_alias) < 2:
                continue
            matched = (
                normalized_text == normalized_alias
                if match_mode == "exact"
                else normalized_alias in normalized_text
            )
            if matched:
                alias_matched = True
                if user_id in exclude_user_ids:
                    continue
                matches.setdefault(user_id, []).append(alias)
    if len(matches) != 1:
        return GroupMemberReferenceResolution("ambiguous" if alias_matched else "unbound")
    user_id, aliases = next(iter(matches.items()))
    # Prefer the longest matching display alias when both card and nickname
    # belong to the same stable QQ identity.
    return GroupMemberReferenceResolution(
        "resolved",
        ResolvedGroupMember(
            user_id=user_id,
            matched_alias=max(aliases, key=lambda alias: len(normalize_member_alias(alias))),
        ),
    )


def resolve_group_member_reference(
    text: str,
    members: Sequence[GroupMemberIdentity],
    *,
    match_mode: Literal["exact", "contained"],
    exclude_user_ids: set[int] | frozenset[int] = frozenset(),
) -> ResolvedGroupMember | None:
    """Resolve an alias only when all matching aliases identify one group user."""

    resolution = classify_group_member_reference(
        text,
        members,
        match_mode=match_mode,
        exclude_user_ids=exclude_user_ids,
    )
    return resolution.member if resolution.status == "resolved" else None
