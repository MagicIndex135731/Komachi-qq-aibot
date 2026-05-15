from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re

from app.core.bbot_bridge import BBOT_TARGET_GROUP_ID, BBOT_TARGET_QQ

_BILIBILI_PUSH_PATTERN = re.compile(
    r"^(?P<name>.+?)（UID:\s*(?P<uid>\d+)）于.+?发布了一条(?:新|视频)?动态",
)
_BILIBILI_LIVE_PATTERN = re.compile(
    r"^(?P<name>.+?)（(?P<uid>\d+)）开始直播了",
)
_TWITTER_PUSH_PATTERN = re.compile(
    r"^(?P<name>.+?)（@(?P<username>[A-Za-z0-9_]+)）于.+?发布了一条(?:视频)?推文",
)
_BILIBILI_LIST_ENTRY_PATTERN = re.compile(
    r"(?:^|\n)\s*\d+[.、]\s*(?P<name>.+?)（UID:\s*(?P<uid>\d+)）",
)
_TWITTER_LIST_ENTRY_PATTERN = re.compile(
    r"(?:^|\n)\s*\d+[.、]\s*(?P<name>.+?)（@(?P<username>[A-Za-z0-9_]+)）",
)


@dataclass(slots=True)
class ListenerCacheEntry:
    platform: str
    external_id: str
    canonical_name: str
    aliases: list[str]
    source: str


def extract_listener_cache_entries(*, group_id: int, user_id: int, plain_text: str) -> list[ListenerCacheEntry]:
    if group_id != BBOT_TARGET_GROUP_ID or user_id != BBOT_TARGET_QQ:
        return []
    text = plain_text.strip()
    if not text:
        return []

    entries: list[ListenerCacheEntry] = []
    entries.extend(_extract_bilibili_push_entries(text))
    entries.extend(_extract_twitter_push_entries(text))
    entries.extend(_extract_bilibili_list_entries(text))
    entries.extend(_extract_twitter_list_entries(text))
    return _dedupe_entries(entries)


def resolve_cached_command_target(*, command_text: str, group_id: int, cache_repo) -> str:
    latest_dynamic_prefix = "最新动态 "
    latest_tweet_prefix = "最新推文 "
    if command_text.startswith(latest_dynamic_prefix):
        target = command_text[len(latest_dynamic_prefix) :].strip()
        if target.isdigit():
            return command_text
        entry = cache_repo.find_best_match(group_id=group_id, platform="bilibili", query=target)
        if entry is not None and str(entry.external_id).strip():
            return f"{latest_dynamic_prefix}{entry.external_id}"
    if command_text.startswith(latest_tweet_prefix):
        target = command_text[len(latest_tweet_prefix) :].strip()
        entry = cache_repo.find_best_match(group_id=group_id, platform="twitter", query=target)
        if entry is not None and str(entry.external_id).strip():
            return f"{latest_tweet_prefix}{entry.external_id}"
    return command_text


def upsert_listener_cache_entries(*, cache_repo, group_id: int, entries: list[ListenerCacheEntry], now: datetime) -> None:
    for entry in entries:
        cache_repo.upsert_entry(
            group_id=group_id,
            platform=entry.platform,
            external_id=entry.external_id,
            canonical_name=entry.canonical_name,
            aliases=entry.aliases,
            source=entry.source,
            updated_at=now,
        )


def _extract_bilibili_push_entries(text: str) -> list[ListenerCacheEntry]:
    entries: list[ListenerCacheEntry] = []
    for pattern, source in (
        (_BILIBILI_PUSH_PATTERN, "bbot_push_dynamic"),
        (_BILIBILI_LIVE_PATTERN, "bbot_push_live"),
    ):
        match = pattern.search(text)
        if match is None:
            continue
        entries.append(
            ListenerCacheEntry(
                platform="bilibili",
                external_id=match.group("uid"),
                canonical_name=match.group("name").strip(),
                aliases=_build_aliases(match.group("name").strip()),
                source=source,
            )
        )
    return entries


def _extract_twitter_push_entries(text: str) -> list[ListenerCacheEntry]:
    match = _TWITTER_PUSH_PATTERN.search(text)
    if match is None:
        return []
    name = match.group("name").strip()
    username = match.group("username").strip()
    return [
        ListenerCacheEntry(
            platform="twitter",
            external_id=username,
            canonical_name=name,
            aliases=_build_aliases(name) + [username],
            source="bbot_push_tweet",
        )
    ]


def _extract_bilibili_list_entries(text: str) -> list[ListenerCacheEntry]:
    entries = []
    for match in _BILIBILI_LIST_ENTRY_PATTERN.finditer(text):
        name = match.group("name").strip()
        entries.append(
            ListenerCacheEntry(
                platform="bilibili",
                external_id=match.group("uid"),
                canonical_name=name,
                aliases=_build_aliases(name),
                source="bbot_list_bilibili",
            )
        )
    return entries


def _extract_twitter_list_entries(text: str) -> list[ListenerCacheEntry]:
    entries = []
    for match in _TWITTER_LIST_ENTRY_PATTERN.finditer(text):
        name = match.group("name").strip()
        username = match.group("username").strip()
        entries.append(
            ListenerCacheEntry(
                platform="twitter",
                external_id=username,
                canonical_name=name,
                aliases=_build_aliases(name) + [username],
                source="bbot_list_twitter",
            )
        )
    return entries


def _build_aliases(name: str) -> list[str]:
    aliases = [name]
    leading_cjk = re.match(r"^[\u4e00-\u9fff]+", name)
    if leading_cjk is not None:
        aliases.append(leading_cjk.group(0))
    condensed = re.sub(r"[\s_]+", "", name)
    if condensed and condensed not in aliases:
        aliases.append(condensed)
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _dedupe_entries(entries: list[ListenerCacheEntry]) -> list[ListenerCacheEntry]:
    deduped: dict[tuple[str, str], ListenerCacheEntry] = {}
    for entry in entries:
        deduped[(entry.platform, entry.external_id)] = entry
    return list(deduped.values())
