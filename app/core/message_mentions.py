"""Detect whether a raw group message explicitly addresses the bot."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable


def calls_bot_by_name(text: str, bot_names: Iterable[str]) -> bool:
    """A vocative or explicit call, rather than a name inside a comparison."""
    names = '|'.join(re.escape(name) for name in bot_names if name)
    if not names:
        return False
    return bool(re.search(r'(?:^|[\s，,。！!？?；;])(?:@)?(?:' + names +
        r')(?=$|[\s，,:：！!？?]|你|帮|来|请|看|查|搜|说|画|生成|能|可以|把|给|解释|分析)', text.strip(), re.I))


def addressing_evidence(raw, *, bot_id, bot_names=(), text='', reply_author_id=None):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = {}
    raw = raw if isinstance(raw, dict) else {}
    segments = raw.get('message', raw.get('raw_message', ''))
    targets = []
    values = ([item['data'].get('qq') for item in segments[:256]
               if isinstance(item, dict) and item.get('type') == 'at' and isinstance(item.get('data'), dict)] if isinstance(segments, list)
              else re.findall(r'\[CQ:at,qq=([0-9]+)(?:,[^\]]*)?\]', str(segments)))
    for value in values[:100]:
        if re.fullmatch(r'[0-9]{1,19}', str(value)) and 0 < int(value) < 2**63:
            targets.append(int(value))
    at_bot = bot_id in targets
    other = (any(value != bot_id for value in targets) or
        reply_author_id is not None and reply_author_id != bot_id) and not at_bot and reply_author_id != bot_id
    return {'bot_id': bot_id, 'at_user_ids': list(dict.fromkeys(targets)),
        'reply_author_id': reply_author_id, 'bot_explicitly_called': at_bot or calls_bot_by_name(text, bot_names),
        'other_addressee': bool(other)}


ADDRESSING_POLICY = ('你的账号身份以addressing.bot_id为准，author_id是发言者，at_user_ids和回复作者表示被叫到的人。'
    '正文中的“你”“自己”优先指真实@或回复的对象；提到小町可能只是比较或谈论，不能据此把给他人的任务认领成自己。'
    '他人照片、身体、经历不属于你；不确定指代时依据上下文核实，别编造身份混淆或嘲讽用户认错人。')


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
