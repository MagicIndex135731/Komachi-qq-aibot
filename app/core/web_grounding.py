from __future__ import annotations

import re

from app.providers.web_search import WebPageContent, WebSearchResult


FLOOR_PATTERN = re.compile(r"(?:(\d+)|([一二三四五六七八九十两]))\s*([楼层])")
LOCAL_LOOKUP_HINTS = (
    "店",
    "餐厅",
    "饭店",
    "校区",
    "大学",
    "学校",
    "医院",
    "酒店",
    "景点",
    "附近",
    "具体位置",
    "位置",
    "在哪",
    "哪里",
    "哪层",
    "几层",
    "几楼",
    "怎么走",
    "推荐",
    "好吃",
    "好喝",
    "地址",
    "牛肉拉面",
)
LOCATION_DETAIL_HINTS = (
    "具体位置",
    "位置",
    "在哪",
    "哪里",
    "哪层",
    "几层",
    "几楼",
    "怎么走",
)
CHINESE_NUMERAL_TO_INT = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def build_grounding_notes(
    *,
    target_text: str,
    external_lookup: bool,
    web_results: list[WebSearchResult],
    web_pages: list[WebPageContent],
    recent_bot_replies: list[str],
) -> list[str]:
    normalized_target = target_text.strip()
    if not normalized_target:
        return []

    local_lookup = external_lookup or _looks_like_local_lookup(normalized_target)
    if not local_lookup:
        return []

    evidence_text = "\n".join(
        [
            *(
                f"{result.title} {result.snippet} {result.source} {result.date}"
                for result in web_results
            ),
            *(f"{page.title} {page.url} {page.content}" for page in web_pages),
        ]
    )
    evidence_floors = _extract_floor_tokens(evidence_text)
    previous_floors = _extract_floor_tokens("\n".join(recent_bot_replies[-3:]))

    notes: list[str] = []
    if not evidence_floors:
        notes.append(
            "Search evidence does not confirm any exact floor or precise spot. You may give a broad recommendation, but do not claim an exact floor, building corner, or concrete location detail unless the sources say so."
        )
        if previous_floors:
            notes.append(
                f"A previous reply mentioned {', '.join(previous_floors)}, but the current evidence does not confirm it. Do not repeat that detail as certain."
            )
        return notes

    if len(evidence_floors) > 1:
        notes.append(
            f"Current search evidence contains conflicting floor details: {', '.join(evidence_floors)}. Say the sources conflict or the location may have changed, and do not present one floor as certain."
        )

    if len(previous_floors) == 1 and len(evidence_floors) == 1 and previous_floors[0] != evidence_floors[0]:
        notes.append(
            f"Your previous reply said {previous_floors[0]}, but current evidence points to {evidence_floors[0]}. Acknowledge the earlier answer may have been inaccurate and correct it."
        )
    elif previous_floors and not set(previous_floors).intersection(evidence_floors):
        notes.append(
            f"Your earlier grounded details {', '.join(previous_floors)} do not match the current source evidence {', '.join(evidence_floors)}. Do not silently reuse the old detail."
        )

    if _is_location_detail_request(normalized_target) and len(evidence_floors) == 1:
        notes.append(
            f"The currently grounded floor detail is {evidence_floors[0]}. If you mention it, phrase it as coming from the current source evidence rather than from memory."
        )

    return notes


def _looks_like_local_lookup(text: str) -> bool:
    return any(hint in text for hint in LOCAL_LOOKUP_HINTS)


def _is_location_detail_request(text: str) -> bool:
    return any(hint in text for hint in LOCATION_DETAIL_HINTS)


def _extract_floor_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for digit_text, chinese_text, suffix in FLOOR_PATTERN.findall(text):
        if digit_text:
            number = int(digit_text)
        else:
            number = CHINESE_NUMERAL_TO_INT.get(chinese_text, 0)
        if number <= 0:
            continue
        del suffix
        token = f"{number}层"
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens
