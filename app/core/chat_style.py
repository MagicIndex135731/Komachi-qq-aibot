from __future__ import annotations

import json
import re


MARKDOWN_FENCE_PATTERN = re.compile(r"```+")
MARKDOWN_INLINE_PATTERN = re.compile(r"[*_`~]+")
MARKDOWN_HEADING_PATTERN = re.compile(r"^\s{0,3}(?:#{1,6}|>+)\s*")
MODEL_THINK_BLOCK_PATTERN = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
LIST_PREFIX_PATTERN = re.compile(r"^\s{0,3}(?:[-*+]\s+|(?:\d+|[A-Za-z])[.)]\s+)")
ORDERING_PREFIX_PATTERN = re.compile(
    r"^(?:第[一二三四五六七八九十百千万0-9]+[、，.]?|首先[:：]?\s*|其次[:：]?\s*|再次[:：]?\s*|最后[:：]?\s*|另外[:：]?\s*|然后[:：]?\s*|再说[:：]?\s*|一是[:：]?\s*|二是[:：]?\s*)"
)
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
CLAUSE_PATTERN = re.compile(r"[^。！？!?~，、；;:：]+(?:[。！？!?~，、；;:：]|$)")
SENTENCE_ENDINGS = "。！？!?~."
CLAUSE_ENDINGS = "，、；;:："
PROACTIVE_FORMAL_LEADIN_PATTERN = re.compile(
    r"^(?:总的来说|总体来说|简单来说|从这个角度看|从这个角度来说|某种程度上|归根结底|本质上|由此可见|可以看出|这意味着|这说明)(?:[，,:：]\s*)?"
)


def build_human_chat_style_lines(*, proactive_turn: bool = False) -> list[str]:
    lines = [
        "Talk like a real person chatting in a group.",
        "Do not use Markdown, headings, bullet lists, numbered lists, or checklist formatting in normal replies.",
        "If someone wants a detailed explanation, stay conversational and explain in natural paragraphs instead of notes or tutorial formatting.",
        "Do not use stock assistant transitions like first, second, in summary, or here are a few points.",
    ]
    if proactive_turn:
        lines.extend(
            [
                "For proactive interjections, sound like a real person casually chiming in.",
                "For proactive interjections, answer with one complete short sentence, usually 8-16 Chinese characters.",
                "For proactive interjections, make the model output short directly. Do not rely on later truncation.",
                "For proactive interjections, prefer one compact QQ message instead of multiple lines or fragments.",
                "For proactive interjections, prefer casual everyday Chinese phrasing like '那确实有点贵啊''这也太坑了吧''有点离谱了'.",
                "For proactive interjections, use spoken Chinese you might actually see between friends on QQ, not polished written prose.",
                "For proactive interjections, have a mild opinion of your own; not just agree with the previous chat.",
                "For proactive interjections, add a small fresh angle, light disagreement, or specific judgment when it fits.",
                "For proactive interjections, avoid empty filler-only replies like '是哦''确实' and keep one tiny concrete reaction tied to the topic.",
                "For proactive interjections, do not turn the reply into a mini-analysis, recap, or tidy conclusion.",
            ]
        )
    return lines


def _sentence_punctuation(text: str) -> str:
    return "。" if CHINESE_PATTERN.search(text) else "."


def _strip_line(text: str) -> tuple[str, bool]:
    cleaned = MARKDOWN_FENCE_PATTERN.sub("", text).strip()
    if not cleaned:
        return "", False

    cleaned = MARKDOWN_HEADING_PATTERN.sub("", cleaned)

    was_list_line = bool(LIST_PREFIX_PATTERN.match(cleaned))
    if was_list_line:
        cleaned = LIST_PREFIX_PATTERN.sub("", cleaned)

    cleaned = ORDERING_PREFIX_PATTERN.sub("", cleaned)
    cleaned = MARKDOWN_INLINE_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, was_list_line


def _strip_leading_control_json(text: str) -> str:
    cleaned = text.lstrip()
    if not cleaned.startswith("{"):
        return text

    try:
        payload, end_index = json.JSONDecoder().raw_decode(cleaned)
    except ValueError:
        return text
    if not isinstance(payload, dict):
        return text

    normalized_keys = {str(key).strip().lower() for key in payload}
    if normalized_keys.isdisjoint({"queries", "sourcefilter", "sources", "tool", "tools", "filelibrary"}):
        return text

    remainder = cleaned[end_index:].lstrip()
    return remainder or text


def normalize_chat_reply(text: str) -> str:
    text = MODEL_THINK_BLOCK_PATTERN.sub("", text)
    text = _strip_leading_control_json(text)
    pieces: list[str] = []
    for raw_line in text.splitlines():
        cleaned, was_list_line = _strip_line(raw_line)
        if not cleaned:
            continue
        if was_list_line and cleaned[-1] not in SENTENCE_ENDINGS:
            cleaned += _sentence_punctuation(cleaned)
        pieces.append(cleaned)

    if not pieces:
        fallback = re.sub(r"\s+", " ", MARKDOWN_INLINE_PATTERN.sub("", text)).strip()
        return fallback

    normalized = pieces[0]
    for piece in pieces[1:]:
        if normalized.endswith(tuple(CLAUSE_ENDINGS) + tuple(SENTENCE_ENDINGS) + (":",)):
            normalized += piece
            continue
        normalized += f" {piece}"

    return re.sub(r"\s+", " ", normalized).strip()


def _compact_budget(text: str, *, chinese_budget: int, non_chinese_budget: int) -> int:
    return chinese_budget if CHINESE_PATTERN.search(text) else non_chinese_budget


def _proactive_budget(text: str) -> int:
    return _compact_budget(text, chinese_budget=24, non_chinese_budget=12)


def _measure_segment(text: str) -> int:
    if CHINESE_PATTERN.search(text):
        return len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))
    return len(re.findall(r"[A-Za-z0-9_]+", text))


def _truncate_segment(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if CHINESE_PATTERN.search(text):
        kept: list[str] = []
        units = 0
        for char in text:
            if re.match(r"[\u4e00-\u9fffA-Za-z0-9]", char):
                if units >= budget:
                    break
                units += 1
            kept.append(char)
        return "".join(kept).rstrip(CLAUSE_ENDINGS + " ")

    words = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text)
    kept_words: list[str] = []
    units = 0
    for token in words:
        if re.match(r"[A-Za-z0-9_]+", token):
            if units >= budget:
                break
            units += 1
        kept_words.append(token)
    return re.sub(r"\s+", " ", "".join(kept_words)).strip().rstrip(",:;")


def _ensure_sentence_ending(text: str) -> str:
    tightened = text.strip().rstrip(CLAUSE_ENDINGS + " ")
    if not tightened:
        return ""
    if tightened[-1] in SENTENCE_ENDINGS:
        return tightened
    return tightened + _sentence_punctuation(tightened)


def _strip_proactive_formal_leadins(text: str) -> str:
    tightened = text.strip()
    while tightened:
        updated = PROACTIVE_FORMAL_LEADIN_PATTERN.sub("", tightened, count=1).strip()
        if updated == tightened or not updated:
            return tightened
        tightened = updated
    return text.strip()


def _normalize_compact_chat_reply(
    text: str,
    *,
    budget: int,
    strip_formal_leadins: bool,
    prefer_first_sentence_only: bool,
) -> str:
    normalized = normalize_chat_reply(text)
    if not normalized:
        return normalized
    if strip_formal_leadins:
        normalized = _strip_proactive_formal_leadins(normalized)

    sentence_match = re.match(rf"^(.+?[{re.escape(SENTENCE_ENDINGS)}])", normalized)
    if sentence_match and prefer_first_sentence_only:
        first_sentence = sentence_match.group(1).strip()
        if _measure_segment(first_sentence) <= budget:
            return first_sentence

    segments = [segment.strip() for segment in CLAUSE_PATTERN.findall(normalized) if segment.strip()]
    if not segments:
        return _ensure_sentence_ending(_truncate_segment(normalized, budget))

    selected: list[str] = []
    used = 0
    sentence_count = 0
    for segment in segments:
        segment_units = _measure_segment(segment)
        if not selected and segment_units > budget:
            return _ensure_sentence_ending(_truncate_segment(segment, budget))
        if selected and used + segment_units > budget:
            break
        selected.append(segment)
        used += segment_units
        if segment[-1] in SENTENCE_ENDINGS:
            sentence_count += 1
            if prefer_first_sentence_only or sentence_count >= 2:
                break

    tightened = "".join(selected).strip()
    if not tightened:
        return _ensure_sentence_ending(_truncate_segment(normalized, budget))
    return _ensure_sentence_ending(tightened)


def normalize_proactive_chat_reply(text: str) -> str:
    normalized = normalize_chat_reply(text)
    if not normalized:
        return normalized
    return _strip_proactive_formal_leadins(normalized)


def normalize_brief_group_interjection_reply(text: str) -> str:
    normalized = normalize_chat_reply(text)
    if not normalized:
        return normalized
    return _strip_proactive_formal_leadins(normalized)
