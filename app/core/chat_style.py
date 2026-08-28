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


def build_human_chat_style_lines(
    *, proactive_turn: bool = False, komachi_style: bool = True
) -> list[str]:
    lines = [
        "Talk like a real person chatting in a group.",
    ]
    if komachi_style:
        lines.extend(
            [
                "Keep Komachi's mesugaki edge in every reply: smug, sharp-tongued, lightly superior, and end with a teasing jab instead of a neutral wrap-up.",
                "Never soften into polite customer-service tone; a short mocking or smug retort is the default, not an exception.",
            ]
        )
    lines.extend(
        [
        "Do not use Markdown, headings, bullet lists, numbered lists, or checklist formatting in normal replies.",
        "If someone wants a detailed explanation, stay conversational and explain in natural paragraphs instead of notes or tutorial formatting.",
        "Do not use stock assistant transitions like first, second, in summary, or here are a few points.",
        ]
    )
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
        if komachi_style:
            lines.extend(
                [
                    "For proactive interjections, lean into Komachi's mesugaki personality: smug, cheeky, lightly teasing the speaker like catching them doing something silly.",
                    "For proactive interjections, short teasing quips are welcome (like 不会吧不会吧、这都要小町来提醒、欸~), vary them and tie them to the topic.",
                    "For proactive interjections, keep the teasing playful and light, never mean or lecturing.",
                    "For proactive interjections, be sharp and provocative: mock the point, play superior, and land a smug jab instead of agreeing.",
                    "For proactive interjections, favor teasing put-downs like 就这？ or 不会吧不会吧 when the topic invites it.",
                ]
            )
        lines.append(
            "For proactive interjections, this is a sharp roast, not a speech: output exactly ONE short line (usually 10-20 Chinese characters), land the jab, and stop."
        )
        lines.append(
            "For proactive interjections, never write a paragraph, never join clauses into a long run-on, never explain, recap, or conclude."
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
    normalized = _strip_proactive_formal_leadins(normalized)
    # Keep the first complete sentence as a safety net for long drafts; never
    # cut mid-sentence. The model is instructed to output one short roast line.
    sentence_match = re.match(rf"^(.+?[{re.escape(SENTENCE_ENDINGS)}])", normalized)
    if sentence_match:
        return sentence_match.group(1).strip()
    return normalized


def split_burst_reply(text: str, burst: dict | None) -> list[str]:
    """Split one reply into a short message burst when the persona allows it.

    A burst is enabled by a persona-level ``burst`` mapping with ``enabled:
    true``. The model joins burst messages with the configured separator; this
    function splits them back into individual QQ messages. Personas without
    burst configuration always reply with a single message.
    """

    normalized = str(text or "").strip()
    if not normalized or not isinstance(burst, dict) or not burst.get("enabled"):
        return [normalized] if normalized else []
    separator = str(burst.get("separator") or "|")
    max_messages = max(1, min(6, int(burst.get("max_messages") or 3)))
    parts = [part.strip() for part in normalized.split(separator)]
    parts = [part for part in parts if part]
    max_chars = max(8, int(burst.get("max_chars") or 24))
    if len(parts) == 1 and len(parts[0]) > max_chars:
        parts = _split_long_segment(parts[0], max_chars)
    if len(parts) < 2:
        return [normalized]
    if len(parts) > max_messages:
        overflow = parts[max_messages - 1 :]
        parts = parts[: max_messages - 1] + [separator.join(overflow)]
    return [part for part in parts if part]


def normalize_chat_reply_burst_aware(text: str, burst: dict | None) -> str:
    """Normalize a reply while preserving burst separators and line breaks."""

    if not isinstance(burst, dict) or not burst.get("enabled"):
        return normalize_chat_reply(text)
    separator = str(burst.get("separator") or "|")
    pieces = re.split(rf"{re.escape(separator)}|\n+", str(text or ""))
    normalized = [normalize_chat_reply(piece) for piece in pieces]
    normalized = [piece for piece in normalized if piece]
    if not normalized:
        return normalize_chat_reply(text)
    return separator.join(normalized)


def _split_long_segment(text: str, limit: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for char in str(text):
        current += char
        if len(current) >= limit and char in SENTENCE_ENDINGS:
            chunks.append(current)
            current = ""
    if current.strip():
        chunks.append(current)
    cleaned = [chunk.strip() for chunk in chunks if chunk.strip()]
    return cleaned or [str(text).strip()]


def scrub_banned_address_terms(
    text: str,
    terms: tuple[str, ...] | list[str],
    *,
    replacement: str = "你",
) -> str:
    """Deterministic backstop: remove honorific/maid-style address terms."""

    scrubbed = str(text or "")
    for term in terms:
        if term:
            scrubbed = scrubbed.replace(term, replacement)
    scrubbed = re.sub(r"你{2,}", "你", scrubbed)
    return scrubbed


def _topic_units(text: str) -> set[str]:
    compact = re.sub(r"[\s\u3000]+", "", str(text or "").lower())
    units = {compact[index : index + 2] for index in range(len(compact) - 1)}
    stop_chars = set(
        "的了是我你他她它吗呢啊吧呀在就都很不这那什么怎么哪个有和与跟给对向把被让说做要想能会可以没别去来看回上下中前后里外天点个些再还也才只"
    )
    units.update(
        char
        for char in compact
        if "\u4e00" <= char <= "\u9fff" and char not in stop_chars
    )
    units.update(token for token in re.findall(r"[a-z0-9]+", compact))
    units.discard("")
    return units


def retrieve_relevant_examples(
    bank: list[str] | tuple[str, ...] | list[dict],
    context_lines: list[str],
    *,
    limit: int = 6,
) -> list[dict]:
    """Pick persona examples by topic overlap with the current conversation.

    Each returned entry carries its conversation context so the caller can
    show "上文「…」→ 他回「…」" pairs instead of a bare quote. Context text
    participates in matching, so an example ranks high only when its
    *situation* resembles the current chat.
    """

    context_units = _topic_units(
        "\n".join(str(line).split(":", 1)[-1] for line in context_lines)
    )
    scored: list[tuple[int, int, dict]] = []
    seen: set[str] = set()
    for example in bank or []:
        if isinstance(example, dict):
            text = str(example.get("text") or "").strip()
            context_before = [
                str(item.get("text") or "").strip()
                for item in (example.get("context_before") or [])
                if isinstance(item, dict)
            ]
            reply_target = str(example.get("reply_target") or "").strip()
        else:
            text = str(example).strip()
            context_before = []
            reply_target = ""
        if not text or text in seen:
            continue
        seen.add(text)
        matchable = " ".join([*context_before, reply_target, text])
        overlap = len(context_units & _topic_units(matchable))
        if overlap:
            scored.append(
                (
                    overlap,
                    len(text),
                    {
                        "text": text,
                        "context_before": context_before,
                        "reply_target": reply_target,
                    },
                )
            )
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [entry for _, _, entry in scored[: max(0, limit)]]


def format_example_pairs(entries: list[dict], *, max_pairs: int = 4) -> str:
    """Render retrieved examples as context→reply pairs."""

    pairs: list[str] = []
    for entry in entries[: max(0, max_pairs)]:
        lead = entry.get("reply_target") or (
            (entry.get("context_before") or [""])[-1]
        )
        if lead:
            pairs.append(f"上文「{lead}」→ 他回「{entry.get('text')}」")
        else:
            pairs.append(f"他回「{entry.get('text')}」")
    return "；".join(pairs)


def retrieve_relevant_facts(
    facts: list[dict] | tuple[dict, ...],
    context_lines: list[str],
    *,
    limit: int = 5,
) -> list[dict]:
    """Pick member facts by topic overlap with the current conversation."""

    context_units = _topic_units(
        "\n".join(str(line).split(":", 1)[-1] for line in context_lines)
    )
    scored: list[tuple[int, int, dict]] = []
    seen: set[str] = set()
    for fact in facts or []:
        if isinstance(fact, dict):
            text = f"{fact.get('category') or ''} {fact.get('fact') or ''}"
        else:
            text = str(fact)
        key = str(fact.get("fact") if isinstance(fact, dict) else fact).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        overlap = len(context_units & _topic_units(text))
        if overlap:
            scored.append((overlap, len(text), fact if isinstance(fact, dict) else {"fact": text}))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [fact for _, _, fact in scored[: max(0, limit)]]
