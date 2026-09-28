"""Deterministic, precision-first retrieval for persona speech examples.

This module owns the boundary between chat content and dynamic style evidence.
It deliberately permits an empty result: stable persona instructions are a
safer fallback than an unrelated verbatim example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
import hashlib
import math
import re
from statistics import median
from typing import Iterable, Mapping, Sequence


DOCUMENT_SCHEMA = "style-situation-v2"
MAX_SELECTED_EXAMPLES = 3
MAX_QUERY_FRAGMENT_CHARS = 160
MAX_QUERY_FRAGMENTS = 4
MAX_QUERY_CHARS = 480
MAX_SITUATION_CHARS = 80
MAX_REPLY_CHARS = 80
MAX_PROMPT_BLOCK_CHARS = 600
MAX_DYNAMIC_SAMPLE_CHARS = 120
SEMANTIC_WEIGHT = 0.82
LEXICAL_WEIGHT = 0.13
RECENCY_WEIGHT = 0.05
RECENCY_HALF_LIFE_DAYS = 180.0
MMR_RELEVANCE_WEIGHT = 0.82
MMR_DIVERSITY_WEIGHT = 0.18

_MEDIA_ONLY_RE = re.compile(
    r"^(?:\[(?:图片|视频|语音|表情|文件|image|video|audio|sticker|file)[^\]]*\]|"
    r"\[CQ:[^\]]+\]|[\s\W_])+$",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_URL_ONLY_RE = re.compile(r"^(?:\s*(?:https?://\S+|www\.\S+)\s*)+$", re.IGNORECASE)
_CONTROL_RE = re.compile(
    r"^(?:切换人格为\s*[:：]|/|！|!)(?:\s|\S)*$|^(?:重载|reload|status|help)$",
    re.IGNORECASE,
)
_LEADING_MENTION_RE = re.compile(r"^\s*@\S+\s*")
_SPEAKER_PREFIX_RE = re.compile(r"^.{1,32}?[：:]\s*(.+)$", re.DOTALL)
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)、]|[一二三四五六七八九十]+[、.])\s+", re.MULTILINE)
_WORD_RE = re.compile(r"[a-z][a-z0-9_+.-]*|\d+(?:\.\d+)?", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u9fff]+")
_EFFECTIVE_RE = re.compile(r"[\u3400-\u9fffa-zA-Z0-9]")
_SHORT_FOLLOWUP_RE = re.compile(
    r"(?:^|[，,。.!！？?\s])(?:谁|哪个|哪位|名字呢?|这个呢?|那个呢?|然后呢?|后来呢?|"
    r"为什么|为啥|咋了|怎么了|真的吗|还有呢?|具体呢?|什么意思)(?:[啊呀呢嘛吗吧？?！!。.]*)$"
)

# Only whole lexical units are filtered. Single characters are never used as
# evidence, so common Chinese function words do not create accidental hits.
_LEXICAL_STOP_UNITS = {
    "什么", "怎么", "为什么", "为啥", "这个", "那个", "哪个", "然后",
    "还是", "就是", "可以", "觉得", "一下", "一个", "有没有", "是不是",
    "了吗", "呢啊", "啊啊", "哈哈", "知道", "真的", "现在", "最近",
}


@dataclass(frozen=True, slots=True)
class StyleRetrievalQuery:
    current_text: str
    quoted_text: str | None
    continuation_lines: tuple[str, ...]
    semantic_text: str
    lexical_units: frozenset[str]
    current_message_id: str = ""
    fragment_count: int = 0


@dataclass(frozen=True, slots=True)
class StyleDocument:
    msg_id: str
    user_id: int
    group_id: int
    situation: str
    reply: str
    situation_kind: str
    canonical_text: str
    document_hash: str
    timestamp: datetime | None
    entry: dict = field(compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class StyleExampleMatch:
    entry: dict
    document: StyleDocument
    semantic_score: float
    lexical_score: float
    recency_score: float
    final_score: float


@dataclass(slots=True)
class StyleRetrievalTrace:
    query_fragment_count: int = 0
    query_chars: int = 0
    candidate_count: int = 0
    quality_rejected: int = 0
    threshold_rejected: int = 0
    selected_ids: list[str] = field(default_factory=list)
    selected_scores: list[float] = field(default_factory=list)
    fallback_mode: str = "none"
    rejection_reason: str = ""


def _field(value: object, name: str, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalized_timestamp(value: object) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def normalize_style_text(value: object) -> str:
    text = str(value or "").replace("\u3000", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _without_speaker_prefix(value: object) -> str:
    text = normalize_style_text(value)
    match = _SPEAKER_PREFIX_RE.match(text)
    return normalize_style_text(match.group(1)) if match else text


def _query_fragment(value: object) -> str:
    text = _LEADING_MENTION_RE.sub("", normalize_style_text(value))
    if not text or _CONTROL_RE.match(text) or _URL_ONLY_RE.match(text) or _MEDIA_ONLY_RE.match(text):
        return ""
    return text[:MAX_QUERY_FRAGMENT_CHARS].strip()


def _without_subject_terms(value: object, subject_terms: Iterable[object]) -> str:
    text = normalize_style_text(value)
    terms = sorted(
        {
            normalize_style_text(term)
            for term in subject_terms
            if len(normalize_style_text(term)) >= 2
        },
        key=len,
        reverse=True,
    )
    for term in terms:
        text = re.sub(re.escape(term), "", text, flags=re.IGNORECASE)
    return text.strip(" ，,。.!！？?：:、")


def lexical_units(value: object) -> frozenset[str]:
    """Return discriminative Chinese 2/3-grams and complete latin tokens."""

    text = normalize_style_text(value).lower()
    units: set[str] = set(_WORD_RE.findall(text))
    for run in _CJK_RE.findall(text):
        for width in (2, 3):
            units.update(run[index : index + width] for index in range(len(run) - width + 1))
    return frozenset(unit for unit in units if unit and unit not in _LEXICAL_STOP_UNITS)


def is_short_followup(value: object) -> bool:
    text = _query_fragment(value)
    if not text or len(text) > 12:
        return False
    compact = re.sub(r"[\s，,。.!！？?]", "", text)
    return bool(_SHORT_FOLLOWUP_RE.search(text) or len(compact) <= 3)


def build_style_retrieval_query(
    *,
    current_text: object,
    quoted_text: object | None = None,
    recent_messages: Sequence[object] = (),
    current_timestamp: datetime | None = None,
    current_message_id: object = "",
    bot_user_id: int | None = None,
    subject_terms: Iterable[object] = (),
) -> StyleRetrievalQuery:
    """Build a narrow query; the general prompt history is never accepted."""

    current = _query_fragment(_without_subject_terms(current_text, subject_terms))
    quoted = _query_fragment(_without_subject_terms(quoted_text, subject_terms))
    fragments: list[tuple[str, str]] = []
    if current:
        fragments.append(("当前问题", current))
    if quoted and quoted != current:
        fragments.append(("引用内容", quoted))

    continuation: list[str] = []
    short_quote_anchor = bool(quoted and is_short_followup(current))
    if (not quoted and is_short_followup(current)) or short_quote_anchor:
        now = _normalized_timestamp(current_timestamp)
        ordered_recent = sorted(
            recent_messages,
            key=lambda message: (
                _normalized_timestamp(_field(message, "timestamp", None))
                or datetime.min.replace(tzinfo=UTC),
                str(_field(message, "platform_msg_id", _field(message, "msg_id", "")) or ""),
            ),
        )
        for message in reversed(ordered_recent):
            msg_id = str(_field(message, "platform_msg_id", _field(message, "msg_id", "")) or "")
            if msg_id and msg_id == str(current_message_id or ""):
                continue
            user_id = _field(message, "user_id", None)
            if bot_user_id is not None and str(user_id) == str(bot_user_id):
                continue
            timestamp = _normalized_timestamp(_field(message, "timestamp", None))
            if now is not None and timestamp is not None:
                age = (now - timestamp).total_seconds()
                if age > 120:
                    break
                if age < -5:
                    continue
            fragment = _query_fragment(
                _without_subject_terms(
                    _field(message, "plain_text", _field(message, "text", "")),
                    subject_terms,
                )
            )
            if not fragment or fragment in {current, quoted, *continuation}:
                continue
            continuation.append(fragment)
            if len(continuation) >= 3:
                break
        continuation.reverse()
        fragments.extend(("紧邻上文", value) for value in continuation)

    fragments = fragments[:MAX_QUERY_FRAGMENTS]
    semantic_parts: list[str] = []
    remaining = MAX_QUERY_CHARS
    for label, value in fragments:
        piece = f"{label}：{value}"
        if remaining <= 0:
            break
        semantic_parts.append(piece[:remaining])
        remaining -= len(semantic_parts[-1])
    semantic_text = "\n".join(semantic_parts)
    lexical_text = " ".join(value for _, value in fragments)
    return StyleRetrievalQuery(
        current_text=current,
        quoted_text=quoted or None,
        continuation_lines=tuple(continuation),
        semantic_text=semantic_text,
        lexical_units=lexical_units(lexical_text),
        current_message_id=str(current_message_id or ""),
        fragment_count=len(semantic_parts),
    )


def dynamic_sample_quality_reason(text: object) -> str | None:
    value = normalize_style_text(text)
    if not value:
        return "empty"
    if _CONTROL_RE.match(value):
        return "control"
    if _MEDIA_ONLY_RE.match(value) or _URL_ONLY_RE.match(value):
        return "non_text"
    if len(value) > MAX_DYNAMIC_SAMPLE_CHARS:
        return "too_long"
    if value.count("\n") >= 3 or len(_LIST_LINE_RE.findall(value)) >= 3:
        return "structured_dump"
    if "```" in value or len(_URL_RE.findall(value)) >= 2:
        return "structured_dump"
    if len(_EFFECTIVE_RE.findall(value)) < 2:
        return "too_short"
    return None


def build_style_document(
    entry: Mapping[str, object] | dict,
    *,
    expected_user_id: int | None = None,
    expected_group_id: int | None = None,
) -> StyleDocument | None:
    user_id = int(entry.get("user_id") or expected_user_id or 0)
    group_id = int(entry.get("group_id") or expected_group_id or 0)
    if expected_user_id is not None and user_id != int(expected_user_id):
        return None
    if expected_group_id is not None and group_id != int(expected_group_id):
        return None
    reply = normalize_style_text(entry.get("text"))
    if dynamic_sample_quality_reason(reply) is not None:
        return None
    reply_target = _without_speaker_prefix(entry.get("reply_target"))
    if reply_target and dynamic_sample_quality_reason(reply_target) is not None:
        reply_target = ""
    situation_kind = "quoted" if reply_target else "reply_only"
    situation = reply_target
    if not situation:
        before = entry.get("context_before") or []
        if isinstance(before, Sequence) and not isinstance(before, (str, bytes)):
            for item in reversed(before):
                candidate = _without_speaker_prefix(
                    item.get("text") if isinstance(item, Mapping) else item
                )
                if candidate and dynamic_sample_quality_reason(candidate) is None:
                    situation = candidate
                    situation_kind = "adjacent"
                    break
    if situation:
        situation = situation[:MAX_SITUATION_CHARS].strip()
        canonical = f"对话情境：{situation}\n本人回应：{reply}"
    else:
        canonical = f"本人回应：{reply}"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    timestamp = _normalized_timestamp(entry.get("timestamp"))
    return StyleDocument(
        msg_id=str(entry.get("msg_id") or ""),
        user_id=user_id,
        group_id=group_id,
        situation=situation,
        reply=reply,
        situation_kind=situation_kind,
        canonical_text=canonical,
        document_hash=digest,
        timestamp=timestamp,
        entry=dict(entry),
    )


def cosine_similarity(left: Sequence[float] | None, right: Sequence[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    try:
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        norm_left = math.sqrt(sum(float(a) * float(a) for a in left))
        norm_right = math.sqrt(sum(float(b) * float(b) for b in right))
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not norm_left or not norm_right or not math.isfinite(dot):
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_left * norm_right)))


def _lexical_score(query_units: frozenset[str], document: StyleDocument) -> float:
    if not query_units:
        return 0.0
    candidate_units = lexical_units(f"{document.situation} {document.reply}")
    overlap = query_units & candidate_units
    if not overlap:
        return 0.0
    # Query coverage rewards direct evidence without favoring long copied text.
    return min(1.0, len(overlap) / max(1, min(6, len(query_units))))


def _recency_score(timestamp: datetime | None, now: datetime) -> float:
    if timestamp is None:
        return 0.0
    age_days = max(0.0, (now - timestamp).total_seconds() / 86400.0)
    return math.exp(-math.log(2.0) * age_days / RECENCY_HALF_LIFE_DAYS)


def _semantic_threshold(document: StyleDocument, lexical_score: float) -> float:
    threshold = 0.56 if lexical_score >= 0.25 else 0.68
    if document.situation_kind == "adjacent":
        threshold += 0.01
    elif document.situation_kind == "reply_only":
        threshold += 0.04
    return threshold


def _reply_ngram_jaccard(left: str, right: str) -> float:
    left_units = lexical_units(left)
    right_units = lexical_units(right)
    if not left_units or not right_units:
        return 1.0 if normalize_style_text(left) == normalize_style_text(right) else 0.0
    return len(left_units & right_units) / len(left_units | right_units)


def _near_duplicate_text(left: object, right: object) -> bool:
    """Reject query/recent-output echoes while keeping topical responses."""

    normalized_left = re.sub(
        r"[^\u3400-\u9fffa-zA-Z0-9]+", "", normalize_style_text(left).lower()
    )
    normalized_right = re.sub(
        r"[^\u3400-\u9fffa-zA-Z0-9]+", "", normalize_style_text(right).lower()
    )
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True
    if min(len(normalized_left), len(normalized_right)) < 6:
        return False
    return SequenceMatcher(None, normalized_left, normalized_right).ratio() >= 0.78


def rank_style_examples(
    *,
    query: StyleRetrievalQuery,
    documents: Sequence[StyleDocument],
    vectors_by_id: Mapping[str, Sequence[float]],
    query_vector: Sequence[float] | None,
    limit: int = MAX_SELECTED_EXAMPLES,
    exclude_texts: Iterable[object] = (),
    now: datetime | None = None,
    trace: StyleRetrievalTrace | None = None,
) -> list[StyleExampleMatch]:
    trace = trace if trace is not None else StyleRetrievalTrace()
    trace.query_fragment_count = query.fragment_count
    trace.query_chars = len(query.semantic_text)
    trace.candidate_count = len(documents)
    resolved_now = _normalized_timestamp(now) or datetime.now(UTC)
    excluded = [
        normalize_style_text(value).lower()
        for value in exclude_texts
        if normalize_style_text(value)
    ]
    if query.current_text:
        excluded.append(normalize_style_text(query.current_text).lower())

    def reply_is_excluded(document: StyleDocument) -> bool:
        return any(
            _near_duplicate_text(document.reply, excluded_text)
            for excluded_text in excluded
        )

    if query_vector is None:
        trace.fallback_mode = "lexical"
        lexical_matches: list[StyleExampleMatch] = []
        for document in documents:
            if reply_is_excluded(document):
                continue
            lexical = _lexical_score(query.lexical_units, document)
            if lexical <= 0:
                trace.threshold_rejected += 1
                continue
            recency = _recency_score(document.timestamp, resolved_now)
            lexical_matches.append(
                StyleExampleMatch(
                    entry=document.entry,
                    document=document,
                    semantic_score=0.0,
                    lexical_score=lexical,
                    recency_score=recency,
                    final_score=0.95 * lexical + 0.05 * recency,
                )
            )
        candidates = lexical_matches
    else:
        trace.fallback_mode = "semantic"
        semantic_rows: list[tuple[float, StyleDocument]] = []
        for document in documents:
            vector = vectors_by_id.get(document.msg_id)
            if vector is None:
                continue
            semantic_rows.append((max(0.0, cosine_similarity(query_vector, vector)), document))
        semantic_rows.sort(key=lambda row: (-row[0], row[1].msg_id))
        semantic_rows = semantic_rows[:32]
        candidates = []
        for semantic, document in semantic_rows:
            if reply_is_excluded(document):
                continue
            lexical = _lexical_score(query.lexical_units, document)
            recency = _recency_score(document.timestamp, resolved_now)
            final = SEMANTIC_WEIGHT * semantic + LEXICAL_WEIGHT * lexical + RECENCY_WEIGHT * recency
            strong_lexical = lexical >= 0.25
            if semantic < _semantic_threshold(document, lexical) or (
                not strong_lexical and final < 0.60
            ):
                trace.threshold_rejected += 1
                continue
            candidates.append(
                StyleExampleMatch(
                    entry=document.entry,
                    document=document,
                    semantic_score=semantic,
                    lexical_score=lexical,
                    recency_score=recency,
                    final_score=final,
                )
            )
        all_scores = [row[0] for row in semantic_rows]
        if candidates and not any(match.lexical_score >= 0.25 for match in candidates):
            top = max(match.semantic_score for match in candidates)
            if all_scores and top - median(all_scores) < 0.08:
                trace.threshold_rejected += len(candidates)
                candidates = []
                trace.rejection_reason = "ambiguous_semantic"
        if not candidates:
            # A short quoted scene can be semantically underrepresented in a
            # sparse style bank. Permit one bounded lexical retry only when
            # enough scene fragments survived and one candidate has strong
            # direct overlap. Never lower the global semantic threshold or
            # fill an otherwise unrelated query.
            lexical_fallback: list[StyleExampleMatch] = []
            if query.fragment_count >= 3:
                for document in documents:
                    if reply_is_excluded(document):
                        continue
                    lexical = _lexical_score(query.lexical_units, document)
                    if lexical < 0.5:
                        continue
                    recency = _recency_score(document.timestamp, resolved_now)
                    lexical_fallback.append(
                        StyleExampleMatch(
                            entry=document.entry,
                            document=document,
                            semantic_score=0.0,
                            lexical_score=lexical,
                            recency_score=recency,
                            final_score=0.95 * lexical + 0.05 * recency,
                        )
                    )
            if lexical_fallback:
                candidates = lexical_fallback
                trace.fallback_mode = "semantic_then_lexical"
                trace.rejection_reason = "semantic_empty_strong_lexical"
            else:
                trace.fallback_mode = "safe_empty"
                if not trace.rejection_reason:
                    trace.rejection_reason = "no_qualified_style_match"

    candidates.sort(
        key=lambda item: (
            -item.final_score,
            -(item.document.timestamp.timestamp() if item.document.timestamp else 0.0),
            item.document.msg_id,
        )
    )
    deduped: list[StyleExampleMatch] = []
    for candidate in candidates:
        if any(
            normalize_style_text(candidate.document.reply).lower()
            == normalize_style_text(existing.document.reply).lower()
            or _reply_ngram_jaccard(candidate.document.reply, existing.document.reply) >= 0.85
            for existing in deduped
        ):
            continue
        deduped.append(candidate)

    selected: list[StyleExampleMatch] = []
    remaining = list(deduped[:12])
    max_count = min(MAX_SELECTED_EXAMPLES, max(0, int(limit)))
    while remaining and len(selected) < max_count:
        def mmr_value(match: StyleExampleMatch) -> tuple[float, float, str]:
            vector = vectors_by_id.get(match.document.msg_id)
            similarity = max(
                (cosine_similarity(vector, vectors_by_id.get(item.document.msg_id)) for item in selected),
                default=0.0,
            )
            return (
                MMR_RELEVANCE_WEIGHT * match.final_score - MMR_DIVERSITY_WEIGHT * max(0.0, similarity),
                match.final_score,
                match.document.msg_id,
            )

        best = max(remaining, key=mmr_value)
        remaining.remove(best)
        selected.append(best)

    trace.selected_ids = [match.document.msg_id for match in selected]
    trace.selected_scores = [round(match.final_score, 4) for match in selected]
    if not selected and not trace.rejection_reason:
        trace.rejection_reason = "no_qualified_candidate"
    return selected


def format_style_example_block(
    entries: Sequence[Mapping[str, object] | dict],
    *,
    max_pairs: int = MAX_SELECTED_EXAMPLES,
    max_chars: int = MAX_PROMPT_BLOCK_CHARS,
) -> str:
    """Render bounded style-only examples from canonical v2 fields."""

    pairs: list[str] = []
    budget = max(0, min(MAX_PROMPT_BLOCK_CHARS, int(max_chars)))
    for entry in list(entries)[: min(MAX_SELECTED_EXAMPLES, max(0, int(max_pairs)))]:
        document = build_style_document(entry)
        if document is None:
            continue
        reply = document.reply[:MAX_REPLY_CHARS].strip()
        if document.situation:
            pair = f"情境「{document.situation[:MAX_SITUATION_CHARS]}」→ 本人原话「{reply}」"
        else:
            pair = f"本人原话「{reply}」"
        if len(pair) > budget:
            break
        pairs.append(pair)
        budget -= len(pair) + 1
    return "；".join(pairs)
