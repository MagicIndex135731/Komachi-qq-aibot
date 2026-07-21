from __future__ import annotations

from collections import Counter
from datetime import datetime
import re
from typing import Any

from app.providers.embeddings import tokenize_text


LIKE_PATTERNS = (
    re.compile(r"^(?P<speaker>[^:\uff1a]+)[:\uff1a]\s*I like\s+(?P<thing>.+?)[.!?]?$", re.IGNORECASE),
    re.compile(r"^(?P<speaker>[^:\uff1a]+)[:\uff1a].*?\u559c\u6b22(?P<thing>.+?)[\u3002\uff01\uff1f]?$"),
)
DISLIKE_PATTERNS = (
    re.compile(r"^(?P<speaker>[^:\uff1a]+)[:\uff1a]\s*I (?:do not|don't) like\s+(?P<thing>.+?)[.!?]?$", re.IGNORECASE),
    re.compile(r"^(?P<speaker>[^:\uff1a]+)[:\uff1a].*?\u4e0d\u559c\u6b22(?P<thing>.+?)[\u3002\uff01\uff1f]?$"),
)

PLAN_PATTERNS = (
    re.compile(
        r"^(?P<speaker>[^:\uff1a]+)[:\uff1a]\s*(?P<content>.*?(?:\u6253\u7b97|\u51c6\u5907|\u8ba1\u5212|\u60f3\u8981|\u4f1a\u53bb|\u8981\u53bb|\u7ea6\u4e86).+)$"
    ),
    re.compile(
        r"^(?P<speaker>[^:\uff1a]+)[:\uff1a]\s*(?P<content>.*?\b(?:plan(?:ning)?|going to|will)\b.+)$",
        re.IGNORECASE,
    ),
)
DECISION_PATTERNS = (
    re.compile(r"^(?P<speaker>[^:\uff1a]+)[:\uff1a]\s*(?P<content>.*?(?:\u51b3\u5b9a|\u5c31\u8fd9\u4e48\u5b9a|\u5b9a\u4e86|\u7edf\u4e00|\u6539\u6210|\u53d6\u6d88).+)$"),
    re.compile(
        r"^(?P<speaker>[^:\uff1a]+)[:\uff1a]\s*(?P<content>.*?\b(?:decided|decision|cancel(?:led)?|changed? to)\b.+)$",
        re.IGNORECASE,
    ),
)
CANCELLATION_PATTERNS = (
    re.compile(r"^(?P<speaker>[^:\uff1a]+)[:\uff1a]\s*(?P<content>.*?(?:\u53d6\u6d88|\u4e0d\u53bb\u4e86|\u4e0d\u6253\u7b97|\u8ba1\u5212\u6709\u53d8|\u6539\u671f).*)$"),
    re.compile(
        r"^(?P<speaker>[^:\uff1a]+)[:\uff1a]\s*(?P<content>.*?\b(?:cancel(?:led)?|not going|no longer plan)\b.*)$",
        re.IGNORECASE,
    ),
)

HISTORY_DETAIL_PATTERN = re.compile(
    r"(?:\u4e4b\u524d|\u4ee5\u524d|\u4e0a\u6b21|\u5f53\u65f6|\u540e\u6765|\u5386\u53f2|\u8bb0\u5f97|\u8bf4\u8fc7|\u51b3\u5b9a|\u8ba1\u5212|\u73b0\u5728|\u8fd8\u662f|\u66fe\u7ecf|previously|before|earlier|remember|decided|plan)",
    re.IGNORECASE,
)


def _match_patterns(line: str, patterns: tuple[re.Pattern[str], ...]) -> re.Match[str] | None:
    for pattern in patterns:
        match = pattern.match(line)
        if match:
            return match
    return None


def extract_memory_candidates(*, scope_id: str, source_msg_id: str, lines: list[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for line in lines:
        dislike_match = _match_patterns(line, DISLIKE_PATTERNS)
        if dislike_match:
            speaker = dislike_match.group("speaker").strip()
            thing = dislike_match.group("thing").strip()
            candidates.append(
                {
                    "scope_type": "group",
                    "scope_id": scope_id,
                    "subject_type": "user",
                    "subject_id": speaker,
                    "memory_kind": "taboo",
                    "content": f"{speaker} dislikes {thing}.",
                    "importance": 4,
                    "confidence": 0.8,
                    "source_msg_id": source_msg_id,
                }
            )
            continue

        like_match = _match_patterns(line, LIKE_PATTERNS)
        if like_match:
            speaker = like_match.group("speaker").strip()
            thing = like_match.group("thing").strip()
            candidates.append(
                {
                    "scope_type": "group",
                    "scope_id": scope_id,
                    "subject_type": "user",
                    "subject_id": speaker,
                    "memory_kind": "preference",
                    "content": f"{speaker} likes {thing}.",
                    "importance": 4,
                    "confidence": 0.8,
                    "source_msg_id": source_msg_id,
                }
            )
    return candidates


def extract_structured_memory_candidates(
    *,
    scope_id: str,
    source_msg_id: str,
    lines: list[str],
    observed_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Extract only explicit, source-backed long-lived chat statements.

    This intentionally stays conservative. Every returned record retains its
    source message so later model-assisted consolidation can supersede it
    without turning guesses or jokes into permanent memory.
    """
    candidates: list[dict[str, Any]] = []
    for line in lines:
        normalized = line.strip()
        if not normalized:
            continue
        cancellation_match = _match_patterns(normalized, CANCELLATION_PATTERNS)
        if cancellation_match is not None:
            speaker = cancellation_match.group("speaker").strip()
            content = cancellation_match.group("content").strip().rstrip("\u3002.!?")
            if len(content) >= 4:
                candidates.append(
                    {
                        "scope_type": "group",
                        "scope_id": scope_id,
                        "subject_type": "user",
                        "subject_id": speaker,
                        "memory_kind": "plan",
                        "content": f"{speaker}: {content}.",
                        "importance": 4,
                        "confidence": 0.8,
                        "source_msg_id": source_msg_id,
                        "valid_from": observed_at,
                        # This is consumed by the persistence layer, never sent
                        # to the ORM as a model field.
                        "supersedes_kind": "plan",
                    }
                )
            continue
        for memory_kind, patterns, importance in (
            ("plan", PLAN_PATTERNS, 3),
            ("decision", DECISION_PATTERNS, 4),
        ):
            match = _match_patterns(normalized, patterns)
            if match is None:
                continue
            speaker = match.group("speaker").strip()
            content = match.group("content").strip().rstrip("\u3002.!?")
            if len(content) < 4:
                continue
            candidates.append(
                {
                    "scope_type": "group",
                    "scope_id": scope_id,
                    "subject_type": "user",
                    "subject_id": speaker,
                    "memory_kind": memory_kind,
                    "content": f"{speaker}: {content}.",
                    "importance": importance,
                    "confidence": 0.7,
                    "source_msg_id": source_msg_id,
                    "valid_from": observed_at,
                }
            )
            break
    return candidates


def is_history_detail_query(query: str) -> bool:
    """Whether a question needs a wider, evidence-heavy memory budget."""
    return bool(HISTORY_DETAIL_PATTERN.search(str(query or "")))


def retrieve_relevant_memories(query: str, memories: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    query_counts = Counter(tokenize_text(query))

    def score(memory: dict[str, Any]) -> tuple[int, float, int, float]:
        searchable = " ".join(
            str(memory.get(key, ""))
            for key in ("subject_id", "predicate", "object_text", "content")
        )
        content_tokens = Counter(tokenize_text(searchable))
        overlap = sum((query_counts & content_tokens).values())
        confidence = float(memory.get("confidence", 0.0) or 0.0)
        mention_count = int(memory.get("mention_count", 1) or 1)
        importance = int(memory.get("importance", 0) or 0)
        return overlap, confidence, importance, min(mention_count, 20) / 20

    ranked = [memory for memory in memories if score(memory)[0] > 0]
    ranked.sort(key=score, reverse=True)
    return ranked[:limit]


def history_search_terms(query: str, *, limit: int = 12) -> list[str]:
    """Return compact lexical terms suitable for bounded SQL history recall."""
    normalized = str(query or "").lower()
    terms: list[str] = []
    for chinese_run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        terms.extend(chinese_run[index : index + 2] for index in range(len(chinese_run) - 1))
    terms.extend(token for token in re.findall(r"[a-z0-9_]{2,}", normalized) if len(token) >= 2)
    unique_terms = list(dict.fromkeys(term for term in terms if len(term) >= 2))
    return unique_terms[:limit]


def retrieve_relevant_history(query: str, messages: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    """Rank already-bounded historical candidates and omit zero-signal fallback data."""
    if limit <= 0:
        return []

    query_counts = Counter(tokenize_text(query))
    phrases = history_search_terms(query)
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for message in messages:
        text = str(message.get("plain_text", "")).strip()
        if not text:
            continue
        phrase_hits = sum(1 for phrase in phrases if phrase in text.lower())
        token_overlap = sum((query_counts & Counter(tokenize_text(text))).values())
        if phrase_hits == 0 and token_overlap < 2:
            continue
        score = phrase_hits * 10 + token_overlap
        ranked.append((score, int(message.get("id", 0)), message))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [message for _score, _message_id, message in ranked[:limit]]
