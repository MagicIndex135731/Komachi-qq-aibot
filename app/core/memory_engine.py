from __future__ import annotations

from collections import Counter
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


def retrieve_relevant_memories(query: str, memories: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    query_counts = Counter(tokenize_text(query))

    def score(memory: dict[str, Any]) -> tuple[int, int]:
        content_tokens = Counter(tokenize_text(str(memory.get("content", ""))))
        overlap = sum((query_counts & content_tokens).values())
        return overlap, int(memory.get("importance", 0))

    ranked = sorted(memories, key=score, reverse=True)
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
