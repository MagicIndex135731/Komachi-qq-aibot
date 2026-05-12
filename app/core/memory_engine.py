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
