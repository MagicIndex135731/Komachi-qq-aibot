"""Query-relevance ranking for per-member memory facts."""

from __future__ import annotations

import re
from typing import Any, Protocol, Sequence


_CJK = re.compile(r"[\u4e00-\u9fff]+")


class RankableMemoryFact(Protocol):
    content: str
    predicate: str
    object_text: str
    importance: int
    confidence: float
    id: int


def memory_query_features(
    *,
    query: str,
    entities: Sequence[str] = (),
    topic_terms: Sequence[str] = (),
) -> tuple[str, ...]:
    """Extract matching features from a memory query.

    Whole entities/topic terms plus CJK bigrams and trigrams give Chinese
    retrieval a lexical handle even when the FTS tokenizer drops short terms.
    """
    features: set[str] = set()
    for value in (*entities, *topic_terms, query):
        text = str(value or "").strip()
        if not text:
            continue
        if len(text) >= 2:
            features.add(text)
        for size in (2, 3):
            for index in range(max(0, len(text) - size + 1)):
                piece = text[index : index + size]
                if _CJK.fullmatch(piece):
                    features.add(piece)
    return tuple(sorted(features))


def filter_member_query_features(
    features: Sequence[str],
    *,
    aliases: Sequence[str],
) -> tuple[str, ...]:
    """Drop features that merely echo a member alias.

    The member's nickname/card appears in most of that member's facts, so it
    carries no relevance signal. Any feature containing an alias (including
    CJK bigrams such as "阿渣喜") is removed.
    """
    normalized_aliases = tuple(
        dict.fromkeys(
            str(alias).strip()
            for alias in aliases
            if len(str(alias).strip()) >= 2
        )
    )
    if not normalized_aliases:
        return tuple(features)
    kept: list[str] = []
    for feature in features:
        if any(alias in feature for alias in normalized_aliases):
            continue
        kept.append(feature)
    return tuple(kept)


def rank_member_facts(
    facts: Sequence[RankableMemoryFact],
    *,
    query_features: Sequence[str],
    limit: int,
) -> list[RankableMemoryFact]:
    """Rank member facts by query relevance, then importance/confidence.

    A fact matches when any query feature appears in its content, predicate or
    object text. Matched facts always outrank unmatched ones; within each group
    importance, confidence and id break ties deterministically.
    """
    if limit <= 0:
        return []
    normalized_features = tuple(
        dict.fromkeys(str(value) for value in query_features if str(value).strip())
    )
    scored: list[tuple[bool, float, float, int, RankableMemoryFact]] = []
    for fact in facts:
        haystack = " ".join(
            str(value)
            for value in (fact.content, fact.predicate, fact.object_text)
            if str(value or "").strip()
        )
        matched = any(feature in haystack for feature in normalized_features)
        scored.append(
            (
                matched,
                float(fact.importance or 1),
                float(fact.confidence or 0.0),
                int(fact.id or 0),
                fact,
            )
        )
    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    return [item[4] for item in scored[: int(limit)]]


def matching_member_fact_ids(
    facts: Sequence[RankableMemoryFact],
    *,
    query_features: Sequence[str],
) -> set[int]:
    normalized_features = tuple(
        dict.fromkeys(str(value) for value in query_features if str(value).strip())
    )
    matched_ids: set[int] = set()
    for fact in facts:
        haystack = " ".join(
            str(value)
            for value in (fact.content, fact.predicate, fact.object_text)
            if str(value or "").strip()
        )
        if any(feature in haystack for feature in normalized_features):
            matched_ids.add(int(fact.id or 0))
    return matched_ids
