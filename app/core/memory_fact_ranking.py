"""Query-relevance ranking for per-member memory facts."""

from __future__ import annotations

import re
from typing import Any, Mapping, Protocol, Sequence

from app.core.time_utils import stored_as_utc


_CJK = re.compile(r"[\u4e00-\u9fff]+")

_KIND_INTENT_PATTERNS: tuple[tuple[tuple[str, ...], re.Pattern[str]], ...] = (
    (("taboo", "preference"), re.compile(r"讨厌|不喜欢|反感")),
    (("preference",), re.compile(r"喜欢|偏好|最爱|爱看|爱听|爱吃|爱喝|爱玩|主人|称呼")),
    (("running_joke",), re.compile(r"什么梗|有啥梗|有什么梗|梗")),
    (("relationship",), re.compile(r"什么关系|和谁|和什么人|关系")),
    (("plan",), re.compile(r"打算|计划|准备做")),
    (("decision",), re.compile(r"决定")),
    (("current",), re.compile(r"最近在做什么|在做什么|在干嘛|干什么")),
    (("event",), re.compile(r"最近发生|发生了什么|发生什么")),
    (("profile",), re.compile(r"介绍|是什么样的人|画像|哪里人|做什么的")),
)


def preferred_kinds_for_query(*, query: str, answer_mode: str) -> tuple[str, ...]:
    """Return the fact kinds that match the question intent.

    Intent wins over the generic current-fact default so that "有什么梗" boosts
    running_joke, "我讨厌什么" boosts taboo, and "最近在做什么" boosts current
    instead of being crowded out by preference/profile facts.
    """
    text = str(query or "").strip()
    for kinds, pattern in _KIND_INTENT_PATTERNS:
        if pattern.search(text):
            return kinds
    if answer_mode == "current_fact":
        return ("preference", "taboo", "profile")
    return ()


def _recency_value(fact: RankableMemoryFact) -> float:
    for attribute in ("last_seen_at", "valid_from"):
        value = getattr(fact, attribute, None)
        if value is None:
            continue
        try:
            return float(stored_as_utc(value).timestamp())
        except (AttributeError, OSError, ValueError, TypeError):
            continue
    return 0.0


class RankableMemoryFact(Protocol):
    content: str
    predicate: str
    object_text: str
    memory_kind: str
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
    preferred_kinds: Sequence[str] = (),
    semantic_scores: Mapping[int, float] | None = None,
) -> list[RankableMemoryFact]:
    """Rank member facts by query relevance, then importance/confidence.

    A fact matches when any query feature appears in its content, predicate or
    object text. Facts whose kind is preferred (for example preference/taboo
    when the question asks about likes) outrank non-preferred kinds; matched
    facts outrank unmatched ones; importance, confidence and id break ties
    deterministically.
    """
    if limit <= 0:
        return []
    normalized_features = tuple(
        dict.fromkeys(str(value) for value in query_features if str(value).strip())
    )
    preferred = frozenset(
        str(kind).strip() for kind in preferred_kinds if str(kind).strip()
    )
    normalized_semantic = {
        int(fact_id): max(0.0, min(1.0, float(score)))
        for fact_id, score in (semantic_scores or {}).items()
    }
    scored: list[
        tuple[bool, bool, float, float, float, float, int, RankableMemoryFact]
    ] = []
    for fact in facts:
        haystack = " ".join(
            str(value)
            for value in (fact.content, fact.predicate, fact.object_text)
            if str(value or "").strip()
        )
        matched = any(feature in haystack for feature in normalized_features)
        kind = str(getattr(fact, "memory_kind", "") or "").strip()
        semantic = normalized_semantic.get(int(fact.id or 0), 0.0)
        recency = _recency_value(fact)
        scored.append(
            (
                kind in preferred,
                matched,
                semantic,
                float(fact.importance or 1),
                float(fact.confidence or 0.0),
                recency,
                int(fact.id or 0),
                fact,
            )
        )
    scored.sort(
        key=lambda item: (
            item[0],
            item[2],
            item[1],
            item[3],
            item[4],
            item[5],
            item[6],
        ),
        reverse=True,
    )
    return [item[7] for item in scored[: int(limit)]]


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
