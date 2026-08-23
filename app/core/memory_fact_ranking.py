"""Query-relevance ranking for per-member memory facts."""

from __future__ import annotations

import re
from typing import Any, Mapping, Protocol, Sequence

from app.core.time_utils import stored_as_utc


_CJK = re.compile(r"[\u4e00-\u9fff]+")
_CURRENT_VIEWING_INTENT_PATTERN = re.compile(
    r"(?:(?:最近|现在|目前|近期|当下).{0,12}?(?:正在|在)?"
    r"(?:看|追|补)(?:着)?(?:什么|啥)|"
    r"(?:正在|在)(?:看|追|补)(?:着)?(?:什么|啥))"
)
_CURRENT_VIEWING_FEATURES = ("在看", "观看", "追看", "追番", "补番", "补剧")

# Storage kinds stay separate so each fact keeps one lifecycle and canonical
# identity.  A portrait is a read-time view over these stable personal kinds.
PERSON_PORTRAIT_KINDS = (
    "profile",
    "preference",
    "taboo",
    "relationship",
    "fact",
)
_COMPOSITE_PORTRAIT_PATTERN = re.compile(r"完整.{0,4}画像|个人画像|画像|介绍|是什么样的人")

_KIND_INTENT_PATTERNS: tuple[tuple[tuple[str, ...], re.Pattern[str]], ...] = (
    (("taboo", "preference"), re.compile(r"讨厌|不喜欢|反感")),
    (("preference",), re.compile(r"喜欢|偏好|最爱|爱看|爱听|爱吃|爱喝|爱玩|主人|称呼")),
    (("running_joke",), re.compile(r"什么梗|有啥梗|有什么梗|梗")),
    (("relationship",), re.compile(r"什么关系|和谁|和什么人|关系")),
    (("plan",), re.compile(r"打算|计划|准备做")),
    (("decision",), re.compile(r"决定")),
    (("current", "event"), _CURRENT_VIEWING_INTENT_PATTERN),
    (("current",), re.compile(r"最近在做什么|在做什么|在干嘛|干什么")),
    (("event",), re.compile(r"最近发生|发生了什么|发生什么")),
    (PERSON_PORTRAIT_KINDS, _COMPOSITE_PORTRAIT_PATTERN),
    (("profile",), re.compile(r"哪里人|做什么的")),
)


_RECENCY_INTENT_PATTERN = re.compile(r"最近|现在|目前|近期|当下|刚刚|刚")
_TEMPORAL_FACT_KINDS = (
    "current",
    "event",
    "plan",
    "decision",
    "relationship",
    "profile",
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
    if _RECENCY_INTENT_PATTERN.search(text):
        # Temporal questions are not limited to one activity vocabulary.
        # Location, employment, study, relationship and other state changes
        # may be stored under several time-bearing fact kinds.
        return _TEMPORAL_FACT_KINDS
    if answer_mode == "current_fact":
        return ("preference", "taboo", "profile")
    return ()


def is_composite_portrait_query(query: str) -> bool:
    """Whether a question asks for a broad portrait rather than one attribute."""
    return bool(_COMPOSITE_PORTRAIT_PATTERN.search(str(query or "").strip()))


def select_diverse_portrait_facts(
    facts: Sequence[RankableMemoryFact],
    *,
    limit: int,
) -> list[RankableMemoryFact]:
    """Keep a bounded, category-diverse stable portrait.

    The input is already relevance-ranked.  Reserve one slot for every
    available portrait section, then fill remaining slots in the original
    order.  Temporary activity kinds are intentionally excluded.
    """
    if limit <= 0:
        return []
    eligible = [
        fact
        for fact in facts
        if str(getattr(fact, "memory_kind", "") or "") in PERSON_PORTRAIT_KINDS
    ]
    selected: list[RankableMemoryFact] = []
    selected_ids: set[int] = set()
    for kind in PERSON_PORTRAIT_KINDS:
        match = next(
            (
                fact
                for fact in eligible
                if str(getattr(fact, "memory_kind", "") or "") == kind
                and int(fact.id or 0) not in selected_ids
            ),
            None,
        )
        if match is None:
            continue
        selected.append(match)
        selected_ids.add(int(match.id or 0))
        if len(selected) >= limit:
            return selected
    for fact in eligible:
        fact_id = int(fact.id or 0)
        if fact_id in selected_ids:
            continue
        selected.append(fact)
        selected_ids.add(fact_id)
        if len(selected) >= limit:
            break
    return selected


def temporal_recency_required(*, query: str) -> bool:
    """True when the query asks about recent or current facts.

    Temporal questions should rank the freshest facts above older
    high-importance facts so "recent plans" never returns a stale plan.
    """
    return bool(_RECENCY_INTENT_PATTERN.search(str(query or "")))


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
    intent_query: str | None = None,
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
    if _CURRENT_VIEWING_INTENT_PATTERN.search(
        str(intent_query if intent_query is not None else query)
    ):
        features.update(_CURRENT_VIEWING_FEATURES)
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
    recency_boost: bool = False,
) -> list[RankableMemoryFact]:
    """Rank member facts by query relevance, then importance/confidence.

    A fact matches when any query feature appears in its content, predicate or
    object text. Facts whose kind is preferred (for example preference/taboo
    when the question asks about likes) outrank non-preferred kinds; matched
    facts outrank unmatched ones. With ``recency_boost`` (temporal questions),
    the freshest facts outrank older facts; otherwise importance, confidence
    and id break ties deterministically.
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
        tuple[
            bool,
            bool,
            float,
            tuple[float, float],
            float,
            float,
            int,
            RankableMemoryFact,
        ]
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
                (
                    recency if recency_boost else 0.0,
                    float(fact.importance or 1),
                ),
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


def select_temporal_current_facts(
    facts: Sequence[RankableMemoryFact],
    *,
    matching_fact_ids: set[int],
    topic_specific: bool,
    broad_limit: int = 5,
    broad_horizon_days: int = 14,
) -> list[RankableMemoryFact]:
    """Reduce a current-state query to the freshest relevant fact set.

    A topic-specific question (watching, work, location, study, and so on)
    should not expose stale alternatives to the answer model. A broad
    "recently doing what" question may need several activities, but only from
    the same recent window as the freshest matching evidence.
    """
    if not facts:
        return []
    matched = [fact for fact in facts if int(fact.id or 0) in matching_fact_ids]
    candidates = matched or list(facts)
    if topic_specific:
        return candidates[:1]
    newest = max((_recency_value(fact) for fact in candidates), default=0.0)
    cutoff = newest - max(1, int(broad_horizon_days)) * 86_400 if newest else 0.0
    selected = [
        fact
        for fact in candidates
        if not newest or _recency_value(fact) >= cutoff
    ]
    return selected[: max(1, int(broad_limit))]


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
