"""Query-relevance ranking for per-member memory facts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence

from app.core.time_utils import stored_as_utc


_CJK = re.compile(r"[\u4e00-\u9fff]+")
_CURRENT_VIEWING_INTENT_PATTERN = re.compile(
    r"(?:(?:最近|现在|目前|近期|当下).{0,12}?(?:正在|在)?"
    r"(?:看|追|补)(?:着)?(?:什么|啥|哪个|哪些)|"
    r"(?:正在|在)(?:看|追|补)(?:着)?(?:什么|啥|哪个|哪些)|"
    r"(?:最近|现在|目前|近期|当下).{0,8}?(?:什么|啥|哪个|哪些)"
    r"(?:动画|作品|番|剧|游戏|书|歌).{0,6}?(?:在)?(?:看|追|补|玩|读|听))"
)
_CURRENT_VIEWING_FEATURES = ("在看", "观看", "追看", "追番", "补番", "补剧")
_CURRENT_ACTIVITY_INTENT_PATTERN = re.compile(
    r"(?:最近|现在|目前|近期|当下).{0,12}?"
    r"(?:在|正在)?(?:看|追|补|玩|做|学|忙|干)(?:着)?"
    r"(?:什么|啥|哪个|哪些)|"
    r"(?:正在|在)(?:玩|做|学|忙|干)(?:着)?(?:什么|啥)"
)
_CURRENT_STATE_INTENT_PATTERN = re.compile(
    r"(?:(?:现在|目前|最近|近期|当下).{0,12}?)?"
    r"(?:在哪(?!个|些|位)|在哪里|哪里工作|在哪工作|忙什么|什么状态)"
)
_PROCESS_STATE_INTENT_PATTERN = re.compile(
    r"(?:"
    r"(?:最近|现在|目前|近期|当下).{0,18}?"
    r"(?:工作|求职|面试|论文|项目|学习|考试|搬家|装修|健身|治疗|申请|实习|研究|准备)"
    r".{0,8}?(?:怎么样|咋样|如何|什么情况)|"
    r"(?:工作|求职|面试|论文|项目|学习|考试|搬家|装修|健身|治疗|申请|实习|研究|准备)?"
    r"(?:找|写|做|学|准备|推进|申请|面试|装修|搬|练|治疗)(?:得|的)?"
    r"(?:怎么样|咋样|如何|到哪(?:一步)?|什么情况)|"
    r"[A-Za-z0-9_\-\u4e00-\u9fff]{1,20}(?:的)?进展(?:怎么样|咋样|如何|到哪(?:一步)?|什么情况)?"
    r")"
)

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
    (("plan",), re.compile(r"打算|计划|准备(?:做|去|学|看|玩)?|接下来|下一步")),
    (("decision",), re.compile(r"决定")),
    (("current", "event"), _CURRENT_VIEWING_INTENT_PATTERN),
    (("current", "event"), _CURRENT_STATE_INTENT_PATTERN),
    (("current",), _CURRENT_ACTIVITY_INTENT_PATTERN),
    (("event",), re.compile(r"最近发生|发生了什么|发生什么")),
    (PERSON_PORTRAIT_KINDS, _COMPOSITE_PORTRAIT_PATTERN),
    (("profile",), re.compile(r"哪里人|做什么的")),
)

FactCoverage = Literal["single", "plural", "multi_facet"]


@dataclass(frozen=True, slots=True)
class FactIntentPolicy:
    allowed_kinds: tuple[str, ...]
    preferred_kinds: tuple[str, ...]
    temporal: bool
    coverage: FactCoverage
    reason: str


_PLURAL_FACT_PATTERN = re.compile(r"哪些|哪几|几部|多部|都(?:有|是|在)|分别")


def fact_kinds_for_query(*, query: str, answer_mode: str) -> tuple[str, ...]:
    """Return the bounded storage-kind policy for member-fact injection.

    This is intentionally shared with normal fact intent inference.  Callers
    may rank within the returned kinds, but must not broaden it to every kind:
    a one-off current activity is not a durable preference, and an expired
    plan is never profile knowledge.
    """

    return fact_intent_policy(query=query, answer_mode=answer_mode).allowed_kinds


_RECENCY_INTENT_PATTERN = re.compile(
    r"最近|现在|目前|近期|当下|刚刚|刚|接下来|之后|下一步|未来|打算|计划|准备"
)
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
    return fact_intent_policy(query=query, answer_mode=answer_mode).preferred_kinds


def fact_intent_policy(*, query: str, answer_mode: str) -> FactIntentPolicy:
    """Return the single storage and coverage policy for a member-fact query."""

    text = str(query or "").strip()
    plural = bool(_PLURAL_FACT_PATTERN.search(text))
    temporal = bool(_RECENCY_INTENT_PATTERN.search(text))

    if _CURRENT_VIEWING_INTENT_PATTERN.search(text):
        return FactIntentPolicy(
            allowed_kinds=("current", "event"),
            preferred_kinds=("current", "event"),
            temporal=True,
            coverage="plural" if plural else "single",
            reason="current_viewing",
        )
    if _CURRENT_ACTIVITY_INTENT_PATTERN.search(text):
        return FactIntentPolicy(
            allowed_kinds=("current", "event"),
            preferred_kinds=("current",),
            temporal=True,
            coverage="plural" if plural else "single",
            reason="current_activity",
        )
    if _CURRENT_STATE_INTENT_PATTERN.search(text):
        return FactIntentPolicy(
            allowed_kinds=("current", "event"),
            preferred_kinds=("current", "event"),
            temporal=True,
            coverage="single",
            reason="current_state",
        )
    if _PROCESS_STATE_INTENT_PATTERN.search(text):
        return FactIntentPolicy(
            allowed_kinds=_TEMPORAL_FACT_KINDS,
            preferred_kinds=_TEMPORAL_FACT_KINDS,
            temporal=True,
            coverage="plural" if plural else "multi_facet",
            reason="temporal_process",
        )

    for kinds, pattern in _KIND_INTENT_PATTERNS:
        if not pattern.search(text):
            continue
        preferred = tuple(kinds)
        preferred_set = frozenset(preferred)
        if preferred == PERSON_PORTRAIT_KINDS:
            allowed = PERSON_PORTRAIT_KINDS
            coverage: FactCoverage = "multi_facet"
        elif preferred_set & {"plan", "decision"}:
            allowed = ("plan", "decision")
            coverage = "plural" if plural else "single"
        elif preferred_set & {"preference", "taboo"}:
            allowed = tuple(
                kind for kind in ("preference", "taboo") if kind in preferred_set
            )
            coverage = "plural" if plural else "single"
        elif "relationship" in preferred_set:
            allowed = ("relationship", "profile", "fact")
            coverage = "multi_facet"
        elif "profile" in preferred_set:
            allowed = ("profile", "fact")
            coverage = "single"
        elif preferred_set & {"current", "event"}:
            allowed = ("current", "event")
            coverage = "plural" if plural else "single"
        else:
            allowed = preferred
            coverage = "plural" if plural else "single"
        return FactIntentPolicy(
            allowed_kinds=allowed,
            preferred_kinds=preferred,
            temporal=temporal,
            coverage=coverage,
            reason="explicit_kind_intent",
        )

    if temporal and answer_mode == "current_fact":
        return FactIntentPolicy(
            allowed_kinds=_TEMPORAL_FACT_KINDS,
            preferred_kinds=_TEMPORAL_FACT_KINDS,
            temporal=True,
            coverage="plural" if plural else "multi_facet",
            reason="temporal_process",
        )

    if answer_mode == "current_fact":
        return FactIntentPolicy(
            allowed_kinds=("fact", "relationship", "profile", "preference", "taboo"),
            preferred_kinds=("preference", "taboo", "profile"),
            temporal=temporal,
            coverage="plural" if plural else "single",
            reason="current_fact_fallback",
        )
    return FactIntentPolicy(
        allowed_kinds=("fact", "relationship"),
        preferred_kinds=(),
        temporal=temporal,
        coverage="single",
        reason="durable_fallback",
    )


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


def recent_viewing_event_fallback(
    facts: Sequence[RankableMemoryFact], *, query: str, now: Any,
    horizon_days: int = 14,
) -> RankableMemoryFact | None:
    """Use a recent explicit viewing event when a current-viewing query has no match."""
    if not _CURRENT_VIEWING_INTENT_PATTERN.search(str(query or "")):
        return None
    now_seconds = float(stored_as_utc(now).timestamp())
    candidates = [
        fact for fact in facts
        if str(getattr(fact, "memory_kind", "")) == "event"
        and re.search(r"(?:看完|补完|追完|看了|补了|追了).{0,18}《[^》]+》", str(fact.content or ""))
        and 0 <= now_seconds - _recency_value(fact) <= horizon_days * 86400
    ]
    return max(candidates, key=_recency_value) if candidates else None


def prefer_recent_viewing_event(
    event: RankableMemoryFact | None,
    *,
    matched_current: Sequence[RankableMemoryFact],
) -> bool:
    """A direct finished-viewing event can outrank older generic current text."""
    if event is None:
        return False
    newest_current = max((_recency_value(fact) for fact in matched_current), default=0.0)
    return _recency_value(event) > newest_current


def viewing_fallback_match_ids(
    matching_fact_ids: set[int],
    *,
    viewing_event: RankableMemoryFact,
    coverage: FactCoverage,
) -> set[int]:
    """Keep a single-item fallback from collapsing plural evidence.

    The fallback exists for queries whose lexical features (for example
    ``动画``) do not occur in a concrete title fact.  A singular question may
    safely narrow to the freshest explicit viewing event.  A plural question
    must keep the broader ranked candidate set when no lexical match exists;
    otherwise one fallback row silently turns ``哪些`` back into a one-item
    answer before the diversity selector can run.
    """

    event_id = int(viewing_event.id or 0)
    if coverage == "single":
        return {event_id}
    if matching_fact_ids:
        return {*matching_fact_ids, event_id}
    return set()


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
    intent_text = str(intent_query if intent_query is not None else query)
    if _CURRENT_VIEWING_INTENT_PATTERN.search(intent_text):
        features.update(_CURRENT_VIEWING_FEATURES)
    if re.search(r"动画|动漫|番剧", intent_text):
        # Distillation categories use “动漫”, while people commonly ask
        # “什么动画”.  Keep both terms so a title-only fact (for example an
        # abbreviation such as RW0) can still match through its predicate.
        features.update(("动画", "动漫", "番剧"))
    if re.search(r"哪家|哪个|哪些|哪位|什么|谁", intent_text):
        # Clarification-derived facts preserve the original slot wording. A
        # query may ask “哪个公司” while the source question says “哪家”；
        # treat these slot forms as one retrieval family.
        features.update(("哪家", "哪个", "哪些", "哪位", "什么", "谁"))
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
    coverage: FactCoverage = "single",
    broad_limit: int = 10,
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
    if topic_specific and coverage == "single":
        return candidates[:1]
    newest = max((_recency_value(fact) for fact in candidates), default=0.0)
    cutoff = newest - max(1, int(broad_horizon_days)) * 86_400 if newest else 0.0
    recent = [
        fact
        for fact in candidates
        if not newest or _recency_value(fact) >= cutoff
    ]
    if coverage == "single":
        return recent[: max(1, int(broad_limit))]

    selected: list[RankableMemoryFact] = []
    seen_facets: set[tuple[str, ...]] = set()
    for fact in recent:
        kind = str(getattr(fact, "memory_kind", "") or "")
        predicate = " ".join(str(getattr(fact, "predicate", "") or "").split()).casefold()
        object_text = " ".join(str(getattr(fact, "object_text", "") or "").split()).casefold()
        if coverage == "multi_facet":
            # Same-kind facts may still represent different alternatives or
            # states (for example two ``decision/work`` rows for separate
            # companies). Keep the normalized content as a tie-break facet
            # when no structured object is available.
            facet = (
                kind,
                predicate,
                object_text or " ".join(str(fact.content).split()).casefold(),
            )
        else:
            # Multi-item questions need distinct objects even when all rows
            # share the same lifecycle kind and category.
            facet = (predicate, object_text or " ".join(str(fact.content).split()).casefold())
        if facet in seen_facets:
            continue
        seen_facets.add(facet)
        selected.append(fact)
        if len(selected) >= max(1, int(broad_limit)):
            break
    return selected


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
