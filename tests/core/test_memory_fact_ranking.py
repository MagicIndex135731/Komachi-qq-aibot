from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from app.core.memory_fact_ranking import (
    filter_member_query_features,
    matching_member_fact_ids,
    memory_query_features,
    preferred_kinds_for_query,
    rank_member_facts,
    select_temporal_current_facts,
    temporal_recency_required,
)


def _fact(
    fact_id: int,
    content: str,
    *,
    importance: int = 1,
    confidence: float = 0.5,
    predicate: str = "",
    object_text: str = "",
    memory_kind: str = "fact",
    last_seen_at=None,
    valid_from=None,
):
    return SimpleNamespace(
        id=fact_id,
        content=content,
        predicate=predicate,
        object_text=object_text,
        memory_kind=memory_kind,
        importance=importance,
        confidence=confidence,
        last_seen_at=last_seen_at,
        valid_from=valid_from,
    )


def test_preferred_kinds_for_query_intent_mapping() -> None:
    cases = (
        ("阿渣喜欢什么动画？", ("preference",)),
        ("阿渣讨厌什么？", ("taboo", "preference")),
        ("阿渣不喜欢什么？", ("taboo", "preference")),
        ("阿渣有什么梗？", ("running_joke",)),
        ("阿渣和谁是什么关系？", ("relationship",)),
        ("阿渣打算做什么？", ("plan",)),
        ("阿渣决定了什么？", ("decision",)),
        ("阿渣最近在做什么？", ("current",)),
        ("阿渣最近在看什么动画？", ("current", "event")),
        ("阿渣现在在追什么番？", ("current", "event")),
        ("阿渣最近发生了什么？", ("event",)),
        (
            "阿渣目前在哪里工作？",
            ("current", "event", "plan", "decision", "relationship", "profile"),
        ),
        ("介绍一下阿渣", ("profile",)),
        ("阿渣是什么样的人？", ("profile",)),
        ("今天天气怎么样", ()),
    )
    for query, expected in cases:
        assert preferred_kinds_for_query(
            query=query,
            answer_mode="general_history",
        ) == expected, query


def test_preferred_kinds_for_query_current_fact_fallback() -> None:
    assert preferred_kinds_for_query(
        query="还记得阿渣吗？",
        answer_mode="current_fact",
    ) == ("preference", "taboo", "profile")
    assert preferred_kinds_for_query(
        query="还记得阿渣吗？",
        answer_mode="general_history",
    ) == ()


def test_memory_query_features_extract_entities_and_cjk_grams() -> None:
    features = memory_query_features(
        query="阿渣喜欢什么动画？",
        entities=("阿渣",),
        topic_terms=("动画",),
    )
    assert "阿渣" in features
    assert "动画" in features
    assert "喜欢" in features
    assert "什么" in features


def test_memory_query_features_expands_current_viewing_synonyms() -> None:
    features = memory_query_features(
        query="阿渣最近在看什么动画？",
        topic_terms=("动画",),
    )

    assert {"在看", "观看", "追看", "追番", "补番"}.issubset(features)


def test_filter_member_query_features_drops_alias_echoes() -> None:
    features = memory_query_features(
        query="阿渣喜欢什么动画？",
        entities=("阿渣",),
        topic_terms=("动画",),
    )
    filtered = filter_member_query_features(
        features,
        aliases=("阿渣",),
    )
    assert "动画" in filtered
    assert "喜欢" in filtered
    assert all("阿渣" not in feature for feature in filtered)


def test_rank_member_facts_puts_relevant_facts_first() -> None:
    unrelated = _fact(1, "用户表示自己在做前后端", importance=4, confidence=0.9)
    relevant = _fact(2, "用户表示自己一直在看海贼王动画", importance=2, confidence=0.8)
    features = memory_query_features(
        query="阿渣喜欢什么动画？",
        topic_terms=("动画",),
    )
    features = filter_member_query_features(features, aliases=("阿渣",))
    ranked = rank_member_facts(
        (unrelated, relevant),
        query_features=features,
        limit=6,
    )
    assert [fact.id for fact in ranked] == [2, 1]
    assert matching_member_fact_ids(
        (unrelated, relevant),
        query_features=features,
    ) == {2}


def test_rank_member_facts_respects_limit_and_tie_break() -> None:
    facts = (
        _fact(1, "甲", importance=3, confidence=0.9),
        _fact(2, "乙", importance=3, confidence=0.8),
        _fact(3, "丙", importance=3, confidence=0.8),
    )
    ranked = rank_member_facts(facts, query_features=("不存在",), limit=2)
    assert [fact.id for fact in ranked] == [1, 3]


def test_rank_member_facts_prefers_kind_before_importance() -> None:
    unrelated_current = _fact(
        1,
        "阿渣在做前后端",
        importance=5,
        confidence=0.9,
        memory_kind="current",
    )
    unrelated_preference = _fact(
        2,
        "阿渣一直在看海贼王",
        importance=2,
        confidence=0.8,
        memory_kind="preference",
    )
    matched_preference = _fact(
        3,
        "阿渣喜欢看动画",
        importance=1,
        confidence=0.7,
        memory_kind="preference",
    )
    features = memory_query_features(
        query="阿渣喜欢什么动画？",
        topic_terms=("动画",),
    )
    features = filter_member_query_features(features, aliases=("阿渣",))
    ranked = rank_member_facts(
        (unrelated_current, unrelated_preference, matched_preference),
        query_features=features,
        limit=10,
        preferred_kinds=("preference", "taboo", "profile"),
    )
    assert [fact.id for fact in ranked] == [3, 2, 1]


def test_rank_member_facts_recency_breaks_ties() -> None:
    older = _fact(
        1,
        "用户表示自己在做前后端",
        importance=3,
        confidence=0.8,
        memory_kind="current",
        last_seen_at=datetime(2026, 7, 20, tzinfo=UTC),
    )
    newer = _fact(
        2,
        "用户表示自己在做前后端",
        importance=3,
        confidence=0.8,
        memory_kind="current",
        last_seen_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    no_date = _fact(
        3,
        "用户表示自己在做前后端",
        importance=3,
        confidence=0.8,
        memory_kind="current",
    )
    ranked = rank_member_facts(
        (older, no_date, newer),
        query_features=("不存在",),
        limit=3,
    )
    assert [fact.id for fact in ranked] == [2, 1, 3]

def test_temporal_recency_required_detects_recent_intent() -> None:
    temporal_queries = (
        "阿渣最近有什么计划",
        "茔草现在在忙什么",
        "阿渣目前在哪里工作",
        "阿渣近期决定了什么",
        "阿渣刚刚和谁建立了关系",
        "最近聊过什么",
    )
    assert all(temporal_recency_required(query=query) for query in temporal_queries)
    assert not temporal_recency_required(query="阿渣喜欢什么动画")
    assert not temporal_recency_required(query="介绍一下阿渣")


def test_rank_member_facts_recency_boost_prefers_fresh_facts() -> None:
    recent = _fact(
        2,
        "用户表示下周去考古",
        importance=1,
        confidence=0.5,
        valid_from=datetime(2026, 8, 10, tzinfo=UTC),
    )
    stale = _fact(
        1,
        "用户表示打算开发一键上号功能",
        importance=9,
        confidence=0.9,
        valid_from=datetime(2026, 3, 1, tzinfo=UTC),
    )
    features = memory_query_features(query="阿渣最近有什么计划")
    ranked = rank_member_facts(
        (stale, recent),
        query_features=features,
        limit=6,
        recency_boost=True,
    )
    assert [fact.id for fact in ranked] == [2, 1]

    # Without the boost the old high-importance fact still wins.
    ranked_plain = rank_member_facts(
        (stale, recent),
        query_features=features,
        limit=6,
    )
    assert [fact.id for fact in ranked_plain] == [1, 2]


def test_recent_viewing_ranking_prefers_fresh_equivalent_activity() -> None:
    old_explicit_current = _fact(
        1,
        "用户正在看《向阳素描》",
        confidence=0.99,
        memory_kind="current",
        last_seen_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    recent_viewing_event = _fact(
        2,
        "用户与朋友共同观看并讨论《结城友奈是勇者》",
        confidence=0.90,
        memory_kind="event",
        last_seen_at=datetime(2026, 8, 20, tzinfo=UTC),
    )
    old_preference = _fact(
        3,
        "用户一直在看《海贼王》最新内容",
        confidence=0.99,
        memory_kind="preference",
        last_seen_at=datetime(2026, 7, 21, tzinfo=UTC),
    )
    query = "阿渣最近在看什么动画？"
    features = filter_member_query_features(
        memory_query_features(query=query, topic_terms=("动画",)),
        aliases=("阿渣",),
    )

    ranked = rank_member_facts(
        (old_explicit_current, recent_viewing_event, old_preference),
        query_features=features,
        limit=3,
        preferred_kinds=preferred_kinds_for_query(
            query=query,
            answer_mode="current_fact",
        ),
        recency_boost=True,
    )

    assert [fact.id for fact in ranked] == [2, 1, 3]


def test_topic_specific_temporal_current_fact_keeps_only_the_newest_match() -> None:
    recent_match = _fact(
        2,
        "用户最近在北京工作",
        last_seen_at=datetime(2026, 8, 20, tzinfo=UTC),
    )
    stale_match = _fact(
        1,
        "用户之前在西安工作",
        last_seen_at=datetime(2026, 7, 20, tzinfo=UTC),
    )
    newer_unrelated = _fact(
        3,
        "用户今天买了饮料",
        last_seen_at=datetime(2026, 8, 22, tzinfo=UTC),
    )

    selected = select_temporal_current_facts(
        (recent_match, newer_unrelated, stale_match),
        matching_fact_ids={1, 2},
        topic_specific=True,
    )

    assert [fact.id for fact in selected] == [2]


def test_broad_temporal_current_fact_keeps_multiple_recent_activities() -> None:
    newest = _fact(
        3,
        "用户最近在学日语",
        last_seen_at=datetime(2026, 8, 22, tzinfo=UTC),
    )
    recent = _fact(
        2,
        "用户最近在做机器人",
        last_seen_at=datetime(2026, 8, 18, tzinfo=UTC),
    )
    stale = _fact(
        1,
        "用户之前在准备考试",
        last_seen_at=datetime(2026, 7, 1, tzinfo=UTC),
    )

    selected = select_temporal_current_facts(
        (newest, recent, stale),
        matching_fact_ids={1, 2, 3},
        topic_specific=False,
    )

    assert [fact.id for fact in selected] == [3, 2]
