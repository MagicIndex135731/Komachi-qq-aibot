from __future__ import annotations

from types import SimpleNamespace

from app.core.memory_fact_ranking import (
    filter_member_query_features,
    matching_member_fact_ids,
    memory_query_features,
    rank_member_facts,
)


def _fact(
    fact_id: int,
    content: str,
    *,
    importance: int = 1,
    confidence: float = 0.5,
    predicate: str = "",
    object_text: str = "",
):
    return SimpleNamespace(
        id=fact_id,
        content=content,
        predicate=predicate,
        object_text=object_text,
        importance=importance,
        confidence=confidence,
    )


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
