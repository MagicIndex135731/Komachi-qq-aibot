from __future__ import annotations

from types import SimpleNamespace

from app.core.memory_fact_ranking import (
    filter_member_query_features,
    memory_query_features,
    rank_member_facts,
)
from app.core.memory_fact_semantics import SemanticFactRanker


class _FakeEmbedder:
    available = True

    def embed_query(self, query: str):
        del query
        return [1.0, 0.0, 0.0]

    def embed_documents(self, documents):
        vectors = []
        for text in documents:
            if "海贼" in text:
                vectors.append([1.0, 0.2, 0.0])
            elif "动画" in text:
                vectors.append([0.2, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.2, 0.1])
        return vectors


def _fact(fact_id: int, content: str, memory_kind: str, importance: int, confidence: float):
    return SimpleNamespace(
        id=fact_id,
        content=content,
        predicate="",
        object_text="",
        memory_kind=memory_kind,
        importance=importance,
        confidence=confidence,
    )


def test_semantic_ranker_scores_and_caches_by_fact_id() -> None:
    embedder = _FakeEmbedder()
    ranker = SemanticFactRanker(embedder)
    facts = (
        _fact(1, "阿渣在做前后端", "current", 5, 0.95),
        _fact(2, "阿渣一直在看海贼王", "preference", 2, 0.8),
        _fact(3, "阿渣喜欢看动画", "preference", 1, 0.7),
    )
    scores = ranker.score("喜欢什么动画", facts)
    assert scores[2] > scores[3] > scores[1]
    cached = ranker.score("喜欢什么动画", facts)
    assert cached == scores


def test_semantic_scores_lift_specific_work_over_habit_and_current() -> None:
    facts = (
        _fact(1, "阿渣在做前后端", "current", 5, 0.95),
        _fact(2, "阿渣一直在看海贼王", "preference", 2, 0.8),
        _fact(3, "阿渣喜欢看动画", "preference", 1, 0.7),
    )
    features = filter_member_query_features(
        memory_query_features(query="阿渣喜欢什么动画？", topic_terms=("动画",)),
        aliases=("阿渣",),
    )
    semantic_scores = {2: 0.95, 3: 0.6, 1: 0.2}
    ranked = rank_member_facts(
        facts,
        query_features=features,
        limit=10,
        preferred_kinds=("preference", "taboo", "profile"),
        semantic_scores=semantic_scores,
    )
    assert [fact.id for fact in ranked] == [2, 3, 1]


def test_semantic_ranker_degrades_when_embedder_unavailable() -> None:
    class Unavailable:
        available = False

        def embed_query(self, _query):
            return None

        def embed_documents(self, _documents):
            return None

    ranker = SemanticFactRanker(Unavailable())
    assert ranker.score("query", (_fact(1, "x", "fact", 1, 0.5),)) == {}
