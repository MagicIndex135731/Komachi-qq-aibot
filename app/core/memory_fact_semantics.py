"""On-the-fly semantic scoring for member memory facts."""

from __future__ import annotations

from collections import OrderedDict
import math
import threading
from typing import Any, Mapping, Protocol, Sequence


class FactEmbedder(Protocol):
    available: bool

    def embed_query(self, query: str) -> list[float] | None: ...

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]] | None: ...


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return None
    return dot / (left_norm * right_norm)


class SemanticFactRanker:
    """Score facts by query-fact cosine similarity with an in-memory cache.

    The embedder is optional: any failure or unavailable provider degrades to
    an empty score map so callers keep their existing literal ranking.
    """

    def __init__(
        self,
        embedder: FactEmbedder,
        *,
        cache_size: int = 2048,
    ) -> None:
        self._embedder = embedder
        self._cache: OrderedDict[int, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self._cache_size = max(1, int(cache_size))

    def score(
        self,
        query: str,
        facts: Sequence[Any],
    ) -> dict[int, float]:
        if (
            not bool(getattr(self._embedder, "available", False))
            or not facts
            or not str(query or "").strip()
        ):
            return {}
        try:
            query_vector = self._embedder.embed_query(str(query))
            if query_vector is None:
                return {}
        except Exception:
            return {}

        with self._lock:
            cached: dict[int, list[float]] = {}
            missing: list[tuple[int, str]] = []
            for fact in facts:
                fact_id = int(getattr(fact, "id", 0) or 0)
                if not fact_id:
                    continue
                vector = self._cache.get(fact_id)
                if vector is not None:
                    cached[fact_id] = vector
                else:
                    missing.append((fact_id, str(getattr(fact, "content", "") or "")))
            if missing:
                try:
                    new_vectors = self._embedder.embed_documents(
                        [text for _, text in missing]
                    ) or []
                except Exception:
                    new_vectors = []
                for index, (fact_id, _text) in enumerate(missing):
                    vector = new_vectors[index] if index < len(new_vectors) else None
                    if vector:
                        cached[fact_id] = vector
                        self._cache[fact_id] = vector
                        while len(self._cache) > self._cache_size:
                            self._cache.popitem(last=False)

        scores: dict[int, float] = {}
        for fact_id, vector in cached.items():
            similarity = _cosine(query_vector, vector)
            if similarity is not None:
                scores[fact_id] = round(similarity, 6)
        return scores
