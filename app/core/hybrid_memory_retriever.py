from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, Callable, Mapping, Sequence


logger = logging.getLogger(__name__)
TEMPORAL_RELEVANCE_FINAL_LIMIT = 60
TEMPORAL_RELEVANCE_PIN_LIMIT = 48
DATED_HISTORY_FINAL_LIMIT = 1
VECTOR_CUTOFF_TIE_EPSILON = 1e-6

DEFAULT_CHANNEL_WEIGHTS: dict[str, float] = {
    "exact_quote": 6.0,
    "reply_graph": 4.0,
    "entity": 3.0,
    "fact": 2.5,
    "bm25": 1.8,
    "vector": 1.8,
    "temporal": 1.2,
}


class MemoryScopeViolation(RuntimeError):
    """A V2 candidate cannot be proven to belong to the requested group."""


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    document_id: int
    group_id: int
    document_kind: str
    episode_id: int | None
    source_msg_ids: tuple[str, ...]
    start_at: datetime
    end_at: datetime
    channel_score: float = 0.0


@dataclass(frozen=True, slots=True)
class FusedRetrievalCandidate:
    document_id: int
    group_id: int
    document_kind: str
    episode_id: int | None
    source_msg_ids: tuple[str, ...]
    start_at: datetime
    end_at: datetime
    routes: tuple[str, ...]
    route_ranks: tuple[tuple[str, int], ...]
    fused_score: float


@dataclass(frozen=True, slots=True)
class HybridRetrievalResult:
    candidates: tuple[FusedRetrievalCandidate, ...]
    failed_channels: tuple[str, ...] = ()
    attempted_channels: tuple[str, ...] = ()
    channel_candidate_counts: tuple[tuple[str, int], ...] = ()

    @property
    def all_channels_failed(self) -> bool:
        return bool(self.attempted_channels) and set(self.attempted_channels) <= set(
            self.failed_channels
        )


RetrievalChannel = Callable[..., Sequence[RetrievalCandidate]]


class HybridMemoryRetriever:
    """Run independent scoped channels and combine their ranks with weighted RRF.

    Channel callables own their database sessions. They must not share one
    SQLAlchemy Session across the worker threads used here.
    """

    def __init__(
        self,
        *,
        channels: Mapping[str, RetrievalChannel],
        channel_weights: Mapping[str, float] | None = None,
        candidate_limit: int = 30,
        final_limit: int = 30,
        rrf_k: int = 60,
        channel_timeout_seconds: float = 0.5,
    ) -> None:
        if channel_timeout_seconds <= 0:
            raise ValueError("channel_timeout_seconds must be positive")
        self.channels = dict(channels)
        self.channel_weights = {
            **DEFAULT_CHANNEL_WEIGHTS,
            **dict(channel_weights or {}),
        }
        self.candidate_limit = max(1, int(candidate_limit))
        self.final_limit = max(1, int(final_limit))
        self.rrf_k = max(1, int(rrf_k))
        self.channel_timeout_seconds = float(channel_timeout_seconds)

    def retrieve(self, *, group_id: int, resolved_query: Any) -> HybridRetrievalResult:
        if not self.channels:
            return HybridRetrievalResult(())

        reference_msg_ids = tuple(
            str(value).strip()
            for value in getattr(resolved_query, "reference_msg_ids", ())
            if str(value).strip()
        )
        if reference_msg_ids:
            planned_channels = ("exact_quote", "reply_graph")
        elif getattr(resolved_query, "answer_mode", "") == "mention":
            planned_channels = ("temporal",)
        elif getattr(resolved_query, "retrieval_mode", "") == "temporal":
            # The time range remains a hard filter inside every scoped channel.
            # Semantic/lexical routes rank the relevant message within a busy
            # interval; the temporal route preserves complete window coverage.
            planned_channels = ("bm25", "vector", "entity", "temporal")
        else:
            planned_channels = tuple(self.channels)
        channel_names = tuple(
            channel for channel in planned_channels if channel in self.channels
        )
        # Transport-level quotes and mention plans already have deterministic
        # provenance channels. Temporal history still runs scoped semantic
        # ranking because all participating channels enforce the same range.
        if not channel_names:
            return HybridRetrievalResult(())
        channel_results: dict[str, Sequence[RetrievalCandidate]] = {}
        failed_channels: list[str] = []
        executor = ThreadPoolExecutor(
            max_workers=len(channel_names),
            thread_name_prefix="memory-retrieval",
        )
        futures = {}
        try:
            futures = {
                channel: executor.submit(
                    self.channels[channel],
                    group_id=group_id,
                    resolved_query=resolved_query,
                    limit=self.candidate_limit,
                )
                for channel in channel_names
            }
            done, _ = wait(
                tuple(futures.values()),
                timeout=self.channel_timeout_seconds,
            )
            for channel in channel_names:
                future = futures[channel]
                if future not in done:
                    failed_channels.append(channel)
                    future.cancel()
                    logger.warning(
                        "memory_retrieval_channel_failed channel=%s error_type=TimeoutError",
                        channel,
                    )
                    continue
                try:
                    channel_results[channel] = tuple(future.result())
                except Exception as exc:
                    failed_channels.append(channel)
                    logger.warning(
                        "memory_retrieval_channel_failed channel=%s error_type=%s",
                        channel,
                        type(exc).__name__,
                    )
        finally:
            for future in futures.values():
                if not future.done():
                    future.cancel()
            # A timed-out provider may ignore cancellation. Never let executor
            # cleanup turn a finite channel deadline back into a reply-path wait.
            executor.shutdown(wait=False, cancel_futures=True)

        # Validate the entire batch before using any candidate. A single
        # cross-scope row indicates a repository/provenance failure, so partial
        # V2 output is unsafe; the orchestrator must fall back independently.
        for candidates in channel_results.values():
            for item in candidates:
                if int(item.group_id) != int(group_id):
                    raise MemoryScopeViolation(
                        f"candidate scope mismatch document_id={item.document_id}"
                    )

        if (
            getattr(resolved_query, "coverage_mode", "relevance") == "relevance"
            and "vector" in channel_results
        ):
            channel_results["vector"] = self._stabilize_vector_cutoff(
                channel_results["vector"],
                cutoff=self.final_limit,
            )

        accumulated: dict[int, dict[str, Any]] = {}
        for channel in channel_names:
            weight = float(self.channel_weights.get(channel, 1.0))
            for rank, item in enumerate(channel_results.get(channel, ()), start=1):
                state = accumulated.setdefault(
                    item.document_id,
                    {
                        "candidate": item,
                        "routes": [],
                        "route_ranks": [],
                        "score": 0.0,
                        "source_msg_ids": [],
                    },
                )
                if channel not in state["routes"]:
                    state["routes"].append(channel)
                    state["route_ranks"].append((channel, rank))
                    state["score"] += weight / (self.rrf_k + rank)
                state["source_msg_ids"].extend(item.source_msg_ids)

        fused: list[FusedRetrievalCandidate] = []
        for state in accumulated.values():
            item = state["candidate"]
            fused.append(
                FusedRetrievalCandidate(
                    document_id=item.document_id,
                    group_id=item.group_id,
                    document_kind=item.document_kind,
                    episode_id=item.episode_id,
                    source_msg_ids=tuple(dict.fromkeys(state["source_msg_ids"])),
                    start_at=item.start_at,
                    end_at=item.end_at,
                    routes=tuple(state["routes"]),
                    route_ranks=tuple(state["route_ranks"]),
                    fused_score=float(state["score"]),
                )
            )

        fused.sort(
            key=lambda item: (
                -int("exact_quote" in item.routes),
                -item.fused_score,
                -item.end_at.timestamp(),
                item.document_id,
            )
        )
        relevance_order = tuple(fused)
        coverage_mode = getattr(resolved_query, "coverage_mode", "relevance")
        if coverage_mode == "chronological":
            fused.sort(key=lambda item: (item.start_at, item.document_id))
        elif coverage_mode == "time_buckets":
            fused = self._time_bucket_coverage(fused, self.final_limit)
        resolved_final_limit = self.final_limit
        if (
            getattr(resolved_query, "retrieval_mode", "") == "temporal"
            and getattr(resolved_query, "answer_mode", "") != "mention"
        ):
            # A bounded date window can still contain hundreds of chat lines.
            # Single-event lookups expose only the direct top hit; summaries
            # keep a wider relevance/coverage packet for interval synthesis.
            resolved_final_limit = min(
                resolved_final_limit,
                DATED_HISTORY_FINAL_LIMIT
                if getattr(resolved_query, "answer_mode", "") == "dated_history"
                else TEMPORAL_RELEVANCE_FINAL_LIMIT,
            )
            fused = self._pin_temporal_relevance(
                coverage_order=fused,
                relevance_order=relevance_order,
                limit=resolved_final_limit,
            )
        return HybridRetrievalResult(
            candidates=tuple(fused[:resolved_final_limit]),
            failed_channels=tuple(failed_channels),
            attempted_channels=channel_names,
            channel_candidate_counts=tuple(
                (channel, len(channel_results.get(channel, ())))
                for channel in channel_names
            ),
        )

    @staticmethod
    def _pin_temporal_relevance(
        *,
        coverage_order: Sequence[FusedRetrievalCandidate],
        relevance_order: Sequence[FusedRetrievalCandidate],
        limit: int,
    ) -> list[FusedRetrievalCandidate]:
        """Keep direct semantic hits ahead of bounded timeline coverage.

        Temporal queries need both the most relevant messages and enough of the
        requested interval to answer summaries.  Coverage reordering alone can
        bury an exact semantic hit behind dozens of adjacent messages, so reserve
        most of the bounded packet for relevance and fill the remainder from the
        deterministic coverage order.
        """

        resolved_limit = max(1, int(limit))
        pin_limit = min(TEMPORAL_RELEVANCE_PIN_LIMIT, resolved_limit)
        selected = list(relevance_order[:pin_limit])
        selected_ids = {item.document_id for item in selected}
        for item in coverage_order:
            if item.document_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item.document_id)
            if len(selected) >= resolved_limit:
                break
        return selected

    @staticmethod
    def _stabilize_vector_cutoff(
        candidates: Sequence[RetrievalCandidate],
        *,
        cutoff: int,
    ) -> tuple[RetrievalCandidate, ...]:
        """Interleave a narrow semantic boundary without widening the hard limit.

        Dense-vector scores immediately around a large top-K boundary are often
        effectively tied.  Keeping the high-confidence prefix fixed while
        interleaving equal-width slices on both sides of the cutoff prevents a
        tiny score perturbation from deterministically hiding every candidate
        just below K.  The candidate count and the final hard limit are unchanged.
        """

        ordered = tuple(candidates)
        resolved_cutoff = max(1, int(cutoff))
        if len(ordered) <= resolved_cutoff or resolved_cutoff < 4:
            return ordered
        boundary_gap = abs(
            float(ordered[resolved_cutoff - 1].channel_score)
            - float(ordered[resolved_cutoff].channel_score)
        )
        if boundary_gap > VECTOR_CUTOFF_TIE_EPSILON:
            return ordered
        boundary_score = float(ordered[resolved_cutoff - 1].channel_score)
        tie_start = resolved_cutoff - 1
        while (
            tie_start > 0
            and abs(float(ordered[tie_start - 1].channel_score) - boundary_score)
            <= VECTOR_CUTOFF_TIE_EPSILON
        ):
            tie_start -= 1
        tie_end = resolved_cutoff + 1
        while (
            tie_end < len(ordered)
            and abs(float(ordered[tie_end].channel_score) - boundary_score)
            <= VECTOR_CUTOFF_TIE_EPSILON
        ):
            tie_end += 1
        width = min(
            8,
            resolved_cutoff // 4,
            resolved_cutoff - tie_start,
            tie_end - resolved_cutoff,
        )
        if width <= 0:
            return ordered
        prefix_end = resolved_cutoff - width
        lower = ordered[prefix_end:resolved_cutoff]
        upper = ordered[resolved_cutoff : resolved_cutoff + width]
        boundary = tuple(
            item
            for pair in zip(lower, upper, strict=True)
            for item in pair
        )
        return (
            *ordered[:prefix_end],
            *boundary,
            *ordered[resolved_cutoff + width :],
        )

    @staticmethod
    def _time_bucket_coverage(
        candidates: Sequence[FusedRetrievalCandidate],
        limit: int,
    ) -> list[FusedRetrievalCandidate]:
        """Allocate a fair quota across equal spans of the available timeline."""

        resolved_limit = max(1, int(limit))
        ordered = sorted(
            candidates,
            key=lambda item: (item.start_at, item.document_id),
        )
        if len(ordered) <= resolved_limit:
            return ordered
        if resolved_limit == 1:
            return [ordered[-1]]

        start_value = ordered[0].start_at.timestamp()
        end_value = ordered[-1].start_at.timestamp()
        if end_value <= start_value:
            return sorted(
                ordered,
                key=lambda item: (
                    -item.fused_score,
                    -item.end_at.timestamp(),
                    item.document_id,
                ),
            )[:resolved_limit]

        bucket_count = min(12, resolved_limit, len(ordered))
        buckets: list[list[FusedRetrievalCandidate]] = [
            [] for _ in range(bucket_count)
        ]
        span = end_value - start_value
        for candidate in ordered:
            ratio = (candidate.start_at.timestamp() - start_value) / span
            bucket_index = min(
                bucket_count - 1,
                max(0, int(ratio * bucket_count)),
            )
            buckets[bucket_index].append(candidate)

        for index, bucket in enumerate(buckets):
            bucket.sort(
                key=lambda item: (
                    -item.fused_score,
                    item.start_at,
                    item.document_id,
                )
            )
            if index == 0:
                bucket.sort(key=lambda item: (item.start_at, item.document_id))
            elif index == bucket_count - 1:
                bucket.sort(
                    key=lambda item: (item.start_at, item.document_id),
                    reverse=True,
                )

        selected: list[FusedRetrievalCandidate] = []
        depth = 0
        while len(selected) < resolved_limit:
            added = False
            for bucket in buckets:
                if depth >= len(bucket):
                    continue
                selected.append(bucket[depth])
                added = True
                if len(selected) >= resolved_limit:
                    break
            if not added:
                break
            depth += 1
        return selected
