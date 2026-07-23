from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, Callable, Mapping, Sequence


logger = logging.getLogger(__name__)

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

        channel_names = tuple(self.channels)
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
        return HybridRetrievalResult(
            candidates=tuple(fused[: self.final_limit]),
            failed_channels=tuple(failed_channels),
            attempted_channels=channel_names,
            channel_candidate_counts=tuple(
                (channel, len(channel_results.get(channel, ())))
                for channel in channel_names
            ),
        )
