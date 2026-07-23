"""Synchronous V2 query-side pipeline composed from offline-testable stages."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from app.core.memory_context_packer import (
    EvidenceMessage,
    MemoryContextPacker,
    MemoryFact,
    MemorySummary,
)
from app.core.hybrid_memory_retriever import MemoryScopeViolation
from app.core.memory_orchestrator import MemoryContextResult
from app.core.memory_query_resolver import RecentMemoryMessage, ResolvedMemoryQuery


class QueryResolver(Protocol):
    def resolve(
        self,
        query: str,
        *,
        recent_messages: Sequence[RecentMemoryMessage],
        quoted_message: RecentMemoryMessage | None,
        now: datetime | None,
    ) -> ResolvedMemoryQuery: ...


class Retriever(Protocol):
    def retrieve(self, *, group_id: int, resolved_query: object) -> object: ...


class Expander(Protocol):
    def expand(self, *, group_id: int, candidates: Sequence[object], mode: str) -> Sequence[object]: ...


FactLoader = Callable[..., Sequence[MemoryFact]]
SummaryLoader = Callable[..., Sequence[MemorySummary]]
SourceScopeValidator = Callable[[int, tuple[str, ...]], bool]


@dataclass(frozen=True, slots=True)
class MemoryV2Request:
    group_id: int
    query: str
    recent_messages: tuple[EvidenceMessage, ...]
    quoted_message: EvidenceMessage | None
    target_message_id: str | None
    available_input: int
    now: datetime | None = None


@dataclass(frozen=True, slots=True)
class MemoryV2EvaluationTrace:
    result: MemoryContextResult
    resolved_query: ResolvedMemoryQuery
    retrieved_source_msg_ids: tuple[str, ...]
    retrieved_source_units: tuple[tuple[str, ...], ...]
    candidate_scores: tuple[tuple[int, float], ...]
    attempted_channels: tuple[str, ...] = ()
    failed_channels: tuple[str, ...] = ()
    channel_candidate_counts: tuple[tuple[str, int], ...] = ()


class MemoryV2ContextProvider:
    """Run resolver -> retrieval -> expansion -> packing as one fallback unit."""

    def __init__(
        self,
        *,
        resolver: QueryResolver,
        retriever: Retriever,
        expander: Expander,
        packer: MemoryContextPacker,
        source_scope_validator: SourceScopeValidator,
        fact_loader: FactLoader | None = None,
        summary_loader: SummaryLoader | None = None,
    ) -> None:
        self._resolver = resolver
        self._retriever = retriever
        self._expander = expander
        self._packer = packer
        self._source_scope_validator = source_scope_validator
        self._fact_loader = fact_loader or (lambda **_: ())
        self._summary_loader = summary_loader or (lambda **_: ())

    def __call__(self, request: MemoryV2Request) -> MemoryContextResult:
        return self.evaluate(request).result

    def evaluate(self, request: MemoryV2Request) -> MemoryV2EvaluationTrace:
        """Run V2 and expose an in-memory, content-free evaluation trace."""
        self._validate_recent_scope(request)
        resolved = self._resolver.resolve(
            request.query,
            recent_messages=request.recent_messages,
            quoted_message=request.quoted_message,
            now=request.now,
        )
        retrieval_result = self._retriever.retrieve(
            group_id=request.group_id,
            resolved_query=resolved,
        )
        if bool(getattr(retrieval_result, "all_channels_failed", False)):
            raise RuntimeError("all memory retrieval channels failed")
        candidates = tuple(getattr(retrieval_result, "candidates"))
        mode = "detail" if resolved.needs_detail else "normal"
        segments = tuple(
            self._expander.expand(
                group_id=request.group_id,
                candidates=candidates,
                mode=mode,
            )
        )
        facts = tuple(
            self._fact_loader(
                group_id=request.group_id,
                resolved_query=resolved,
            )
        )
        summaries = tuple(
            self._summary_loader(
                group_id=request.group_id,
                resolved_query=resolved,
            )
        )
        self._validate_derived_scope(
            group_id=request.group_id,
            facts=facts,
            summaries=summaries,
        )
        packed = self._packer.pack(
            mode,
            available_input=request.available_input,
            target_message_id=request.target_message_id,
            recent_messages=request.recent_messages,
            evidence_segments=segments,
            facts=facts,
            summaries=summaries,
        )
        if packed.source_msg_ids and not self._source_scope_validator(
            request.group_id,
            packed.source_msg_ids,
        ):
            raise MemoryScopeViolation("packed memory source scope mismatch")
        result = MemoryContextResult(
            group_id=request.group_id,
            packed_context=packed,
            selected_source_msg_ids=packed.source_msg_ids,
            estimated_tokens=packed.estimated_tokens,
            mode="v2",
        )
        retrieved_source_msg_ids = tuple(
            dict.fromkeys(
                str(source_id)
                for candidate in candidates
                for source_id in getattr(candidate, "source_msg_ids", ())
                if str(source_id)
            )
        )
        return MemoryV2EvaluationTrace(
            result=result,
            resolved_query=resolved,
            retrieved_source_msg_ids=retrieved_source_msg_ids,
            retrieved_source_units=tuple(
                tuple(
                    dict.fromkeys(
                        str(source_id)
                        for source_id in getattr(candidate, "source_msg_ids", ())
                        if str(source_id)
                    )
                )
                for candidate in candidates
                if str(getattr(candidate, "document_kind", "")) == "episode"
            ),
            candidate_scores=tuple(
                (int(getattr(candidate, "document_id")), float(getattr(candidate, "fused_score")))
                for candidate in candidates
            ),
            attempted_channels=tuple(
                str(channel)
                for channel in getattr(retrieval_result, "attempted_channels", ())
            ),
            failed_channels=tuple(
                str(channel)
                for channel in getattr(retrieval_result, "failed_channels", ())
            ),
            channel_candidate_counts=tuple(
                (str(channel), int(count))
                for channel, count in getattr(
                    retrieval_result,
                    "channel_candidate_counts",
                    (),
                )
            ),
        )

    @staticmethod
    def _validate_recent_scope(request: MemoryV2Request) -> None:
        scoped = (*request.recent_messages,) + (
            (request.quoted_message,) if request.quoted_message is not None else ()
        )
        if any(
            message.group_id is None or int(message.group_id) != int(request.group_id)
            for message in scoped
        ):
            raise ValueError("memory recent snapshot scope mismatch")

    @staticmethod
    def _validate_derived_scope(
        *,
        group_id: int,
        facts: Sequence[MemoryFact],
        summaries: Sequence[MemorySummary],
    ) -> None:
        for item in (*facts, *summaries):
            if item.group_id is None or int(item.group_id) != int(group_id):
                raise MemoryScopeViolation("derived memory scope mismatch")
            if not item.source_msg_ids or any(not str(source_id) for source_id in item.source_msg_ids):
                raise MemoryScopeViolation("derived memory provenance is missing")
