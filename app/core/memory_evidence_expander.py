"""Expand fused retrieval hits into group- and episode-bounded evidence."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from app.core.hybrid_memory_retriever import (
    FusedRetrievalCandidate,
    MemoryScopeViolation,
)
from app.core.memory_context_packer import EvidenceMessage, EvidenceSegment


ExpansionMode = Literal["normal", "detail"]
EpisodeLoader = Callable[..., Sequence[EvidenceMessage]]


class MemoryEvidenceExpander:
    """Build chronological evidence windows without using global ID arithmetic.

    The injected loader must apply both ``group_id`` and ``episode_id`` in SQL.
    This class validates the returned batch again before constructing prompt
    evidence, so a repository regression fails the whole V2 path closed.
    """

    def __init__(
        self,
        *,
        episode_loader: EpisodeLoader,
        normal_radius: int = 5,
        detail_radius: int = 10,
        normal_segment_limit: int = 4,
        detail_segment_limit: int = 6,
        max_reply_depth: int = 2,
    ) -> None:
        if min(normal_radius, detail_radius, max_reply_depth) < 0:
            raise ValueError("expansion radii and reply depth cannot be negative")
        if min(normal_segment_limit, detail_segment_limit) <= 0:
            raise ValueError("segment limits must be positive")
        self._episode_loader = episode_loader
        self._radii = {"normal": normal_radius, "detail": detail_radius}
        self._limits = {
            "normal": normal_segment_limit,
            "detail": detail_segment_limit,
        }
        self._max_reply_depth = max_reply_depth

    def expand(
        self,
        *,
        group_id: int,
        candidates: Sequence[FusedRetrievalCandidate],
        mode: ExpansionMode,
    ) -> tuple[EvidenceSegment, ...]:
        if mode not in self._radii:
            raise ValueError(f"unknown expansion mode: {mode}")
        segments: list[EvidenceSegment] = []
        for candidate in candidates[: self._limits[mode]]:
            segment = self._expand_candidate(
                group_id=group_id,
                candidate=candidate,
                radius=self._radii[mode],
            )
            if segment is not None:
                segments.append(segment)
        return tuple(segments)

    def _expand_candidate(
        self,
        *,
        group_id: int,
        candidate: FusedRetrievalCandidate,
        radius: int,
    ) -> EvidenceSegment | None:
        if int(candidate.group_id) != int(group_id):
            raise MemoryScopeViolation(
                f"candidate scope mismatch document_id={candidate.document_id}"
            )
        if candidate.episode_id is None:
            raise MemoryScopeViolation(
                f"candidate has no episode document_id={candidate.document_id}"
            )
        loaded = tuple(
            self._episode_loader(
                group_id=group_id,
                episode_id=candidate.episode_id,
            )
        )
        ordered = tuple(sorted(loaded, key=lambda row: (row.sent_at, row.source_msg_id)))
        if any(row.group_id is None or int(row.group_id) != int(group_id) for row in ordered):
            raise MemoryScopeViolation(
                f"episode scope mismatch episode_id={candidate.episode_id}"
            )

        by_id: dict[str, EvidenceMessage] = {}
        for row in ordered:
            if row.source_msg_id in by_id:
                raise MemoryScopeViolation(
                    f"duplicate episode source episode_id={candidate.episode_id}"
                )
            by_id[row.source_msg_id] = row
        missing = tuple(source_id for source_id in candidate.source_msg_ids if source_id not in by_id)
        if missing:
            raise MemoryScopeViolation(
                f"unverified provenance document_id={candidate.document_id}"
            )

        indexes = {row.source_msg_id: index for index, row in enumerate(ordered)}
        selected_ids: set[str] = set()
        for source_id in candidate.source_msg_ids:
            center = indexes[source_id]
            start = max(0, center - radius)
            end = min(len(ordered), center + radius + 1)
            selected_ids.update(row.source_msg_id for row in ordered[start:end])

        for source_id in candidate.source_msg_ids:
            current = by_id[source_id]
            seen = {source_id}
            for _ in range(self._max_reply_depth):
                parent_id = current.reply_to_msg_id
                if not parent_id or parent_id in seen or parent_id not in by_id:
                    break
                seen.add(parent_id)
                selected_ids.add(parent_id)
                current = by_id[parent_id]

        atomic_groups: list[tuple[str, ...]] = []
        questions = set(selected_ids)
        for row in ordered:
            if row.is_bot and row.reply_to_msg_id in questions:
                selected_ids.add(row.source_msg_id)
                pair = (str(row.reply_to_msg_id), row.source_msg_id)
                if pair not in atomic_groups:
                    atomic_groups.append(pair)

        selected_with_blocked = tuple(
            row for row in ordered if row.source_msg_id in selected_ids
        )
        blocked_output_present = any(row.blocked for row in selected_with_blocked)
        selected = tuple(row for row in selected_with_blocked if not row.blocked)
        # Blocked raw rows may remain in the recent continuity section, but an
        # expanded segment is derived prompt evidence and must not reproduce it.
        if not selected:
            return None
        return EvidenceSegment(
            episode_id=str(candidate.episode_id),
            fused_score=candidate.fused_score,
            messages=selected,
            hit_source_msg_ids=candidate.source_msg_ids,
            document_id=str(candidate.document_id),
            atomic_source_groups=tuple(atomic_groups),
            pinned="exact_quote" in candidate.routes,
            blocked_output_present=blocked_output_present,
        )
