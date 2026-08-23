from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Protocol

from app.core.memory_context_packer import (
    MEMORY_GROUNDING_NO_EVIDENCE,
    PackedMemoryContext,
)

logger = logging.getLogger(__name__)


class GroupMemoryRequest(Protocol):
    group_id: int


@dataclass(frozen=True, slots=True)
class MemoryContextResult:
    group_id: int
    packed_context: Any
    selected_source_msg_ids: tuple[str, ...]
    estimated_tokens: int
    mode: str
    resolved_answer_mode: str = ""
    resolved_subject_ids: tuple[str, ...] | None = None
    resolved_subject_binding: str = ""


@dataclass(frozen=True, slots=True)
class ShadowJobRequest:
    """ID/version-only shadow payload safe for persistence and logging."""

    group_id: int
    current_msg_id: str
    config_version: str = ""
    index_generation: str = ""


ContextProvider = Callable[[GroupMemoryRequest], MemoryContextResult]
ShadowEnqueuer = Callable[[ShadowJobRequest], None]
HistoryRequestPredicate = Callable[[GroupMemoryRequest], bool]


class MemoryOrchestrator:
    """Select V1/V2 memory context without letting V2 block normal replies."""

    def __init__(
        self,
        *,
        v2_enabled: bool,
        shadow_mode: bool,
        v2_provider: ContextProvider,
        legacy_provider: ContextProvider,
        recent_provider: ContextProvider,
        shadow_enqueue: ShadowEnqueuer | None = None,
        strict_scoped_fallback: bool = False,
        history_request_predicate: HistoryRequestPredicate | None = None,
    ) -> None:
        self.v2_enabled = bool(v2_enabled)
        self.shadow_mode = bool(shadow_mode) and self.v2_enabled
        self.v2_provider = v2_provider
        self.legacy_provider = legacy_provider
        self.recent_provider = recent_provider
        self.shadow_enqueue = shadow_enqueue
        self.strict_scoped_fallback = bool(strict_scoped_fallback)
        self.history_request_predicate = history_request_predicate

    def build_context(self, request: GroupMemoryRequest) -> MemoryContextResult:
        if not self.v2_enabled:
            return self._legacy_or_recent(request)

        if self.shadow_mode:
            selected = self._legacy_or_recent(request)
            if self.shadow_enqueue is not None:
                try:
                    self.shadow_enqueue(self._build_shadow_job_request(request))
                except Exception as exc:
                    logger.warning(
                        "memory_shadow_enqueue_failed group_id=%s error_type=%s",
                        request.group_id,
                        type(exc).__name__,
                    )
            return selected

        try:
            return self._validate_scope(self.v2_provider(request), request.group_id)
        except Exception as exc:
            logger.warning(
                "memory_v2_safe_fallback group_id=%s error_type=%s "
                "strict_scoped=%s",
                request.group_id,
                type(exc).__name__,
                self.strict_scoped_fallback,
            )
            if self.strict_scoped_fallback:
                is_history = True
                if self.history_request_predicate is not None:
                    try:
                        is_history = bool(self.history_request_predicate(request))
                    except Exception:
                        is_history = True
                if is_history:
                    logger.warning(
                        "memory_fallback route=v3_safe_empty group_id=%s "
                        "reason=v2_exception",
                        request.group_id,
                    )
                    return self._empty_context(
                        request.group_id,
                        mode="v3_safe_empty",
                    )
                logger.warning(
                    "memory_fallback route=scoped_recent group_id=%s "
                    "reason=ordinary_v2_exception",
                    request.group_id,
                )
                return self._recent_or_empty(request)
            return self._legacy_or_recent(request)

    def _recent_or_empty(self, request: GroupMemoryRequest) -> MemoryContextResult:
        try:
            return self._validate_scope(
                self.recent_provider(request),
                request.group_id,
            )
        except Exception as exc:
            logger.warning(
                "memory_recent_fallback_failed group_id=%s error_type=%s",
                request.group_id,
                type(exc).__name__,
            )
            return self._empty_context(request.group_id)

    def _legacy_or_recent(self, request: GroupMemoryRequest) -> MemoryContextResult:
        try:
            return self._validate_scope(self.legacy_provider(request), request.group_id)
        except Exception as exc:
            logger.warning(
                "memory_v1_fallback group_id=%s error_type=%s",
                request.group_id,
                type(exc).__name__,
            )
            try:
                return self._validate_scope(
                    self.recent_provider(request),
                    request.group_id,
                )
            except Exception as recent_exc:
                logger.warning(
                    "memory_recent_fallback_failed group_id=%s error_type=%s",
                    request.group_id,
                    type(recent_exc).__name__,
                )
                return self._empty_context(request.group_id)

    @staticmethod
    def _build_shadow_job_request(request: GroupMemoryRequest) -> ShadowJobRequest:
        message_id = getattr(request, "current_msg_id", None)
        if not isinstance(message_id, (str, int)) or isinstance(message_id, bool) or not str(message_id).strip():
            message_id = getattr(request, "target_message_id", None)
        if not isinstance(message_id, (str, int)) or isinstance(message_id, bool) or not str(message_id).strip():
            raise ValueError("shadow request is missing a current message ID")

        config_version = getattr(request, "config_version", "")
        index_generation = getattr(request, "index_generation", "")
        return ShadowJobRequest(
            group_id=int(request.group_id),
            current_msg_id=str(message_id).strip(),
            config_version=(
                str(config_version).strip()
                if isinstance(config_version, (str, int)) and not isinstance(config_version, bool)
                else ""
            ),
            index_generation=(
                str(index_generation).strip()
                if isinstance(index_generation, (str, int)) and not isinstance(index_generation, bool)
                else ""
            ),
        )

    @staticmethod
    def _empty_context(group_id: int, *, mode: str = "empty") -> MemoryContextResult:
        packed = PackedMemoryContext(
            mode="normal",
            budget=256,
            estimated_tokens=len(MEMORY_GROUNDING_NO_EVIDENCE.split()),
            text=MEMORY_GROUNDING_NO_EVIDENCE,
            grounding_policy=MEMORY_GROUNDING_NO_EVIDENCE,
        )
        return MemoryContextResult(
            group_id=int(group_id),
            packed_context=packed,
            selected_source_msg_ids=(),
            estimated_tokens=packed.estimated_tokens,
            mode=mode,
        )

    @staticmethod
    def _validate_scope(result: MemoryContextResult, group_id: int) -> MemoryContextResult:
        if int(result.group_id) != int(group_id):
            raise ValueError("memory context scope mismatch")
        return result
