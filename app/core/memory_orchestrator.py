from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Protocol


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


@dataclass(frozen=True, slots=True)
class ShadowJobRequest:
    """ID/version-only shadow payload safe for persistence and logging."""

    group_id: int
    current_msg_id: str
    config_version: str = ""
    index_generation: str = ""


ContextProvider = Callable[[GroupMemoryRequest], MemoryContextResult]
ShadowEnqueuer = Callable[[ShadowJobRequest], None]


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
    ) -> None:
        self.v2_enabled = bool(v2_enabled)
        self.shadow_mode = bool(shadow_mode) and self.v2_enabled
        self.v2_provider = v2_provider
        self.legacy_provider = legacy_provider
        self.recent_provider = recent_provider
        self.shadow_enqueue = shadow_enqueue

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
                "memory_v2_fallback group_id=%s error_type=%s",
                request.group_id,
                type(exc).__name__,
            )
            return self._legacy_or_recent(request)

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
    def _empty_context(group_id: int) -> MemoryContextResult:
        return MemoryContextResult(
            group_id=int(group_id),
            packed_context="",
            selected_source_msg_ids=(),
            estimated_tokens=0,
            mode="empty",
        )

    @staticmethod
    def _validate_scope(result: MemoryContextResult, group_id: int) -> MemoryContextResult:
        if int(result.group_id) != int(group_id):
            raise ValueError("memory context scope mismatch")
        return result
