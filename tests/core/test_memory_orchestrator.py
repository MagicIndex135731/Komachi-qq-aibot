from __future__ import annotations

from dataclasses import dataclass

from app.core.memory_orchestrator import (
    MemoryContextResult,
    MemoryOrchestrator,
    ShadowJobRequest,
)


@dataclass(frozen=True)
class Request:
    group_id: int
    current_msg_id: str = "m-1"
    query: str = "敏感查询正文"
    recent_messages: tuple[str, ...] = ("敏感最近消息",)


def result(group_id: int, mode: str) -> MemoryContextResult:
    return MemoryContextResult(
        group_id=group_id,
        packed_context={"mode": mode},
        selected_source_msg_ids=(f"{mode}-source",),
        estimated_tokens=10,
        mode=mode,
    )


def test_disabled_mode_uses_only_legacy_provider() -> None:
    calls: list[str] = []
    orchestrator = MemoryOrchestrator(
        v2_enabled=False,
        shadow_mode=True,
        v2_provider=lambda _request: calls.append("v2") or result(100, "v2"),
        legacy_provider=lambda _request: calls.append("v1") or result(100, "v1"),
        recent_provider=lambda _request: calls.append("recent") or result(100, "recent"),
        shadow_enqueue=lambda _request: calls.append("shadow"),
    )

    resolved = orchestrator.build_context(Request(group_id=100))

    assert resolved.mode == "v1"
    assert calls == ["v1"]


def test_shadow_mode_returns_v1_and_only_enqueues_v2_work() -> None:
    calls: list[str] = []
    jobs: list[ShadowJobRequest] = []
    orchestrator = MemoryOrchestrator(
        v2_enabled=True,
        shadow_mode=True,
        v2_provider=lambda _request: calls.append("v2") or result(100, "v2"),
        legacy_provider=lambda _request: calls.append("v1") or result(100, "v1"),
        recent_provider=lambda _request: calls.append("recent") or result(100, "recent"),
        shadow_enqueue=lambda job: jobs.append(job) or calls.append(f"shadow:{job.current_msg_id}"),
    )

    resolved = orchestrator.build_context(Request(group_id=100))

    assert resolved.mode == "v1"
    assert calls == ["v1", "shadow:m-1"]
    assert jobs == [
        ShadowJobRequest(
            group_id=100,
            current_msg_id="m-1",
        )
    ]
    assert not hasattr(jobs[0], "query")
    assert not hasattr(jobs[0], "recent_messages")


def test_active_mode_returns_v2_context() -> None:
    orchestrator = MemoryOrchestrator(
        v2_enabled=True,
        shadow_mode=False,
        v2_provider=lambda _request: result(100, "v2"),
        legacy_provider=lambda _request: result(100, "v1"),
        recent_provider=lambda _request: result(100, "recent"),
    )

    assert orchestrator.build_context(Request(group_id=100)).mode == "v2"


def test_active_failure_falls_back_to_independent_v1() -> None:
    def broken(_request):
        raise RuntimeError("sensitive provider body")

    orchestrator = MemoryOrchestrator(
        v2_enabled=True,
        shadow_mode=False,
        v2_provider=broken,
        legacy_provider=lambda _request: result(100, "v1"),
        recent_provider=lambda _request: result(100, "recent"),
    )

    assert orchestrator.build_context(Request(group_id=100)).mode == "v1"


def test_cross_group_v2_result_is_discarded_and_falls_back() -> None:
    orchestrator = MemoryOrchestrator(
        v2_enabled=True,
        shadow_mode=False,
        v2_provider=lambda _request: result(200, "v2"),
        legacy_provider=lambda _request: result(100, "v1"),
        recent_provider=lambda _request: result(100, "recent"),
    )

    assert orchestrator.build_context(Request(group_id=100)).mode == "v1"


def test_legacy_failure_uses_minimal_recent_context() -> None:
    def broken(_request):
        raise RuntimeError("legacy failed")

    orchestrator = MemoryOrchestrator(
        v2_enabled=True,
        shadow_mode=False,
        v2_provider=broken,
        legacy_provider=broken,
        recent_provider=lambda _request: result(100, "recent"),
    )

    assert orchestrator.build_context(Request(group_id=100)).mode == "recent"


def test_recent_failure_returns_safe_empty_context_instead_of_blocking_reply() -> None:
    def broken(_request):
        raise RuntimeError("provider failed")

    orchestrator = MemoryOrchestrator(
        v2_enabled=True,
        shadow_mode=False,
        v2_provider=broken,
        legacy_provider=broken,
        recent_provider=broken,
    )

    resolved = orchestrator.build_context(Request(group_id=100))

    assert resolved == MemoryContextResult(
        group_id=100,
        packed_context="",
        selected_source_msg_ids=(),
        estimated_tokens=0,
        mode="empty",
    )


def test_shadow_job_uses_target_message_id_when_current_alias_is_absent() -> None:
    @dataclass(frozen=True)
    class TargetRequest:
        group_id: int
        target_message_id: str
        query: str = "不得进入 shadow job"

    jobs: list[ShadowJobRequest] = []
    orchestrator = MemoryOrchestrator(
        v2_enabled=True,
        shadow_mode=True,
        v2_provider=lambda _request: result(100, "v2"),
        legacy_provider=lambda _request: result(100, "v1"),
        recent_provider=lambda _request: result(100, "recent"),
        shadow_enqueue=jobs.append,
    )

    resolved = orchestrator.build_context(
        TargetRequest(group_id=100, target_message_id="target-7")
    )

    assert resolved.mode == "v1"
    assert jobs == [ShadowJobRequest(group_id=100, current_msg_id="target-7")]


def test_shadow_enqueue_failure_never_changes_the_v1_reply_context() -> None:
    def broken_enqueue(_request):
        raise RuntimeError("queue unavailable")

    orchestrator = MemoryOrchestrator(
        v2_enabled=True,
        shadow_mode=True,
        v2_provider=lambda _request: result(100, "v2"),
        legacy_provider=lambda _request: result(100, "v1"),
        recent_provider=lambda _request: result(100, "recent"),
        shadow_enqueue=broken_enqueue,
    )

    assert orchestrator.build_context(Request(group_id=100)).mode == "v1"
