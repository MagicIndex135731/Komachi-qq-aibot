import asyncio
from contextvars import ContextVar
import threading

import pytest

from app.runtime.blocking import AgentModelBusyError, BoundedBlockingRunner, blocking_runner_scope, run_blocking


@pytest.mark.asyncio
async def test_cancelled_sync_call_keeps_capacity_until_real_work_finishes():
    entered, release = threading.Event(), threading.Event()
    runner = BoundedBlockingRunner(max_workers=1)

    def slow():
        entered.set()
        release.wait(timeout=2)
        return "done"

    task = asyncio.create_task(runner.run(slow))
    try:
        async with asyncio.timeout(1):
            while not entered.is_set():
                await asyncio.sleep(0.003)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(AgentModelBusyError):
            await runner.run(lambda: "should not queue")
        release.set()
        async with asyncio.timeout(1):
            while True:
                try:
                    assert await runner.run(lambda: "capacity restored") == "capacity restored"
                    break
                except AgentModelBusyError:
                    await asyncio.sleep(0.003)
    finally:
        release.set()
        runner.close()


@pytest.mark.asyncio
async def test_nested_router_blocking_call_preserves_scoped_context():
    marker = ContextVar("isolated-test-marker", default="none")
    runner = BoundedBlockingRunner(max_workers=1)
    token = marker.set("agent-generation")
    try:
        with blocking_runner_scope(runner):
            assert await run_blocking(marker.get) == "agent-generation"
    finally:
        marker.reset(token)
        runner.close()
    assert marker.get() == "none"
    with pytest.raises(AgentModelBusyError):
        await runner.run(lambda: "closed")
