"""Bound synchronous model work even when its awaiting turn is cancelled."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import partial
import threading


class AgentModelBusyError(RuntimeError):
    pass


class DeferredCleanup:
    """Keep temporary pixels alive until a cancelled model thread has finished."""
    def __init__(self, cleanup):
        self.cleanup = cleanup
        self._lock = threading.Lock()
        self._users = 0
        self._closed = False

    @contextmanager
    def hold(self):
        with self._lock:
            if self._closed:
                raise AgentModelBusyError("turn resources already closed")
            self._users += 1
        try:
            yield
        finally:
            with self._lock:
                self._users -= 1
                if self._closed and self._users == 0 and self.cleanup is not None:
                    self.cleanup()
                    self.cleanup = None

    def close(self):
        with self._lock:
            self._closed = True
            if self._users == 0 and self.cleanup is not None:
                self.cleanup()
                self.cleanup = None


class BoundedBlockingRunner:
    def __init__(self, max_workers: int = 2) -> None:
        self._slots = threading.BoundedSemaphore(max_workers)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="qq-agent")
        self._closed = False

    async def run(self, function, *args, **kwargs):
        if self._closed or not self._slots.acquire(blocking=False):
            raise AgentModelBusyError("Agent blocking work capacity unavailable")
        try:
            future = self._pool.submit(copy_context().run, partial(function, *args, **kwargs))
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        # Cancelling the turn revokes its send guard, while this slot remains
        # occupied until the real HTTP/thread work finishes.
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(lambda result: result.exception() if not result.cancelled() else None)
        return await asyncio.shield(wrapped)

    def close(self) -> None:
        self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)


_runner: ContextVar[BoundedBlockingRunner | None] = ContextVar("qq_agent_blocking", default=None)


@contextmanager
def blocking_runner_scope(runner: BoundedBlockingRunner) -> Iterator[None]:
    token = _runner.set(runner)
    try:
        yield
    finally:
        _runner.reset(token)


async def run_blocking(function, *args, **kwargs):
    runner = _runner.get()
    if runner is not None:
        return await runner.run(function, *args, **kwargs)
    return await asyncio.to_thread(function, *args, **kwargs)
