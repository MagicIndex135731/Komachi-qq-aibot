"""User media work is metered, without a cumulative application quota."""
from __future__ import annotations

import asyncio


class RequestWorkTimeout(TimeoutError):
    """Only the enclosing ordinary turn timer expired."""


def unmetered_media(workspace=None, guard=None):
    workspace = workspace or getattr(guard, 'request_workspace', None)
    budget = getattr(workspace, 'media_budget', None)
    return bool(budget is not None and budget.unmetered)


def bind_request_workspace(guard, workspace):
    """Expose a late model-selected media goal to enclosing execution timers."""
    seen = set()
    while guard is not None and id(guard) not in seen:
        seen.add(id(guard))
        guard.request_workspace = workspace
        guard = getattr(guard, 'parent', None) or getattr(guard, 'dispatch_parent', None)


async def await_request_work(work, *, timeout, workspace=None, guard=None):
    """Keep ordinary turn timeouts; media continues under per-operation timeouts.

    Check again at expiry because set_task_goal may discover the media intent
    after execution starts. Cancellation still cancels and joins the work.
    """
    loop=asyncio.get_running_loop()
    timer=asyncio.timeout(None)
    def expire():
        try:
            if unmetered_media(workspace,guard):return
        except (ValueError,RuntimeError,OSError):
            pass  # Invalid authority must not disable the ordinary timer.
        timer.reschedule(loop.time())
    handle=loop.call_later(timeout,expire)
    try:
        async with timer:
            return await work
    except TimeoutError as exc:
        if timer.expired():raise RequestWorkTimeout() from exc
        raise
    finally:
        handle.cancel()
