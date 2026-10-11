"""Turn-scoped QQ dispatch checks, shared by text and image send paths."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


class GroupDispatchRevokedError(RuntimeError):
    """The controller revoked a group action before its API dispatch."""


@dataclass(slots=True)
class GroupDispatchScope:
    group_id: int
    can_send: Callable[[], bool]
    on_dispatch: Callable[[], None]
    receipts: list[str] = field(default_factory=list)
    dispatched: bool = False
    uncertain: bool = False
    failed: bool = False


_group_scope: ContextVar[GroupDispatchScope | None] = ContextVar("qq_group_dispatch", default=None)


@contextmanager
def group_dispatch_scope(scope: GroupDispatchScope) -> Iterator[GroupDispatchScope]:
    token = _group_scope.set(scope)
    try:
        yield scope
    finally:
        _group_scope.reset(token)


def current_group_dispatch(group_id: int) -> GroupDispatchScope | None:
    scope = _group_scope.get()
    if scope is not None and scope.group_id != int(group_id):
        raise ValueError("QQ action escaped the current Agent group")
    return scope
