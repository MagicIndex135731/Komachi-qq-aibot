"""Private chat admission; friend identity comes only from the QQ gateway."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import logging


logger = logging.getLogger(__name__)


class PrivateChatAccess:
    def __init__(
        self, *, owner_qq: int, allowlist: set[int] | None = None,
        all_friends: bool = False,
        friend_reader: Callable[[], Awaitable[tuple[int, ...]]] | None = None,
        timeout_seconds: float = 8,
    ) -> None:
        self.owner_qq = owner_qq
        self.allowlist = allowlist if allowlist is not None else set()
        self.all_friends = all_friends
        self.friend_reader = friend_reader
        self.timeout_seconds = timeout_seconds
        self._friend_ids: frozenset[int] = frozenset()

    def allows(self, user_id: int) -> bool:
        """Use the last verified roster inside a synchronous dispatch fence."""
        return (user_id == self.owner_qq or user_id in self.allowlist
            or self.all_friends and user_id in self._friend_ids)

    async def authorize(self, user_id: int) -> bool:
        """Refresh for each friend message and queued turn, with no stale fallback."""
        if user_id == self.owner_qq or user_id in self.allowlist:
            return True
        if not self.all_friends or self.friend_reader is None:
            return False
        try:
            friends = await asyncio.wait_for(self.friend_reader(), self.timeout_seconds)
        except asyncio.CancelledError:
            self._friend_ids = frozenset()
            raise
        except Exception as exc:
            self._friend_ids = frozenset()
            logger.warning("private_friend_access_read_failed error_type=%s", type(exc).__name__)
            return False
        self._friend_ids = frozenset(friends)
        return user_id in self._friend_ids
