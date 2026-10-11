import asyncio

import pytest

from app.private_chat.access import PrivateChatAccess


OWNER, FRIEND, STRANGER = 900000101, 900000102, 900000103


async def test_static_access_does_not_read_friends():
    async def unavailable():
        raise AssertionError("fixed permissions must not need the gateway")

    access = PrivateChatAccess(owner_qq=OWNER, allowlist={FRIEND}, friend_reader=unavailable)
    assert await access.authorize(OWNER)
    assert await access.authorize(FRIEND)
    assert not await access.authorize(STRANGER)


async def test_all_friends_tracks_additions_and_removals():
    friends = {FRIEND}

    async def read():
        return tuple(friends)

    access = PrivateChatAccess(owner_qq=OWNER, all_friends=True, friend_reader=read)
    assert await access.authorize(FRIEND)
    assert access.allows(FRIEND)
    assert not await access.authorize(STRANGER)
    friends.add(STRANGER)
    assert await access.authorize(STRANGER)
    friends.remove(FRIEND)
    assert not await access.authorize(FRIEND)
    assert not access.allows(FRIEND)


async def test_friend_failure_clears_previous_permission(caplog):
    failure = False

    async def read():
        if failure:
            raise RuntimeError("synthetic private error content")
        return (FRIEND,)

    access = PrivateChatAccess(owner_qq=OWNER, all_friends=True, friend_reader=read)
    assert await access.authorize(FRIEND)
    failure = True
    assert not await access.authorize(FRIEND)
    assert not access.allows(FRIEND)
    assert await access.authorize(OWNER)
    assert "synthetic private error content" not in caplog.text


async def test_missing_reader_does_not_open_to_strangers():
    access = PrivateChatAccess(owner_qq=OWNER, all_friends=True)
    assert not await access.authorize(STRANGER)


async def test_friend_read_timeout_is_bounded():
    async def read():
        await asyncio.Event().wait()

    access = PrivateChatAccess(owner_qq=OWNER, all_friends=True, friend_reader=read, timeout_seconds=0.01)
    assert not await access.authorize(FRIEND)
    assert not access.allows(FRIEND)


async def test_friend_read_cancellation_propagates():
    started = asyncio.Event()

    async def read():
        started.set()
        await asyncio.Event().wait()

    access = PrivateChatAccess(owner_qq=OWNER, all_friends=True, friend_reader=read)
    task = asyncio.create_task(access.authorize(FRIEND))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not access.allows(FRIEND)
