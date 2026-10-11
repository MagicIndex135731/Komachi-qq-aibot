from __future__ import annotations

import asyncio
import base64
import pytest
from PIL import Image

from app.adapters.account_profile import (
    AccountProfileAdapter,
    ProfileReceipt,
    ProfileRejectedError,
    ProfileResultUncertainError,
    ProfileUnavailableError,
)
from app.adapters.sender import QQMessageDeliveryUncertainError


ACCOUNT = 900000101
GROUP = 900000001
OK = {"status": "ok", "retcode": 0, "data": None}


class Gateway:
    def __init__(self, response=OK):
        self.response = response
        self.calls = []
        self.websocket = object()
        self.started = asyncio.Event()

    async def call_api(self, action, params):
        self.started.set()
        self.calls.append((action, params))
        if isinstance(self.response, BaseException):
            raise self.response
        if callable(self.response):
            return await self.response()
        return self.response


def adapter(gateway, **kwargs):
    return AccountProfileAdapter(
        gateway, account_id=ACCOUNT, allowed_group_ids=(GROUP,), **kwargs
    )


async def test_nickname_is_account_profile_only_and_has_real_acknowledgement():
    gateway = Gateway()
    result = await adapter(gateway).nickname("比企谷小町")
    assert result == ProfileReceipt(action="set_qq_profile", account_id=ACCOUNT)
    assert result.acknowledged
    assert gateway.calls == [("set_qq_profile", {"nickname": "比企谷小町"})]
    assert not hasattr(result, "receipt_id")


async def test_card_changes_only_the_own_member_in_an_allowed_group():
    gateway = Gateway()
    result = await adapter(gateway).card(GROUP, ACCOUNT, "画画的小町")
    assert result.group_id == GROUP
    assert gateway.calls == [
        ("set_group_card", {"group_id": GROUP, "user_id": ACCOUNT, "card": "画画的小町"})
    ]


@pytest.mark.parametrize("group_id,bot_id", [(GROUP + 1, ACCOUNT), (GROUP, ACCOUNT + 1)])
async def test_card_rejects_other_groups_and_other_members(group_id, bot_id):
    gateway = Gateway()
    with pytest.raises(ProfileRejectedError):
        await adapter(gateway).card(group_id, bot_id, "new")
    assert gateway.calls == []


async def test_group_allowlist_is_empty_by_default():
    gateway = Gateway()
    with pytest.raises(ProfileRejectedError):
        await AccountProfileAdapter(gateway, account_id=ACCOUNT).card(GROUP, ACCOUNT, "new")
    assert gateway.calls == []


@pytest.mark.parametrize("name", ["", "  ", "x" * 33, "name\x00", "name\n", "\ud800", 42, True])
async def test_nickname_validation_happens_before_submission(name):
    gateway = Gateway()
    with pytest.raises(ValueError):
        await adapter(gateway).nickname(name)
    assert gateway.calls == []


@pytest.mark.parametrize("name", ["", "x" * 61, "name\x1b", "name\t", None])
async def test_card_validation_happens_before_submission(name):
    gateway = Gateway()
    with pytest.raises(ValueError):
        await adapter(gateway).card(GROUP, ACCOUNT, name)
    assert gateway.calls == []


async def test_avatar_is_verified_and_submitted_as_bounded_bytes(tmp_path):
    image = tmp_path / "avatar.png"
    Image.new("RGB", (32, 32), "pink").save(image)
    gateway = Gateway()
    result = await adapter(gateway, media_roots=(tmp_path,)).avatar(str(image))
    action, params = gateway.calls[0]
    assert action == "set_qq_avatar"
    assert base64.b64decode(params["file"][9:]) == image.read_bytes()
    assert params["file"].startswith("base64://")
    assert len(result.artifact_sha256) == 64


@pytest.mark.parametrize("file", ["https://qpic.cn/image.png", "base64://YWJj", "relative.png", "file:///tmp/a.png", "\\\\server\\share\\avatar.png"])
async def test_avatar_rejects_arbitrary_references_without_source_provenance(tmp_path, file):
    gateway = Gateway()
    with pytest.raises(ValueError):
        await adapter(gateway, media_roots=(tmp_path,)).avatar(file)
    assert gateway.calls == []


async def test_avatar_rejects_images_outside_authorized_roots(tmp_path):
    root = tmp_path / "authorized"
    root.mkdir()
    other = tmp_path / "other.png"
    Image.new("RGB", (10, 10)).save(other)
    gateway = Gateway()
    with pytest.raises(ProfileRejectedError):
        await adapter(gateway, media_roots=(root,)).avatar(str(other))
    assert gateway.calls == []


async def test_avatar_has_no_file_access_without_explicit_roots(tmp_path):
    image = tmp_path / "avatar.png"
    Image.new("RGB", (10, 10)).save(image)
    gateway = Gateway()
    with pytest.raises(ProfileRejectedError):
        await adapter(gateway).avatar(str(image))
    assert gateway.calls == []


async def test_avatar_follows_resolved_path_and_rejects_symlink_escape(tmp_path):
    root = tmp_path / "authorized"
    root.mkdir()
    other = tmp_path / "other.png"
    Image.new("RGB", (10, 10)).save(other)
    alias = root / "alias.png"
    try:
        alias.symlink_to(other)
    except OSError:
        # Windows host lacks CreateSymbolicLink privilege; use a real relative
        # path escape to exercise the same resolved containment predicate.
        alias = root / ".." / "other.png"
    gateway = Gateway()
    with pytest.raises(ProfileRejectedError):
        await adapter(gateway, media_roots=(root,)).avatar(str(alias))
    assert gateway.calls == []


@pytest.mark.parametrize("contents", [b"not an image", b"\x89PNG\r\n\x1a\n"])
async def test_avatar_rejects_invalid_image_contents(tmp_path, contents):
    image = tmp_path / "avatar.png"
    image.write_bytes(contents)
    gateway = Gateway()
    with pytest.raises(ValueError):
        await adapter(gateway, media_roots=(tmp_path,)).avatar(str(image))
    assert gateway.calls == []


async def test_avatar_rejects_truncated_jpeg(tmp_path):
    image = tmp_path / "avatar.jpg"
    Image.new("RGB", (100, 100), "pink").save(image)
    image.write_bytes(image.read_bytes()[:-10])
    gateway = Gateway()
    with pytest.raises(ValueError):
        await adapter(gateway, media_roots=(tmp_path,)).avatar(str(image))
    assert gateway.calls == []


async def test_avatar_rejects_large_file_before_decoding(tmp_path):
    image = tmp_path / "avatar.png"
    image.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    gateway = Gateway()
    with pytest.raises(ValueError):
        await adapter(gateway, media_roots=(tmp_path,)).avatar(str(image))
    assert gateway.calls == []


async def test_avatar_rejects_pixel_bomb(tmp_path):
    image = tmp_path / "avatar.png"
    Image.new("1", (4096, 4096)).save(image)
    gateway = Gateway()
    with pytest.raises(ValueError):
        await adapter(gateway, media_roots=(tmp_path,)).avatar(str(image))
    assert gateway.calls == []


async def test_final_guard_runs_inside_submission_and_can_revoke():
    gateway = Gateway()

    def revoke():
        raise ProfileRejectedError("revoked")

    with pytest.raises(ProfileRejectedError):
        await adapter(gateway).nickname("new", before_dispatch=revoke)
    assert gateway.calls == []


async def test_final_guard_and_dispatch_have_no_scheduling_boundary():
    events = []
    gateway = Gateway()
    original = gateway.call_api

    async def call(action, params):
        events.append("call")
        return await original(action, params)

    gateway.call_api = call

    def guarded():
        events.append("guard")
        asyncio.get_running_loop().call_soon(events.append, "queued")

    await adapter(gateway).nickname("new", before_dispatch=guarded)
    assert events[:2] == ["guard", "call"]


async def test_disconnected_gateway_is_definite_failure_without_dispatch():
    gateway = Gateway()
    gateway.websocket = None
    with pytest.raises(ProfileUnavailableError):
        await adapter(gateway).nickname("new")
    assert gateway.calls == []


async def test_connection_is_rechecked_at_coroutine_api_entry():
    gateway = Gateway()
    target = adapter(gateway)
    task = asyncio.create_task(target.nickname("new"))
    gateway.websocket = None
    with pytest.raises(ProfileUnavailableError):
        await task
    assert gateway.calls == []


@pytest.mark.parametrize("response", [TimeoutError("secret URL"), ConnectionError("secret"), asyncio.CancelledError()])
async def test_gateway_failure_after_submission_is_unknown_and_never_retried(response):
    gateway = Gateway(response)
    with pytest.raises(ProfileResultUncertainError) as caught:
        await adapter(gateway).nickname("new")
    assert isinstance(caught.value, QQMessageDeliveryUncertainError)
    assert "secret" not in str(caught.value)
    assert len(gateway.calls) == 1


async def test_timeout_after_submission_is_unknown():
    async def stalled():
        await asyncio.Future()

    gateway = Gateway(stalled)
    with pytest.raises(ProfileResultUncertainError):
        await adapter(gateway, timeout_seconds=0.01).nickname("new")
    assert len(gateway.calls) == 1


async def test_cancellation_after_submission_is_unknown():
    async def stalled():
        await asyncio.Future()

    gateway = Gateway(stalled)
    task = asyncio.create_task(adapter(gateway).nickname("new"))
    await gateway.started.wait()
    task.cancel()
    with pytest.raises(ProfileResultUncertainError):
        await task
    assert len(gateway.calls) == 1


async def test_cancellation_before_submission_preserves_cancelled_state():
    gateway = Gateway()
    task = asyncio.create_task(adapter(gateway).nickname("new"))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gateway.calls == []


@pytest.mark.parametrize("retcode", [1400, 1404])
async def test_known_pre_write_bridge_rejections_are_definite(retcode):
    gateway = Gateway({"status": "failed", "retcode": retcode, "data": None})
    with pytest.raises(ProfileRejectedError):
        await adapter(gateway).nickname("new")
    assert len(gateway.calls) == 1


@pytest.mark.parametrize("retcode", [100, 1200, 500, -1])
async def test_generic_bridge_failure_does_not_prove_no_write(retcode):
    gateway = Gateway({"status": "failed", "retcode": retcode, "data": None})
    with pytest.raises(ProfileResultUncertainError):
        await adapter(gateway).nickname("new")
    assert len(gateway.calls) == 1


@pytest.mark.parametrize("response", [
    None, {}, {"status": "ok", "retcode": False, "data": None},
    {"status": "ok", "retcode": 0},
    {"status": "ok", "retcode": 0, "data": {}},
    {"status": "ok", "retcode": 100, "data": None},
    {"status": "failed", "retcode": 0, "data": None},
    {"status": "failed", "retcode": 1400},
    {"status": "failed", "retcode": 1404, "data": {}},
])
async def test_malformed_acknowledgement_is_unknown(response):
    gateway = Gateway(response)
    with pytest.raises(ProfileResultUncertainError):
        await adapter(gateway).nickname("new")
    assert len(gateway.calls) == 1


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "900000101", 2**63])
def test_account_and_scope_ids_are_strict(value):
    with pytest.raises(ValueError):
        AccountProfileAdapter(Gateway(), account_id=value)
    with pytest.raises(ValueError):
        AccountProfileAdapter(Gateway(), account_id=ACCOUNT, allowed_group_ids=(value,))


def test_group_ids_validate_every_element_before_duplicate_removal():
    with pytest.raises(ValueError):
        AccountProfileAdapter(Gateway(), account_id=ACCOUNT, allowed_group_ids=(1, True))


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_timeout_is_finite_positive(value):
    with pytest.raises(ValueError):
        adapter(Gateway(), timeout_seconds=value)


async def test_non_callable_callback_is_rejected_without_dispatch():
    gateway = Gateway()
    with pytest.raises(ValueError):
        await adapter(gateway).nickname("new", before_dispatch="invalid")
    assert gateway.calls == []
