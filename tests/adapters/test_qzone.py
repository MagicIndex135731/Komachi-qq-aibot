from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from app.adapters.qzone import (
    QzoneAdapter,
    QzonePost,
    QzoneProtocolError,
    QzoneRejectedError,
    QzoneUnavailableError,
)
from app.adapters.sender import QQMessageDeliveryUncertainError
from app.agent.social_contracts import SocialPost
from app.agent.social_runtime import SocialStoppedError


OWNER = 900000101
OTHER = 900000102
STAMP = 1791500000
TID = "synthetic-tid-1"


def ok(data):
    return {"status": "ok", "retcode": 0, "data": data}


def row(tid=TID, **changes):
    return {"tid": tid, "content": "合成动态", "time": STAMP,
            "comment_num": 0, "is_private": False, "images": [], **changes}


def post(tid=TID, owner=OWNER):
    return QzonePost(owner, tid, "合成动态", datetime.fromtimestamp(STAMP, timezone.utc))


class Gateway:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def call_api(self, action, params):
        self.calls.append((action, params))
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


@pytest.mark.asyncio
async def test_posts_are_immutable_scoped_observations_with_inert_html_and_images():
    gateway = Gateway([ok({"total": 1, "msglist": [row(
        content='<p>春日 &amp; 测试</p><script>execute_secret()</script>',
        images=["https://img.example.test/one.png", "https://img.example.test/two.gif"],
    )]})])
    posts = await QzoneAdapter(gateway).get_posts(OWNER, count=2)

    assert QzonePost is SocialPost
    assert posts == (QzonePost(OWNER, TID, "春日 & 测试", datetime.fromtimestamp(STAMP, timezone.utc),
                               ("https://img.example.test/one.png", "https://img.example.test/two.gif")),)
    assert gateway.calls == [("get_qzone_msg_list", {"target_uin": OWNER, "pos": 0, "num": 2})]
    with pytest.raises(AttributeError):
        posts[0].post_id = "changed"


@pytest.mark.asyncio
async def test_feed_handle_is_resolved_to_real_tid_and_rechecked_before_comment():
    feed = {"uin": OWNER, "nickname": "合成作者", "time": STAMP, "appid": 311,
            "key": "feed-handle-is-not-tid", "html":
            f'<li data-uin="{OWNER}" data-tid="{TID}"><p>合成动态</p></li>'}
    gateway = Gateway([
        ok({"feeds": [feed], "has_more": False}),
        ok({"total": 1, "msglist": [row()]}),
        ok({"total": 1, "msglist": [row()]}),
        ok({"comment_id": "synthetic-comment-1"}),
    ])
    adapter = QzoneAdapter(gateway)
    posts = await adapter.get_feeds()
    assert posts == (post(),)
    assert await adapter.comment(posts[0], "合成评论") == "synthetic-comment-1"
    assert gateway.calls[-1] == ("comment_qzone", {"target_uin": OWNER, "tid": TID, "content": "合成评论"})
    assert len(gateway.calls) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("html,key", [
    ("<p>没有可靠标识</p>", "guessed-tid"),
    (f'<li data-uin="{OTHER}" data-tid="{TID}"></li>', TID),
    (f'<a href="https://evil.example.test/{OWNER}/mood/{TID}">动态</a>', "guessed-tid"),
    (f'<li data-uin="{OWNER}" data-tid="{TID}"></li><li data-uin="{OWNER}" data-tid="other-tid"></li>', TID),
])
async def test_unresolved_or_conflicting_feed_identity_never_becomes_comment_target(html, key):
    gateway = Gateway([ok({"feeds": [{"uin": OWNER, "nickname": "合成", "time": STAMP,
                                       "appid": 311, "key": key, "html": html}], "has_more": False}),
                       ok({"total": 1, "msglist": [row()]})])
    assert await QzoneAdapter(gateway).get_feeds() == ()
    assert all(action != "comment_qzone" for action, _ in gateway.calls)


@pytest.mark.asyncio
async def test_feed_key_requires_exact_match_in_actual_owner_post_list_and_deduplicates():
    feed = {"uin": OWNER, "nickname": "合成", "time": STAMP, "appid": 311, "key": TID, "html": ""}
    gateway = Gateway([ok({"feeds": [feed, feed], "has_more": True}),
                       ok({"total": 1, "msglist": [row()]})])
    assert await QzoneAdapter(gateway).get_feeds() == (post(),)
    assert len(gateway.calls) == 2


@pytest.mark.asyncio
async def test_feed_system_row_without_author_does_not_block_verified_friend_post():
    # SnowLuma normalizes absent authors to zero on non-mood feed entries.
    system = {"uin": 0, "nickname": "", "time": STAMP, "appid": 4,
              "key": "synthetic-system-feed", "html": "<p>系统动态</p>"}
    friend = {"uin": OWNER, "nickname": "合成", "time": STAMP, "appid": 311,
              "key": TID, "html": ""}
    gateway = Gateway([ok({"feeds": [system, friend], "has_more": False}),
                       ok({"total": 1, "msglist": [row()]})])
    assert await QzoneAdapter(gateway).get_feeds(allowed_owner_ids=(OWNER,)) == (post(),)
    assert gateway.calls == [("get_qzone_feeds", {"page_num": 1, "count": 20}),
                             ("get_qzone_msg_list", {"target_uin": OWNER, "pos": 0, "num": 100})]


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [{"uin": 0}, {"appid": "311"}])
async def test_commentable_feed_still_requires_typed_app_and_positive_author(changes):
    feed = {"uin": OWNER, "nickname": "合成", "time": STAMP, "appid": 311,
            "key": TID, "html": "", **changes}
    gateway = Gateway([ok({"feeds": [feed], "has_more": False})])
    with pytest.raises(QzoneProtocolError):
        await QzoneAdapter(gateway).get_feeds(allowed_owner_ids=(OWNER,))
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
async def test_comment_does_not_write_when_post_disappeared_or_owner_was_fabricated():
    gateway = Gateway([ok({"total": 0, "msglist": []})])
    with pytest.raises(QzoneRejectedError):
        await QzoneAdapter(gateway).comment(post(owner=OTHER), "合成评论")
    assert gateway.calls == [("get_qzone_msg_list", {"target_uin": OTHER, "pos": 0, "num": 100})]


@pytest.mark.asyncio
async def test_comment_rejects_changed_body_even_if_owner_tid_and_time_are_identical():
    gateway = Gateway([ok({"total": 1, "msglist": [row(content="已经修改的合成动态")]})])
    with pytest.raises(QzoneRejectedError):
        await QzoneAdapter(gateway).comment(post(), "合成评论")
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("revocation", ["stop", "disconnect_reconnect", "policy_change"])
async def test_revocation_during_comment_preflight_blocks_write_before_durable_marker(revocation):
    generation = 1
    calls = []
    marked = []

    class RevokingGateway:
        async def call_api(self, action, params):
            nonlocal generation
            calls.append(action)
            # Reconnecting may restore a boolean connected flag, but cannot
            # restore the generation/policy under which this turn was made.
            generation += 2 if revocation == "disconnect_reconnect" else 1
            return ok({"total": 1, "msglist": [row()]})

    def before_dispatch():
        if generation != 1:
            raise SocialStoppedError("scheduled action revoked")
        marked.append(True)

    with pytest.raises(SocialStoppedError):
        await QzoneAdapter(RevokingGateway()).comment(post(), "合成评论", before_dispatch=before_dispatch)
    assert calls == ["get_qzone_msg_list"]
    assert marked == []


@pytest.mark.asyncio
async def test_publish_and_comment_preserve_multiple_approved_images_without_fetching(tmp_path):
    local = tmp_path / "synthetic.png"
    images = (str(local), "https://img.example.test/approved.png", "base64://c3ludGhldGlj")
    gateway = Gateway([ok({"tid": "synthetic-new-tid", "time": STAMP}),
                       ok({"total": 1, "msglist": [row()]}), ok({"comment_id": ""})])
    adapter = QzoneAdapter(gateway)

    assert await adapter.publish("合成发布", images=images) == "synthetic-new-tid"
    assert await adapter.comment(post(), "合成评论", images=images) is None
    expected_images = [local.resolve().as_uri(), images[1], images[2]]
    assert gateway.calls[0] == ("send_qzone_msg", {"content": "合成发布", "ugc_right": 4, "images": expected_images})
    assert gateway.calls[-1] == ("comment_qzone", {"target_uin": OWNER, "tid": TID,
                                                "content": "合成评论", "images": expected_images})
    assert not local.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [None, {}, {"status": "ok", "data": {"tid": TID}},
    {"status": [], "retcode": 0, "data": {"tid": TID}},
    {"status": "ok", "retcode": True, "data": {"tid": TID}},
    {"status": "ok", "retcode": "0", "data": {"tid": TID}},
    ok(None), ok({}), ok({"tid": ""}), ok({"tid": [TID]})])
async def test_malformed_publish_acknowledgement_is_uncertain_and_never_retried(response):
    gateway = Gateway([response])
    with pytest.raises(QQMessageDeliveryUncertainError):
        await QzoneAdapter(gateway).publish("sensitive-synthetic-text")
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
async def test_explicit_rejection_is_definite_and_omits_gateway_sensitive_message():
    gateway = Gateway([{"status": "failed", "retcode": 1400, "data": None,
                        "message": "sensitive-synthetic-text token=synthetic-secret"}])
    with pytest.raises(QzoneRejectedError) as error:
        await QzoneAdapter(gateway).publish("sensitive-synthetic-text")
    assert "1400" in str(error.value)
    assert "sensitive" not in str(error.value)
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wording", ["network down", "发表说说失败：响应缺少 tid", "timeout after write"])
async def test_generic_bridge_error_after_possible_write_is_uncertain_even_with_nonzero_code(wording):
    gateway = Gateway([{"status": "failed", "retcode": 100, "data": None, "wording": wording}])
    with pytest.raises(QQMessageDeliveryUncertainError):
        await QzoneAdapter(gateway).publish("合成发布")
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wording", ["qzone publish failed: code=-3 synthetic-denied",
                                    "qzone comment failed: subcode=5 synthetic-denied",
                                    "第 2 张图片上传失败: network down"])
async def test_known_business_rejection_or_prepublication_upload_failure_is_definite(wording):
    gateway = Gateway([{"status": "failed", "retcode": 100, "data": None, "wording": wording}])
    with pytest.raises(QzoneRejectedError):
        await QzoneAdapter(gateway).publish("合成发布")
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [asyncio.TimeoutError(), asyncio.CancelledError(),
                                    ConnectionError("synthetic-secret"), RuntimeError("synthetic-secret")])
async def test_post_submission_transport_failure_or_cancellation_is_uncertain_once(failure):
    gateway = Gateway([failure])
    with pytest.raises(QQMessageDeliveryUncertainError) as error:
        await QzoneAdapter(gateway).publish("合成发布")
    assert "synthetic-secret" not in str(error.value)
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
async def test_real_deadline_cancels_single_inflight_write_and_never_resubmits():
    class SlowGateway:
        calls = 0
        cancelled = False

        async def call_api(self, action, params):
            self.calls += 1
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True

    gateway = SlowGateway()
    with pytest.raises(QQMessageDeliveryUncertainError):
        await QzoneAdapter(gateway, timeout_seconds=0.01).publish("合成发布")
    assert gateway.calls == 1 and gateway.cancelled


@pytest.mark.asyncio
async def test_shutdown_cancellation_after_submission_retains_uncertain_outcome():
    entered = asyncio.Event()
    calls = []

    class WaitingGateway:
        async def call_api(self, action, params):
            calls.append(action)
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(QzoneAdapter(WaitingGateway()).publish("合成发布"))
    await entered.wait()
    task.cancel()
    with pytest.raises(QQMessageDeliveryUncertainError):
        await task
    assert calls == ["send_qzone_msg"]


@pytest.mark.asyncio
async def test_final_fence_and_gateway_entry_have_no_scheduling_gap():
    active = True
    entered = []

    def revoke():
        nonlocal active
        active = False

    def before_dispatch():
        assert active
        asyncio.get_running_loop().call_soon(revoke)

    class EntryGateway:
        async def call_api(self, action, params):
            entered.append(active)
            return ok({"tid": TID, "time": STAMP})

    assert await QzoneAdapter(EntryGateway()).publish("合成发布", before_dispatch=before_dispatch) == TID
    assert entered == [True]


@pytest.mark.asyncio
async def test_disconnected_gateway_fails_before_submission_and_read_cancellation_propagates():
    disconnected = Gateway([])
    disconnected.websocket = None
    with pytest.raises(QzoneUnavailableError):
        await QzoneAdapter(disconnected).publish("合成发布")
    assert disconnected.calls == []
    gateway = Gateway([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await QzoneAdapter(gateway).get_posts(OWNER)


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [None, {}, {"total": 1, "msglist": None},
    {"total": 1, "msglist": [row(time=True)]}, {"total": 1, "msglist": [row(tid="bad id")]},
    {"total": 1, "msglist": [row(images=["javascript:execute()"])]},
    {"total": 1, "msglist": [row(images="https://img.example.test/a")]},
    {"total": 1, "msglist": [row(content=["bad"])]}])
async def test_malformed_read_data_never_becomes_an_observation(data):
    with pytest.raises(QzoneProtocolError):
        await QzoneAdapter(Gateway([ok(data)])).get_posts(OWNER)


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"content": ""}, {"content": "x" * 2001},
    {"content": "synthetic", "visibility": 1}, {"content": "synthetic", "visibility": True},
    {"content": "synthetic", "images": ("javascript:execute()",)},
    {"content": "synthetic", "images": ("https://user:secret@img.example.test/a",)},
    {"content": "synthetic", "images": ("https://img.example.test/a",) * 10}])
async def test_invalid_write_parameters_fail_without_gateway_call(kwargs):
    gateway = Gateway([])
    with pytest.raises(ValueError):
        await QzoneAdapter(gateway).publish(**kwargs)
    assert gateway.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id,count", [(True, 20), (0, 20), (OWNER, True), (OWNER, 101)])
async def test_read_parameters_are_bounded_before_gateway_call(user_id, count):
    gateway = Gateway([])
    with pytest.raises(ValueError):
        await QzoneAdapter(gateway).get_posts(user_id, count)
    assert gateway.calls == []


@pytest.mark.asyncio
async def test_feed_author_reads_are_bounded_and_non_post_feeds_are_excluded():
    feeds = [{"uin": OWNER + index, "nickname": "合成", "time": STAMP, "appid": 311,
              "key": f"synthetic-tid-{index}", "html": ""} for index in range(6)]
    feeds.insert(0, {**feeds[0], "appid": 4})
    gateway = Gateway([ok({"feeds": feeds, "has_more": True})]
                      + [ok({"total": 1, "msglist": [row(f"synthetic-tid-{index}")]}) for index in range(5)])
    posts = await QzoneAdapter(gateway).get_feeds()
    assert len(posts) == 5
    assert len(gateway.calls) == 6
    assert len({post.owner_id for post in posts}) == 5


@pytest.mark.asyncio
async def test_allowed_author_is_selected_before_the_five_author_read_budget():
    feeds = [{"uin": OWNER + index, "nickname": "合成", "time": STAMP, "appid": 311,
              "key": f"synthetic-tid-{index}", "html": ""} for index in range(7)]
    allowed_owner = OWNER + 6
    gateway = Gateway([ok({"feeds": feeds, "has_more": False}),
                       ok({"total": 1, "msglist": [row("synthetic-tid-6")]})])
    posts = await QzoneAdapter(gateway).get_feeds(allowed_owner_ids=(allowed_owner,))
    assert posts == (post(tid="synthetic-tid-6", owner=allowed_owner),)
    assert gateway.calls == [("get_qzone_feeds", {"page_num": 1, "count": 20}),
                             ("get_qzone_msg_list", {"target_uin": allowed_owner, "pos": 0, "num": 100})]


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [(), (True,), (0,), tuple(range(1, 102)), "900000101"])
async def test_empty_or_invalid_feed_allowlist_never_reads_any_account(allowed):
    gateway = Gateway([])
    if allowed == ():
        assert await QzoneAdapter(gateway).get_feeds(allowed_owner_ids=allowed) == ()
    else:
        with pytest.raises(ValueError):
            await QzoneAdapter(gateway).get_feeds(allowed_owner_ids=allowed)
    assert gateway.calls == []


@pytest.mark.asyncio
async def test_like_uses_verified_post_timestamp_and_successful_empty_receipt():
    gateway = Gateway([ok({"total": 1, "msglist": [row()]}), ok(None)])
    assert await QzoneAdapter(gateway).like(post()) is None
    assert gateway.calls[-1] == ("like_qzone", {"target_uin": OWNER, "tid": TID, "abstime": STAMP})


@pytest.mark.asyncio
async def test_self_only_publish_is_explicit():
    gateway = Gateway([ok({"tid": TID, "time": STAMP})])
    assert await QzoneAdapter(gateway).publish("合成发布", visibility=64) == TID
    assert gateway.calls[0][1]["ugc_right"] == 64
