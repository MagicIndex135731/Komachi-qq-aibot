"""Current QQ friendship, rather than a copied/stale account allowlist."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.adapters.qzone import QzoneAdapter, QzoneProtocolError, QzoneRejectedError
from app.agent.public_tools import build_public_tools
from app.agent.social_contracts import SocialDecision, SocialObservation
from app.agent.social_executor import SocialExecutor
from app.agent.social_policy import configured_social_jobs
from tests.adapters.test_qzone import Gateway, OWNER, OTHER, ok, post, row
from tests.agent.test_social_policy import runtime
from tests.agent.test_social_executor import Guard


def setup(engine, adapter):
    config = runtime({"enabled": True, "qzone": {"comment": {
        "enabled": True, "all_friends": True, "quiet_hours": None,
        "interval_seconds": 7200, "cooldown_seconds": 21600, "max_daily": 2}}})
    config.settings.bot_qq = 900000501
    job = configured_social_jobs(config)[0]
    assert job.all_friends and not job.friend_ids
    router = SimpleNamespace(runtime=config, engine=engine)
    executor = SimpleNamespace(router=router, qzone=adapter, clock=lambda: post().published_at)
    executor.legacy = SocialExecutor(router=router, sender=None, qzone_adapter=adapter, clock=executor.clock)
    observation = SocialObservation(job, executor.clock(), "synthetic public persona")
    return executor, observation


@pytest.mark.asyncio
async def test_friend_list_accepts_more_than_static_limit_and_deduplicates():
    ids = tuple(range(900000201, 900000402))
    adapter = QzoneAdapter(Gateway([ok([{"user_id": i} for i in (*ids, ids[0])])]))
    assert await adapter.get_friend_ids() == ids


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [{}, [True], [{"user_id": True}], [{"user_id": -1}],
                                   [{"user_id": "900000201"}], [{"user_id": 2**63}]])
async def test_malformed_friend_list_never_authorizes(data):
    with pytest.raises(QzoneProtocolError):
        await QzoneAdapter(Gateway([ok(data)])).get_friend_ids()


@pytest.mark.asyncio
async def test_friend_deleted_during_post_verification_prevents_write():
    gateway = Gateway([ok([{"user_id": OWNER}]), ok({"total": 1, "msglist": [row()]}), ok([])])
    guard = Guard()
    with pytest.raises(QzoneRejectedError):
        await QzoneAdapter(gateway).comment(post(), "synthetic comment", require_friend=True,
                                           before_dispatch=guard.on_dispatch)
    assert [action for action, _ in gateway.calls] == ["get_friend_list", "get_qzone_msg_list", "get_friend_list"]
    assert not guard.dispatched


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 5, 20])
async def test_all_friends_tools_discover_feed_without_scheduler_selected_target(sqlite_engine, count):
    feed = {"uin": OWNER, "nickname": "synthetic", "time": int(post().published_at.timestamp()),
            "appid": 311, "key": post().post_id, "html": ""}
    friends = ok([{"user_id": OWNER}])
    gateway = Gateway([friends, ok({"feeds": [feed] * (count + 1), "has_more": False}), friends,
                       ok({"total": 1, "msglist": [row()]}), friends, friends])
    executor, observation = setup(sqlite_engine, QzoneAdapter(gateway))
    tools = {tool.name: tool for tool in build_public_tools(executor, observation, Guard(), object())}
    result = await tools["read_friend_feed"].handler({"count": count})
    assert result.status == "ok"
    assert result.data["posts"][0]["owner_id"] == OWNER
    assert result.data["posts"][0]["post_id"] == post().post_id


@pytest.mark.asyncio
@pytest.mark.parametrize("friend_response", [ok([]), RuntimeError("synthetic read failure"),
    {"status": "failed", "retcode": 100, "data": None}])
async def test_nonfriend_or_failed_lookup_cannot_read_posts(sqlite_engine, friend_response):
    gateway = Gateway([friend_response])
    executor, observation = setup(sqlite_engine, QzoneAdapter(gateway))
    tools = {tool.name: tool for tool in build_public_tools(executor, observation, Guard(), object())}
    result = await tools["read_friend_posts"].handler({"owner_id": OTHER})
    assert result.status == "blocked"
    assert [action for action, _ in gateway.calls] == ["get_friend_list"]


@pytest.mark.asyncio
@pytest.mark.parametrize("removed", [False, True])
async def test_native_agent_discovers_current_friend_and_rechecks_before_comment(sqlite_engine, tmp_path, removed):
    from tests.agent.test_autonomous_social import composed, wake, writes, FRIEND
    from tests.agent.test_mature_composition import model_steps, latest
    current_friends = [FRIEND]
    def choose(body):
        found = latest(body, "read_friend_feed")["data"]["posts"]
        assert found[0]["owner_id"] == FRIEND
        if removed:
            current_friends.clear()
        return "comment_qzone", {"owner_id": FRIEND, "post_id": found[0]["post_id"],
                                  "text": "synthetic comment", "artifact_ids": []}
    client, requests, _ = model_steps(("read_friend_feed", {"count": 20}), choose, None)
    async with composed(sqlite_engine, tmp_path, client, runtime="harness", kind="qzone_comment") as fixture:
        permission = fixture.router.runtime.social_policy["qzone"]["comment"]
        permission.pop("friend_ids")
        permission["all_friends"] = True
        original = fixture.gateway.call_api
        async def api(action, params):
            if action == "get_friend_list":
                fixture.gateway.calls.append((action, params))
                return ok([{"user_id": i} for i in current_friends])
            return await original(action, params)
        fixture.gateway.call_api = api
        await wake(fixture)
        assert [action for action, _ in writes(fixture)] == ([] if removed else ["comment_qzone"])
        if removed:
            assert latest(requests[-1], "comment_qzone")["status"] == "blocked"


@pytest.mark.parametrize("kind,options", [
    ("comment", {"all_friends": "true"}),
    ("comment", {"all_friends": True, "friend_ids": [OWNER]}),
    ("publish", {"all_friends": True, "topics": ["synthetic"]}),
])
def test_dynamic_friend_permission_cannot_be_ambiguous_or_grant_other_surface(kind, options):
    config = runtime({"enabled": True, "qzone": {kind: {"enabled": True, **options}}})
    with pytest.raises(ValueError):
        configured_social_jobs(config)


@pytest.mark.asyncio
async def test_deleted_friend_cannot_fetch_observed_pixels(sqlite_engine):
    gateway = Gateway([ok([])])
    executor, observation = setup(sqlite_engine, QzoneAdapter(gateway))
    observation = replace(observation, posts=(post(),))
    tools = {tool.name: tool for tool in build_public_tools(executor, observation, Guard(), object())}
    result = await tools["observe_post_images"].handler({"owner_id": OWNER, "post_id": post().post_id})
    assert result.status == "blocked"
    assert not result.images


@pytest.mark.asyncio
async def test_deleted_friend_blocks_legacy_executor_before_post_read(sqlite_engine):
    gateway = Gateway([ok([])])
    executor, observation = setup(sqlite_engine, QzoneAdapter(gateway))
    observation = replace(observation, posts=(post(),), authorized_friend_ids=(OWNER,))
    guard = Guard()
    result = await executor.legacy.execute(observation,
        SocialDecision("comment_qzone", "synthetic", post().post_id, OWNER), guard)
    assert result.status == "blocked" and not guard.dispatched
    assert [action for action, _ in gateway.calls] == ["get_friend_list"]
