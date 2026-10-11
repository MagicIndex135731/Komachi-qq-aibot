"""Owner directives can select old posts without widening autonomous authority."""
from datetime import timedelta

import pytest

from app.agent.public_tools import build_public_tools
from app.adapters.qzone import QzoneAdapter
from app.agent.social_contracts import SocialDecision, SocialObservation
from app.agent.tool_dispatch import ToolDispatchBoundary
from app.storage.agent_effects import AgentToolEffectRepository
from tests.adapters.test_qzone import Gateway, OWNER, ok, post, row
from tests.agent.test_current_friend_permissions import setup
from tests.agent.test_social_executor import Guard


@pytest.mark.parametrize("manual", [None, False, True])
@pytest.mark.parametrize("age", [86400, 86401, 315360000, -60])
async def test_post_read_age_depends_on_live_owner_authority(sqlite_engine, manual, age):
    friends = ok([{"user_id": OWNER}])
    executor, observation = setup(sqlite_engine, QzoneAdapter(
        Gateway([friends, ok({"total": 1, "msglist": [row(
            time=int(post().published_at.timestamp()) - age)]}), friends])))
    guard = Guard()
    if manual is not None:
        guard.owner_authority = lambda: manual
    tools = {tool.name: tool for tool in build_public_tools(executor, observation, guard, object())}
    outcome = await tools["read_friend_posts"].handler({"owner_id": OWNER})
    expected = age >= 0 and (manual is True or age <= observation.job.max_post_age_seconds)
    assert bool(outcome.data["posts"]) is expected


@pytest.mark.parametrize("manual", [None, False, True])
@pytest.mark.parametrize("age", [86400, 86401, -60])
async def test_final_comment_recheck_keeps_autonomous_age_limit(sqlite_engine, manual, age):
    observed = post()
    observed = type(observed)(observed.owner_id, observed.post_id, observed.text,
                              observed.published_at - timedelta(seconds=age), observed.images)
    friends = ok([{"user_id": OWNER}])
    gateway = Gateway([friends, ok({"total": 1, "msglist": [row(
        time=int(observed.published_at.timestamp()))]}), friends,
        ok({"total": 1, "msglist": [row(time=int(observed.published_at.timestamp()))]}),
        friends, ok({"comment_id": "synthetic-comment-receipt"})])
    executor, observation = setup(sqlite_engine, QzoneAdapter(gateway))
    observation = SocialObservation(observation.job, observation.now, observation.persona,
                                    posts=(observed,), authorized_friend_ids=(OWNER,))
    guard = Guard()
    if manual is not None:
        guard.owner_authority = lambda: manual
    result = await executor.legacy.execute(observation,
        SocialDecision("comment_qzone", "synthetic comment", observed.post_id, OWNER), guard)
    expected = age >= 0 and (manual is True or age <= observation.job.max_post_age_seconds)
    assert result.status == ("sent" if expected else "blocked")
    assert [name for name, _ in gateway.calls].count("comment_qzone") == int(expected)


async def test_revoked_owner_cannot_dispatch_previously_read_old_post(sqlite_engine):
    friends = ok([{"user_id": OWNER}])
    old = row(time=int(post().published_at.timestamp()) - 172800)
    gateway = Gateway([friends, ok({"total": 1, "msglist": [old]}), friends])
    executor, observation = setup(sqlite_engine, QzoneAdapter(gateway))
    live = [True]
    guard = Guard()
    guard.owner_authority = lambda: live[0]
    boundary = ToolDispatchBoundary(engine=sqlite_engine, account_id=executor.router.runtime.settings.bot_qq,
        scope="owner:public:synthetic", guard=guard, clock=executor.clock, owner_authority=guard.owner_authority)
    tools = {tool.name: tool for tool in build_public_tools(executor, observation, boundary, object())}
    assert (await boundary.invoke(tools["read_friend_posts"], {"owner_id": OWNER}, "read")).data["posts"]
    live[0] = False
    with pytest.raises(RuntimeError, match="authority revoked"):
        await boundary.invoke(tools["comment_qzone"], {"owner_id": OWNER, "post_id": old["tid"],
            "text": "synthetic comment", "artifact_ids": []}, "write")
    assert "comment_qzone" not in [name for name, _ in gateway.calls]


@pytest.mark.parametrize("private_origin", [False, True])
async def test_native_owner_old_post_comment_returns_real_receipt_without_autonomous_cost(
        sqlite_engine, tmp_path, private_origin):
    from app.agent.owner_actions import OwnerActionService, OwnerActionWorker, OwnerNoticeWorker
    from tests.agent.test_owner_actions import source
    from tests.agent.test_autonomous_social import composed, FRIEND, BOT, GROUP, NOW, writes
    from tests.agent.test_mature_composition import model_steps, latest
    def choose(body):
        found = latest(body, "read_friend_posts")["data"]["posts"]
        assert len(found) == 1 and found[0]["owner_id"] == FRIEND
        return "comment_qzone", {"owner_id": FRIEND, "post_id": found[0]["post_id"],
                                  "text": "合成主人旧帖评论", "artifact_ids": []}
    client, requests, _ = model_steps(("read_friend_posts", {"owner_id": FRIEND}), choose,
        ("report_owner_result", {"text": "已获得真实评论回执", "artifact_ids": []}), None)
    async with composed(sqlite_engine, tmp_path, client, kind="qzone_comment", runtime="harness",
            options={"max_daily": 1, "max_daily_per_friend": 1, "quiet_hours": ["00:00", "23:59"]}) as fixture:
        await fixture.agent.stop()
        fixture.gateway.posts[0]["time"] = int(NOW.timestamp()) - 172800
        service = OwnerActionService(sqlite_engine, runtime=fixture.router.runtime, clock=lambda: NOW)
        source_id = source(sqlite_engine, author=fixture.router.runtime.settings.owner_qq,
                           group=None if private_origin else GROUP, text="评论合成好友最新空间")
        service.enqueue(source_id, target="public")
        result = await OwnerActionWorker(service, fixture.executor, connected=lambda: True).run_once()
        assert result["status"] == "sent"
        assert latest(requests[-1], "comment_qzone")["status"] == "sent"
        assert [name for name, _ in writes(fixture)] == ["comment_qzone"]
        ledger = AgentToolEffectRepository(sqlite_engine, account_id=BOT, clock=lambda: NOW)
        assert ledger.count_dispatched(scope=f"public:qzone_comment:friend:{FRIEND}") == 0
        assert ledger.count_dispatched(scope=f"owner:public:qzone_comment:friend:{FRIEND}") == 1
        assert "PRIVATE_PERSONA_FACT" not in str(requests[0])
        class PrivateSender:
            async def send_private_text(self, message):
                return ("synthetic-private-notice",)
        sender = PrivateSender() if private_origin else fixture.executor.sender
        notice = OwnerNoticeWorker(service, sender, surface="private" if private_origin else "group",
                                   connected=lambda: True)
        assert (await notice.run_once()).status == "sent"
        assert await notice.run_once() is None
