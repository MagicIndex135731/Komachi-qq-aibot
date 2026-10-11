import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.agent.social_contracts import SocialDecision, SocialJob, SocialObservation, SocialPost, SocialResult, SocialRuntimeOptions
from app.agent.social_runtime import SocialRuntime
from app.storage.db import session_scope
from app.storage.social_agent import SocialActionRecord


ACCOUNT = 900000999
NOW = datetime(2026, 10, 9, 6, tzinfo=UTC)


def job(kind="qzone_publish", **kwargs):
    return replace(SocialJob(kind, kind, ACCOUNT, interval_seconds=10, cooldown_seconds=1,
                             max_daily=3, quiet_hours=None, friend_ids=(900000101,)), **kwargs)


def runtime(engine, jobs, observe, decide, execute, policy_check=lambda *_: True, **kwargs):
    return SocialRuntime(engine=engine, account_id=ACCOUNT, jobs=lambda: tuple(jobs), observe=observe,
                         decide=decide, execute=execute, policy_check=policy_check, clock=lambda: NOW,
                         options=SocialRuntimeOptions(poll_seconds=999, observation_timeout_seconds=1,
                                                      decision_timeout_seconds=1, execution_timeout_seconds=1,
                                                      lease_seconds=10, shutdown_timeout_seconds=0.02), **kwargs)


async def observation(item, now):
    return SocialObservation(item, now, "synthetic persona")


async def publishing(_):
    return SocialDecision("publish_qzone", "a public thought")


@pytest.mark.asyncio
async def test_idle_account_wakes_persistently_without_any_inbound_event(sqlite_engine):
    sent = []

    async def execute(obs, decision, guard):
        guard.on_dispatch()
        sent.append((obs.job.key, decision.text))
        return SocialResult("sent", "real-receipt")

    jobs = [job()]
    agent = runtime(sqlite_engine, jobs, observation, publishing, execute)
    await agent.start()
    agent.set_connected(True)
    try:
        await agent.tick(NOW)
        assert sent == []
        await agent.tick(NOW + timedelta(seconds=10))
        assert len(sent) == 1
        await agent.tick(NOW + timedelta(seconds=10))
        assert len(sent) == 1
    finally:
        await agent.stop()


@pytest.mark.asyncio
async def test_friend_post_is_commented_once_across_repeated_polls(sqlite_engine):
    sent = []

    async def observe(item, now):
        return SocialObservation(item, now, "persona", posts=(SocialPost(900000101, "real-tid", "public post", NOW),))

    async def decide(_):
        return SocialDecision("comment_qzone", "a friendly comment", "real-tid", 900000101)

    async def execute(obs, decision, guard):
        guard.on_dispatch()
        sent.append(decision.post_id)
        return SocialResult("failed", reason="explicit rejection")

    agent = runtime(sqlite_engine, [job("qzone_comment")], observe, decide, execute)
    await agent.start()
    agent.set_connected(True)
    try:
        await agent.tick(NOW)
        for seconds in (10, 20, 30):
            await agent.tick(NOW + timedelta(seconds=seconds))
        assert sent == ["real-tid"]
    finally:
        await agent.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [SocialDecision("comment_qzone", "fake", "invented", 900000101),
                                       SocialDecision("comment_qzone", "wrong author", "real-tid", 900000102),
                                       SocialDecision("publish_qzone", "wrong surface"),
                                       SocialDecision("comment_qzone", "chat export", "real-tid", 900000101, (123,))])
async def test_fabricated_or_cross_surface_decisions_never_dispatch(sqlite_engine, decision):
    async def observe(item, now):
        return SocialObservation(item, now, "persona", posts=(SocialPost(900000101, "real-tid", "public", NOW),))

    async def decide(_):
        return decision

    async def forbidden(*_):
        pytest.fail("invalid decision reached the external write callback")

    agent = runtime(sqlite_engine, [job("qzone_comment")], observe, decide, forbidden)
    await agent.start()
    agent.set_connected(True)
    try:
        await agent.tick(NOW)
        await agent.tick(NOW + timedelta(seconds=10))
        with session_scope(sqlite_engine) as session:
            assert list(session.scalars(select(SocialActionRecord))) == []
    finally:
        await agent.stop()


@pytest.mark.asyncio
async def test_policy_revocation_at_dispatch_and_quiet_hours_prevent_external_work(sqlite_engine):
    allowed = [True]
    observed, dispatched = [], []

    async def observe(item, now):
        observed.append(now)
        return await observation(item, now)

    async def execute(obs, decision, guard):
        allowed[0] = False
        guard.on_dispatch()
        dispatched.append(True)
        return SocialResult("sent", "receipt")

    agent = runtime(sqlite_engine, [job(quiet_hours=("23:00", "08:00"))], observe, publishing, execute,
                    policy_check=lambda *_: allowed[0])
    await agent.start()
    agent.set_connected(True)
    try:
        await agent.tick(NOW.replace(hour=16))
        await agent.tick(NOW.replace(hour=16) + timedelta(seconds=10))
        assert observed == []
    finally:
        await agent.stop()
    agent = runtime(sqlite_engine, [job()], observe, publishing, execute, policy_check=lambda *_: allowed[0])
    await agent.start()
    agent.set_connected(True)
    try:
        await agent.tick(NOW + timedelta(days=1))
        assert len(observed) == 1
        assert dispatched == []
    finally:
        await agent.stop()


@pytest.mark.asyncio
async def test_unknown_dispatch_is_not_replayed_after_restart(sqlite_engine):
    calls = []

    async def execute(obs, decision, guard):
        guard.on_dispatch()
        calls.append(decision.text)
        raise TimeoutError("synthetic ambiguous timeout")

    agent = runtime(sqlite_engine, [job(max_daily=1)], observation, publishing, execute)
    await agent.start()
    agent.set_connected(True)
    await agent.tick(NOW)
    await agent.tick(NOW + timedelta(seconds=10))
    await agent.stop()
    successor = runtime(sqlite_engine, [job(max_daily=1)], observation, publishing, execute)
    await successor.start()
    successor.set_connected(True)
    try:
        await successor.tick(NOW + timedelta(seconds=20))
        assert len(calls) == 1
        with session_scope(sqlite_engine) as session:
            assert list(session.scalars(select(SocialActionRecord.status))) == ["uncertain"]
    finally:
        await successor.stop()


@pytest.mark.asyncio
async def test_provider_failure_waits_for_next_due_instead_of_retrying_each_tick(sqlite_engine):
    calls = []

    async def decide(_):
        calls.append(True)
        raise RuntimeError("synthetic provider failure")

    agent = runtime(sqlite_engine, [job()], observation, decide, publishing)
    await agent.start()
    agent.set_connected(True)
    try:
        await agent.tick(NOW)
        for seconds in (10, 11, 12, 19):
            await agent.tick(NOW + timedelta(seconds=seconds))
        assert len(calls) == 1
        await agent.tick(NOW + timedelta(seconds=20))
        assert len(calls) == 2
    finally:
        await agent.stop()


@pytest.mark.asyncio
async def test_disconnect_and_stop_fence_late_model_results(sqlite_engine):
    entered, release = asyncio.Event(), asyncio.Event()
    sent = []

    async def decide(_):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return SocialDecision("publish_qzone", "late model result")

    async def execute(obs, decision, guard):
        guard.on_dispatch()
        sent.append(decision.text)
        return SocialResult("sent", "late-receipt")

    agent = runtime(sqlite_engine, [job()], observation, decide, execute)
    await agent.start()
    agent.set_connected(True)
    await agent.tick(NOW)
    pending = asyncio.create_task(agent.tick(NOW + timedelta(seconds=10)))
    await entered.wait()
    agent.set_connected(False)
    await agent.stop()
    release.set()
    await asyncio.wait_for(asyncio.gather(pending, return_exceptions=True), 1)
    assert sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,expected", [("qzone_comment", "sent"), ("qzone_publish", "uncertain"), ("group", "uncertain")])
async def test_missing_receipt_only_allows_explicit_comment_success(sqlite_engine, kind, expected):
    async def observe(item, now):
        return SocialObservation(item, now, "persona", posts=(SocialPost(900000101, "comment-tid", "post", NOW),))

    async def decide(_):
        return SocialDecision({"group": "send_group", "qzone_publish": "publish_qzone", "qzone_comment": "comment_qzone"}[kind],
                              "text", "comment-tid" if kind == "qzone_comment" else "",
                              900000101 if kind == "qzone_comment" else 0)

    async def execute(obs, decision, guard):
        guard.on_dispatch()
        return SocialResult("sent")

    agent = runtime(sqlite_engine, [job(kind)], observe, decide, execute)
    await agent.start()
    agent.set_connected(True)
    try:
        await agent.tick(NOW)
        await agent.tick(NOW + timedelta(seconds=10))
        with session_scope(sqlite_engine) as session:
            assert session.scalar(select(SocialActionRecord.status)) == expected
    finally:
        await agent.stop()


@pytest.mark.asyncio
async def test_two_runtimes_cannot_execute_for_the_same_account(sqlite_engine):
    sent = []

    async def execute(obs, decision, guard):
        guard.on_dispatch()
        sent.append(guard.action_id)
        return SocialResult("sent", "receipt")

    first = runtime(sqlite_engine, [job()], observation, publishing, execute)
    second = runtime(sqlite_engine, [job()], observation, publishing, execute)
    await first.start()
    await second.start()
    first.set_connected(True)
    second.set_connected(True)
    try:
        await first.tick(NOW)
        await second.tick(NOW)
        await first.tick(NOW + timedelta(seconds=10))
        await second.tick(NOW + timedelta(seconds=10))
        assert len(sent) == 1
    finally:
        await first.stop()
        await second.stop()


@pytest.mark.asyncio
async def test_stop_after_dispatch_keeps_uncertain_despite_late_success(sqlite_engine):
    entered, release = asyncio.Event(), asyncio.Event()

    async def execute(obs, decision, guard):
        guard.on_dispatch()
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return SocialResult("sent", "late-receipt")

    agent = runtime(sqlite_engine, [job()], observation, publishing, execute)
    await agent.start()
    agent.set_connected(True)
    await agent.tick(NOW)
    pending = asyncio.create_task(agent.tick(NOW + timedelta(seconds=10)))
    await entered.wait()
    await agent.stop()
    with session_scope(sqlite_engine) as session:
        assert session.scalar(select(SocialActionRecord.status)) == "uncertain"
    release.set()
    await asyncio.wait_for(asyncio.gather(pending, return_exceptions=True), 1)
    with session_scope(sqlite_engine) as session:
        assert session.scalar(select(SocialActionRecord.status)) == "uncertain"


@pytest.mark.asyncio
async def test_real_group_executor_uses_durable_guard_and_a_real_qq_receipt(sqlite_engine):
    from app.adapters.sender import Sender
    from app.agent.social_executor import SocialExecutor
    from app.agent.social_policy import configured_social_jobs
    from app.storage.models import Message
    from tests.agent.test_social_executor import group_setup

    class Gateway:
        calls = []

        async def call_api(self, action, params):
            self.calls.append((action, params))
            return {"status": "ok", "data": {"message_id": "actual-social-receipt"}}

    router, _ = group_setup(sqlite_engine)
    gateway = Gateway()
    sender = Sender(gateway)
    executor = SocialExecutor(router=router, sender=sender, qzone_adapter=None, clock=lambda: NOW)

    async def decide(_):
        return SocialDecision("send_group", "秋天到了，适合出门散步。")

    jobs = configured_social_jobs(router.runtime)
    agent = SocialRuntime(engine=sqlite_engine, account_id=router.runtime.settings.bot_qq,
                          jobs=lambda: jobs, observe=observation, decide=decide, execute=executor.execute,
                          policy_check=executor.policy_check, clock=lambda: NOW,
                          options=SocialRuntimeOptions(poll_seconds=999))
    await agent.start()
    agent.set_connected(True)
    try:
        await agent.tick(NOW)
        await agent.tick(NOW + timedelta(seconds=jobs[0].interval_seconds))
        assert len(gateway.calls) == 1
        assert gateway.calls[0][1]["group_id"] == jobs[0].scope_id
        with session_scope(sqlite_engine) as session:
            action = session.scalar(select(SocialActionRecord))
            assert action.status == "sent"
            assert action.receipt_id == "actual-social-receipt"
            rows = list(session.scalars(select(Message)))
            assert len(rows) == 1
            assert rows[0].user_id == router.runtime.settings.bot_qq
            assert rows[0].raw_json["social_action_id"] == action.action_id
            assert rows[0].raw_json["delivery_state"] == "sent"
    finally:
        await agent.stop()
