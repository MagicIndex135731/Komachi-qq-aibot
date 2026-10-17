"""Policy gates are durable across midnight, restart and competing consumers."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from app.agent.social_contracts import SocialJob, SocialPost
from app.storage.autonomous_agent import AutonomousRepository
from app.storage.agent_effects import AgentToolEffectRepository
from app.storage.social_agent import SocialAgentRepository

ACCOUNT = 900000099
NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


def patrol_job(kind="qzone_publish", **changes):
    return replace(SocialJob(kind, kind, ACCOUNT, quiet_hours=("23:00", "08:00"),
        public_topics=("音乐",), autonomous_tools=True),
        patrol_interval_seconds=(5400, 10800), morning_delay_seconds=(1800, 5400),
        shared_cooldown_seconds=3600, **changes)


def effect(repo, label, *, friend=None, owner=False, run=None):
    operation = "publish_qzone" if friend is None else "comment_qzone"
    scope = "public:qzone_publish" if friend is None else f"public:qzone_comment:friend:{friend}"
    if owner:
        scope = "owner:" + scope
    return repo.reserve(scope, run_id=run or label, tool_call_id=label,
        operation=operation, payload_hash=__import__("hashlib").sha256(label.encode()).hexdigest())["effect_id"]


@pytest.mark.parametrize("kind", ["model:group_interactive", "model:group_autonomous",
    "model:qzone_publish", "model:qzone_comment"])
def test_chat_usage_cannot_starve_space_budget(sqlite_engine, kind):
    repo = AutonomousRepository(sqlite_engine, account_id=ACCOUNT, clock=lambda: NOW)
    for _ in range(70):
        assert repo.reserve_cost("model", 500)
    assert repo.cost_available(kind, 1)
    assert repo.reserve_cost(kind, 1)
    assert not repo.cost_available(kind, 1)
    assert not repo.reserve_cost(kind, 1)
    assert repo.reserve_cost("owner:model", 1)
    assert repo.reserve_cost("private:model", 1)


def test_shared_space_interval_survives_midnight_and_owner_is_separate(sqlite_engine):
    clock = [NOW.replace(hour=15, minute=50)]  # Shanghai 23:50
    repo = AgentToolEffectRepository(sqlite_engine, account_id=ACCOUNT, clock=lambda: clock[0])
    first = effect(repo, "first")
    assert repo.mark_dispatching(first, daily_limit=1, cooldown_seconds=64800,
        shared_cooldown_seconds=3600)
    repo.finish(first, "sent", receipt="synthetic-post")
    manual = effect(repo, "manual", friend=900000101, owner=True)
    assert repo.mark_dispatching(manual, daily_limit=None)
    clock[0] += timedelta(minutes=20)
    blocked = effect(repo, "cross-day", friend=900000102)
    assert not repo.mark_dispatching(blocked, daily_limit=1, operation_daily_limit=3,
        shared_cooldown_seconds=3600)
    assert repo.get(blocked)["reason"] == "space_shared_cooldown"
    clock[0] += timedelta(minutes=41)
    assert repo.mark_dispatching(effect(repo, "later", friend=900000102), daily_limit=1,
        operation_daily_limit=3, shared_cooldown_seconds=3600)
    # Daily reset must not reset the 18-hour publication spacing.
    assert not repo.mark_dispatching(effect(repo, "publish-too-soon"), daily_limit=1,
        cooldown_seconds=64800)


def test_comments_have_both_global_and_per_friend_limits(sqlite_engine):
    clock = [NOW]
    repo = AgentToolEffectRepository(sqlite_engine, account_id=ACCOUNT, clock=lambda: clock[0])
    for n in range(3):
        current = effect(repo, str(n), friend=900000101 + n)
        assert repo.mark_dispatching(current, daily_limit=1, operation_daily_limit=3,
            shared_cooldown_seconds=3600)
        repo.finish(current, "sent")
        clock[0] += timedelta(hours=1)
    assert not repo.mark_dispatching(effect(repo, "fourth", friend=900000104), daily_limit=1,
        operation_daily_limit=3, shared_cooldown_seconds=3600)
    assert not repo.mark_dispatching(effect(repo, "same-friend", friend=900000101), daily_limit=1)


def test_one_space_write_per_original_wake_even_with_post_dedupe_run(sqlite_engine):
    clock = [NOW]
    repo = AgentToolEffectRepository(sqlite_engine, account_id=ACCOUNT, clock=lambda: clock[0])
    first = effect(repo, "a", friend=900000101, run="post:a")
    assert repo.mark_dispatching(first, daily_limit=1, wake_id="patrol:1", max_writes_per_wake=1)
    repo.finish(first, "sent")
    clock[0] += timedelta(hours=2)
    second = effect(repo, "b", friend=900000102, run="post:b")
    assert not repo.mark_dispatching(second, daily_limit=1, wake_id="patrol:1", max_writes_per_wake=1)
    assert repo.get(second)["reason"] == "space_wake_budget"


def test_global_comment_reservations_are_atomic(sqlite_engine):
    repo = AgentToolEffectRepository(sqlite_engine, account_id=ACCOUNT, clock=lambda: NOW)
    ids = [effect(repo, f"race-{n}", friend=900000101+n) for n in range(6)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda identifier: repo.mark_dispatching(identifier,
            daily_limit=1, operation_daily_limit=3), ids))
    assert sum(results) == 3


def test_patrol_is_one_persistent_random_schedule(sqlite_engine):
    jobs = (patrol_job(), patrol_job("qzone_comment", max_daily_per_friend=1))
    repo = SocialAgentRepository(sqlite_engine, random_seconds=lambda low, high: low)
    lease = repo.acquire(account_id=ACCOUNT, owner="one", now=NOW, lease_seconds=20000)
    assert repo.claim_due_jobs(account_id=ACCOUNT, owner="one", epoch=lease.epoch, jobs=jobs, now=NOW) == ()
    later = NOW + timedelta(seconds=5400)
    due = repo.claim_due_jobs(account_id=ACCOUNT, owner="one", epoch=lease.epoch, jobs=jobs, now=later)
    assert len(due) == 2 and len({row.due_at for row in due}) == 1
    # Candidate pair is one patrol; the runtime chooses at most one model turn.
    restarted = SocialAgentRepository(sqlite_engine, random_seconds=lambda low, high: high)
    assert restarted.claim_due_jobs(account_id=ACCOUNT, owner="one", epoch=lease.epoch, jobs=jobs, now=later) == ()


def test_midnight_patrol_moves_to_random_morning(sqlite_engine):
    repo = SocialAgentRepository(sqlite_engine, random_seconds=lambda low, high: low)
    midnight = NOW.replace(hour=16, minute=0)
    lease = repo.acquire(account_id=ACCOUNT, owner="one", now=midnight, lease_seconds=50000)
    jobs = (patrol_job(),)
    assert not repo.claim_due_jobs(account_id=ACCOUNT, owner="one", epoch=lease.epoch, jobs=jobs, now=midnight)
    morning = midnight + timedelta(hours=8)
    assert not repo.claim_due_jobs(account_id=ACCOUNT, owner="one", epoch=lease.epoch, jobs=jobs, now=morning)
    assert repo.claim_due_jobs(account_id=ACCOUNT, owner="one", epoch=lease.epoch, jobs=jobs,
        now=morning + timedelta(minutes=30))


def test_actual_assessment_expires_and_content_change_is_immediately_eligible(sqlite_engine):
    repo = SocialAgentRepository(sqlite_engine)
    post = SocialPost(900000101, "post-a", "公开内容", NOW)
    assert repo.unreviewed_posts(account_id=ACCOUNT, posts=(post,)) == (post,)
    repo.assess_post(account_id=ACCOUNT, post=post, status='no_response', now=NOW, minutes=30)
    assert not SocialAgentRepository(sqlite_engine).unreviewed_posts(account_id=ACCOUNT, posts=(post,), now=NOW)
    changed = replace(post, text="新增内容")
    assert repo.unreviewed_posts(account_id=ACCOUNT, posts=(changed,), now=NOW) == (changed,)
    assert repo.unreviewed_posts(account_id=ACCOUNT, posts=(post,), now=NOW+timedelta(minutes=31)) == (post,)


@pytest.mark.parametrize("choose_high,expected_minutes", [(False, 30), (True, 90)])
def test_first_start_after_quiet_end_uses_morning_window(sqlite_engine, choose_high, expected_minutes):
    from app.storage.social_agent import SocialScheduledJob
    from app.storage.db import session_scope
    from sqlalchemy import select
    morning = NOW.replace(hour=0, minute=5)  # Shanghai 08:05
    repo = SocialAgentRepository(sqlite_engine, random_seconds=lambda low, high: high if choose_high else low)
    lease = repo.acquire(account_id=ACCOUNT, owner="first", now=morning, lease_seconds=20000)
    assert not repo.claim_due_jobs(account_id=ACCOUNT, owner="first", epoch=lease.epoch, jobs=(patrol_job(),), now=morning)
    with session_scope(sqlite_engine) as session:
        due = session.scalar(select(SocialScheduledJob.next_due).where(SocialScheduledJob.job_key == "qzone_patrol"))
    assert due == morning.replace(minute=0, tzinfo=None) + timedelta(minutes=expected_minutes)


def enable_patrol(fixture):
    from tests.agent.test_autonomous_social import FRIEND
    qzone = fixture.router.runtime.social_policy["qzone"]
    qzone["patrol"] = {"enabled": True, "interval_seconds": [5400, 10800],
        "morning_delay_seconds": [1800, 5400], "shared_cooldown_seconds": 3600}
    qzone.setdefault("publish", {"enabled": True, "autonomous_tools": True, "topics": ["音乐"],
        "quiet_hours": None, "max_daily": 1, "cooldown_seconds": 64800})
    qzone["publish"].update(max_daily=1, cooldown_seconds=64800)
    qzone.setdefault("comment", {"enabled": True, "autonomous_tools": True, "friend_ids": [FRIEND],
        "quiet_hours": None, "max_daily": 3, "max_daily_per_friend": 1, "cooldown_seconds": 3600})
    qzone["comment"].update(max_daily=3, max_daily_per_friend=1, cooldown_seconds=3600)
    fixture.agent.repository.random_seconds = lambda low, high: low


@pytest.mark.parametrize("chosen", ["qzone_publish", "qzone_comment"])
async def test_real_harness_patrol_selects_only_one_surface(sqlite_engine, tmp_path, chosen):
    from tests.agent.test_autonomous_social import composed, FRIEND, NOW as START, writes
    from tests.agent.test_mature_composition import model_steps
    operation = "publish_qzone" if chosen == "qzone_publish" else "comment_qzone"
    args = {"text": "合成主动内容", "artifact_ids": []}
    if chosen == "qzone_comment":
        args.update(owner_id=FRIEND, post_id="observed-photo")
    client, requests, _ = model_steps((operation, args), None)
    async with composed(sqlite_engine, tmp_path, client, kind="qzone_publish", runtime="harness") as fixture:
        enable_patrol(fixture)
        # A busy chat exhausted the former shared 60-request limit.
        for _ in range(70):
            assert fixture.executor.repository.reserve_cost("model", 500)
        fixture.agent.choose_patrol = lambda rows: next(row for row in rows if row.job.kind == chosen)
        await fixture.agent.tick(fixture.clock[0])
        fixture.clock[0] += timedelta(seconds=5401)
        await fixture.agent.tick(fixture.clock[0])
        assert len(requests) > 0
        assert [name for name, _ in writes(fixture)] == ["send_qzone_msg" if chosen == "qzone_publish" else operation]
        from app.storage.db import session_scope
        from app.storage.social_agent import SocialActionRecord
        from sqlalchemy import select
        with session_scope(sqlite_engine) as session:
            records = list(session.scalars(select(SocialActionRecord)))
            assert len(records) == 1 and records[0].status == "sent", [(r.status, r.reason) for r in records]
        assert not fixture.executor.repository.cost_available("model:" + chosen, len(requests))


async def test_quota_and_empty_feed_prevent_all_model_calls(sqlite_engine, tmp_path):
    from tests.agent.test_autonomous_social import composed
    from tests.agent.test_mature_composition import model_steps
    client, requests, _ = model_steps(None)
    async with composed(sqlite_engine, tmp_path, client, kind="qzone_publish", runtime="harness") as fixture:
        enable_patrol(fixture)
        fixture.gateway.posts = []
        effects = AgentToolEffectRepository(sqlite_engine, account_id=fixture.router.runtime.settings.bot_qq,
            clock=lambda: fixture.clock[0])
        sent = effect(effects, "already-published")
        assert effects.mark_dispatching(sent, daily_limit=1)
        effects.finish(sent, "sent", receipt="synthetic-post")
        await fixture.agent.tick(fixture.clock[0])
        fixture.clock[0] += timedelta(seconds=5401)
        await fixture.agent.tick(fixture.clock[0])
        assert not requests


async def test_silence_reviews_new_post_and_wakeup_cannot_advance_next_patrol(sqlite_engine, tmp_path):
    from tests.agent.test_autonomous_social import composed
    from tests.agent.test_mature_composition import model_steps
    client, requests, _ = model_steps(None)
    async with composed(sqlite_engine, tmp_path, client, kind="qzone_publish", runtime="harness") as fixture:
        enable_patrol(fixture)
        # Disable publishing so the only opportunity is a fresh friend post.
        fixture.router.runtime.social_policy["qzone"]["publish"]["enabled"] = False
        await fixture.agent.tick(fixture.clock[0])
        fixture.clock[0] += timedelta(seconds=5401)
        await fixture.agent.tick(fixture.clock[0])
        assert len(requests) == 1
        job = fixture.agent.jobs()[0]
        scope = fixture.agent.scope_for_job(job)
        fixture.executor.repository.write_intention(scope, {"goal": "观察公开动态"}, (),
            next_due=fixture.clock[0] + timedelta(seconds=60), status="waiting")
        fixture.clock[0] += timedelta(seconds=61)
        await fixture.agent.tick(fixture.clock[0])
        assert len(requests) == 1
        fixture.clock[0] += timedelta(seconds=5401)
        await fixture.agent.tick(fixture.clock[0])
        assert len(requests) == 1  # Same declined post causes no further model calls.


def test_additive_wake_column_migration_preserves_old_effect(sqlite_engine):
    from app.storage.db import create_all
    with sqlite_engine.begin() as connection:
        connection.exec_driver_sql('DROP TABLE agent_tool_effects')
        connection.exec_driver_sql('CREATE TABLE agent_tool_effects (action_id VARCHAR(80) PRIMARY KEY)')
        connection.exec_driver_sql("INSERT INTO agent_tool_effects VALUES ('synthetic-existing')")
    create_all(sqlite_engine)
    create_all(sqlite_engine)
    with sqlite_engine.connect() as connection:
        assert connection.exec_driver_sql('SELECT action_id,wake_id,reservation_attempts FROM agent_tool_effects').fetchall() == [
            ('synthetic-existing', '', 1)]
