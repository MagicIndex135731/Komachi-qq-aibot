"""Independent architecture review probes for mature runtime boundaries."""
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.agent.artifacts import AutonomousArtifactStore
from app.agent.social_contracts import SocialDecision, SocialJob, SocialObservation, SocialResult, SocialRuntimeOptions
from app.agent.social_runtime import SocialRuntime
from app.storage.autonomous_agent import AutonomousRepository
from app.storage.social_agent import SocialAgentRepository


ACCOUNT = 900000101
NOW = datetime(2026, 10, 9, 8, tzinfo=UTC)


def test_mature_public_artifact_namespace_exists_and_excludes_group_material(sqlite_engine, tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    group_store = AutonomousArtifactStore(root, account_id=ACCOUNT, scope="group:900000001",
        engine=sqlite_engine, source_message_ids=lambda: ())
    material = root / "synthetic-private.txt"
    material.write_text("synthetic group-only material", encoding="utf8")
    artifact_id = group_store.add(material, "synthetic_group_work")
    public_store = AutonomousArtifactStore(root, account_id=ACCOUNT, scope="public",
        engine=sqlite_engine, source_message_ids=lambda: ())
    assert public_store.get(artifact_id) is None
    assert not public_store.list()


def test_mature_comment_wake_does_not_require_scheduler_selected_post(sqlite_engine):
    repo = SocialAgentRepository(sqlite_engine)
    lease = repo.acquire(account_id=ACCOUNT, owner="synthetic-worker", now=NOW, lease_seconds=60)
    job = SocialJob("comments", "qzone_comment", ACCOUNT, friend_ids=(900000201,),
        autonomous_tools=True, quiet_hours=None)
    action_id = repo.reserve_action(account_id=ACCOUNT, owner=lease.owner, epoch=lease.epoch,
        job=job, decision=SocialDecision("run_tools", "observe and choose tools"),
        dedupe_key="synthetic mature wake", now=NOW)
    assert action_id is not None


@pytest.mark.asyncio
async def test_due_goal_is_not_consumed_before_runtime_can_observe_it(sqlite_engine):
    notes = AutonomousRepository(sqlite_engine, account_id=ACCOUNT, clock=lambda: NOW)
    notes.write_intention("public", {"goal": "synthetic cross-wake goal"}, next_due=NOW, status="waiting")
    quiet = SocialJob("publish", "qzone_publish", ACCOUNT, autonomous_tools=True,
        quiet_hours=("00:00", "23:59"), public_topics=("synthetic topic",))
    executions = []

    async def observe(job, now):
        return SocialObservation(job, now, "synthetic public persona")

    async def decide(observation):
        return SocialDecision("run_tools", "synthetic wake")

    async def execute(observation, decision, guard):
        executions.append(observation.job.key)
        return SocialResult("silent")

    runtime = SocialRuntime(engine=sqlite_engine, account_id=ACCOUNT, jobs=lambda: (quiet,),
        observe=observe, decide=decide, execute=execute, policy_check=lambda *_: True,
        clock=lambda: NOW, intention_repository=notes, scope_for_job=lambda _: "public",
        options=SocialRuntimeOptions(poll_seconds=999, observation_timeout_seconds=1,
            decision_timeout_seconds=1, execution_timeout_seconds=1, lease_seconds=10,
            shutdown_timeout_seconds=0.1))
    await runtime.start()
    runtime.set_connected(True)
    try:
        await runtime.tick(NOW)
    finally:
        await runtime.stop()
    assert executions == []
    assert notes.read_intention_schedule("public")["next_due"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("rotation_reason", ["source_budget", "in_flight"])
async def test_real_runtime_rotates_prior_session_at_source_budget_or_interruption(sqlite_engine, tmp_path, rotation_reason):
    from app.agent.social_policy import configured_social_jobs
    from test_autonomous_social import GROUP, NOW as FIXTURE_NOW, PolicyGuard, add_source, composed
    from test_harness_runtime import model_client, responses_sse

    client, requests, _ = model_client(lambda _, number: responses_sse(number=number))
    old_sources = tuple(add_source(sqlite_engine, platform_id=f"rotation-old-{number}")
                        for number in range(200 if rotation_reason == "source_budget" else 1))
    new_source = add_source(sqlite_engine, platform_id="rotation-current", text="synthetic new event")
    async with composed(sqlite_engine, tmp_path, client, runtime="harness") as fixture:
        job = next(job for job in configured_social_jobs(fixture.router.runtime) if job.kind == "group")
        scope = f"group:{GROUP}"
        old_identity = "synthetic-old-session"
        checkpoint = {"session_id": old_identity, "session_path": str(fixture.executor.sessions.session_path(scope))}
        if rotation_reason == "in_flight":
            checkpoint["in_flight"] = True
        fixture.executor.sessions.save(scope, [
            {"role": "user", "content": "synthetic closed prior event"},
            {"role": "assistant", "content": "synthetic closed prior result"}], checkpoint,
            old_sources, expected_revision=0)
        observation = SocialObservation(job, FIXTURE_NOW, "synthetic persona", source_message_ids=(new_source,))
        result = await fixture.executor.run_tools(observation, PolicyGuard(fixture.executor, observation))
        assert result.status == "silent"
        restored = fixture.executor.sessions.read(scope)
        assert restored["checkpoint"]["session_id"] != old_identity
        assert not restored["checkpoint"].get("in_flight")
        assert new_source in restored["source_message_ids"]
        assert len(restored["source_message_ids"]) <= 200
        if rotation_reason == "source_budget":
            assert restored["source_message_ids"] == [new_source]
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_production_group_long_native_checkpoint_resumes_without_rotating(sqlite_engine, tmp_path):
    import json
    from app.agent.harness_session_files import native_file_path, native_session_size
    from app.agent.social_policy import configured_social_jobs
    from test_autonomous_social import GROUP, NOW as FIXTURE_NOW, PolicyGuard, add_source, composed
    from test_harness_runtime import model_client, responses_sse
    from test_harness_session_files import long_session_home

    client, requests, _ = model_client(lambda _, number: responses_sse(
        text="production long-path first receipt" if number == 1 else "restored second wake", number=number))
    source = add_source(sqlite_engine, platform_id="long-path-production-source")
    async with composed(sqlite_engine, tmp_path, client, runtime="harness") as fixture:
        fixture.executor.sessions.session_root = long_session_home(tmp_path) / "agent_sessions"
        job = next(job for job in configured_social_jobs(fixture.router.runtime) if job.kind == "group")
        observation = SocialObservation(job, FIXTURE_NOW, "synthetic persona", source_message_ids=(source,))
        scope = f"group:{GROUP}"
        identity = None
        for _ in range(2):
            result = await fixture.executor.run_tools(observation, PolicyGuard(fixture.executor, observation))
            assert result.status == "silent"
            state = fixture.executor.sessions.read(scope)
            checkpoint = state["checkpoint"]
            logical_path = Path(checkpoint["session_path"])
            assert len(str(logical_path)) > 260
            assert not str(logical_path).startswith("\\\\?\\")
            assert native_session_size((logical_path,)) > 0
            assert "production long-path first receipt" in native_file_path(logical_path).read_text(encoding="utf-8")
            assert not checkpoint.get("in_flight") and source in state["source_message_ids"]
            if identity is not None:
                assert checkpoint["session_id"] == identity
            identity = checkpoint["session_id"]
    assert len(requests) == 2
    assert "production long-path first receipt" in json.dumps(requests[1])
