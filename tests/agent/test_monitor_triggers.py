"""Trigger evidence comes from the real claimed schedule, without model calls."""
from datetime import UTC, datetime

import pytest

from app.agent.social_contracts import SocialDecision, SocialJob, SocialObservation, SocialResult, SocialRuntimeOptions
from app.agent.social_runtime import SocialRuntime
from app.storage.autonomous_agent import AutonomousRepository


@pytest.mark.asyncio
async def test_claimed_goal_preserves_due_cause_and_scope(sqlite_engine):
    now = datetime(2026, 10, 14, 8, tzinfo=UTC)
    notes = AutonomousRepository(sqlite_engine, account_id=900000101, clock=lambda: now)
    notes.write_intention('public', {'goal': 'synthetic source check'}, next_due=now, status='waiting')
    job = SocialJob('synthetic-public', 'qzone_publish', 900000101, autonomous_tools=True,
        quiet_hours=None, public_topics=('synthetic',))
    seen = []
    async def observe(item, clock):
        return SocialObservation(item, clock, 'synthetic persona')
    async def decide(_):
        return SocialDecision('run_tools', 'synthetic wake')
    async def execute(observation, decision, guard):
        seen.append(guard.trace_trigger)
        return SocialResult('silent')
    runtime = SocialRuntime(engine=sqlite_engine, account_id=900000101, jobs=lambda:(job,),
        observe=observe, decide=decide, execute=execute, policy_check=lambda *_:True,
        clock=lambda:now, intention_repository=notes, scope_for_job=lambda _:'public',
        options=SocialRuntimeOptions(poll_seconds=999, observation_timeout_seconds=1,
            decision_timeout_seconds=1, execution_timeout_seconds=1, lease_seconds=10,
            shutdown_timeout_seconds=.1))
    await runtime.start();runtime.set_connected(True)
    try:
        await runtime.tick(now)
    finally:
        await runtime.stop()
    assert len(seen)==1
    assert seen[0]['trigger_kind']=='public_goal' and seen[0]['trigger_reason_code']=='intention_due'
    evidence=seen[0]['trigger_evidence']
    assert evidence['due_at']==now.isoformat() and evidence['goal']=='synthetic source check'
    assert evidence['goal_scope']=='public' and evidence['schedule_revision']>0
    assert notes.read_intention_schedule('public')['next_due'] is None


@pytest.mark.asyncio
async def test_native_group_timer_without_changes_skips_model(sqlite_engine, tmp_path):
    from app.agent.runtime_trace import TraceReader
    from test_autonomous_social import composed, wake
    from test_harness_runtime import model_client, responses_sse
    client, requests, _ = model_client(lambda _, n: responses_sse(number=n))
    try:
        async with composed(sqlite_engine, tmp_path, client, runtime='harness') as fixture:
            fixture.router.runtime.settings.agent_monitor_enabled=True
            await wake(fixture)
            traces=TraceReader(fixture.router.runtime.settings.data_dir/'agent_monitor').list_traces()
            assert traces == [] and requests == []
    finally:
        client.http_client.close()
