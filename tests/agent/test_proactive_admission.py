from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.autonomous_social import AutonomousSocialExecutor
from app.agent.observation import GroupObservationService
from app.agent.social_contracts import SocialJob, SocialObservation
from app.agent.tool_agent import ToolTurnResult
from tests.agent.test_autonomous_social import add_source, NOW, BOT, GROUP


def executor(engine):
    router = SimpleNamespace(engine=engine, runtime=SimpleNamespace(settings=SimpleNamespace(bot_qq=BOT), group_policy={}),
        _group_policy_int=lambda **kw: kw['default'])
    service = object.__new__(AutonomousSocialExecutor)
    service.router, service.clock, service.runtime_mode = router, lambda: NOW, 'harness'
    service._scope_locks = {}
    from app.storage.autonomous_agent import AutonomousRepository
    service.repository = AutonomousRepository(engine, account_id=BOT, clock=lambda: NOW)
    service._run_tools = AsyncMock(return_value=ToolTurnResult('silent'))
    return service


def observed(ids):
    return SocialObservation(SocialJob(key=f'group:{GROUP}', kind='group', scope_id=GROUP,
        autonomous_tools=True), NOW, 'synthetic persona', source_message_ids=tuple(ids))


def guard(ids):
    return SimpleNamespace(require_valid=lambda: None, valid=lambda: True,
        trace_trigger={'trigger_evidence': {'source_ids': list(ids)}})


async def test_single_messages_only_wake_after_eight_new_sources(sqlite_engine,monkeypatch):
    service = executor(sqlite_engine)
    judge=AsyncMock(return_value=(ToolTurnResult('silent'),None))
    monkeypatch.setattr('app.agent.participation_judge.judge_participation',judge)
    ids = []
    for index in range(8):
        ids.append(add_source(sqlite_engine, platform_id=f'synthetic-{index}', text=f'讨论内容 {index}'))
        await service.run_tools(observed(ids), guard([ids[-1]]), interactive=True)
        assert judge.await_count == 0
        assert service._run_tools.await_count == int(index == 7)
    # New arrivals cannot immediately trigger another silent judgment.
    next_id = add_source(sqlite_engine, platform_id='synthetic-next', text='新的话题')
    await service.run_tools(observed([*ids, next_id]), guard([next_id]), interactive=True)
    assert judge.await_count == 0
    assert service._run_tools.await_count == 1


async def test_direct_request_and_result_delivery_bypass_proactive_budget(sqlite_engine):
    service = executor(sqlite_engine)
    source = add_source(sqlite_engine, platform_id='direct', text='小町请回答')
    for kwargs in [{'direct_group_reply': True}, {'research_result': {}}, {'owner_command': {'target': 'group',
                    'group_id': GROUP, 'task_id': 'synthetic'}}]:
        await service.run_tools(observed([source]), guard([source]), interactive=True, **kwargs)
    assert service._run_tools.await_count == 3


def test_recent_observation_is_thirty_and_preserves_canonical_ids(sqlite_engine):
    service = executor(sqlite_engine)
    ids = [add_source(sqlite_engine, platform_id=f'window-{i}', text=f'窗口消息 {i}') for i in range(37)]
    from app.storage.db import session_scope
    from app.storage.repositories import MessageRepository
    with session_scope(sqlite_engine) as session:
        rows = GroupObservationService(service.router)._recent(MessageRepository(session), GROUP)
        assert [row.id for row in rows] == ids[-30:]


@pytest.mark.parametrize('deferral', ['proactive_admission:proactive_model_limit', 'patrol_superseded'])
async def test_due_goal_survives_local_quota_deferral(sqlite_engine, deferral):
    from app.agent.social_runtime import SocialRuntime
    from app.agent.social_contracts import SocialDecision, SocialResult, SocialRuntimeOptions
    from app.storage.autonomous_agent import AutonomousRepository
    notes = AutonomousRepository(sqlite_engine, account_id=BOT, clock=lambda: NOW)
    scope = f'group:{GROUP}'
    notes.write_intention(scope, {'goal': '到期后继续核对', 'progress': '', 'next_step': '等待额度'}, ())
    notes.schedule_intention(scope, next_due=NOW, status='active')
    job = SocialJob(key=scope, kind='group', scope_id=GROUP, autonomous_tools=True, quiet_hours=None)
    async def observe(job, now):
        return SocialObservation(job, now, 'synthetic persona')
    async def decide(observation):
        return SocialDecision('run_tools', 'synthetic due goal')
    async def execute(*args):
        return SocialResult('blocked' if deferral == 'patrol_superseded' else 'silent', reason=deferral)
    runtime = SocialRuntime(engine=sqlite_engine, account_id=BOT, jobs=lambda: (job,),
        observe=observe, decide=decide, execute=execute, policy_check=lambda *_: True,
        clock=lambda: NOW, intention_repository=notes, scope_for_job=lambda _: scope,
        options=SocialRuntimeOptions(poll_seconds=999, observation_timeout_seconds=1,
            decision_timeout_seconds=1, execution_timeout_seconds=1, lease_seconds=10,
            shutdown_timeout_seconds=.1))
    await runtime.start()
    runtime.set_connected(True)
    try:
        await runtime.tick(NOW)
        assert notes.read_intention_schedule(scope)['next_due'] is not None
        assert notes.read_intention(scope)['working_intention']['goal'] == '到期后继续核对'
    finally:
        await runtime.stop()
