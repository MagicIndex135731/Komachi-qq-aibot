"""Regressions for rejected old task admission and independent scheduled turns."""
import json

import pytest

from app.agent.research_tasks import ResearchTaskWorkspace
from app.agent.social_contracts import SocialObservation
from app.agent.social_policy import configured_social_jobs
from app.agent.tool_agent import ToolOutcome, ToolTurnResult
from test_autonomous_social import FRIEND, NOW, PolicyGuard, ScriptedModel, add_source, composed


@pytest.mark.asyncio
@pytest.mark.parametrize('rejection', ['source_budget', 'memory_revoked'])
async def test_rejected_old_task_does_not_poison_direct_turn(sqlite_engine, tmp_path, monkeypatch, rejection):
    sources = [add_source(sqlite_engine, platform_id=f'admission-{i}') for i in range(201)]
    captured = []

    async def memory(context, **kwargs):
        context.observed_sources.update(sources[1:] if rejection == 'source_budget' else [sources[-1]])
        return ToolOutcome({'evidence': '', 'memory_search_allowed': True})

    class Harness:
        def __init__(self, **kwargs):
            pass

        async def run(self, prompt, tools, guard, **kwargs):
            guard.require_valid()
            captured.append(prompt)
            return ToolTurnResult('silent', reason='model_finished')

    monkeypatch.setattr('app.agent.harness_runtime.HarnessToolAgent', Harness)
    monkeypatch.setattr('app.agent.memory_integration.build_agent_memory', memory)
    async with composed(sqlite_engine, tmp_path, ScriptedModel(), runtime='harness') as fixture:
        job = configured_social_jobs(fixture.router.runtime)[0]
        old = ResearchTaskWorkspace(fixture.router.runtime.settings.data_dir / 'research_tasks',
            scope=job.key, actor_id=FRIEND, source_id=sources[0], request='OLD_TASK_MARKER',
            valid_source=lambda _: True)
        old.begin()
        state = json.loads((old.root / 'active.json').read_text())
        state['dependencies'] = {'sources': [sources[0]], 'memory': {'fact:999999': 'revoked'}
                                 if rejection == 'memory_revoked' else {}}
        (old.root / 'active.json').write_text(json.dumps(state))
        observed = SocialObservation(job, NOW, '小町', source_message_ids=(sources[-1],))
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed),
            interactive=True, direct_group_reply=True, target_user_id=FRIEND, target_message_id='admission-200')
        checkpoint = fixture.executor.sessions.read(job.key)
        assert captured
        assert sources[0] not in checkpoint['source_message_ids']
        assert not checkpoint['checkpoint'].get('memory_dependencies')
        assert 'OLD_TASK_MARKER' not in '\n'.join(captured[0])


@pytest.mark.asyncio
async def test_patrol_checkpoint_and_native_home_are_separate_from_main(sqlite_engine, tmp_path, monkeypatch):
    captured = []

    class Harness:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        async def run(self, prompt, tools, guard, **kwargs):
            guard.require_valid()
            return ToolTurnResult('silent', reason='model_finished')

    monkeypatch.setattr('app.agent.harness_runtime.HarnessToolAgent', Harness)
    source = add_source(sqlite_engine, platform_id='main-then-patrol')
    async with composed(sqlite_engine, tmp_path, ScriptedModel(), runtime='harness') as fixture:
        job = configured_social_jobs(fixture.router.runtime)[0]
        observed = SocialObservation(job, NOW, '小町', source_message_ids=(source,))
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed),
            interactive=True, direct_group_reply=True, target_user_id=FRIEND, target_message_id='main-then-patrol')
        main = fixture.executor.sessions.read(job.key)
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed, scheduled=True))
        assert fixture.executor.sessions.read(job.key) == main
        background = fixture.executor.sessions.read('background:' + job.key)
        assert background['revision'] > 0
        assert captured[0]['session_home'] != captured[1]['session_home']
        assert captured[0]['session_id'] != captured[1]['session_id']
        assert captured[1]['trace_context']['context_kind'] == 'background_group'
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed),
            interactive=True, direct_group_reply=True, target_user_id=FRIEND, target_message_id='main-then-patrol')
        assert captured[2]['session_home'] == captured[0]['session_home']
        assert fixture.executor.sessions.read('background:' + job.key) == background


@pytest.mark.asyncio
async def test_main_can_query_public_action_after_independent_patrol(sqlite_engine, tmp_path, monkeypatch):
    from tests.storage.test_self_activity import effect
    results = []

    class Harness:
        def __init__(self, **kwargs):
            pass

        async def run(self, prompt, tools, guard, **kwargs):
            tool = next(t for t in tools if t.name == 'read_self_activity')
            result = await tool.handler({})
            results.append(result.data)
            return ToolTurnResult('silent', reason='model_finished')

    monkeypatch.setattr('app.agent.harness_runtime.HarnessToolAgent', Harness)
    source = add_source(sqlite_engine, platform_id='own-activity-query')
    async with composed(sqlite_engine, tmp_path, ScriptedModel(), runtime='harness') as fixture:
        job = configured_social_jobs(fixture.router.runtime)[0]
        observed = SocialObservation(job, NOW, '小町', source_message_ids=(source,))
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed),
            interactive=True, direct_group_reply=True, target_user_id=FRIEND, target_message_id='own-activity-query')
        main = fixture.executor.sessions.read(job.key)
        # A different execution context finishes a real public tool effect.
        effect(sqlite_engine, 'independent-public-write', stamp=NOW)
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed, scheduled=True))
        assert fixture.executor.sessions.read(job.key) == main
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed),
            interactive=True, direct_group_reply=True, target_user_id=FRIEND, target_message_id='own-activity-query')
        assert results[0]['published'] is False
        assert results[1]['published'] is True and results[2]['published'] is True
        assert results[2]['confirmed_count'] == 1
