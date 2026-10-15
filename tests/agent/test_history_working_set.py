from datetime import timedelta
import json

import httpx
import pytest

from app.agent.harness_runtime import HarnessToolAgent
from app.agent.social_contracts import SocialObservation
from app.agent.social_policy import configured_social_jobs
from app.agent.tool_agent import ToolTurnResult
from app.storage.proactive import ProactiveLimits, ProactiveRepository
from tests.agent.test_autonomous_social import add_source, composed, ScriptedModel, PolicyGuard, NOW, GROUP, FRIEND, BOT
from tests.agent.test_harness_runtime import bridge_factory, request_body, ModelHandler


@pytest.mark.parametrize('direct,budget,status,rotates', [(True, 24000, 'sent', True),
    (False, 8000, 'sent', True), (False, 8000, 'silent', False)])
async def test_history_budget_rotates_even_if_native_compaction_disabled(sqlite_engine, tmp_path, monkeypatch, direct, budget, status, rotates):
    sessions = []
    class Harness:
        def __init__(self, **kwargs):
            sessions.append(kwargs['session_id'])
        async def run(self, *args, **kwargs):
            return ToolTurnResult(status, trace=({'status': 'model_request', 'outcome': 'completed',
                'output_disposition': 'committed',
                'context_history_estimated_tokens': budget + 1},))
    monkeypatch.setattr('app.agent.harness_runtime.HarnessToolAgent', Harness)
    source = add_source(sqlite_engine, platform_id='history-budget-request', text='小町解释问题')
    async with composed(sqlite_engine, tmp_path, ScriptedModel(), runtime='harness') as fixture:
        fixture.router.runtime.settings.agent_context_compaction_enabled = False
        job = configured_social_jobs(fixture.router.runtime)[0]
        observation = SocialObservation(job, NOW, '小町', source_message_ids=(source,))
        for _ in range(2):
            await fixture.executor.run_tools(observation, PolicyGuard(fixture.executor, observation, scheduled=not direct),
                interactive=True, direct_group_reply=direct, target_message_id='history-budget-request', target_user_id=FRIEND)
            fixture.clock[0] += timedelta(minutes=31)
        assert (sessions[0] != sessions[1]) == rotates


def test_transport_retries_and_compaction_share_proactive_request_cap(sqlite_engine, bridge_factory):
    repo = ProactiveRepository(sqlite_engine, account_id=BOT, clock=lambda: NOW)
    limits = ProactiveLimits(hourly_models=2)
    permit, _, _ = repo.claim(GROUP, candidates=[(i, str(i)) for i in range(1, 9)],
        bootstrap_cursor=0, max_source_id=8, limits=limits)
    bridge, requests, _ = bridge_factory(lambda *_: httpx.Response(503, json={'error': 'synthetic'}))
    bridge.owner.before_call = lambda kind: repo.reserve(permit, 'model', limits=limits)
    bridge.forward(ModelHandler(), request_body())
    assert len(requests) == 2
    compactor, compact_requests, _ = bridge_factory(lambda *_: httpx.Response(503))
    compactor.owner.before_call = lambda kind: repo.reserve(permit, 'model', limits=limits)
    compactor.forward(ModelHandler(), request_body(), compaction=True)
    assert len(requests) == 2 and compact_requests == []
