"""Explicit main-to-child delegation boundaries, using synthetic identities."""
import asyncio
from types import SimpleNamespace

import pytest

from app.agent.delegated_research import (DelegationBudget, run_delegated_research,
                                         child_context_policy, selected_task_sources, _summary)
import json
from app.agent.research_tasks import ResearchTaskWorkspace
from app.agent.tool_agent import AgentTool, ToolTurnResult


class Guard:
    available = True

    def require_valid(self):
        if not self.available:
            raise RuntimeError('source_revoked')


def test_child_has_small_budget_and_only_explicit_sources():
    policy = child_context_policy(SimpleNamespace(llm_context_window_tokens=258000))
    assert policy.context_window == 32000 and policy.summary_tokens == 1024
    assert policy.compaction_enabled and policy.threshold_ratio == .55
    assert child_context_policy(SimpleNamespace(llm_context_window_tokens=16000)).context_window == 16000
    assert selected_task_sources({}, [1, 2, 3], 3) == (3,)
    assert selected_task_sources({'source_ids': [1, 3]}, [1, 2, 3], 3) == (3, 1)
    with pytest.raises(ValueError):
        selected_task_sources({'source_ids': [99]}, [1, 2, 3], 3)
    with pytest.raises(ValueError):
        selected_task_sources({'source_ids': [True]}, [1, 2, 3], 3)


def test_result_is_a_small_summary_not_raw_task_state():
    class Workspace:
        def summary(self):
            return {'goal': '目' * 2000, 'progress': '进' * 2000, 'next_step': '缺' * 1000,
                'findings': ['结' * 1000] * 12, 'last_delivery': 'RAW_BODY', 'dependencies': {'secret': 'RAW_BODY'},
                'evidence': [{'evidence_id': str(n), 'sources': ['https://example.com/' + 'a' * 1900] * 4,
                              'content_chars': 96000, 'content': 'RAW_BODY'} for n in range(20)]}
    result = _summary(Workspace(), status='completed')
    encoded = json.dumps(result, ensure_ascii=False)
    assert len(encoded) <= 6000 and 'RAW_BODY' not in encoded
    assert result['next_step'] and result['findings'] and result['source_bound']


def test_child_budget_is_single_use_and_reserves_parent_delivery():
    now = [100.0]
    budget = DelegationBudget(120, clock=lambda: now[0])
    now[0] += 20
    assert budget.take() == 55
    with pytest.raises(ValueError, match='limit'):
        budget.take()
    now[0] += 40
    deadline = DelegationBudget(55, clock=lambda: now[0])
    with pytest.raises(ValueError, match='deadline'):
        deadline.take()


@pytest.mark.parametrize('mode', ['completed', 'blocked', 'missing', 'failed', 'late_failure', 'revoked'])
async def test_child_checkpoint_requires_successful_model_finish_and_readonly_tools(tmp_path, mode):
    guard = Guard()
    workspace = ResearchTaskWorkspace(tmp_path / 'tasks', scope='private:900000101', actor_id=900000101,
        source_id=123, request='synthetic', valid_source=lambda _: guard.available)
    captured = {}
    costs = []

    class Runtime:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self, prompt, tools, child_guard):
            assert 'PARENT_BULK_HISTORY' not in str(prompt)
            assert {tool.name for tool in tools} == {'search_web', 'update_research_task'}
            assert captured['before_call']('model') is True
            if mode != 'missing':
                workspace.update(progress='核对一项', next_step='未读其他文件', findings=['source-bound finding'],
                                 status='blocked' if mode == 'blocked' else 'completed')
            if mode == 'revoked':
                guard.available = False
            return ToolTurnResult('failed' if mode == 'failed' else 'silent', reason='model_provider_failed' if mode in
                {'failed', 'late_failure'} else '', trace=[{'status': 'model_request',
                'outcome': 'failed' if mode == 'late_failure' else 'completed'}])

        def close(self):
            captured['closed'] = True

    names = ('search_web', 'update_research_task', 'send_private', 'delegate_research_task',
             'memory_write', 'schedule_private_wakeup', 'execute_python', 'request_owner_action')
    result = await run_delegated_research(runtime_factory=Runtime, decision_client=object(), parent_guard=guard,
        workspace=workspace, read_tools=tuple(AgentTool(name, name, {}, lambda _: None) for name in names),
        goal='check synthetic source', deliverable='findings and gaps', reason='multiple large files', source_id=123,
        before_call=lambda kind: costs.append(kind) or True, settings=SimpleNamespace(data_dir=tmp_path),
        timeout_seconds=30)
    assert costs == ['model'] and captured['closed']
    assert captured['trace_context']['parent_context_scope'] == 'private:900000101'
    assert captured['context_policy'].context_window == 32000
    assert '小町' not in captured['system_identity']
    assert result['status'] == ('completed' if mode == 'completed' else 'blocked')
    if mode in {'missing', 'failed', 'late_failure', 'revoked'}:
        assert result['findings'] == []
    if mode in {'failed', 'late_failure', 'missing'}:
        assert workspace.summary()['status'] == 'blocked'


async def test_cancelled_child_cannot_leave_a_completed_checkpoint(tmp_path):
    guard, entered = Guard(), asyncio.Event()
    workspace = ResearchTaskWorkspace(tmp_path, scope='group:900000001', actor_id=900000101,
        source_id=123, request='synthetic', valid_source=lambda _: True)

    class Runtime:
        def __init__(self, **kwargs):
            pass

        async def run(self, prompt, tools, child_guard):
            workspace.update(progress='尚未结束', next_step='', findings=['uncommitted'], status='completed')
            entered.set()
            await asyncio.Event().wait()

        def close(self):
            pass

    running = asyncio.create_task(run_delegated_research(runtime_factory=Runtime, decision_client=object(),
        parent_guard=guard, workspace=workspace, read_tools=(), goal='source check', deliverable='summary',
        reason='large sources', source_id=123, before_call=lambda _: True, settings=SimpleNamespace(data_dir=tmp_path)))
    await entered.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert workspace.summary()['status'] == 'blocked'
    assert workspace.summary()['findings'] == []
