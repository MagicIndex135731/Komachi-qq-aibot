from types import SimpleNamespace

import pytest

from app.agent.self_activity import build_self_activity_tool
from app.storage.agent_effects import AgentToolEffectRepository
from tests.storage.test_self_activity import ACCOUNT, NOW, effect


@pytest.mark.asyncio
async def test_self_activity_checks_guard_before_and_after_query(sqlite_engine):
    checked = []
    class Blocking:
        async def run(self, fn, **kwargs):
            assert checked == ['guard']
            return fn(**kwargs)
    tool = build_self_activity_tool(AgentToolEffectRepository(sqlite_engine, account_id=ACCOUNT,
        clock=lambda: NOW), Blocking(), SimpleNamespace(require_valid=lambda: checked.append('guard')))
    effect(sqlite_engine, 'published-in-another-context')
    result = await tool.handler({})
    assert checked == ['guard', 'guard']
    assert result.status == 'ok' and result.data['published'] is True
    assert tool.effect == 'read' and tool.name == 'read_self_activity'


@pytest.mark.asyncio
async def test_self_activity_revoked_and_invalid_arguments(sqlite_engine):
    class Blocking:
        async def run(self, fn, **kwargs):
            return fn(**kwargs)
    def revoked():
        raise RuntimeError('revoked')
    repo = AgentToolEffectRepository(sqlite_engine, account_id=ACCOUNT, clock=lambda: NOW)
    tool = build_self_activity_tool(repo, Blocking(), SimpleNamespace(require_valid=revoked))
    with pytest.raises(RuntimeError, match='revoked'):
        await tool.handler({})
    tool = build_self_activity_tool(repo, Blocking(), SimpleNamespace(require_valid=lambda: None))
    assert (await tool.handler({'date': 'invalid'})).status == 'blocked'


@pytest.mark.asyncio
async def test_self_activity_revocation_during_read_does_not_return_facts(sqlite_engine):
    valid = [True]
    class Blocking:
        async def run(self, fn, **kwargs):
            result = fn(**kwargs)
            valid[0] = False
            return result
    def require_valid():
        if not valid[0]:
            raise RuntimeError('revoked_after_read')
    repo = AgentToolEffectRepository(sqlite_engine, account_id=ACCOUNT, clock=lambda: NOW)
    tool = build_self_activity_tool(repo, Blocking(), SimpleNamespace(require_valid=require_valid))
    with pytest.raises(RuntimeError, match='revoked_after_read'):
        await tool.handler({})
