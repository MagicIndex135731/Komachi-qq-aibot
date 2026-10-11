from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.agent.tool_agent import AgentTool, ToolOutcome
from app.agent.tool_dispatch import ToolDispatchBoundary
from app.storage.agent_effects import AgentToolEffectRepository

NOW = datetime(2026, 10, 11, 0, tzinfo=UTC)
BOT = 900000099

class Guard:
    action_id = 'synthetic-owner-one'
    dispatched = False
    def valid(self): return True
    def require_valid(self): pass
    def on_dispatch(self): self.dispatched = True

def boundary(engine, *, owner=False, clock=lambda: NOW, run='synthetic-one'):
    guard = Guard(); guard.action_id = run
    return ToolDispatchBoundary(engine=engine, account_id=BOT, scope='public', guard=guard,
        clock=clock, max_daily=1, cooldown_seconds=21600, owner_authority=(lambda: True) if owner else None)

def tool(b, name='publish_qzone', *, per_friend=False):
    async def send(args):
        b.on_dispatch()
        return ToolOutcome({'acknowledged': True}, 'sent', 'synthetic-receipt')
    return AgentTool(name, 'synthetic', {}, send, 'write', scope='public:'+name,
        daily_limit=1, cooldown_seconds=21600, per_target_daily_limit=per_friend)

async def test_owner_actions_bypass_and_do_not_charge_autonomous_quotas(sqlite_engine):
    first = boundary(sqlite_engine, owner=True)
    assert (await first.invoke(tool(first), {'text':'主人发第一条'}, 'one')).status == 'sent'
    second = boundary(sqlite_engine, owner=True, run='synthetic-owner-two')
    assert (await second.invoke(tool(second), {'text':'主人发第二条'}, 'two')).status == 'sent'
    auto = boundary(sqlite_engine, run='synthetic-auto')
    assert (await auto.invoke(tool(auto), {'text':'自主发表'}, 'auto')).status == 'sent'
    blocked = boundary(sqlite_engine, run='synthetic-auto-two')
    assert (await blocked.invoke(tool(blocked), {'text':'第二次自主发表'}, 'auto2')).status != 'sent'
    repo = AgentToolEffectRepository(sqlite_engine, account_id=BOT, clock=lambda: NOW)
    assert repo.count_dispatched(scope='public:publish_qzone') == 1
    assert repo.count_dispatched(scope='owner:public:publish_qzone') == 2

async def test_comments_limit_each_friend_independently_and_reset_at_shanghai_midnight(sqlite_engine):
    instant = [datetime(2026, 10, 11, 15, 59, 59, tzinfo=UTC)]
    async def comment(friend, post):
        b = boundary(sqlite_engine, clock=lambda: instant[0], run='run-'+post)
        return await b.invoke(tool(b, 'comment_qzone', per_friend=True),
            {'owner_id': friend, 'post_id': post, 'text':'合成评论'}, post)
    assert (await comment(900000201, 'a')).status == 'sent'
    assert (await comment(900000202, 'b')).status == 'sent'
    assert (await comment(900000201, 'c')).status != 'sent'
    instant[0] += timedelta(seconds=1)
    # Natural-day quota replaces the global comment cooldown.
    assert (await comment(900000201, 'd')).status == 'sent'

async def test_owner_authority_revocation_blocks_dispatch_and_unknown_is_not_retried(sqlite_engine):
    valid = [True]
    b = boundary(sqlite_engine, owner=True)
    b.owner_authority = lambda: valid[0]
    t = tool(b)
    valid[0] = False
    with pytest.raises(RuntimeError):
        await b.invoke(t, {'text':'撤销后不能发送'}, 'revoked')
    valid[0] = True
    async def unknown(args):
        b.on_dispatch()
        return ToolOutcome({}, 'uncertain')
    t = AgentTool('publish_qzone', 'synthetic', {}, unknown, 'write', scope='public:publish_qzone')
    assert (await b.invoke(t, {'text':'未知结果'}, 'first')).status == 'uncertain'
    again = boundary(sqlite_engine, owner=True, run='another-owner-message')
    assert (await again.invoke(tool(again), {'text':'未知结果'}, 'second')).status == 'uncertain'

async def test_unknown_public_effect_is_not_replayed_when_authority_changes(sqlite_engine):
    b = boundary(sqlite_engine, run='auto-unknown')
    async def unknown(args):
        b.on_dispatch()
        return ToolOutcome({}, 'uncertain')
    t = AgentTool('publish_qzone','synthetic',{},unknown,'write',scope='public:publish_qzone')
    assert (await b.invoke(t, {'text':'跨权限未知结果'}, 'one')).status == 'uncertain'
    manual = boundary(sqlite_engine, owner=True, run='owner-other')
    assert (await manual.invoke(tool(manual), {'text':'跨权限未知结果'}, 'two')).status == 'uncertain'

def test_per_friend_limit_serializes_competing_workers_and_survives_restart(sqlite_engine):
    from concurrent.futures import ThreadPoolExecutor
    scope='public:qzone_comment:friend:900000201'
    def dispatch(index):
        repo=AgentToolEffectRepository(sqlite_engine,account_id=BOT,clock=lambda:NOW)
        record=repo.reserve(scope,run_id=f'competing-{index}',tool_call_id='comment',
            operation='comment_qzone',payload_hash=str(index)*64)
        return repo.mark_dispatching(record['action_id'],daily_limit=1,cooldown_seconds=0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(dispatch,(1,2)))==1
    restarted=AgentToolEffectRepository(sqlite_engine,account_id=BOT,clock=lambda:NOW)
    assert restarted.count_dispatched(scope=scope)==1
    assert not dispatch(3)
