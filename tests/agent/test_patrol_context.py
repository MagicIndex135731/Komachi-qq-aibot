import asyncio
from datetime import timedelta
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.agent.context_policy import RequestContextPolicy, estimate_request_tokens
from app.agent.context_recovery import application_turn_marker
from app.agent.patrol_context import OwnerTurnGate, ObservationContextPolicy, observation_context
from app.agent.social_contracts import SocialPost
from app.storage.social_agent import SocialAgentRepository, SocialPatrolCandidate
from app.storage.db import session_scope
from tests.agent.test_qzone_patrol import NOW, ACCOUNT


def parent():
    return SimpleNamespace(valid=lambda: True, require_valid=lambda: None, on_dispatch=lambda: None,
        record_tool_dispatch=lambda _: None)


@pytest.mark.parametrize('dispatched', [False, True])
async def test_direct_input_supersedes_only_unsent_patrol_and_serializes_owner(dispatched):
    gate, order = OwnerTurnGate(), []
    async with gate.turn(parent(), direct=False) as guard:
        if dispatched:
            guard.record_tool_dispatch('synthetic-effect')
        async def direct():
            async with gate.turn(parent(), direct=True):
                order.append('direct')
        async def patrol():
            async with gate.turn(parent(), direct=False):
                order.append('patrol')
        waiting_patrol = asyncio.create_task(patrol())
        waiting_direct = asyncio.create_task(direct())
        await asyncio.sleep(0)
        assert guard.valid() is dispatched
        assert guard.superseded.is_set() is not dispatched
        assert order == []
    await asyncio.gather(waiting_patrol, waiting_direct)
    assert order == ['direct', 'patrol']


async def test_cancellation_does_not_leave_an_owner_locked():
    gate = OwnerTurnGate()
    async with gate.turn(parent(), direct=False):
        async def direct():
            async with gate.turn(parent(), direct=True):
                raise AssertionError('must remain queued')
        task = asyncio.create_task(direct())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    async with gate.turn(parent(), direct=False):
        assert gate.direct_waiters == 0


@pytest.mark.parametrize('protocol', ['openai-responses', 'openai-completions'])
def test_observation_anchor_keeps_whole_tool_pairs_current_turn_and_canonical_storage(protocol):
    key = 'input' if protocol == 'openai-responses' else 'messages'
    first = [{'role': 'user', 'content': application_turn_marker('old')+'小町对星星很感兴趣'},
        {'type':'function_call', 'call_id':'old-call', 'name':'read_context', 'arguments':'{}'},
        {'type':'function_call_output', 'call_id':'old-call', 'output':'{}'},
        {'role':'assistant', 'content':'观察星空'}]
    second = [{'role':'user', 'content':application_turn_marker('large')+'旧资料'*2000},
        {'role':'assistant', 'content':'已闭合'}]
    current = [{'role':'user', 'content':application_turn_marker('now')+'新的讨论'}]
    body = {key:[{'role':'system', 'content':'小町人格'}, *first, *second, *current], 'tools':[]}
    policy = ObservationContextPolicy(RequestContextPolicy(), budget=300)
    projected, metrics = policy.project(body, protocol=protocol, current_turn_id='now', current_call_ids=set())
    assert projected[key] == [body[key][0], *first, *current]
    assert body[key] == [body[key][0], *first, *second, *current]
    assert metrics['patrol_omitted_history_items'] == 2
    assert metrics['patrol_anchor_tokens'] <= 300
    assert estimate_request_tokens(projected) < estimate_request_tokens(body)


def test_dynamic_slice_bounds_large_sources_without_modifying_original():
    from app.agent.context_policy import estimate_text_tokens
    source = ('长消息'*4000,)
    projected = observation_context(source)
    assert estimate_text_tokens('\n'.join(projected)) <= 1500
    assert source[0] == '长消息'*4000


def test_offered_is_not_evaluated_and_cannot_overwrite_explicit_defer(sqlite_engine):
    repo = SocialAgentRepository(sqlite_engine)
    posts = tuple(SocialPost(900000201+n, f'synthetic-{n}', '新动态', NOW) for n in range(3))
    repo.assess_post(account_id=ACCOUNT, post=posts[0], status='deferred', now=NOW, minutes=90)
    repo.note_posts_offered(account_id=ACCOUNT, posts=posts, now=NOW)
    with session_scope(sqlite_engine) as session:
        states = list(session.query(SocialPatrolCandidate).all())
        assert sorted(row.status for row in states) == ['deferred','offered','offered']
    later = NOW + timedelta(minutes=31)
    assert not repo.unreviewed_posts(account_id=ACCOUNT, posts=posts, now=later)
    assert repo.unreviewed_posts(account_id=ACCOUNT, posts=posts, now=NOW+timedelta(minutes=91)) == posts[:1]
    assert repo.unreviewed_posts(account_id=ACCOUNT+1, posts=posts, now=NOW) == posts
    changed = replace(posts[0], text='更新了内容')
    assert repo.unreviewed_posts(account_id=ACCOUNT, posts=(changed,), now=NOW) == (changed,)


def test_fenced_json_still_validates_real_target():
    from app.agent.participation_judge import parse_decision
    assert parse_decision('```json\n{"decision":"participate","source_id":3,"reason":"natural_hook"}\n```', {3}) == (3, 'natural_hook')
    assert parse_decision('```json\n{"decision":"participate","source_id":9,"reason":"natural_hook"}\n```', {3})[0] is None


def test_routine_wait_does_not_become_a_personality_preference():
    from app.agent.patrol_context import routine_wait
    assert routine_wait({'goal':'空间例行巡检', 'progress':'刚才选择安静', 'next_step':'等待下次统一巡检唤醒'})
    assert not routine_wait({'goal':'核对木星照片', 'next_step':'等待下次原帖更新后继续核对'})
    assert not routine_wait({'goal':'空间例行巡检后画猫', 'next_step':'等待下次提供参考图'})


async def test_unsent_observation_is_cancelled_without_cancelling_its_consumer():
    from app.agent.patrol_context import owner_work
    gate, started, cancelled = OwnerTurnGate(), asyncio.Event(), asyncio.Event()
    async def slow_read():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    async def observe():
        async with gate.turn(parent(), direct=False) as guard:
            return await owner_work(slow_read(), guard, interruptible=True)
    task = asyncio.create_task(observe())
    await started.wait()
    async with asyncio.timeout(1):
        async with gate.turn(parent(), direct=True):
            assert cancelled.is_set()
    result = await task
    assert result.status == 'blocked' and result.reason == 'patrol_superseded'


def test_compact_prompt_keeps_persona_and_authoritative_participation_feedback_rules():
    from app.agent.patrol_context import observation_prompt
    from app.agent.participation_feedback import PROACTIVE_PARTICIPATION_RULES
    from app.agent.context_policy import serious_request_policy
    identity = 'System persona: 当前人格:喜欢星空的小町'
    feedback = 'Group policy: ' + PROACTIVE_PARTICIPATION_RULES
    projected = observation_prompt([identity, feedback, 'Group policy: '+serious_request_policy(),
        'Current observation JSON:\n{}'])
    assert identity in projected and feedback in projected
    assert not any(serious_request_policy() in line for line in projected)
    assert any('不能接梗反击' in line for line in projected)
