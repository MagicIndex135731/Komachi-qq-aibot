"""Every addressed request retains its own chronological inbox turn."""
import asyncio
from dataclasses import replace

import pytest
from sqlalchemy import select

from app.agent.contracts import AgentActionResult, AgentBatch, GroupAgentOptions
from app.agent.participation import GroupParticipationPlanner
from app.agent.runtime import GroupAgentRuntime
from app.storage.db import session_scope
from app.storage.group_agent import GroupAgentInbox, GroupAgentRepository, GroupAgentTurn
from test_participation import FakeGateway, FakeLlm, event, router_for


@pytest.mark.asyncio
@pytest.mark.parametrize('at_first', [False, True])
async def test_addressed_target_is_first_in_message_order_not_at_priority(sqlite_engine, tmp_path, at_first):
    llm = FakeLlm()
    router = router_for(sqlite_engine, tmp_path, llm, FakeGateway())
    named = replace(event('named-request', user_id=20002), plain_text='小町你在吗？')
    addressed = replace(event('real-at', user_id=20003, mentioned=True), plain_text='现在几点？')
    events = (addressed, named) if at_first else (named, addressed)
    for incoming in events:
        router.ingest_live_group_message(incoming)
    planner = GroupParticipationPlanner(router=router, decision_client=llm)
    try:
        decision = await planner.decide(AgentBatch(10001, events, 'chronological-target'))
        assert decision.reason == 'direct_trigger' and decision.target_message_id == events[0].platform_msg_id
        assert llm.prompts == []
    finally:
        planner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('first_status', ['sent', 'uncertain'])
async def test_named_request_then_at_are_both_consumed_once_in_order(sqlite_engine, tmp_path, first_status):
    llm = FakeLlm()
    router = router_for(sqlite_engine, tmp_path, llm, FakeGateway())
    named = replace(event('named-request', user_id=20002), plain_text='小町你在吗？')
    addressed = replace(event('real-at', user_id=20003, mentioned=True), plain_text='现在几点？')
    for incoming in (named, addressed):
        router.ingest_live_group_message(incoming)
    planner = GroupParticipationPlanner(router=router, decision_client=llm)
    completed = []

    async def execute(target, batch, decision, guard):
        guard.on_dispatch()
        completed.append((target.platform_msg_id, tuple(e.platform_msg_id for e in batch.events)))
        status = first_status if len(completed) == 1 else 'sent'
        return AgentActionResult(status, ('receipt-' + target.platform_msg_id,))

    runtime = GroupAgentRuntime(engine=sqlite_engine, bot_qq=router.runtime.settings.bot_qq, bot_name='小町',
        decide=planner.decide, execute=execute,
        options=GroupAgentOptions(quiet_seconds=0.02, max_batch_wait_seconds=0.05))
    runtime.set_connected(True)
    await runtime.start()
    try:
        await runtime.observe(named)
        await runtime.observe(addressed)
        async with asyncio.timeout(2):
            while len(completed) < 2:
                await asyncio.sleep(0.005)
        assert completed == [('named-request', ('named-request',)), ('real-at', ('real-at',))]
        with session_scope(sqlite_engine) as session:
            inbox = list(session.scalars(select(GroupAgentInbox).order_by(GroupAgentInbox.id)))
            turns = list(session.scalars(select(GroupAgentTurn).order_by(GroupAgentTurn.created_at)))
            assert [item.status for item in inbox] == ['done', 'done']
            assert [turn.status for turn in turns] == [first_status, 'sent']
            assert [turn.source_message_ids for turn in turns] == [[inbox[0].source_message_id], [inbox[1].source_message_id]]
            assert not await runtime.observe(named) and not await runtime.observe(addressed)
    finally:
        await runtime.stop()
        planner.close()


@pytest.mark.parametrize('dispatched', [False, True])
def test_narrowed_turn_restart_keeps_unanswered_suffix_and_never_replays_sent_prefix(sqlite_engine, dispatched):
    from test_group_agent_runtime import BOT, GROUP, persist

    repository = GroupAgentRepository(sqlite_engine)
    first = persist(sqlite_engine, 'first-name', text='Komachi你在吗？')
    second = persist(sqlite_engine, 'second-at', mentioned=True, text='现在几点？')
    for incoming in (first, second):
        repository.enqueue(group_id=GROUP, platform_msg_id=incoming.platform_msg_id, bot_qq=BOT, max_pending=20)
    state = repository.begin_session(GROUP)
    claimed = repository.claim(group_id=GROUP, epoch=state.epoch, max_messages=20, max_age_seconds=180, bot_qq=BOT)
    assert len(claimed.events) == 2
    assert repository.retain_through_target(turn_id=claimed.turn_id, group_id=GROUP, epoch=state.epoch,
        target_message_id=first.platform_msg_id) == 1
    if dispatched:
        repository.record_decision(turn_id=claimed.turn_id, action='reply', reason='direct_trigger')
        assert repository.mark_dispatching(turn_id=claimed.turn_id, group_id=GROUP, epoch=state.epoch)
    repository.recover()
    resumed_state = repository.get_session(GROUP)
    resumed = repository.claim(group_id=GROUP, epoch=resumed_state.epoch, max_messages=20, max_age_seconds=180, bot_qq=BOT)
    expected = ['second-at'] if dispatched else ['first-name', 'second-at']
    assert [e.platform_msg_id for e in resumed.events] == expected
    if dispatched:
        assert repository.turn_status(claimed.turn_id) == 'uncertain'
    with session_scope(sqlite_engine) as session:
        inbox = list(session.scalars(select(GroupAgentInbox).where(GroupAgentInbox.turn_id == resumed.turn_id)
            .order_by(GroupAgentInbox.id)))
        turn = session.get(GroupAgentTurn, resumed.turn_id)
        assert turn.source_message_ids == [item.source_message_id for item in inbox]


@pytest.mark.parametrize('invalid', ['stale_epoch', 'wrong_group', 'missing_target', 'already_dispatching'])
def test_narrowing_rejects_stale_or_unowned_targets_without_releasing_suffix(sqlite_engine, invalid):
    from test_group_agent_runtime import BOT, GROUP, persist

    repository = GroupAgentRepository(sqlite_engine)
    first = persist(sqlite_engine, 'first-name')
    second = persist(sqlite_engine, 'second-at', mentioned=True)
    for incoming in (first, second):
        repository.enqueue(group_id=GROUP, platform_msg_id=incoming.platform_msg_id, bot_qq=BOT, max_pending=20)
    state = repository.begin_session(GROUP)
    claimed = repository.claim(group_id=GROUP, epoch=state.epoch, max_messages=20, max_age_seconds=180, bot_qq=BOT)
    if invalid == 'already_dispatching':
        repository.record_decision(turn_id=claimed.turn_id, action='reply', reason='direct_trigger')
        assert repository.mark_dispatching(turn_id=claimed.turn_id, group_id=GROUP, epoch=state.epoch)
    with pytest.raises(ValueError):
        repository.retain_through_target(turn_id=claimed.turn_id,
            group_id=GROUP + 1 if invalid == 'wrong_group' else GROUP,
            epoch=state.epoch + 1 if invalid == 'stale_epoch' else state.epoch,
            target_message_id='missing' if invalid == 'missing_target' else first.platform_msg_id)
    with session_scope(sqlite_engine) as session:
        inbox = list(session.scalars(select(GroupAgentInbox).order_by(GroupAgentInbox.id)))
        assert [item.status for item in inbox] == ['processing', 'processing']
        assert len(session.get(GroupAgentTurn, claimed.turn_id).source_message_ids) == 2
