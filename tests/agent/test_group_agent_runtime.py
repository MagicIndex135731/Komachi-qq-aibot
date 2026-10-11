from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.adapters.onebot_models import GroupMessageEvent, parse_group_message_event
from app.agent.contracts import AgentActionResult, AgentDecision, AgentStoppedError, GroupAgentOptions
from app.agent.runtime import GroupAgentRuntime
from app.storage.db import session_scope
from app.storage.group_agent import GroupAgentInbox, GroupAgentRepository, GroupAgentSession, GroupAgentTurn
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository


BOT = 900000999
GROUP = 900000001


def persist(engine, msg_id="m-1", *, group_id=GROUP, user_id=900000101, text="hello", age=0, mentioned=False):
    at = datetime.now(UTC) - timedelta(seconds=age)
    message = [{"type": "text", "data": {"text": text}}]
    if mentioned:
        message.insert(0, {"type": "at", "data": {"qq": str(BOT)}})
    payload = {
        "post_type": "message", "message_type": "group", "group_id": group_id,
        "message_id": msg_id, "user_id": user_id, "time": at.timestamp(),
        "sender": {"nickname": f"member-{user_id}", "card": ""}, "message": message,
    }
    event = parse_group_message_event(payload, bot_qq=BOT, bot_name="Komachi")
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(group_id=group_id, group_name="synthetic", enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=user_id, nickname=f"member-{user_id}", group_card="")
        MessageRepository(session).add_group_message(
            platform_msg_id=msg_id, group_id=group_id, user_id=user_id, timestamp=at,
            plain_text=text, raw_json=payload, msg_type="text", reply_to_msg_id=None,
            mentioned_bot=mentioned,
        )
    return event


def options(**kwargs):
    baseline = GroupAgentOptions(
        quiet_seconds=0.015, max_batch_wait_seconds=0.05,
        decision_timeout_seconds=1, execution_timeout_seconds=1,
        shutdown_timeout_seconds=0.2, max_wait_seconds=0.04,
    )
    return replace(baseline, **kwargs)


async def eventually(predicate, timeout=1.5):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.003)


def make_runtime(engine, decide, execute, **kwargs):
    return GroupAgentRuntime(
        engine=engine, bot_qq=BOT, bot_name="Komachi", decide=decide,
        execute=execute, options=options(**kwargs),
    )


@pytest.mark.asyncio
async def test_batches_keep_real_authors_and_persist_note_receipts(sqlite_engine):
    batches, sent = [], []

    async def decide(batch):
        batches.append(batch)
        return AgentDecision("reply", batch.events[-1].platform_msg_id, working_note="awaiting a response")

    async def execute(event, batch, decision, guard):
        guard.on_dispatch()
        sent.append((event.platform_msg_id, batch.turn_id))
        return AgentActionResult("sent", ("qq-receipt-1",))

    runtime = make_runtime(sqlite_engine, decide, execute)
    runtime.set_connected(True)
    await runtime.start()
    try:
        one = persist(sqlite_engine, "batch-1", user_id=900000101)
        two = persist(sqlite_engine, "batch-2", user_id=900000102)
        await runtime.observe(one)
        await runtime.observe(two)
        await eventually(lambda: len(sent) == 1)
        assert len(batches) == 1
        assert [event.user_id for event in batches[0].events] == [900000101, 900000102]
        assert [event.platform_msg_id for event in batches[0].events] == ["batch-1", "batch-2"]
        state = runtime.repository.get_session(GROUP)
        assert state.working_note == "awaiting a response"
        assert state.last_sent_at is not None
        with session_scope(sqlite_engine) as session:
            turn = session.get(GroupAgentTurn, sent[0][1])
            assert turn.status == "sent"
            assert turn.receipt_ids == ["qq-receipt-1"]
            assert "hello" not in str(turn.source_message_ids)
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_duplicate_observation_and_restart_do_not_repeat_completed_turn(sqlite_engine):
    calls = []

    async def decide(batch):
        calls.append(batch.turn_id)
        return AgentDecision("silent")

    async def execute(*args):
        raise AssertionError("silent turn must not execute")

    event = persist(sqlite_engine)
    runtime = make_runtime(sqlite_engine, decide, execute)
    runtime.set_connected(True)
    await runtime.start()
    assert await runtime.observe(event)
    assert not await runtime.observe(event)
    await eventually(lambda: len(calls) == 1 and runtime.repository.turn_status(calls[0]) == "silent")
    await runtime.stop()
    other = make_runtime(sqlite_engine, decide, execute)
    other.set_connected(True)
    await other.start()
    try:
        assert not await other.observe(event)
        await asyncio.sleep(0.04)
        assert len(calls) == 1
    finally:
        await other.stop()

async def test_owner_directive_receipt_is_saved_without_refreshing_proactive_cadence(sqlite_engine):
    async def decide(batch): return AgentDecision('reply',batch.events[-1].platform_msg_id)
    async def execute(event,batch,decision,guard):
        guard.on_dispatch()
        return AgentActionResult('sent',('synthetic-owner-receipt',),charge_proactive_cadence=False)
    runtime=make_runtime(sqlite_engine,decide,execute)
    runtime.set_connected(True); await runtime.start()
    try:
        await runtime.observe(persist(sqlite_engine,'owner-cadence-exempt'))
        await eventually(lambda: runtime.repository.get_session(GROUP).cursor>0)
        assert runtime.repository.get_session(GROUP).last_sent_at is None
        with session_scope(sqlite_engine) as db:
            turn=db.scalar(select(GroupAgentTurn).where(GroupAgentTurn.group_id==GROUP))
            assert turn.status=='sent' and turn.receipt_ids==['synthetic-owner-receipt']
    finally: await runtime.stop()


@pytest.mark.asyncio
async def test_scope_and_self_echo_cannot_enter_the_agent_inbox(sqlite_engine):
    async def unexpected(*args):
        raise AssertionError("out-of-scope event invoked a model")

    runtime = make_runtime(sqlite_engine, unexpected, unexpected)
    runtime.set_connected(True)
    await runtime.start()
    try:
        normal = persist(sqlite_engine)
        self_echo = persist(sqlite_engine, "bot-echo", user_id=BOT)
        assert not await runtime.observe(replace(normal, group_id=GROUP + 1))
        assert not await runtime.observe(self_echo)
        with session_scope(sqlite_engine) as session:
            assert list(session.scalars(select(GroupAgentInbox))) == []
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_one_consumer_per_group_and_global_model_concurrency(sqlite_engine):
    entered, active = [], set()
    release = asyncio.Event()
    finished = []

    async def decide(batch):
        assert batch.group_id not in active
        active.add(batch.group_id)
        entered.append(batch.group_id)
        await release.wait()
        active.remove(batch.group_id)
        finished.append(batch.group_id)
        return AgentDecision("silent")

    async def execute(*args):
        raise AssertionError("silent")

    runtime = make_runtime(sqlite_engine, decide, execute, model_concurrency=2)
    runtime.set_connected(True)
    await runtime.start()
    try:
        await runtime.observe(persist(sqlite_engine, "concurrent-1"))
        await runtime.observe(persist(sqlite_engine, "concurrent-2", group_id=GROUP + 1))
        await eventually(lambda: len(entered) == 2)
        assert len(active) == 2
        await runtime.observe(persist(sqlite_engine, "concurrent-3"))
        await asyncio.sleep(0.03)
        assert len(entered) == 2
        release.set()
        await eventually(lambda: len(finished) == 3)
        assert entered.count(GROUP) == 2
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_restart_recovers_only_before_dispatch(sqlite_engine):
    entered = asyncio.Event()
    turn_ids = []

    async def interrupted_decide(batch):
        turn_ids.append(batch.turn_id)
        entered.set()
        await asyncio.Event().wait()

    async def unexpected(*args):
        raise AssertionError("not dispatched")

    event = persist(sqlite_engine, "pre-dispatch")
    first = make_runtime(sqlite_engine, interrupted_decide, unexpected)
    first.set_connected(True)
    await first.start()
    await first.observe(event)
    await entered.wait()
    await first.stop()
    assert first.repository.turn_status(turn_ids[0]) == "interrupted"
    completed = []

    async def resumed_decide(batch):
        completed.append(batch.turn_id)
        return AgentDecision("silent")

    second = make_runtime(sqlite_engine, resumed_decide, unexpected)
    second.set_connected(True)
    await second.start()
    try:
        await eventually(lambda: completed and second.repository.turn_status(completed[0]) == "silent")
        assert completed == turn_ids
    finally:
        await second.stop()


@pytest.mark.asyncio
async def test_stop_after_dispatch_records_unknown_and_never_replays(sqlite_engine):
    dispatched = asyncio.Event()
    turns, guards = [], []

    async def decide(batch):
        turns.append(batch.turn_id)
        return AgentDecision("reply", batch.events[0].platform_msg_id)

    async def execute(event, batch, decision, guard):
        guards.append(guard)
        guard.on_dispatch()
        dispatched.set()
        await asyncio.Event().wait()

    first = make_runtime(sqlite_engine, decide, execute)
    first.set_connected(True)
    await first.start()
    await first.observe(persist(sqlite_engine, "unknown-send"))
    await dispatched.wait()
    await first.stop()
    assert first.repository.turn_status(turns[0]) == "uncertain"
    assert not guards[0].valid()
    with pytest.raises(AgentStoppedError):
        guards[0].on_dispatch()

    async def unexpected(*args):
        raise AssertionError("unknown action replayed")

    second = make_runtime(sqlite_engine, unexpected, unexpected)
    second.set_connected(True)
    await second.start()
    await asyncio.sleep(0.03)
    await second.stop()
    assert len(turns) == 1


def test_crash_recovery_dispatch_marker_fences_old_epoch(sqlite_engine):
    event = persist(sqlite_engine, "crash-dispatch")
    repository = GroupAgentRepository(sqlite_engine)
    repository.enqueue(group_id=GROUP, platform_msg_id=event.platform_msg_id, bot_qq=BOT, max_pending=20)
    state = repository.begin_session(GROUP)
    turn = repository.claim(group_id=GROUP, epoch=state.epoch, max_messages=20, max_age_seconds=180)
    repository.record_decision(turn_id=turn.turn_id, action="reply", reason="natural")
    assert repository.mark_dispatching(turn_id=turn.turn_id, group_id=GROUP, epoch=state.epoch)
    assert repository.recover() == ()
    assert repository.turn_status(turn.turn_id) == "uncertain"
    assert not repository.mark_dispatching(turn_id=turn.turn_id, group_id=GROUP, epoch=state.epoch)


@pytest.mark.asyncio
async def test_disconnection_invalidates_a_late_model_result_even_after_reconnect(sqlite_engine):
    entered, release = asyncio.Event(), asyncio.Event()
    calls, sent = [], []

    async def decide(batch):
        calls.append(batch.turn_id)
        if len(calls) == 1:
            entered.set()
            await release.wait()
        return AgentDecision("reply", batch.events[-1].platform_msg_id)

    async def execute(event, batch, decision, guard):
        guard.on_dispatch()
        sent.append(event.platform_msg_id)
        return AgentActionResult("sent", ("new-connection-receipt",))

    runtime = make_runtime(sqlite_engine, decide, execute)
    await runtime.start()
    try:
        await runtime.observe(persist(sqlite_engine, "connection-turn"))
        await asyncio.sleep(0.03)
        assert calls == []
        runtime.set_connected(True)
        await entered.wait()
        runtime.set_connected(False)
        runtime.set_connected(True)
        release.set()
        await eventually(lambda: len(sent) == 1)
        assert len(calls) == 2
        assert sent == ["connection-turn"]
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_invalid_target_and_malformed_decision_are_silent(sqlite_engine):
    calls = []

    async def decide(batch):
        calls.append(batch.turn_id)
        return AgentDecision("reply", "other-group-source") if len(calls) == 1 else {"action": "reply"}

    async def unexpected(*args):
        raise AssertionError("invalid decision executed")

    runtime = make_runtime(sqlite_engine, decide, unexpected)
    runtime.set_connected(True)
    await runtime.start()
    try:
        await runtime.observe(persist(sqlite_engine, "bad-target"))
        await eventually(lambda: calls and runtime.repository.turn_status(calls[0]) == "silent")
        await runtime.observe(persist(sqlite_engine, "bad-contract"))
        await eventually(lambda: len(calls) == 2 and runtime.repository.turn_status(calls[1]) == "silent")
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_wait_does_not_resample_idle_batch_but_resumes_for_new_messages(sqlite_engine):
    calls = []

    async def decide(batch):
        calls.append(batch)
        return AgentDecision("wait", working_note="speaker may continue", wait_seconds=0.04)

    async def unexpected(*args):
        raise AssertionError("wait must not send")

    runtime = make_runtime(sqlite_engine, decide, unexpected)
    runtime.set_connected(True)
    await runtime.start()
    try:
        await runtime.observe(persist(sqlite_engine, "wait-1"))
        await eventually(lambda: len(calls) == 1)
        await runtime.observe(persist(sqlite_engine, "wait-2"))
        await eventually(lambda: len(calls) == 2 and runtime.repository.turn_status(calls[1].turn_id) == "silent")
        assert calls[1].working_note == "speaker may continue"
        await asyncio.sleep(0.06)
        assert len(calls) == 2
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_stale_canonical_message_never_becomes_new_autonomous_activity(sqlite_engine):
    async def unexpected(*args):
        raise AssertionError("stale source activated a model")

    runtime = make_runtime(sqlite_engine, unexpected, unexpected, max_message_age_seconds=20)
    runtime.set_connected(True)
    await runtime.start()
    try:
        await runtime.observe(persist(sqlite_engine, "stale-source", age=120))
        await eventually(lambda: not runtime.repository.has_pending(GROUP))
        with session_scope(sqlite_engine) as session:
            assert session.scalar(select(GroupAgentInbox.status)) == "expired"
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_consumer_limit_defers_other_group_without_losing_inbox(sqlite_engine):
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def decide(batch):
        seen.append(batch.group_id)
        if len(seen) == 1:
            entered.set()
            await release.wait()
        return AgentDecision("silent")

    async def unexpected(*args):
        raise AssertionError("silent")

    runtime = make_runtime(sqlite_engine, decide, unexpected, max_active_groups=1)
    runtime.set_connected(True)
    await runtime.start()
    try:
        assert await runtime.observe(persist(sqlite_engine, "capacity-1"))
        await entered.wait()
        assert await runtime.observe(persist(sqlite_engine, "capacity-2", group_id=GROUP + 1))
        assert seen == [GROUP]
        release.set()
        await eventually(lambda: seen == [GROUP, GROUP + 1])
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_timeout_before_dispatch_does_not_retry_forever(sqlite_engine):
    turns = []

    async def decide(batch):
        turns.append(batch.turn_id)
        await asyncio.Event().wait()

    async def unexpected(*args):
        raise AssertionError("not decided")

    runtime = make_runtime(sqlite_engine, decide, unexpected, decision_timeout_seconds=0.025)
    runtime.set_connected(True)
    await runtime.start()
    try:
        await runtime.observe(persist(sqlite_engine, "model-timeout"))
        await eventually(lambda: turns and runtime.repository.turn_status(turns[0]) == "failed")
        await asyncio.sleep(0.04)
        assert len(turns) == 1
    finally:
        await runtime.stop()


def test_registration_is_repeatable_and_working_state_does_not_duplicate_raw_messages(sqlite_engine):
    from app.storage.db import create_all
    create_all(sqlite_engine)
    create_all(sqlite_engine)
    event = persist(sqlite_engine, "inbox-canonical")
    repository = GroupAgentRepository(sqlite_engine)
    assert repository.enqueue(group_id=GROUP, platform_msg_id=event.platform_msg_id, bot_qq=BOT, max_pending=2)
    assert not repository.enqueue(group_id=GROUP, platform_msg_id=event.platform_msg_id, bot_qq=BOT, max_pending=2)
    with session_scope(sqlite_engine) as session:
        inbox = session.scalar(select(GroupAgentInbox))
        assert inbox.source_message_id > 0
        assert not hasattr(inbox, "plain_text")
        assert not hasattr(inbox, "raw_json")
        assert session.get(GroupAgentSession, GROUP) is None


@pytest.mark.asyncio
async def test_late_executor_cannot_send_after_bounded_stop(sqlite_engine):
    entered, release = asyncio.Event(), asyncio.Event()
    guards, sent = [], []

    async def decide(batch):
        return AgentDecision("reply", batch.events[0].platform_msg_id)

    async def execute(event, batch, decision, guard):
        guards.append(guard)
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # Simulate a callback that cannot stop its synchronous provider yet.
            await release.wait()
        guard.on_dispatch()
        sent.append(event.platform_msg_id)
        return AgentActionResult("sent")

    runtime = make_runtime(sqlite_engine, decide, execute, shutdown_timeout_seconds=0.015)
    runtime.set_connected(True)
    await runtime.start()
    await runtime.observe(persist(sqlite_engine, "late-executor"))
    await entered.wait()
    await runtime.stop()
    assert not guards[0].valid()
    release.set()
    await eventually(lambda: runtime.repository.turn_status(guards[0].turn_id) == "interrupted")
    assert sent == []


def test_recalled_source_is_revalidated_at_dispatch_and_old_finish_cannot_change_new_turn(sqlite_engine):
    source = persist(sqlite_engine, "source-recalled")
    repository = GroupAgentRepository(sqlite_engine)
    repository.enqueue(group_id=GROUP, platform_msg_id=source.platform_msg_id, bot_qq=BOT, max_pending=20)
    state = repository.begin_session(GROUP)
    claimed = repository.claim(group_id=GROUP, epoch=state.epoch, max_messages=20, max_age_seconds=180)
    repository.record_decision(turn_id=claimed.turn_id, action="reply", reason="participant")
    with session_scope(sqlite_engine) as session:
        MessageRepository(session).mark_group_message_deleted(group_id=GROUP, platform_msg_id=source.platform_msg_id)
    assert not repository.mark_dispatching(turn_id=claimed.turn_id, group_id=GROUP, epoch=state.epoch)
    repository.finish(turn_id=claimed.turn_id, status="interrupted", epoch=state.epoch)
    repository.recover()
    newer = repository.begin_session(GROUP)
    with session_scope(sqlite_engine) as session:
        turn = session.get(GroupAgentTurn, claimed.turn_id)
        turn.status = "deciding"
        turn.epoch = newer.epoch
    repository.finish(turn_id=claimed.turn_id, status="failed", epoch=state.epoch)
    assert repository.turn_status(claimed.turn_id) == "deciding"


@pytest.mark.asyncio
async def test_global_budget_serializes_groups_when_one_model_slot(sqlite_engine):
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def decide(batch):
        seen.append(batch.group_id)
        if len(seen) == 1:
            entered.set()
            await release.wait()
        return AgentDecision("silent")

    async def unexpected(*args):
        raise AssertionError("silent")

    runtime = make_runtime(sqlite_engine, decide, unexpected, model_concurrency=1)
    runtime.set_connected(True)
    await runtime.start()
    try:
        await runtime.observe(persist(sqlite_engine, "one-slot-1"))
        await runtime.observe(persist(sqlite_engine, "one-slot-2", group_id=GROUP + 1))
        await entered.wait()
        await asyncio.sleep(0.03)
        assert len(seen) == 1
        release.set()
        await eventually(lambda: len(seen) == 2)
    finally:
        await runtime.stop()


def test_pending_overflow_expires_only_inbox_references_and_preserves_canonical_rows(sqlite_engine):
    repository = GroupAgentRepository(sqlite_engine)
    for index in range(4):
        event = persist(sqlite_engine, f"overflow-{index}")
        repository.enqueue(group_id=GROUP, platform_msg_id=event.platform_msg_id, bot_qq=BOT, max_pending=2)
    with session_scope(sqlite_engine) as session:
        inbox = list(session.scalars(select(GroupAgentInbox).order_by(GroupAgentInbox.id)))
        assert [item.status for item in inbox] == ["expired", "expired", "pending", "pending"]
        assert MessageRepository(session).count_group_messages(GROUP) == 4


def test_options_reject_unbounded_or_negative_runtime_settings():
    with pytest.raises(ValueError):
        GroupAgentOptions(model_concurrency=0)
    with pytest.raises(ValueError):
        GroupAgentOptions(quiet_seconds=-1)
    with pytest.raises(ValueError):
        GroupAgentOptions(working_note_max_sources=0)


@pytest.mark.asyncio
async def test_two_mentions_in_one_window_are_processed_as_two_real_requests(sqlite_engine):
    batches, replies = [], []

    async def decide(batch):
        batches.append(batch)
        target = next(event for event in batch.events if event.mentioned_bot)
        return AgentDecision("reply", target.platform_msg_id, "direct_trigger")

    async def execute(event, batch, decision, guard):
        guard.on_dispatch()
        replies.append((event.platform_msg_id, event.user_id))
        return AgentActionResult("sent", (f"receipt-{len(replies)}",))

    runtime = make_runtime(sqlite_engine, decide, execute)
    runtime.set_connected(True)
    await runtime.start()
    try:
        await runtime.observe(persist(sqlite_engine, "mention-1", user_id=900000101, mentioned=True))
        await runtime.observe(persist(sqlite_engine, "mention-2", user_id=900000102, mentioned=True))
        await eventually(lambda: len(replies) == 2)
        assert replies == [("mention-1", 900000101), ("mention-2", 900000102)]
        assert [[event.platform_msg_id for event in batch.events] for batch in batches] == [["mention-1"], ["mention-2"]]
    finally:
        await runtime.stop()


def test_working_note_can_be_cleared_and_expires_across_restart(sqlite_engine):
    repository = GroupAgentRepository(sqlite_engine)
    event = persist(sqlite_engine, "note-source")
    repository.enqueue(group_id=GROUP, platform_msg_id=event.platform_msg_id, bot_qq=BOT, max_pending=20)
    state = repository.begin_session(GROUP)
    turn = repository.claim(group_id=GROUP, epoch=state.epoch, max_messages=20, max_age_seconds=180)
    repository.finish(turn_id=turn.turn_id, status="silent", working_note="temporary old topic")
    with session_scope(sqlite_engine) as session:
        state_row = session.get(GroupAgentSession, GROUP)
        state_row.working_note_updated_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
    repository.recover()  # Recovery updates epoch but must not refresh the note's age.
    assert repository.begin_session(GROUP).working_note == ""
    next_event = persist(sqlite_engine, "note-clear-source")
    repository.enqueue(group_id=GROUP, platform_msg_id=next_event.platform_msg_id, bot_qq=BOT, max_pending=20)
    state = repository.begin_session(GROUP)
    turn = repository.claim(group_id=GROUP, epoch=state.epoch, max_messages=20, max_age_seconds=180)
    with session_scope(sqlite_engine) as session:
        session.get(GroupAgentSession, GROUP).working_note = "note must clear"
    repository.finish(turn_id=turn.turn_id, status="silent", working_note="")
    assert repository.get_session(GROUP).working_note == ""


def test_prune_preserves_unknown_actions_and_their_deduplication_reference(sqlite_engine):
    repository = GroupAgentRepository(sqlite_engine)
    event = persist(sqlite_engine, "unknown-prune")
    repository.enqueue(group_id=GROUP, platform_msg_id=event.platform_msg_id, bot_qq=BOT, max_pending=20)
    state = repository.begin_session(GROUP)
    turn = repository.claim(group_id=GROUP, epoch=state.epoch, max_messages=20, max_age_seconds=180)
    repository.finish(turn_id=turn.turn_id, status="uncertain")
    repository.prune(completed_before=datetime.now(UTC) + timedelta(days=1))
    assert repository.turn_status(turn.turn_id) == "uncertain"
    assert not repository.enqueue(group_id=GROUP, platform_msg_id=event.platform_msg_id, bot_qq=BOT, max_pending=20)


@pytest.mark.parametrize("invalid_source", ["recalled", "expired", "foreign"])
def test_claim_revalidates_note_without_restarting_the_group(sqlite_engine, invalid_source):
    repository = GroupAgentRepository(sqlite_engine)
    source = persist(sqlite_engine, "old-note-source")
    repository.enqueue(group_id=GROUP, platform_msg_id=source.platform_msg_id, bot_qq=BOT, max_pending=20)
    state = repository.begin_session(GROUP)
    turn = repository.claim(group_id=GROUP, epoch=state.epoch, max_messages=20, max_age_seconds=180)
    repository.finish(turn_id=turn.turn_id, status="silent", working_note="previous topic")
    foreign = persist(sqlite_engine, "foreign-note-source", group_id=GROUP + 1)
    with session_scope(sqlite_engine) as session:
        note = session.get(GroupAgentSession, GROUP)
        if invalid_source == "recalled":
            MessageRepository(session).mark_group_message_deleted(group_id=GROUP, platform_msg_id=source.platform_msg_id)
        elif invalid_source == "expired":
            note.working_note_updated_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
        else:
            foreign_row = MessageRepository(session).get_by_platform_msg_id(foreign.platform_msg_id)
            note.working_note_source_message_ids = [foreign_row.id]
    fresh = persist(sqlite_engine, "fresh-note-source")
    repository.enqueue(group_id=GROUP, platform_msg_id=fresh.platform_msg_id, bot_qq=BOT, max_pending=20)
    next_turn = repository.claim(group_id=GROUP, epoch=state.epoch, max_messages=20, max_age_seconds=180)
    assert next_turn.session.working_note == ""
    with session_scope(sqlite_engine) as session:
        note = session.get(GroupAgentSession, GROUP)
        assert note.working_note_source_message_ids == []
        assert note.working_note_updated_at is None


def test_working_note_source_overflow_clears_content_and_all_provenance(sqlite_engine):
    repository = GroupAgentRepository(sqlite_engine)
    state = repository.begin_session(GROUP)
    for index in range(4):
        event = persist(sqlite_engine, f"note-limit-{index}")
        repository.enqueue(group_id=GROUP, platform_msg_id=event.platform_msg_id, bot_qq=BOT, max_pending=20)
        turn = repository.claim(group_id=GROUP, epoch=state.epoch, max_messages=20, max_age_seconds=180)
        repository.finish(
            turn_id=turn.turn_id, status="silent", working_note="all previous sources may contribute",
            working_note_max_sources=3,
        )
        with session_scope(sqlite_engine) as session:
            note = session.get(GroupAgentSession, GROUP)
            if index < 3:
                assert note.working_note
                assert len(note.working_note_source_message_ids) == index + 1
            else:
                assert note.working_note == ""
                assert note.working_note_source_message_ids == []
                assert note.working_note_updated_at is None
            assert MessageRepository(session).get_by_platform_msg_id(event.platform_msg_id) is not None


def test_claim_clears_a_legacy_note_above_the_current_source_limit(sqlite_engine):
    repository = GroupAgentRepository(sqlite_engine)
    sources = [persist(sqlite_engine, f"legacy-note-{index}") for index in range(4)]
    state = repository.begin_session(GROUP)
    with session_scope(sqlite_engine) as session:
        row = session.get(GroupAgentSession, GROUP)
        row.working_note = "old state may reference every previous source"
        row.working_note_updated_at = datetime.now(UTC).replace(tzinfo=None)
        row.working_note_source_message_ids = [
            MessageRepository(session).get_by_platform_msg_id(source.platform_msg_id).id
            for source in sources
        ]
    fresh = persist(sqlite_engine, "legacy-note-fresh")
    repository.enqueue(group_id=GROUP, platform_msg_id=fresh.platform_msg_id, bot_qq=BOT, max_pending=20)
    claimed = repository.claim(
        group_id=GROUP, epoch=state.epoch, max_messages=20, max_age_seconds=180,
        working_note_max_sources=3,
    )
    assert claimed.session.working_note == ""
    with session_scope(sqlite_engine) as session:
        assert session.get(GroupAgentSession, GROUP).working_note_source_message_ids == []


@pytest.mark.asyncio
async def test_paused_worker_admission_is_bounded_and_restart_recovers(sqlite_engine):
    class FailingClaimRepository(GroupAgentRepository):
        failures = 0
        failing = True

        def claim(self, **kwargs):
            if self.failing and kwargs["group_id"] == GROUP:
                self.failures += 1
                raise RuntimeError("synthetic persistent repository failure")
            return super().claim(**kwargs)

    repository = FailingClaimRepository(sqlite_engine)
    completed = []

    async def decide(batch):
        completed.append(batch.group_id)
        return AgentDecision("silent")

    runtime = make_runtime(sqlite_engine, decide, decide, quiet_seconds=0)
    runtime.repository = repository
    await runtime.start()
    runtime.set_connected(True)
    try:
        await runtime.observe(persist(sqlite_engine, "worker-failure-first"))
        await eventually(lambda: GROUP in runtime._worker_paused)
        assert repository.failures == 3
        await runtime.observe(persist(sqlite_engine, "worker-failure-reset"))
        await eventually(lambda: GROUP in runtime._worker_paused)
        assert repository.failures == 6
        await runtime.observe(persist(sqlite_engine, "worker-failure-reset-exhausted"))
        await runtime.observe(persist(sqlite_engine, "other-group-finishes", group_id=GROUP + 1))
        await eventually(lambda: completed == [GROUP + 1] and not runtime.active_group_ids)
        assert repository.failures == 6
        assert repository.has_pending(GROUP)
        assert GROUP in runtime._worker_paused
        await runtime.stop()
        repository.failing = False
        await runtime.start()
        await eventually(lambda: GROUP in completed and not repository.has_pending(GROUP))
        assert repository.failures == 6
        assert GROUP not in runtime._worker_paused
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_recalling_consumed_note_source_during_model_work_revokes_dispatch(sqlite_engine):
    repository = GroupAgentRepository(sqlite_engine)
    old_source = persist(sqlite_engine, "consumed-note-source")
    repository.enqueue(group_id=GROUP, platform_msg_id=old_source.platform_msg_id, bot_qq=BOT, max_pending=20)
    state = repository.begin_session(GROUP)
    old_turn = repository.claim(group_id=GROUP, epoch=state.epoch, max_messages=20, max_age_seconds=180)
    repository.finish(turn_id=old_turn.turn_id, status="silent", working_note="context from old source")
    entered, release = asyncio.Event(), asyncio.Event()
    blocked, sent = [], []

    async def decide(batch):
        return AgentDecision("reply", batch.events[0].platform_msg_id, working_note=batch.working_note)

    async def execute(event, batch, decision, guard):
        if batch.working_note:
            assert batch.working_note == "context from old source"
            entered.set()
            await release.wait()
            try:
                guard.on_dispatch()
            except AgentStoppedError:
                blocked.append(batch.turn_id)
                raise
        else:
            guard.on_dispatch()
        sent.append((event.platform_msg_id, batch.working_note))
        return AgentActionResult("sent", ("fresh-note-receipt",))

    runtime = make_runtime(sqlite_engine, decide, execute, quiet_seconds=0)
    await runtime.start()
    runtime.set_connected(True)
    try:
        fresh = persist(sqlite_engine, "consumed-note-fresh")
        await runtime.observe(fresh)
        await entered.wait()
        with session_scope(sqlite_engine) as session:
            MessageRepository(session).mark_group_message_deleted(
                group_id=GROUP, platform_msg_id=old_source.platform_msg_id,
            )
        release.set()
        await eventually(lambda: bool(sent))
        assert len(blocked) == 1
        assert sent == [(fresh.platform_msg_id, "")]
    finally:
        release.set()
        await runtime.stop()


def test_each_burst_dispatch_revalidates_consumed_working_note_sources(sqlite_engine):
    repository = GroupAgentRepository(sqlite_engine)
    old_source = persist(sqlite_engine, "burst-note-source")
    repository.enqueue(group_id=GROUP, platform_msg_id=old_source.platform_msg_id, bot_qq=BOT, max_pending=20)
    state = repository.begin_session(GROUP)
    old_turn = repository.claim(group_id=GROUP, epoch=state.epoch, max_messages=20, max_age_seconds=180)
    repository.finish(turn_id=old_turn.turn_id, status="silent", working_note="burst context")
    fresh = persist(sqlite_engine, "burst-note-fresh")
    repository.enqueue(group_id=GROUP, platform_msg_id=fresh.platform_msg_id, bot_qq=BOT, max_pending=20)
    claimed = repository.claim(group_id=GROUP, epoch=state.epoch, max_messages=20, max_age_seconds=180)
    repository.record_decision(turn_id=claimed.turn_id, action="reply", reason="test")
    arguments = dict(
        turn_id=claimed.turn_id, group_id=GROUP, epoch=state.epoch,
        working_note_source_message_ids=claimed.session.working_note_source_message_ids,
    )
    assert repository.mark_dispatching(**arguments)
    with session_scope(sqlite_engine) as session:
        MessageRepository(session).mark_group_message_deleted(group_id=GROUP, platform_msg_id=old_source.platform_msg_id)
    assert not repository.mark_dispatching(**arguments)
    repository.finish(turn_id=claimed.turn_id, status="uncertain", epoch=state.epoch)
    assert repository.turn_status(claimed.turn_id) == "uncertain"
