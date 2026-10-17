"""Actual tool effects have independent dispatch receipts and write budgets."""
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import json
from threading import Barrier

import pytest

from app.storage.agent_effects import AgentToolEffectRepository
from app.storage.db import build_engine, create_all


ACCOUNT = 900000101
SCOPE = "group:900000001"
NOW = datetime(2026, 10, 9, 8, tzinfo=UTC)
HASH = "a" * 64


def repository(engine, *, account_id=ACCOUNT, clock=lambda: NOW):
    return AgentToolEffectRepository(engine, account_id=account_id, clock=clock)


def reserve(repo, call="c", *, operation="send_group", scope=SCOPE, run_id="synthetic-run", payload_hash=HASH):
    return repo.reserve(scope, run_id=run_id, tool_call_id=call, operation=operation,
                        payload_hash=payload_hash, sources=(1, 2))


def test_effect_reservations_are_persistent_and_idempotent(sqlite_engine):
    repo = repository(sqlite_engine)
    first = reserve(repo)
    assert first["fresh"] is True and first["status"] == "reserved"
    repeated = reserve(repository(sqlite_engine))
    assert repeated["fresh"] is False and repeated["effect_id"] == first["effect_id"]
    assert repeated["source_message_ids"] == [1, 2]
    assert reserve(repository(sqlite_engine, account_id=900000102))["effect_id"] != first["effect_id"]
    with pytest.raises(ValueError, match="identity"):
        reserve(repo, scope="group:900000002")
    with pytest.raises(ValueError, match="identity"):
        reserve(repo, payload_hash="b" * 64)


def test_initial_receipt_is_atomic_copied_and_only_applied_to_fresh_claims(sqlite_engine):
    repo = repository(sqlite_engine)
    metadata = {'asset_claim_version':2, 'artifact_ids':['art-a']}
    def claim(initial):
        return repo.reserve(SCOPE, run_id='synthetic-run', tool_call_id='c', operation='send_group',
            payload_hash=HASH, sources=(1, 2), initial_receipt=initial)
    first = claim(metadata)
    assert first['fresh'] and first['receipt'] == metadata
    metadata['artifact_ids'].append('art-b')
    assert repository(sqlite_engine).read(first['action_id'])['receipt'] == first['receipt']
    assert claim({'asset_claim_version':99})['receipt'] == first['receipt']
    repo.finish(first['action_id'], 'failed')
    repaired = claim(metadata)
    assert repaired['fresh'] and repaired['receipt'] == metadata and repaired['reservation_attempts'] == 2
    repo.mark_dispatching(first['action_id'], daily_limit=None)
    repo.finish(first['action_id'], 'uncertain', receipt={'error':'lost receipt'})
    assert claim(metadata)['receipt'] == {'error':'lost receipt'}


def test_initial_receipt_cannot_relabel_existing_legacy_reservation(sqlite_engine):
    repo = repository(sqlite_engine)
    first = reserve(repo)
    again = repo.reserve(SCOPE, run_id='synthetic-run', tool_call_id='c', operation='send_group',
        payload_hash=HASH, sources=(1, 2), initial_receipt={'asset_claim_version':2})
    assert not again['fresh'] and again['receipt'] is None and again['action_id'] == first['action_id']


def test_completed_call_is_never_dispatched_twice(sqlite_engine):
    repo = repository(sqlite_engine)
    effect = reserve(repo)
    assert repo.mark_dispatching(effect["effect_id"], daily_limit=10)
    assert not repo.mark_dispatching(effect["effect_id"], daily_limit=10)
    assert repo.finish(effect["effect_id"], status="sent", receipt={"message_id": "synthetic-platform-id"})
    assert not repo.mark_dispatching(effect["effect_id"], daily_limit=10)
    restored = reserve(repository(sqlite_engine))
    assert restored["status"] == "sent"
    assert restored["receipt"] == {"message_id": "synthetic-platform-id"}


def test_restart_recovers_dispatching_to_uncertain_without_replay(sqlite_engine):
    repo = repository(sqlite_engine)
    effect = reserve(repo)
    assert repo.mark_dispatching(effect["effect_id"])
    restarted = repository(sqlite_engine)
    assert restarted.recover(SCOPE) == 1
    assert restarted.recover(SCOPE) == 0
    assert restarted.read(effect["effect_id"])["status"] == "uncertain"
    assert not restarted.mark_dispatching(effect["effect_id"])
    assert not restarted.finish(effect["effect_id"], status="failed")


def test_same_wakeup_can_combine_different_effects_while_each_is_counted(sqlite_engine):
    repo = repository(sqlite_engine)
    send = reserve(repo, "send")
    post = reserve(repo, "post", operation="publish_qzone")
    assert repo.mark_dispatching(send["effect_id"], daily_limit=2, cooldown_seconds=60)
    assert repo.mark_dispatching(post["effect_id"], daily_limit=2, cooldown_seconds=60)
    assert repo.count_dispatched(operation="send_group") == 1
    assert repo.count_dispatched(operation="publish_qzone") == 1
    assert repo.finish(send["effect_id"], status="uncertain")
    duplicate = reserve(repo, "another-send", payload_hash="b" * 64)
    assert not repo.mark_dispatching(duplicate["effect_id"], daily_limit=2)
    assert repo.read(duplicate["effect_id"])["reason"] == "daily_scope_budget_exhausted"


def test_cooldown_is_operation_and_scope_scoped_and_shanghai_day_resets(sqlite_engine):
    clock = {"now": datetime(2026, 10, 9, 15, 59, 59, tzinfo=UTC)}
    repo = repository(sqlite_engine, clock=lambda: clock["now"])
    first = reserve(repo)
    assert repo.mark_dispatching(first["effect_id"], daily_limit=1, cooldown_seconds=10)
    assert repo.finish(first["effect_id"], "sent", receipt={"message_id": "synthetic-confirmed"})
    other_scope = reserve(repo, "other-group", scope="group:900000002")
    # Each data scope has its own actual write budget.
    assert repo.mark_dispatching(other_scope["effect_id"], daily_limit=1)
    clock["now"] += timedelta(seconds=1)
    next_day = reserve(repo, "new-day")
    assert not repo.mark_dispatching(next_day["effect_id"], daily_limit=1, cooldown_seconds=10)
    assert repo.read(next_day["effect_id"])["reason"] == "operation_cooldown"
    clock["now"] += timedelta(seconds=10)
    after = reserve(repo, "after-cooldown")
    assert repo.mark_dispatching(after["effect_id"], daily_limit=1, cooldown_seconds=10)


def test_explicit_direct_reply_exempts_social_quota_but_still_records_and_checks_unknown_effects(sqlite_engine):
    repo = repository(sqlite_engine)
    first = reserve(repo, "first-direct", payload_hash="a" * 64)
    assert repo.mark_dispatching(first["effect_id"], daily_limit=None)
    assert repo.finish(first["effect_id"], "sent", receipt="synthetic-receipt")
    second = reserve(repo, "second-direct", payload_hash="b" * 64)
    assert repo.mark_dispatching(second["effect_id"], daily_limit=None)
    assert repo.finish(second["effect_id"], "uncertain")
    assert repo.count_dispatched() == 2
    repeated = reserve(repo, "third-direct", run_id="different-direct", payload_hash="b" * 64)
    assert repeated["effect_id"] == second["effect_id"] and not repeated["fresh"]
    assert not repo.mark_dispatching(second["effect_id"], daily_limit=None)
    autonomous = reserve(repo, "autonomous-after-replies", payload_hash="c" * 64)
    assert not repo.mark_dispatching(autonomous["effect_id"], daily_limit=1)
    assert repo.read(autonomous["effect_id"])["reason"] == "daily_scope_budget_exhausted"


def test_reservation_alone_does_not_count_as_platform_attempt(sqlite_engine):
    repo = repository(sqlite_engine)
    effect = reserve(repo)
    assert repo.count_dispatched() == 0
    assert repo.finish(effect["effect_id"], status="failed", reason="cancelled_before_dispatch")
    assert not repo.mark_dispatching(effect["effect_id"])
    assert repo.count_dispatched() == 0


def test_concurrent_dispatches_obey_actual_operation_quota(tmp_path):
    path = tmp_path / "effect-quota.db"
    engine = build_engine(path)
    create_all(engine)
    repo = repository(engine)
    effects = [reserve(repo, f"call-{number}", payload_hash=f"{number:064x}")["effect_id"] for number in range(6)]
    engine.dispose()
    barrier = Barrier(6)

    def dispatch(effect):
        engine = build_engine(path)
        try:
            repo = repository(engine)
            barrier.wait(timeout=10)
            return repo.mark_dispatching(effect, daily_limit=2)
        finally:
            engine.dispose()

    with ThreadPoolExecutor(max_workers=6) as pool:
        dispatched = list(pool.map(dispatch, effects))
    assert dispatched.count(True) == 2


def test_other_account_cannot_finish_or_read_effect(sqlite_engine):
    repo = repository(sqlite_engine)
    effect = reserve(repo)
    other = repository(sqlite_engine, account_id=900000102)
    assert other.read(effect["effect_id"]) == {}
    assert not other.mark_dispatching(effect["effect_id"])
    assert not other.finish(effect["effect_id"], status="failed")


@pytest.mark.parametrize("recover", [True, False])
def test_unknown_tool_send_marks_only_own_scoped_reserved_message_uncertain(sqlite_engine, recover):
    from sqlalchemy import select
    from app.storage.db import session_scope
    from app.storage.models import Group, Message, User
    repo = repository(sqlite_engine)
    effect = reserve(repo)
    effect_id = effect["effect_id"]
    with session_scope(sqlite_engine) as session:
        session.add_all([Group(group_id=900000001), Group(group_id=900000002),
                         User(user_id=ACCOUNT), User(user_id=900000201)])
        session.flush()
        for label, group_id, user_id, state in (
            ("own", 900000001, ACCOUNT, "reserved"),
            ("other-group", 900000002, ACCOUNT, "reserved"),
            ("other-user", 900000001, 900000201, "reserved"),
            ("deleted", 900000001, ACCOUNT, "deleted"),
        ):
            session.add(Message(platform_msg_id="synthetic:" + label, group_id=group_id, user_id=user_id,
                timestamp=NOW, plain_text="synthetic output", raw_json={"social_action_id": effect_id,
                                                                         "delivery_state": state}))
    assert repo.mark_dispatching(effect_id)
    if recover:
        assert repo.recover(SCOPE) == 1
    else:
        assert repo.finish(effect_id, "uncertain", reason="synthetic timeout")
    with session_scope(sqlite_engine) as session:
        states = {message.platform_msg_id: message.raw_json["delivery_state"]
                  for message in session.scalars(select(Message))}
    assert states == {"synthetic:own": "uncertain", "synthetic:other-group": "reserved",
                      "synthetic:other-user": "reserved", "synthetic:deleted": "deleted"}
    assert repo.read(effect_id)["source_message_ids"] == [1, 2]


@pytest.mark.parametrize("limit", [True, -1, 501, "1"])
def test_effect_limits_validate_before_state_transition(sqlite_engine, limit):
    repo = repository(sqlite_engine)
    effect = reserve(repo)
    with pytest.raises(ValueError):
        repo.mark_dispatching(effect["effect_id"], daily_limit=limit)
    assert repo.read(effect["effect_id"])["status"] == "reserved"


def test_authority_revocation_before_dispatch_commit_rolls_back(sqlite_engine):
    repo = repository(sqlite_engine)
    effect = reserve(repo)

    def revoked():
        raise RuntimeError("synthetic source revoked")

    with pytest.raises(RuntimeError):
        repo.mark_dispatching(effect["effect_id"], before_commit=revoked)
    assert repo.read(effect["effect_id"])["status"] == "reserved"
    assert repo.count_dispatched() == 0


@pytest.mark.parametrize("status", ["dispatching", "uncertain"])
def test_new_wake_cannot_replay_same_unresolved_effect(sqlite_engine, status):
    repo = repository(sqlite_engine)
    first = reserve(repo)
    assert repo.mark_dispatching(first["effect_id"], daily_limit=10)
    if status == "uncertain":
        assert repo.finish(first["effect_id"], status, receipt={"error": "synthetic timeout"})
    restarted = repository(sqlite_engine)
    repeated = restarted.reserve(SCOPE, run_id="new-wake", tool_call_id="new-call",
                                 operation="send_group", payload_hash=HASH, sources=(3,))
    assert repeated["fresh"] is False and repeated["effect_id"] == first["effect_id"]
    assert repeated["status"] == status and repeated["replay_allowed"] is False
    assert repeated["source_message_ids"] == [1, 2]
    assert not restarted.mark_dispatching(repeated["effect_id"], daily_limit=10)
    assert reserve(restarted, run_id="different-args", payload_hash="b" * 64)["fresh"]
    assert reserve(restarted, run_id="different-scope", scope="group:900000002")["fresh"]
    assert reserve(restarted, run_id="different-operation", operation="publish_qzone")["fresh"]
    assert reserve(repository(sqlite_engine, account_id=900000102), run_id="other-account")["fresh"]


@pytest.mark.parametrize("status", ["sent", "failed"])
def test_known_terminal_result_does_not_globally_block_new_wake(sqlite_engine, status):
    repo = repository(sqlite_engine)
    first = reserve(repo)
    assert repo.mark_dispatching(first["effect_id"], daily_limit=10)
    assert repo.finish(first["effect_id"], status, receipt={"confirmed_status": status})
    assert reserve(repo, run_id="new-wake")["fresh"]
    assert not reserve(repo)["fresh"]


def test_separate_reservations_racing_to_dispatch_cannot_repeat_unresolved_payload(tmp_path):
    path = tmp_path / "effect-duplicate-race.db"
    engine = build_engine(path)
    create_all(engine)
    repo = repository(engine)
    effects = [reserve(repo, run_id=f"wake-{number}")["effect_id"] for number in range(6)]
    engine.dispose()
    barrier = Barrier(6)

    def dispatch(effect):
        engine = build_engine(path)
        try:
            barrier.wait(timeout=10)
            return repository(engine).mark_dispatching(effect, daily_limit=10)
        finally:
            engine.dispose()

    with ThreadPoolExecutor(max_workers=6) as pool:
        dispatched = list(pool.map(dispatch, effects))
    assert dispatched.count(True) == 1
    engine = build_engine(path)
    try:
        records = repository(engine).recent(SCOPE)
        assert sum(record["status"] == "dispatching" for record in records) == 1
        assert sum(record["reason"] == "prior_unknown_effect" for record in records) == 5
        assert repository(engine).count_dispatched(scope=SCOPE) == 1
    finally:
        engine.dispose()


def test_recent_receipts_are_bounded_json_ready_and_account_scope_isolated(sqlite_engine):
    clock = {"now": NOW}
    repo = repository(sqlite_engine, clock=lambda: clock["now"])
    first = reserve(repo)
    assert repo.mark_dispatching(first["effect_id"])
    assert repo.finish(first["effect_id"], "sent", receipt={"message_id": "synthetic-confirmed"})
    clock["now"] += timedelta(seconds=1)
    second = reserve(repo, run_id="second-wake", payload_hash="b" * 64)
    assert repo.mark_dispatching(second["effect_id"])
    assert repo.finish(second["effect_id"], "uncertain", receipt={"error": "synthetic-timeout"})
    reserve(repo, run_id="other-scope", scope="group:900000002")
    reserve(repository(sqlite_engine, account_id=900000102), run_id="other-account")
    records = repo.recent(SCOPE, limit=64)
    assert [record["effect_id"] for record in records] == [second["effect_id"], first["effect_id"]]
    assert records[0]["source_message_ids"] == [1, 2]
    assert records[0]["receipt"] == {"error": "synthetic-timeout"}
    assert records[0]["dispatched_at"].endswith("+00:00")
    assert len(json.loads(json.dumps(records))) == 2
    records[0]["receipt"]["error"] = "caller-mutation"
    assert repo.get(second["effect_id"])["receipt"]["error"] == "synthetic-timeout"
    assert len(repo.recent(SCOPE, limit=1)) == 1
    for invalid in (True, 0, -1, 65, "1"):
        with pytest.raises(ValueError):
            repo.recent(SCOPE, limit=invalid)


@pytest.mark.parametrize("status", ["failed", "blocked"])
@pytest.mark.parametrize("change_payload", [False, True])
def test_preflight_failure_has_one_bounded_repair_reservation(sqlite_engine, status, change_payload):
    repo = repository(sqlite_engine)
    first = reserve(repo, operation="comment_qzone", run_id="synthetic-post")
    assert repo.finish(first["effect_id"], status, reason="image_not_observed")
    replacement_hash = "b" * 64 if change_payload else HASH
    repaired = repo.reserve(SCOPE, operation="comment_qzone", run_id="synthetic-post", tool_call_id="c",
                            payload_hash=replacement_hash, sources=(3, 4))
    assert repaired["fresh"] is True and repaired["effect_id"] == first["effect_id"]
    assert repaired["status"] == "reserved" and repaired["reservation_attempts"] == 2
    assert repaired["source_message_ids"] == [3, 4] and repaired["payload_hash"] == replacement_hash
    assert repaired["receipt"] is None and repaired["reason"] == ""
    assert repo.count_dispatched() == 0
    assert repo.finish(first["effect_id"], status, reason="second_preflight_failure")
    repeated = reserve(repo, operation="comment_qzone", run_id="synthetic-post", payload_hash=replacement_hash)
    assert repeated["fresh"] is False and repeated["reservation_attempts"] == 2
    with pytest.raises(ValueError, match="identity"):
        reserve(repo, operation="comment_qzone", run_id="synthetic-post", payload_hash="c" * 64)


@pytest.mark.parametrize("status", ["sent", "uncertain", "failed", "blocked"])
def test_dispatched_effect_is_never_revived_or_given_changed_payload(sqlite_engine, status):
    repo = repository(sqlite_engine)
    first = reserve(repo)
    assert repo.mark_dispatching(first["effect_id"], daily_limit=10)
    assert repo.finish(first["effect_id"], status, receipt={"synthetic": status})
    repeated = reserve(repo)
    assert repeated["fresh"] is False and repeated["reservation_attempts"] == 1
    assert repeated["status"] == status
    with pytest.raises(ValueError, match="identity"):
        reserve(repo, payload_hash="b" * 64)


def test_repair_cannot_bypass_another_unresolved_effect(sqlite_engine):
    repo = repository(sqlite_engine)
    repair = reserve(repo, run_id="synthetic-post", operation="comment_qzone")
    assert repo.finish(repair["effect_id"], "blocked")
    unknown = reserve(repo, run_id="another-post", operation="comment_qzone", payload_hash="b" * 64)
    assert repo.mark_dispatching(unknown["effect_id"], daily_limit=10)
    repeated = reserve(repo, run_id="synthetic-post", operation="comment_qzone", payload_hash="b" * 64)
    assert not repeated["fresh"] and repeated["effect_id"] == unknown["effect_id"]
    assert repo.get(repair["effect_id"])["reservation_attempts"] == 1
