"""Durable agent budgets and intentions retain their account/scope boundary."""
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest
from sqlalchemy import select

from app.storage.autonomous_agent import AutonomousIntention, AutonomousRepository, AutonomousUsage
from app.storage.db import build_engine, create_all, session_scope


ACCOUNT = 900000101
OTHER_ACCOUNT = 900000102
SCOPE = "group:900000001"
OTHER_SCOPE = "group:900000002"
NOW = datetime(2026, 10, 9, 8, tzinfo=UTC)


def repository(engine, *, account_id=ACCOUNT, clock=lambda: NOW):
    return AutonomousRepository(engine, account_id=account_id, clock=clock)


def test_concurrent_reservations_limit_two_across_independent_sqlite_connections(tmp_path):
    path = tmp_path / "concurrent-autonomous.db"
    initializer = build_engine(path)
    create_all(initializer)
    initializer.dispose()
    barrier = Barrier(8)

    def reserve(_):
        engine = build_engine(path)
        try:
            repo = repository(engine)
            barrier.wait(timeout=10)
            return repo.reserve_cost("model", 2)
        finally:
            engine.dispose()

    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(pool.map(reserve, range(8)))
    assert accepted.count(True) == 2
    assert accepted.count(False) == 6
    reopened = build_engine(path)
    try:
        assert not repository(reopened).reserve_cost("model", 2)
        with session_scope(reopened) as session:
            rows = list(session.scalars(select(AutonomousUsage)))
            assert len(rows) == 2
            assert {row.account_id for row in rows} == {ACCOUNT}
    finally:
        reopened.dispose()


def test_daily_quota_resets_at_shanghai_midnight_not_utc_midnight(sqlite_engine):
    clock = {"now": datetime(2026, 10, 9, 15, 59, 59, tzinfo=UTC)}
    repo = repository(sqlite_engine, clock=lambda: clock["now"])
    assert repo.reserve_cost("model", 1)
    assert not repo.reserve_cost("model", 1)
    clock["now"] = datetime(2026, 10, 9, 16, 0, 0, tzinfo=UTC)
    assert repo.reserve_cost("model", 1)
    clock["now"] = datetime(2026, 10, 10, 0, 0, 0, tzinfo=UTC)
    assert not repo.reserve_cost("model", 1)


def test_model_image_profile_group_and_account_budgets_use_distinct_buckets(sqlite_engine):
    repo = repository(sqlite_engine)
    kinds = ("model", "image", "profile:nickname", "profile:avatar",
             "profile:card:900000001", "profile:card:900000002")
    for kind in kinds:
        assert repo.reserve_cost(kind, 1)
        assert not repo.reserve_cost(kind, 1)
    assert repository(sqlite_engine, account_id=OTHER_ACCOUNT).reserve_cost("model", 1)
    with session_scope(sqlite_engine) as session:
        rows = list(session.scalars(select(AutonomousUsage)))
        assert len(rows) == len(kinds) + 1


def test_failed_external_call_does_not_refund_a_durable_reservation(sqlite_engine):
    repo = repository(sqlite_engine)
    with pytest.raises(TimeoutError):
        assert repo.reserve_cost("image", 1)
        raise TimeoutError("synthetic upstream outcome unknown")
    restarted = repository(sqlite_engine)
    assert not restarted.reserve_cost("image", 1)
    assert restarted.reserve_cost("model", 1)


def test_zero_budget_and_upper_limit_boundary_are_enforced(sqlite_engine):
    repo = repository(sqlite_engine)
    assert not repo.reserve_cost("model", 0)
    assert repo.reserve_cost("model", 500)


@pytest.mark.parametrize("limit", [True, False, -1, 501, 1.5, "2", None])
def test_cost_limit_rejects_boolean_noninteger_and_out_of_range_values(sqlite_engine, limit):
    with pytest.raises(ValueError):
        repository(sqlite_engine).reserve_cost("model", limit)
    with session_scope(sqlite_engine) as session:
        assert list(session.scalars(select(AutonomousUsage))) == []


@pytest.mark.parametrize("kind", [True, None, 5, "", "other", "profile:card:",
                                  "profile:card:0", "profile:card:-1", "profile:card:True",
                                  "profile:card:001", "profile:card:" + "1" * 201])
def test_cost_kind_rejects_unknown_or_noncanonical_profile_group_ids(sqlite_engine, kind):
    with pytest.raises(ValueError):
        repository(sqlite_engine).reserve_cost(kind, 1)


@pytest.mark.parametrize("account", [True, False, 0, -1, "900000101", 1.5, None])
def test_account_boundary_requires_a_positive_integer(sqlite_engine, account):
    with pytest.raises(ValueError):
        repository(sqlite_engine, account_id=account)


def test_intentions_persist_and_isolate_account_scope_and_caller_mutation(sqlite_engine):
    first = repository(sqlite_engine)
    first.write_intention(SCOPE, {"goal": "synthetic music", "steps": ["history"]}, (1, 2))
    first.write_intention(OTHER_SCOPE, {"goal": "synthetic drawing"}, (3,))
    other = repository(sqlite_engine, account_id=OTHER_ACCOUNT)
    assert other.read_intention(SCOPE) == {}
    other.write_intention(SCOPE, {"goal": "another synthetic account"}, (4,))
    restarted = repository(sqlite_engine)
    note = restarted.read_intention(SCOPE)
    assert note == {"working_intention": {"goal": "synthetic music", "steps": ["history"]},
                    "source_message_ids": [1, 2]}
    note["working_intention"]["steps"].append("caller mutation")
    note["source_message_ids"].append(9)
    assert restarted.read_intention(SCOPE)["source_message_ids"] == [1, 2]
    assert restarted.read_intention(SCOPE)["working_intention"]["steps"] == ["history"]
    assert restarted.read_intention(OTHER_SCOPE)["source_message_ids"] == [3]
    assert other.read_intention(SCOPE)["source_message_ids"] == [4]


def test_intention_source_budget_accepts_two_hundred_and_rejects_overwrite_overflow(sqlite_engine):
    repo = repository(sqlite_engine)
    sources = tuple(range(1, 201))
    repo.write_intention(SCOPE, {"goal": "synthetic source-bound intention"}, sources)
    with pytest.raises(ValueError):
        repo.write_intention(SCOPE, {"goal": "overflow"}, tuple(range(1, 202)))
    assert repo.read_intention(SCOPE)["source_message_ids"] == list(sources)


@pytest.mark.parametrize("sources", [(True,), (False,), (0,), (-1,), ("1",), (1.5,),
                                     (None,), ({"id": 1},), ([1],), "12", None])
def test_intention_rejects_illegal_source_ids_before_storage(sqlite_engine, sources):
    with pytest.raises(ValueError):
        repository(sqlite_engine).write_intention(SCOPE, {"goal": "synthetic"}, sources)
    with session_scope(sqlite_engine) as session:
        assert list(session.scalars(select(AutonomousIntention))) == []


@pytest.mark.parametrize("scope", ["", " ", "x" * 161, True, None, 1])
def test_intention_rejects_invalid_scope_on_read_and_write(sqlite_engine, scope):
    repo = repository(sqlite_engine)
    with pytest.raises(ValueError):
        repo.write_intention(scope, {"goal": "synthetic"})
    with pytest.raises(ValueError):
        repo.read_intention(scope)


@pytest.mark.parametrize("note", [True, None, [], "synthetic", {"goal": "x" * 2001}])
def test_intention_rejects_nonobject_and_oversize_note_without_overwriting(sqlite_engine, note):
    repo = repository(sqlite_engine)
    repo.write_intention(SCOPE, {"goal": "kept synthetic"}, (1,))
    with pytest.raises(ValueError):
        repo.write_intention(SCOPE, note, (2,))
    assert repo.read_intention(SCOPE) == {"working_intention": {"goal": "kept synthetic"},
                                        "source_message_ids": [1]}


def test_due_goal_claim_consumes_only_one_wakeup_and_preserves_legacy_note(sqlite_engine):
    repo = repository(sqlite_engine)
    note = {"goal": "synthetic long goal", "progress": "history read", "next_step": "inspect"}
    repo.write_intention(SCOPE, note, (1, 2))
    repo.schedule_intention(SCOPE, next_due=NOW, status="waiting")
    claims = repo.claim_due_intentions("worker", limit=4, lease_seconds=60)
    assert len(claims) == 1
    assert claims[0].scope == SCOPE
    assert claims[0].working_intention == note
    assert claims[0].source_message_ids == (1, 2)
    assert repository(sqlite_engine).claim_due_intentions("other") == ()
    assert repo.finish_intention_claim(claims[0])
    assert repo.claim_due_intentions("worker") == ()
    assert repo.read_intention(SCOPE)["working_intention"] == note
    assert repo.read_intention_schedule(SCOPE)["next_due"] is None


def test_agent_reschedule_during_execution_survives_claim_finalization(sqlite_engine):
    repo = repository(sqlite_engine)
    repo.write_intention(SCOPE, {"goal": "synthetic"}, (1,))
    repo.schedule_intention(SCOPE, next_due=NOW)
    claim = repo.claim_due_intentions("worker")[0]
    repo.write_intention(SCOPE, {"goal": "synthetic", "next_step": "later"}, (1,))
    later = NOW + timedelta(hours=1)
    repo.schedule_intention(SCOPE, next_due=later, status="waiting")
    assert repo.finish_intention_claim(claim)
    assert repo.read_intention_schedule(SCOPE)["next_due"] == later
    assert repo.claim_due_intentions("worker") == ()


def test_goal_not_entering_agent_can_release_or_defer_without_losing_schedule(sqlite_engine):
    repo = repository(sqlite_engine)
    repo.write_intention(SCOPE, {"goal": "synthetic"}, next_due=NOW)
    claim = repo.claim_due_intentions("first")[0]
    assert repo.finish_intention_claim(claim, consume=False)
    assert repo.read_intention_schedule(SCOPE)["next_due"] == NOW
    retried = repo.claim_due_intentions("second")[0]
    later = NOW + timedelta(minutes=10)
    assert repo.finish_intention_claim(retried, consume=False, retry_at=later)
    assert repo.read_intention_schedule(SCOPE)["next_due"] == later
    assert repo.claim_due_intentions("third") == ()


def test_release_or_defer_cannot_overwrite_new_agent_schedule(sqlite_engine):
    repo = repository(sqlite_engine)
    repo.write_intention(SCOPE, {"goal": "synthetic"}, next_due=NOW)
    claim = repo.claim_due_intentions("first")[0]
    selected = NOW + timedelta(days=1)
    repo.schedule_intention(SCOPE, next_due=selected)
    assert repo.finish_intention_claim(claim, consume=False, retry_at=NOW + timedelta(minutes=1))
    assert repo.read_intention_schedule(SCOPE)["next_due"] == selected


def test_expired_claim_recovers_after_restart_and_fences_stale_worker(sqlite_engine):
    clock = {"now": NOW}
    repo = repository(sqlite_engine, clock=lambda: clock["now"])
    repo.write_intention(SCOPE, {"goal": "synthetic"})
    repo.schedule_intention(SCOPE, next_due=NOW)
    stale = repo.claim_due_intentions("old", lease_seconds=10)[0]
    clock["now"] += timedelta(seconds=11)
    resumed = repository(sqlite_engine, clock=lambda: clock["now"])
    claim = resumed.claim_due_intentions("new", lease_seconds=10)[0]
    assert claim.token != stale.token
    assert not repo.finish_intention_claim(stale)
    assert not repo.renew_intention_claim(stale, lease_seconds=10)
    assert resumed.renew_intention_claim(claim, lease_seconds=10)
    assert resumed.finish_intention_claim(claim)


def test_due_goals_account_scope_order_terminal_and_cancellation(sqlite_engine):
    repo = repository(sqlite_engine)
    repo.write_intention(OTHER_SCOPE, {"goal": "later"})
    repo.schedule_intention(OTHER_SCOPE, next_due=NOW + timedelta(seconds=1))
    repo.write_intention(SCOPE, {"goal": "due"}, (1,))
    repo.schedule_intention(SCOPE, next_due=NOW)
    other = repository(sqlite_engine, account_id=OTHER_ACCOUNT)
    other.write_intention(SCOPE, {"goal": "other account"})
    other.schedule_intention(SCOPE, next_due=NOW)
    assert [claim.scope for claim in repo.claim_due_intentions("one")] == [SCOPE]
    assert len(other.claim_due_intentions("two")) == 1
    assert repo.invalidate_intention_source(SCOPE, 1)
    assert repo.read_intention(SCOPE) == {}
    assert repo.read_intention_schedule(SCOPE) == {}


def test_completed_goal_never_becomes_due(sqlite_engine):
    repo = repository(sqlite_engine)
    repo.write_intention(SCOPE, {"goal": "done"})
    repo.schedule_intention(SCOPE, next_due=None, status="completed")
    assert repo.claim_due_intentions("worker") == ()
    with pytest.raises(ValueError):
        repo.schedule_intention(SCOPE, next_due=NOW, status="completed")


def test_concurrent_due_claims_have_one_winner(tmp_path):
    path = tmp_path / "concurrent-due.db"
    initializer = build_engine(path)
    create_all(initializer)
    repo = repository(initializer)
    repo.write_intention(SCOPE, {"goal": "synthetic"})
    repo.schedule_intention(SCOPE, next_due=NOW)
    initializer.dispose()
    barrier = Barrier(6)

    def claim(number):
        engine = build_engine(path)
        try:
            repo = repository(engine)
            barrier.wait(timeout=10)
            return repo.claim_due_intentions(f"worker-{number}")
        finally:
            engine.dispose()

    with ThreadPoolExecutor(max_workers=6) as pool:
        claims = list(pool.map(claim, range(6)))
    assert sum(map(len, claims)) == 1


@pytest.mark.parametrize("kwargs", [
    {"next_due": datetime(2026, 10, 9)}, {"next_due": "later"},
    {"next_due": NOW, "status": "running"}, {"next_due": NOW, "status": "cancelled"},
])
def test_goal_schedule_rejects_invalid_fields(sqlite_engine, kwargs):
    repo = repository(sqlite_engine)
    repo.write_intention(SCOPE, {"goal": "kept"})
    with pytest.raises(ValueError):
        repo.schedule_intention(SCOPE, **kwargs)
    assert repo.read_intention_schedule(SCOPE) == {}


def test_intention_and_schedule_roll_back_if_authority_changes_before_commit(sqlite_engine):
    repo = repository(sqlite_engine)
    repo.write_intention(SCOPE, {"goal": "kept"}, (1,))

    def cancelled():
        raise RuntimeError("synthetic authority revoked")

    with pytest.raises(RuntimeError):
        repo.write_intention(SCOPE, {"goal": "must rollback"}, (2,), next_due=NOW,
                             before_commit=cancelled)
    assert repo.read_intention(SCOPE)["working_intention"] == {"goal": "kept"}
    assert repo.read_intention_schedule(SCOPE) == {}


def test_listing_goals_is_read_only_json_ready_and_scoped(sqlite_engine):
    repo = repository(sqlite_engine)
    repo.write_intention(SCOPE, {"goal": "due"}, (1,), next_due=NOW, status="waiting")
    repo.write_intention(OTHER_SCOPE, {"goal": "legacy dormant"})
    listed = repo.list_intentions(scopes=(SCOPE,))
    assert len(listed) == 1
    assert listed[0]["next_due"] == NOW.isoformat()
    assert listed[0]["status"] == "waiting"
    assert repository(sqlite_engine, account_id=OTHER_ACCOUNT).list_intentions() == ()
    assert repo.claim_due_intentions("worker", scopes=(OTHER_SCOPE,)) == ()
    assert len(repo.claim_due_intentions("worker", scopes=(SCOPE,))) == 1


def test_additive_migration_retains_legacy_intention_without_scheduling_it(tmp_path):
    from sqlalchemy import text
    path = tmp_path / "legacy-intention.db"
    engine = build_engine(path)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE autonomous_intentions (key VARCHAR(200) PRIMARY KEY, "
            "account_id INTEGER NOT NULL, scope VARCHAR(160) NOT NULL, note JSON NOT NULL, "
            "source_message_ids JSON NOT NULL, updated_at DATETIME NOT NULL)"))
        connection.execute(text("INSERT INTO autonomous_intentions VALUES (:key,:account,:scope,:note,:sources,:at)"),
            {"key": f"{ACCOUNT}:{SCOPE}", "account": ACCOUNT, "scope": SCOPE,
             "note": '{"goal":"legacy synthetic","next_step":"continue"}', "sources": "[1]", "at": "2026-10-09 08:00:00"})
    create_all(engine)
    create_all(engine)
    repo = repository(engine)
    assert repo.read_intention(SCOPE)["working_intention"] == {"goal": "legacy synthetic", "next_step": "continue"}
    assert repo.read_intention_schedule(SCOPE) == {}
    assert repo.claim_due_intentions("worker") == ()
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA integrity_check")).scalar_one() == "ok"
    engine.dispose()
