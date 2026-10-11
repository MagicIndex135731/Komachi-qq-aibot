"""Scoped framework checkpoints never replay uncertain writes or orphan calls."""
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.storage.agent_sessions import AgentSessionRepository, SessionRevisionConflict


ACCOUNT = 900000101
SCOPE = "group:900000001"
NOW = datetime(2026, 10, 9, 8, tzinfo=UTC)


def repository(engine, tmp_path, *, account_id=ACCOUNT, **kwargs):
    return AgentSessionRepository(engine, account_id=account_id,
                                  session_root=tmp_path / "sessions", clock=lambda: NOW, **kwargs)


def turn(number):
    return [{"role": "user", "content": f"synthetic {number}"},
            {"type": "function_call", "call_id": f"call-{number}", "name": "history", "arguments": "{}"},
            {"type": "function_call_output", "call_id": f"call-{number}", "output": "synthetic receipt"},
            {"role": "assistant", "content": "done"}]


def test_scoped_checkpoint_survives_restart_and_isolated_paths(sqlite_engine, tmp_path):
    repo = repository(sqlite_engine, tmp_path)
    path = repo.session_path(SCOPE)
    saved = repo.save(SCOPE, turn(1), {"framework": "deepseek-harness", "session_path": str(path),
                     "session_id": "synthetic-1", "receipts": []}, (1, 2), expected_revision=0)
    assert saved["revision"] == 1
    assert repository(sqlite_engine, tmp_path).read(SCOPE) == saved
    assert repository(sqlite_engine, tmp_path, account_id=900000102).read(SCOPE)["items"] == []
    assert repo.read("group:900000002")["items"] == []
    assert repo.session_path("group:900000002") != path
    assert repository(sqlite_engine, tmp_path, account_id=900000102).session_path(SCOPE) != path
    saved["items"][0]["content"] = "caller mutation"
    assert repo.read(SCOPE)["items"][0]["content"] == "synthetic 1"


def test_stale_worker_cannot_overwrite_checkpoint(sqlite_engine, tmp_path):
    repo = repository(sqlite_engine, tmp_path)
    repo.save(SCOPE, turn(1), {}, (), expected_revision=0)
    with pytest.raises(SessionRevisionConflict):
        repo.save(SCOPE, turn(2), {}, (), expected_revision=0)
    assert repo.read(SCOPE)["items"] == turn(1)


def test_compression_drops_whole_closed_turns_without_a_model(sqlite_engine, tmp_path):
    repo = repository(sqlite_engine, tmp_path, max_items=5)
    saved = repo.save(SCOPE, turn(1) + turn(2), {}, (1,), expected_revision=0)
    assert saved["items"] == turn(2)
    assert saved["dropped_items"] == 4


def test_character_limit_is_strict_and_can_drop_one_oversize_turn(sqlite_engine, tmp_path):
    repo = repository(sqlite_engine, tmp_path, max_characters=300)
    saved = repo.save(SCOPE, [{"role": "user", "content": "x" * 400}], {}, (), expected_revision=0)
    assert saved["items"] == []
    assert saved["dropped_items"] == 1


@pytest.mark.parametrize("items", [
    [{"type": "function_call", "call_id": "c", "name": "send", "arguments": "{}"}],
    [{"type": "function_call_output", "call_id": "c", "output": "orphan"}],
    [{"role": "assistant", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "send"}}]}],
])
def test_orphan_tool_items_are_rejected_before_saving(sqlite_engine, tmp_path, items):
    with pytest.raises(ValueError, match="tool"):
        repository(sqlite_engine, tmp_path).save(SCOPE, items, {}, (), expected_revision=0)


def test_chat_completion_tool_calls_remain_closed(sqlite_engine, tmp_path):
    items = [{"role": "user", "content": "synthetic"},
             {"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "history"}}]},
             {"role": "tool", "tool_call_id": "c", "content": "receipt"}]
    assert repository(sqlite_engine, tmp_path).save(SCOPE, items, {}, (), expected_revision=0)["items"] == items


def test_dispatching_and_uncertain_receipts_are_never_replayable(sqlite_engine, tmp_path):
    repo = repository(sqlite_engine, tmp_path)
    saved = repo.save(SCOPE, turn(1), {"receipts": [
        {"call_id": "a", "status": "dispatching", "replay_allowed": True},
        {"call_id": "b", "status": "uncertain", "replay_allowed": True},
        {"call_id": "c", "status": "sent", "receipt_id": "synthetic-platform-id"},
    ]}, (1,), expected_revision=0)
    restored = repository(sqlite_engine, tmp_path).read(SCOPE)
    assert saved["checkpoint"]["receipts"] == restored["checkpoint"]["receipts"]
    assert [r["status"] for r in restored["checkpoint"]["receipts"]] == ["uncertain", "uncertain", "sent"]
    assert all(not r["replay_allowed"] for r in restored["checkpoint"]["receipts"][:2])
    assert restored["checkpoint"]["receipts"][2]["receipt_id"] == "synthetic-platform-id"


def test_recall_invalidates_context_and_harness_path_but_keeps_receipt_fence(sqlite_engine, tmp_path):
    repo = repository(sqlite_engine, tmp_path)
    repo.save(SCOPE, turn(1), {"session_path": str(repo.session_path(SCOPE)), "session_id": "old",
        "receipts": [{"status": "uncertain", "call_id": "write"}]}, (1, 2), expected_revision=0)
    assert not repo.clear_if_source(SCOPE, 3)
    assert repo.clear_if_source(SCOPE, 1)
    cleared = repo.read(SCOPE)
    assert cleared["items"] == []
    assert cleared["source_message_ids"] == []
    assert "session_path" not in cleared["checkpoint"]
    assert cleared["checkpoint"]["receipts"][0]["replay_allowed"] is False
    assert cleared["revision"] == 2


def test_checkpoint_path_cannot_point_at_other_scope_or_escape_root(sqlite_engine, tmp_path):
    repo = repository(sqlite_engine, tmp_path)
    for path in (tmp_path / "outside.jsonl", repo.session_path("group:900000002")):
        with pytest.raises(ValueError, match="path"):
            repo.save(SCOPE, [], {"session_path": str(path)}, (), expected_revision=0)


@pytest.mark.parametrize("sources", [(True,), (0,), (1, 1), tuple(range(1, 202)), "12"])
def test_session_sources_validate_with_same_boundary_as_intentions(sqlite_engine, tmp_path, sources):
    with pytest.raises(ValueError):
        repository(sqlite_engine, tmp_path).save(SCOPE, [], {}, sources, expected_revision=0)


def test_pruned_checkpoint_still_retains_uncertain_call_fence(sqlite_engine, tmp_path):
    repo = repository(sqlite_engine, tmp_path)
    repo.save(SCOPE, turn(1), {"receipts": [{"call_id": "write", "status": "uncertain"}]}, (), expected_revision=0)
    saved = repo.save(SCOPE, turn(2), {}, (), expected_revision=1)
    assert saved["checkpoint"]["receipts"] == [{"call_id": "write", "status": "uncertain", "replay_allowed": False}]


def test_concurrent_checkpoint_updates_have_one_cas_winner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from app.storage.db import build_engine, create_all
    path = tmp_path / "session-concurrent.db"
    engine = build_engine(path)
    create_all(engine)
    engine.dispose()
    barrier = Barrier(4)

    def save(number):
        engine = build_engine(path)
        try:
            repo = repository(engine, tmp_path)
            barrier.wait(timeout=10)
            try:
                repo.save(SCOPE, turn(number), {}, (), expected_revision=0)
                return True
            except SessionRevisionConflict:
                return False
        finally:
            engine.dispose()

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(save, range(4)))
    assert outcomes.count(True) == 1


def test_begin_turn_persists_inflight_identity_and_sources_before_framework_runs(sqlite_engine, tmp_path):
    repo = repository(sqlite_engine, tmp_path)
    saved = repo.save(SCOPE, turn(1), {"session_id": "closed", "session_path": str(repo.session_path(SCOPE))},
        (1,), expected_revision=0)
    inflight = repo.begin_turn(SCOPE, {"session_id": "closed", "session_path": str(repo.session_path(SCOPE))},
        (2,), expected_revision=saved["revision"])
    restarted = repository(sqlite_engine, tmp_path).read(SCOPE)
    assert restarted == inflight
    assert restarted["checkpoint"]["in_flight"] is True
    assert restarted["source_message_ids"] == [1, 2]
    assert restarted["items"] == turn(1)
    assert repo.clear_if_source(SCOPE, 2)
    assert "session_id" not in repo.read(SCOPE)["checkpoint"]


def test_completed_turn_clears_inflight_flag_and_fences_original_revision(sqlite_engine, tmp_path):
    repo = repository(sqlite_engine, tmp_path)
    pending = repo.begin_turn(SCOPE, {"session_id": "current"}, (1,), expected_revision=0)
    with pytest.raises(SessionRevisionConflict):
        repo.save(SCOPE, turn(1), {"session_id": "current"}, (1,), expected_revision=0)
    complete = repo.save(SCOPE, turn(1), {"session_id": "current", "last_status": "silent"},
        (1,), expected_revision=pending["revision"])
    assert "in_flight" not in complete["checkpoint"]
    assert complete["revision"] == 2
