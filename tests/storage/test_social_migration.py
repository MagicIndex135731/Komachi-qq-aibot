from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import text

from app.storage.db import create_all


def test_earlier_tool_effect_schema_preserves_unknown_receipt_on_upgrade(sqlite_engine):
    from app.storage.agent_effects import AgentToolEffectRepository
    repository = AgentToolEffectRepository(sqlite_engine, account_id=900000999)
    effect = repository.reserve("public", run_id="synthetic-old-wake", tool_call_id="synthetic-call",
        operation="publish_qzone", payload_hash="a" * 64)
    assert repository.mark_dispatching(effect["effect_id"])
    repository.finish(effect["effect_id"], "uncertain", receipt={"state": "unknown"})
    with sqlite_engine.begin() as connection:
        connection.execute(text("ALTER TABLE agent_tool_effects DROP COLUMN reservation_attempts"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: create_all(sqlite_engine), range(2)))
    row = repository.read(effect["effect_id"])
    assert row["status"] == "uncertain" and row["receipt"] == {"state": "unknown"}
    assert row["reservation_attempts"] == 1
    blocked = repository.reserve("public", run_id="synthetic-new-wake", tool_call_id="new-call",
        operation="publish_qzone", payload_hash="a" * 64)
    assert not blocked["fresh"] and blocked["effect_id"] == effect["effect_id"]


def test_old_social_ledger_migrates_without_losing_uncertain_actions(sqlite_engine):
    with sqlite_engine.begin() as connection:
        connection.execute(text("ALTER TABLE social_action_records DROP COLUMN dispatched_at"))
        connection.execute(text(
            "INSERT INTO social_account_leases (account_id, owner, epoch, lease_until) "
            "VALUES (900000999, '', 1, '2026-10-09 00:00:00')"
        ))
        connection.execute(text(
            "INSERT INTO social_action_records "
            "(action_id, account_id, owner, epoch, job_key, kind, scope_id, dedupe_key, "
            "action, text, post_id, post_owner_id, source_message_ids, status, reason, created_at, updated_at) "
            "VALUES ('synthetic-uncertain', 900000999, 'old', 1, 'publish', 'qzone_publish', "
            "900000999, 'synthetic-dedupe', 'publish_qzone', 'public thought', '', 0, '[]', "
            "'uncertain', 'unknown_receipt', '2026-10-09 01:00:00', '2026-10-09 01:00:00')"
        ))

    # Group/private startup may inspect the same earlier schema concurrently.
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: create_all(sqlite_engine), range(2)))
    create_all(sqlite_engine)

    with sqlite_engine.connect() as connection:
        columns = [row[1] for row in connection.execute(text("PRAGMA table_info(social_action_records)"))]
        row = connection.execute(text(
            "SELECT action_id, status, reason, dispatched_at, text FROM social_action_records"
        )).one()
    assert columns.count("dispatched_at") == 1
    assert tuple(row) == ("synthetic-uncertain", "uncertain", "unknown_receipt", None, "public thought")
