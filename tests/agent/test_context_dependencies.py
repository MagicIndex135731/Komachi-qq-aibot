"""Historical provenance survives handoff without inflating the 200-source turn."""
from datetime import UTC, datetime, timedelta

from app.agent.context_dependencies import context_snapshot_valid
from app.storage.agent_sessions import AgentSessionRepository, read_context_snapshot
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.repositories import UserRepository


NOW = datetime(2026, 10, 14, 12, tzinfo=UTC)
ACCOUNT, USER = 900000099, 900000101
SCOPE = f'private:{USER}'


def seed(engine, index):
    with session_scope(engine) as session:
        UserRepository(session).upsert_user(user_id=USER, nickname='synthetic', group_card='')
        session.flush()
        row = Message(platform_msg_id=f'snapshot-source-{index}', group_id=None, user_id=USER,
            timestamp=NOW, plain_text=f'synthetic {index}', raw_json={})
        session.add(row)
        session.flush()
        return row.id


def test_frozen_dependencies_survive_reset_but_not_recall(sqlite_engine, tmp_path):
    repo = AgentSessionRepository(sqlite_engine, account_id=ACCOUNT, session_root=tmp_path, clock=lambda: NOW)
    old, current, later = [seed(sqlite_engine, n) for n in range(3)]
    repo.save(SCOPE, [], {}, [old], expected_revision=0)
    repo.save(SCOPE, [], {}, [current], expected_revision=1)
    identifier = repo.freeze_dependencies(SCOPE, [current])
    manifest = read_context_snapshot(sqlite_engine, account_id=ACCOUNT, scope=SCOPE, identifier=identifier, now=NOW)
    assert manifest['sources'] == [old, current]
    repo.save(SCOPE, [], {}, [later], expected_revision=2, reset_dependencies=True)
    assert repo.read(SCOPE)['context_source_message_ids'] == [later]
    assert context_snapshot_valid(sqlite_engine, account_id=ACCOUNT, scope=SCOPE,
        identifier=identifier, private_user_id=USER, now=NOW)
    with session_scope(sqlite_engine) as session:
        session.get(Message, old).raw_json = {'delivery_state': 'deleted'}
    assert not context_snapshot_valid(sqlite_engine, account_id=ACCOUNT, scope=SCOPE,
        identifier=identifier, private_user_id=USER, now=NOW)


def test_context_snapshot_scope_account_and_lifetime_are_verified(sqlite_engine, tmp_path):
    repo = AgentSessionRepository(sqlite_engine, account_id=ACCOUNT, session_root=tmp_path, clock=lambda: NOW)
    identifier = repo.freeze_dependencies(SCOPE, [seed(sqlite_engine, 1)])
    assert read_context_snapshot(sqlite_engine, account_id=ACCOUNT+1, scope=SCOPE, identifier=identifier, now=NOW) is None
    assert read_context_snapshot(sqlite_engine, account_id=ACCOUNT, scope='private:900000102', identifier=identifier, now=NOW) is None
    assert read_context_snapshot(sqlite_engine, account_id=ACCOUNT, scope=SCOPE, identifier=identifier, now=NOW+timedelta(days=8)) is None
