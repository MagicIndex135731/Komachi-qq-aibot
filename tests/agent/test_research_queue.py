from datetime import UTC, datetime, timedelta

import pytest

from app.agent.research_queue import ResearchDelegationQueue
from app.storage.db import session_scope
from app.storage.models import Job, Message
from app.storage.repositories import UserRepository


def fixture_queue(engine, *, surface='private', execute=None, deliver=None):
    now = [datetime(2026, 10, 14, 12, tzinfo=UTC)]
    with session_scope(engine) as session:
        UserRepository(session).upsert_user(user_id=900000101, nickname='synthetic', group_card='')
        session.flush()
        row = Message(platform_msg_id='private-queue-source', group_id=None, user_id=900000101,
                      timestamp=now[0], plain_text='source', raw_json={})
        session.add(row)
        session.flush()
        source_id = row.id
    payload = {'account_id': 900000099, 'surface': 'private', 'scope_id': 900000101,
               'scope': 'private:900000101', 'actor_id': 900000101, 'source_id': source_id,
               'sources': [source_id], 'memory': {}, 'task_id': 'a' * 24, 'goal': 'synthetic goal'}
    async def complete(payload, guard):
        guard.require_valid()
        return {'status': 'completed', 'findings': ['source-bound finding']}
    async def sent(payload, guard):
        guard.require_valid()
        guard.on_dispatch()
        return 'completed'
    queue = ResearchDelegationQueue(engine, account_id=900000099, surface=surface,
        execute=execute or complete, deliver=deliver or sent, authorized=lambda _: True, clock=lambda: now[0])
    queue.running = True
    return queue, payload, now


async def test_persisted_child_is_idempotent_and_only_own_process_claims_result(sqlite_engine):
    queue, payload, now = fixture_queue(sqlite_engine)
    result = queue.enqueue(payload, parent_turn=10, before_commit=lambda: None)
    assert queue.enqueue(payload, parent_turn=10, before_commit=lambda: None)['job_id'] == result['job_id']
    other = ResearchDelegationQueue(sqlite_engine, account_id=900000099, surface='group',
        execute=queue.execute, deliver=queue.deliver, authorized=lambda _: True, clock=lambda: now[0])
    assert other.claim() is None
    assert await queue.tick()
    with session_scope(sqlite_engine) as session:
        assert session.get(Job, result['job_id']).status == 'result_ready'
    restarted = ResearchDelegationQueue(sqlite_engine, account_id=900000099, surface='private',
        execute=queue.execute, deliver=queue.deliver, authorized=lambda _: True, clock=lambda: now[0])
    restarted.running = True
    assert await restarted.tick()
    assert not await restarted.tick()
    with session_scope(sqlite_engine) as session:
        assert session.get(Job, result['job_id']).status == 'completed'


@pytest.mark.parametrize('failure', ['revoked', 'cross_user', 'expired', 'cancelled'])
async def test_revoked_or_cross_user_result_never_delivers(sqlite_engine, failure):
    delivered = []
    async def send(payload, guard):
        delivered.append(payload)
        return 'completed'
    queue, payload, now = fixture_queue(sqlite_engine, deliver=send)
    if failure == 'cross_user':
        payload['actor_id'] = 900000102
    result = queue.enqueue(payload, parent_turn=10, before_commit=lambda: None)
    if failure == 'cancelled':
        queue.cancel_scope(payload['scope_id'])
    elif failure == 'expired':
        now[0] += timedelta(days=8)
    elif failure == 'revoked':
        with session_scope(sqlite_engine) as session:
            session.get(Message, payload['source_id']).raw_json = {'delivery_state': 'deleted'}
    await queue.tick()
    assert delivered == []
    with session_scope(sqlite_engine) as session:
        assert session.get(Job, result['job_id']).status in {'blocked', 'cancelled'}


async def test_unknown_delivery_owner_loss_never_replays_and_read_recovery_is_bounded(sqlite_engine):
    queue, payload, now = fixture_queue(sqlite_engine)
    result = queue.enqueue(payload, parent_turn=10, before_commit=lambda: None)
    job_id, _, phase = queue.claim()
    assert phase == 'running'
    now[0] += timedelta(seconds=901)
    assert queue.claim()[2] == 'running'
    now[0] += timedelta(seconds=901)
    assert queue.claim() is None
    with session_scope(sqlite_engine) as session:
        assert session.get(Job, job_id).status == 'failed'
    second = queue.enqueue(payload, parent_turn=11, before_commit=lambda: None)
    assert await queue.tick()
    assert queue.claim()[2] == 'delivering'
    now[0] += timedelta(seconds=901)
    assert queue.claim() is None
    with session_scope(sqlite_engine) as session:
        assert session.get(Job, second['job_id']).status == 'uncertain'
