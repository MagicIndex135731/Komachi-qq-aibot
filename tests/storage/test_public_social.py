from datetime import timedelta
import json

import pytest

from tests.agent.test_autonomous_social import BOT, NOW, FRIEND


def repo(engine, **kw):
    from app.storage.public_social import PublicSocialRepository
    return PublicSocialRepository(engine, account_id=BOT, clock=lambda: NOW, **kw)


def test_account_isolation_detail_pagination_and_bounded_output(sqlite_engine):
    from app.storage.public_social import PublicSocialRepository
    r = repo(sqlite_engine)
    for n in range(12):
        r.save(event_key=str(n), kind='thought', operation='save_work_note', status='ok',
               payload={'note': 'PUBLIC_' + str(n) + '长' * 1100}, created_at=NOW + timedelta(seconds=n))
    result = r.read(limit=5)
    assert len(result['records']) == 5 and result['next_before']
    assert len(json.dumps(result, ensure_ascii=False)) < 4500
    record_id = result['records'][0]['record_id']
    assert '长' * 1000 in r.read(record_id=record_id)['record']['payload']['note']
    assert PublicSocialRepository(sqlite_engine, account_id=BOT + 1).read(record_id=record_id)['record'] is None
    earlier = r.read(before=result['next_before'], limit=5)
    assert not ({x['record_id'] for x in result['records']} & {x['record_id'] for x in earlier['records']})


def test_retention_expiry_idempotency_and_reinitialization(sqlite_engine):
    from app.storage.db import create_all
    r = repo(sqlite_engine, max_records=3)
    for n in range(6):
        r.save(event_key=str(n), kind='thought', operation='remember_intention', status='ok',
               payload={'goal': str(n)}, created_at=NOW - timedelta(seconds=10 - n))
    assert len(r.read(limit=10)['records']) == 3
    r.save(event_key='5', kind='thought', operation='remember_intention', status='ok', payload={'goal': 'same'})
    assert len(r.read(limit=10)['records']) == 3
    r.save(event_key='expired', kind='thought', operation='remember_intention', status='ok',
           payload={'goal': 'old'}, created_at=NOW - timedelta(days=15))
    assert all(x.get('preview') != 'old' for x in r.read(limit=10)['records'])
    create_all(sqlite_engine)
    assert len(repo(sqlite_engine).read(limit=10)['records']) == 3


@pytest.mark.parametrize('status', ['sent', 'uncertain', 'blocked', 'failed'])
def test_comment_status_comes_from_matching_effect_and_unknown_is_never_replayable(sqlite_engine, status):
    from app.storage.agent_effects import AgentToolEffectRepository
    effects = AgentToolEffectRepository(sqlite_engine, account_id=BOT, clock=lambda: NOW)
    effect = effects.reserve('public:qzone_comment', run_id='post:synthetic', tool_call_id='comment_qzone',
                             operation='comment_qzone', payload_hash='a' * 64)
    if status in {'sent', 'uncertain'}:
        effects.mark_dispatching(effect['effect_id'], daily_limit=5)
    r = repo(sqlite_engine)
    r.save(event_key='comment', kind='comment', operation='comment_qzone', status='pending',
           owner_id=FRIEND, post_id='synthetic', payload={'text': 'public draft'},
           effect_id=effect['effect_id'], payload_hash='a' * 64)
    effects.finish(effect['effect_id'], status, reason='synthetic-result')
    row = r.read(kind='comment')['records'][0]
    assert row['status'] == status
    assert row['write_dispatched'] == (status in {'sent', 'uncertain'})
    assert row['automatic_replay_allowed'] is False
    if status == 'blocked':
        effects.reserve('public:qzone_comment', run_id='post:synthetic', tool_call_id='comment_qzone',
                        operation='comment_qzone', payload_hash='b' * 64)
        assert r.read(kind='comment')['records'][0]['status'] == 'unverified'


def test_record_without_live_effect_cannot_claim_success(sqlite_engine):
    r = repo(sqlite_engine)
    r.save(event_key='unverified', kind='comment', operation='comment_qzone', status='sent',
           owner_id=FRIEND, post_id='synthetic', payload={'text': 'not a receipt'})
    assert r.read(kind='comment')['records'][0]['status'] == 'unverified'


def test_observation_churn_cannot_evict_comment_and_private_fields_are_dropped(sqlite_engine):
    r = repo(sqlite_engine, max_records=3)
    r.save(event_key='draft', kind='comment', operation='comment_qzone', status='blocked',
           payload={'text': 'keep draft', 'private_instruction': 'PRIVATE_SECRET'})
    for n in range(8):
        r.save(event_key=str(n), kind='observation', operation='read_friend_posts', status='observed',
               payload={'post_text': 'public post'}, created_at=NOW + timedelta(seconds=n))
    result = r.read(kind='comment')
    assert len(result['records']) == 1
    assert 'PRIVATE_SECRET' not in json.dumps(r.read(record_id=result['records'][0]['record_id']))


def test_date_filter_uses_shanghai_midnight_and_owner_filter(sqlite_engine):
    from datetime import UTC, datetime
    from app.storage.public_social import PublicSocialRepository
    now = datetime(2026, 10, 17, 16, 30, tzinfo=UTC)
    r = PublicSocialRepository(sqlite_engine, account_id=BOT, clock=lambda: now)
    for event, owner, stamp in [('yesterday', FRIEND, now-timedelta(minutes=31)),
                                 ('today', FRIEND, now-timedelta(minutes=29)),
                                 ('other-friend', FRIEND+1, now)]:
        r.save(event_key=event, kind='observation', operation='read_friend_posts', status='observed',
               owner_id=owner, payload={'post_text': event}, created_at=stamp)
    assert [x['preview'] for x in r.read(date='today', owner_id=FRIEND)['records']] == ['today']
    assert [x['preview'] for x in r.read(date='2026-10-17')['records']] == ['yesterday']


def test_native_backfill_only_imports_public_tools_and_exact_effects(sqlite_engine, tmp_path):
    from hashlib import sha256
    from app.agent.social_memory import import_public_native
    from app.storage.agent_effects import AgentToolEffectRepository
    from sqlalchemy import select
    from sqlalchemy.orm import Session
    from app.storage.agent_effects import AgentToolEffect
    args = {'owner_id': FRIEND, 'post_id': 'synthetic', 'text': 'old public draft', 'artifact_ids': []}
    digest = sha256(json.dumps(args, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    effects = AgentToolEffectRepository(sqlite_engine, account_id=BOT, clock=lambda: NOW)
    effect = effects.reserve('public:qzone_comment', run_id='old-post', tool_call_id='comment_qzone',
                             operation='comment_qzone', payload_hash=digest)
    effects.finish(effect['effect_id'], 'blocked')
    events = [
        {'type': 'assistant/message', 'time': NOW.timestamp()*1000, 'data': {'text': 'HIDDEN_PRIVATE_TEXT'}},
        {'type': 'tool/call', 'time': NOW.timestamp()*1000,
         'data': {'name': 'comment_qzone', 'callId': 'old-call', 'arguments': json.dumps(args)}},
        {'type': 'tool/result', 'time': NOW.timestamp()*1000, 'data': {'message': {
            'source': {'callId': 'old-call'}, 'content': [{'type': 'tool-result', 'content': [
                {'type': 'text', 'text': json.dumps({'status': 'blocked', 'data': {}})}]}]}}},
    ]
    path = tmp_path/'native.jsonl'
    path.write_text('\n'.join(json.dumps(e) for e in events))
    with Session(sqlite_engine) as session:
        authoritative = list(session.scalars(select(AgentToolEffect)))
        for _ in range(2):
            assert len(import_public_native(repo(sqlite_engine), path, authoritative, now=NOW)) == 1
    data = repo(sqlite_engine).read(kind='comment')
    assert len(data['records']) == 1 and data['records'][0]['status'] == 'blocked'
    detail = repo(sqlite_engine).read(record_id=data['records'][0]['record_id'])
    assert detail['record']['payload']['text'] == args['text']
    assert 'HIDDEN_PRIVATE_TEXT' not in json.dumps(detail)
