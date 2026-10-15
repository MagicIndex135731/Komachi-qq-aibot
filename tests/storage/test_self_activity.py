"""Own public action facts survive context separation without chat disclosure."""
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest

from app.storage.agent_effects import AgentToolEffect, AgentToolEffectRepository
from app.storage.db import session_scope
from app.storage.social_agent import SocialAccountLease, SocialActionRecord, utc_naive

ACCOUNT = 900000101
NOW = datetime(2026, 10, 14, 18, tzinfo=UTC)


def effect(engine, key, *, account=ACCOUNT, scope='public:qzone_publish', operation='publish_qzone',
           status='sent', receipt='synthetic-post', stamp=NOW, dispatched=True, run_id=None):
    repo = AgentToolEffectRepository(engine, account_id=account, clock=lambda: stamp)
    value = repo.reserve(scope, run_id=run_id or key, tool_call_id=key, operation=operation,
                         payload_hash=sha256(key.encode()).hexdigest())
    with session_scope(engine) as session:
        row = session.get(AgentToolEffect, value['effect_id'])
        row.status, row.receipt = status, receipt
        row.dispatched_at = utc_naive(stamp) if dispatched else None
    return value['effect_id']


def legacy(engine, key, *, action='publish_qzone', status='sent', receipt='legacy-post'):
    with session_scope(engine) as session:
        if session.get(SocialAccountLease, ACCOUNT) is None:
            session.add(SocialAccountLease(account_id=ACCOUNT, owner='synthetic', epoch=1,
                                          lease_until=utc_naive(NOW + timedelta(hours=1))))
            session.flush()
        session.add(SocialActionRecord(action_id=key, account_id=ACCOUNT, owner='synthetic', epoch=1,
            job_key='qzone_publish', kind='qzone_publish', scope_id=ACCOUNT, dedupe_key=key,
            action=action, text='MUST_NOT_DISCLOSE_PRIVATE_PROCESS', source_message_ids=[],
            status=status, receipt_id=receipt, created_at=utc_naive(NOW),
            dispatched_at=utc_naive(NOW), updated_at=utc_naive(NOW)))


def read(engine, **kwargs):
    # A fresh repository proves this is independent of the native chat context.
    return AgentToolEffectRepository(engine, account_id=ACCOUNT, clock=lambda: NOW).self_publications(**kwargs)


def test_self_publications_account_type_scope_and_envelope_isolation(sqlite_engine):
    first = effect(sqlite_engine, 'public-post', run_id='same-wake')
    effect(sqlite_engine, 'owner-post', scope='owner:public:qzone_publish', receipt='owner-post')
    effect(sqlite_engine, 'other-account', account=900000102)
    effect(sqlite_engine, 'group-leak', scope='group:900000001')
    effect(sqlite_engine, 'private-leak', scope='private:900000103')
    effect(sqlite_engine, 'friend-comment', scope='public:qzone_comment:friend:900000102', operation='comment_qzone')
    legacy(sqlite_engine, 'same-wake')
    legacy(sqlite_engine, 'real-legacy')
    legacy(sqlite_engine, 'envelope', action='run_tools')
    result = read(sqlite_engine)
    assert result['date'] == '2026-10-15' and result['timezone'] == 'Asia/Shanghai'
    assert result['confirmed_count'] == 3 and result['published'] is True
    assert len(result['records']) == 3 and first in {r['action_id'] for r in result['records']}
    assert 'MUST_NOT_DISCLOSE' not in str(result)
    assert 'source_message_ids' not in str(result) and 'group:' not in str(result)
    assert result['coverage'] == 'system_publication_receipts_only'


def test_self_publications_unknown_pending_failure_and_absence(sqlite_engine):
    effect(sqlite_engine, 'unknown', status='uncertain', receipt=None)
    effect(sqlite_engine, 'in-flight', status='dispatching', receipt=None)
    effect(sqlite_engine, 'bad-sent', receipt=None)
    effect(sqlite_engine, 'not-dispatched-sent', receipt='bad', dispatched=False)
    effect(sqlite_engine, 'failed', status='failed', receipt=None)
    effect(sqlite_engine, 'blocked', status='blocked', receipt=None, dispatched=False)
    effect(sqlite_engine, 'pending', status='reserved', receipt=None, dispatched=False)
    result = read(sqlite_engine)
    assert result['confirmed_count'] == 0 and result['unknown_count'] == 4
    assert result['failed_count'] == 2 and result['pending_count'] == 1
    assert result['published'] is None
    empty = read(sqlite_engine, date='2026-10-14')
    assert empty['published'] is False and not empty['records']
    assert empty['coverage'] == 'system_publication_receipts_only'


def test_self_publications_shanghai_midnight_and_truncated_exact_counts(sqlite_engine):
    effect(sqlite_engine, 'previous-day', stamp=datetime(2026, 10, 14, 15, 59, 59, tzinfo=UTC))
    for i in range(65):
        effect(sqlite_engine, f'today-{i:03}', receipt=f'post-{i}',
               stamp=datetime(2026, 10, 14, 16, tzinfo=UTC) + timedelta(seconds=i))
    result = read(sqlite_engine)
    assert result['confirmed_count'] == 65 and len(result['records']) == 64
    assert result['records_truncated'] is True
    assert read(sqlite_engine, date='2026-10-14')['confirmed_count'] == 1


@pytest.mark.parametrize('date', [True, '2026-02-30', '2026-10-16', 'anything', '2026-1-1'])
def test_self_publications_reject_invalid_or_future_date(sqlite_engine, date):
    with pytest.raises(ValueError):
        read(sqlite_engine, date=date)
