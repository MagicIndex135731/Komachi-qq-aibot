from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.core.memory_source_guard import is_usable_memory_source
from app.storage.agent_sessions import AgentSessionRepository, SessionRevisionConflict
from app.storage.context_resets import ContextResetRepository
from app.storage.db import create_all, session_scope
from app.storage.models import Message, Job
from app.storage.repositories import DevSessionRepository, GroupRepository, JobRepository, MessageRepository, UserRepository


ACCOUNT, OWNER, GROUP, OTHER = 900000099, 900000101, 900000001, 900000002


def source(engine, name, *, group=GROUP, user=OWNER, text='上下文原文', direction=None):
    with session_scope(engine) as session:
        UserRepository(session).upsert_user(user_id=user, nickname='合成', group_card='')
        repo = MessageRepository(session)
        raw = {'message': [{'type': 'text', 'data': {'text': text}}]}
        if direction:
            raw.update(direction=direction, recipient_user_id=OWNER, delivery_state='sent')
        if group is not None:
            GroupRepository(session).upsert_group(group_id=group, group_name='合成', enabled=True, speak_enabled=True)
            row = repo.add_group_message(platform_msg_id=name, group_id=group, user_id=user,
                timestamp=datetime.now(UTC), plain_text=text, raw_json=raw, msg_type='text',
                reply_to_msg_id=None, mentioned_bot=True)
        else:
            row = repo.add_private_message(platform_msg_id=name, user_id=user,
                timestamp=datetime.now(UTC), plain_text=text, raw_json=raw, msg_type='text')
        session.flush()
        return row.id


def test_reset_is_persistent_exact_bounded_and_keeps_originals(sqlite_engine, tmp_path):
    older = source(sqlite_engine, 'older')
    excluded = [source(sqlite_engine, str(i)) for i in range(3)]
    other = source(sqlite_engine, 'other', group=OTHER)
    command = source(sqlite_engine, 'reset', text='重置上下文 3')
    newer = source(sqlite_engine, 'newer')
    sessions = AgentSessionRepository(sqlite_engine, account_id=ACCOUNT, session_root=tmp_path)
    state = sessions.save(f'group:{GROUP}', [{'role': 'user', 'content': '污染内容'}],
        {'runtime_id': 'old-native', 'receipts': [{'call_id': 'unknown', 'status': 'uncertain', 'text': '污染内容'}]},
        excluded, expected_revision=0)
    reset = ContextResetRepository(sqlite_engine, account_id=ACCOUNT)
    result = reset.reset(scope=f'group:{GROUP}', actor_id=OWNER, source_id=command, count=3)
    assert result['excluded_count'] == 3 and result['applied']
    assert not reset.reset(scope=f'group:{GROUP}', actor_id=OWNER, source_id=command, count=3)['applied']
    create_all(sqlite_engine)  # Fresh initializer observes the same persisted mask.
    with session_scope(sqlite_engine) as session:
        assert all(not is_usable_memory_source(session.get(Message, i)) for i in [*excluded, command])
        assert all(is_usable_memory_source(session.get(Message, i)) for i in [older, other, newer])
        for i in excluded:
            row = session.get(Message, i)
            assert row.plain_text == '上下文原文' and row.raw_json['message'][0]['data']['text'] == row.plain_text
    restored = sessions.read(f'group:{GROUP}')
    assert restored['items'] == [] and restored['context_source_message_ids'] == []
    assert restored['checkpoint']['runtime_id'] != 'old-native'
    assert '污染内容' not in json.dumps(restored, ensure_ascii=False)
    assert restored['checkpoint']['receipts'][0]['status'] == 'uncertain'
    with pytest.raises(SessionRevisionConflict):
        sessions.save(f'group:{GROUP}', state['items'], state['checkpoint'], excluded,
            expected_revision=state['revision'])


def test_private_reset_counts_only_this_conversation(sqlite_engine):
    inbound = source(sqlite_engine, 'private-in', group=None)
    outbound = source(sqlite_engine, 'private-out', group=None, user=ACCOUNT, direction='outbound')
    other_user = source(sqlite_engine, 'private-other', group=None, user=900000103)
    command = source(sqlite_engine, 'private-reset', group=None, text='重置上下文 2')
    result = ContextResetRepository(sqlite_engine, account_id=ACCOUNT).reset(
        scope=f'private:{OWNER}', actor_id=OWNER, source_id=command, count=2)
    assert result['excluded_count'] == 2
    with session_scope(sqlite_engine) as session:
        assert not is_usable_memory_source(session.get(Message, inbound))
        assert not is_usable_memory_source(session.get(Message, outbound))
        assert is_usable_memory_source(session.get(Message, other_user))


def test_cannot_reset_another_scope(sqlite_engine):
    command = source(sqlite_engine, 'reset-other', group=OTHER, text='重置上下文')
    with pytest.raises(ValueError):
        ContextResetRepository(sqlite_engine, account_id=ACCOUNT).reset(
            scope=f'group:{GROUP}', actor_id=OWNER, source_id=command, count=30)


def test_reset_cancels_old_returns_but_preserves_newly_arrived_jobs(sqlite_engine):
    old = source(sqlite_engine, 'old-return', group=None)
    command = source(sqlite_engine, 'reset-return', group=None, text='重置上下文 1')
    new = source(sqlite_engine, 'new-request', group=None)
    with session_scope(sqlite_engine) as session:
        for original in (old, new):
            for job_type, prefix in [('private_agent_turn', 'private-agent'), ('agent_media_operation', 'media'),
                    ('delegated_research', 'research')]:
                key = f'{prefix}:{ACCOUNT}:' + ('' if prefix == 'private-agent' else 'private:') + f'{OWNER}:{original}'
                JobRepository(session).add_job(job_type=job_type, job_key=key, status='queued',
                    run_at=datetime.now(UTC), payload_json={'account_id': ACCOUNT, 'surface': 'private',
                        'user_id': OWNER, 'scope_id': OWNER, 'source_id': original})
    ContextResetRepository(sqlite_engine, account_id=ACCOUNT).reset(scope=f'private:{OWNER}',
        actor_id=OWNER, source_id=command, count=1)
    with session_scope(sqlite_engine) as session:
        for job in session.scalars(select(Job)):
            assert job.status == ('cancelled' if job.payload_json['source_id'] == old else 'queued')


def test_late_summary_completion_cannot_reactivate_pre_reset_daily_session(sqlite_engine):
    with session_scope(sqlite_engine) as session:
        previous = DevSessionRepository(session).create_owner_session(owner_qq=OWNER, session_mode='daily')
        previous_id = previous.id
    command = source(sqlite_engine, 'summary-reset', group=None, text='重置上下文')
    ContextResetRepository(sqlite_engine, account_id=ACCOUNT).reset(scope=f'private:{OWNER}',
        actor_id=OWNER, source_id=command, count=40)
    with session_scope(sqlite_engine) as session:
        repo = DevSessionRepository(session)
        repo.update_session(session_id=previous_id, summary='LATE_SUMMARY_POISON')
        assert repo.get_latest_owner_session(owner_qq=OWNER, session_mode='daily').summary == ''


def test_private_drawing_from_before_reset_cannot_deliver_even_outside_mask(sqlite_engine):
    from app.core.group_image_generation import PrivateImageGenerationService
    older = source(sqlite_engine, 'drawing-old', group=None)
    source(sqlite_engine, 'drawing-recent', group=None)
    command = source(sqlite_engine, 'drawing-reset', group=None, text='重置上下文 1')
    new = source(sqlite_engine, 'drawing-new', group=None)
    ContextResetRepository(sqlite_engine, account_id=ACCOUNT).reset(scope=f'private:{OWNER}',
        actor_id=OWNER, source_id=command, count=1)
    service = PrivateImageGenerationService.__new__(PrivateImageGenerationService)
    service.engine = sqlite_engine
    request = SimpleNamespace(user_id=OWNER, agent_source_message_id='drawing-old', agent_reference_source_ids=())
    with session_scope(sqlite_engine) as session:
        assert is_usable_memory_source(session.get(Message, older))
    assert not service._agent_private_source_usable(request)
    request.agent_source_message_id = 'drawing-new'
    assert service._agent_private_source_usable(request)


def test_reset_invalidates_old_dependency_snapshot_and_running_group_generation(sqlite_engine, tmp_path):
    from app.core.context_reset import ContextResetGuard
    from app.storage.agent_sessions import read_context_snapshot
    older = source(sqlite_engine, 'older-unmasked')
    sessions = AgentSessionRepository(sqlite_engine, account_id=ACCOUNT, session_root=tmp_path)
    scope = f'group:{GROUP}'
    snapshot = sessions.freeze_dependencies(scope, (older,), authority_scope=scope)
    repo = ContextResetRepository(sqlite_engine, account_id=ACCOUNT)
    guard = ContextResetGuard(SimpleNamespace(valid=lambda: True), repo, scope)
    assert guard.valid()
    source(sqlite_engine, 'last-masked')
    command = source(sqlite_engine, 'generation-reset', text='重置上下文 1')
    repo.reset(scope=scope, actor_id=OWNER, source_id=command, count=1)
    assert not guard.valid()
    assert read_context_snapshot(sqlite_engine, account_id=ACCOUNT, scope=scope,
        identifier=snapshot, now=datetime.now(UTC)) is None
    with session_scope(sqlite_engine) as session:
        assert is_usable_memory_source(session.get(Message, older))
