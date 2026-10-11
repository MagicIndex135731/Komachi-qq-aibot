from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event

from app.storage.db import build_engine, create_all, session_scope
from app.storage.models import Message
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository
from app.core.time_utils import ASIA_SHANGHAI


@pytest.mark.parametrize('method', [
    'list_recent_group_messages',
    'list_recent_group_messages_for_summarization',
    'list_recent_group_messages_for_user',
    'list_recent_group_messages_for_user_since',
    'list_recent_private_messages_for_user_since',
    'list_recent_group_inbound_messages',
    'last_bot_reply_at',
])
def test_recent_reads_materialize_only_needed_pages_and_keep_delivery_policy(tmp_path, method):
    engine = build_engine(tmp_path / 'bot.db')
    create_all(engine)
    private = 'private' in method
    now = datetime(2026, 10, 10, tzinfo=UTC)
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(group_id=10001, group_name='fixture', enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=20001, nickname='fixture', group_card='')
        session.flush()
        # Identical timestamps exercise stable ID pagination. The newest rows
        # force a second page so LIMIT-before-filter cannot silently lose data.
        session.add_all([Message(platform_msg_id=f'm-{i}', group_id=None if private else 10001,
            user_id=20001, timestamp=now.replace(tzinfo=None), plain_text='fixture',
            raw_json={'payload': 'x' * 2048, **({'delivery_state': 'reserved'} if i >= 297 else
                {'delivery_state': 'uncertain', 'failure_kind': 'delivery_result_unknown'} if i == 296 else {})},
            msg_type='text', mentioned_bot=False) for i in range(300)])
    loaded = []
    def observe(row, context):
        loaded.append(row.id)
    event.listen(Message, 'load', observe)
    try:
        with session_scope(engine) as session:
            repo = MessageRepository(session)
            kwargs = {'limit': 4, **({} if private else {'group_id': 10001})}
            if 'user' in method:
                kwargs['user_id'] = 20001
            if 'since' in method:
                kwargs['since'] = now - timedelta(days=1)
            if 'inbound' in method:
                kwargs['bot_user_id'] = 20002
            if method == 'last_bot_reply_at':
                del kwargs['limit']
                kwargs['bot_user_id'] = 20001
                value = getattr(repo, method)(**kwargs)
                assert value == now.replace(tzinfo=ASIA_SHANGHAI).astimezone(UTC)
                identifiers = []
            else:
                rows = getattr(repo, method)(**kwargs)
                identifiers = [row.platform_msg_id for row in rows]
        skip_unconfirmed = method in {'list_recent_group_messages_for_summarization',
            'list_recent_group_messages_for_user', 'list_recent_group_inbound_messages'}
        start = 292 if skip_unconfirmed else 293
        if method != 'last_bot_reply_at':
            assert identifiers == [f'm-{i}' for i in range(start, start + 4)]
        assert len(loaded) <= 8, 'recent context must not materialize the full historical message table'
    finally:
        event.remove(Message, 'load', observe)
        engine.dispose()
