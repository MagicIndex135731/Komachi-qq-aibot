"""Only a possible or confirmed bot delivery may refresh chat cooldown."""
from datetime import UTC, datetime, timedelta

import pytest

from app.storage.db import session_scope
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository


@pytest.mark.parametrize('state,counts', [
    ('deleted', False), ('reserved', False), ('sent', True), ('uncertain', True), (None, True),
])
def test_failed_output_does_not_refresh_cooldown_but_unknown_does(sqlite_engine, state, counts):
    group, bot = 900000001, 900000200
    now = datetime(2026, 10, 17, 12, 39, 48, tzinfo=UTC)
    before = now - timedelta(hours=1)
    with session_scope(sqlite_engine) as session:
        GroupRepository(session).upsert_group(group_id=group, group_name='', enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=bot, nickname='synthetic bot', group_card='')
        messages = MessageRepository(session)
        for identifier, stamp, delivery in [('old-sent', before, 'sent'), ('latest-output', now, state)]:
            messages.add_group_message(platform_msg_id=identifier, group_id=group, user_id=bot,
                timestamp=stamp, plain_text='synthetic output', msg_type='text', reply_to_msg_id=None,
                mentioned_bot=False, raw_json={'delivery_state': delivery} if delivery else {})
    with session_scope(sqlite_engine) as session:
        messages = MessageRepository(session)
        assert messages.last_bot_reply_at(group_id=group, bot_user_id=bot) == (now if counts else before)
        assert messages.last_bot_reply_at(group_id=900000002, bot_user_id=bot) is None
        assert messages.last_bot_reply_at(group_id=group, bot_user_id=900000201) is None
