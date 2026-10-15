"""Validate immutable provenance inherited by asynchronous work and artifacts."""
from datetime import UTC, datetime

from sqlalchemy import select

from app.core.memory_source_guard import is_usable_memory_source
from app.storage.agent_sessions import read_context_snapshot
from app.storage.db import session_scope
from app.storage.models import Message


def context_snapshot_valid(engine, *, account_id, scope, identifier, group_id=None,
                           private_user_id=None, now=None):
    now = now or datetime.now(UTC)
    value = read_context_snapshot(engine, account_id=account_id, scope=scope, identifier=identifier, now=now)
    if value is None:
        return False
    sources = value['sources']
    with session_scope(engine) as session:
        for index in range(0, len(sources), 400):
            batch = sources[index:index + 400]
            rows = list(session.scalars(select(Message).where(Message.id.in_(batch))))
            if len(rows) != len(batch) or any(row.group_id != group_id
                    or not is_usable_memory_source(row, group_id=group_id,
                        **({'author_id': private_user_id} if private_user_id is not None else {}))
                    or private_user_id is not None and (row.raw_json or {}).get('direction') == 'outbound'
                    for row in rows):
                return False
    if group_id is None:
        return not value['memory']
    from app.agent.memory_integration import retained_memory_dependencies_valid
    return retained_memory_dependencies_valid(engine, group_id=group_id, dependencies=value['memory'], now=now)
