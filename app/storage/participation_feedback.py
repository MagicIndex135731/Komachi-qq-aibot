"""Source-bound temporary group participation preferences, never long-term facts."""
from datetime import UTC, datetime, timedelta

from sqlalchemy import DateTime, ForeignKeyConstraint, Integer, String
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.dialects.sqlite import insert

from app.core.memory_source_guard import is_usable_memory_source
from app.core.time_utils import shanghai_aware
from app.storage.models import Base, Message


class GroupParticipationFeedback(Base):
    __tablename__ = 'group_participation_feedback'
    __table_args__ = (ForeignKeyConstraint(
        ['source_message_id', 'group_id'], ['messages.id', 'messages.group_id'], ondelete='CASCADE'),)

    group_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_message_id: Mapped[int] = mapped_column(Integer)
    quiet_until: Mapped[datetime] = mapped_column(DateTime)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    reason: Mapped[str] = mapped_column(String(48))


def feedback_for(session: Session, *, group_id: int, bot_id: int, now: datetime) -> GroupParticipationFeedback | None:
    row = session.get(GroupParticipationFeedback, group_id)
    if row is None or row.expires_at <= now.astimezone(UTC).replace(tzinfo=None):
        return None
    source = session.get(Message, row.source_message_id)
    if not is_usable_memory_source(source, group_id=group_id) or source.user_id == bot_id:
        return None
    return row


def save_feedback(session: Session, *, source: Message, reason: str, quiet_seconds: int, duration_seconds: int) -> bool:
    # Durable canonical IDs make replays/restarts idempotent. Use source time,
    # never wall-clock replay time, so old requests cannot extend the penalty.
    received = shanghai_aware(source.timestamp).astimezone(UTC).replace(tzinfo=None)
    values = dict(group_id=source.group_id, source_message_id=source.id,
        quiet_until=received + timedelta(seconds=quiet_seconds),
        expires_at=received + timedelta(seconds=duration_seconds), reason=reason)
    statement = insert(GroupParticipationFeedback).values(**values)
    result = session.execute(statement.on_conflict_do_update(index_elements=['group_id'],
        set_={key:value for key,value in values.items() if key!='group_id'},
        where=GroupParticipationFeedback.source_message_id < source.id))
    return bool(result.rowcount)
