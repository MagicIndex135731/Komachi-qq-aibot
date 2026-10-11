from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from zoneinfo import ZoneInfo

from sqlalchemy import DateTime, ForeignKeyConstraint, Index, Integer, JSON, String, Text, UniqueConstraint, delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, mapped_column

from app.adapters.onebot_models import GroupMessageEvent
from app.core.message_content import extract_images_from_raw_payload
from app.core.memory_source_guard import is_usable_memory_source
from app.storage.models import Base, Message


def _utc_naive(value: datetime | None = None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


class GroupAgentSession(Base):
    __tablename__ = "group_agent_sessions"

    group_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cursor: Mapped[int] = mapped_column(Integer, default=0)
    epoch: Mapped[int] = mapped_column(Integer, default=0)
    working_note: Mapped[str] = mapped_column(Text, default="")
    working_note_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    working_note_source_message_ids: Mapped[list] = mapped_column(JSON, default=list)
    last_sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive)


class GroupAgentInbox(Base):
    __tablename__ = "group_agent_inbox"
    __table_args__ = (
        UniqueConstraint("group_id", "platform_msg_id", name="ux_group_agent_inbox_source"),
        ForeignKeyConstraint(
            ["source_message_id", "group_id"], ["messages.id", "messages.group_id"],
            name="fk_group_agent_inbox_source_group",
        ),
        Index("ix_group_agent_inbox_pending", "group_id", "status", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(Integer)
    source_message_id: Mapped[int] = mapped_column(Integer)
    platform_msg_id: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(24), default="pending")
    turn_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive)


class GroupAgentTurn(Base):
    __tablename__ = "group_agent_turns"
    __table_args__ = (Index("ix_group_agent_turn_group", "group_id", "created_at"),)

    turn_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    group_id: Mapped[int] = mapped_column(Integer)
    epoch: Mapped[int] = mapped_column(Integer)
    source_message_ids: Mapped[list] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(24), default="deciding")
    decision: Mapped[str] = mapped_column(String(16), default="")
    reason: Mapped[str] = mapped_column(String(160), default="")
    receipt_ids: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive)


@dataclass(frozen=True, slots=True)
class AgentSessionSnapshot:
    group_id: int
    cursor: int
    epoch: int
    working_note: str
    last_sent_at: datetime | None
    working_note_source_message_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ClaimedAgentTurn:
    turn_id: str
    events: tuple[GroupMessageEvent, ...]
    session: AgentSessionSnapshot


class GroupAgentRepository:
    """Only source references and a bounded working note are Agent-owned data.

    Canonical messages retain all identities, images, originals and deletion
    metadata. Inbox entries are explicitly live observations: startup history
    ingestion alone must never enqueue them.
    """

    FINAL_STATUSES = frozenset({"sent", "uncertain", "blocked", "failed", "silent", "expired"})

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def _session_scope(self):
        # db imports this module to register mappings before create_all.
        from app.storage.db import session_scope
        return session_scope(self.engine)

    @staticmethod
    def _snapshot(row: GroupAgentSession) -> AgentSessionSnapshot:
        last_sent = row.last_sent_at
        if last_sent is not None and last_sent.tzinfo is None:
            last_sent = last_sent.replace(tzinfo=UTC)
        note_sources = tuple(row.working_note_source_message_ids or ()) if row.working_note else ()
        return AgentSessionSnapshot(
            row.group_id, row.cursor, row.epoch, row.working_note, last_sent, note_sources,
        )

    @staticmethod
    def _usable_note_sources(session, *, group_id: int, source_ids: list[int]) -> bool:
        if not source_ids or any(not str(value).isdigit() or int(value) <= 0 for value in source_ids):
            return False
        source_ids = tuple(dict.fromkeys(int(value) for value in source_ids))
        sources = list(session.scalars(select(Message).where(
            Message.group_id == group_id, Message.id.in_(source_ids),
        )))
        return len(sources) == len(source_ids) and all(
            is_usable_memory_source(source, group_id=group_id) for source in sources
        )

    @staticmethod
    def _clear_working_note(row: GroupAgentSession) -> None:
        row.working_note = ""
        row.working_note_updated_at = None
        row.working_note_source_message_ids = []

    def _validate_working_note(
        self, session, row: GroupAgentSession, *, max_age_seconds: float, max_sources: int,
    ) -> bool:
        """Clear stale or untraceable notes before exposing them to any turn.

        Provenance is capped before the SQL query, including legacy rows that
        predate the bound. Truncating source IDs while retaining the text would
        hide which omitted messages contributed to that text.
        """
        if not row.working_note:
            self._clear_working_note(row)
            return True
        source_ids = row.working_note_source_message_ids or []
        if (
            row.working_note_updated_at is None
            or _utc_naive(row.working_note_updated_at) < _utc_naive() - timedelta(seconds=max_age_seconds)
            or len(source_ids) > max_sources
            or not self._usable_note_sources(session, group_id=row.group_id, source_ids=source_ids)
        ):
            self._clear_working_note(row)
            return False
        return True

    def enqueue(self, *, group_id: int, platform_msg_id: str, bot_qq: int, max_pending: int) -> bool:
        with self._session_scope() as session:
            duplicate = session.scalar(select(GroupAgentInbox.id).where(
                GroupAgentInbox.group_id == group_id,
                GroupAgentInbox.platform_msg_id == platform_msg_id,
            ))
            if duplicate is not None:
                return False
            source = session.scalar(select(Message).where(
                Message.group_id == group_id,
                Message.platform_msg_id == platform_msg_id,
                Message.user_id != bot_qq,
            ))
            if source is None:
                return False
            raw = source.raw_json if isinstance(source.raw_json, dict) else {}
            if raw.get("direction") == "outbound" or not is_usable_memory_source(source, group_id=group_id):
                return False
            session.add(GroupAgentInbox(
                group_id=group_id, source_message_id=source.id,
                platform_msg_id=platform_msg_id, received_at=_utc_naive(),
            ))
            session.flush()
            pending = list(session.scalars(select(GroupAgentInbox).where(
                GroupAgentInbox.group_id == group_id,
                GroupAgentInbox.status == "pending",
            ).order_by(GroupAgentInbox.id)))
            for old in pending[:-max(1, max_pending)]:
                old.status = "expired"
            return True

    def recover(self) -> tuple[int, ...]:
        """Never replay an operation that crossed the durable dispatch marker."""
        with self._session_scope() as session:
            for state in session.scalars(select(GroupAgentSession)):
                state.epoch += 1
                state.updated_at = _utc_naive()
            interrupted = list(session.scalars(select(GroupAgentTurn).where(
                GroupAgentTurn.status.in_(["deciding", "ready", "dispatching"]),
            )))
            for turn in interrupted:
                dispatched = turn.status == "dispatching"
                turn.status = "uncertain" if dispatched else "interrupted"
                turn.reason = "recovered_after_dispatch" if dispatched else "recovered_before_dispatch"
                turn.updated_at = _utc_naive()
                inbox = list(session.scalars(select(GroupAgentInbox).where(
                    GroupAgentInbox.turn_id == turn.turn_id,
                    GroupAgentInbox.group_id == turn.group_id,
                    GroupAgentInbox.status == "processing",
                )))
                for item in inbox:
                    item.status = "done" if dispatched else "pending"
                    if not dispatched:
                        item.turn_id = None
                if dispatched:
                    state = session.get(GroupAgentSession, turn.group_id)
                    if state is not None and inbox:
                        state.cursor = max(state.cursor, max(item.id for item in inbox))
            session.flush()
            return tuple(session.scalars(select(GroupAgentInbox.group_id).where(
                GroupAgentInbox.status == "pending",
            ).distinct().order_by(GroupAgentInbox.group_id)))

    def begin_session(
        self, group_id: int, *, working_note_max_age_seconds: float = 1800.0,
        working_note_max_sources: int = 200,
    ) -> AgentSessionSnapshot:
        with self._session_scope() as session:
            row = session.get(GroupAgentSession, group_id)
            if row is None:
                row = GroupAgentSession(group_id=group_id, epoch=1)
                session.add(row)
            else:
                self._validate_working_note(
                    session, row, max_age_seconds=working_note_max_age_seconds,
                    max_sources=working_note_max_sources,
                )
                row.epoch += 1
                row.updated_at = _utc_naive()
            session.flush()
            return self._snapshot(row)

    def get_session(self, group_id: int) -> AgentSessionSnapshot | None:
        with self._session_scope() as session:
            row = session.get(GroupAgentSession, group_id)
            return self._snapshot(row) if row is not None else None

    def epoch_is_current(self, group_id: int, epoch: int) -> bool:
        with self._session_scope() as session:
            value = session.scalar(select(GroupAgentSession.epoch).where(GroupAgentSession.group_id == group_id))
            return value == epoch

    def has_pending(self, group_id: int) -> bool:
        with self._session_scope() as session:
            return session.scalar(select(GroupAgentInbox.id).where(
                GroupAgentInbox.group_id == group_id,
                GroupAgentInbox.status == "pending",
            ).limit(1)) is not None

    def pending_group_ids(self) -> tuple[int, ...]:
        with self._session_scope() as session:
            return tuple(session.scalars(select(GroupAgentInbox.group_id).where(
                GroupAgentInbox.status == "pending",
            ).distinct().order_by(GroupAgentInbox.group_id)))

    @staticmethod
    def _event(source: Message) -> GroupMessageEvent:
        raw = dict(source.raw_json) if isinstance(source.raw_json, dict) else {}
        sender = raw.get("sender") if isinstance(raw.get("sender"), dict) else {}
        timestamp = source.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        return GroupMessageEvent(
            platform_msg_id=source.platform_msg_id, group_id=int(source.group_id),
            user_id=source.user_id, nickname=str(sender.get("nickname", "")),
            group_card=str(sender.get("card", "")), plain_text=source.plain_text,
            raw_payload=raw, timestamp=timestamp.astimezone(UTC), msg_type=source.msg_type,
            images=extract_images_from_raw_payload(raw), mentioned_bot=source.mentioned_bot,
            reply_to_msg_id=source.reply_to_msg_id,
        )

    def claim(
        self, *, group_id: int, epoch: int, max_messages: int, max_age_seconds: float,
        bot_qq: int | None = None, working_note_max_age_seconds: float = 1800.0,
        working_note_max_sources: int = 200,
    ) -> ClaimedAgentTurn | None:
        with self._session_scope() as session:
            state = session.get(GroupAgentSession, group_id)
            if state is None or state.epoch != epoch:
                return None
            self._validate_working_note(
                session, state, max_age_seconds=working_note_max_age_seconds,
                max_sources=working_note_max_sources,
            )
            pending = list(session.scalars(select(GroupAgentInbox).where(
                GroupAgentInbox.group_id == group_id,
                GroupAgentInbox.status == "pending",
            ).order_by(GroupAgentInbox.id).limit(max(1, max_messages))))
            if not pending:
                return None
            cutoff = _utc_naive() - timedelta(seconds=max_age_seconds)
            retained: list[GroupAgentInbox] = []
            events: list[GroupMessageEvent] = []
            for item in pending:
                source = session.get(Message, item.source_message_id)
                source_at = (
                    source.timestamp.replace(tzinfo=ZoneInfo("Asia/Shanghai")).astimezone(UTC).replace(tzinfo=None)
                    if source is not None and source.timestamp.tzinfo is None
                    else _utc_naive(source.timestamp) if source is not None else None
                )
                if item.received_at < cutoff or source is None or source.group_id != group_id or source_at < cutoff:
                    item.status = "expired"
                    state.cursor = max(state.cursor, item.id)
                    continue
                raw = source.raw_json if isinstance(source.raw_json, dict) else {}
                if raw.get("direction") == "outbound" or not is_usable_memory_source(source, group_id=group_id):
                    item.status = "expired"
                    state.cursor = max(state.cursor, item.id)
                    continue
                retained.append(item)
                events.append(self._event(source))
                quoted_bot = False
                if source.reply_to_msg_id:
                    quoted = session.scalar(select(Message).where(
                        Message.group_id == group_id,
                        Message.platform_msg_id == source.reply_to_msg_id,
                    ))
                    # Check the known account identity, including canonical
                    # live echoes whose ID came from the QQ receipt.
                    quoted_bot = bool(quoted is not None and (
                        quoted.user_id == bot_qq
                        or (bot_qq is None and isinstance(quoted.raw_json, dict)
                            and quoted.raw_json.get("direction") == "outbound")
                    ))
                if source.mentioned_bot or quoted_bot:
                    # Direct requests remain independent turns even when two
                    # users mention the bot during the same quiet window.
                    break
            if not retained:
                return None
            digest = sha256((str(group_id) + ":" + ",".join(str(item.id) for item in retained)).encode()).hexdigest()
            turn_id = "agent-" + digest[:48]
            turn = session.get(GroupAgentTurn, turn_id)
            if turn is not None and turn.status in self.FINAL_STATUSES:
                for item in retained:
                    item.status = "done"
                state.cursor = max(state.cursor, retained[-1].id)
                return None
            if turn is None:
                turn = GroupAgentTurn(
                    turn_id=turn_id, group_id=group_id, epoch=epoch,
                    source_message_ids=[item.source_message_id for item in retained],
                )
                session.add(turn)
            else:
                turn.status = "deciding"
                turn.epoch = epoch
                turn.updated_at = _utc_naive()
            for item in retained:
                item.status = "processing"
                item.turn_id = turn_id
            session.flush()
            return ClaimedAgentTurn(turn_id, tuple(events), self._snapshot(state))

    def record_decision(self, *, turn_id: str, action: str, reason: str) -> None:
        with self._session_scope() as session:
            turn = session.get(GroupAgentTurn, turn_id)
            if turn is None or turn.status != "deciding":
                raise ValueError("Agent turn is not deciding")
            turn.decision = action[:16]
            turn.reason = reason[:160]
            turn.status = "ready"
            turn.updated_at = _utc_naive()

    def mark_dispatching(
        self, *, turn_id: str, group_id: int, epoch: int,
        working_note_source_message_ids: tuple[int, ...] = (),
    ) -> bool:
        with self._session_scope() as session:
            state = session.get(GroupAgentSession, group_id)
            turn = session.get(GroupAgentTurn, turn_id)
            if state is None or state.epoch != epoch or turn is None or turn.group_id != group_id or turn.epoch != epoch:
                return False
            if turn.status not in {"ready", "dispatching"}:
                return False
            if working_note_source_message_ids and not self._usable_note_sources(
                session, group_id=group_id, source_ids=working_note_source_message_ids,
            ):
                return False
            sources = list(session.scalars(select(Message).where(
                Message.group_id == group_id, Message.id.in_(turn.source_message_ids),
            )))
            if len(sources) != len(turn.source_message_ids) or any(
                not is_usable_memory_source(source, group_id=group_id)
                or isinstance(source.raw_json, dict) and source.raw_json.get("direction") == "outbound"
                for source in sources
            ):
                return False
            turn.status = "dispatching"
            turn.updated_at = _utc_naive()
            return True

    def finish(
        self, *, turn_id: str, status: str, working_note: str | None = None,
        receipt_ids: tuple[str, ...] = (), reason: str = "", epoch: int | None = None,
        working_note_max_age_seconds: float = 1800.0, working_note_max_sources: int = 200,
        charge_proactive_cadence: bool = True,
    ) -> None:
        if type(charge_proactive_cadence) is not bool or status not in self.FINAL_STATUSES | {"interrupted"}:
            raise ValueError("invalid Agent terminal status")
        with self._session_scope() as session:
            turn = session.get(GroupAgentTurn, turn_id)
            if turn is None or turn.status in self.FINAL_STATUSES:
                return
            if epoch is not None and turn.epoch != epoch:
                return
            turn.status = status
            turn.updated_at = _utc_naive()
            turn.receipt_ids = list(dict.fromkeys(str(value)[:128] for value in receipt_ids if str(value)))[:20]
            if reason:
                turn.reason = reason[:160]
            state = session.get(GroupAgentSession, turn.group_id)
            inbox = list(session.scalars(select(GroupAgentInbox).where(
                GroupAgentInbox.group_id == turn.group_id,
                GroupAgentInbox.turn_id == turn_id,
                GroupAgentInbox.status == "processing",
            )))
            for item in inbox:
                item.status = "pending" if status == "interrupted" else "done"
                if status == "interrupted":
                    item.turn_id = None
            if state is not None and state.epoch == turn.epoch:
                if working_note is not None:
                    note = working_note[:2000]
                    prior_valid = self._validate_working_note(
                        session, state, max_age_seconds=working_note_max_age_seconds,
                        max_sources=working_note_max_sources,
                    )
                    source_ids = list(dict.fromkeys(
                        list(state.working_note_source_message_ids or ())
                        + list(turn.source_message_ids or ())
                    ))
                    if note and (
                        not prior_valid or len(source_ids) > working_note_max_sources
                        or not self._usable_note_sources(session, group_id=turn.group_id, source_ids=source_ids)
                    ):
                        note = ""
                    state.working_note = note
                    state.working_note_updated_at = _utc_naive() if note else None
                    state.working_note_source_message_ids = source_ids if note else []
                if status == "sent" and charge_proactive_cadence:
                    state.last_sent_at = _utc_naive()
                if status != "interrupted" and inbox:
                    state.cursor = max(state.cursor, max(item.id for item in inbox))
                state.updated_at = _utc_naive()

    def turn_status(self, turn_id: str) -> str | None:
        with self._session_scope() as session:
            return session.scalar(select(GroupAgentTurn.status).where(GroupAgentTurn.turn_id == turn_id))

    def prune(self, *, completed_before: datetime) -> None:
        """Keep canonical messages and all unresolved turn records untouched."""
        with self._session_scope() as session:
            old_ids = select(GroupAgentTurn.turn_id).where(
                GroupAgentTurn.status.in_(self.FINAL_STATUSES - {"uncertain"}),
                GroupAgentTurn.updated_at < _utc_naive(completed_before),
            )
            session.execute(delete(GroupAgentInbox).where(
                GroupAgentInbox.status.in_(["done", "expired"]),
                GroupAgentInbox.turn_id.in_(old_ids),
            ))
            session.execute(delete(GroupAgentTurn).where(GroupAgentTurn.turn_id.in_(old_ids)))
            session.execute(delete(GroupAgentInbox).where(
                GroupAgentInbox.status == "expired", GroupAgentInbox.turn_id.is_(None),
                GroupAgentInbox.received_at < _utc_naive(completed_before),
            ))
