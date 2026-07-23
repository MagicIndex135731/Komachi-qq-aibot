from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Group(Base):
    __tablename__ = "groups"

    group_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    group_name: Mapped[str] = mapped_column(String(255), default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    speak_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    reply_mode: Mapped[str] = mapped_column(String(32), default="balanced")
    cooldown_seconds: Mapped[int] = mapped_column(Integer, default=90)
    persona_variant: Mapped[str] = mapped_column(String(64), default="default")


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        Index("ix_users_nickname", "nickname"),
        Index("ix_users_group_card", "group_card"),
    )

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nickname: Mapped[str] = mapped_column(String(255), default="")
    group_card: Mapped[str] = mapped_column(String(255), default="")
    first_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    profile_summary: Mapped[str] = mapped_column(Text, default="")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("id", "group_id", name="ux_messages_id_group"),
        Index("ix_messages_group_reply", "group_id", "reply_to_msg_id"),
        Index("ix_messages_group_user", "group_id", "user_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform_msg_id: Mapped[str] = mapped_column(String(128), unique=True)
    group_id: Mapped[int | None] = mapped_column(ForeignKey("groups.group_id"), nullable=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw_json: Mapped[dict] = mapped_column(JSON)
    plain_text: Mapped[str] = mapped_column(Text, default="")
    msg_type: Mapped[str] = mapped_column(String(32), default="text")
    reply_to_msg_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    mentioned_bot: Mapped[bool] = mapped_column(Boolean, default=False)


class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope_type: Mapped[str] = mapped_column(String(32))
    scope_id: Mapped[str] = mapped_column(String(64))
    summary_level: Mapped[str] = mapped_column(String(32))
    start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    content: Mapped[str] = mapped_column(Text)
    source_count: Mapped[int] = mapped_column(Integer)
    # A stable topic or range key lets recursive summarization update its output
    # without retaining duplicate summaries for the same source material.
    summary_key: Mapped[str] = mapped_column(String(255), default="", index=True)
    source_start_msg_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_end_msg_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_summary_ids: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="active")


class MemoryItem(Base):
    __tablename__ = "memory_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope_type: Mapped[str] = mapped_column(String(32))
    scope_id: Mapped[str] = mapped_column(String(64))
    subject_type: Mapped[str] = mapped_column(String(32))
    subject_id: Mapped[str] = mapped_column(String(64))
    memory_kind: Mapped[str] = mapped_column(String(32))
    canonical_key: Mapped[str] = mapped_column(String(255), default="", index=True)
    predicate: Mapped[str] = mapped_column(String(96), default="")
    object_text: Mapped[str] = mapped_column(Text, default="")
    content: Mapped[str] = mapped_column(Text)
    importance: Mapped[int] = mapped_column(Integer, default=1)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    source_msg_id: Mapped[str] = mapped_column(String(128))
    source_msg_ids: Mapped[list] = mapped_column(JSON, default=list)
    mention_count: Mapped[int] = mapped_column(Integer, default=1)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Keep extraction provenance and the validity lifecycle separate: an old
    # fact remains auditable after a newer fact supersedes it.
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    supersedes_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    superseded_by_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_type: Mapped[str] = mapped_column(String(64))
    job_key: Mapped[str] = mapped_column(String(255), default="", index=True)
    payload_json: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    requested_generation: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    processed_generation: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    claimed_generation: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, server_default="3")
    locked_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error_code: Mapped[str] = mapped_column(String(96), default="", server_default="")
    backfill_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_generation: Mapped[str] = mapped_column(String(128), default="", server_default="")


class ConversationEpisode(Base):
    __tablename__ = "conversation_episodes"
    __table_args__ = (
        CheckConstraint(
            "status IN ('open','closed','processing','processed','ready','failed','superseded')",
            name="ck_conversation_episodes_status",
        ),
        ForeignKeyConstraint(
            ["start_message_id", "group_id"],
            ["messages.id", "messages.group_id"],
            name="fk_conversation_episodes_start_message_group",
        ),
        ForeignKeyConstraint(
            ["end_message_id", "group_id"],
            ["messages.id", "messages.group_id"],
            name="fk_conversation_episodes_end_message_group",
        ),
        UniqueConstraint("id", "group_id", name="ux_conversation_episodes_id_group"),
        UniqueConstraint(
            "group_id",
            "segmentation_version",
            "start_message_id",
            "end_message_id",
            name="ux_conversation_episodes_identity",
        ),
        Index(
            "ux_conversation_episodes_open_group",
            "group_id",
            unique=True,
            sqlite_where=text("status = 'open' AND is_current = 1"),
        ),
        Index(
            "ix_conversation_episodes_group_time",
            "group_id",
            "started_at",
            "ended_at",
            "id",
        ),
        Index(
            "ix_conversation_episodes_group_status",
            "group_id",
            "status",
            "ended_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.group_id"), nullable=False)
    segmentation_version: Mapped[str] = mapped_column(String(64), default="v2")
    status: Mapped[str] = mapped_column(String(32), default="open")
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    start_message_id: Mapped[int] = mapped_column(Integer, nullable=False)
    end_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    boundary_reason: Mapped[str] = mapped_column(String(64), default="")
    title: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str] = mapped_column(String(128), default="")
    compaction_version: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EpisodeMessage(Base):
    __tablename__ = "episode_messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["episode_id", "group_id"],
            ["conversation_episodes.id", "conversation_episodes.group_id"],
            name="fk_episode_messages_episode_group",
        ),
        ForeignKeyConstraint(
            ["message_id", "group_id"],
            ["messages.id", "messages.group_id"],
            name="fk_episode_messages_message_group",
        ),
        UniqueConstraint("episode_id", "message_id", name="ux_episode_messages_membership"),
        UniqueConstraint("episode_id", "ordinal", name="ux_episode_messages_ordinal"),
        UniqueConstraint("message_id", name="ux_episode_messages_message"),
        Index("ix_episode_messages_group_message", "group_id", "message_id"),
    )

    episode_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    group_id: Mapped[int] = mapped_column(Integer, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class RetrievalDocument(Base):
    __tablename__ = "retrieval_documents"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','active','inactive','superseded','failed')",
            name="ck_retrieval_documents_status",
        ),
        CheckConstraint(
            "embedding_status IN ('pending','ready','failed','disabled','stale')",
            name="ck_retrieval_documents_embedding_status",
        ),
        ForeignKeyConstraint(
            ["episode_id", "group_id"],
            ["conversation_episodes.id", "conversation_episodes.group_id"],
            name="fk_retrieval_documents_episode_group",
        ),
        UniqueConstraint("id", "group_id", name="ux_retrieval_documents_id_group"),
        UniqueConstraint(
            "scope_type",
            "scope_id",
            "document_kind",
            "source_table",
            "source_id",
            "content_hash",
            name="ux_retrieval_documents_source_version",
        ),
        Index(
            "ix_retrieval_documents_group_status_time",
            "group_id",
            "status",
            "start_at",
            "end_at",
        ),
        Index(
            "ix_retrieval_documents_episode",
            "group_id",
            "episode_id",
            "document_kind",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope_type: Mapped[str] = mapped_column(String(32))
    scope_id: Mapped[str] = mapped_column(String(64))
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.group_id"), nullable=False)
    episode_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    document_kind: Mapped[str] = mapped_column(String(64))
    source_table: Mapped[str] = mapped_column(String(64))
    source_id: Mapped[str] = mapped_column(String(255))
    start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    content_hash: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="active")
    embedding_provider: Mapped[str] = mapped_column(String(64), default="")
    embedding_model: Mapped[str] = mapped_column(String(255), default="")
    embedding_version: Mapped[str] = mapped_column(String(128), default="")
    embedding_dimensions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Only window/message evidence that is part of the vector contract is
    # eligible. Summary/fact documents are intentionally FTS-only unless a
    # future embedding policy opts them in explicitly.
    embedding_eligible: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    embedding_status: Mapped[str] = mapped_column(String(32), default="disabled")
    last_error_code: Mapped[str] = mapped_column(String(96), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class RetrievalDocumentMessage(Base):
    __tablename__ = "retrieval_document_messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "group_id"],
            ["retrieval_documents.id", "retrieval_documents.group_id"],
            name="fk_retrieval_document_messages_document_group",
        ),
        ForeignKeyConstraint(
            ["message_id", "group_id"],
            ["messages.id", "messages.group_id"],
            name="fk_retrieval_document_messages_message_group",
        ),
        UniqueConstraint(
            "document_id",
            "message_id",
            "role",
            name="ux_retrieval_document_messages_source",
        ),
        UniqueConstraint(
            "document_id",
            "ordinal",
            "role",
            name="ux_retrieval_document_messages_ordinal",
        ),
        Index("ix_retrieval_document_messages_group_message", "group_id", "message_id"),
    )

    document_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[str] = mapped_column(String(32), primary_key=True, default="source")
    group_id: Mapped[int] = mapped_column(Integer, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class RetrievalIndexState(Base):
    __tablename__ = "retrieval_index_state"
    __table_args__ = (
        UniqueConstraint("channel", "generation", name="ux_retrieval_index_state_generation"),
        Index(
            "ux_retrieval_index_state_active_channel",
            "channel",
            unique=True,
            sqlite_where=text("is_active = 1"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    channel: Mapped[str] = mapped_column(String(32))
    generation: Mapped[int] = mapped_column(Integer)
    physical_table: Mapped[str] = mapped_column(String(255))
    provider: Mapped[str] = mapped_column(String(64), default="")
    model: Mapped[str] = mapped_column(String(255), default="")
    dimensions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    version: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[str] = mapped_column(String(32), default="building")
    total_documents: Mapped[int] = mapped_column(Integer, default=0)
    indexed_documents: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class MemoryBackfillRun(Base):
    __tablename__ = "memory_backfill_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','running','completed','failed','cancelled')",
            name="ck_memory_backfill_runs_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_key: Mapped[str] = mapped_column(String(255), unique=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    snapshot_watermarks_json: Mapped[dict] = mapped_column(JSON, default=dict)
    manifest_json: Mapped[dict] = mapped_column(JSON, default=dict)
    segmentation_generation: Mapped[str] = mapped_column(String(128), default="")
    compaction_generation: Mapped[str] = mapped_column(String(128), default="")
    index_generation: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error_code: Mapped[str] = mapped_column(String(96), default="")


class MemoryLateArrivalPreparation(Base):
    """Idempotency ledger for destructive late-arrival resegmentation."""

    __tablename__ = "memory_late_arrival_preparations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["message_id", "group_id"],
            ["messages.id", "messages.group_id"],
            name="fk_memory_late_arrival_message_group",
        ),
        UniqueConstraint(
            "group_id",
            "message_id",
            "segmentation_generation",
            name="ux_memory_late_arrival_preparation_key",
        ),
        Index(
            "ix_memory_late_arrival_preparations_group_message",
            "group_id",
            "message_id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.group_id"), nullable=False)
    message_id: Mapped[int] = mapped_column(Integer, nullable=False)
    segmentation_generation: Mapped[str] = mapped_column(String(128))
    compaction_generation: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class UsageRecord(Base):
    __tablename__ = "usage_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    model: Mapped[str] = mapped_column(String(64))
    endpoint: Mapped[str] = mapped_column(String(32))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)


class BbotListenerCacheEntry(Base):
    __tablename__ = "bbot_listener_cache_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(Integer, index=True)
    platform: Mapped[str] = mapped_column(String(32), index=True)
    external_id: Mapped[str] = mapped_column(String(128), index=True)
    canonical_name: Mapped[str] = mapped_column(String(255), default="")
    aliases_json: Mapped[list] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(64), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DevSession(Base):
    __tablename__ = "dev_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_qq: Mapped[int] = mapped_column(Integer, index=True)
    session_mode: Mapped[str] = mapped_column(String(32), default="project")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    summary: Mapped[str] = mapped_column(Text, default="")
    last_task_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class DevTask(Base):
    __tablename__ = "dev_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("dev_sessions.id"), index=True)
    requested_by_qq: Mapped[int] = mapped_column(Integer, index=True)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw_request_text: Mapped[str] = mapped_column(Text)
    intent_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    files_read_json: Mapped[list] = mapped_column(JSON, default=list)
    files_changed_json: Mapped[list] = mapped_column(JSON, default=list)
    commands_run_json: Mapped[list] = mapped_column(JSON, default=list)
    restart_required: Mapped[bool] = mapped_column(Boolean, default=False)
    restart_result: Mapped[str] = mapped_column(String(64), default="")
    failure_reason: Mapped[str] = mapped_column(Text, default="")
    checkpoint_dir: Mapped[str] = mapped_column(Text, default="")
    result_text: Mapped[str] = mapped_column(Text, default="")


class DevTaskArtifact(Base):
    __tablename__ = "dev_task_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("dev_tasks.id"), index=True)
    artifact_type: Mapped[str] = mapped_column(String(64))
    artifact_path: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
