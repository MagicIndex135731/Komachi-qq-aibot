from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


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

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nickname: Mapped[str] = mapped_column(String(255), default="")
    group_card: Mapped[str] = mapped_column(String(255), default="")
    first_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    profile_summary: Mapped[str] = mapped_column(Text, default="")


class Message(Base):
    __tablename__ = "messages"

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


class MemoryItem(Base):
    __tablename__ = "memory_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope_type: Mapped[str] = mapped_column(String(32))
    scope_id: Mapped[str] = mapped_column(String(64))
    subject_type: Mapped[str] = mapped_column(String(32))
    subject_id: Mapped[str] = mapped_column(String(64))
    memory_kind: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    importance: Mapped[int] = mapped_column(Integer, default=1)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    source_msg_id: Mapped[str] = mapped_column(String(128))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_type: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


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
