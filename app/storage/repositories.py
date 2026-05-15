from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.storage.models import (
    BbotListenerCacheEntry,
    DevSession,
    DevTask,
    DevTaskArtifact,
    Group,
    Job,
    MemoryItem,
    Message,
    Summary,
    UsageRecord,
    User,
)


def _normalize_utc_sqlite_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


class GroupRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_group(self, *, group_id: int, group_name: str, enabled: bool, speak_enabled: bool) -> Group:
        group = self.session.get(Group, group_id) or Group(group_id=group_id)
        group.group_name = group_name
        group.enabled = enabled
        group.speak_enabled = speak_enabled
        self.session.add(group)
        return group

    def set_speak_enabled(self, group_id: int, value: bool) -> None:
        group = self.session.get(Group, group_id) or Group(group_id=group_id)
        group.speak_enabled = value
        self.session.add(group)

    def set_enabled(self, group_id: int, value: bool) -> None:
        group = self.session.get(Group, group_id) or Group(group_id=group_id)
        group.enabled = value
        self.session.add(group)

    def get_group(self, group_id: int) -> Group | None:
        return self.session.get(Group, group_id)


class UserRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_user(self, *, user_id: int, nickname: str, group_card: str) -> User:
        user = self.session.get(User, user_id) or User(user_id=user_id)
        user.nickname = nickname
        user.group_card = group_card
        now = datetime.now().astimezone()
        user.first_seen_at = user.first_seen_at or now
        user.last_seen_at = now
        self.session.add(user)
        return user

    def get_users_by_ids(self, user_ids: list[int]) -> dict[int, User]:
        unique_ids = list(dict.fromkeys(user_ids))
        if not unique_ids:
            return {}
        stmt = select(User).where(User.user_id.in_(unique_ids))
        return {user.user_id: user for user in self.session.scalars(stmt)}


class MessageRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def _is_reserved_outbound(message: Message) -> bool:
        raw_json = message.raw_json
        return isinstance(raw_json, dict) and raw_json.get("delivery_state") == "reserved"

    def add_group_message(
        self,
        *,
        platform_msg_id: str,
        group_id: int,
        user_id: int,
        timestamp: datetime,
        plain_text: str,
        raw_json: dict[str, Any],
        msg_type: str,
        reply_to_msg_id: str | None,
        mentioned_bot: bool,
    ) -> Message:
        self.session.flush()
        message = Message(
            platform_msg_id=platform_msg_id,
            group_id=group_id,
            user_id=user_id,
            timestamp=timestamp,
            plain_text=plain_text,
            raw_json=raw_json,
            msg_type=msg_type,
            reply_to_msg_id=reply_to_msg_id,
            mentioned_bot=mentioned_bot,
        )
        self.session.add(message)
        return message

    def add_private_message(
        self,
        *,
        platform_msg_id: str,
        user_id: int,
        timestamp: datetime,
        plain_text: str,
        raw_json: dict[str, Any],
        msg_type: str = "text",
        reply_to_msg_id: str | None = None,
        mentioned_bot: bool = False,
    ) -> Message:
        self.session.flush()
        message = Message(
            platform_msg_id=platform_msg_id,
            group_id=None,
            user_id=user_id,
            timestamp=timestamp,
            plain_text=plain_text,
            raw_json=raw_json,
            msg_type=msg_type,
            reply_to_msg_id=reply_to_msg_id,
            mentioned_bot=mentioned_bot,
        )
        self.session.add(message)
        return message

    def get_by_platform_msg_id(self, platform_msg_id: str) -> Message | None:
        stmt = select(Message).where(Message.platform_msg_id == platform_msg_id).limit(1)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_recent_group_messages(self, *, group_id: int, limit: int) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.group_id == group_id)
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        recent_messages = []
        for message in self.session.scalars(stmt):
            if self._is_reserved_outbound(message):
                continue
            recent_messages.append(message)
            if len(recent_messages) >= limit:
                break
        return list(reversed(recent_messages))

    def count_group_messages(self, group_id: int) -> int:
        stmt = select(func.count()).select_from(Message).where(Message.group_id == group_id)
        return self.session.scalar(stmt) or 0

    def list_recent_group_messages_for_user(self, *, group_id: int, user_id: int, limit: int) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.group_id == group_id, Message.user_id == user_id)
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        recent_messages = []
        for message in self.session.scalars(stmt):
            if self._is_reserved_outbound(message):
                continue
            recent_messages.append(message)
            if len(recent_messages) >= limit:
                break
        return list(reversed(recent_messages))

    def list_recent_group_messages_for_user_since(
        self,
        *,
        group_id: int,
        user_id: int,
        since: datetime,
        limit: int,
    ) -> list[Message]:
        stmt = (
            select(Message)
            .where(
                Message.group_id == group_id,
                Message.user_id == user_id,
                Message.timestamp >= since,
            )
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        recent_messages = []
        for message in self.session.scalars(stmt):
            if self._is_reserved_outbound(message):
                continue
            recent_messages.append(message)
            if len(recent_messages) >= limit:
                break
        return list(reversed(recent_messages))

    def list_recent_private_messages_for_user_since(
        self,
        *,
        user_id: int,
        since: datetime,
        limit: int,
    ) -> list[Message]:
        stmt = (
            select(Message)
            .where(
                Message.group_id.is_(None),
                Message.user_id == user_id,
                Message.timestamp >= since,
            )
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        recent_messages = []
        for message in self.session.scalars(stmt):
            if self._is_reserved_outbound(message):
                continue
            recent_messages.append(message)
            if len(recent_messages) >= limit:
                break
        return list(reversed(recent_messages))

    def list_recent_group_user_ids(self, *, group_id: int, limit: int) -> list[int]:
        latest_message_at = func.max(Message.timestamp).label("latest_message_at")
        stmt = (
            select(Message.user_id, latest_message_at)
            .where(Message.group_id == group_id)
            .group_by(Message.user_id)
            .order_by(latest_message_at.desc())
            .limit(limit)
        )
        return [int(user_id) for user_id, _latest in self.session.execute(stmt)]

    def last_bot_reply_at(self, *, group_id: int, bot_user_id: int) -> datetime | None:
        stmt = (
            select(Message)
            .where(Message.group_id == group_id, Message.user_id == bot_user_id)
            .order_by(Message.timestamp.desc(), Message.id.desc())
        )
        timestamp = None
        for message in self.session.scalars(stmt):
            if self._is_reserved_outbound(message):
                continue
            timestamp = message.timestamp
            break
        if timestamp is None:
            return None
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=UTC)
        return timestamp


class BbotListenerCacheRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_entry(
        self,
        *,
        group_id: int,
        platform: str,
        external_id: str,
        canonical_name: str,
        aliases: list[str],
        source: str,
        updated_at: datetime,
    ) -> BbotListenerCacheEntry:
        stmt = (
            select(BbotListenerCacheEntry)
            .where(
                BbotListenerCacheEntry.group_id == group_id,
                BbotListenerCacheEntry.platform == platform,
                BbotListenerCacheEntry.external_id == external_id,
            )
            .limit(1)
        )
        entry = self.session.execute(stmt).scalar_one_or_none()
        if entry is None:
            entry = BbotListenerCacheEntry(
                group_id=group_id,
                platform=platform,
                external_id=external_id,
            )
        entry.canonical_name = canonical_name
        entry.aliases_json = aliases
        entry.source = source
        entry.updated_at = updated_at
        self.session.add(entry)
        return entry

    def find_best_match(self, *, group_id: int, platform: str, query: str) -> BbotListenerCacheEntry | None:
        stmt = (
            select(BbotListenerCacheEntry)
            .where(
                BbotListenerCacheEntry.group_id == group_id,
                BbotListenerCacheEntry.platform == platform,
            )
            .order_by(BbotListenerCacheEntry.updated_at.desc(), BbotListenerCacheEntry.id.desc())
        )
        normalized_query = self._normalize(query)
        if not normalized_query:
            return None

        best_entry = None
        best_score = -1
        for entry in self.session.scalars(stmt):
            names = [str(entry.canonical_name or "")] + [str(alias) for alias in (entry.aliases_json or [])]
            normalized_candidates = [candidate for candidate in (self._normalize(name) for name in names) if candidate]
            score = self._score_candidates(normalized_query, normalized_candidates)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_score <= 0:
            return None
        return best_entry

    def _score_candidates(self, query: str, candidates: list[str]) -> int:
        score = 0
        for candidate in candidates:
            if query == candidate:
                score = max(score, 100)
            elif query in candidate:
                score = max(score, 80)
            elif candidate in query:
                score = max(score, 60)
        return score

    def _normalize(self, value: str) -> str:
        lowered = value.strip().lower()
        return "".join(character for character in lowered if character.isalnum() or "\u4e00" <= character <= "\u9fff")


class SummaryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_summary(
        self,
        *,
        scope_type: str,
        scope_id: str,
        summary_level: str,
        start_at: datetime,
        end_at: datetime,
        content: str,
        source_count: int,
    ) -> Summary:
        summary = Summary(
            scope_type=scope_type,
            scope_id=scope_id,
            summary_level=summary_level,
            start_at=start_at,
            end_at=end_at,
            content=content,
            source_count=source_count,
        )
        self.session.add(summary)
        return summary

    def list_recent_group_summaries(self, scope_id: str, limit: int) -> list[str]:
        stmt = (
            select(Summary)
            .where(Summary.scope_type == "group", Summary.scope_id == scope_id)
            .order_by(Summary.end_at.desc(), Summary.id.desc())
            .limit(limit)
        )
        summaries = [summary.content for summary in self.session.scalars(stmt)]
        return list(reversed(summaries))


class MemoryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_memory(
        self,
        *,
        scope_type: str,
        scope_id: str,
        subject_type: str,
        subject_id: str,
        memory_kind: str,
        content: str,
        importance: int,
        confidence: float,
        source_msg_id: str,
    ) -> MemoryItem:
        memory = MemoryItem(
            scope_type=scope_type,
            scope_id=scope_id,
            subject_type=subject_type,
            subject_id=subject_id,
            memory_kind=memory_kind,
            content=content,
            importance=importance,
            confidence=confidence,
            source_msg_id=source_msg_id,
        )
        self.session.add(memory)
        return memory

    def list_group_memories(self, scope_id: str, limit: int) -> list[MemoryItem]:
        stmt = (
            select(MemoryItem)
            .where(MemoryItem.scope_type == "group", MemoryItem.scope_id == scope_id)
            .order_by(MemoryItem.importance.desc(), MemoryItem.id.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def list_group_memories_for_subject(self, *, scope_id: str, subject_id: str, limit: int) -> list[MemoryItem]:
        stmt = (
            select(MemoryItem)
            .where(
                MemoryItem.scope_type == "group",
                MemoryItem.scope_id == scope_id,
                MemoryItem.subject_id == subject_id,
            )
            .order_by(MemoryItem.importance.desc(), MemoryItem.id.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class JobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_job(self, *, job_type: str, payload_json: dict[str, Any], run_at: datetime, status: str = "queued") -> Job:
        job = Job(job_type=job_type, payload_json=payload_json, run_at=run_at, status=status)
        self.session.add(job)
        self.session.flush()
        return job

    def count_active_jobs(self, *, job_type: str, statuses: list[str] | None = None) -> int:
        active_statuses = statuses or ["queued", "running"]
        stmt = select(func.count(Job.id)).where(Job.job_type == job_type, Job.status.in_(active_statuses))
        return int(self.session.execute(stmt).scalar_one() or 0)

    def list_jobs(self, *, job_type: str, statuses: list[str]) -> list[Job]:
        if not statuses:
            return []
        stmt = (
            select(Job)
            .where(Job.job_type == job_type, Job.status.in_(statuses))
            .order_by(Job.id.asc())
        )
        return list(self.session.scalars(stmt))

    def claim_oldest_queued_job(self, *, job_type: str, now: datetime | None = None) -> Job | None:
        run_before = _normalize_utc_sqlite_timestamp(now or datetime.now().astimezone())
        stmt = (
            select(Job)
            .where(Job.job_type == job_type, Job.status == "queued", Job.run_at <= run_before)
            .order_by(Job.id.asc())
            .limit(1)
        )
        job = self.session.execute(stmt).scalar_one_or_none()
        if job is None:
            return None
        job.status = "running"
        self.session.add(job)
        self.session.flush()
        return job

    def mark_job_status(self, *, job_id: int, status: str, payload_json: dict[str, Any] | None = None) -> Job | None:
        job = self.session.get(Job, job_id)
        if job is None:
            return None
        job.status = status
        if payload_json is not None:
            job.payload_json = payload_json
        self.session.add(job)
        return job

    def requeue_running_jobs(self, *, job_type: str) -> int:
        jobs = self.list_jobs(job_type=job_type, statuses=["running"])
        for job in jobs:
            job.status = "queued"
            self.session.add(job)
        return len(jobs)


class UsageRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_usage(
        self,
        *,
        timestamp: datetime,
        model: str,
        endpoint: str,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
    ) -> UsageRecord:
        record = UsageRecord(
            timestamp=_normalize_utc_sqlite_timestamp(timestamp),
            model=model,
            endpoint=endpoint,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
        )
        self.session.add(record)
        return record

    def summarize_usage(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        model: str | None = None,
    ) -> dict[str, int]:
        normalized_start_at = _normalize_utc_sqlite_timestamp(start_at)
        normalized_end_at = _normalize_utc_sqlite_timestamp(end_at)
        stmt = select(
            func.count(UsageRecord.id),
            func.sum(UsageRecord.input_tokens),
            func.sum(UsageRecord.cached_input_tokens),
            func.sum(UsageRecord.output_tokens),
        ).where(
            UsageRecord.timestamp >= normalized_start_at,
            UsageRecord.timestamp <= normalized_end_at,
        )
        if model is not None:
            stmt = stmt.where(UsageRecord.model == model)
        call_count, input_tokens, cached_input_tokens, output_tokens = self.session.execute(stmt).one()
        return {
            "call_count": int(call_count or 0),
            "input_tokens": int(input_tokens or 0),
            "cached_input_tokens": int(cached_input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
        }


class DevSessionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_latest_owner_session(self, *, owner_qq: int, session_mode: str = "project") -> DevSession | None:
        stmt = (
            select(DevSession)
            .where(DevSession.owner_qq == owner_qq, DevSession.session_mode == session_mode)
            .order_by(DevSession.last_active_at.desc(), DevSession.id.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def create_owner_session(self, *, owner_qq: int, session_mode: str = "project") -> DevSession:
        now = datetime.now().astimezone()
        dev_session = DevSession(
            owner_qq=owner_qq,
            session_mode=session_mode,
            started_at=now,
            last_active_at=now,
            summary="",
        )
        self.session.add(dev_session)
        self.session.flush()
        return dev_session

    def get_or_create_owner_session(self, *, owner_qq: int, session_mode: str = "project") -> DevSession:
        dev_session = self.get_latest_owner_session(owner_qq=owner_qq, session_mode=session_mode)
        if dev_session is None:
            dev_session = self.create_owner_session(owner_qq=owner_qq, session_mode=session_mode)
        else:
            dev_session.last_active_at = datetime.now().astimezone()
            self.session.add(dev_session)
        self.session.add(dev_session)
        self.session.flush()
        return dev_session

    def list_recent_owner_sessions(
        self,
        *,
        owner_qq: int,
        limit: int,
        session_modes: list[str] | None = None,
    ) -> list[DevSession]:
        stmt = select(DevSession).where(DevSession.owner_qq == owner_qq)
        if session_modes:
            stmt = stmt.where(DevSession.session_mode.in_(session_modes))
        stmt = stmt.order_by(DevSession.last_active_at.desc(), DevSession.id.desc()).limit(limit)
        return list(self.session.scalars(stmt))

    def update_session(
        self,
        *,
        session_id: int,
        summary: str | None = None,
        last_task_id: int | None = None,
    ) -> DevSession | None:
        dev_session = self.session.get(DevSession, session_id)
        if dev_session is None:
            return None
        dev_session.last_active_at = datetime.now().astimezone()
        if summary is not None:
            dev_session.summary = summary
        if last_task_id is not None:
            dev_session.last_task_id = last_task_id
        self.session.add(dev_session)
        return dev_session


class DevTaskRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_task(
        self,
        *,
        session_id: int,
        requested_by_qq: int,
        raw_request_text: str,
        intent_type: str,
        status: str = "queued",
    ) -> DevTask:
        task = DevTask(
            session_id=session_id,
            requested_by_qq=requested_by_qq,
            requested_at=datetime.now().astimezone(),
            raw_request_text=raw_request_text,
            intent_type=intent_type,
            status=status,
            summary="",
            files_read_json=[],
            files_changed_json=[],
            commands_run_json=[],
            restart_required=False,
            restart_result="",
            failure_reason="",
            checkpoint_dir="",
            result_text="",
        )
        self.session.add(task)
        self.session.flush()
        return task

    def list_tasks_by_status(self, status: str) -> list[DevTask]:
        stmt = select(DevTask).where(DevTask.status == status).order_by(DevTask.id.asc())
        return list(self.session.scalars(stmt))

    def list_tasks_for_session_by_status(self, *, session_id: int, statuses: list[str]) -> list[DevTask]:
        if not statuses:
            return []
        stmt = (
            select(DevTask)
            .where(DevTask.session_id == session_id, DevTask.status.in_(statuses))
            .order_by(DevTask.id.asc())
        )
        return list(self.session.scalars(stmt))

    def list_tasks_by_statuses(self, *, statuses: list[str], intent_types: list[str] | None = None) -> list[DevTask]:
        if not statuses:
            return []
        stmt = select(DevTask).where(DevTask.status.in_(statuses))
        if intent_types:
            stmt = stmt.where(DevTask.intent_type.in_(intent_types))
        stmt = stmt.order_by(DevTask.id.asc())
        return list(self.session.scalars(stmt))

    def list_recent_tasks_for_session(self, *, session_id: int, limit: int) -> list[DevTask]:
        stmt = (
            select(DevTask)
            .where(DevTask.session_id == session_id)
            .order_by(DevTask.id.desc())
            .limit(limit)
        )
        return list(reversed(list(self.session.scalars(stmt))))

    def claim_oldest_queued_task(self, *, intent_types: list[str] | None = None) -> DevTask | None:
        stmt = select(DevTask).where(DevTask.status == "queued")
        if intent_types:
            stmt = stmt.where(DevTask.intent_type.in_(intent_types))
        stmt = stmt.order_by(DevTask.id.asc()).limit(1)
        task = self.session.execute(stmt).scalar_one_or_none()
        if task is None:
            return None
        task.status = "running"
        self.session.add(task)
        self.session.flush()
        return task

    def get_task(self, task_id: int) -> DevTask | None:
        return self.session.get(DevTask, task_id)

    def mark_completed(
        self,
        *,
        task_id: int,
        summary: str,
        result_text: str,
        files_read: list[str],
        files_changed: list[str],
        commands_run: list[str],
        restart_required: bool,
        restart_result: str,
        checkpoint_dir: str,
    ) -> DevTask | None:
        task = self.session.get(DevTask, task_id)
        if task is None:
            return None
        task.status = "completed"
        task.summary = summary
        task.result_text = result_text
        task.files_read_json = files_read
        task.files_changed_json = files_changed
        task.commands_run_json = commands_run
        task.restart_required = restart_required
        task.restart_result = restart_result
        task.checkpoint_dir = checkpoint_dir
        self.session.add(task)
        return task

    def mark_failed(self, *, task_id: int, failure_reason: str, checkpoint_dir: str = "") -> DevTask | None:
        task = self.session.get(DevTask, task_id)
        if task is None:
            return None
        task.status = "failed"
        task.failure_reason = failure_reason
        if checkpoint_dir:
            task.checkpoint_dir = checkpoint_dir
        self.session.add(task)
        return task

    def mark_status(self, *, task_id: int, status: str) -> DevTask | None:
        task = self.session.get(DevTask, task_id)
        if task is None:
            return None
        task.status = status
        self.session.add(task)
        return task


class DevTaskArtifactRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_artifact(
        self,
        *,
        task_id: int,
        artifact_type: str,
        artifact_path: str,
        metadata_json: dict[str, Any],
    ) -> DevTaskArtifact:
        artifact = DevTaskArtifact(
            task_id=task_id,
            artifact_type=artifact_type,
            artifact_path=artifact_path,
            metadata_json=metadata_json,
        )
        self.session.add(artifact)
        self.session.flush()
        return artifact
