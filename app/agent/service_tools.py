"""Scoped adapters for existing memory, persona, image and scheduling services.

These tools provide capabilities to the framework's registry. They do not
contain an agent loop or decide which capability should follow another.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.adapters.outbound_guard import GroupDispatchScope, group_dispatch_scope
from app.agent.group_vision import canonical_group_image_inputs
from app.agent.memory_integration import (
    call_agent_memory_tool, create_memory_dependency_receipt,
    memory_dependencies_valid, restore_memory_dependency_receipt,
)
from app.agent.social_planner import SocialPlanner
from app.agent.tool_agent import AgentTool, ToolOutcome
from app.core.memory_source_guard import is_usable_memory_source
from app.core.memory_tool_executor import MemoryToolExecutor
from app.core.memory_tools import DEFAULT_MEMORY_TOOL_TIMEOUT_SECONDS, memory_tool_schemas
from app.core.message_content import ImageAttachment
from app.runtime.blocking import BoundedBlockingRunner
from app.storage.autonomous_agent import AutonomousRepository
from app.storage.db import session_scope
from app.storage.models import Job, Message
from app.storage.repositories import JobRepository, PersonaStyleSyncStateRepository


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ServiceToolContext:
    router: Any
    observation: Any
    guard: Any
    observed_sources: set[int]
    blocking: Any
    store: Any
    clock: Callable[[], datetime] | None = None
    target_user_id: int | None = None
    target_message_id: str | None = None
    task_service: Any = None
    schedule_wakeup: Any = None
    list_wakeups: Any = None
    cancel_wakeup: Any = None
    memory_search_allowed: bool = True
    memory_dependencies: dict[str, str] = field(default_factory=dict)
    available_input_budget: int | None = None


def _object(**properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _text(maximum=500, minimum=1):
    return {"type": "string", "minLength": minimum, "maxLength": maximum}


def _sources(engine, group_id, ids):
    ids = tuple(sorted(set(ids)))
    if not ids:
        return ()
    if len(ids) > 200:
        raise ValueError("source_budget_exhausted")
    with session_scope(engine) as session:
        rows = tuple(session.scalars(select(Message).where(Message.group_id == group_id, Message.id.in_(ids))))
        if len(rows) != len(ids) or any(not is_usable_memory_source(row, group_id=group_id) for row in rows):
            raise ValueError("source_unavailable")
        # Return immutable values: ORM instances are detached after this scope.
        return tuple((row.id, row.platform_msg_id, row.user_id, row.raw_json) for row in rows)


class PersonaTaskService:
    """Persistent persona refresh requests using the existing jobs repository.

    A request does not imply permission to spend money. The processing entry
    point requires an explicit operator-owned flag; tool arguments cannot set it.
    """
    job_type = "agent_persona_refresh"

    def __init__(self, engine, *, account_id: int, clock=None):
        self.engine, self.account_id = engine, account_id
        self.clock = clock or (lambda: datetime.now(UTC))

    def _prefix(self, group_id):
        return f"agent-persona:{self.account_id}:{int(group_id)}:"

    @contextmanager
    def _transaction(self, before_commit=lambda: None):
        with self.engine.connect() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            with Session(bind=connection) as session:
                try:
                    yield session
                    session.flush()
                    before_commit()
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise

    @staticmethod
    def _data(row):
        return {"task_id": row.id, "status": row.status, "reason": row.last_error_code,
                "persona_key": row.payload_json.get("persona_key"), "run_at": row.run_at.isoformat()}

    def enqueue(self, *, group_id, persona_key, source_user_id, source_message_ids, request_key, guard):
        if type(source_user_id) is not int or source_user_id <= 0:
            return {"error": "persona_source_user_invalid"}
        key = self._prefix(group_id) + hashlib.sha256(request_key.encode("utf-8")).hexdigest()
        guard.require_valid()
        _sources(self.engine, group_id, source_message_ids)
        with self._transaction(guard.require_valid) as session:
            pending = tuple(session.scalars(select(Job).where(Job.job_type == self.job_type,
                Job.job_key.startswith(self._prefix(group_id)), Job.status.in_(("queued", "running"))).limit(20)))
            if len(pending) >= 20 and not any(row.job_key == key for row in pending):
                return {"error": "persona_task_capacity"}
            row = JobRepository(session).add_job(job_type=self.job_type, job_key=key, status="queued",
                run_at=self.clock().astimezone(UTC).replace(tzinfo=None),
                payload_json={"account_id": self.account_id, "group_id": group_id, "persona_key": persona_key,
                    "source_user_id": source_user_id, "source_message_ids": list(source_message_ids)})
            if row.payload_json.get("persona_key") != persona_key or row.payload_json.get("source_user_id") != source_user_id:
                return {"error": "request_key_conflict"}
            return self._data(row)

    def status(self, *, group_id, task_id):
        with Session(self.engine) as session:
            row = session.get(Job, task_id)
            if row is None or row.job_type != self.job_type or not row.job_key.startswith(self._prefix(group_id)):
                return {"error": "task_out_of_scope"}
            return self._data(row)

    def process_next(self, persona_service, *, allow_paid: bool = False, execution_guard=None):
        """Run one approved request with the existing source-validating service.

        Calling this with the default flag records a real blocked result without
        constructing a model. There is no automatic retry of failed refreshes.
        """
        if execution_guard is not None:
            execution_guard()
        with self._transaction() as session:
            now = self.clock().astimezone(UTC).replace(tzinfo=None)
            interrupted = tuple(session.scalars(select(Job).where(Job.job_type == self.job_type,
                Job.job_key.startswith(f"agent-persona:{self.account_id}:"), Job.status == "running",
                Job.lease_until <= now)))
            for lost in interrupted:
                lost.status, lost.last_error_code = "failed", "interrupted_not_replayed"
                lost.completed_at = now
            row = session.scalars(select(Job).where(Job.job_type == self.job_type,
                Job.job_key.startswith(f"agent-persona:{self.account_id}:"), Job.status == "queued", Job.run_at <= now)
                .order_by(Job.run_at, Job.id).limit(1)).first()
            if row is None:
                return None
            if allow_paid is not True:
                row.status, row.last_error_code = "blocked", "paid_calls_paused"
                return self._data(row)
            row.status, row.locked_by = "running", uuid4().hex
            row.locked_at = now
            row.lease_until = now + timedelta(seconds=600)
            row.attempt_count += 1
            task_id, owner, payload = row.id, row.locked_by, dict(row.payload_json)
        def require_valid():
            if execution_guard is not None:
                execution_guard()
            with session_scope(self.engine) as session:
                current = session.get(Job, task_id)
                if (current is None or current.status != "running" or current.locked_by != owner
                        or current.lease_until <= self.clock().astimezone(UTC).replace(tzinfo=None)):
                    raise RuntimeError("persona_task_authority_revoked")
        status, reason = "failed", "persona_refresh_failed"
        try:
            require_valid()
            group_id, user_id, key = payload["group_id"], payload["source_user_id"], payload["persona_key"]
            _sources(self.engine, group_id, payload["source_message_ids"])
            profile = persona_service.personas.get(key) or {}
            if (key == "default" or profile.get("live_refresh") is not True
                    or profile.get("source_group_id") != group_id or profile.get("source_user_id") != user_id):
                status, reason = "blocked", "persona_permission_revoked"
            else:
                def refreshed_at():
                    with session_scope(self.engine) as session:
                        state = PersonaStyleSyncStateRepository(session).get(group_id=group_id, user_id=user_id)
                        return state.last_refresh_at if state else None
                # Hold the same per-profile lock as the existing timer across
                # the complete request, so its receipt cannot claim another
                # consumer's refresh or pay for the same window twice.
                with persona_service._profile_execution(key, user_id, group_id, require_valid):
                    persona_service._sync_examples(key, user_id, group_id, execution_guard=require_valid)
                    before = refreshed_at()
                    persona_service._maybe_refresh_profile(key, user_id, group_id, execution_guard=require_valid)
                    require_valid()
                    after = refreshed_at()
                status, reason = ("completed", "source_validated_refresh") if after and after != before else ("blocked", "not_due_or_refresh_declined")
        except Exception as exc:
            reason = type(exc).__name__
        with self._transaction() as session:
            row = session.get(Job, task_id)
            if row.status == "running" and row.locked_by == owner:
                row.status, row.last_error_code = status, reason
                row.completed_at = self.clock().astimezone(UTC).replace(tzinfo=None)
            return self._data(row)

    def recover(self):
        """Account-scoped startup recovery; interrupted paid jobs are not replayed."""
        with self._transaction() as session:
            rows = tuple(session.scalars(select(Job).where(Job.job_type == self.job_type,
                Job.job_key.startswith(f"agent-persona:{self.account_id}:"), Job.status == "running")))
            for row in rows:
                row.status, row.last_error_code = "failed", "interrupted_not_replayed"
            return len(rows)


class PersonaTaskWorker:
    """Optional account-scoped consumer, gated by operator-owned configuration.

    The tool cannot enable this worker. Disabling permission leaves requests
    queued and fences any model result that arrives after a stop or revocation.
    """

    def __init__(self, tasks: PersonaTaskService, persona_service, *, permission_check=None,
                 blocking=None, interval_seconds: float = 2.0):
        self.tasks, self.persona_service = tasks, persona_service
        self.permission_check = permission_check or (lambda: False)
        self.blocking = blocking or BoundedBlockingRunner(1)
        self.owns_blocking = blocking is None
        self.interval_seconds = max(0.01, float(interval_seconds))
        self.running = False
        self._task = None

    def _require_execution(self) -> None:
        if not self.running or self.permission_check() is not True:
            raise RuntimeError("persona_task_execution_disabled")

    async def start(self) -> bool:
        if self.running:
            return True
        if self.permission_check() is not True:
            return False
        self.running = True
        self._task = asyncio.create_task(self._run(), name="agent-persona-task-worker")
        return True

    async def _run(self) -> None:
        while self.running:
            if self.permission_check() is True:
                try:
                    await self.blocking.run(self.tasks.process_next, self.persona_service,
                        allow_paid=True, execution_guard=self._require_execution)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("agent_persona_task_failed error_type=%s", type(exc).__name__)
            await asyncio.sleep(self.interval_seconds)

    async def stop(self) -> None:
        self.running = False
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self.owns_blocking:
            self.blocking.close()


def build_service_tools(context: ServiceToolContext) -> tuple[AgentTool, ...]:
    router, observation, guard = context.router, context.observation, context.guard
    runtime, job = router.runtime, context.observation.job
    clock = context.clock or (lambda: datetime.now(UTC))
    account_id = runtime.settings.bot_qq
    services = dict(runtime.social_policy.get("services", {}))
    data_scope = getattr(context.store, "scope", job.key)
    note_scope = "work-note:" + hashlib.sha256(data_scope.encode("utf-8")).hexdigest()
    notes = AutonomousRepository(router.engine, account_id=account_id, clock=clock)

    def require_valid():
        guard.require_valid()
        if dict(runtime.social_policy.get("services", {})) != services:
            raise RuntimeError("service_permission_revoked")
        if job.kind == "group":
            _sources(router.engine, job.scope_id, context.observed_sources)

    def valid():
        try:
            require_valid()
            return True
        except (ValueError, RuntimeError):
            return False

    async def read_persona(args):
        require_valid()
        if job.kind == "group":
            from app.agent.persona_integration import build_group_persona_context
            def read_group_persona():
                target = None
                if context.target_message_id:
                    with session_scope(router.engine) as session:
                        target = session.scalar(select(Message).where(Message.group_id == job.scope_id,
                            Message.platform_msg_id == context.target_message_id))
                return build_group_persona_context(router, group_id=job.scope_id,
                    source_message_ids=tuple(sorted(context.observed_sources)),
                    target_message_id=context.target_message_id,
                    addressed=bool(target and target.mentioned_bot), now=observation.now)
            current = await context.blocking.run(read_group_persona)
            require_valid()
            candidates = set(current.source_message_ids)
            if len(context.observed_sources | candidates) > 200:
                return ToolOutcome({"error": "source_budget_exhausted"}, "failed")
            _sources(router.engine, job.scope_id, candidates)
            context.observed_sources.update(candidates)
            return ToolOutcome({"persona_key": current.persona_key, "expression_style": current.text,
                "profile_fingerprint": current.profile_fingerprint,
                "source_message_ids": current.source_message_ids, "kind": "persona_not_fact_evidence"})
        key = job.persona_key
        raw = runtime.persona if key == "default" else runtime.personas.get(key, {})
        return ToolOutcome({"persona_key": key, "expression_style": SocialPlanner._render_public_persona(raw), "kind": "persona_not_fact_evidence"})

    async def save_note(args):
        require_valid()
        dependencies = dict(context.memory_dependencies) if job.kind == "group" else {}
        try:
            payload = {"note": args["note"], "memory_dependencies": create_memory_dependency_receipt(dependencies)}
        except ValueError:
            return ToolOutcome({"error": "note_dependency_invalid"}, "blocked")
        if len(str(payload)) > 2000:
            return ToolOutcome({"error": "note_budget_exhausted"}, "blocked")
        def require_note_valid():
            require_valid()
            if job.kind == "group" and not memory_dependencies_valid(router.engine,
                    group_id=job.scope_id, dependencies=dependencies, now=clock()):
                raise RuntimeError("note_memory_dependency_revoked")
        def write():
            require_note_valid()
            notes.write_intention(note_scope, payload, tuple(sorted(context.observed_sources)),
                before_commit=require_note_valid)
        await context.blocking.run(write)
        require_note_valid()
        return ToolOutcome({"saved": True, "kind": "revocable_work_note_not_fact"})

    async def read_note(args):
        require_valid()
        note = await context.blocking.run(notes.read_intention, note_scope)
        payload = note.get("working_intention", {})
        try:
            if job.kind == "group":
                source_ids = note.get("source_message_ids", ())
                _sources(router.engine, job.scope_id, source_ids)
                if len(context.observed_sources | set(source_ids)) > 200:
                    return ToolOutcome({"error": "source_budget_exhausted"}, "blocked")
                dependencies = {}
                if payload:
                    if "memory_dependencies" not in payload:
                        return ToolOutcome({"error": "note_dependency_metadata_missing"}, "blocked")
                    dependencies = await context.blocking.run(restore_memory_dependency_receipt,
                        router.engine, group_id=job.scope_id, receipt=payload["memory_dependencies"], now=clock())
                    if dependencies is None:
                        return ToolOutcome({"error": "note_memory_dependency_revoked"}, "blocked")
                    if len(set(context.memory_dependencies) | set(dependencies)) > 200:
                        return ToolOutcome({"error": "memory_dependency_budget_exhausted"}, "blocked")
                context.observed_sources.update(source_ids)
                context.memory_dependencies.update(dependencies)
        except ValueError:
            return ToolOutcome({"error": "note_source_revoked"}, "blocked")
        require_valid()
        # Dependency receipts are infrastructure metadata, never model evidence.
        return ToolOutcome({"note": {"note": payload["note"]} if "note" in payload else {},
                            "kind": "revocable_work_note_not_fact"})

    tools = [AgentTool("read_persona", "读取当前作用域人格及表达风格；不是事实记忆", _object(), read_persona),
             AgentTool("save_work_note", "保存可撤销工作笔记；不会写长期事实或人格语料", _object(note=_text(1200, 0)), save_note),
             AgentTool("read_work_note", "读取本账号当前作用域工作笔记，失效来源会拒绝", _object(), read_note)]
    async def invoke(callback, args):
        require_valid()
        if inspect.iscoroutinefunction(callback):
            value = await callback(args)
        else:
            value = await context.blocking.run(callback, args)
            if inspect.isawaitable(value):
                value = await value
        require_valid()
        return value if isinstance(value, ToolOutcome) else ToolOutcome(value if isinstance(value, dict) else {"result": value},
            "blocked" if isinstance(value, dict) and value.get("error") else "ok")
    if context.schedule_wakeup is not None:
        async def schedule(args):
            return await invoke(context.schedule_wakeup, args)
        tools.append(AgentTool("schedule_wakeup", "保存当前授权范围稍后继续的目标/提醒，到期唤醒Agent决定行动，不预定发送内容",
            _object(goal=_text(500), delay_seconds={"type": "integer", "minimum": 1, "maximum": 604800}), schedule))
    if context.list_wakeups is not None:
        async def list_wakeups(args):
            return await invoke(context.list_wakeups, args)
        tools.append(AgentTool("list_wakeups", "读取当前授权范围持久的到期唤醒/提醒及真实状态", _object(), list_wakeups))
    if context.cancel_wakeup is not None:
        async def cancel(args):
            return await invoke(context.cancel_wakeup, args)
        tools.append(AgentTool("cancel_wakeup", "取消当前授权范围持久目标的以后唤醒", _object(task_id=_text(128)), cancel))
    if job.kind != "group":
        return tuple(tools)

    owner_authority=getattr(guard,'owner_authority',None)
    if getattr(router, 'persona_manager', None) is not None:
        async def set_group_persona(args):
            require_valid()
            if owner_authority is None or owner_authority() is not True:
                return ToolOutcome({'error':'owner_required'},'blocked')
            manager=router.persona_manager
            from app.core.persona_switch import persona_aliases
            requested=args['persona'].strip()
            matches=[key for key,persona in manager.personas.items()
                if requested==key or requested in persona_aliases(persona)]
            if len(matches)!=1:
                return ToolOutcome({'error':'persona_not_unique','available':[
                    {'key':key,'name':persona.get('name',key)} for key,persona in manager.personas.items()]},'blocked')
            guard.require_valid()
            confirmation=await router.persona_switch_service.switch(group_id=job.scope_id,target_key=matches[0])
            require_valid()
            return ToolOutcome({'applied':manager.active_key(job.scope_id)==matches[0],
                'persona_key':matches[0],'confirmation':confirmation})
        tools.append(AgentTool('set_group_persona','仅真实主人可切换当前授权群人格，支持当前配置名称或key；不会修改QQ头像/群名片',
            _object(persona=_text(100)),set_group_persona))

    async def memory_call(name, args):
        require_valid()
        if not getattr(router, "_group_memory_enabled", lambda **kwargs: True)(group_id=job.scope_id):
            return ToolOutcome({"error": "memory_disabled"}, "blocked")
        if not context.memory_search_allowed:
            return ToolOutcome({"error": "memory_search_not_allowed_for_conversation_quote"}, "blocked")
        if name != "memory_write":
            return await call_agent_memory_tool(context, name, args)
        if name == "memory_write" and services.get("memory_write") is not True:
            return ToolOutcome({"error": "memory_write_not_authorized"}, "blocked")
        source_rows = _sources(router.engine, job.scope_id, observation.source_message_ids)
        subjects = tuple(sorted({row[2] for row in source_rows if row[2] != account_id
            and not (isinstance(row[3], dict) and row[3].get("direction") == "outbound")}))
        member_names = {str(row[2]): row[2] for row in source_rows}
        for row in source_rows:
            sender = row[3].get("sender", {}) if isinstance(row[3], dict) else {}
            if isinstance(sender, dict):
                for label in (sender.get("nickname"), sender.get("card")):
                    if isinstance(label, str) and label.strip():
                        member_names.setdefault(label.strip(), row[2])
        executor = MemoryToolExecutor(engine=router.engine, group_id=job.scope_id,
            current_user_id=context.target_user_id, authorized_subject_ids=subjects, now=observation.now,
            recent_source_msg_ids=tuple(row[1] for row in source_rows), member_names=member_names,
            timeout_seconds=getattr(runtime.settings, "memory_memory_tool_timeout_seconds", DEFAULT_MEMORY_TOOL_TIMEOUT_SECONDS),
            max_results=getattr(runtime.settings, "memory_memory_tool_max_results", 5))
        def execute():
            with group_dispatch_scope(GroupDispatchScope(job.scope_id, valid, lambda: None)):
                require_valid()
                return executor.execute(name, args)
        raw = await context.blocking.run(execute)
        require_valid()
        try:
            value = json.loads(raw)
        except (ValueError, TypeError):
            value = {"evidence": raw[:12000], "kind": "source_bound_memory_evidence"}
        if isinstance(value, dict) and value.get("error"):
            return ToolOutcome(value, "failed")
        return ToolOutcome(value if isinstance(value, dict) else {"result": value})

    if getattr(runtime.settings, "memory_memory_tools_enabled", False):
        for specification in memory_tool_schemas():
            name = specification["name"]
            if name == "memory_write" and services.get("memory_write") is not True:
                continue
            parameters = dict(specification["parameters"])
            parameters["properties"] = {key: dict(value) for key, value in parameters["properties"].items()}
            for value in parameters["properties"].values():
                if value.get("type") == "string":
                    value["maxLength"] = 500
                if value.get("type") == "array":
                    value["maxItems"] = 20
            if name == "memory_search":
                maximum = max(1, min(20, int(getattr(runtime.settings, "memory_memory_tool_max_results", 5))))
                parameters["properties"]["limit"].update(maximum=maximum, default=maximum)
            async def handler(args, name=name):
                return await memory_call(name, args)
            tools.append(AgentTool(name, specification["description"], parameters, handler))

    async def observe_images(args):
        require_valid()
        if not getattr(runtime.settings, "llm_supports_vision_input", False):
            return ToolOutcome({"error": "vision_not_enabled"}, "blocked")
        requested = set(args["source_ids"])
        if not requested.issubset(context.observed_sources):
            return ToolOutcome({"error": "image_source_out_of_scope"}, "blocked")
        def materialize():
            require_valid()
            with canonical_group_image_inputs(router.engine, job.scope_id, tuple(sorted(requested)),
                    cache_dir=runtime.settings.data_dir / "image_cache") as pictures:
                result = []
                for picture in pictures:
                    require_valid()
                    # The artifact store accepts only its owned input roots.
                    # Copy decoded canonical pixels there, never a model path.
                    context.store.root.mkdir(parents=True, exist_ok=True)
                    source_path = Path(picture.local_path)
                    temporary = context.store.root / (".observed-" + uuid4().hex + source_path.suffix)
                    try:
                        temporary.write_bytes(source_path.read_bytes())
                        artifact_id = context.store.add(temporary, "observed_group_pixels")
                    finally:
                        temporary.unlink(missing_ok=True)
                    path = context.store.get(artifact_id)
                    result.append((artifact_id, ImageAttachment(url="", local_path=str(path),
                        source_message_id=picture.source_message_id, source_user_id=picture.source_user_id,
                        reference_subject=picture.reference_subject)))
                return result
        result = await context.blocking.run(materialize)
        require_valid()
        return ToolOutcome({"images": [{"artifact_id": item[0], "message_id": item[1].source_message_id,
            "author_id": item[1].source_user_id} for item in result], "actual_pixels": bool(result)},
            "ok" if result else "failed", images=tuple(item[1] for item in result))
    tools.append(AgentTool("observe_group_images", "读取已观察来源的真实群图片并保存可复用产物；不会猜测未见像素",
        _object(source_ids={"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 2**63 - 1}, "maxItems": 24}), observe_images))

    tasks = context.task_service
    if tasks is not None:
        async def task_status(args):
            require_valid()
            value = await context.blocking.run(tasks.status, group_id=job.scope_id, task_id=args["task_id"])
            require_valid()
            return ToolOutcome(value, "blocked" if value.get("error") else "ok")
        tools.append(AgentTool("persona_task_status", "读取本群人格任务真实状态；排队不表示刷新成功",
            _object(task_id={"type": "integer", "minimum": 1, "maximum": 2**63 - 1}), task_status))
        if services.get("persona_refresh") is True:
            async def request_refresh(args):
                require_valid()
                persona = runtime.personas.get(args["persona_key"], {})
                if (args["persona_key"] == "default" or persona.get("live_refresh") is not True
                        or persona.get("source_group_id") != job.scope_id):
                    return ToolOutcome({"error": "persona_out_of_scope_or_refresh_disabled"}, "blocked")
                value = await context.blocking.run(tasks.enqueue, group_id=job.scope_id,
                    persona_key=args["persona_key"], source_user_id=persona.get("source_user_id"),
                    source_message_ids=tuple(sorted(context.observed_sources)), request_key=args["request_key"], guard=guard)
                require_valid()
                return ToolOutcome({**value, "paid_execution":
                    "enabled" if services.get("persona_task_execution") is True else "paused_by_configuration"},
                    "blocked" if value.get("error") else "ok")
            tools.append(AgentTool("request_persona_refresh", "请求已允许人格的来源校验刷新；保持原服务门限，付费暂停时不会执行",
                _object(persona_key=_text(100), request_key=_text(100)), request_refresh))
    return tuple(tools)
