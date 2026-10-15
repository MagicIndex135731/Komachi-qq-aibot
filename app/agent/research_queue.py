"""Durable read-only child jobs, separately consumed by group/private processes."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
import hashlib
import json
import logging
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.memory_source_guard import is_usable_memory_source
from app.agent.runtime_trace import AgentTrace, current_trace_context, execution_event, observe_with
from app.storage.db import session_scope
from app.storage.models import Job, Message

logger = logging.getLogger(__name__)
ACTIVE_STATES = ('queued', 'running', 'result_ready', 'delivering')
WAITING_STATES = ('pending', *ACTIVE_STATES)


class ResearchJobGuard:
    def __init__(self, queue, job_id, payload, phase):
        self.queue, self.job_id, self.payload, self.phase = queue, job_id, payload, phase
        self.sources = set(payload['sources'])
        self.memory = dict(payload.get('memory', {}))
        self.action_id = 'research-result:' + str(job_id)
        self.dispatched = False
        self.trace_trigger = {'trigger_kind': 'research_result', 'trigger_reason_code': 'delegated_result_ready',
            'trigger_reason': '独立资料任务已结束，将摘要回传原主会话。',
            'trigger_evidence': {'task_id': payload['task_id'], 'source_id': payload['source_id']}}

    @property
    def source_message_ids(self):
        return tuple(sorted(self.sources))

    def valid_source(self, identifier, *, author=False):
        return self.valid_sources((identifier,), author=author)

    def valid_sources(self, identifiers, *, author=False):
        if any(type(identifier) is not int for identifier in identifiers):
            return False
        identifiers = set(identifiers)
        def confirmed_bot_dependency(row):
            return (not author and getattr(self.queue, 'allow_confirmed_bot_dependencies', False)
                and row.user_id == self.queue.account_id and (row.raw_json or {}).get('direction') == 'outbound')
        with session_scope(self.queue.engine) as session:
            rows = tuple(session.scalars(select(Message).where(Message.id.in_(identifiers))))
            valid = len(rows) == len(identifiers) and all(is_usable_memory_source(row, group_id=self.payload['scope_id'] if self.queue.surface == 'group' else None)
                and row.group_id == (self.payload['scope_id'] if self.queue.surface == 'group' else None)
                and (not author and self.queue.surface == 'group' or row.user_id == self.payload['actor_id'] or confirmed_bot_dependency(row))
                and ((row.raw_json or {}).get('direction') != 'outbound' or confirmed_bot_dependency(row)) for row in rows)
        return valid

    def valid_dependencies(self, metadata):
        if metadata.get('context_snapshot'):
            from app.agent.context_dependencies import context_snapshot_valid
            if not context_snapshot_valid(self.queue.engine, account_id=self.queue.account_id,
                    scope=self.payload['scope'], identifier=metadata['context_snapshot'],
                    group_id=self.payload['scope_id'] if self.queue.surface == 'group' else None,
                    private_user_id=self.payload['actor_id'] if self.queue.surface == 'private' else None,
                    now=self.queue.clock()):
                return False
        sources, memory = metadata.get('sources', []), metadata.get('memory', {})
        if (not isinstance(sources, list) or not isinstance(memory, dict)
                or len(self.sources | set(sources)) > 200 or len(self.memory.keys() | memory.keys()) > 200
                or not self.valid_sources(sources)):
            return False
        if self.queue.surface == 'group':
            from app.agent.memory_integration import memory_dependencies_valid
            if not memory_dependencies_valid(self.queue.engine, group_id=self.payload['scope_id'],
                    dependencies=memory, now=self.queue.clock()):
                return False
        elif memory:
            return False
        self.sources.update(sources)
        self.memory.update(memory)
        return True

    def snapshot(self):
        return {'sources': sorted(self.sources), 'memory': dict(self.memory),
            **({'context_snapshot': self.payload['context_snapshot']} if self.payload.get('context_snapshot') else {})}

    def valid(self):
        if not self.queue.running or not self.queue.authorized(self.payload):
            return False
        if self.payload.get('context_snapshot'):
            from app.agent.context_dependencies import context_snapshot_valid
            if not context_snapshot_valid(self.queue.engine, account_id=self.queue.account_id,
                    scope=self.payload['scope'], identifier=self.payload['context_snapshot'],
                    group_id=self.payload['scope_id'] if self.queue.surface == 'group' else None,
                    private_user_id=self.payload['actor_id'] if self.queue.surface == 'private' else None,
                    now=self.queue.clock()):
                return False
        with session_scope(self.queue.engine) as session:
            row = session.get(Job, self.job_id)
            if not (row and row.job_type == self.queue.job_type and row.locked_by == self.queue.owner
                    and row.status == self.phase and row.lease_until and row.lease_until.replace(tzinfo=UTC) > self.queue.clock()
                    and self.queue.clock().timestamp() - self.payload['created_at'] <= 7 * 86400):
                return False
        return self.valid_source(self.payload['source_id'], author=True) and self.valid_dependencies(self.snapshot())

    def require_valid(self):
        if not self.valid():
            raise RuntimeError('research_job_authority_revoked')

    def on_dispatch(self):
        self.require_valid()
        self.dispatched = True


class ResearchDelegationQueue:
    job_type = 'delegated_research'
    max_execution_attempts = 2
    interrupted_execution_status = 'queued'
    exhausted_execution_status = 'failed'
    guard_type = ResearchJobGuard

    def __init__(self, engine, *, account_id, surface, execute, deliver, authorized, clock=None,
                 task_workspace_root=None, trace_root=None):
        self.engine, self.account_id, self.surface = engine, account_id, surface
        self.execute, self.deliver, self.authorized = execute, deliver, authorized
        self.clock = clock or (lambda: datetime.now(UTC))
        self.task_workspace_root = task_workspace_root
        self.trace_root = trace_root
        self.owner, self.running, self.worker = uuid4().hex, False, None
        self.wake = asyncio.Event()

    def prefix(self, scope_id=None):
        return f'research:{self.account_id}:{self.surface}:' + (f'{scope_id}:' if scope_id is not None else '')

    @contextmanager
    def transaction(self, before_commit=None):
        with self.engine.connect() as connection:
            connection.exec_driver_sql('BEGIN IMMEDIATE')
            with Session(bind=connection) as session:
                try:
                    yield session
                    session.flush()
                    if before_commit:
                        before_commit()
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise

    def enqueue(self, payload, *, parent_turn, before_commit):
        before_commit()
        if (not self.authorized(payload) or len(payload['sources']) > 200
                or len(json.dumps(payload, ensure_ascii=False)) > 64000):
            return {'status': 'blocked', 'reason': 'delegation_source_or_size_invalid'}
        key = self.prefix(payload['scope_id']) + hashlib.sha256(str(parent_turn).encode()).hexdigest()[:24]
        now = self.clock().astimezone(UTC).replace(tzinfo=None)
        with self.transaction(before_commit) as session:
            row = session.scalar(select(Job).where(Job.job_type == self.job_type, Job.job_key == key))
            if row is None:
                waiting = tuple(session.scalars(select(Job).where(Job.job_type == self.job_type,
                    Job.job_key.startswith(self.prefix()), Job.status.in_(WAITING_STATES)).limit(512)))
                own = lambda item: (item.payload_json['scope_id'] == payload['scope_id']
                    and item.payload_json['actor_id'] == payload['actor_id'])
                if len(waiting) >= 512 or sum(own(item) for item in waiting) >= 128:
                    return {'status': 'blocked', 'reason': 'delegation_queue_capacity'}
                active = [item for item in waiting if item.status in ACTIVE_STATES]
                state = 'pending' if len(active) >= 16 or sum(own(item) for item in active) >= 4 else 'queued'
                row = Job(job_type=self.job_type, job_key=key, status=state, run_at=now,
                          max_attempts=self.max_execution_attempts, payload_json={**payload,
                              '_trace_parent': current_trace_context(), 'created_at': self.clock().timestamp()})
                session.add(row)
                session.flush()
            result = {'status': 'queued', 'task_id': row.payload_json['task_id'], 'job_id': row.id,
                      'execution_status': row.status,
                      'delivery': 'background_result_returns_to_main', 'guidance': '排队不代表完成；主对话可继续处理新消息。'}
        self.wake.set()
        execution_event('job.queued', {'job_id': result['job_id'], 'task_id': result['task_id'],
            'status': result['execution_status'], 'operation': payload.get('operation'),
            'execution_kind': 'background_tool' if self.job_type == 'agent_media_operation' else 'delegation_job',
            'delivery_mode': payload.get('delivery_mode', 'review')})
        return result

    async def start(self):
        if not self.running:
            self.running = True
            self.worker = asyncio.create_task(self._run(), name=self.prefix() + 'worker')

    async def stop(self):
        self.running = False
        self.wake.set()
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
            self.worker = None

    def cancel_scope(self, scope_id, *, before_source_id=None):
        with self.transaction() as session:
            for row in session.scalars(select(Job).where(Job.job_type == self.job_type,
                    Job.job_key.startswith(self.prefix(scope_id)), Job.status.in_(WAITING_STATES))):
                if before_source_id is not None and (row.payload_json or {}).get('source_id', 0) > before_source_id:
                    continue
                row.status, row.last_error_code = 'cancelled', 'scope_cancelled'

    def cancel_task(self, scope_id, actor_id, task_id, *, before_commit):
        with self.transaction(before_commit) as session:
            for row in session.scalars(select(Job).where(Job.job_type == self.job_type,
                    Job.job_key.startswith(self.prefix(scope_id)), Job.status.in_(WAITING_STATES))):
                if (row.payload_json.get('actor_id') == actor_id
                        and row.payload_json.get('parent_task_id') == task_id):
                    row.status, row.last_error_code = 'cancelled', 'task_cancelled'

    def claim(self):
        now = self.clock().astimezone(UTC).replace(tzinfo=None)
        with self.transaction() as session:
            # Capacity limits execution, not the size of an authorized request.
            # Promotion is durable; tick revalidates source/permission/TTL before
            # every read and delivery. It never creates timer authority.
            active = list(session.scalars(select(Job).where(Job.job_type == self.job_type,
                Job.job_key.startswith(self.prefix()), Job.status.in_(ACTIVE_STATES))))
            for pending in session.scalars(select(Job).where(Job.job_type == self.job_type,
                    Job.job_key.startswith(self.prefix()), Job.status == 'pending')
                    .order_by(Job.id).limit(512)):
                if len(active) >= 16:
                    break
                if sum(item.payload_json['scope_id'] == pending.payload_json['scope_id']
                        and item.payload_json['actor_id'] == pending.payload_json['actor_id'] for item in active) < 4:
                    pending.status = 'queued'
                    active.append(pending)
            session.flush()
            rows = tuple(session.scalars(select(Job).where(Job.job_type == self.job_type,
                Job.job_key.startswith(self.prefix()), Job.status.in_(('queued', 'running', 'result_ready', 'delivering')))
                .order_by(Job.run_at, Job.id).limit(16)))
            for row in rows:
                if row.status in {'running', 'delivering'}:
                    if row.lease_until and row.lease_until > now:
                        continue
                    if row.status == 'delivering':
                        row.status, row.last_error_code = 'uncertain', 'delivery_owner_lost'
                        continue
                    if row.attempt_count >= row.max_attempts:
                        row.status, row.last_error_code = self.exhausted_execution_status, 'research_attempts_exhausted'
                        continue
                    row.status = 'queued'
                if row.run_at > now:
                    continue
                phase = 'running' if row.status == 'queued' else 'delivering'
                if phase == 'running' and row.attempt_count >= row.max_attempts:
                    row.status, row.last_error_code = self.exhausted_execution_status, 'research_attempts_exhausted'
                    continue
                row.status, row.locked_by, row.locked_at = phase, self.owner, now
                row.lease_until = now + timedelta(seconds=900)
                if phase == 'running':
                    row.attempt_count += 1
                return row.id, dict(row.payload_json), phase

    def finish(self, job_id, phase, *, status, payload=None, reason=''):
        completed_payload = None
        with self.transaction() as session:
            row = session.get(Job, job_id)
            if row and row.status == phase and row.locked_by == self.owner:
                row.status, row.last_error_code = status, reason[:96]
                if payload is not None:
                    row.payload_json = payload
                row.completed_at = self.clock().astimezone(UTC).replace(tzinfo=None)
                completed_payload = dict(row.payload_json)
        if (completed_payload and self.task_workspace_root and completed_payload.get('parent_task_id')
                and status not in WAITING_STATES):
            try:
                from app.agent.research_tasks import ResearchTaskWorkspace
                check = ResearchJobGuard(self, job_id, completed_payload, phase)
                if not self.authorized(completed_payload):
                    return status
                task = ResearchTaskWorkspace(self.task_workspace_root, scope=completed_payload['scope'],
                    actor_id=completed_payload['actor_id'], source_id=completed_payload['source_id'],
                    request=completed_payload.get('goal', ''), valid_source=check.valid_source,
                    valid_dependencies=check.valid_dependencies, clock=lambda: self.clock().timestamp())
                task.resume(completed_payload['parent_task_id'])
                task.record_job_result(job_id, status)
            except (ValueError, OSError, RuntimeError):
                logger.warning('agent_task_checkpoint_unavailable job_id=%s', job_id)
        return status if completed_payload is not None else None

    async def tick(self):
        claimed = self.claim()
        if not claimed:
            return False
        job_id, payload, phase = claimed
        guard = self.guard_type(self, job_id, payload, phase)
        media = self.job_type == 'agent_media_operation'
        lineage = payload.get('_trace_parent')
        observer = AgentTrace(self.trace_root, context={
            **(lineage if isinstance(lineage, dict) else {}),
            'surface': self.surface, 'scope_id': payload['scope_id'], 'actor_id': payload['actor_id'],
            'source_id': payload['source_id'], 'context_scope': payload['scope'],
            'context_kind': 'media_task' if media else 'research_job',
            'execution_kind': 'background_tool' if media else 'delegation_job',
            'job_id': job_id, 'task_id': payload['task_id'], 'job_phase': phase,
            'operation': payload.get('operation'), 'delivery_mode': payload.get('delivery_mode', 'review'),
            'request_preview': payload.get('operation') or payload.get('goal', '')[:180],
            'trigger_kind': 'background_execution', 'trigger_reason': '执行原请求已持久排队的后台任务；交付以实际回执为准。'},
            session_key=f'job:{self.account_id}:{job_id}:{phase}',
            conversation_key=f'job:{self.account_id}:{job_id}')
        settled, reason = None, 'job_state_changed'
        observer.event('job.phase.started', {'job_id': job_id, 'phase': phase,
            'operation': payload.get('operation'), 'delivery_mode': payload.get('delivery_mode', 'review')})
        try:
            with observe_with(observer):
                guard.require_valid()
                if phase == 'running':
                    if media:
                        observer.event('tool.started', {'name': payload['operation'], 'call_id': 'media-execute',
                            'effect': 'image' if payload['operation'] == 'generate_image' else 'read',
                            'arguments': payload['arguments']})
                    result = await asyncio.wait_for(self.execute(payload, guard), timeout=330)
                    if media:
                        observer.event('tool.completed', {'name': payload['operation'], 'call_id': 'media-execute',
                            'status': result.get('status', 'unknown'), 'data': result.get('data', {})})
                    guard.require_valid()
                    settled = self.finish(job_id, phase, status='result_ready', payload={**payload, **guard.snapshot(), 'result': result})
                else:
                    result = await asyncio.wait_for(self.deliver(payload, guard), timeout=650)
                    guard.require_valid()
                    settled = self.finish(job_id, phase, status=result if result in {'completed', 'uncertain', 'failed', 'blocked'} else 'failed')
                reason = 'background_result_ready' if phase == 'running' else 'background_delivery_finished'
        except asyncio.CancelledError:
            # Reads may be safely resumed at most once; delivery may already
            # have happened and must never be automatically replayed.
            settled = self.finish(job_id, phase, status=self.interrupted_execution_status if phase == 'running' else 'uncertain', reason='worker_stopped')
            reason = 'worker_stopped'
            raise
        except Exception as exc:
            settled = self.finish(job_id, phase, status='uncertain' if guard.dispatched else 'blocked', reason=type(exc).__name__)
            reason = type(exc).__name__
            logger.warning('delegated_research_job_failed surface=%s error_type=%s', self.surface, type(exc).__name__)
        finally:
            observer.event('job.phase.completed', {'job_id': job_id, 'phase': phase,
                'status': settled or 'interrupted', 'reason': reason,
                'delivery_mode': payload.get('delivery_mode', 'review')})
            observer.finish('ok' if settled in {'result_ready', 'completed'} else settled or 'interrupted', reason)
            await asyncio.to_thread(observer.close)
        return True

    async def _run(self):
        while self.running:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning('delegated_research_scheduler_failed surface=%s error_type=%s', self.surface, type(exc).__name__)
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=1)
            except TimeoutError:
                pass
