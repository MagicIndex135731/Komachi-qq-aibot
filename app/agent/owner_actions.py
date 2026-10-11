"""Durable owner directives shared by ingress surfaces, executed by the group process.

Only the canonical owner's original message grants authority. The public agent
receives that directive, never surrounding group/private conversation.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import logging
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.social_policy import configured_agent_jobs
from app.agent.tool_agent import AgentTool, ToolOutcome
from app.core.memory_source_guard import is_usable_memory_source
from app.storage.db import session_scope
from app.storage.models import Job, Message
from app.storage.repositories import JobRepository

logger = logging.getLogger(__name__)
OWNER_CONTROL_TOOLS = frozenset({'remember_intention', 'save_work_note', 'generate_image', 'generate_private_image',
    'execute_python', 'memory_write', 'request_persona_refresh', 'schedule_wakeup', 'cancel_wakeup',
    'schedule_private_wakeup', 'cancel_private_wakeup', 'set_nickname', 'set_avatar', 'set_group_card', 'set_group_persona'})

def restrict_directed_tools(tools, *, owner, allow_group_images=False):
    """Keep drawing available only when the caller authorizes this group scope."""
    if owner:
        return tools
    return [tool for tool in tools
        if (allow_group_images and tool.name == 'generate_image' and tool.effect == 'image')
        or (tool.name not in OWNER_CONTROL_TOOLS
            and (tool.effect not in {'image', 'write'} or tool.name in {'send_group', 'send_private'}))]


class OwnerActionService:
    job_type = 'owner_agent_action'

    def __init__(self, engine, *, runtime, clock=None):
        self.engine, self.runtime = engine, runtime
        self.clock = clock or (lambda: datetime.now(UTC))

    @contextmanager
    def transaction(self):
        with self.engine.connect() as connection:
            connection.exec_driver_sql('BEGIN IMMEDIATE')
            with Session(bind=connection) as db:
                try:
                    yield db
                    db.flush(); connection.commit()
                except BaseException:
                    connection.rollback(); raise

    def source(self, source_id):
        with session_scope(self.engine) as db:
            row = db.get(Message, source_id)
            if (row is None or row.user_id != self.runtime.settings.owner_qq
                    or not is_usable_memory_source(row, author_id=self.runtime.settings.owner_qq)
                    or not isinstance(row.raw_json, dict) or row.raw_json.get('direction') == 'outbound'
                    or not row.plain_text.strip() or len(row.plain_text) > 4000):
                return None
            if row.group_id is not None and not self.group_allowed(row.group_id):
                return None
            if row.group_id is None and row.raw_json.get('message_type') != 'private':
                return None
            fingerprint = sha256(json.dumps([row.id, row.platform_msg_id, row.group_id, row.user_id, row.plain_text,
                row.reply_to_msg_id, row.raw_json.get('message',row.raw_json.get('raw_message',''))],
                ensure_ascii=False).encode()).hexdigest()
            return {'source_id': row.id, 'source_group_id': row.group_id,
                'instruction': row.plain_text, 'fingerprint': fingerprint}

    def group_allowed(self, group_id):
        policy = self.runtime.group_policy
        raw = policy.get('groups', {}).get(str(group_id), {})
        values = {**policy.get('default_group_behavior', {}), **raw}
        return bool(raw) and all(values.get(flag) is True for flag in ('enabled', 'speak'))

    def target_job(self, target, group_id=0):
        jobs = configured_agent_jobs(self.runtime)
        if target == 'group' and self.group_allowed(group_id):
            return next((job for job in jobs if job.kind == 'group' and job.scope_id == group_id), None)
        if target == 'public':
            configured = next((job for job in jobs if job.kind == 'qzone_publish'), None) or next(
                (job for job in jobs if job.kind != 'group'), None)
            if configured is not None:
                return configured
            # Owner utility turns are not scheduled autonomous jobs. They
            # expose only enabled services, with no profile/social write tool.
            if self.runtime.settings.agent_runtime == 'harness':
                from app.agent.social_contracts import SocialJob
                return SocialJob('owner_utility', 'profile', self.runtime.settings.bot_qq,
                    autonomous_tools=True, allow_images=True, quiet_hours=None,
                    max_steps=32, max_model_calls_daily=500, max_image_calls_daily=10)
        return None

    def enqueue(self, source_id, *, target, group_id=0, include_images=False, before_commit=None,
                delay_seconds=0, goal=''):
        original = self.source(source_id)
        if original is None:
            return {'error': 'owner_required'}
        if type(group_id) is not int or type(include_images) is not bool or target not in {'public', 'group'}:
            return {'error': 'invalid_owner_target'}
        if type(delay_seconds) is not int or not 0 <= delay_seconds <= 604800 or not isinstance(goal,str) or len(goal)>500:
            return {'error': 'invalid_owner_schedule'}
        if target == 'group':
            group_id = group_id or original['source_group_id'] or 0
            if original['source_group_id'] is not None and group_id != original['source_group_id']:
                return {'error': 'target_not_authorized'}
        else:
            group_id = 0
        if self.target_job(target, group_id) is None:
            return {'error': 'target_not_authorized'}
        account = self.runtime.settings.bot_qq
        key = f'owner-action:{account}:{source_id}:{target}:{group_id}'
        if delay_seconds:
            key += ':' + sha256(json.dumps([goal,delay_seconds],ensure_ascii=False).encode()).hexdigest()[:24]
        with self.transaction() as db:
            existing = db.scalar(select(Job).where(Job.job_type == self.job_type, Job.job_key == key))
            if existing is not None:
                return self.data(existing)
            pending = db.scalar(select(Job.id).where(Job.job_type == self.job_type,
                Job.job_key.startswith(f'owner-action:{account}:'), Job.status.in_(['queued', 'running']))
                .offset(19).limit(1))
            if pending is not None:
                return {'error': 'owner_queue_capacity'}
            if before_commit:
                before_commit()
            if self.source(source_id) != original:
                return {'error': 'owner_source_revoked'}
            row = JobRepository(db).add_job(job_type=self.job_type, job_key=key,
                run_at=self.clock().astimezone(UTC).replace(tzinfo=None)+timedelta(seconds=delay_seconds), status='queued',
                payload_json={**original, 'target': target, 'group_id': group_id, 'include_images': include_images,
                    'kind': 'wakeup' if delay_seconds else 'directive', 'goal': goal})
            return self.data(row)

    @staticmethod
    def data(row):
        value = dict(row.payload_json)
        return {'task_id': row.id, 'status': row.status, 'reason': row.last_error_code,
            'target': value['target'], 'result': value.get('result', {}), 'run_at':row.run_at.isoformat(),
            'goal':value.get('goal',''), 'queued_is_not_completed': row.status == 'queued'}

    def wakeups(self,source_id):
        actor=self.source(source_id)
        if actor is None: return {'error':'owner_required'}
        with session_scope(self.engine) as db:
            rows=db.scalars(select(Job).where(Job.job_type==self.job_type,
                Job.job_key.startswith(f'owner-action:{self.runtime.settings.bot_qq}:')).order_by(Job.id.desc()).limit(100))
            return {'wakeups':[self.data(row) for row in rows if row.payload_json.get('kind')=='wakeup'
                and (actor['source_group_id'] is None or row.payload_json.get('source_group_id')==actor['source_group_id'])][:20]}

    def cancel(self,source_id,task_id):
        actor=self.source(source_id)
        if actor is None: return {'error':'owner_required'}
        with self.transaction() as db:
            row=db.get(Job,task_id)
            if (row is None or row.job_type!=self.job_type or row.payload_json.get('kind')!='wakeup'
                    or not row.job_key.startswith(f'owner-action:{self.runtime.settings.bot_qq}:')
                    or actor['source_group_id'] is not None and row.payload_json.get('source_group_id')!=actor['source_group_id']):
                return {'error':'owner_task_out_of_scope'}
            if row.status=='queued':
                row.status,row.last_error_code='cancelled','owner_goal_cancelled'
            return self.data(row)

    def get(self, task_id, *, source_id=None):
        actor=self.source(source_id) if source_id is not None else None
        if source_id is not None and actor is None:
            return {'error':'owner_required'}
        with session_scope(self.engine) as db:
            row = db.get(Job, task_id)
            if row is None or row.job_type != self.job_type or not row.job_key.startswith(f'owner-action:{self.runtime.settings.bot_qq}:'):
                return {'error': 'owner_task_unavailable'}
            if actor and actor['source_group_id'] is not None and row.payload_json.get('source_group_id')!=actor['source_group_id']:
                return {'error':'owner_task_out_of_scope'}
            if self.source(row.payload_json.get('source_id')) is None:
                return {'error':'owner_task_source_revoked','status':row.status}
            return self.data(row)

    def claim(self, worker, *, lease_seconds):
        now = self.clock().astimezone(UTC).replace(tzinfo=None)
        with self.transaction() as db:
            rows = list(db.scalars(select(Job).where(Job.job_type == self.job_type,
                Job.job_key.startswith(f'owner-action:{self.runtime.settings.bot_qq}:'),
                Job.status.in_(['queued', 'running'])).order_by(Job.id)))
            for row in rows:
                if row.status == 'running':
                    if row.lease_until and row.lease_until <= now:
                        from app.storage.agent_effects import AgentToolEffect
                        effects=list(db.scalars(select(AgentToolEffect).where(
                            AgentToolEffect.account_id==self.runtime.settings.bot_qq,
                            AgentToolEffect.run_id=='owner-job:'+str(row.id))))
                        unresolved=any(effect.status in {'dispatching','uncertain'} for effect in effects)
                        receipts=[{'tool':effect.operation,'status':effect.status,'receipt_id':effect.receipt}
                            for effect in effects if effect.status=='sent']
                        state='uncertain' if unresolved else 'failed'
                        row.status, row.last_error_code, row.completed_at = state, 'interrupted_not_replayed', now
                        row.payload_json={**row.payload_json,'result':{'status':state,
                            'reason':'interrupted_not_replayed','receipts':receipts}}
                    continue
                if row.run_at > now:
                    continue
                payload = dict(row.payload_json)
                original = self.source(payload['source_id'])
                if (original is None or original['fingerprint'] != payload['fingerprint']
                        or self.target_job(payload['target'], payload['group_id']) is None):
                    row.status, row.last_error_code, row.completed_at = 'blocked', 'owner_source_or_target_revoked', now
                    continue
                row.status, row.locked_by, row.locked_at = 'running', worker, now
                row.lease_until = now + timedelta(seconds=lease_seconds)
                return {**payload, 'task_id': row.id, 'worker': worker}
        return None

    def valid(self, claim):
        original = self.source(claim['source_id'])
        if original is None or original['fingerprint'] != claim['fingerprint'] or self.target_job(claim['target'], claim['group_id']) is None:
            return False
        with session_scope(self.engine) as db:
            row = db.get(Job, claim['task_id'])
            now = self.clock().astimezone(UTC).replace(tzinfo=None)
            return bool(row and row.status == 'running' and row.locked_by == claim['worker'] and row.lease_until > now)

    def finish(self, claim, result):
        with self.transaction() as db:
            row = db.get(Job, claim['task_id'])
            if row and row.status == 'running' and row.locked_by == claim['worker']:
                row.payload_json = {**row.payload_json, 'result': result}
                row.status = 'completed' if result.get('status') in {'sent', 'ok', 'silent'} else result.get('status', 'failed')
                row.last_error_code = str(result.get('reason', ''))[:96]
                row.completed_at = self.clock().astimezone(UTC).replace(tzinfo=None)
                original = self.source(claim['source_id'])
                if original is not None and original['fingerprint'] == claim['fingerprint']:
                    JobRepository(db).add_job(job_type='owner_action_notice',
                        job_key=f"owner-notice:{self.runtime.settings.bot_qq}:{claim['task_id']}",
                        run_at=row.completed_at, status='queued', payload_json={**original, 'result': result})


def build_owner_tools(service, source_id, guard):
    if service.source(source_id) is None:
        return ()
    async def request(args):
        guard.require_valid()
        value = service.enqueue(source_id, target=args['target'], group_id=args['group_id'],
            include_images=args['include_current_images'], before_commit=guard.require_valid)
        return ToolOutcome(value, 'blocked' if value.get('error') else 'ok')
    async def status(args):
        guard.require_valid()
        if service.source(source_id) is None:
            return ToolOutcome({'error': 'owner_required'}, 'blocked')
        value=service.get(args['task_id'],source_id=source_id)
        return ToolOutcome(value,'blocked' if value.get('error') else 'ok')
    async def capabilities(args):
        guard.require_valid()
        if service.source(source_id) is None:
            return ToolOutcome({'error': 'owner_required'}, 'blocked')
        jobs = configured_agent_jobs(service.runtime)
        return ToolOutcome({'enabled_surfaces': sorted({job.kind for job in jobs}),
            'groups': [job.scope_id for job in jobs if job.kind == 'group'],
            'compute': service.runtime.social_policy.get('compute', {}).get('enabled') is True,
            'services': service.runtime.social_policy.get('services', {}),
            'qzone_publish': any(job.kind=='qzone_publish' for job in jobs),
            'qzone_comment': any(job.kind=='qzone_comment' for job in jobs),
            'public_actions': '按配置启用好友动态/评论/空间发布/绘图/参考图/计算/扩展',
            'group_actions': '指定已授权群的历史/记忆/人格/图片/转发/绘图/计算/唤醒及当前已启用扩展',
            'authority': 'canonical_owner_message_only', 'autonomous_quota_charged': False})
    empty = {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}
    return (AgentTool('request_owner_action',
        '仅主人可用：将当前主人原消息交给对应能力Agent执行，不转交聊天上下文。公开空间/好友动态/评论/绘图/计算选public；私聊调用群历史、记忆、人格、转发等选group并指定已授权群。排队不是成功，完成会回到当前会话；不要重复排队。',
        {'type': 'object', 'properties': {'target': {'type': 'string', 'enum': ['public', 'group']},
            'group_id': {'type': 'integer', 'minimum': 0, 'maximum': 2**63-1},
            'include_current_images': {'type': 'boolean'}},
            'required': ['target', 'group_id', 'include_current_images'], 'additionalProperties': False}, request),
        AgentTool('owner_action_status', '读取主人委派任务真实状态，确认实际回执；不会重新执行',
            {'type': 'object', 'properties': {'task_id': {'type': 'integer', 'minimum': 1, 'maximum': 2**63-1}},
                'required': ['task_id'], 'additionalProperties': False}, status),
        AgentTool('list_owner_capabilities', '查看当前已启用能力和可指定群；不会打开任何关闭功能', empty, capabilities))


class OwnerActionWorker:
    def __init__(self, service, executor, *, connected=lambda: False):
        self.service, self.executor, self.connected = service, executor, connected
        self.worker = uuid4().hex
        self.running = False
        self._task = None

    async def start(self):
        self.running = True
        self._task = asyncio.create_task(self._run(), name='owner-actions')

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self):
        while self.running:
            if self.connected():
                try:
                    await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception('owner_action_worker_failed error_type=%s', type(exc).__name__)
            await asyncio.sleep(1)

    async def run_once(self):
        claim = self.service.claim(self.worker, lease_seconds=self.executor.router.runtime.settings.agent_background_turn_timeout_seconds + 120)
        if claim is None:
            return None
        worker = self
        class Guard:
            action_id = 'owner-job:' + str(claim['task_id'])
            dispatched = False
            def valid(self): return worker.connected() and worker.service.valid(claim)
            def require_valid(self):
                if not self.valid(): raise RuntimeError('owner_action_authority_revoked')
            def on_dispatch(self): self.require_valid(); self.dispatched = True
        guard = Guard()
        try:
            result = await self.executor.execute_owner_action(claim, guard)
        except asyncio.CancelledError:
            self.service.finish(claim, {'status': 'uncertain' if guard.dispatched else 'failed', 'reason': 'interrupted_not_replayed'})
            raise
        except Exception as exc:
            result = {'status': 'uncertain' if guard.dispatched else 'failed', 'reason': type(exc).__name__}
        self.service.finish(claim, result)
        logger.info('owner_action_finished task_id=%s status=%s', claim['task_id'], result.get('status'))
        return result


class OwnerNoticeWorker:
    """Return completion through the original surface's own process, at most once."""
    def __init__(self, service, sender, *, surface, connected=lambda: False):
        self.service, self.sender, self.surface, self.connected = service, sender, surface, connected
        self.worker, self.running, self._task = uuid4().hex, False, None

    async def start(self):
        self.running = True
        self._task = asyncio.create_task(self._run(), name='owner-notices-'+self.surface)

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel(); await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self):
        while self.running:
            if self.connected():
                try: await self.run_once()
                except asyncio.CancelledError: raise
                except Exception as exc:
                    logger.exception('owner_notice_failed error_type=%s', type(exc).__name__)
            await asyncio.sleep(1)

    def claim(self):
        now = self.service.clock().astimezone(UTC).replace(tzinfo=None)
        with self.service.transaction() as db:
            rows = list(db.scalars(select(Job).where(Job.job_type=='owner_action_notice',
                Job.job_key.startswith(f'owner-notice:{self.service.runtime.settings.bot_qq}:'),
                Job.status.in_(['queued','running'])).order_by(Job.id)))
            for row in rows:
                data = dict(row.payload_json)
                if (data['source_group_id'] is None) != (self.surface=='private'):
                    continue
                if row.status=='running':
                    if row.lease_until and row.lease_until<=now:
                        row.status, row.last_error_code = 'uncertain','interrupted_not_replayed'
                    continue
                if not self.valid_source(data):
                    row.status, row.last_error_code = 'blocked','owner_notice_source_revoked'
                    continue
                row.status, row.locked_by, row.lease_until = 'running',self.worker,now+timedelta(seconds=120)
                return {**data,'task_id':row.id}
        return None

    def valid_source(self, data):
        original = self.service.source(data['source_id'])
        if original is None or original['fingerprint']!=data['fingerprint']:
            return False
        result = data.get('result',{})
        if result.get('group_id'):
            if not self.service.group_allowed(result['group_id']): return False
            from app.agent.memory_integration import memory_dependencies_valid
            with session_scope(self.service.engine) as db:
                sources = result.get('source_message_ids',[])
                rows = list(db.scalars(select(Message).where(Message.id.in_(sources), Message.group_id==result['group_id'])))
                if len(rows)!=len(sources) or any(not is_usable_memory_source(row, group_id=result['group_id']) for row in rows):
                    return False
            if not memory_dependencies_valid(self.service.engine,group_id=result['group_id'],
                    dependencies=result.get('memory_dependencies',{}),now=self.service.clock()):
                return False
        return True

    async def run_once(self):
        from app.adapters.outbound_guard import GroupDispatchScope, group_dispatch_scope
        from app.adapters.sender import OutboundMessage, OutboundPrivateMessage
        from app.agent.artifacts import AutonomousArtifactStore
        from app.agent.tool_dispatch import ToolDispatchBoundary
        data = self.claim()
        if data is None: return None
        worker = self
        class Guard:
            action_id = 'owner-notice:'+str(data['task_id'])
            dispatched = False
            def valid(self):
                if not worker.connected() or not worker.valid_source(data): return False
                with session_scope(worker.service.engine) as db:
                    row=db.get(Job,data['task_id'])
                    return bool(row and row.status=='running' and row.locked_by==worker.worker
                        and row.lease_until>worker.service.clock().astimezone(UTC).replace(tzinfo=None))
            def require_valid(self):
                if not self.valid(): raise RuntimeError('owner_notice_authority_revoked')
            def on_dispatch(self): self.require_valid(); self.dispatched=True
        guard=Guard()
        settings=self.service.runtime.settings
        boundary=ToolDispatchBoundary(engine=self.service.engine,account_id=settings.bot_qq,
            scope='private:'+str(settings.owner_qq) if self.surface=='private' else 'group:'+str(data['source_group_id']),
            guard=guard,clock=self.service.clock,owner_authority=guard.valid)
        result=data['result']; report=result.get('report',{})
        receipts=[r for r in result.get('receipts',[]) if r.get('status')=='sent']
        status=('实际完成：'+', '.join(dict.fromkeys(r['tool'] for r in receipts))) if receipts else (
            '查询结果' if result.get('status')=='ok' else '任务未完成：'+str(result.get('reason','unknown')))
        text=(status+'\n'+report.get('text','')).strip()[:2000]
        images=[]
        if report.get('artifact_ids'):
            store=AutonomousArtifactStore(settings.data_dir/'autonomous_artifacts',account_id=settings.bot_qq,
                scope=report['artifact_scope'],engine=self.service.engine,source_message_ids=lambda: ())
            for artifact_id in report['artifact_ids']:
                path=store.get(artifact_id)
                if path is None or path.suffix not in {'.png','.jpg','.webp','.gif'}:
                    return self.finish(data,'blocked','owner_artifact_revoked')
                images.append(str(path))
        async def send(args):
            guard.require_valid()
            if self.surface=='private':
                boundary.on_dispatch()
                actual=list(await self.sender.send_private_text(OutboundPrivateMessage(settings.owner_qq,text)))
                for image in images:
                    guard.require_valid()
                    receipt=await self.sender.send_private_image(user_id=settings.owner_qq,image_file=image)
                    if not receipt: raise RuntimeError('missing_owner_image_receipt')
                    actual.append(receipt)
            else:
                from types import SimpleNamespace
                from app.agent.social_executor import SocialExecutor
                from app.agent.social_contracts import SocialJob,SocialObservation,SocialDecision
                writer=SocialExecutor(router=SimpleNamespace(engine=self.service.engine,runtime=self.service.runtime),
                    sender=self.sender,qzone_adapter=None,clock=self.service.clock)
                observation=SocialObservation(SocialJob('owner_notice','group',data['source_group_id']),
                    self.service.clock(),'',source_message_ids=(data['source_id'],))
                provisional=writer._reserve_group_output(observation,
                    SocialDecision('send_group',text=text,image_files=tuple(images)),boundary.action_id)
                if provisional is None: raise RuntimeError('duplicate_owner_notice')
                scope=GroupDispatchScope(data['source_group_id'],boundary.valid,boundary.on_dispatch)
                try:
                    with group_dispatch_scope(scope):
                        actual=list(await self.sender.send_group_media(group_id=data['source_group_id'],text=text,image_files=tuple(images))
                            if images else await self.sender.send_group_text(OutboundMessage(data['source_group_id'],text)))
                    if not actual: raise RuntimeError('missing_owner_notice_receipt')
                    writer._finish_group_output(provisional,state='sent',receipt=actual[0])
                except BaseException:
                    writer._finish_group_output(provisional,state='uncertain' if boundary.dispatched else 'deleted')
                    raise
            if not actual: raise RuntimeError('missing_owner_notice_receipt')
            return ToolOutcome({'message_ids':actual},'sent',actual[0])
        tool=AgentTool('owner_action_notice','return actual completion',{},send,'write')
        try:
            outcome=await boundary.invoke(tool,{'text':text,'artifact_ids':report.get('artifact_ids',[])},'notice')
        except asyncio.CancelledError:
            self.finish(data,'uncertain' if boundary.dispatched else 'failed','interrupted_not_replayed'); raise
        except Exception:
            self.finish(data,'uncertain' if boundary.dispatched else 'failed','notice_failed'); raise
        self.finish(data,'completed' if outcome.status=='sent' else outcome.status,'actual_notice:'+outcome.status)
        return outcome

    def finish(self,data,status,reason):
        with self.service.transaction() as db:
            row=db.get(Job,data['task_id'])
            if row and row.status=='running' and row.locked_by==self.worker:
                row.status,row.last_error_code,row.completed_at=status,reason,self.service.clock().astimezone(UTC).replace(tzinfo=None)
