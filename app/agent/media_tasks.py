"""Durable deterministic media work. No child model or native chat session."""
from __future__ import annotations

from contextvars import ContextVar
from datetime import UTC
from hashlib import sha256
import json
from types import SimpleNamespace
from uuid import uuid4
from sqlalchemy import select

from app.agent.research_queue import ResearchDelegationQueue, ResearchJobGuard
from app.agent.tool_agent import AgentTool, ToolOutcome, validate_arguments
from app.storage.db import session_scope
from app.storage.models import Job
from app.storage.models import Message

MEDIA_OPERATIONS = frozenset({'generate_image', 'download_web_images', 'download_web_video', 'download_twitter_media'})
local_fallback_plan = ContextVar('local_media_fallback_plan', default=None)


def download_delivery_properties():
    return {'quality':{'type':'string','enum':['best_available','original_only'],
        'description':'默认best_available按预算选原生版本；用户明确必须原图且不接受降档才original_only'},
        'delivery_mode': {'type':'string', 'enum':['auto','review'],
        'description':'用户明确给定来源全部媒体原样发送可选auto；按人物/主题筛选用review（默认），取结果后以原帖正文、日期和类型判断，无须逐份预览。auto绑定原请求，按完成先后交付，不另调用发送工具。'},
        'delivery_text': {'type':'string','maxLength':300,
            'description':'auto完成后的简短说明，不得提前宣称已发送；可留空。'}}


def build_media_batch_tool(tools, guard):
    """Compose offered operations; each handler still checks its own authority.

    No URLs, site names, tool order or permission decisions are inferred here.
    The durable queue manages pending work after this model turn has finished.
    """
    offered = {tool.name: tool for tool in tools if tool.name in MEDIA_OPERATIONS}
    if not offered:
        return ()
    async def batch(args):
        operations = args['operations']
        # Validate the whole batch before any child can be queued. A schema
        # matching a different operation must not authorize this operation.
        if any(item['operation'] not in offered or not validate_arguments(
                item['arguments'], offered[item['operation']].parameters) for item in operations):
            return ToolOutcome({'error': 'invalid_media_batch_arguments'}, 'blocked')
        jobs, failures = [], []
        for index, item in enumerate(operations):
            guard.require_valid()
            tool = offered[item['operation']]
            if tool.authorization is not None and tool.authorization() is not True:
                failures.append({'index': index, 'error': 'operation_not_authorized'})
                continue
            # Generations retain their paid image budget boundary; this batch
            # is for deterministic downloads only, not a way around it.
            outcome = await tool.handler(item['arguments'])
            if outcome.status == 'ok' and type(outcome.data.get('job_id')) is int:
                jobs.append({'index': index, 'operation': item['operation'], **outcome.data})
            else:
                failures.append({'index': index, 'error': outcome.data.get('error') or
                    outcome.data.get('reason') or 'media_not_queued', 'status': outcome.status})
        guard.require_valid()
        return ToolOutcome({'jobs': jobs, 'failures': failures, 'requested_count': len(operations),
            'accepted_count': len(jobs), 'all_accepted': not failures, 'task_status': 'waiting' if jobs else 'blocked',
            'guidance': '任务已持久排队，pending等待容量后继续；不是已完成。保留未接受项和覆盖缺口，不重复提交已接受任务。'},
            'ok' if jobs else 'blocked')
    offered = {name: tool for name, tool in offered.items() if name != 'generate_image'}
    if not offered:
        return ()
    return (AgentTool('queue_media_batch',
        '将已发现的多项下载操作持久排队，每次最多16项、每进程最多4个媒体worker并行。各项仍受原工具权限/大小/格式限制；等待容量后继续，先完成的可先交付，排队不代表完成。分析筛选用review，纯下载交付可用auto。',
        {'type': 'object', 'properties': {'operations': {'type': 'array', 'minItems': 1, 'maxItems': 16,
            'items': {'type': 'object', 'properties': {'operation': {'type': 'string', 'enum': sorted(offered)},
                'arguments': {'anyOf': [tool.parameters for tool in offered.values()]}},
                'required': ['operation', 'arguments'], 'additionalProperties': False}}},
            'required': ['operations'], 'additionalProperties': False}, batch,
            side_effects=('enqueue_work',), execution='background'),)


class MediaJobGuard(ResearchJobGuard):
    def __init__(self, *args):
        super().__init__(*args)
        self.action_id = 'media-result:' + str(self.job_id)
        self.trace_trigger.update(trigger_kind='media_result', trigger_reason_code='media_result_ready',
            trigger_reason='媒体后台执行已结束，将真实产物和状态回传原主会话。')


class MediaTaskQueue(ResearchDelegationQueue):
    job_type = 'agent_media_operation'
    concurrency = 4
    # Losing a worker must not silently buy the same image a second time.
    max_execution_attempts = 1
    interrupted_execution_status = 'uncertain'
    exhausted_execution_status = 'uncertain'
    guard_type = MediaJobGuard
    allow_confirmed_bot_dependencies = True
    claim_scan_limit = 512

    def __init__(self, engine, *, authorized, available=lambda: True, **kwargs):
        self.available = available
        self._deliver_result = kwargs.pop('deliver')
        kwargs['deliver'] = self._deliver_request
        super().__init__(engine, authorized=lambda payload: authorized(payload)
            and payload.get('source_fingerprints') == self.source_fingerprints(payload['sources']), **kwargs)

    @staticmethod
    def _review_download(payload):
        return (payload.get('delivery_mode', 'review') == 'review'
            and payload.get('operation') in MEDIA_OPERATIONS - {'generate_image'})

    def occupies_capacity(self, row):
        # READY files wait for their request's siblings, not for a worker slot.
        return super().occupies_capacity(row) and not (
            row.status == 'result_ready' and self._review_download(row.payload_json))

    def _review_siblings(self, session, payload):
        return list(session.scalars(select(Job).where(Job.job_type == self.job_type,
            Job.job_key.startswith(self.prefix(payload['scope_id'])),
            Job.payload_json['scope'].as_string() == payload['scope'],
            Job.payload_json['actor_id'].as_integer() == payload['actor_id'],
            Job.payload_json['source_id'].as_integer() == payload['source_id'],
            Job.payload_json['parent_task_id'].as_string() == payload['parent_task_id']
                if payload.get('parent_task_id') else Job.payload_json['parent_task_id'].as_string().is_(None)
            ).order_by(Job.id)))

    async def _deliver_request(self, payload, guard):
        if not self._review_download(payload):return await self._deliver_result(payload, guard)
        guard.require_valid()
        with session_scope(self.engine) as session:
            rows = [(row.id, row.status, dict(row.payload_json))
                for row in self._review_siblings(session, payload) if self._review_download(row.payload_json)]
        returned = []
        for identifier, state, item in rows:
            if state not in {'completed', 'result_ready', 'delivering', 'failed', 'blocked'}:continue
            if not self.authorized(item) or not guard.valid_dependencies({
                    'sources': item['sources'], 'memory': item.get('memory', {}),
                    **({'context_snapshot': item['context_snapshot']} if item.get('context_snapshot') else {})}):
                raise RuntimeError('media_batch_source_revoked')
            returned.append((identifier, item.get('item_result', item.get('result', {
                'status': 'failed', 'data': {'error': 'media_execution_failed'}}))))
        result = merge_media_results(returned, payload)
        outcome = await self._deliver_result({**payload, 'result': result}, guard)
        guard.require_valid()
        # The leader's durable result handoff is idempotent. Settle siblings
        # only after it succeeds; a lost leader fences the request as unknown.
        if outcome == 'completed':
            with self.transaction(guard.require_valid) as session:
                for row in self._review_siblings(session, payload):
                    if row.id != guard.job_id and row.status == 'result_ready' and self._review_download(row.payload_json):
                        row.status = 'completed';row.completed_at = self.clock().astimezone(UTC).replace(tzinfo=None)
                        row.last_error_code = 'media_result_aggregated'
        return outcome

    def claim(self):
        # A transient disconnected gateway is not revoked request authority.
        # Do not spend the single execution attempt during process startup.
        return super().claim() if self.available() else None

    def admit_claim(self, session, row, phase, now):
        if self._review_download(row.payload_json):
            siblings = [s for s in self._review_siblings(session, row.payload_json)
                if self._review_download(s.payload_json)]
            if any(s.status in {'uncertain', 'cancelled'} for s in siblings):
                row.status = 'blocked';row.last_error_code = 'media_request_unknown_or_cancelled';row.completed_at = now
                return False
            if phase == 'delivering' and any(s.id != row.id and s.status in {
                    'pending', 'queued', 'running', 'delivering'} for s in siblings):return False
        plan=row.payload_json.get('fallback_plan')
        if not plan:return True
        payload=row.payload_json
        siblings=list(session.scalars(select(Job).where(Job.job_type==self.job_type,
            Job.job_key.startswith(self.prefix(payload['scope_id'])),
            Job.payload_json['fallback_plan']['key'].as_string()==plan['key'],
            Job.payload_json['scope'].as_string()==payload['scope'],
            Job.payload_json['actor_id'].as_integer()==payload['actor_id'],
            Job.payload_json['source_id'].as_integer()==payload['source_id'])))
        if any(j.status in {'uncertain','cancelled'} for j in siblings):
            row.status='blocked';row.last_error_code='media_fallback_unknown_or_cancelled'
            row.completed_at=now
            return False
        if phase!='running':return True
        # Only confirmed deliveries count. Byte/logical identities prevent a
        # repeated clip from satisfying the goal twice.
        ready={};types=set()
        for sibling in siblings:
            if sibling.status!='completed':continue
            result=sibling.payload_json.get('result',{})
            if result.get('status')!='ok':continue
            data=result.get('data',{})
            for item in data.get('media',[]):
                kind='image' if item.get('kind')=='photo' else 'video'
                identity=item.get('sha256') or item.get('logical_media_id') or item.get('artifact_id')
                if identity:ready[identity]=kind;types.add(kind)
            if not data.get('media'):
                for kind,ids in (('image',data.get('artifact_ids',[])),
                        ('video',data.get('video_artifact_ids',[]) or ([data['video_artifact_id']] if data.get('video_artifact_id') else []))):
                    for identifier in ids:ready[identifier]=kind;types.add(kind)
        if len(ready)>=plan['minimum_count'] and set(plan['media_types'])<=types:
            row.status='completed';row.last_error_code='media_fallback_goal_satisfied'
            row.completed_at=now
            return False
        earlier=[j for j in siblings if j.payload_json['fallback_plan']['wave']<plan['wave']]
        return not any(j.status in {'pending','queued','running','result_ready','delivering'} for j in earlier)

    def source_fingerprints(self, sources):
        with session_scope(self.engine) as session:
            result = {}
            for row in session.scalars(select(Message).where(Message.id.in_(sources))):
                result[str(row.id)] = sha256(json.dumps([row.id, row.group_id, row.user_id,
                    row.platform_msg_id, row.plain_text, row.reply_to_msg_id, row.raw_json],
                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            return result if len(result) == len(set(sources)) else None

    def prefix(self, scope_id=None):
        return f'media:{self.account_id}:{self.surface}:' + (f'{scope_id}:' if scope_id is not None else '')

    def submit(self, payload, *, parent_turn, before_commit, continuation=False):
        if payload.get('operation') not in MEDIA_OPERATIONS:
            return ToolOutcome({'error': 'media_operation_not_allowed'}, 'blocked')
        arguments = dict(payload['arguments'])
        mode, caption = arguments.pop('delivery_mode', 'review'), arguments.pop('delivery_text', '')
        if (mode not in {'auto','review'} or not isinstance(caption,str) or len(caption)>300 or '[CQ:' in caption
                or mode=='auto' and (payload['operation']=='generate_image'
                    or payload['surface']=='group' and payload.get('request_addressed') is not True)):
            return ToolOutcome({'error':'invalid_media_delivery'},'blocked')
        payload = {**payload, 'arguments':arguments, 'delivery_mode':mode, 'delivery_text':caption}
        plan=local_fallback_plan.get()
        if plan is not None:
            if mode!='auto' or any(plan.get(k)!=payload.get(k) for k in ('scope','actor_id','source_id')):
                return ToolOutcome({'error':'media_fallback_out_of_scope'},'blocked')
            payload={**payload,'fallback_plan':{k:plan[k] for k in ('key','wave','minimum_count','media_types')}}
        identity = sha256(json.dumps([payload['operation'], arguments],
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        result = self.enqueue({**payload, 'source_fingerprints': self.source_fingerprints(payload['sources']),
            'task_id': 'media-' + uuid4().hex},
            parent_turn=str(parent_turn) + ':' + identity, before_commit=before_commit)
        if result['status'] == 'queued':
            with session_scope(self.engine) as session:
                # Reusing an existing job cannot change its delivery contract.
                mode = session.get(Job,result['job_id']).payload_json.get('delivery_mode','review')
            if continuation and result['execution_status'] in {'completed', 'failed', 'blocked', 'uncertain', 'cancelled'}:
                return ToolOutcome({'error': 'media_task_already_settled', 'job_id': result['job_id'],
                    'previous_status': result['execution_status'], 'replay_allowed': False,
                    'guidance': '原任务已结算，查询已有结果；缺份数时选择另一个已核验来源，不重复提交。'}, 'blocked')
            result['delivery'] = ('automatic_delivery_to_original_request' if mode=='auto'
                else 'background_media_result_returns_to_main')
            if mode=='auto':
                result['guidance'] = '程序会按原请求自动交付；可简短告知排队，不手动重复发送或重新提交。'
        return ToolOutcome(result, 'blocked' if result['status'] == 'blocked' else 'ok')


    def status_tool(self, *, scope_id, actor_id, guard):
        async def status(args):
            guard.require_valid()
            with session_scope(self.engine) as session:
                row = session.get(Job, args['job_id'])
                if not (row and row.job_type == self.job_type and row.job_key.startswith(self.prefix(scope_id))
                        and row.payload_json.get('actor_id') == actor_id):
                    return ToolOutcome({'error': 'media_task_out_of_scope'}, 'blocked')
                payload, state, reason = dict(row.payload_json), row.status, row.last_error_code
            check = ResearchJobGuard(self, args['job_id'], payload, 'running')
            if (not self.authorized(payload) or not check.valid_source(payload['source_id'], author=True)
                    or self.clock().timestamp() - payload['created_at'] > 7 * 86400
                    or not check.valid_dependencies(check.snapshot())):
                return ToolOutcome({'error': 'media_task_source_revoked'}, 'blocked')
            guard.require_valid()
            return ToolOutcome({'job_id': args['job_id'], 'status': state, 'reason': reason,
                'result': payload.get('result'), 'delivery': 'only_platform_receipt_confirms_send',
                'replay_allowed': False})
        return AgentTool('media_task_status', '查询当前提问者在本会话的媒体后台任务状态和实际产物；排队/下载/生图不代表QQ已发送，不重新执行任务。',
            {'type': 'object', 'properties': {'job_id': {'type': 'integer', 'minimum': 1, 'maximum': 2**63-1}},
             'required': ['job_id'], 'additionalProperties': False}, status)


def merge_media_results(results, payload):
    """Retain ready handles, identities and failures across one original request."""
    data = {'media_artifact_ids': [], 'artifact_ids': [], 'video_artifact_ids': [],
        'media': [], 'failures': [], 'jobs': [], 'quality_fallbacks': [], 'video_quality_fallbacks': []}
    seen = set()
    for identifier, result in results:
        item = result.get('data', {})
        data['jobs'].append({'job_id': identifier, 'status': 'completed'})
        if result.get('status') != 'ok':
            data['failures'].append({'job_id': identifier, 'error': item.get('error', 'media_download_failed')})
            continue
        for key in ('media_artifact_ids', 'artifact_ids', 'video_artifact_ids'):
            values = item.get(key, [])
            if key == 'media_artifact_ids' and not values:
                values = item.get('artifact_ids', []) + ([item['video_artifact_id']] if item.get('video_artifact_id') else [])
            if key == 'video_artifact_ids' and item.get('video_artifact_id'):values = [*values, item['video_artifact_id']]
            data[key].extend(i for i in values if i not in data[key])
        for medium in item.get('media', []):
            if medium.get('artifact_id') not in seen:
                data['media'].append(medium);seen.add(medium.get('artifact_id'))
        for key in ('failures', 'quality_fallbacks', 'video_quality_fallbacks'):data[key].extend(item.get(key, []))
    data['downloaded_count'] = len(data['media_artifact_ids'])
    data['requested_count'] = data['downloaded_count'] + len(data['failures'])
    data['all_requested_downloaded'] = not data['failures']
    data['request_aggregated'] = True
    data['total_bytes'] = sum(m.get('bytes', 0) for m in data['media'])
    if len(results) == 1:data = {**results[0][1].get('data', {}), **data}
    data['guidance'] = ('同一原请求的下载结果已汇总；失败不撤销成功文件。仍缺数量时可按原主题、日期与类型选择新来源有限续补，'
        '原任务不可重复执行。新下载排队后等待下一次汇总，最终合并发送已核验成果；真实未知结果禁止重放。')
    return media_result(ToolOutcome(data, 'ok' if data['media_artifact_ids'] else 'failed'), payload)


def media_result(outcome, payload):
    """Return handles and a bounded observation, never transport bytes or logs."""
    data = dict(outcome.data)
    if len(json.dumps(data, ensure_ascii=False)) > 12000:
        # Resource descriptions may be long; handles are never replaced by an error.
        keep = {'artifact_id','artifact_ids','media_artifact_ids','video_artifact_id',
            'video_artifact_ids','requested_count','downloaded_count','all_requested_downloaded',
            'source_url','post_id','total_bytes','image_roles','quality_fallbacks','video_quality_fallbacks',
            'failure_detail','error','jobs','request_aggregated'}
        compact = {key:value for key,value in data.items() if key in keep}
        compact['media'] = [{key:value for key,value in item.items() if key in {
            'artifact_id','logical_media_id','source_ref','kind','index','status','bytes','image_size','sha256','quality','fallback_reason'}}
            for item in data.get('media',[]) if isinstance(item,dict)]
        from app.core.media_source import media_source_key
        for item in compact['media']:
            if item.get('source_ref'):item['source_ref']=media_source_key(item['source_ref'])
        compact['failures'] = [{key:str(value)[:160] for key,value in item.items()
            if key in {'index','image_id','error'}} for item in data.get('failures',[]) if isinstance(item,dict)]
        for original, failure in zip((item for item in data.get('failures',[]) if isinstance(item,dict)), compact['failures']):
            if isinstance(original.get('details'),dict):
                failure['details']={k:v for k,v in original['details'].items() if k in {
                    'stage','variant_count','byte_limit','duration_limit_seconds','rejected_by_reason'}}
        compact.update(result_compacted=True, guidance='描述已压缩，保留真实成果句柄；失败和部分完成须如实说明。')
        data = compact
    # Downloads return handles; explicit read_artifact is the visual inspection
    # step. Generations retain their requested preview, not unrelated web pixels.
    preview_ids = []
    if payload['operation'] in {'generate_image', 'generate_private_image'}:
        preview_ids = data.get('artifact_ids', ())[:2]
        if data.get('artifact_id'):preview_ids = [data['artifact_id']]
    return {'kind': 'media_result', 'operation': payload['operation'], 'task_id': payload['task_id'],
        'status': outcome.status, 'data': data, 'preview_artifact_ids': list(preview_ids),
        'guidance': '这是后台真实执行结果，下载回传不读取像素；主题筛选默认依据原帖正文、日期、类型。只有文字有歧义时才少量查看静态图片，不预览视频或动图。直接下载请求可复用auto交付。失败/未知不声称成功，不重复提交。'}


async def execute_web_media(payload, guard, *, engine, settings, web_client, runner):
    from app.agent.shared_links import build_share_access, wrap_share_web_tools
    from app.agent.web_images import build_web_image_tools
    guard.require_valid()
    group_id = payload['scope_id'] if payload['surface'] == 'group' else None
    store = media_store(payload, guard, engine=engine, settings=settings)
    from app.agent.request_budget import worker_budget
    budget=worker_budget(payload,guard,settings)
    tools = build_web_image_tools(web_client, runner=runner, guard=guard, store=store, settings=settings,
        prepared_candidates=payload.get('candidates', {}), transfer_budget=budget)
    access = build_share_access(SimpleNamespace(engine=engine), group_id=group_id,
        trigger_source_ids=(payload['source_id'],), bot_qq=payload['account_id'],
        addressed_source_ids=(payload['source_id'],) if payload.get('request_addressed') is True else (),
        **({'private_user_id': payload['actor_id']} if group_id is None else {}))
    tools = wrap_share_web_tools(tools, access, observed_source_ids=guard.source_message_ids)
    tool = next(item for item in tools if item.name == payload['operation'])
    if not validate_arguments(payload['arguments'], tool.parameters):
        return media_result(ToolOutcome({'error': 'invalid_media_arguments'}, 'blocked'), payload)
    from app.core.request_cost import request_cost_scope
    with request_cost_scope(budget):
        outcome = await tool.handler(payload['arguments'])
    guard.require_valid()
    return media_result(outcome, payload)


def media_store(payload, guard, *, engine, settings):
    from app.agent.artifacts import AutonomousArtifactStore
    return AutonomousArtifactStore(settings.data_dir / 'autonomous_artifacts', account_id=payload['account_id'],
        scope=payload['scope'], engine=engine, source_message_ids=lambda: guard.source_message_ids,
        context_snapshot_factory=(lambda: payload['context_snapshot']) if payload.get('context_snapshot') else None)


def media_previews(result, store, guard, settings):
    from app.core.downloaded_media import DownloadedMediaCache
    from app.core.message_content import ImageAttachment
    previews = []
    for identifier in result.get('preview_artifact_ids', ())[:2]:
        guard.require_valid()
        cache = DownloadedMediaCache.for_store(store, settings)
        cache.pin((identifier,))
        path = store.get(identifier)
        if path is not None and path.suffix in {'.png', '.jpg', '.webp', '.gif'}:
            previews.append(ImageAttachment(url='', local_path=str(path), reference_subject='后台真实媒体产物 ' + identifier))
    guard.require_valid()
    return previews


async def deliver_download(payload, guard, *, engine, settings, runner, send, clock, conversation_key):
    """Deliver only an explicitly queued download, through the existing ledger."""
    from app.agent.media_delivery import resolve_media_bundle,pack_media_files,artifact_delivery_identities
    from app.agent.runtime_trace import AgentTrace, current_trace_context
    from app.agent.tool_dispatch import ToolDispatchBoundary
    guard.require_valid()
    if payload.get('delivery_mode')!='auto' or payload['operation'] not in MEDIA_OPERATIONS-{'generate_image'}:
        raise ValueError('automatic_media_delivery_not_authorized')
    store = media_store(payload,guard,engine=engine,settings=settings)
    result = payload['result']
    data = result.get('data',{})
    identifiers = data.get('media_artifact_ids') or ([data['video_artifact_id']]
        if data.get('video_artifact_id') else data.get('artifact_ids',[]))
    files = ()
    text = payload.get('delivery_text') or '下载完成，媒体在这里。'
    download_ok = result.get('status')=='ok' and bool(identifiers)
    if not download_ok and payload.get('fallback_plan'):
        # A failed candidate does not produce a warning between fallback
        # waves. Only an exhausted request with no delivered media needs one.
        with session_scope(engine) as session:
            siblings=list(session.scalars(select(Job).where(Job.job_type=='agent_media_operation',
                Job.job_key.startswith(guard.queue.prefix(payload['scope_id'])),
                Job.payload_json['fallback_plan']['key'].as_string()==payload['fallback_plan']['key'],
                Job.payload_json['scope'].as_string()==payload['scope'],
                Job.payload_json['actor_id'].as_integer()==payload['actor_id'],
                Job.payload_json['source_id'].as_integer()==payload['source_id'])))
            if any(j.status in {'uncertain','cancelled'} for j in siblings):return 'blocked'
            others=[j for j in siblings if j.id!=guard.job_id]
            if any(j.status in {'pending','queued','running','result_ready','delivering','completed'} for j in others):
                return 'failed'
    if download_ok:
        def before_read(identifier):
            guard.require_valid()
            provenance = store.provenance(identifier)
            if not guard.valid_dependencies({'sources':list(provenance),'memory':{}}):
                raise ValueError('media_delivery_source_revoked')
            return store.get(identifier)
        files = await runner.run(resolve_media_bundle,store,settings,identifiers,before_read=before_read)
        if data.get('all_requested_downloaded') is False:
            text = f"成功下载{len(files)}项，另有{len(data.get('failures',[]))}项未完成；这里只发送已取得的媒体。"
        if data.get('quality_fallbacks'):
            text += f" 其中{len(data['quality_fallbacks'])}张图片原图不可用，已降为large尺寸（非原图）。"
        if data.get('video_quality_fallbacks'):
            text += f" 其中{len(data['video_quality_fallbacks'])}段视频采用较低清晰度的原生版本。"
    else:
        text = '这次媒体下载未完成，暂时无法发送。'
    from app.agent.request_budget import worker_budget
    budget=worker_budget(payload,guard,settings)
    boundary = ToolDispatchBoundary(engine=engine,account_id=payload['account_id'],scope=payload['scope'],
        guard=guard,max_daily=None if budget and budget.unmetered else 500,cooldown_seconds=0,clock=clock,
        delivery_source_id=payload['source_id'],
        media_identity=lambda i:artifact_delivery_identities(store,i),
        interactive_group_send=payload['surface']=='group')
    operation = 'send_'+payload['surface']
    async def handler(args):
        selected=tuple(files[identifiers.index(i)] for i in args.get('media',[]))
        return await send(boundary, selected, args['text'])
    tool = AgentTool(operation,'交付原请求已授权的下载结果',{},handler,effect='write')
    observer = AgentTrace(settings.data_dir/'agent_monitor' if getattr(settings,'agent_monitor_enabled',False) else None,
        context={**current_trace_context(), **guard.trace_trigger,
            'execution_kind':'tool_delivery', 'context_kind':'media_task','surface':payload['surface'],
            'context_scope':payload['scope'],'scope_id':payload['scope_id'],'actor_id':payload['actor_id'],
            'source_id':payload['source_id'],'trigger_kind':'media_delivery','request_preview':'下载结果直接交付'},
        session_key=str(conversation_key)+':media-delivery',
        conversation_key=f"job:{payload['account_id']}:{guard.job_id}")
    try:
        async def dispatch(args,call_id):
            observer.event('tool.started',{'name':operation,'call_id':call_id,'effect':'write','arguments':args})
            outcome = await boundary.invoke(tool,args,call_id)
            observer.event('tool.completed',{'name':operation,'call_id':call_id,'effect':'write',
                'status':outcome.status,'data':outcome.data,'receipt_id':outcome.receipt_id})
            return outcome
        packets=await runner.run(pack_media_files,identifiers,files) if files else [[]]
        for index,packet in enumerate(packets):
            guard.require_valid()
            if budget:budget.spend('writes',finalization=True)
            outcome=await dispatch({'text':text+ (f'（第{index+1}/{len(packets)}批）' if len(packets)>1 else ''),
                'media':packet},'auto-media-'+str(index))
            if outcome.status!='sent':break
        if files and outcome.status=='failed' and outcome.data.get('error') in {
                'qq_send_rejected','qq_video_requires_standalone_message'}:
            # A definite rejection permits one notice, never another upload.
            await dispatch({'text':'媒体已下载，但本次发送被拒绝，暂未送达。','media':[]},'failure-notice')
        observer.finish(outcome.status,'automatic_media_delivery')
        return ('completed' if download_ok else 'failed') if outcome.status=='sent' else (
            outcome.status if outcome.status in {'uncertain','blocked'} else 'failed')
    finally:
        observer.close()
