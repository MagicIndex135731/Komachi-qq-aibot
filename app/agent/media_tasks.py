"""Durable deterministic media work. No child model or native chat session."""
from __future__ import annotations

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

MEDIA_OPERATIONS = frozenset({'generate_image', 'download_web_images', 'download_web_video'})


class MediaJobGuard(ResearchJobGuard):
    def __init__(self, *args):
        super().__init__(*args)
        self.action_id = 'media-result:' + str(self.job_id)
        self.trace_trigger.update(trigger_kind='media_result', trigger_reason_code='media_result_ready',
            trigger_reason='媒体后台执行已结束，将真实产物和状态回传原主会话。')


class MediaTaskQueue(ResearchDelegationQueue):
    job_type = 'agent_media_operation'
    # Losing a worker must not silently buy the same image a second time.
    max_execution_attempts = 1
    interrupted_execution_status = 'uncertain'
    exhausted_execution_status = 'uncertain'
    guard_type = MediaJobGuard
    allow_confirmed_bot_dependencies = True

    def __init__(self, engine, *, authorized, available=lambda: True, **kwargs):
        self.available = available
        super().__init__(engine, authorized=lambda payload: authorized(payload)
            and payload.get('source_fingerprints') == self.source_fingerprints(payload['sources']), **kwargs)

    def claim(self):
        # A transient disconnected gateway is not revoked request authority.
        # Do not spend the single execution attempt during process startup.
        return super().claim() if self.available() else None

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

    def submit(self, payload, *, parent_turn, before_commit):
        if payload.get('operation') not in MEDIA_OPERATIONS:
            return ToolOutcome({'error': 'media_operation_not_allowed'}, 'blocked')
        identity = sha256(json.dumps([payload['operation'], payload['arguments']],
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        result = self.enqueue({**payload, 'source_fingerprints': self.source_fingerprints(payload['sources']),
            'task_id': 'media-' + uuid4().hex},
            parent_turn=str(parent_turn) + ':' + identity, before_commit=before_commit)
        if result['status'] == 'queued':
            result['delivery'] = 'background_media_result_returns_to_main'
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


def media_result(outcome, payload):
    """Return handles and a bounded observation, never transport bytes or logs."""
    data = dict(outcome.data)
    if len(json.dumps(data, ensure_ascii=False)) > 12000:
        outcome = ToolOutcome({'error': 'media_result_too_large'}, 'failed')
        data = outcome.data
    preview_ids = data.get('artifact_ids', ())[:2]
    if data.get('artifact_id'):
        preview_ids = [data['artifact_id']]
    return {'kind': 'media_result', 'operation': payload['operation'], 'task_id': payload['task_id'],
        'status': outcome.status, 'data': data, 'preview_artifact_ids': list(preview_ids),
        'guidance': '这是后台真实执行结果；核对产物后用发送工具交付。失败或不确定不得称成功，不自动重做。'}


async def execute_web_media(payload, guard, *, engine, settings, web_client, runner):
    from app.agent.shared_links import build_share_access, wrap_share_web_tools
    from app.agent.web_images import build_web_image_tools
    guard.require_valid()
    group_id = payload['scope_id'] if payload['surface'] == 'group' else None
    store = media_store(payload, guard, engine=engine, settings=settings)
    tools = build_web_image_tools(web_client, runner=runner, guard=guard, store=store, settings=settings,
        prepared_candidates=payload.get('candidates', {}))
    access = build_share_access(SimpleNamespace(engine=engine), group_id=group_id,
        trigger_source_ids=(payload['source_id'],), bot_qq=payload['account_id'],
        addressed_source_ids=(payload['source_id'],) if payload.get('request_addressed') is True else (),
        **({'private_user_id': payload['actor_id']} if group_id is None else {}))
    tools = wrap_share_web_tools(tools, access, observed_source_ids=guard.source_message_ids)
    tool = next(item for item in tools if item.name == payload['operation'])
    if not validate_arguments(payload['arguments'], tool.parameters):
        return media_result(ToolOutcome({'error': 'invalid_media_arguments'}, 'blocked'), payload)
    outcome = await tool.handler(payload['arguments'])
    guard.require_valid()
    return media_result(outcome, payload)


def media_store(payload, guard, *, engine, settings):
    from app.agent.artifacts import AutonomousArtifactStore
    return AutonomousArtifactStore(settings.data_dir / 'autonomous_artifacts', account_id=payload['account_id'],
        scope=payload['scope'], engine=engine, source_message_ids=lambda: guard.source_message_ids)


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
