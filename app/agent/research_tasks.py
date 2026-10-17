"""Requester-scoped working checkpoints and bounded external evidence, never facts."""
from __future__ import annotations

import hashlib
import json
from copy import copy
from pathlib import Path
import re
import time
from functools import wraps
from threading import RLock, local
from uuid import uuid4

from app.agent.tool_agent import AgentTool, ToolOutcome

_LOCKS = {}
_HELD = local()


def serialized_checkpoint(method):
    """Serialize read/modify/replace across model turns and queue workers."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if not self.formal and method.__name__ in {'observe_tool', 'record_delivery'}:
            return method(self, *args, **kwargs)
        key = str(self.root)
        with _LOCKS.setdefault(key, RLock()):
            held = getattr(_HELD, 'roots', set())
            if key in held:
                return method(self, *args, **kwargs)
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / '.checkpoint.lock').open('a+b') as lock:
                import os
                if os.name == 'nt':
                    import msvcrt
                    if lock.tell() == 0:
                        lock.write(b'0'); lock.flush()
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock, fcntl.LOCK_EX)
                _HELD.roots = held | {key}
                try:
                    return method(self, *args, **kwargs)
                finally:
                    _HELD.roots = held
                    if os.name == 'nt':
                        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(lock, fcntl.LOCK_UN)
    return wrapped


class ResearchTaskWorkspace:
    def __init__(self, root: Path, *, scope: str, actor_id: int, source_id: int,
                 request: str, valid_source, clock=time.time, dependency_snapshot=None, valid_dependencies=None,
                 task_content_budget=96000, task_key: str = '', cancel_background=None):
        if not scope or not isinstance(actor_id, int) or not isinstance(source_id, int):
            raise ValueError('invalid_task_scope')
        self.scope, self.actor_id, self.source_id = scope, actor_id, source_id
        self.request, self.valid_source, self.clock = request[:4000], valid_source, clock
        self.dependency_snapshot, self.valid_dependencies = dependency_snapshot, valid_dependencies
        self.cancel_background = cancel_background
        self.task_content_budget = task_content_budget
        self.archived_read_chars = self.web_result_chars = 0
        owner = hashlib.sha256(f'{scope}:{actor_id}'.encode()).hexdigest()
        self.root = Path(root).resolve() / owner
        # Routing is deliberately decided by the current main conversation.
        # A keyword in quoted or ordinary text must never fork its context.
        # `begin_research_task` or an explicit delegated child activates the
        # bounded workspace after the model has made that decision.
        self.formal = False
        self.request_control = {}
        identity = f'{owner}:{source_id}' + (f':{task_key}' if task_key else '')
        self.task_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
        self.context_scope = scope

    def activate(self):
        """Enable task storage; callers select the main session before activation."""
        self.formal = True
        self.context_scope = f'research:{self.scope}:{self.actor_id}:{self.task_id}'
        return self

    def resume(self, task_id):
        """Select an explicitly named checkpoint in this requester scope."""
        if not isinstance(task_id, str) or not re.fullmatch(r'[a-f0-9]{24}', task_id):
            raise ValueError('invalid_task_id')
        self._check()
        state = self._load(self.root / task_id / 'checkpoint.json')
        if not state:
            raise ValueError('task_checkpoint_unavailable')
        self._check(state)
        self.task_id = task_id
        return self.activate()

    def _check(self, state=None):
        if not self.valid_source(self.source_id):
            raise ValueError('task_source_unavailable')
        if state is not None and (state.get('scope') != self.scope or state.get('actor_id') != self.actor_id
                or self.clock() - state.get('updated_at', 0) > 7 * 86400
                or not self.valid_source(state.get('source_id'))):
            raise ValueError('task_source_unavailable')
        if state is not None and (state.get('dependencies') or self.valid_dependencies is not None):
            metadata = state.get('dependencies') or {}
            sources = metadata.get('sources', []) if isinstance(metadata, dict) else None
            if (not isinstance(sources, list) or any(type(item) is not int or item <= 0 for item in sources)
                    or self.valid_dependencies is None):
                raise ValueError('task_memory_dependency_unavailable')
            # Source checks above are read-only. Admit the complete provenance
            # only after the caller validates its dependencies and budget.
            metadata = {**metadata, 'sources': list(dict.fromkeys(
                [*sources, self.source_id, state['source_id']]))}
            if not self.valid_dependencies(metadata):
                raise ValueError('task_memory_dependency_unavailable')

    def _file(self, name='checkpoint.json'):
        return self.root / self.task_id / name

    def _load(self, path):
        if not path.is_file():
            return None
        if path.stat().st_size > 256 * 1024:
            raise ValueError('task_record_too_large')
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('task_record_invalid')
        return data

    def _write(self, path, data):
        self._check(data)
        encoded = json.dumps(data, ensure_ascii=False, sort_keys=True)
        if len(encoded.encode('utf-8')) > 256 * 1024:
            raise ValueError('task_record_too_large')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
        try:
            temporary.write_text(encoded, encoding='utf-8')
            self._check(data)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _active(self):
        try:
            data = self._load(self.root / 'active.json')
            if data:
                self._check(data)
            return data
        except (ValueError, OSError):
            return None

    def summary(self):
        self._check()
        state = self._load(self._file())
        if state:
            self._check(state)
            return state
        return {'task_id': self.task_id, 'scope': self.scope, 'actor_id': self.actor_id,
                'source_id': self.source_id, 'goal': self.request, 'progress': '',
                'findings': [], 'next_step': '', 'status': 'active',
                'updated_at': self.clock(), 'evidence': []}

    def recent_summary(self):
        state = self._active()
        return ({k: state[k] for k in ('task_id', 'goal', 'progress', 'next_step', 'status') if k in state}
                if state else {})

    def prompt_summary(self):
        state = self.summary() if self.formal else self.recent_summary()
        return {k: state[k] for k in ('task_id', 'goal', 'deliverable', 'progress', 'remaining', 'next_step', 'status') if k in state}

    @serialized_checkpoint
    def begin(self, goal='', deliverable='', remaining=None, delivery='none'):
        if delivery not in {'none', 'text', 'media'} or remaining is not None and len(remaining) > 24:
            raise ValueError('invalid_task_checkpoint')
        self.activate()
        state = self.summary()
        if goal:
            state['goal'] = goal[:2000]
        if deliverable:
            state['deliverable'] = deliverable[:1000]
        if remaining is not None:
            state['remaining'] = [str(item)[:300] for item in remaining]
        if delivery != 'none':
            state['delivery_requirement'] = delivery
        if self.dependency_snapshot:
            state['dependencies'] = self.dependency_snapshot()
        state['updated_at'] = self.clock()
        self._write(self._file(), state)
        self._write(self.root / 'active.json', state)
        return state

    @serialized_checkpoint
    def update(self, *, progress, next_step, findings, status='active', remaining=None):
        if status not in {'active', 'waiting', 'partial', 'completed', 'blocked', 'cancelled'} or len(findings) > 12:
            raise ValueError('invalid_task_checkpoint')
        state = self.begin()
        if status == 'cancelled' and self.cancel_background:
            self.cancel_background(self.task_id)
            state['background_jobs'] = {key: 'cancelled' if value in {'pending', 'queued', 'running',
                'result_ready', 'delivering'} else value for key, value in state.get('background_jobs', {}).items()}
        state.update(progress=progress[:2000], next_step=next_step[:1000],
                     findings=[str(value)[:1000] for value in findings], status=status, updated_at=self.clock())
        if remaining is not None:
            if len(remaining) > 24:
                raise ValueError('invalid_task_checkpoint')
            state['remaining'] = [str(item)[:300] for item in remaining]
        # A model declaration cannot override deterministic missing conditions.
        if status == 'completed':
            state.pop('completion_gaps', None)
            gaps = self.completion_gaps(state)
            if gaps:
                state.update(status='partial', completion_gaps=gaps)
            else:
                state['completion_basis'] = 'recorded_tool_results; semantic_coverage_model_reported'
        self._write(self._file(), state)
        self._write(self.root / 'active.json', state)
        return {k: state[k] for k in ('task_id', 'goal', 'progress', 'next_step', 'findings', 'status',
            'remaining', 'completion_gaps', 'completion_basis') if k in state}

    def completion_gaps(self, state=None):
        state = state or self.summary()
        receipts = state.get('execution_receipts', [])
        gaps = ['remaining_work'] if state.get('remaining') else []
        if any(value in {'pending', 'queued', 'running', 'result_ready', 'delivering'}
                for value in state.get('background_jobs', {}).values()):
            gaps.append('background_pending')
        if state.get('unknown_external_effect') or any(item.get('status') == 'uncertain'
                or item.get('job_status') == 'uncertain' for item in receipts):
            gaps.append('unknown_external_effect')
        delivery = state.get('delivery_requirement', 'none')
        sends = [item for item in receipts if item.get('status') == 'sent' and item.get('delivery')]
        if delivery != 'none' and not sends:
            gaps.append('delivery_unconfirmed')
        if delivery == 'media' and not any(item.get('media') for item in sends):
            gaps.append('media_delivery_unconfirmed')
        return gaps

    def execution_status(self):
        if not self.formal:
            return {}
        state = self.summary()
        return {**{k: state[k] for k in ('task_id', 'status', 'remaining', 'next_step') if k in state},
            'completion_gaps': self.completion_gaps(state),
            'coverage_verification': 'model_reported_not_independent_semantic_validation'}

    @serialized_checkpoint
    def claim_media_fallback(self):
        """One local recovery per source, shared across turns and workers."""
        self._check()
        path=self._file('media-fallback.json')
        if self._load(path):return False
        state={'scope':self.scope,'actor_id':self.actor_id,'source_id':self.source_id,
            'task_id':self.task_id,'updated_at':self.clock()}
        if self.dependency_snapshot:state['dependencies']=self.dependency_snapshot()
        self._check(state)
        self._write(path,state)
        return True

    @serialized_checkpoint
    def save_request_control(self, control):
        """Lightweight evidence/goal record; does not activate research budgets."""
        data={'scope':self.scope,'actor_id':self.actor_id,'source_id':self.source_id,
            'task_id':self.task_id,'updated_at':self.clock(),'control':control}
        if self.dependency_snapshot:data['dependencies']=self.dependency_snapshot()
        # Only the latest lightweight request is retained in this requester
        # scope. Formal checkpoints and queued jobs own durable continuation;
        # ordinary chat must not create an ever-growing directory per message.
        self._write(self.root/'request-goal.json',data)
        self.request_control = dict(control)

    @serialized_checkpoint
    def media_selection_for_source(self):
        """Resume source-checked candidates and goal, never old artifact authority."""
        data=self._load(self.root/'request-goal.json')
        if not data or data.get('source_id')!=self.source_id or data.get('task_id')!=self.task_id:
            return {}
        self._check(data)
        value=data.get('control',{})
        return {k:value[k] for k in ('minimum_media_count','media_selection','candidate_evidence','constraint_evidence','last_delivery','delivered_artifacts','delivered_content_groups','delivered_media_types') if k in value}

    @serialized_checkpoint
    def save_execution_experience(self, version, experience):
        """Only callers with an actual verified delivery may record a method."""
        self._check()
        rows=self.execution_experiences(version,experience['delivery'])
        entry={'scope':self.scope,'actor_id':self.actor_id,'source_id':self.source_id,
            'updated_at':self.clock(),'version':version,**experience}
        if self.dependency_snapshot:entry['dependencies']=self.dependency_snapshot()
        signature=(entry['delivery'],entry['requirements'],entry['tools'])
        rows=[r for r in rows if (r['delivery'],r['requirements'],r['tools'])!=signature]
        rows.append(entry)
        # Per requester/scope; no chat text, artifact payloads or inferred facts.
        self._write(self.root/'execution-experiences.json',{'scope':self.scope,
            'actor_id':self.actor_id,'source_id':self.source_id,'updated_at':self.clock(),
            'experiences':rows[-12:]})

    def execution_experiences(self, version, delivery):
        self._check()
        data=self._load(self.root/'execution-experiences.json') or {}
        result=[]
        for item in data.get('experiences',[])[-12:]:
            if not isinstance(item,dict) or item.get('version')!=version or item.get('delivery')!=delivery:
                continue
            try:self._check(item)
            except (ValueError,OSError):continue
            if item.get('verified_basis')=='observed_tools_and_confirmed_request_delivery':result.append(item)
        return result[-3:]

    @serialized_checkpoint
    def observe_tool(self, name, args, outcome, call_id):
        if not self.formal or name in {'begin_research_task', 'update_research_task', 'read_research_task',
                'resume_research_task', 'read_research_evidence', 'read_research_resources'}:
            return
        state = self.begin()
        receipts = state.get('execution_receipts', [])
        entries = outcome.data.get('jobs', []) if name == 'queue_media_batch' else [outcome.data]
        jobs = state.get('background_jobs', {})
        for index, data in enumerate(entries):
            entry = {'call_id': call_id + ':' + str(index), 'tool': name, 'status': outcome.status}
            if type(data.get('job_id')) is int:
                entry.update(job_id=data['job_id'], job_status=data.get('execution_status', data.get('status')))
                # Status reads replace the queue observation for this job.
                receipts = [item for item in receipts if item.get('job_id') != entry['job_id']]
                if len(jobs) >= 512 and str(entry['job_id']) not in jobs:
                    state['unknown_external_effect'] = True  # bounded fail-closed checkpoint
                else:
                    jobs[str(entry['job_id'])] = entry['job_status']
            if entry.get('status') == 'uncertain' or entry.get('job_status') == 'uncertain':
                state['unknown_external_effect'] = True
            if name in {'send_group', 'send_private'}:
                entry.update(delivery=True, media=bool(args.get('artifact_ids') or
                    args.get('video_artifact_id') or args.get('media_artifact_ids')),
                    receipt_id=outcome.receipt_id)
            receipts = [item for item in receipts if item.get('call_id') != entry['call_id']]
            receipts.append(entry)
        state['execution_receipts'] = receipts[-128:]
        state['background_jobs'] = jobs
        if jobs and not state.get('remaining') and state['status'] == 'active':
            state['status'] = 'waiting'
        self._write(self._file(), state)
        self._write(self.root / 'active.json', state)

    @serialized_checkpoint
    def record_job_result(self, job_id, status):
        """Persist a queue terminal state without waking a model or changing
        another request's active checkpoint. This is transport/work coverage,
        never a claim that the semantic goal has been independently verified.
        """
        state = self.summary()
        jobs = state.get('background_jobs', {})
        jobs[str(job_id)] = status
        state['background_jobs'] = jobs
        if status == 'uncertain':
            state['unknown_external_effect'] = True
        if state.get('status') == 'waiting' and not any(value in {
                'pending', 'queued', 'running', 'result_ready', 'delivering'} for value in jobs.values()):
            state['status'] = 'partial' if any(value != 'completed' for value in jobs.values()) else 'active'
            state['next_step'] = '后台处理结束；按实际交付与覆盖范围核验整体目标'
        state['updated_at'] = self.clock()
        self._write(self._file(), state)
        active = self._active()
        if active and active.get('task_id') == self.task_id:
            self._write(self.root / 'active.json', state)

    def _select_read_task(self, task_id=None):
        """Read a named requester-local checkpoint without changing this goal."""
        self._check()
        if task_id is None or task_id == self.task_id:
            return self
        selected = copy(self)
        selected.resume(task_id)
        return selected

    def resource_page(self, task_id=None, *, offset=0, limit=5):
        if type(offset) is not int or not 0 <= offset <= 10000 or type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError('invalid_resource_page')
        from app.core.url_policy import public_web_url
        selected = self._select_read_task(task_id)
        state = selected.summary()
        selected._check(state)
        resources = {}
        for entry in state.get('evidence', []):
            for value in entry.get('sources', []):
                try:
                    url = public_web_url(value)
                except (TypeError, ValueError):
                    continue
                record = resources.setdefault(url, {'url': url, 'kind': 'web_page', 'evidence_ids': []})
                if entry['evidence_id'] not in record['evidence_ids']:
                    record['evidence_ids'].append(entry['evidence_id'])
            for item in entry.get('resources', []):
                url = item.get('source_url')
                if url in resources:
                    resources[url].update({k: v for k, v in item.items() if k != 'source_url'})
                    resources[url]['kind'] = 'inspected_post'
        values = list(resources.values())
        return {'task_id': selected.task_id, 'resources': values[offset:offset+limit], 'total': len(values),
            'next_offset': offset+limit if offset+limit < len(values) else None,
            'trust': 'untrusted_external_sources', 'coverage': 'discovered_sources_only_not_exhaustive',
            'guidance': '使用url交给接受URL的工具；evidence_id用于原文读取，不能当URL或artifact_id。'}

    @serialized_checkpoint
    def record_delivery(self, text, status):
        """A confirmed/ambiguous transport outcome never proves the audit complete."""
        if not self.formal or status not in {'sent', 'uncertain'}:
            return
        state = self.begin()
        state['last_delivery'] = {'text': text[:12000], 'status': status}
        if not state.get('progress'):
            state['progress'] = '上次交付内容（完成范围须继续核验）：' + text[:1600]
        self._write(self._file(), state)
        self._write(self.root / 'active.json', state)

    def evidence_reference(self, data):
        content = re.sub(r'web_[a-f0-9]{12}_\d+', 'source_ref', str(data.get('content', '')))
        sources = [source.get('source', '') for source in data.get('sources', []) if isinstance(source, dict)]
        return hashlib.sha256(json.dumps([content, sources, data.get('resources', [])], ensure_ascii=False,
            sort_keys=True).encode()).hexdigest()[:24]

    @serialized_checkpoint
    def save_evidence(self, data):
        self._check()
        ref = self.evidence_reference(data)
        state = self.summary()
        if self.dependency_snapshot:
            state['dependencies'] = self.dependency_snapshot()
            self._check(state)
        state['updated_at'] = self.clock()
        entries = state.get('evidence', [])
        if ref not in {item['evidence_id'] for item in entries}:
            if len(entries) >= 128:
                raise ValueError('task_evidence_budget_exhausted')
            entries.append({'evidence_id': ref, 'sources': [s.get('source') for s in data.get('sources', [])
                            if isinstance(s, dict)], 'content_chars': len(data.get('content', '')),
                'resources': [{k: item[k] for k in ('source_url', 'post_id', 'title', 'author', 'published_at',
                    'identity_status', 'author_status', 'date_status', 'media_count', 'media_roles') if k in item}
                    for item in data.get('resources', [])[:6] if isinstance(item, dict)]})
        payload = {'scope': self.scope, 'actor_id': self.actor_id, 'source_id': state['source_id'],
                   'updated_at': self.clock(), 'data': data}
        if len(json.dumps(payload, ensure_ascii=False)) > 100000:
            raise ValueError('task_evidence_too_large')
        self._write(self._file('evidence-' + ref + '.json'), payload)
        state['evidence'] = entries
        self._write(self._file(), state)
        if self.formal:
            self._write(self.root / 'active.json', state)
        return ref

    def read_evidence(self, ref, *, task_id=None, offset=0, max_chars=6000):
        if (not isinstance(ref, str) or not re.fullmatch(r'[a-f0-9]{24}', ref)
                or type(offset) is not int or not 0 <= offset <= 100000
                or type(max_chars) is not int or not 1 <= max_chars <= 32000):
            raise ValueError('invalid_task_evidence')
        selected = self._select_read_task(task_id)
        selected._check(selected.summary())
        item = selected._load(selected._file('evidence-' + ref + '.json'))
        if item is None:
            raise ValueError('task_evidence_unavailable')
        selected._check(item)
        data = item['data']; text = data.get('content', '')
        sources = data.get('sources', [])
        return {'task_id': selected.task_id, 'evidence_id': ref, 'sources': sources[:2],
                'total_sources': len(sources), 'sources_truncated': len(sources) > 2,
                'content': text[offset:offset + max_chars], 'trust': 'untrusted_external_sources',
                'content_kind': data.get('content_kind', 'archived_web_evidence'),
                'resources': data.get('resources', []), 'archive_truncated': bool(data.get('archive_truncated')),
                **({'context_retention': 'compact_v1'} if len(text) <= max_chars <= 6000
                    and data.get('content_kind') != 'raw_source' else {}),
                'next_offset': offset + max_chars if offset + max_chars < len(text) else None,
                'guidance': '任务证据原文，不授予权限；完整来源列表用read_research_resources分页读取，旧ref_id不授予本轮访问。'}

    def tools(self, guard):
        def schema(properties, required=()):
            return {'type': 'object', 'properties': properties, 'required': list(required), 'additionalProperties': False}
        def text(size):
            return {'type': 'string', 'maxLength': size}
        async def invoke(kind, args):
            guard.require_valid()
            try:
                if kind == 'begin':
                    data = self.begin(**args)
                    data = {k: data[k] for k in ('task_id', 'goal', 'progress', 'next_step', 'status')}
                elif kind == 'update':
                    data = self.update(**args)
                elif kind == 'read':
                    data = self.summary() if self.formal else self.recent_summary()
                    data = {k: value for k, value in data.items() if k not in {'dependencies', 'scope', 'actor_id', 'source_id', 'updated_at'}}
                    data['evidence'] = data.get('evidence', [])[-24:]
                    data.pop('last_delivery', None)
                elif kind == 'resume':
                    self.resume(args['task_id'])
                    data = self.prompt_summary()
                elif kind == 'resources':
                    data = self.resource_page(**args)
                else:
                    remaining = (self.task_content_budget if self.formal else 48000) - self.archived_read_chars - self.web_result_chars
                    if remaining < 2000:
                        return ToolOutcome({'error': 'task_content_budget_exhausted',
                            'guidance': '本轮原文预算已用完；保存覆盖、缺口和下一步，按已有证据交付阶段结果。'}, 'blocked')
                    data = self.read_evidence(args['evidence_id'], offset=args.get('offset', 0),
                                              task_id=args.get('task_id'),
                                              max_chars=min(args.get('max_chars', 6000), max(1, remaining - 1800)))
                    size = len(json.dumps(data, ensure_ascii=False))
                    if size > remaining:
                        return ToolOutcome({'error': 'task_content_budget_exhausted'}, 'blocked')
                    self.archived_read_chars += size
                guard.require_valid()
                return ToolOutcome(data)
            except (ValueError, OSError) as exc:
                code = str(exc) if isinstance(exc, ValueError) and str(exc) in {
                    'invalid_task_id', 'invalid_task_evidence', 'invalid_resource_page',
                    'task_checkpoint_unavailable', 'task_source_unavailable',
                    'task_memory_dependency_unavailable'} else 'task_evidence_unavailable'
                return ToolOutcome({'error': code, 'task_id': args.get('task_id', self.task_id),
                    'guidance': '证据按task_id与evidence_id定位；跨轮读取请同时传原结果的task_id，'
                        '不必切换当前目标。查无旧证据不代表网站或下载不可用；可按当前请求重新搜索/打开来源。'
                        '来源撤回、过期或依赖失效的旧内容不能恢复。'}, 'blocked')
        return (
            AgentTool('begin_research_task', '多步骤工作保存目标、交付要求和剩余项，可组合任何获准工具；资料任务扩容网页预算，不增加权限。普通闲聊和单步请求无需建任务',
                      schema({'goal': text(2000), 'deliverable': text(1000),
                          'remaining': {'type': 'array', 'items': text(300), 'maxItems': 24},
                          'delivery': {'type': 'string', 'enum': ['none', 'text', 'media']}}), lambda args: invoke('begin', args)),
            AgentTool('update_research_task', '保存目标进度、证据结论和下一步；不是长期事实。完成前如实标注未验范围',
                schema({'progress': text(2000), 'next_step': text(1000), 'findings': {'type': 'array',
                       'items': text(1000), 'maxItems': 12}, 'status': {'type': 'string',
                       'enum': ['active', 'waiting', 'partial', 'completed', 'blocked', 'cancelled']},
                       'remaining': {'type': 'array', 'items': text(300), 'maxItems': 24}}, ('progress', 'next_step', 'findings', 'status')),
                lambda args: invoke('update', args)),
            AgentTool('read_research_task', '读取本提问者的任务覆盖、进度和证据索引，不携带全部原文',
                      schema({}), lambda args: invoke('read', args)),
            AgentTool('resume_research_task', '主模型明确选择本提问者的旧任务；复验来源、依赖及有效期，不自动切换主会话',
                      schema({'task_id': text(24)}, ('task_id',)), lambda args: invoke('resume', args)),
            AgentTool('read_research_evidence', '按evidence_id分段重读网页证据，默认6000字符；跨轮同时传原结果task_id，只读同提问者旧任务而不切换当前目标，不能读取其他群或用户任务',
                schema({'evidence_id': text(24), 'task_id': text(24), 'offset': {'type': 'integer', 'minimum': 0, 'maximum': 100000},
                       'max_chars': {'type': 'integer', 'minimum': 1, 'maximum': 32000}}, ('evidence_id',)),
                lambda args: invoke('evidence', args)),
            AgentTool('read_research_resources', '分页读取当前提问者任务的全部已发现来源URL和证据ID；子任务摘要只给索引，用此工具续读再组合后续工具',
                schema({'task_id': text(24), 'offset': {'type': 'integer', 'minimum': 0, 'maximum': 10000},
                    'limit': {'type': 'integer', 'minimum': 1, 'maximum': 20}}), lambda args: invoke('resources', args)),
        )
