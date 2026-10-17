"""Shared, deterministic native-context selection and bounded patrol progress.

Local checkpoints own authority and recovery. Provider response IDs and cache
keys are transport metadata, never a source of platform permissions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from uuid import uuid4

from app.agent.context_policy import RequestContextPolicy
from app.agent.context_recovery import should_rotate_context


def context_kind(scope: str) -> str:
    if scope.startswith('background:group:'):
        return 'background_group'
    if scope.startswith('research:'):
        return 'research'
    if scope.startswith('owner:'):
        return 'owner_task'
    if scope.startswith('private:'):
        return 'private'
    if scope.startswith('group:'):
        return 'group'
    return 'public'


def context_policy(settings, *, background=False, ephemeral=False):
    policy = RequestContextPolicy.from_settings(settings)
    if ephemeral:
        return replace(policy, history_tokens=0, retain_tokens=0)
    history = int(getattr(settings, 'agent_background_history_tokens', 8000)
                  if background else getattr(settings, 'agent_context_history_tokens', 24000))
    return replace(policy, history_tokens=history, retain_tokens=min(policy.retain_tokens, history),
        soft_input_tokens=min(policy.soft_input_tokens or 32000, 32000) if background else policy.soft_input_tokens,
        target_input_tokens=min(policy.target_input_tokens or 20000, 20000) if background else policy.target_input_tokens,
        summary_tokens=min(policy.summary_tokens, 4096) if background else policy.summary_tokens)


@dataclass(frozen=True)
class NativeContext:
    session_id: str
    restore: bool
    metadata: dict


def select_native_context(scope: str, checkpoint: dict, *, version, history_present: bool,
                          history_bytes=0, restored_items=0, id_field='session_id', new_id=None) -> NativeContext:
    """Use the same reason to select an ID and explain it in the monitor."""
    make_id = new_id or (lambda: uuid4().hex)
    previous = checkpoint.get(id_field)
    reset = checkpoint.get('last_status') == 'context_reset'
    if reset and not previous:
        previous = checkpoint.get('runtime_id')
    current = previous or make_id()
    reason, rotate = 'same_native_session', False
    if reset:
        reason = 'context_reset'
    elif checkpoint.get('in_flight'):
        reason, rotate = 'interrupted_turn', True
    elif should_rotate_context(checkpoint):
        reason, rotate = 'previous_failed', True
    elif checkpoint.get('shared_native_dirty'):
        reason, rotate = 'parallel_completion', True
    elif not previous:
        reason = checkpoint.get('last_status') if checkpoint.get('last_status') in {
            'source_invalidated', 'memory_changed', 'persona_changed'} else 'first_turn'
    elif checkpoint.get('context_format_version') != version:
        reason, rotate = 'model_format_changed', True
    elif history_bytes > 8 * 1024 * 1024:
        reason, rotate = 'native_size_limit', True
    elif not history_present:
        if checkpoint.get('last_status') == 'silent' and (checkpoint.get('native_history_retained') is False
                or 'native_history_retained' not in checkpoint and not restored_items):
            reason = 'idle_history_discarded'
        else:
            reason, rotate = 'history_unavailable', True
    if rotate:
        current = make_id()
    restore = not history_present or rotate or reset
    count = max(0, restored_items) if restore and not reset else 0
    state = ('continued' if reason == 'same_native_session' and history_present
             else 'restored' if count else 'fresh' if reason == 'idle_history_discarded' else 'new')
    metadata = {'context_kind': context_kind(scope), 'context_scope': scope,
        'context_state': state, 'context_reason': reason, 'restored_context_items': count,
        'context_previous_version': checkpoint.get('context_format_version'), 'context_version': version,
        'context_id_changed': bool(previous and previous != current),
        'context_native_history_available': bool(history_present), 'context_native_bytes': max(0, history_bytes),
        'context_lifetime': 'persistent', 'context_previous_status': checkpoint.get('last_status')}
    return NativeContext(current, restore, metadata)


_WAIT = re.compile(r'等待下次|等下次|下次(?:统一)?巡检|wait(?:ing)?\s+(?:for\s+)?(?:the\s+)?next', re.I)
_ROUTINE = re.compile(r'例行巡检|空间巡检|日常巡检|routine\s+(?:patrol|inspection)', re.I)
_WORK_TOOLS = frozenset({'save_work_note', 'generate_image', 'generate_private_image', 'execute_python',
    'begin_research_task', 'update_research_task', 'read_artifact', 'observe_post_images',
    'search_web', 'open_web_page', 'find_web_page', 'search_history', 'inspect_web_resources',
    'list_x_posts', 'search_x_posts', 'search_xhs_posts', 'read_research_evidence', 'read_research_resources'})


class PatrolProgress:
    """Finish repeated waiting checkpoints without changing useful tool flows."""
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.meaningful = False
        self.finished = False
        self.finish_reason = ''
        self.progress_tools: set[str] = set()
        self._seen_work: set[str] = set()
        self._waiting_goal = None
        self._waiting_note = None
        self._revision = self._waiting_revision = 0

    def before(self, name, arguments):
        if not self.enabled or name != 'remember_intention' or not isinstance(arguments, dict):
            return None
        goal = re.sub(r'\s+', '', str(arguments.get('goal', '')))
        note = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        waiting = bool(_WAIT.search(str(arguments.get('next_step', ''))))
        repeated = self._waiting_goal == goal and self._waiting_revision == self._revision
        if waiting and repeated and (_ROUTINE.search(goal) or note == self._waiting_note):
            self.finished, self.finish_reason = True, 'patrol_no_progress'
            return {'status': 'ok', 'data': {'saved': True, 'unchanged': True,
                'handler_executed': False, 'turn_finished': True, 'reason': self.finish_reason}}
        return None

    def observe(self, name, arguments, result):
        if not self.enabled or result.get('status') not in {'ok', 'sent'}:
            return
        data = result.get('data') or {}
        if data.get('error'):
            return
        if name == 'finish_patrol':
            self.finished, self.finish_reason = True, 'patrol_finished'
        elif name == 'remember_intention':
            if _WAIT.search(str(arguments.get('next_step', ''))):
                self._waiting_goal = re.sub(r'\s+', '', str(arguments.get('goal', '')))
                self._waiting_note = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
                self._waiting_revision = self._revision
            else:
                self._advance(name)
        elif name in _WORK_TOOLS or name == 'read_friend_feed' and data.get('posts'):
            fingerprint = hashlib.sha256(json.dumps([name, arguments, data],
                sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if fingerprint not in self._seen_work:
                self._seen_work.add(fingerprint)
                self._advance(name)

    def _advance(self, name):
        self.meaningful = True
        self._revision += 1
        self.progress_tools.add(name)

    def retention(self):
        return {'retain_history': self.meaningful,
            'reason': 'useful_work_retained' if self.meaningful else 'idle_history_discarded',
            'progress_tools': sorted(self.progress_tools)}
