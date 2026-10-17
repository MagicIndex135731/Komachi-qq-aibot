"""Bounded Responses delta transport over an authoritative local transcript.

Continuation is restricted to a live application turn. A new turn or process
starts from local history. Only explicit pre-output state rejection permits a
full-input fallback; streamed or uncertain model output is never replayed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from urllib.parse import urlsplit


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _item(value):
    if not isinstance(value, dict):
        return value
    kind = value.get('type')
    if kind == 'function_call':
        args = value.get('arguments')
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except (ValueError, TypeError):
            pass
        return {'type': kind, 'call_id': value.get('call_id'), 'name': value.get('name'), 'arguments': args}
    if value.get('role') == 'assistant':
        content = value.get('content')
        if isinstance(content, list):
            content = [{k: v for k, v in part.items() if k not in {'annotations', 'logprobs'}}
                       if isinstance(part, dict) else part for part in content]
        return {'role': 'assistant', 'content': content,
                **({'phase': value['phase']} if value.get('phase') is not None else {})}
    return value


def continuation_supported(mode, base_url, protocol):
    if mode not in {'off', 'auto', 'on'}:
        raise ValueError('invalid Responses continuation mode')
    if protocol != 'openai-responses' or mode == 'off':
        return False
    url = urlsplit(base_url)
    return mode == 'on' or url.scheme == 'https' and url.hostname == 'api.openai.com'


class ResponsesContinuation:
    def __init__(self, *, enabled=False, max_bytes=8 * 1024 * 1024):
        self.enabled, self.max_bytes = enabled, max_bytes
        self.response_id = self.chain_id = None
        self._canonical_input = self._wire_input = self._output = None
        self._signature = None
        self._disabled_reason = ''

    @property
    def uses_provider_state(self):
        return self.enabled and not self._disabled_reason

    def prepare(self, body: dict, *, canonical_input=None):
        """Keep previous live clock/observation items in the full fallback too.

        The caller supplies the SDK-owned input separately from its bounded live
        suffix. This makes wire continuation append-only within one turn while
        a fresh turn reconstructs clean context from canonical local storage.
        """
        full = deepcopy(body)
        full.pop('previous_response_id', None)
        full.pop('conversation', None)
        # Explicit opt-out of provider storage must not silently become a
        # remotely retained conversation, even when continuation is enabled.
        storage_disabled = body.get('store') is False
        canonical = deepcopy(canonical_input if canonical_input is not None else body.get('input', []))
        supplied = body.get('input', [])
        signature = _canonical({k: v for k, v in body.items()
                                if k not in {'input', 'previous_response_id', 'conversation', 'store'}})
        reason = self._disabled_reason or ('not_enabled' if not self.enabled else
            'response_storage_disabled' if storage_disabled else 'first_response')
        wire = full
        if self.enabled and not storage_disabled and not self._disabled_reason and self.response_id and isinstance(canonical, list):
            baseline = [*(self._canonical_input or []), *(self._output or [])]
            prefix_ok = len(canonical) >= len(baseline) and all(
                _canonical(_item(a)) == _canonical(_item(b)) for a, b in zip(baseline, canonical))
            if self._signature == signature and prefix_ok:
                # Dynamic guidance is a suffix after the SDK-owned input.
                suffix = supplied[len(canonical):] if supplied[:len(canonical)] == canonical else []
                delta = [*canonical[len(baseline):], *suffix]
                expanded = [*(self._wire_input or []), *(self._output or []), *delta]
                if len(_canonical(expanded).encode()) <= self.max_bytes:
                    full['input'] = expanded
                    wire = {**full, 'input': delta, 'previous_response_id': self.response_id}
                    wire.pop('conversation', None)
                    reason = 'previous_response'
                else:
                    reason = 'continuation_size_limit'
            else:
                reason = 'request_changed' if self._signature != signature else 'history_changed'
        if reason != 'previous_response' and self.enabled:
            self.response_id = self.chain_id = None
        # All fallback data remains in this live, isolated bridge only.
        self._pending = (canonical, deepcopy(full.get('input', [])), signature)
        self._pending_reusable = not storage_disabled
        metrics = {'continuation_mode': 'incremental' if reason == 'previous_response' else 'full',
            'continuation_reason': reason, 'provider_previous_response_id': wire.get('previous_response_id'),
            'provider_chain_id': self.chain_id,
            'wire_input_items': len(wire.get('input', [])), 'canonical_input_items': len(canonical)}
        return wire, full, metrics

    def complete(self, response):
        if not self.enabled or self._disabled_reason or not isinstance(response, dict):
            return
        if not getattr(self, '_pending_reusable', False):
            self.reset('response_storage_disabled')
            return
        identifier = response.get('id')
        output = response.get('output')
        if (response.get('status') != 'completed' or not provider_identity(identifier) or not isinstance(output, list)):
            self.reset('response_not_reusable')
            return
        if len(_canonical(output).encode()) > self.max_bytes:
            self.reset('response_too_large')
            return
        self._canonical_input, self._wire_input, self._signature = self._pending
        self._output = deepcopy(output)
        self.response_id = identifier
        self.chain_id = self.chain_id or identifier

    def reset(self, reason='state_reset', *, disable=False):
        self.response_id = self.chain_id = None
        self._canonical_input = self._wire_input = self._output = None
        self._signature = None
        if disable:
            self._disabled_reason = reason

    def reject(self, status, payload):
        """Only an explicitly rejected continuation can be resent in full."""
        error = payload.get('error', {}) if isinstance(payload, dict) else {}
        if not isinstance(error, dict) or status not in {400, 404, 422}:
            return None
        code, param = error.get('code'), error.get('param')
        if code in {'previous_response_not_found', 'response_not_found', 'invalid_previous_response_id'}:
            return 'previous_response_unavailable'
        if param == 'previous_response_id' and code in {'unsupported_parameter', 'unknown_parameter', 'invalid_request_error'}:
            return 'continuation_unsupported'
        return None


def cache_identity(key):
    return hashlib.sha256(str(key).encode()).hexdigest()[:24] if key else None


def provider_identity(value):
    return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.:/-]{1,240}', value) else None
