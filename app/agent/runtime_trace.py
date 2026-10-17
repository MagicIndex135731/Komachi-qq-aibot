"""Opt-in private Agent inspection, isolated from ordinary application logs.

The viewer receives sanitized final requests and public execution events. Raw
reasoning, encrypted replay, credentials and binary images never enter records.
Bounded asynchronous writes may be dropped; they never control a model/tool run.
"""
from __future__ import annotations

from itertools import islice
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import logging
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
import threading
import time
from uuid import uuid4


logger = logging.getLogger(__name__)
TRACE_ID = re.compile(r"[a-f0-9]{32}\Z")
ITEM_ID = re.compile(r"[a-z0-9_-]{1,64}\Z")
CONVERSATION_ID = re.compile(r"[cs]-[a-f0-9]{24}\Z")
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024
MAX_EVENT_BYTES = 32 * 1024
MAX_STREAM_CHARS = 65536
MAX_TURN_BYTES = 32 * 1024 * 1024
_SECRETS = {"authorization", "api_key", "apikey", "access_token", "password", "secret", "cookie", "cookies", "headers"}
_PRIVATE_REASONING = {"encrypted_content", "reasoning_content", "reasoning_text", "thinking", "thought_signature", "signature"}
_ASSIGNMENT = re.compile(r"(?i)(\b(?:api[_-]?key|access[_-]?token|password|authorization|secret|cookie)\b[\"']?\s*[:=]\s*[\"']?)([^\s\"',;}]+)")
_URL_SECRET = re.compile(r"(?i)([?&](?:token|key|api_key|access_token|rkey|secret|password)=)[^&#\s\"']+")
_ACTIVE_TRACE = ContextVar('agent_private_observation', default=None)
BACKGROUND_EXECUTION = frozenset({'background_tool', 'delegation_job', 'tool_delivery'})


def _background_execution(meta: dict) -> bool:
    kind = meta.get('execution_kind')
    return (isinstance(kind, str) and kind in BACKGROUND_EXECUTION or
        meta.get('trigger_kind') == 'media_delivery' and meta.get('model_requests') == 0)


def _conversation_identity(meta):
    logical = meta.get('conversation_key')
    if isinstance(logical, str) and re.fullmatch('[a-f0-9]{24}', logical):
        if meta.get('context_lifetime') == 'ephemeral':
            # A fresh judge is never a continuation of a persistent owner,
            # even if an old producer used the same logical entry key.
            return 'c-' + hashlib.sha256(('ephemeral:' + logical).encode()).hexdigest()[:24], 'ephemeral'
        return 'c-' + logical, 'logical'
    native = meta.get('context_session_key')
    if meta.get('context_lifetime') != 'ephemeral' and isinstance(native, str) and re.fullmatch('[a-f0-9]{24}', native):
        return 'c-' + native, 'native_context'
    native = meta.get('session_key')
    return 's-' + (native if isinstance(native, str) and re.fullmatch('[a-f0-9]{24}', native)
                   else meta['trace_id'][:24]), 'native_only'


def _view_meta(meta):
    # This named entry point has always used a fresh, tool-free SDK home.
    # Legacy records predate the explicit lifetime field; their native IDs
    # identify individual judgments, not twenty persistent conversations.
    if meta.get('trigger_kind') == 'participation_judge' and not meta.get('context_lifetime'):
        return {**meta, 'context_lifetime':'ephemeral', 'context_state':'new',
            'context_reason':'ephemeral_judgment', 'context_identity_legacy':True}
    if _background_execution(meta) and not meta.get('execution_kind'):
        return {**meta, 'execution_kind':'tool_delivery'}
    return meta


@contextmanager
def observe_with(trace, *, call_id=None):
    """Link real execution without adding trace data to model input or authority."""
    token = _ACTIVE_TRACE.set((trace, call_id))
    try:
        yield
    finally:
        _ACTIVE_TRACE.reset(token)


def current_trace_context() -> dict:
    active = _ACTIVE_TRACE.get()
    if not active or not active[0].enabled or active[0].closed:
        return {}
    trace, call_id = active
    meta = trace.meta
    result = {'parent_trace_id': trace.trace_id,
        'workflow_id': meta.get('workflow_id') or trace.trace_id}
    conversation = meta.get('workflow_conversation_id')
    if not conversation:
        conversation = _conversation_identity(meta)[0]
    if isinstance(conversation, str) and CONVERSATION_ID.fullmatch(conversation):
        result['workflow_conversation_id'] = conversation
    if type(meta.get('job_id')) is int:
        result['job_id'] = meta['job_id']
    if isinstance(call_id, str):
        result['parent_tool_call_id'] = call_id[:500]
    return result


def execution_event(kind: str, data: dict) -> None:
    active = _ACTIVE_TRACE.get()
    if active:
        active[0].event(kind, data)


def _text(value: str) -> str:
    value = _URL_SECRET.sub(r"\1[已省略凭据]", value)
    value = _ASSIGNMENT.sub(r"\1[已省略凭据]", value)
    value = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [已省略凭据]", value)
    return value


def sanitize(value, *, depth=0, truncated=None):
    if depth > 32:
        if truncated is not None:
            truncated[0] = True
        return "[已省略过深结构]"
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value[:120]:
            return f"[已省略图片二进制；编码字符数={len(value)}]"
        if len(value) > 1024 * 1024 and truncated is not None:
            truncated[0] = True
        text = _text(value[:1024 * 1024])
        return text + ("[内容截断]" if len(value) > 1024 * 1024 else "")
    if isinstance(value, dict):
        if value.get("type") == "reasoning_text":
            return {"type": "reasoning_text", "text": "[已省略隐藏推理]"}
        image = value.get("type") in {"image", "input_image", "image_url"} or "mediaType" in value or str(value.get("mime_type", "")).startswith("image/")
        result = {}
        for key, item in islice(value.items(), 4096):
            name = str(key)
            if name.lower() in _SECRETS or name.lower() in _PRIVATE_REASONING or name.lower() == "reasoning" and isinstance(item, str):
                result[name] = "[已省略凭据或隐藏推理]"
            elif name == "data" and image:
                result[name] = f"[已省略图片二进制；编码字符数={len(item) if isinstance(item,str) else 0}]"
            else:
                result[name] = sanitize(item, depth=depth + 1, truncated=truncated)
        if len(value) > 4096:
            result["_truncated"] = True
            result["omitted_keys"] = len(value) - 4096
            if truncated is not None:
                truncated[0] = True
        return result
    if isinstance(value, (list, tuple)):
        result = [sanitize(item, depth=depth + 1, truncated=truncated) for item in value[:4096]]
        if len(value) > 4096:
            result.append({"_truncated": True, "omitted_items": len(value) - 4096})
            if truncated is not None:
                truncated[0] = True
        return result
    return value if value is None or type(value) in (int, float, bool) else "[已省略非JSON对象]"


def _encode(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf8")


def _bounded(value, maximum):
    truncated = [False]
    clean = sanitize(value, truncated=truncated)
    raw = _encode(clean)
    if len(raw) <= maximum:
        return clean, truncated[0], len(raw)
    return {"_truncated": True, "original_bytes": len(raw),
            "preview": raw[:max(0, maximum - 512)].decode("utf8", errors="ignore")}, True, len(raw)


def _atomic(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(_encode(value))
    temporary.replace(path)


def _load(path: Path, maximum=MAX_SNAPSHOT_BYTES + 65536):
    if path.is_symlink():
        raise ValueError("trace_path_invalid")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("trace_record_too_large")
    return json.loads(data)


def _prefix(body):
    return _encode({"model": body.get("model"), "instructions": body.get("instructions"),
                    "tools": body.get("tools"), "history": body.get("input", body.get("messages"))}).decode()


def context_observation(scope: str, *, previous_id, current_id, checkpoint: dict,
                        history_present: bool, restored_count: int, format_version=None) -> dict:
    """Describe an already-selected session; never select or modify model context."""
    kind = ('background_group' if scope.startswith('background:group:')
            else 'research' if scope.startswith('research:') else 'owner_task' if scope.startswith('owner:')
            else 'private' if scope.startswith('private:') else 'group' if scope.startswith('group:') else 'public')
    continued = bool(previous_id and previous_id == current_id and history_present)
    reason = 'same_native_session' if continued else 'first_turn'
    if not continued and not previous_id and checkpoint.get('last_status') in {
            'source_invalidated', 'memory_changed', 'persona_changed'}:
        reason = checkpoint['last_status']
    if not continued and previous_id:
        reason = ('interrupted_turn' if checkpoint.get('in_flight') else 'previous_failed'
            if checkpoint.get('last_status') in {'failed','blocked','uncertain'} or checkpoint.get('context_reusable') is False
            else 'model_format_changed' if format_version is not None and checkpoint.get('context_format_version') != format_version
            else 'history_unavailable' if not history_present else 'session_rotation')
    return {'context_kind': kind, 'context_scope': scope,
            'context_state': 'continued' if continued else 'restored' if restored_count else 'new',
            'context_reason': reason, 'restored_context_items': max(0, restored_count)}


class AgentTrace:
    def __init__(self, root: Path | None, *, context: dict, session_key: str,
                 conversation_key: str | None = None, max_turns=100,
                 max_bytes=256 * 1024 * 1024, snapshot_bytes=MAX_SNAPSHOT_BYTES):
        self.trace_id = uuid4().hex
        self.enabled = False
        self.failed = False
        self.dropped = 0
        self.closed = False
        self.finished = False
        self._seq = self._bytes = 0
        self._record_sizes = {}
        self._streams = {}
        self._stream_lock = threading.Lock()
        self._queue = Queue(maxsize=16)
        self._thread = None
        self._snapshot_bytes = min(MAX_SNAPSHOT_BYTES, max(1024, snapshot_bytes))
        self._last_request = None
        self._last_prefix = ""
        self._max_turns, self._max_bytes = max(1, max_turns), max_bytes
        if root is None:
            return
        try:
            self.root = Path(root).absolute()
            if self.root.is_symlink():
                raise ValueError("trace_path_invalid")
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.path = self.root / self.trace_id
            self.path.mkdir(mode=0o700)
            now = time.time()
            self.meta = {"trace_id": self.trace_id, "started_at": now, "updated_at": now, "status": "running",
                "session_key": hashlib.sha256(session_key.encode()).hexdigest()[:24],
                "surface": "unknown", "source_id": None, "model_requests": 0,
                "last_seq": 0, "dropped_events": 0, **sanitize(context)}
            if conversation_key is not None:
                self.meta['conversation_key'] = hashlib.sha256(conversation_key.encode()).hexdigest()[:24]
            _atomic(self.path / "meta.json", self.meta)
            self.enabled = True
            self._thread = threading.Thread(target=self._worker, name="agent-private-trace", daemon=True)
            self._thread.start()
            self.event("turn.started", {"surface": self.meta["surface"], "source_id": self.meta.get("source_id")})
        except (OSError, ValueError, TypeError):
            self.failed = True
            logger.warning("agent_private_trace_unavailable")

    def _submit(self, item):
        if not self.enabled or self.closed:
            return
        try:
            self._queue.put_nowait(item)
        except Full:
            self.dropped += 1
            if item[0] == "finish":
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    self._queue.put_nowait(item)
                except (Empty, Full):
                    pass

    def event(self, kind: str, data: dict):
        if not self.enabled or self.closed:
            return
        try:
            clean, truncated, _ = _bounded(data, MAX_EVENT_BYTES)
            self._submit(("event", kind, {**clean, "_truncated": True} if truncated else clean))
        except Exception:
            self.dropped += 1

    def request(self, request_id: str, body: dict, metrics: dict, *, purpose: str):
        if not self.enabled or self.closed:
            return
        try:
            if not ITEM_ID.fullmatch(request_id):
                raise ValueError("trace_request_id_invalid")
            clean, truncated, original = _bounded(body, self._snapshot_bytes)
            self._submit(("request", request_id, {"body": clean, "meta": {"request_id": request_id,
                "purpose": purpose, "captured_at": time.time(), "truncated": truncated,
                "snapshot_bytes": len(_encode(clean)), "original_bytes": original,
                "redacted": True, "binary_omitted": True, **sanitize(metrics)}}))
        except Exception:
            self.dropped += 1

    def stream(self, request_id: str, chunk: dict, *, force=False):
        if not self.enabled or self.closed:
            return
        try:
            if not ITEM_ID.fullmatch(request_id):
                return
            with self._stream_lock:
                if request_id not in self._streams and len(self._streams) >= 128:
                    self.dropped += 1
                    return
                state = self._streams.setdefault(request_id, {"output": "", "summary": "", "reasoning_chars": 0,
                    "summary_status": "not_provided", "last_publish": 0.0, "truncated": False})
                event, delta = chunk.get("type"), chunk.get("delta")
                if event == "response.output_text.delta" and isinstance(delta, str):
                    state["output"] += delta
                elif event == "response.reasoning_summary_text.delta" and isinstance(delta, str):
                    state["summary"] += delta
                    state["summary_status"] = "provided"
                elif event == "response.reasoning_text.delta" and isinstance(delta, str):
                    state["reasoning_chars"] += len(delta)
                elif event in {"response.output_text.done", "response.reasoning_summary_text.done"} and isinstance(chunk.get("text"), str):
                    key = "output" if event == "response.output_text.done" else "summary"
                    state[key] = chunk["text"]
                    if key == "summary":
                        state["summary_status"] = "provided"
                for choice in chunk.get("choices", []) if isinstance(chunk.get("choices"), list) else []:
                    content = choice.get("delta", choice.get("message", {})) if isinstance(choice, dict) else {}
                    if isinstance(content, dict):
                        if isinstance(content.get("content"), str):
                            state["output"] += content["content"]
                        for key in ("reasoning_content", "reasoning"):
                            if isinstance(content.get(key), str):
                                state["reasoning_chars"] += len(content[key])
                # Non-stream Responses may expose only their completed output.
                response = chunk.get("response", chunk)
                for item in response.get("output", []) if isinstance(response, dict) and isinstance(response.get("output"), list) else []:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "reasoning":
                        summary = "".join(p.get("text", "") for p in item.get("summary", []) if isinstance(p, dict) and p.get("type") == "summary_text")
                        if summary:
                            state["summary"], state["summary_status"] = summary, "provided"
                    elif item.get("type") == "message":
                        text = "".join(p.get("text", "") for p in item.get("content", []) if isinstance(p, dict) and p.get("type") == "output_text")
                        if text:
                            state["output"] = text
                for key in ("output", "summary"):
                    if len(state[key]) > MAX_STREAM_CHARS:
                        state[key] = state[key][:MAX_STREAM_CHARS]
                        state["truncated"] = True
                if force or time.monotonic() - state["last_publish"] >= 0.25:
                    state["last_publish"] = time.monotonic()
                    clean = {k: sanitize(v) for k, v in state.items() if k != "last_publish"}
                    self._submit(("stream", request_id, clean))
        except Exception:
            self.dropped += 1

    def finish(self, status: str, reason: str):
        if self.finished:
            return
        self.finished = True
        for request_id in list(self._streams):
            self.stream(request_id, {}, force=True)
        self._submit(("finish", status, {"reason": sanitize(reason)}))

    def close(self):
        if self.closed:
            return
        if not self.finished:
            self.finish("interrupted", "runtime_closed")
        if self._thread:
            deadline = time.monotonic() + 1.5
            while self._queue.unfinished_tasks and time.monotonic() < deadline:
                time.sleep(0.01)
        self.closed = True
        if self._thread:
            self._thread.join(timeout=0.1)

    def _append(self, kind, data):
        event = {"seq": self._seq + 1, "at": time.time(), "kind": kind, "data": data}
        raw = _encode(event) + b"\n"
        if self._bytes + len(raw) > MAX_TURN_BYTES and kind != "turn.completed":
            self.dropped += 1
            return
        self._seq += 1
        fd = os.open(self.path / "events.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "ab") as stream:
            stream.write(raw)
        self._bytes += len(raw)
        self.meta.update(last_seq=self._seq, updated_at=event["at"], dropped_events=self.dropped)

    def _record(self, name, data):
        size = len(_encode(data))
        total = self._bytes - self._record_sizes.get(name, 0) + size
        if total > MAX_TURN_BYTES:
            self.dropped += 1
            return False
        _atomic(self.path / name, data)
        self._bytes, self._record_sizes[name] = total, size
        return True

    def _previous(self):
        for meta in TraceReader(self.root).list_traces(limit=200):
            if meta.get("trace_id") == self.trace_id or meta.get("session_key") != self.meta["session_key"]:
                continue
            previous = meta.get("last_request_id")
            if previous:
                record = TraceReader(self.root).read_request(meta["trace_id"], previous)
                self._last_prefix = _prefix(record["body"])
                self._last_request = (meta["trace_id"], previous)
                break

    def _retention(self):
        records = TraceReader(self.root).list_traces(limit=500)
        total = 0
        for index, meta in enumerate(records):
            directory = self.root / meta["trace_id"]
            files = [p for p in directory.iterdir() if p.is_file() and not p.is_symlink()]
            size = sum(p.stat().st_size for p in files)
            total += size
            active = meta.get("status") == "running" and time.time() - meta.get("updated_at", 0) < 3600
            if not active and (index >= self._max_turns or total > self._max_bytes or time.time() - meta.get("updated_at", 0) > 86400):
                for p in files:
                    if p.name in {"meta.json", "events.jsonl"} or p.name.startswith(("request-", "stream-")) and p.suffix in {".json", ".tmp"}:
                        p.unlink(missing_ok=True)
                try:
                    directory.rmdir()
                    total -= size
                except OSError:
                    pass

    def _worker(self):
        try:
            self._previous()
            self._retention()
        except (OSError, ValueError, TypeError, KeyError):
            pass
        while not self.closed or not self._queue.empty():
            try:
                operation, key, data = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                if self._bytes > MAX_TURN_BYTES and operation != "finish":
                    self.dropped += 1
                    continue
                if operation == "request":
                    prefix = _prefix(data["body"])
                    same = 0
                    for a, b in zip(prefix, self._last_prefix):
                        if a != b:
                            break
                        same += 1
                    data["meta"].update(shared_prefix_chars=same, previous_trace_id=self._last_request[0] if self._last_request else None,
                        previous_request_id=self._last_request[1] if self._last_request else None)
                    if not self._record(f"request-{key}.json", data):
                        continue
                    self._last_request, self._last_prefix = (self.trace_id, key), prefix
                    self.meta.update(last_request_id=key, model_requests=self.meta["model_requests"] + 1)
                    self._append("model.request", data["meta"])
                elif operation == "stream":
                    if not self._record(f"stream-{key}.json", {"request_id": key, **data}):
                        continue
                    self._append("model.progress", {"request_id": key, "output_chars": len(data["output"]),
                        "summary_chars": len(data["summary"]), "reasoning_chars": data["reasoning_chars"]})
                elif operation == "finish":
                    self.meta.update(status=key, finished_at=time.time())
                    self._append("turn.completed", {"status": key, **data})
                else:
                    if key == 'model.finished':
                        for field in ('provider_response_id', 'provider_previous_response_id', 'provider_chain_id',
                                      'continuation_mode', 'continuation_reason', 'prompt_cache_key_id'):
                            if data.get(field) is not None:
                                self.meta[field] = data[field]
                        chain = data.get('provider_chain_id')
                        if isinstance(chain, str):
                            self.meta['provider_chain_ids'] = list(dict.fromkeys([
                                *self.meta.get('provider_chain_ids', []), chain]))[-32:]
                    elif key == 'context.retention':
                        self.meta['context_retention'] = data
                    self._append(key, data)
                _atomic(self.path / "meta.json", self.meta)
                if operation == "finish":
                    self._retention()
            except Exception:
                if not self.failed:
                    logger.warning("agent_private_trace_write_failed")
                self.failed = True
                self.dropped += 1
            finally:
                self._queue.task_done()


class TraceReader:
    def __init__(self, root: Path):
        self.root = Path(root).absolute()
        self._job_link_cache = {}

    def _recorded_jobs(self, meta):
        """Read explicit queue results only; source/time proximity is not lineage."""
        trace_id = meta['trace_id']
        path = self._directory(trace_id) / 'events.jsonl'
        try:
            stat = path.stat()
            if path.is_symlink() or stat.st_size > MAX_TURN_BYTES + 65536:
                return {}
            signature = (stat.st_mtime_ns, stat.st_size)
            cached = self._job_link_cache.get(trace_id)
            if cached and cached[0] == signature:
                return cached[1]
            jobs = {}
            with path.open('rb') as handle:
                for raw in handle:
                    if len(raw) > MAX_EVENT_BYTES + 65536:
                        continue
                    try:
                        event = json.loads(raw)
                    except ValueError:
                        continue
                    if not isinstance(event, dict) or not isinstance(event.get('data'), dict):
                        continue
                    data = event['data']
                    if event.get('kind') == 'job.queued':
                        entries = [data]
                    elif event.get('kind') == 'tool.completed' and data.get('status') in {'ok', 'queued'}:
                        result = data.get('data')
                        if not isinstance(result, dict):
                            continue
                        entries = [result, *(result.get('jobs') if isinstance(result.get('jobs'), list) else [])]
                    else:
                        continue
                    for item in entries:
                        job_id = item.get('job_id') if isinstance(item, dict) else None
                        if type(job_id) is int and job_id > 0:
                            jobs[job_id] = data.get('call_id')
            self._job_link_cache[trace_id] = (signature, jobs)
            return jobs
        except OSError:
            return {}

    def _directory(self, trace_id):
        if not isinstance(trace_id, str) or not TRACE_ID.fullmatch(trace_id):
            raise ValueError("trace_id_invalid")
        path = self.root / trace_id
        if self.root.is_symlink() or path.is_symlink() or path.resolve().parent != self.root.resolve():
            raise ValueError("trace_path_invalid")
        return path

    def list_traces(self, *, limit=100):
        if not self.root.exists():
            return []
        if self.root.is_symlink():
            raise ValueError("trace_path_invalid")
        results = []
        for p in islice(self.root.iterdir(), 1000):
            if not TRACE_ID.fullmatch(p.name) or p.is_symlink() or not p.is_dir():
                continue
            try:
                value = _load(p / "meta.json", 65536)
                if isinstance(value, dict) and value.get("trace_id") == p.name:
                    results.append(value)
            except (OSError, ValueError):
                continue
        return sorted(results, key=lambda item: item.get("started_at", 0), reverse=True)[:min(500, limit)]

    def read_request(self, trace_id: str, request_id: str):
        if not isinstance(request_id, str) or not ITEM_ID.fullmatch(request_id):
            raise ValueError("trace_request_id_invalid")
        return _load(self._directory(trace_id) / f"request-{request_id}.json")

    def list_conversations(self, *, limit=100):
        groups = {}
        trace_groups = {}
        traces = [_view_meta(meta) for meta in self.list_traces(limit=500)]
        live = {meta['trace_id'] for meta in traces}
        self._job_link_cache = {key: value for key, value in self._job_link_cache.items() if key in live}
        # Some application-side tools ran outside the optional observation
        # context. Their saved tool result still proves the exact queued job.
        workers = [meta for meta in traces if _background_execution(meta)]
        candidates = {}
        for worker in workers:
            if type(worker.get('job_id')) is not int or type(worker.get('source_id')) is not int:
                continue
            scope = tuple(worker.get(key) for key in ('surface', 'scope_id', 'source_id', 'actor_id'))
            candidates.setdefault(scope, []).append(worker)
        for scope, items in candidates.items():
            owners = {}
            for meta in traces:
                if _background_execution(meta) or tuple(meta.get(key) for key in ('surface', 'scope_id', 'source_id', 'actor_id')) != scope:
                    continue
                for job_id, call_id in self._recorded_jobs(meta).items():
                    owners.setdefault(job_id, []).append((meta, call_id))
            for worker in items:
                matches = owners.get(worker['job_id'], [])
                if len(matches) != 1:
                    continue
                owner, call_id = matches[0]
                # An existing main-parent association remains authoritative.
                known = next((m for m in traces if m['trace_id'] == worker.get('parent_trace_id')), None)
                if known and not _background_execution(known):
                    continue
                worker.update(parent_trace_id=owner['trace_id'], workflow_id=owner['trace_id'],
                    workflow_conversation_id=_conversation_identity(owner)[0],
                    parent_tool_call_id=call_id, lineage_recovered='recorded_job_result')
        by_trace = {meta['trace_id']:meta for meta in traces}
        for meta in traces:
            key, quality = _conversation_identity(meta)
            trace_groups[meta['trace_id']] = key
            groups.setdefault(key, {'conversation_id': key, 'identity_quality': quality, 'turns': []})['turns'].append(meta)
        results = []
        for group in groups.values():
            turns = sorted(group['turns'], key=lambda t: t.get('started_at', 0))
            last = turns[-1]
            group.update(turns=turns, turn_count=len(turns), surface=last.get('surface', 'unknown'),
                scope_id=last.get('scope_id'), actor_id=last.get('actor_id'), context_kind=last.get('context_kind', 'unknown'),
                context_scope=last.get('context_scope'),
                status=last.get('status'), updated_at=max(t.get('updated_at', 0) for t in turns),
                native_session_count=len({t.get('context_session_key') or t.get('session_key', t['trace_id']) for t in turns
                    if not _background_execution(t) and t.get('context_lifetime') != 'ephemeral'}),
                ephemeral_execution_count=sum(t.get('context_lifetime') == 'ephemeral' for t in turns),
                context_session_key=last.get('context_session_key'),
                provider_chain_ids=list(dict.fromkeys(chain for t in turns
                    for chain in (t.get('provider_chain_ids') if isinstance(t.get('provider_chain_ids'), list) else [])
                    if isinstance(chain, str)))[:64],
                request_preview=last.get('request_preview'), model=last.get('model'))
            results.append(group)
        by_id = {group['conversation_id']: group for group in results}
        # Retain old native-context URLs and persisted worker ancestry after
        # changing the sidebar to stable owner identities. Ambiguous aliases
        # must never cross logical owners or broaden a deletion selection.
        aliases = {}
        for meta in traces:
            native = meta.get('context_session_key')
            if meta.get('context_lifetime') != 'ephemeral' and isinstance(native, str) and re.fullmatch('[a-f0-9]{24}', native):
                aliases.setdefault('c-' + native, set()).add(trace_groups[meta['trace_id']])
        for alias, owners in aliases.items():
            if len(owners) == 1 and alias not in by_id:
                group = by_id[next(iter(owners))]
                group.setdefault('conversation_aliases', []).append(alias)
                by_id[alias] = group
        hidden = set()
        for meta in traces:
            parent_id = trace_groups.get(meta.get('parent_trace_id')) if isinstance(meta.get('parent_trace_id'), str) else None
            if _background_execution(by_trace.get(meta.get('parent_trace_id'), {}) if isinstance(meta.get('parent_trace_id'), str) else {}):
                parent_id = meta.get('workflow_conversation_id') or parent_id
            parent_id = parent_id or meta.get('workflow_conversation_id')
            parent = by_id.get(parent_id) if isinstance(parent_id, str) else None
            is_worker = _background_execution(meta)
            if parent and (is_worker or meta.get('context_kind') == 'research'):
                # Require the same real scope, never infer ancestry from a
                # timestamp, preview, native path, or an equal-looking title.
                if (parent.get('surface'), parent.get('scope_id')) != (meta.get('surface'), meta.get('scope_id')):
                    continue
                own_id = trace_groups[meta['trace_id']]
                if own_id == parent['conversation_id']:
                    continue
                parent.setdefault('task_traces', []).append(meta)
                if is_worker:
                    hidden.add(own_id)
        # Workers do not have a native model session. Keep them beneath their
        # parent; if that parent expired/deleted, retain an honest orphan view.
        results = [group for group in results if group['conversation_id'] not in hidden]
        for group in results:
            children = group.get('task_traces', [])
            children.sort(key=lambda item: (item.get('started_at', 0), item['trace_id']))
            group['task_count'] = len({item['job_id'] if type(item.get('job_id')) is int and item['job_id'] > 0
                else item['trace_id'] for item in children})
            if children:
                group['updated_at'] = max(group['updated_at'], *(item.get('updated_at', 0) for item in children))
        return sorted(results, key=lambda g: g['updated_at'], reverse=True)[:min(100, max(1, limit))]

    def read_conversation(self, conversation_id: str):
        if not isinstance(conversation_id, str) or not CONVERSATION_ID.fullmatch(conversation_id):
            raise ValueError('conversation_id_invalid')
        for group in self.list_conversations():
            if group['conversation_id'] == conversation_id or conversation_id in group.get('conversation_aliases', ()):
                return group
        raise FileNotFoundError('conversation_not_found')

    def read_trace(self, trace_id: str, *, after=0):
        directory = self._directory(trace_id)
        meta = _load(directory / "meta.json", 65536)
        events, streams, cursor = [], {}, max(0, after)
        path = directory / "events.jsonl"
        if path.exists():
            if path.is_symlink() or path.stat().st_size > MAX_TURN_BYTES + 65536:
                raise ValueError("trace_record_too_large")
            with path.open("rb") as handle:
                for raw in handle:
                    if len(raw) > MAX_EVENT_BYTES + 65536:
                        continue
                    try:
                        event = json.loads(raw)
                    except ValueError:
                        continue
                    if not isinstance(event, dict) or type(event.get("seq")) is not int or event["seq"] <= after:
                        continue
                    events.append(event)
                    cursor = event["seq"]
                    if len(events) >= 200:
                        break
        for event in events:
            if event.get("kind") == "model.progress":
                request_id = event.get("data", {}).get("request_id")
                if isinstance(request_id, str) and ITEM_ID.fullmatch(request_id):
                    try:
                        streams[request_id] = _load(directory / f"stream-{request_id}.json", 1024 * 1024)
                    except (OSError, ValueError):
                        pass
        return {"meta": meta, "events": events, "streams": streams, "cursor": cursor}
