"""Read-only, bounded usage summaries; no prompts or model calls are read here."""
from __future__ import annotations

from collections import OrderedDict
import json
import threading
import time

from app.agent.runtime_trace import ITEM_ID, MAX_EVENT_BYTES, MAX_TURN_BYTES, _background_execution


def token(value):
    return value if type(value) is int and 0 <= value <= 2**53 - 1 else None


class ConversationMetrics:
    def __init__(self, reader, *, max_cache=500, max_scan_bytes=2 * 1024 * 1024):
        self.reader = reader
        self.cache = OrderedDict()
        self.max_cache = max(1, min(500, max_cache))
        self.max_scan_bytes = max(1024, min(2 * 1024 * 1024, max_scan_bytes))
        self.lock = threading.Lock()

    def _requests(self, trace_id, budget):
        path = self.reader._directory(trace_id) / 'events.jsonl'
        if path.is_symlink():
            raise ValueError('trace_path_invalid')
        try:
            stat = path.stat()
        except FileNotFoundError:
            return {}, True
        if stat.st_size > MAX_TURN_BYTES + 65536:
            return {}, False
        state = self.cache.get(trace_id)
        identity = (stat.st_dev, stat.st_ino)
        if (state is None or state['identity'] != identity or stat.st_size < state['offset']
                or stat.st_size == state['offset'] and stat.st_mtime_ns != state['mtime']):
            state = {'identity':identity, 'offset':0, 'mtime':0, 'requests':{}, 'complete':True}
            self.cache[trace_id] = state
        self.cache.move_to_end(trace_id)
        while len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)
        if state['offset'] == stat.st_size and state['mtime'] == stat.st_mtime_ns:
            return state['requests'], state['complete']
        with path.open('rb') as stream:
            stream.seek(state['offset'])
            for _ in range(20000):
                if budget[0] <= 0:
                    break
                start = stream.tell()
                raw = stream.readline(min(MAX_EVENT_BYTES + 65537, budget[0]))
                budget[0] -= len(raw)
                if not raw:
                    break
                if not raw.endswith(b'\n'):
                    # An append may still be in flight. Re-read its tail next poll.
                    stream.seek(start)
                    break
                state['offset'] = stream.tell()
                if len(raw) > MAX_EVENT_BYTES + 65536:
                    state['complete'] = False
                    continue
                try:
                    event = json.loads(raw)
                except (ValueError, UnicodeError):
                    state['complete'] = False
                    continue
                if not isinstance(event, dict) or event.get('kind') not in {'model.request','model.finished'}:
                    continue
                data = event.get('data')
                rid = data.get('request_id') if isinstance(data, dict) else None
                if not isinstance(rid, str) or not ITEM_ID.fullmatch(rid):
                    continue
                if rid not in state['requests'] and len(state['requests']) >= 128:
                    state['complete'] = False
                    continue
                record = state['requests'].setdefault(rid, {'request_id':rid, 'order':token(event.get('seq')) or 0, 'begin':{}, 'end':{}})
                fields = ('context_estimated_tokens','context_window_tokens','context_input_ceiling',
                    'context_compaction_threshold_tokens','input_tokens','cached_input_tokens','output_tokens')
                values = {k:token(data[k]) for k in fields if k in data}
                for key, allowed in [('purpose',('agent','compaction')),('usage_status',('reported','partial','unknown','invalid'))]:
                    if key in data:
                        values[key] = data[key] if isinstance(data[key],str) and data[key] in allowed else None
                record['begin' if event['kind']=='model.request' else 'end'] = values
            state['mtime'] = stat.st_mtime_ns
        return state['requests'], state['complete'] and state['offset'] == stat.st_size

    def read(self, conversation_id):
        conversation = self.reader.read_conversation(conversation_id)
        turns = conversation['turns']
        latest_meta = next((meta for meta in reversed(turns) if not _background_execution(meta)), turns[-1])
        current_session = latest_meta.get('session_key')
        inputs, outputs, cache_inputs, cache_counts, totals = [], [], [], [], []
        requests = 0
        latest = None
        complete = True
        with self.lock:
            budget = [self.max_scan_bytes]
            for meta in reversed(turns):
                records, scanned = self._requests(meta['trace_id'], budget)
                complete = complete and scanned and not meta.get('dropped_events')
                requests += max(token(meta.get('model_requests')) or 0, len(records))
                for record in sorted(records.values(), key=lambda r:r['order'], reverse=True):
                    end = record['end'] if record['end'].get('usage_status') != 'invalid' else {}
                    inp, out, cached = (token(end.get(k)) for k in ('input_tokens','output_tokens','cached_input_tokens'))
                    if inp is not None:
                        inputs.append(inp)
                    if out is not None:
                        outputs.append(out)
                    if inp is not None and out is not None:
                        # This adapter's input already includes cache reads.
                        # Pair within one request; never add cached tokens again.
                        totals.append(inp+out)
                    if inp is not None and cached is not None and cached <= inp:
                        cache_inputs.append(inp);cache_counts.append(cached)
                    if (latest is None and meta.get('session_key') == current_session
                            and record['begin'].get('purpose') != 'compaction' and not record['request_id'].startswith('c')):
                        latest = (meta, record, inp, cached)
            context = {'input_tokens':None, 'window_tokens':token(latest_meta.get('context_window_tokens')),
                'source':'unknown', 'ratio':None, 'cache_ratio':None, 'cached_input_tokens':None,
                'trace_id':None, 'request_id':None, 'input_ceiling':None, 'compaction_threshold_tokens':None}
            if latest:
                meta, record, inp, cached = latest
                begin = record['begin']
                estimate = token(begin.get('context_estimated_tokens'))
                occupied = inp if inp is not None else estimate
                window = token(begin.get('context_window_tokens')) or context['window_tokens']
                paired = inp is not None and cached is not None and cached <= inp
                context.update(input_tokens=occupied, window_tokens=window,
                    source='reported' if inp is not None else 'estimated' if estimate is not None else 'unknown',
                    ratio=occupied/window if occupied is not None and window else None,
                    cache_ratio=cached/inp if paired and inp else None,
                    cached_input_tokens=cached if paired else None,
                    trace_id=meta['trace_id'], request_id=record['request_id'],
                    input_ceiling=token(begin.get('context_input_ceiling')),
                    compaction_threshold_tokens=token(begin.get('context_compaction_threshold_tokens')))
            context['window_source'] = 'captured' if context['window_tokens'] else 'unknown'
            if not context['window_tokens']:
                path = self.reader.root / 'window-config.json'
                try:
                    if path.is_symlink() or path.stat().st_size > 4096:
                        raise ValueError('configuration_reference_invalid')
                    config = json.loads(path.read_bytes())
                    captured = config.get('captured_at')
                    window = token(config.get('window_tokens'))
                    if (config.get('version') == 1 and window and type(captured) in (int,float)
                            and 0 <= time.time() - captured <= 86400):
                        context.update(window_tokens=window, window_source='configured_reference',
                            ratio=context['input_tokens']/window if context['input_tokens'] is not None else None)
                except (OSError,ValueError,TypeError,AttributeError):
                    pass
            usage = {'request_count':requests, 'input_tokens':sum(inputs) if inputs else None,
                'output_tokens':sum(outputs) if outputs else None,
                'real_total_tokens':sum(totals) if totals else None,
                'total_known_requests':len(totals),
                'cached_input_tokens':sum(cache_counts) if cache_counts else None,
                'uncached_input_tokens':sum(cache_inputs) - sum(cache_counts) if cache_inputs else None,
                'cache_creation_tokens':None, 'cache_creation_status':'not_reported_separately',
                'input_token_semantics':'includes_cache',
                'uncached_known_requests':len(cache_counts),
                'cache_ratio':sum(cache_counts)/sum(cache_inputs) if sum(cache_inputs) else None,
                'input_known_requests':len(inputs), 'output_known_requests':len(outputs),
                'cache_known_requests':len(cache_counts), 'events_complete':complete}
        return {'conversation_id':conversation_id, 'context':context, 'usage':usage,
                'scope':'retained_conversation_requests', 'read_only':True}
