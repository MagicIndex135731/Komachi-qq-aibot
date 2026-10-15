"""Scope-local replay hygiene; canonical messages and action receipts stay intact."""
from __future__ import annotations

from copy import deepcopy
import json


TURN_PREFIX = 'Current application turn: '
_MEDIA_TYPES = frozenset({'input_image','image_url','image','input_audio','audio','video','input_video'})
_FAILED = frozenset({'failed','blocked','uncertain'})
_UNSAFE_REASONS = frozenset({'model_input_rejected', 'model_image_input_rejected',
    'model_provider_failed', 'model_stream_incomplete', 'model_stream_buffer_exceeded',
    'model_request_timeout', 'model_first_output_timeout', 'model_progress_timeout',
    'foreground_timeout', 'turn_timeout'})


def _safe_trace(trace) -> bool:
    # Delivery success is independent of generation success. A real receipt
    # survives, while rejected/partial model context must not be bootstrapped.
    if not isinstance(trace, (list, tuple)):
        return False
    requests = {}
    for row in trace:
        if not isinstance(row, dict):
            return False
        if row.get('reason') in _UNSAFE_REASONS or str(row.get('reason', '')).startswith(('runtime:', 'runtime_error')):
            return False
        if row.get('status') == 'model_request':
            requests[(row.get('purpose', 'agent'), row.get('model_call_index', 0))] = row
    return all(row.get('outcome') == 'completed' and row.get('output_disposition') != 'discarded'
        for row in requests.values())


def application_turn_marker(turn_id: str) -> str:
    return TURN_PREFIX + turn_id + '\n'


def project_turn_media(entries: list, *, current_turn_id: str, protocol: str, include_current_media: bool):
    """Remove previous pixels, retaining every media result from the current turn."""
    boundary = None
    marker = application_turn_marker(current_turn_id) if current_turn_id else ''
    for index, entry in enumerate(entries):
        if not isinstance(entry,dict): continue
        content=entry.get('content',[])
        parts=content if isinstance(content,list) else [{'text':content}] if isinstance(content,str) else []
        if marker and any(isinstance(p,dict) and isinstance(p.get('text'),str) and p['text'].startswith(marker) for p in parts):
            boundary=index
    # Older callers without a marker retain their latest user input. Production
    # always supplies an application-owned marker, including multi-step tools.
    if boundary is None:
        boundary=next((i for i in range(len(entries)-1,-1,-1) if isinstance(entries[i],dict) and entries[i].get('role')=='user'),len(entries))
    removed=0
    text_type='input_text' if protocol=='openai-responses' else 'text'
    def strip(value):
        nonlocal removed
        if isinstance(value,dict):
            if value.get('type') in _MEDIA_TYPES:
                removed+=1
                return {'type':text_type,'text':'[前轮媒体像素已省略；需要明确来源重新读取，不能声称本轮看过。]'}
            return {k:strip(v) for k,v in value.items()}
        if isinstance(value,list): return [strip(v) for v in value]
        return value
    for index,entry in enumerate(entries):
        if include_current_media and index>=boundary: continue
        entries[index]=strip(entry)
        entry=entries[index]
        if not isinstance(entry,dict): continue
        field='output' if entry.get('type')=='function_call_output' else 'content' if entry.get('role')=='tool' else None
        if field and isinstance(entry.get(field),str):
            try: decoded=json.loads(entry[field])
            except (ValueError,TypeError): continue
            previous=removed
            decoded=strip(decoded)
            if removed>previous: entry[field]=json.dumps(decoded,ensure_ascii=False,separators=(',',':'))
    return removed


def safe_closed_items(items: list) -> list:
    """Bootstrap successful exchanges, excluding rejected/pending/unknown turns."""
    kept=[];turn=[]
    def retain():
        if not turn or turn[-1].get('role')!='assistant': return
        assistant=turn[-1]
        status=assistant.get('status')
        payload=assistant
        if isinstance(assistant.get('content'),str):
            try: payload=json.loads(assistant['content'])
            except (ValueError,TypeError): payload={}
            if not isinstance(payload,dict): payload={}
            if status is None: status=payload.get('status')
        if status not in _FAILED and _safe_trace(payload.get('trace', [])):
            kept.extend(deepcopy(turn))
    for item in items:
        if not isinstance(item,dict): continue
        if item.get('role')=='user' and turn:
            retain();turn=[]
        turn.append(item)
    retain()
    return kept


def should_rotate_context(checkpoint: dict) -> bool:
    return checkpoint.get('context_reusable') is False or checkpoint.get('last_status') in _FAILED


def context_reusable(result) -> bool:
    return result.status not in _FAILED and _safe_trace(result.trace)


def failure_notice_text(reason: str) -> str:
    if reason=='model_image_input_rejected':
        return '这次图片被模型接口拒绝，暂时无法处理；可以继续普通文字聊天。'
    if reason=='model_input_rejected':
        return '这次输入被模型接口拒绝，暂时无法处理；可以换个普通话题继续聊天。'
    return '这次回复没能完成，稍后再试一下。'
