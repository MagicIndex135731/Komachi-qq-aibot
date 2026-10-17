"""Public-only field projection shared by dispatch, tools and bounded backfill."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import re

from app.storage.public_social import OPERATIONS


def public_scope(scope):
    return isinstance(scope, str) and (scope == 'public' or bool(re.fullmatch(r'owner:public:[1-9][0-9]*', scope)))


def capture_public_tool(repository, *, scope, run_id, call_id, name, arguments, outcome,
                        effect_id='', created_at=None, before_commit=None):
    if not public_scope(scope) or name not in OPERATIONS:
        return []
    if name in {'remember_intention', 'save_work_note', 'assess_qzone_post'} and scope != 'public':
        return []  # Owner instructions and task-local notes remain private.
    if name == 'remember_intention':
        from app.agent.patrol_context import routine_wait
        if routine_wait(arguments):
            return []
    args, data = arguments, outcome.data
    key = ':'.join((scope, run_id, call_id))
    common = dict(operation=name, created_at=created_at, before_commit=before_commit)
    if name in {'read_friend_feed', 'read_friend_posts'}:
        if outcome.status != 'ok':
            return []
        return [repository.save(event_key=key + ':' + str(p['owner_id']) + ':' + p['post_id'],
            kind='observation', status='observed', owner_id=p['owner_id'], post_id=p['post_id'],
            payload={'post_text': p.get('text', ''), 'published_at': p.get('published_at', ''),
                     'image_count': p.get('image_count', 0)}, **common) for p in data.get('posts', [])[:20]]
    owner, post = args.get('owner_id', 0), args.get('post_id', '')
    if name == 'comment_qzone':
        return [repository.save(event_key=key, kind='comment', status=outcome.status,
            owner_id=owner, post_id=post, payload={'text': args.get('text', ''),
            'attachment_count': len(args.get('artifact_ids', []))}, effect_id=effect_id,
            payload_hash=sha256(json.dumps(args, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            result_reason=data.get('reason', data.get('error', '')), **common)]
    if outcome.status != 'ok' or data.get('saved') is False:
        return []
    kind = 'observation' if name == 'observe_post_images' else 'thought'
    status = args.get('decision', 'ok') if name == 'assess_qzone_post' else 'ok'
    payload = (dict(args) if name != 'observe_post_images' else
               {k: data[k] for k in ('viewed_image_count', 'unseen_image_count') if k in data})
    return [repository.save(event_key=key, kind=kind, status=status, owner_id=owner, post_id=post,
                            payload=payload, **common)]


def import_public_native(repository, path, effects, *, now, max_bytes=2_000_000):
    """Read tool events only. Never ingest assistant reasoning or monitor traces.

    The caller must bind path to a validated own-public session directory.
    Success and dispatch status are taken from an exact matching ledger payload.
    """
    from app.agent.tool_agent import ToolOutcome
    if path.stat().st_size > max_bytes:
        raise ValueError('public native history exceeds import bound')
    calls, imported = {}, []
    for line in path.read_text(encoding='utf-8').splitlines()[-4096:]:
        event = json.loads(line)
        if event.get('type') not in {'tool/call', 'tool/result'}:
            continue
        stamp = datetime.fromtimestamp(event.get('time', 0) / 1000, UTC)
        if not now - timedelta(days=repository.RETENTION_DAYS) <= stamp <= now + timedelta(seconds=60):
            continue
        data = event.get('data', {})
        if event['type'] == 'tool/call':
            if data.get('name') not in OPERATIONS:
                continue
            args = json.loads(data['arguments']) if isinstance(data.get('arguments'), str) else data.get('arguments')
            if isinstance(args, dict):
                calls[data['callId']] = (data['name'], args, stamp)
            continue
        message = data.get('message', {})
        call_id = message.get('source', {}).get('callId')
        call = calls.pop(call_id, None)
        if call is None:
            continue
        name, args, called_at = call
        results = [p.get('text', '') for c in message.get('content', []) if c.get('type') == 'tool-result'
                   for p in c.get('content', []) if p.get('type') == 'text']
        if not results:
            continue
        result, _ = json.JSONDecoder().raw_decode(results[0])
        effect_id = ''
        if name == 'comment_qzone':
            digest = sha256(json.dumps(args, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            candidates = [e for e in effects if e.payload_hash == digest and e.operation == name
                          and abs((e.created_at.replace(tzinfo=UTC) - called_at).total_seconds()) < 180]
            if len(candidates) != 1:
                continue
            effect_id = candidates[0].action_id
        imported.extend(capture_public_tool(repository, scope='public', run_id='legacy:' + str(path),
            call_id=call_id, name=name, arguments=args,
            outcome=ToolOutcome(result.get('data', {}), result.get('status', 'unverified')),
            effect_id=effect_id, created_at=called_at))
    return imported
