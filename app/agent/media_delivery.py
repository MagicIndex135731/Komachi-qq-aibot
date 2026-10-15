"""Resolve ordered mixed media through scoped artifact and cache validation."""
from collections.abc import Awaitable, Callable
from hashlib import sha256
import json

from app.agent.tool_agent import ToolOutcome
from app.core.downloaded_media import DownloadedMediaCache
from app.storage.agent_effects import AgentToolEffectRepository


async def send_media_once(repository: AgentToolEffectRepository, *, scope: str, source_id: int | None,
                          operation: str, arguments: dict, require_valid: Callable[[], None],
                          invoke: Callable[[], Awaitable[ToolOutcome]]) -> ToolOutcome:
    """Share one durable delivery claim across polling, return and auto turns.

    This claim has its own scope and operation so the real send keeps its
    existing authorization, dispatch fence and quota exactly once. Reserved
    claims also fence concurrent/crashed callers; a receipt is required before
    reusing success. Changing captions or artifact order cannot buy a resend.
    """
    if operation not in {'send_group', 'send_private'} or type(source_id) is not int or source_id <= 0:
        return await invoke()
    identifiers = []
    for key in ('artifact_ids', 'media_artifact_ids', 'media'):
        values = arguments.get(key, [])
        if not isinstance(values, (list, tuple)):
            return await invoke()
        identifiers.extend(values)
    if arguments.get('video_artifact_id'):
        identifiers.append(arguments['video_artifact_id'])
    if (not identifiers or len(identifiers) > 20
            or any(not isinstance(value, str) or not 1 <= len(value) <= 64 for value in identifiers)):
        return await invoke()
    require_valid()
    canonical_scope = scope.removeprefix('owner:')
    scope_hash = sha256(canonical_scope.encode()).hexdigest()
    digest = sha256(json.dumps([source_id, sorted(set(identifiers))], separators=(',', ':')).encode()).hexdigest()
    record = repository.reserve('media-delivery:' + scope_hash,
        run_id=f'media-source:{scope_hash}:{source_id}', tool_call_id=operation + ':' + digest,
        operation='media_delivery_once', payload_hash=digest, sources=(source_id,))
    if not record['fresh']:
        receipt = record.get('receipt') or {}
        receipt_id = receipt.get('receipt_id') if isinstance(receipt, dict) else None
        if record['status'] == 'sent' and isinstance(receipt_id, str) and receipt_id:
            return ToolOutcome({'already_delivered': True, 'delivery_id': record['action_id'],
                'replay_allowed': False, 'guidance': '同一原请求的这批媒体已有真实发送回执；复用回执结束，不再次发送。'},
                'sent', receipt_id)
        return ToolOutcome({'error': 'media_delivery_already_claimed',
            'previous_status': record['status'], 'replay_allowed': False,
            'guidance': '这批媒体已由另一执行路径处理，结果待定或未确认；不要重发。'},
            'uncertain' if record['status'] in {'reserved', 'dispatching', 'uncertain'} else 'blocked')
    try:
        result = await invoke()
    except BaseException:
        # The real dispatch fence owns precise network state. Across crashes
        # or cancellation we conservatively retain this cross-turn claim.
        repository.mark_dispatching(record['action_id'], daily_limit=None)
        repository.finish(record['action_id'], 'uncertain', reason='media_delivery_interrupted')
        raise
    status = result.status
    if status == 'sent' and not result.receipt_id:
        status = 'uncertain'
        result = ToolOutcome({'error': 'media_delivery_missing_receipt', 'replay_allowed': False}, status)
    if status in {'sent', 'uncertain'}:
        # Settle only after the actual send. A crash before settlement leaves
        # a reserved claim, which cannot be replayed. Definite pre-send errors
        # remain repairable once through the repository's existing bound.
        repository.mark_dispatching(record['action_id'], daily_limit=None)
    repository.finish(record['action_id'], status if status in {'sent', 'uncertain', 'failed', 'blocked'} else 'failed',
        receipt={'receipt_id': result.receipt_id}, reason='media_delivery_settled')
    return result


def resolve_media_bundle(store, settings, identifiers, *, before_read):
    if (not isinstance(identifiers, list) or not 1 <= len(identifiers) <= 20
            or any(not isinstance(value, str) for value in identifiers) or len(set(identifiers)) != len(identifiers)):
        raise ValueError('invalid_media_bundle')
    cache = DownloadedMediaCache.for_store(store, settings)
    cache.pin(identifiers)
    files = []
    for identifier in identifiers:
        path = before_read(identifier)
        description = store.describe(identifier)
        if description and description.get('origin') == 'web_page_video':
            path, _ = cache.video(store, identifier)
            kind = 'video'
        elif path is not None and path.suffix in {'.png', '.jpg', '.webp', '.gif'}:
            kind = 'image'
        else:
            raise ValueError('invalid_media_bundle_artifact')
        files.append((kind, str(path)))
    return tuple(files)
