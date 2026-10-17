"""Resolve ordered mixed media through scoped artifact and cache validation."""
from collections.abc import Awaitable, Callable
from hashlib import sha256
import json
import re

from app.agent.tool_agent import ToolOutcome
from app.core.downloaded_media import DownloadedMediaCache
from app.storage.agent_effects import AgentToolEffectRepository


def canonical_delivery_scope(scope):
    match=re.fullmatch(r'owner:(group|private):([1-9][0-9]*)(?::[1-9][0-9]*)?',scope)
    return match[1]+':'+match[2] if match else scope.removeprefix('owner:')


async def send_media_once(repository: AgentToolEffectRepository, *, scope: str, source_id: int | None,
                          operation: str, arguments: dict, require_valid: Callable[[], None],
                          invoke: Callable[[], Awaitable[ToolOutcome]], media_identity=None) -> ToolOutcome:
    """Claim individual logical assets before the existing bundle/send fences."""
    identifiers=[]
    for key in ('artifact_ids','media_artifact_ids','media'):
        values=arguments.get(key,[])
        if not isinstance(values,(list,tuple)):return await invoke()
        identifiers.extend(values)
    if arguments.get('video_artifact_id'):identifiers.append(arguments['video_artifact_id'])
    if (operation not in {'send_group','send_private'} or type(source_id) is not int or source_id<=0
            or not 1<=len(identifiers)<=20 or any(not isinstance(i,str) or not 1<=len(i)<=64 for i in identifiers)):
        return await invoke()
    require_valid()
    canonical_scope=canonical_delivery_scope(scope)
    scope_hash=sha256(canonical_scope.encode()).hexdigest()
    original_digest=sha256(json.dumps([source_id,sorted(set(identifiers))],separators=(',',':')).encode()).hexdigest()
    hashes={scope_hash,sha256(scope.removeprefix('owner:').encode()).hexdigest()}
    for legacy_hash in hashes:
        for legacy in repository.media_delivery_claims('media-delivery:'+legacy_hash,f'media-source:{legacy_hash}:{source_id}'):
            receipt=legacy.get('receipt')
            if isinstance(receipt,dict) and receipt.get('asset_claim_version')==2:continue
            receipt_id=receipt.get('receipt_id') if isinstance(receipt,dict) else receipt if isinstance(receipt,str) else None
            if legacy['payload_hash']==original_digest and legacy['status']=='sent' and receipt_id:
                return ToolOutcome({'already_delivered':True,'replay_allowed':False,'sent_artifact_ids':[],
                    'confirmed_artifact_ids':identifiers,'legacy_receipt':True},'sent',receipt_id)
            # An old bundle hash cannot prove a subset, or buy another attempt
            # after the claim namespace changes. Keep its actual receipt/fence.
            return ToolOutcome({'error':'legacy_media_delivery_subset_unverified','replay_allowed':False,'previous_status':legacy['status'],
                'guidance':'旧版原请求只有整批认领，无法安全证明新增子集；保留已有回执，需要新请求继续。'},
                'uncertain' if legacy['status'] in {'reserved','dispatching','uncertain'} else 'blocked')
    claims=[];new=[];seen=set();previous_receipt=None
    for identifier in identifiers:
        identity=media_identity(identifier) if media_identity else identifier
        identities=(identity,) if isinstance(identity,str) else tuple(identity)
        if any(value in seen for value in identities):continue
        seen.update(identities)
        item_claims=[];duplicate_receipt=None
        for value in identities:
            require_valid()
            digest=sha256(json.dumps([source_id,value],separators=(',',':')).encode()).hexdigest()
            record=repository.reserve('media-asset:'+scope_hash,run_id=f'media-asset:{scope_hash}:{source_id}',
                tool_call_id=operation+':'+digest,operation='media_asset_delivery_once',payload_hash=digest,sources=(source_id,))
            if record['fresh']:item_claims.append(record)
            elif record['status']=='sent' and isinstance(record.get('receipt'),dict) and record['receipt'].get('receipt_id'):
                duplicate_receipt=record['receipt']['receipt_id']
            else:
                for claim in [*claims,*item_claims]:repository.finish(claim['action_id'],'blocked',reason='sibling_delivery_unresolved')
                return ToolOutcome({'error':'media_delivery_already_claimed','previous_status':record['status'],
                    'replay_allowed':False},'uncertain' if record['status'] in {'reserved','dispatching','uncertain'} else 'blocked')
        if duplicate_receipt:
            previous_receipt=duplicate_receipt
            for claim in item_claims:
                repository.mark_dispatching(claim['action_id'],daily_limit=None)
                repository.finish(claim['action_id'],'sent',receipt={'receipt_id':duplicate_receipt},reason='same_media_receipt')
        else:claims.extend(item_claims);new.append(identifier)
    if not new:
        return ToolOutcome({'already_delivered':True,'replay_allowed':False,
            'sent_artifact_ids':[],'confirmed_artifact_ids':identifiers},'sent',previous_receipt)
    # invoke closes over this same validated dict. Filter all aliases consistently.
    for key in ('artifact_ids','media_artifact_ids','media'):
        if key in arguments:arguments[key]=list(dict.fromkeys(i for i in arguments[key] if i in new))
    if arguments.get('video_artifact_id') not in new:arguments.pop('video_artifact_id',None)
    try:
        result=await _send_bundle_once(repository,scope=scope,source_id=source_id,operation=operation,
            arguments=arguments,require_valid=require_valid,invoke=invoke)
    except BaseException:
        for record in claims:
            repository.mark_dispatching(record['action_id'],daily_limit=None)
            repository.finish(record['action_id'],'uncertain',reason='media_delivery_interrupted')
        raise
    status=result.status if result.status in {'sent','uncertain','failed','blocked'} else 'failed'
    if status=='sent' and not result.receipt_id:status='uncertain'
    for record in claims:
        if status in {'sent','uncertain'}:repository.mark_dispatching(record['action_id'],daily_limit=None)
        repository.finish(record['action_id'],status,receipt={'receipt_id':result.receipt_id},reason='media_asset_settled')
    return ToolOutcome({**result.data,'sent_artifact_ids':new if status=='sent' else [],
        'confirmed_artifact_ids':identifiers if status=='sent' else [],
        'logical_media_count':len(new)},status,result.receipt_id)


def artifact_delivery_identities(store,identifier):
    description=store.describe(identifier) or {}
    identities=[description.get('logical_media_id',identifier)]
    content=description.get('content_sha256') or (description.get('hash') if description.get('origin')=='web_page_image' else None)
    if content:identities.append('bytes:'+content)
    return tuple(identities)


async def _send_bundle_once(repository: AgentToolEffectRepository, *, scope: str, source_id: int | None,
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
    canonical_scope = canonical_delivery_scope(scope)
    scope_hash = sha256(canonical_scope.encode()).hexdigest()
    digest = sha256(json.dumps([source_id, sorted(set(identifiers))], separators=(',', ':')).encode()).hexdigest()
    # Sibling assets may arrive while this bundle is still awaiting a receipt.
    # The version must exist in the reservation transaction, not only at finish.
    claim_metadata = {'asset_claim_version': 2, 'artifact_ids': identifiers}
    record = repository.reserve('media-delivery:' + scope_hash,
        run_id=f'media-source:{scope_hash}:{source_id}', tool_call_id=operation + ':' + digest,
        operation='media_delivery_once', payload_hash=digest, sources=(source_id,), initial_receipt=claim_metadata)
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
        repository.finish(record['action_id'], 'uncertain', receipt=claim_metadata, reason='media_delivery_interrupted')
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
        receipt={**claim_metadata, 'receipt_id': result.receipt_id}, reason='media_delivery_settled')
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


def pack_media_files(identifiers, files):
    """Pack small media at 20/32MiB; a larger valid video occupies its own packet."""
    from pathlib import Path
    packets=[];pending=[];size=0
    for identifier,(kind,path) in zip(identifiers,files,strict=True):
        length=Path(path).stat().st_size
        if kind=='video' and length>32*1024*1024:
            if pending:packets.append(pending);pending=[];size=0
            packets.append([identifier])
        else:
            if pending and (len(pending)>=20 or size+length>32*1024*1024):
                packets.append(pending);pending=[];size=0
            if kind=='image' and length>8*1024*1024:raise ValueError('image_exceeds_artifact_budget')
            pending.append(identifier);size+=length
    if pending:packets.append(pending)
    return packets
