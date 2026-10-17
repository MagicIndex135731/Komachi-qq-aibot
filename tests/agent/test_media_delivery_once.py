"""Cross-turn media delivery claims reuse real receipts, never platform writes."""
import asyncio
from datetime import UTC, datetime

import pytest

from app.agent.media_delivery import send_media_once
from app.agent.tool_agent import AgentTool, ToolOutcome
from app.agent.tool_dispatch import ToolDispatchBoundary
from app.storage.agent_effects import AgentToolEffectRepository


def repository(engine, account=900000101):
    return AgentToolEffectRepository(engine, account_id=account, clock=lambda: datetime(2026, 1, 1, tzinfo=UTC))


async def deliver(repo, invoke, *, source=10, scope='private:900000102', **args):
    return await send_media_once(repo, scope=scope, source_id=source, operation='send_private',
        arguments=args or {'artifact_ids': ['art-a', 'art-b']}, require_valid=lambda: None, invoke=invoke)


async def test_receipt_survives_restart_caption_order_and_parameter_aliases(sqlite_engine):
    calls = []
    async def send():
        calls.append(True)
        return ToolOutcome({}, 'sent', 'real-receipt')
    first = await deliver(repository(sqlite_engine), send)
    again = await deliver(repository(sqlite_engine), send, scope='owner:private:900000102',
        text='different caption', media_artifact_ids=['art-b', 'art-a'])
    auto = await deliver(repository(sqlite_engine), send, media=['art-a', 'art-b'])
    assert len(calls) == 1 and first.receipt_id == again.receipt_id == auto.receipt_id
    assert again.data['already_delivered'] and auto.data['replay_allowed'] is False
    assert repository(sqlite_engine).recent('private:900000102') == ()  # No second quota charge.
    await deliver(repository(sqlite_engine), send, source=11)
    await deliver(repository(sqlite_engine), send, scope='private:900000103')
    await deliver(repository(sqlite_engine, 900000104), send)
    assert len(calls) == 4


async def test_concurrent_claim_has_one_sender_and_reserved_crash_is_not_replayed(sqlite_engine):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def send():
        calls.append(True); entered.set(); await release.wait()
        return ToolOutcome({}, 'sent', 'receipt')
    first = asyncio.create_task(deliver(repository(sqlite_engine), send))
    await entered.wait()
    second = await deliver(repository(sqlite_engine), send)
    assert second.status == 'uncertain' and second.data['previous_status'] == 'reserved'
    assert len(calls) == 1
    release.set(); assert (await first).status == 'sent'
    assert (await deliver(repository(sqlite_engine), send)).data['already_delivered']


@pytest.mark.parametrize('interrupted', [False, True])
async def test_unknown_or_cancelled_delivery_cannot_be_replayed(sqlite_engine, interrupted):
    calls = []
    async def send():
        calls.append(True)
        if interrupted:raise asyncio.CancelledError()
        return ToolOutcome({'error': 'delivery_result_unknown'}, 'uncertain')
    if interrupted:
        with pytest.raises(asyncio.CancelledError):await deliver(repository(sqlite_engine), send)
    else:assert (await deliver(repository(sqlite_engine), send)).status == 'uncertain'
    result = await deliver(repository(sqlite_engine), send, media_artifact_ids=['art-b','art-a'])
    assert result.status == 'uncertain' and len(calls) == 1 and result.data['replay_allowed'] is False


async def test_definite_failure_can_be_corrected_once_but_not_retried_forever(sqlite_engine):
    calls = []
    async def send():
        calls.append(True)
        return ToolOutcome({}, 'failed') if len(calls) == 1 else ToolOutcome({}, 'sent', 'receipt')
    assert (await deliver(repository(sqlite_engine), send)).status == 'failed'
    assert (await deliver(repository(sqlite_engine), send)).status == 'sent'
    assert (await deliver(repository(sqlite_engine), send)).data['already_delivered']
    assert len(calls) == 2
    async def fail():calls.append(True); return ToolOutcome({}, 'blocked')
    for _ in range(3):await deliver(repository(sqlite_engine), fail, source=20)
    assert len(calls) == 4


async def test_claim_requires_receipt_and_does_not_intercept_plain_text(sqlite_engine):
    calls = []
    async def missing():calls.append(True); return ToolOutcome({}, 'sent')
    first = await deliver(repository(sqlite_engine), missing)
    second = await deliver(repository(sqlite_engine), missing)
    assert first.status == second.status == 'uncertain' and len(calls) == 1
    async def text():calls.append(True); return ToolOutcome({}, 'sent', 'text-receipt')
    for _ in range(2):
        assert (await deliver(repository(sqlite_engine), text, text='ordinary text')).status == 'sent'
    assert len(calls) == 3


async def test_group_poll_return_and_auto_paths_share_claim_and_keep_real_dispatch_quota(sqlite_engine):
    class Guard:
        action_id = 'foreground'
        source_message_ids = (10,)
        def valid(self):return True
        def require_valid(self):pass
        def on_dispatch(self):pass
    calls = []
    async def run(wake, args, *, source=10):
        guard = Guard(); guard.action_id = wake
        boundary = ToolDispatchBoundary(engine=sqlite_engine, account_id=900000101,
            scope='group:900000102', guard=guard, clock=lambda: datetime(2026,1,1,tzinfo=UTC),
            delivery_source_id=source, interactive_group_send=True)
        async def send(arguments):
            boundary.on_dispatch(); calls.append(arguments)
            return ToolOutcome({}, 'sent', 'receipt-' + str(len(calls)))
        return await boundary.invoke(AgentTool('send_group','synthetic',{},send,'write'),args,wake)
    first = await run('foreground', {'media_artifact_ids':['art-a','art-b'],'text':'first'})
    second = await run('returned-result', {'artifact_ids':['art-b','art-a'],'text':'second'})
    third = await run('auto', {'media':['art-a','art-b'],'text':'automatic'})
    assert len(calls) == 1 and first.receipt_id == second.receipt_id == third.receipt_id
    assert len(repository(sqlite_engine).recent('group:900000102')) == 1
    assert (await run('new-request', {'artifact_ids':['art-a','art-b']}, source=11)).status == 'sent'
    assert len(calls) == 2


async def test_later_overlapping_bundle_sends_only_new_assets(sqlite_engine):
    repo=repository(sqlite_engine);calls=[]
    async def run(ids):
        args={'media_artifact_ids':ids}
        async def send():calls.append(list(args['media_artifact_ids']));return ToolOutcome({},'sent','receipt-'+str(len(calls)))
        return await send_media_once(repo,scope='private:900000102',source_id=10,operation='send_private',
            arguments=args,require_valid=lambda:None,invoke=send)
    await run(['art-a'])
    result=await run(['art-a','art-b'])
    assert calls==[['art-a'],['art-b']] and result.data['sent_artifact_ids']==['art-b']
    assert (await run(['art-b','art-a'])).data['already_delivered']


async def test_same_logical_asset_variant_is_not_a_new_delivery(sqlite_engine):
    calls=[]
    async def send():calls.append(True);return ToolOutcome({},'sent','receipt')
    for identifier in ['art-original','art-small']:
        result=await send_media_once(repository(sqlite_engine),scope='private:900000102',source_id=10,
            operation='send_private',arguments={'artifact_ids':[identifier]},require_valid=lambda:None,
            invoke=send,media_identity=lambda _:'same-source-media-1')
        assert result.status=='sent'
    assert len(calls)==1


async def test_unknown_asset_cannot_replay_inside_changed_bundle(sqlite_engine):
    calls=[]
    async def unknown():calls.append(True);return ToolOutcome({},'uncertain')
    await deliver(repository(sqlite_engine),unknown,artifact_ids=['art-a'])
    result=await deliver(repository(sqlite_engine),unknown,artifact_ids=['art-a','art-b'])
    assert result.status=='uncertain' and len(calls)==1


def test_pack_albums_preserves_order_and_sends_video_alone(tmp_path):
    from app.agent.media_delivery import pack_media_files
    paths=[]
    for i in range(6):
        p=tmp_path/str(i)
        with p.open('wb') as f:f.truncate(8*1024*1024)
        paths.append(('image' if i<5 else 'video',str(p)))
    assert pack_media_files(list('abcdef'),paths)==[list('abcd'),['e','f']]


async def test_aliases_deduplicate_same_source_variants_and_exact_bytes_across_sources(sqlite_engine):
    calls=[]
    aliases={'art-a':('source-a','bytes-one'),'art-a-small':('source-a','bytes-two'),
        'art-repost':('source-b','bytes-one'),'art-new':('source-c','bytes-three')}
    async def run(identifier):
        async def send():calls.append(identifier);return ToolOutcome({},'sent','receipt-'+str(len(calls)))
        return await send_media_once(repository(sqlite_engine),scope='private:900000102',source_id=10,
            operation='send_private',arguments={'artifact_ids':[identifier]},require_valid=lambda:None,invoke=send,
            media_identity=lambda i:aliases[i])
    for identifier in aliases:assert (await run(identifier)).status=='sent'
    assert calls==['art-a','art-new']


async def test_unmapped_legacy_subset_cannot_replay_after_upgrade(sqlite_engine):
    from app.agent.media_delivery import _send_bundle_once
    calls=[]
    async def send():calls.append(True);return ToolOutcome({},'sent','legacy-receipt')
    repo=repository(sqlite_engine)
    await _send_bundle_once(repo,scope='private:900000102',source_id=10,operation='send_private',
        arguments={'artifact_ids':['art-a','art-b']},require_valid=lambda:None,invoke=send)
    # Recreate the historical receipt shape, without changing original messages.
    from app.storage.db import session_scope
    from app.storage.agent_effects import AgentToolEffect
    from sqlalchemy import select
    with session_scope(sqlite_engine) as session:
        row=session.scalars(select(AgentToolEffect).where(AgentToolEffect.operation=='media_delivery_once')).one()
        row.receipt={'receipt_id':'legacy-receipt'}
    result=await deliver(repo,send,artifact_ids=['art-a'])
    assert result.status=='blocked' and result.data['error']=='legacy_media_delivery_subset_unverified' and len(calls)==1
