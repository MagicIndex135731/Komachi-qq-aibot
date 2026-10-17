"""Media jobs share provenance fences, never a model or main-session lock."""
import asyncio
from datetime import timedelta
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from app.agent.media_tasks import MediaTaskQueue, execute_web_media
from app.agent.tool_agent import ToolOutcome
from app.storage.db import session_scope
from app.storage.models import Job, Message
from test_research_queue import fixture_queue


def setup_queue(engine, **kwargs):
    research, payload, now = fixture_queue(engine)
    queue = MediaTaskQueue(engine, account_id=research.account_id, surface='private',
        execute=kwargs.get('execute', research.execute), deliver=kwargs.get('deliver', research.deliver),
        authorized=lambda _: True, clock=lambda: now[0])
    queue.running = True
    payload.update(operation='download_web_video', arguments={'ref_id': 'https://example.com/video'})
    return queue, payload, now


async def test_media_dedup_status_restart_and_no_replay_after_worker_loss(sqlite_engine):
    queue, payload, now = setup_queue(sqlite_engine)
    outcome = queue.submit(payload, parent_turn=42, before_commit=lambda: None)
    other = queue.submit(payload, parent_turn=42, before_commit=lambda: None)
    assert outcome.data['job_id'] == other.data['job_id']
    claimed = queue.claim()
    assert claimed[2] == 'running'
    now[0] += timedelta(seconds=901)
    assert queue.claim() is None
    guard = SimpleNamespace(require_valid=lambda: None)
    status = queue.status_tool(scope_id=payload['scope_id'], actor_id=payload['actor_id'], guard=guard)
    result = await status.handler({'job_id': outcome.data['job_id']})
    assert result.data['status'] == 'uncertain' and result.data['replay_allowed'] is False
    foreign = queue.status_tool(scope_id=payload['scope_id'], actor_id=payload['actor_id']+1, guard=guard)
    assert (await foreign.handler({'job_id': outcome.data['job_id']})).status == 'blocked'
    with session_scope(sqlite_engine) as session:
        session.get(Message, payload['source_id']).raw_json = {'delivery_state': 'deleted'}
    assert (await status.handler({'job_id': outcome.data['job_id']})).status == 'blocked'


async def test_cancelled_media_not_retried(sqlite_engine):
    async def interrupted(payload, guard):
        raise asyncio.CancelledError()
    queue, payload, _ = setup_queue(sqlite_engine, execute=interrupted)
    outcome = queue.submit(payload, parent_turn=42, before_commit=lambda: None)
    with pytest.raises(asyncio.CancelledError):
        await queue.tick()
    assert not await queue.tick()
    with session_scope(sqlite_engine) as session:
        assert session.get(Job, outcome.data['job_id']).status == 'uncertain'


async def test_disconnected_startup_does_not_claim_or_spend_queued_media_attempt(sqlite_engine):
    queue, payload, _ = setup_queue(sqlite_engine)
    outcome = queue.submit(payload, parent_turn=42, before_commit=lambda: None)
    queue.available = lambda: False
    assert not await queue.tick()
    with session_scope(sqlite_engine) as session:
        row = session.get(Job, outcome.data['job_id'])
        assert row.status == 'queued' and row.attempt_count == 0
    queue.available = lambda: True
    assert await queue.tick()
    with session_scope(sqlite_engine) as session:
        assert session.get(Job, outcome.data['job_id']).status == 'result_ready'


@pytest.mark.parametrize('state', ['sent', 'uncertain', 'deleted', 'foreign'])
async def test_confirmed_bot_reply_can_be_dependency_but_never_authority(sqlite_engine, state):
    from app.storage.repositories import UserRepository
    from app.agent.media_tasks import MediaJobGuard
    queue, payload, now = setup_queue(sqlite_engine)
    with session_scope(sqlite_engine) as session:
        author = queue.account_id if state != 'foreign' else queue.account_id+1
        UserRepository(session).upsert_user(user_id=author, nickname='synthetic bot', group_card='')
        session.flush()
        row = Message(platform_msg_id='synthetic-previous-reply', group_id=None, user_id=author,
            timestamp=now[0], plain_text='previous confirmed reply',
            raw_json={'direction': 'outbound', 'delivery_state': state if state != 'foreign' else 'sent'})
        session.add(row); session.flush(); identifier = row.id
    guard = MediaJobGuard(queue, 1, payload, 'running')
    assert guard.valid_source(identifier) is (state == 'sent')
    assert guard.valid_source(identifier, author=True) is False
    payload['sources'].append(identifier)
    outcome = queue.submit(payload, parent_turn=42, before_commit=lambda: None)
    assert await queue.tick()
    with session_scope(sqlite_engine) as session:
        assert session.get(Job, outcome.data['job_id']).status == ('result_ready' if state == 'sent' else 'blocked')


async def test_async_batch_persists_pixels_and_scoped_result_without_model(sqlite_engine, tmp_path, monkeypatch):
    from test_autonomous_social import image_bytes
    from test_web_image_tools import Runner
    from app.providers.web_images import WebImageClient
    from app.agent.artifacts import AutonomousArtifactStore
    calls, delivered = [], []
    client = httpx.Client(transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(200, content=image_bytes())))
    provider = WebImageClient(http_client=client)
    monkeypatch.setattr('app.agent.web_images.WebImageClient', lambda _: provider)
    settings = SimpleNamespace(data_dir=tmp_path)
    async def execute(payload, guard):
        return await execute_web_media(payload, guard, engine=sqlite_engine, settings=settings, web_client=None, runner=Runner())
    async def deliver(payload, guard):
        guard.require_valid()
        delivered.append(payload['result'])
        return 'completed'
    queue, payload, _ = setup_queue(sqlite_engine, execute=execute, deliver=deliver)
    with session_scope(sqlite_engine) as session:
        row = session.get(Message, payload['source_id'])
        row.plain_text = '请打开 https://example.com/page 并读取所有图片'
        row.raw_json = {'message_type': 'private', 'message': [{'type': 'text', 'data': {'text': row.plain_text}}]}
    payload.update(operation='download_web_images', arguments={'image_ids': ['webimg-synthetic']},
        candidates={'webimg-synthetic': {'url': 'https://example.com/a.png', 'page_url': 'https://example.com/page', 'alt': '合成图'}})
    outcome = queue.submit(payload, parent_turn=42, before_commit=lambda: None)
    try:
        assert calls == [] and outcome.data['status'] == 'queued'
        assert await queue.tick() and len(calls) == 1 and not delivered
        assert await queue.tick() and len(calls) == 1
        data = delivered[0]['data']
        assert data['downloaded_count'] == 1 and data['all_requested_downloaded']
        assert len(json.dumps(delivered[0])) < 4000
        store = AutonomousArtifactStore(tmp_path / 'autonomous_artifacts', account_id=payload['account_id'],
            scope=payload['scope'], engine=sqlite_engine, source_message_ids=lambda: (payload['source_id'],))
        identifier = data['artifact_ids'][0]
        assert store.get(identifier).read_bytes() == image_bytes()
        assert store.provenance(identifier) == (payload['source_id'],)
    finally:
        client.close()


async def test_group_media_provider_wait_does_not_hold_main_lock_or_start_model(sqlite_engine, tmp_path, monkeypatch):
    from app.agent.social_contracts import SocialObservation
    from app.agent.social_policy import configured_social_jobs
    from app.agent.tool_agent import ToolTurnResult
    from test_autonomous_social import composed, ScriptedModel, add_source, NOW, FRIEND
    entered, release = threading.Event(), threading.Event()
    turns = []
    def provider():
        entered.set()
        assert release.wait(10)
    class Harness:
        def __init__(self, **kwargs):
            pass
        async def run(self, prompt, tools, guard, **kwargs):
            turns.append(prompt)
            if len(turns) == 1:
                result = await next(t for t in tools if t.name == 'generate_image').handler({'prompt': '合成猫'})
                assert result.data['status'] == 'queued'
            return ToolTurnResult('silent', reason='model_finished')
    monkeypatch.setattr('app.agent.harness_runtime.HarnessToolAgent', Harness)
    async with composed(sqlite_engine, tmp_path, ScriptedModel(), runtime='harness', image_callback=provider) as fixture:
        queue = fixture.executor.media_jobs
        queue.running = True
        job = configured_social_jobs(fixture.router.runtime)[0]
        async def main(identifier):
            source = add_source(sqlite_engine, platform_id=identifier)
            guard = SimpleNamespace(action_id=identifier, valid=lambda: True, require_valid=lambda: None, on_dispatch=lambda: None)
            return await fixture.executor.run_tools(SocialObservation(job, NOW, '小町', source_message_ids=(source,)),
                guard, interactive=True, target_user_id=FRIEND, target_message_id=identifier, direct_group_reply=True)
        await main('media-start')
        task = asyncio.create_task(queue.tick())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert (await asyncio.wait_for(main('media-new-chat'), timeout=5)).status == 'silent'
            assert len(turns) == 2 and not release.is_set()
            release.set()
            await task
            assert len(turns) == 2 and len(fixture.image_client.calls) == 1
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


def test_long_media_result_preserves_handles_and_partial_status():
    from app.agent.media_tasks import media_result
    ids=['art-'+str(i).zfill(32) for i in range(20)]
    result=media_result(ToolOutcome({'artifact_ids':ids,'post':{'text':'x'*20000},
        'all_requested_downloaded':False,'failures':[{'index':21,'error':'size_limit'}]}),
        {'operation':'download_web_images','task_id':'synthetic'})
    assert result['status']=='ok' and result['data']['artifact_ids']==ids
    assert result['data']['all_requested_downloaded'] is False
    assert len(json.dumps(result))<12000


def test_long_media_sources_compact_without_dropping_ready_handles():
    from app.agent.media_tasks import media_result
    media=[{'artifact_id':'art-'+str(i).zfill(32),'logical_media_id':'media-'+str(i),
        'source_ref':'https://example.com/'+str(i)+'?'+'x'*1900,'status':'ready'} for i in range(20)]
    result=media_result(ToolOutcome({'media_artifact_ids':[i['artifact_id'] for i in media],
        'media':media,'post':{'text':'x'*16000}}),{'operation':'download_web_images','task_id':'synthetic'})
    assert len(json.dumps(result))<12000
    assert len(result['data']['media_artifact_ids'])==20 and result['status']=='ok'


def test_compacted_video_result_retains_quality_and_variant_rejection_details():
    from app.agent.media_tasks import media_result
    details={'stage':'video_variant_selection','variant_count':2,'byte_limit':128*1024*1024,
        'duration_limit_seconds':1200,'rejected_by_reason':{'duration_limit':2}}
    value=media_result(ToolOutcome({'post':{'text':'x'*20000},'media_artifact_ids':['art-video'],
        'media':[{'artifact_id':'art-video','kind':'video','quality':'lower_native','fallback_reason':'higher_version_over_budget_or_failed'}],
        'video_quality_fallbacks':[{'index':1,'quality':'lower_native','reason':'higher_version_over_budget_or_failed'}],
        'failures':[{'index':2,'error':'no_mp4_stream_within_budget','details':details}]}),
        {'operation':'download_twitter_media','task_id':'synthetic'})
    assert value['data']['media'][0]['quality']=='lower_native'
    assert value['data']['failures'][0]['details']==details
    assert value['data']['video_quality_fallbacks'] and len(json.dumps(value))<12000
