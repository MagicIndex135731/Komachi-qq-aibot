"""All-excluded web candidates recover without another model decision."""
import time

import pytest

from app.agent.request_task import RequestTaskControl
from app.agent.tool_agent import ToolOutcome
from tests.agent.test_harness_runtime import bridge_factory, responses_sse, tool
from tests.agent.test_request_budget import workspace


URL = 'https://example.test/gallery'


def setup(bridge_factory, tmp_path, *, surface='private'):
    bridge, requests, _ = bridge_factory(lambda *_: responses_sse(), require_reply=True)
    w = workspace(tmp_path, scope=f'{surface}:900000101')
    bridge.owner.task_workspace = w
    bridge.control = RequestTaskControl(w, capability_version='synthetic')
    bridge.control.observe('list_web_images', {'ref_id': URL}, ToolOutcome({
        'page_url': URL, 'images': [{'image_id': 'webimg-a', 'role': 'page_image'},
                                    {'image_id': 'webimg-b', 'role': 'page_image'}]}), 'list')
    bridge.control.describe({'media_selection': {'candidate_reviews': [
        {'ref_id': URL, 'decision': 'excluded', 'reason': '模型认为题材不够典型'}]}})
    calls = []

    async def download(args):
        calls.append(('download', args))
        return ToolOutcome({'job_id': 1, 'status': 'queued',
                            'delivery': 'automatic_delivery_to_original_request'})

    async def send(args):
        calls.append(('send', args))
        return ToolOutcome({}, 'sent', 'synthetic-receipt')

    from app.agent.media_tasks import download_delivery_properties
    bridge.tools = {
        'download_web_images': tool('download_web_images', download, properties={
            'image_ids': {'type': 'array', 'items': {'type': 'string'}},
            **download_delivery_properties()}, required=('image_ids',)),
        f'send_{surface}': tool(f'send_{surface}', send, effect='write', properties={
            'text': {'type': 'string'}, 'task_status': {'type': 'string'},
            'artifact_ids': {'type': 'array', 'items': {'type': 'string'}},
            'as_forward': {'type': 'boolean'}}, required=('text',))}
    return bridge, requests, calls


@pytest.mark.parametrize('surface', ['group', 'private'])
async def test_all_excluded_terminal_send_hands_off_download_once(bridge_factory, tmp_path, surface):
    bridge, requests, calls = setup(bridge_factory, tmp_path, surface=surface)
    args = {'name': f'send_{surface}', 'arguments': {
        'text': '没找到合适的图片', 'task_status': 'partial'}, 'call_id': 'finish'}
    await bridge.call_tool(args)
    await bridge.call_tool(args)
    await bridge.finalize_media()
    assert [name for name, _ in calls] == ['download']
    assert calls[0][1]['image_ids'] == ['webimg-a', 'webimg-b']
    assert calls[0][1]['delivery_mode'] == 'auto'
    assert '候选' in calls[0][1]['delivery_text']
    assert bridge.auto_handoff and not requests


@pytest.mark.parametrize('state', ['selected', 'delivered', 'ready', 'pending',
                                  'uncertain', 'cancelled', 'revoked', 'expired', 'budget'])
async def test_fallback_preserves_admission_and_never_repeats_media(bridge_factory, tmp_path, state):
    bridge, requests, calls = setup(bridge_factory, tmp_path)
    c = bridge.control
    if state == 'selected': next(iter(c.candidate_reviews.values()))['decision'] = 'selected'
    if state == 'delivered': c.delivered_artifacts.add('old')
    if state == 'ready': c.artifacts.add('ready')
    if state == 'pending': c.jobs[8] = 'running'
    if state == 'uncertain': c.unknown = True
    if state == 'cancelled': bridge.cancelled.set()
    if state == 'revoked': bridge.guard.require_valid = lambda: (_ for _ in ()).throw(RuntimeError('revoked'))
    if state == 'expired': bridge.finalization_deadline = time.monotonic() - 1
    if state == 'budget':
        from app.agent.request_budget import RequestBudget
        c.workspace.media_budget.path.unlink()
        c.workspace.media_budget=RequestBudget(c.workspace,limits={})
        c.workspace.media_budget.spend('tool_calls',64)
    await bridge.finalize_media()
    assert not calls and not requests


async def test_model_budget_exhaustion_can_still_queue_local_fallback(bridge_factory, tmp_path):
    bridge, requests, calls = setup(bridge_factory, tmp_path)
    bridge.model_calls = bridge.owner.max_steps
    bridge.stop_reason = 'model_budget_exhausted'
    await bridge.finalize_media()
    assert len(calls) == 1 and bridge.auto_handoff and not requests


@pytest.mark.parametrize('kind', ['image', 'video', 'mixed', 'wrong_type', 'old', 'wrong_site', 'wrong_author'])
async def test_post_fallback_preserves_explicit_constraints(bridge_factory, tmp_path, kind):
    bridge, requests, calls = setup(bridge_factory, tmp_path)
    c = bridge.control
    c.resource_urls.clear(); c.web_image_candidates.clear(); c.candidate_reviews.clear()
    url = 'https://x.com/synthetic/status/900000000000000001'
    roles = ['post_image'] if kind == 'image' else ['post_video']
    if kind == 'mixed': roles = ['post_image', 'post_video']
    c.requested_media_types = {'image'} if kind == 'wrong_type' else {'image', 'video'}
    c.observe('inspect_web_resources', {'ref_ids': [url]}, ToolOutcome({'resources': [{
        'source_url': url, 'post_id': '900000000000000001', 'identity_status': 'matched',
        'author': 'synthetic', 'published_at': '1970-01-01T00:01:00Z', 'media_roles': roles,
        'media_count': 2}]}), 'inspect')
    c.describe({'media_selection': {'candidate_reviews': [
        {'ref_id': url, 'decision': 'excluded', 'reason': '模型主观排除'}]}})
    if kind == 'old':
        from datetime import UTC, datetime
        c.published_after = datetime.fromtimestamp(100, UTC)
        c.publication_filter['provenance'] = 'explicit_user'
    if kind == 'wrong_site': c.requested_sites = {'xhs'}
    if kind == 'wrong_author':
        c.publisher_proof = 'explicit synthetic proof'; c.explicit_publisher_account = 'another'

    async def download(args):
        calls.append(('download', args))
        return ToolOutcome({'job_id': 2, 'status': 'queued',
                            'delivery': 'automatic_delivery_to_original_request'})
    from app.agent.media_tasks import download_delivery_properties
    bridge.tools['download_twitter_media'] = tool('download_twitter_media', download, properties={
        'ref_id': {'type': 'string'}, **download_delivery_properties()}, required=('ref_id',))
    await bridge.finalize_media()
    assert len(calls) == (1 if kind in {'image', 'video', 'mixed'} else 0)
    assert not requests


async def test_generic_unknown_date_never_bypasses_explicit_date(bridge_factory, tmp_path):
    bridge, requests, calls = setup(bridge_factory, tmp_path)
    bridge.control.publication_filter['provenance'] = 'explicit_user'
    await bridge.finalize_media()
    assert not calls and not requests


async def test_claim_survives_wake_but_does_not_cross_source_or_scope(bridge_factory, tmp_path):
    bridge, _, calls = setup(bridge_factory, tmp_path)
    await bridge.finalize_media()
    assert len(calls) == 1
    again, requests, calls = setup(bridge_factory, tmp_path)
    await again.finalize_media()
    assert not calls and not requests
    w = workspace(tmp_path, scope='group:900000102')
    assert w.claim_media_fallback() and not w.claim_media_fallback()
    w.source_id += 1; w.task_id = 'synthetic-new-source'
    assert w.claim_media_fallback()


async def test_failed_download_is_not_retried_and_model_call_policy_stays_review(bridge_factory, tmp_path):
    bridge, requests, calls = setup(bridge_factory, tmp_path)
    bridge.control.media_selection.update(selection_scope='topic', subject='旅行')
    prepared = bridge.control.prepare_tool_arguments('download_web_images', {
        'image_ids': ['webimg-a'], 'delivery_mode': 'auto'})
    assert prepared['delivery_mode'] == 'review'

    async def failed(args):
        calls.append(('failed', args))
        return ToolOutcome({'error': 'synthetic_download_failure'}, 'failed')
    from dataclasses import replace
    bridge.tools['download_web_images'] = replace(bridge.tools['download_web_images'], handler=failed)
    await bridge.finalize_media()
    await bridge.finalize_media()
    assert len(calls) == 1 and not bridge.auto_handoff and not requests


async def test_explicit_original_quality_preserved(bridge_factory, tmp_path):
    bridge, _, calls = setup(bridge_factory, tmp_path)
    bridge.control.workspace.request += '，只要原图'
    await bridge.finalize_media()
    assert calls[0][1]['quality'] == 'original_only'


async def test_uncertain_fallback_stops_before_the_next_candidate(bridge_factory, tmp_path):
    from dataclasses import replace
    bridge, requests, calls = setup(bridge_factory, tmp_path)
    c = bridge.control
    second = 'https://example.test/other'
    c.observe('list_web_images', {'ref_id': second}, ToolOutcome({
        'page_url': second, 'images': [{'image_id': 'webimg-other'}]}), 'other')

    async def uncertain(args):
        calls.append(('uncertain', args))
        return ToolOutcome({'error': 'synthetic_lost_ack'}, 'uncertain')
    bridge.tools['download_web_images'] = replace(bridge.tools['download_web_images'], handler=uncertain)
    await bridge.finalize_media()
    assert len(calls) == 1 and c.unknown and not bridge.auto_handoff and not requests


async def test_model_silently_skipping_candidates_also_recovers(bridge_factory, tmp_path):
    bridge, requests, calls = setup(bridge_factory, tmp_path)
    bridge.control.candidate_reviews.clear()
    await bridge.finalize_media()
    assert len(calls) == 1 and bridge.auto_handoff and not requests


async def test_native_zero_selection_closes_after_local_queue_without_another_model_call(tmp_path):
    from app.agent.harness_runtime import HarnessToolAgent
    from app.agent.media_tasks import download_delivery_properties
    from tests.agent.test_harness_runtime import Guard, model_client
    w = workspace(tmp_path / 'workspace')
    calls = []

    def script(body, n):
        if n == 1:
            return responses_sse(name='list_web_images', arguments={'ref_id': URL}, number=n, call_id=f'c{n}')
        if n == 2:
            return responses_sse(name='set_task_goal', arguments={'goal': '下载网页图片',
                'delivery': 'media', 'requirements': ['media_delivery'], 'media_selection': {
                'candidate_reviews': [{'ref_id': URL, 'decision': 'excluded', 'reason': '模型排除'}]}}, number=n, call_id=f'c{n}')
        assert n == 3
        return responses_sse(name='send_private', arguments={'text': '没有合适的图', 'task_status': 'partial'}, number=n, call_id=f'c{n}')

    async def listing(args):
        return ToolOutcome({'page_url': URL, 'images': [{'image_id': 'webimg-a'}]})

    async def download(args):
        calls.append(('download', args))
        return ToolOutcome({'job_id': 1, 'status': 'queued',
                            'delivery': 'automatic_delivery_to_original_request'})

    async def send(args):
        calls.append(('send', args))
        return ToolOutcome({}, 'sent', 'receipt')

    client, requests, _ = model_client(script)
    agent = HarnessToolAgent(decision_client=client, task_workspace=w, require_reply=True,
        reply_source_id=10, session_home=tmp_path / 'native', max_steps=5)
    tools = (
        tool('list_web_images', listing, properties={'ref_id': {'type': 'string'}}),
        tool('download_web_images', download, properties={
            'image_ids': {'type': 'array', 'items': {'type': 'string'}}, **download_delivery_properties()}),
        tool('send_private', send, effect='write', properties={
            'text': {'type': 'string'}, 'task_status': {'type': 'string'}}))
    try:
        result = await agent.run(['Synthetic web media request'], tools, Guard())
        assert len(requests) == 3 and [name for name, _ in calls] == ['download']
        assert result.reason == 'background_auto_handoff' and not result.receipt_id
    finally:
        agent.close(); client.http_client.close()


@pytest.mark.parametrize('surface', ['group', 'private'])
async def test_fallback_real_queue_downloads_pixels_and_delivers_original_source(
        sqlite_engine, tmp_path, monkeypatch, surface):
    import asyncio
    import base64
    from app.adapters.sender import Sender
    from tests.agent.test_web_image_delivery import webpage
    from test_autonomous_social import composed, image_bytes, add_source, NOW, FRIEND
    from tests.agent.test_harness_runtime import model_client
    from tests.private_chat.test_agent_service import setup as private_setup, event, OTHER
    page = 'https://example.com/madoka'
    web, http, page_requests = webpage(monkeypatch)
    from app.agent.harness_runtime import _Bridge
    original_recover = _Bridge.recover_excluded_media
    recovery = []
    async def diagnose(bridge):
        await original_recover(bridge)
        recovery.append({'media': bridge.control.requires_media if bridge.control else None,
            'sources': bridge.control.excluded_media_sources() if bridge.control else [], 'trace': bridge.trace[-5:]})
    monkeypatch.setattr(_Bridge, 'recover_excluded_media', diagnose)

    def script(body, n):
        if n == 1: name, args = 'list_web_images', {'ref_id': page}
        elif n == 2:
            name, args = 'set_task_goal', {'goal': '下载网页图片', 'delivery': 'media',
                'requirements': ['media_delivery'], 'media_selection': {'candidate_reviews': [
                {'ref_id': page, 'decision': 'excluded', 'reason': '模型全部pass'}]}}
        else:
            assert n == 3
            name, args = f'send_{surface}', {'text': '没找到合适的媒体', 'task_status': 'partial', 'artifact_ids': []}
        return responses_sse(name=name, arguments=args, number=n, call_id=f'fallback-{n}')

    class Gateway:
        def __init__(self): self.calls = []
        async def call_api(self, action, params):
            self.calls.append((action, params))
            return {'status': 'ok', 'retcode': 0, 'data': {'message_id': 'fallback-real-receipt'}}

    gateway = Gateway()
    try:
        if surface == 'group':
            model, requests, _ = model_client(script)
            async with composed(sqlite_engine, tmp_path, model, runtime='harness') as fixture:
                fixture.executor.sender = Sender(gateway)
                fixture.router.web_search_client = web
                queue = fixture.executor.media_jobs; queue.running = True
                from app.agent.social_contracts import SocialObservation
                from app.agent.social_policy import configured_social_jobs
                from types import SimpleNamespace
                job = next(j for j in configured_social_jobs(fixture.router.runtime) if j.kind == 'group')
                source = add_source(sqlite_engine, platform_id='fallback-request',
                    text='小町请下载这个网页的图片或视频 '+page)
                guard = SimpleNamespace(action_id='fallback-request', valid=lambda: True,
                    require_valid=lambda: None, on_dispatch=lambda: None)
                await fixture.executor.run_tools(SocialObservation(job, NOW, 'synthetic persona', source_message_ids=(source,)),
                    guard, interactive=True, target_user_id=FRIEND, target_message_id='fallback-request', direct_group_reply=True)
                assert not gateway.calls
                assert await queue.tick() and await queue.tick(), recovery
                await queue.stop()
        else:
            service, agent, dispatcher, _, requests = private_setup(sqlite_engine, tmp_path, script)
            service.sender = Sender(gateway); service.web_search_client = web
            await service.start()
            try:
                incoming = event('fallback-request', user_id=OTHER,
                    text='从https://example.com/madoka中帮我把小圆的人设图扒下来')
                await dispatcher.handle_private_message(incoming)
                async with asyncio.timeout(20):
                    while not gateway.calls or agent._active: await asyncio.sleep(.05)
                await dispatcher.handle_private_message(incoming)
            finally: await service.stop()
        assert len(requests) == 3 and len(gateway.calls) == 1 and len(page_requests) == 2, recovery
        action, params = gateway.calls[0]
        assert action.startswith(f'send_{surface}')
        segments = params.get('message', [])
        for node in params.get('messages', []): segments += node['data']['content']
        pixels = [base64.b64decode(item['data']['file'][9:]) for item in segments if item['type'] == 'image']
        assert pixels == [image_bytes()]
        assert any('网页候选' in item['data'].get('text', '') for item in segments)
        if surface == 'private': assert params['user_id'] == OTHER
    finally:
        http.close(); web.http_client.close()
