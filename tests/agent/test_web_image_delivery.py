"""Native Harness selects webpage pixels and delivers through actual adapters."""
import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from app.adapters.sender import Sender, QQMessageDeliveryUncertainError
from app.agent.artifacts import AutonomousArtifactStore
from app.providers.web_images import WebImageClient
from app.providers.web_search import WebSearchClient
from app.storage.agent_effects import AgentToolEffect
from app.storage.db import session_scope
from sqlalchemy import select

from test_autonomous_social import composed, image_bytes, writes, add_source, NOW
from test_harness_runtime import model_client, responses_sse
from tests.private_chat.test_agent_service import OWNER, OTHER, event, setup, sse
from tests.providers.test_web_images import NOTE, note_page


async def requested_group_media(fixture):
    """Exercise native immediate media via a real owner's explicit task.

    Member batch/video queuing is exercised separately by media-task tests.
    A scheduled patrol no longer has permission to fetch public web media.
    """
    from app.agent.social_contracts import SocialObservation
    from app.agent.social_policy import configured_social_jobs
    from app.storage.models import Message
    from app.storage.repositories import UserRepository
    text = '请联网下载这个网页的图片或视频并发到当前群'
    source = add_source(fixture.router.engine, platform_id='owner-requested-web-media', text=text)
    owner = fixture.router.runtime.settings.owner_qq
    with session_scope(fixture.router.engine) as session:
        UserRepository(session).upsert_user(user_id=owner, nickname='synthetic owner', group_card='')
        row = session.get(Message, source)
        row.user_id = owner
        row.raw_json = {'message': [{'type': 'text', 'data': {'text': text}}]}
    job = configured_social_jobs(fixture.router.runtime)[0]
    observation = SocialObservation(job, NOW, '小町', source_message_ids=(source,))
    guard = SimpleNamespace(action_id='owner-requested-web-media', valid=lambda: True,
        require_valid=lambda: None, on_dispatch=lambda: None)
    return await fixture.executor.run_tools(observation, guard, interactive=True, owner_command={
        'source_id': source, 'target': 'group', 'group_id': job.scope_id, 'task_id': 1, 'instruction': text})


def output(body):
    value = next(item for item in reversed(body['input']) if item.get('type') == 'function_call_output')
    content = value['output']
    if isinstance(content, list):
        content = next(block['text'] for block in content if block.get('type') in {'input_text', 'text'})
    return json.JSONDecoder().raw_decode(content)[0]['data']


def webpage(monkeypatch):
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, content=image_bytes()) if request.url.path.endswith('.png') else httpx.Response(200,
            headers={'Content-Type': 'text/html'}, text='<template><h2>小圆人设</h2><img alt="鹿目圆设定图" src="/madoka.png"></template>')
    client = httpx.Client(transport=httpx.MockTransport(respond))
    provider = WebImageClient(http_client=client)
    monkeypatch.setattr('app.agent.web_images.WebImageClient', lambda _: provider)
    web = WebSearchClient(provider='ddgs', base_url='', api_key='', timeout_seconds=10)
    return web, client, requests


async def test_native_group_selects_and_sends_page_image(sqlite_engine, tmp_path, monkeypatch):
    web, http, page_requests = webpage(monkeypatch)
    def script(body, n):
        if n == 1:
            name, args = 'list_web_images', {'ref_id': 'https://example.com/madoka'}
        elif n == 2:
            name, args = 'download_web_image', {'image_id': output(body)['images'][0]['image_id']}
        elif n == 3:
            name, args = 'send_group', {'text': '这是网页中的小圆人设图', 'artifact_ids': [output(body)['artifact_id']]}
        else:
            return responses_sse(number=n)
        return responses_sse(name=name, arguments=args, call_id=f'page-{n}', number=n)
    model, requests, _ = model_client(script)
    try:
        async with composed(sqlite_engine, tmp_path, model, runtime='harness') as fixture:
            fixture.router.web_search_client = web
            await requested_group_media(fixture)
            assert len(writes(fixture)) == 1 and len(page_requests) == 2, (len(requests), len(page_requests), str(requests[-1]['input'][-2:])[:1800])
            segments = writes(fixture)[0][1]['message']
            pixels = base64.b64decode(next(item['data']['file'][9:] for item in segments if item['type'] == 'image'))
            assert pixels == image_bytes()
            assert len(requests) == 4
            assert fixture.image_client.calls == [] and fixture.image_client.edit_calls == []
    finally:
        http.close()
        web.http_client.close()


@pytest.mark.parametrize('user_id', [OWNER, OTHER])
@pytest.mark.parametrize('uncertain', [False, True])
async def test_native_private_selects_and_sends_original_once(sqlite_engine, tmp_path, monkeypatch, user_id, uncertain):
    web, http, page_requests = webpage(monkeypatch)
    def script(body, n):
        if n == 1:
            return sse('list_web_images', {'ref_id': 'https://example.com/madoka'}, number=n)
        if n == 2:
            return sse('download_web_image', {'image_id': output(body)['images'][0]['image_id']}, number=n)
        if n == 3:
            return sse('send_private', {'text': '小圆人设图', 'artifact_ids': [output(body)['artifact_id']]}, number=n)
        return sse(number=n)
    service, agent, dispatcher, _, requests = setup(sqlite_engine, tmp_path, script)
    class Gateway:
        calls = []
        async def call_api(self, action, params):
            self.calls.append((action, params))
            if uncertain:
                raise QQMessageDeliveryUncertainError('synthetic lost ack')
            return {'status': 'ok', 'retcode': 0, 'data': {'message_id': 'web-private-receipt'}}
    gateway = Gateway()
    service.sender = Sender(gateway)
    service.web_search_client = web
    await agent.start()
    try:
        incoming = event('web-private-request', user_id=user_id,
            text='从https://example.com/madoka中帮我把小圆的人设图扒下来')
        await dispatcher.handle_private_message(incoming)
        await dispatcher.handle_private_message(incoming)
        assert len(gateway.calls) == 1 and len(page_requests) == 2
        action, params = gateway.calls[0]
        assert action == 'send_private_msg' and params['user_id'] == user_id
        pixels = base64.b64decode(next(item['data']['file'][9:] for item in params['message'] if item['type'] == 'image'))
        assert pixels == image_bytes()
        store = AutonomousArtifactStore(tmp_path / 'autonomous_artifacts', account_id=agent.account_id,
            scope=f'private:{user_id}', engine=sqlite_engine, source_message_ids=lambda: ())
        assert len(store.list()) == 1
        artifact = store.list()[0]['id']
        assert store.provenance(artifact)
        foreign = AutonomousArtifactStore(store.root, account_id=agent.account_id,
            scope=f'private:{OTHER if user_id == OWNER else OWNER}', engine=sqlite_engine, source_message_ids=lambda: ())
        assert foreign.get(artifact) is None
        with session_scope(sqlite_engine) as session:
            effects = list(session.scalars(select(AgentToolEffect).where(AgentToolEffect.operation == 'send_private')))
            assert len(effects) == 1 and effects[0].status == ('uncertain' if uncertain else 'sent')
        assert {'list_web_images', 'download_web_image'} <= {tool['name'] for tool in requests[0]['tools']}
    finally:
        await agent.stop()
        http.close()
        web.http_client.close()


@pytest.mark.parametrize('surface', ['group', 'private'])
async def test_native_album_download_and_merged_forward(sqlite_engine, tmp_path, monkeypatch, surface):
    requests = []
    def respond(request):
        requests.append(request)
        if request.url.host=='cdn.example.com':
            import io
            from PIL import Image
            body=io.BytesIO()
            Image.new('RGB',(32,24),'red' if len(requests)==2 else 'blue').save(body,'PNG')
            return httpx.Response(200,content=body.getvalue())
        return httpx.Response(200,headers={'Content-Type':'text/html'},text=note_page())
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        provider = WebImageClient(http_client=http)
        monkeypatch.setattr('app.agent.web_images.WebImageClient', lambda _: provider)
        web = WebSearchClient(provider='ddgs', base_url='', api_key='', timeout_seconds=10)
        result_steps = []
        def command(body, n, *, quoted=False):
            from test_delegation_composition import observation
            current = observation(body, 'Private observation JSON:\n') if quoted else {}
            if current.get('kind') == 'research_result':
                if result_steps:
                    return None, None
                data = current['delegated_result']['data']
                result_steps.append(True)
                assert data['downloaded_count'] == 2 and data['all_requested_downloaded']
                return 'send_private', {'text': '笔记配图共2张', 'artifact_ids': data['artifact_ids'], 'as_forward': True}
            if quoted and n == 1:
                return 'read_message_links', {'source_id': source[0]}
            step = n - int(quoted)
            if step == 1:
                return 'list_web_images', {'ref_id': output(body)['links'][0]['url'] if quoted else f'https://www.xiaohongshu.com/explore/{NOTE}'}
            if step == 2:
                return 'download_web_images', {'image_ids': [item['image_id'] for item in output(body)['images']]}
            if step == 3:
                data = output(body)
                if quoted:
                    assert data['status'] == 'queued'
                    return 'send_private', {'text': '排队中'}
                assert data['downloaded_count'] == 2 and data['all_requested_downloaded']
                return 'send_group' if surface == 'group' else 'send_private', {
                    'text': '笔记配图共2张', 'artifact_ids': data['artifact_ids'], 'as_forward': True}
            return None, None
        if surface == 'group':
            def model(body, n):
                name, args = command(body, n)
                return responses_sse(name=name, arguments=args, number=n, call_id=f'album-{n}') if name else responses_sse(number=n)
            client, _, _ = model_client(model)
            async with composed(sqlite_engine, tmp_path, client, runtime='harness') as fixture:
                fixture.router.web_search_client = web
                old = fixture.gateway.call_api
                async def gateway(action, params):
                    if action == 'send_group_forward_msg':
                        fixture.gateway.calls.append((action, params))
                        return {'status': 'ok', 'retcode': 0, 'data': {'message_id': 'album-group-receipt'}}
                    return await old(action, params)
                fixture.gateway.call_api = gateway
                await requested_group_media(fixture)
                sent = writes(fixture)
                assert len(sent) == 1 and sent[0][0] == 'send_group_forward_msg'
                assert len(sent[0][1]['messages']) == 2
        else:
            source = []
            def model(body, n):
                name, args = command(body, n, quoted=True)
                return sse(name, args, number=n) if name else sse(number=n)
            service, agent, dispatcher, _, _ = setup(sqlite_engine, tmp_path, model)
            from app.adapters.onebot_models import parse_private_message_event
            from app.storage.models import Message
            shared = parse_private_message_event({'message_id': 'album-card', 'user_id': OTHER, 'time': 100,
                'message': [{'type': 'share', 'data': {'url': f'https://www.xiaohongshu.com/explore/{NOTE}', 'title': '小红书图文'}}]})
            dispatcher._persist_inbound(shared)
            with session_scope(sqlite_engine) as session:
                source.append(session.scalar(select(Message.id).where(Message.platform_msg_id == f'private-inbound-{OTHER}-album-card')))
            class Gateway:
                calls = []
                async def call_api(self, action, params):
                    self.calls.append((action, params))
                    return {'status': 'ok', 'retcode': 0, 'data': {'message_id': 'album-private-receipt'}}
            gateway = Gateway()
            service.sender, service.web_search_client = Sender(gateway), web
            incoming = event('album-request', user_id=OTHER, text='把里面图片都搬出来然后发合并转发')
            incoming.reply_to_msg_id = 'album-card'
            incoming.raw_payload['message'].insert(0, {'type': 'reply', 'data': {'id': 'album-card'}})
            await agent.start()
            try:
                await dispatcher.handle_private_message(incoming)
                for _ in range(160):
                    if any(action == 'send_private_forward_msg' for action, _ in gateway.calls):
                        break
                    import asyncio
                    await asyncio.sleep(0.1)
                sent = [params for action, params in gateway.calls if action == 'send_private_forward_msg']
                assert len(sent) == 1 and len(sent[0]['messages']) == 2
                assert len(gateway.calls) == 2
            finally:
                await agent.stop()
        assert len(requests) == 3
        web.http_client.close()
