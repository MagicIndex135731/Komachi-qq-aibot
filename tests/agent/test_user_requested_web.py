"""Canonical user requests, rather than background goals, authorize web I/O."""
import httpx
import pytest
from types import SimpleNamespace

from app.agent.social_contracts import SocialObservation
from app.agent.social_policy import configured_social_jobs
from app.agent.tool_agent import ToolTurnResult
from app.providers.web_search import WebSearchClient
from app.storage.db import session_scope
from app.storage.models import Message
from test_autonomous_social import FRIEND, NOW, PolicyGuard, ScriptedModel, add_source, composed
from test_shared_links import card


@pytest.mark.parametrize('mode', ['question', 'addressed', 'casual', 'patrol', 'public_patrol', 'negative', 'result_return'])
async def test_group_request_and_patrol_use_same_manifest_but_enforce_network_gate(sqlite_engine, tmp_path, monkeypatch, mode):
    calls, results = [], []
    web = WebSearchClient(provider='wyzlab_alpha', base_url='https://synthetic.invalid/v1',
        api_key='synthetic', timeout_seconds=45, http_client=httpx.Client(transport=httpx.MockTransport(
            lambda request: calls.append(request) or httpx.Response(200, json={'output': 'synthetic evidence'}))))

    class Harness:
        def __init__(self, **kwargs):
            pass

        async def run(self, prompt, tools, guard, **kwargs):
            registry = {t.name: t for t in tools}
            assert len(tools) + 2 <= 64  # Reserve RequestTaskControl tools.
            assert 'inspect_web_resources' in registry
            if 'search_web' in registry:
                results.append(await registry['search_web'].handler({'query': 'synthetic factual lookup'}))
            return ToolTurnResult('silent', reason='model_finished')

    monkeypatch.setattr('app.agent.harness_runtime.HarnessToolAgent', Harness)
    kind = 'qzone_publish' if mode == 'public_patrol' else 'group'
    try:
        async with composed(sqlite_engine, tmp_path, ScriptedModel(), runtime='harness', kind=kind) as fixture:
            fixture.router.web_search_client = web
            text = {'question': '这个合成角色是打什么伤害的，是c还是辅啊', 'result_return': '合成角色强吗', 'addressed': '小町',
                'negative': '小町别搜了'}.get(mode, '今天上课比较晚')
            source = add_source(sqlite_engine, platform_id='current-source', text=text)
            old_card = add_source(sqlite_engine, platform_id='unrequested-old-card')
            with session_scope(sqlite_engine) as session:
                session.get(Message, source).raw_json = {'message': [{'type': 'text', 'data': {'text': text}}]}
                session.get(Message, old_card).raw_json = {'message': [card()]}
            job = configured_social_jobs(fixture.router.runtime)[0]
            observed = SocialObservation(job, NOW, '小町', source_message_ids=(source, old_card) if kind == 'group' else ())
            interactive = mode not in {'patrol', 'public_patrol'}
            await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed, scheduled=True),
                interactive=interactive, target_user_id=FRIEND if interactive else None,
                target_message_id='current-source' if interactive else None,
                direct_group_reply=mode in {'question', 'addressed', 'negative'}, research_result={} if mode == 'result_return' else None)
            assert results and (results[-1].status == 'ok') == (mode in {'question', 'addressed'})
            assert len(calls) == int(mode in {'question', 'addressed'})
            if not calls:
                assert results[-1].data['error'] == 'web_requires_explicit_request'
    finally:
        web.http_client.close()


@pytest.mark.parametrize('question', [False, True, 'natural_request','continued_find_request'])
async def test_child_and_media_rebuild_request_gate_and_ignore_legacy_autonomous_flag(sqlite_engine, tmp_path, monkeypatch, question):
    from app.agent.delegated_research import child_inputs
    from app.agent.media_tasks import execute_web_media
    from app.agent.research_tasks import ResearchTaskWorkspace
    from app.agent.tool_agent import AgentTool, ToolOutcome
    from test_research_queue import fixture_queue
    from test_web_tools import Client, Runner
    _, payload, _ = fixture_queue(sqlite_engine)
    payload.update(task_key='synthetic-child', autonomous_web=True, operation='download_web_images', arguments={})
    with session_scope(sqlite_engine) as session:
        row = session.get(Message, payload['source_id'])
        text = ('帮我去x扒一些synthetic最新的cos视频或图片' if question == 'natural_request'
                else '继续帮我去X找一些关于synthetic的cos视频或图片' if question=='continued_find_request'
                else '合成角色强吗' if question else '今天上课比较晚')
        row.raw_json = {'message': [{'type': 'text', 'data': {'text': text}}]}
    settings = SimpleNamespace(data_dir=tmp_path, bot_qq=payload['account_id'])
    guard = SimpleNamespace(sources=set(payload['sources']), source_message_ids=tuple(payload['sources']),
        require_valid=lambda: None, valid_source=lambda *a, **k: True,
        valid_dependencies=lambda _: True, snapshot=lambda: {'sources': payload['sources'], 'memory': {}})
    workspace = ResearchTaskWorkspace(tmp_path / 'research_tasks', scope=payload['scope'],
        actor_id=payload['actor_id'], source_id=payload['source_id'], request=payload['goal'],
        task_key=payload['task_key'], valid_source=lambda _: True)
    payload['task_id'] = workspace.task_id
    client = Client()
    _, tools = child_inputs(payload, guard, settings=settings, engine=sqlite_engine, web_client=client, runner=Runner())
    result = await next(t for t in tools if t.name == 'search_web').handler({'query': '合成角色'})
    assert (result.status == 'ok') == bool(question) and bool(client.calls) == bool(question)
    media_calls = []
    async def download(args):
        media_calls.append(args)
        return ToolOutcome({'artifact_ids': []})
    monkeypatch.setattr('app.agent.web_images.build_web_image_tools', lambda *a, **k:
        (AgentTool('download_web_images', 'download', {'type': 'object'}, download),))
    result = await execute_web_media(payload, guard, engine=sqlite_engine, settings=settings, web_client=None, runner=Runner())
    assert (result['status'] == 'ok') == bool(question) and bool(media_calls) == bool(question)
