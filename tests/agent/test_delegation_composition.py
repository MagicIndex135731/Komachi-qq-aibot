"""Real Harness: main keeps chatting while an isolated child reads sources."""
import asyncio
import json
import threading

import httpx
import pytest

from app.agent.runtime_trace import TraceReader
from app.providers.web_search import WebSearchClient
from app.agent.social_contracts import SocialObservation
from app.agent.social_policy import configured_social_jobs
from tests.private_chat.test_agent_service import setup, event, sse, OTHER
from test_autonomous_social import composed, add_source, GROUP, FRIEND, NOW, writes
from test_harness_runtime import model_client, responses_sse

DELEGATE = {'goal': '核对多个合成源文件的风险', 'deliverable': '已验范围、证据链接和缺口', 'reason': '多个文件的正文会占用大量上下文'}
CHECKPOINT = {'progress': '已核对合成入口', 'next_step': '认证文件尚未提供', 'findings': ['已验证合成入口，认证部分未检查'], 'status': 'completed'}
RAW = 'CHILD_ONLY_RAW_SOURCE_' * 600


def observation(body, prefix):
    for item in reversed(body.get('input', [])):
        content = item.get('content', '')
        text = '\n'.join(part.get('text', '') for part in content if isinstance(part, dict)) if isinstance(content, list) else content
        if prefix in text:
            return json.JSONDecoder().raw_decode(text.split(prefix, 1)[1])[0]
    raise AssertionError('observation missing: ' + prefix)


def web_client(entered, release):
    def page(request):
        entered.set()
        assert release.wait(15), 'main did not continue while child was reading'
        return httpx.Response(200, json={'output': 'Source (https://example.com/source)\nciteturn0view0\nL0: ' + RAW,
            'results': [{'type': 'text_result', 'url': 'https://example.com/source', 'ref_id': 'turn0view0'}]})
    return WebSearchClient(provider='wyzlab_alpha', base_url='https://synthetic.invalid/v1', api_key='synthetic',
        timeout_seconds=45, http_client=httpx.Client(transport=httpx.MockTransport(page)))


def choose(body, counts, *, surface, checkpoint_status):
    text = json.dumps(body, ensure_ascii=False)
    if 'Child task observation JSON:' in text:
        n = counts['child'] = counts.get('child', 0) + 1
        names = {tool['name'] for tool in body['tools']}
        assert not names & {'send_private', 'send_group', 'delegate_research_task', 'generate_private_image',
                            'schedule_wakeup', 'schedule_private_wakeup', 'memory_write', 'request_owner_action'}
        assert '今天上课比较晚' not in text
        assert 'synthetic persona' not in text and '你是小町' not in text
        if n == 1:
            return 'open_web_page', {'ref_id': 'https://example.com/source'}
        assert 'CHILD_ONLY_RAW_SOURCE_' in text
        return ('update_research_task', {**CHECKPOINT, 'status': checkpoint_status}) if n == 2 else None
    assert 'CHILD_ONLY_RAW_SOURCE_' not in text
    obs = observation(body, 'Private observation JSON:\n' if surface == 'private' else 'Current observation JSON:\n')
    if surface == 'group' and obs.get('current_target', {}).get('request_ref'):
        obs['current_target'] = observation(body, 'Current request JSON:\n')['current_target']
    is_result = obs.get('kind') == 'research_result' if surface == 'private' else 'Delegated result JSON:' in text
    key = ('result' if is_result else 'main', obs.get('message_id') if surface == 'private' else obs.get('current_target', {}).get('source_id'))
    n = counts[key] = counts.get(key, 0) + 1
    send = 'send_private' if surface == 'private' else 'send_group'
    args = lambda value: {'text': value, **({'artifact_ids': []} if surface == 'group' else {})}
    if is_result:
        assert checkpoint_status in text and '已验证合成入口' in text
        assert 'delegate_research_task' in {tool['name'] for tool in body['tools']}
        return (send, {**args('入口已核对，认证文件尚未检查'),'task_status':'partial'}) if n == 1 else None
    message = obs.get('message', '') if surface == 'private' else obs.get('current_target', {}).get('text', '')
    if '合成文件' in message:
        if n == 1:
            return 'delegate_research_task', DELEGATE
        if n == 2:
            assert 'background_result_returns_to_main' in text
            return send, {**args('已安排后台核对'),'task_status':'progress'}
        return None
    return (send, args('晚课结束记得休息')) if n == 1 else None


@pytest.mark.parametrize('checkpoint_status', ['completed', 'blocked'])
async def test_private_background_child_allows_new_chat_and_returns_to_same_main(sqlite_engine, tmp_path, checkpoint_status):
    counts, entered, release = {}, threading.Event(), threading.Event()
    def script(body, n):
        command = choose(body, counts, surface='private', checkpoint_status=checkpoint_status)
        return sse(*command, number=n) if command else sse(number=n)
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path, script)
    web = service.web_search_client = web_client(entered, release)
    agent.settings.agent_monitor_enabled = True
    await service.start()
    try:
        await dispatcher.handle_private_message(event('main-complex', user_id=OTHER, text='请联网核对这些合成文件 https://example.com/source'))
        first = agent.sessions.read(f'private:{OTHER}')['checkpoint']['runtime_id']
        assert await asyncio.to_thread(entered.wait, 5)
        await asyncio.wait_for(dispatcher.handle_private_message(event('main-casual', user_id=OTHER, text='今天上课比较晚')), timeout=5)
        assert [text for _, text in sender.texts] == ['已安排后台核对', '晚课结束记得休息']
        release.set()
        async with asyncio.timeout(15):
            while len(sender.texts) < 3:
                await asyncio.sleep(.05)
        assert sender.texts[-1][1] == '入口已核对，认证文件尚未检查'
        assert agent.sessions.read(f'private:{OTHER}')['checkpoint']['runtime_id'] == first
        traces = TraceReader(tmp_path / 'agent_monitor').list_traces()
        child = next(trace for trace in traces if trace['context_kind'] == 'research')
        assert child['parent_context_scope'] == f'private:{OTHER}'
        assert child['delegation_reason'] == DELEGATE['reason']
        assert len([trace for trace in traces if trace['context_kind'] == 'private']) == 3
    finally:
        release.set()
        await service.stop()
        web.http_client.close()


async def test_group_background_child_returns_to_same_main_after_new_message(sqlite_engine, tmp_path):
    counts, entered, release = {}, threading.Event(), threading.Event()
    def script(body, n):
        command = choose(body, counts, surface='group', checkpoint_status='completed')
        return responses_sse(name=command[0], arguments=command[1], call_id=f'delegate-{n}', number=n) if command else responses_sse(number=n)
    client, requests, _ = model_client(script)
    web = web_client(entered, release)
    try:
        async with composed(sqlite_engine, tmp_path, client, runtime='harness') as fixture:
            fixture.router.web_search_client = web
            fixture.router.runtime.settings.agent_monitor_enabled = True
            queue = fixture.executor.research_jobs
            queue.running = True
            job = configured_social_jobs(fixture.router.runtime)[0]
            async def main(message_id, text):
                source = add_source(sqlite_engine, platform_id=message_id, text=text)
                obs = SocialObservation(job, NOW, 'synthetic persona', (text,), (source,))
                class DirectGuard:
                    action_id = message_id

                    def require_valid(self):
                        pass

                    def valid(self):
                        return True

                    def on_dispatch(self):
                        pass
                guard = DirectGuard()
                return await fixture.executor.run_tools(obs, guard, interactive=True, target_user_id=FRIEND,
                    target_message_id=message_id, direct_group_reply=True)
            assert (await main('group-complex', '请核对这些合成文件')).status == 'sent'
            first = fixture.executor.sessions.read(f'group:{GROUP}')['checkpoint']['session_id']
            child = asyncio.create_task(queue.tick())
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                assert (await asyncio.wait_for(main('group-chat', '今天上课比较晚'), timeout=5)).status == 'sent'
                assert len(writes(fixture)) == 2
                release.set()
                await child
                assert await queue.tick()
                assert len(writes(fixture)) == 3
                assert fixture.executor.sessions.read(f'group:{GROUP}')['checkpoint']['session_id'] == first
                traces = TraceReader(fixture.router.runtime.settings.data_dir / 'agent_monitor').list_traces()
                assert next(trace for trace in traces if trace['context_kind'] == 'research')['parent_context_scope'] == f'group:{GROUP}'
                assert any(trace['trigger_kind'] == 'research_result' for trace in traces)
            finally:
                release.set()
                await asyncio.gather(child, return_exceptions=True)
                await queue.stop()
    finally:
        web.http_client.close()
        client.http_client.close()
