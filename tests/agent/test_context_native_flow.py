"""Real SDK context boundaries and continuation against a synthetic provider."""
import json

import httpx
import pytest

from app.agent.harness_runtime import HarnessToolAgent
from app.agent.harness_session_files import native_session_logs
from app.agent.tool_agent import ToolOutcome
from tests.agent.test_harness_runtime import Guard, model_client, responses_sse, tool


@pytest.mark.asyncio
@pytest.mark.parametrize('reject', [False, True])
async def test_live_response_chain_and_explicit_stale_fallback_preserve_tool_once(tmp_path, reject):
    effects = []
    def script(body, n):
        assert body['store'] is True
        if n == 1:
            return responses_sse(name='inspect', call_id='first', number=n)
        if n == 2 and reject:
            assert body.get('previous_response_id') == 'resp_1'
            return httpx.Response(404, json={'error': {'code': 'previous_response_not_found'}})
        if reject:
            assert 'previous_response_id' not in body
            assert any(x.get('type') == 'function_call_output' for x in body['input'])
        else:
            assert body.get('previous_response_id') == 'resp_1'
            assert not any(x.get('type') == 'function_call' for x in body['input'])
        return responses_sse(number=n)
    async def inspect(args):
        effects.append('inspect')
        return ToolOutcome({'verified': True})
    client, requests, _ = model_client(script)
    agent = HarnessToolAgent(decision_client=client, session_home=tmp_path/'native', session_id='stable',
        continuation_mode='on', max_steps=4)
    try:
        result = await agent.run(['synthetic research'], (tool('inspect', inspect),), Guard())
        assert result.status == 'silent', result
        assert effects == ['inspect'] and len(requests) == (3 if reject else 2)
        assert result.reason == 'model_finished'
        # A later application turn uses authoritative local replay, not a stale remote pointer.
        client.http_client.close()
        client.http_client = httpx.Client(transport=httpx.MockTransport(lambda req:
            httpx.Response(200, content=responses_sse(number=9), headers={'Content-Type':'text/event-stream'})
            if 'previous_response_id' not in json.loads(req.content) else httpx.Response(400)))
        again = await agent.run(['new application turn'], (tool('inspect', inspect),), Guard())
        assert again.reason == 'model_finished', again
    finally:
        agent.close(); client.http_client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('useful', [False, True])
async def test_patrol_repeated_wait_finishes_and_retains_only_useful_work(tmp_path, useful):
    saved = []
    def script(body, n):
        if useful and n == 1:
            return responses_sse(name='read_artifact', call_id='read', number=n)
        return responses_sse(name='remember_intention', arguments={'goal':'空间巡检',
            'progress':'没有新动态', 'next_step':'等待下次巡检'}, call_id='wait_'+str(n), number=n)
    async def remember(args):
        saved.append(args)
        return ToolOutcome({'saved': True})
    async def read(args):
        return ToolOutcome({'text':'Useful creative draft'})
    props = {key:{'type':'string'} for key in ('goal','progress','next_step')}
    client, requests, _ = model_client(script)
    home = tmp_path/'native'
    agent = HarnessToolAgent(decision_client=client, session_home=home, session_id='patrol',
        max_steps=8, idle_patrol=True, ephemeral_context=True)
    try:
        result = await agent.run(['synthetic patrol'], (tool('remember_intention', remember,
            properties=props, required=tuple(props)), tool('read_artifact', read)), Guard())
        assert result.status == 'silent' and result.reason == 'patrol_no_progress', result
        assert len(requests) == (3 if useful else 2) and len(saved) == 1
        assert agent.context_retention['retain_history'] is useful
        assert bool(native_session_logs(home, 'patrol')) is useful
    finally:
        agent.close(); client.http_client.close()
