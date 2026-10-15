import json

from app.agent.context_policy import RequestContextPolicy
from app.agent.harness_runtime import HarnessToolAgent
from app.agent.harness_session_files import native_session_logs, native_file_path
from app.agent.tool_agent import AgentTool, ToolOutcome
from tests.agent.test_harness_runtime import model_client, responses_sse, Guard


def prompt(target, environment):
    return ['Current observation JSON:\n' + json.dumps({'current_target': {
        'source_id': 900100001, 'text': target}, 'context': [environment],
        'authority': {'canonical_owner': False}})]


async def test_native_silent_turns_are_reusable_without_accumulating_observation(tmp_path):
    client, requests, _ = model_client(lambda _, n: responses_sse(number=n))
    agent = HarnessToolAgent(decision_client=client, session_home=tmp_path/'home',
        session_id='live-session', ephemeral_context=True,
        context_policy=RequestContextPolicy(history_tokens=8000))
    try:
        for index in range(2):
            result = await agent.run(prompt(f'target-{index}', f'ENVIRONMENT-{index}'), (), Guard())
            assert result.status == 'silent', result.reason
        assert len(requests) == 2
        body = json.dumps(requests[-1], ensure_ascii=False)
        assert 'ENVIRONMENT-1' in body and 'ENVIRONMENT-0' not in body
        assert 'target-0' not in body
    finally:
        agent.close()
        client.http_client.close()


async def test_native_successful_interaction_survives_next_silent_observation(tmp_path):
    client, requests, _ = model_client(lambda _, n: responses_sse(name='send_group', arguments={'text': 'actual reply'}, number=n)
        if n == 1 else responses_sse(number=n))
    agent = HarnessToolAgent(decision_client=client, session_home=tmp_path/'home',
        session_id='live-session', ephemeral_context=True)
    async def send(args):
        return ToolOutcome({'result': 'sent'}, 'sent', 'synthetic-actual-receipt')
    tool = AgentTool('send_group', 'send', {'type': 'object', 'properties': {'text': {'type': 'string'}},
        'required': ['text'], 'additionalProperties': False}, send, 'write')
    try:
        result = await agent.run(prompt('REAL_OLD_REQUEST', 'ENVIRONMENT_OLD_SECRET'), (tool,), Guard())
        assert result.status == 'sent', result.reason
        before = {str(p): native_file_path(p).read_bytes() for p in native_session_logs(tmp_path/'home', 'live-session')}
        result = await agent.run(prompt('NEW_REQUEST', 'ENVIRONMENT_NEW_ONLY'), (tool,), Guard())
        assert result.status == 'silent', result.reason
        assert before == {str(p): native_file_path(p).read_bytes() for p in native_session_logs(tmp_path/'home', 'live-session')}
        body = json.dumps(requests[-1], ensure_ascii=False)
        assert 'REAL_OLD_REQUEST' in body and 'ENVIRONMENT_NEW_ONLY' in body
        assert 'ENVIRONMENT_OLD_SECRET' not in body
    finally:
        agent.close()
        client.http_client.close()
