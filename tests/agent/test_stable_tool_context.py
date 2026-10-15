"""Actual composition keeps schemas fixed when the canonical speaker changes."""
import json
from types import SimpleNamespace

from app.agent.social_contracts import SocialObservation
from app.agent.social_policy import configured_social_jobs
from app.agent.tool_agent import ToolTurnResult
from app.providers.llm_client import LlmClient
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.repositories import UserRepository
from test_autonomous_social import FRIEND, NOW, PolicyGuard, ScriptedModel, add_source, composed


async def test_owner_member_and_return_use_identical_tools_and_instruction_prefix(sqlite_engine, tmp_path, monkeypatch):
    captured = []
    class Harness:
        def __init__(self, **kwargs):
            pass
        async def run(self, prompt, tools, guard, **kwargs):
            schema = [{'name': t.name, 'description': t.description, 'parameters': t.parameters} for t in tools]
            # Use the production split: actor-specific policy must be in input.
            fixed, inputs = LlmClient._split_prompt_lines(None,
                [line for line in prompt if not line.startswith('Current persona guidance:\n')])
            captured.append((schema, fixed, inputs, {t.name: t for t in tools}))
            if len(captured) == 2:
                for name, args in [('request_owner_action', {'target': 'public', 'group_id': 0, 'include_current_images': False}),
                                   ('remember_intention', {'goal': 'forged', 'progress': '', 'next_step': ''}),
                                   ('execute_python', {'code': 'print(1)'})]:
                    assert (await captured[-1][3][name].handler(args)).status == 'blocked'
            if len(captured) == 3:
                assert (await captured[-1][3]['delegate_research_task'].handler({})).status == 'blocked'
            return ToolTurnResult('silent', reason='model_finished')
    monkeypatch.setattr('app.agent.harness_runtime.HarnessToolAgent', Harness)
    async with composed(sqlite_engine, tmp_path, ScriptedModel(), runtime='harness') as fixture:
        owner = fixture.router.runtime.settings.owner_qq
        fixture.executor.sandbox = object()
        job = configured_social_jobs(fixture.router.runtime)[0]
        sources = [add_source(sqlite_engine, platform_id='prefix-owner'), add_source(sqlite_engine, platform_id='prefix-member')]
        with session_scope(sqlite_engine) as session:
            UserRepository(session).upsert_user(user_id=owner, nickname='synthetic owner', group_card='')
            session.get(Message, sources[0]).user_id = owner
        for source_id, actor, return_value in [(sources[0], owner, None), (sources[1], FRIEND, None),
                                               (sources[1], FRIEND, {'status': 'completed', 'findings': []})]:
            observed = SocialObservation(job, NOW, '小町', source_message_ids=(source_id,))
            guard = SimpleNamespace(action_id='stable-' + str(len(captured)), valid=lambda: True,
                require_valid=lambda: None, on_dispatch=lambda: None)
            await fixture.executor.run_tools(observed, guard, interactive=True,
                target_user_id=actor, target_message_id='prefix-owner' if actor == owner else 'prefix-member',
                direct_group_reply=True, research_result=return_value)
        assert captured[0][0] == captured[1][0] == captured[2][0]
        assert captured[0][1] == captured[1][1] == captured[2][1]
        assert len(captured[0][0]) <= 64
        assert 'canonical_owner": true' in '\n'.join(captured[0][2])
        assert 'canonical_owner": false' in '\n'.join(captured[1][2])


async def test_private_member_cannot_call_owner_tool_and_manifest_matches_owner(sqlite_engine, tmp_path):
    from tests.private_chat.test_agent_service import setup, event, sse, OWNER, OTHER
    from test_owner_actions import setup as owner_setup
    service, agent, dispatcher, sender, requests = setup(sqlite_engine, tmp_path,
        lambda body, n: sse('request_owner_action', {'target': 'public', 'group_id': 0, 'include_current_images': False}, number=n)
        if n == 2 else sse('send_private', {'text': 'synthetic'}, number=n) if n in {1, 3} else sse(number=n))
    agent.owner_service, _ = owner_setup(sqlite_engine, tmp_path)
    await service.start()
    try:
        await dispatcher.handle_private_message(event('prefix-owner', user_id=OWNER))
        await dispatcher.handle_private_message(event('prefix-other', user_id=OTHER))
        assert requests[0]['tools'] == requests[1]['tools']
        assert requests[0]['instructions'] == requests[1]['instructions']
        output = next(item for item in requests[2]['input'] if item.get('type') == 'function_call_output')
        assert json.loads(output['output'])['data']['error'] == 'owner_required'
        assert sender.texts == [(OWNER, 'synthetic'), (OTHER, 'synthetic')]
    finally:
        await service.stop()
