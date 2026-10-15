"""Ordinary group drawing through the installed Harness and synthetic providers."""
from __future__ import annotations

import pytest

from app.adapters.onebot_models import parse_group_message_event
from app.agent.contracts import AgentBatch, AgentDecision
from app.agent.owner_actions import restrict_directed_tools
from app.agent.tool_agent import AgentTool, ToolOutcome
from test_autonomous_social import (BOT, FRIEND, GROUP, OTHER_GROUP, NOW,
    InteractiveIngressGuard, composed, writes)
from test_harness_runtime import model_client, responses_sse
from test_image_tools import saved_reference
from test_mature_composition import latest


async def unused(args):
    return ToolOutcome({})


@pytest.mark.parametrize("group_images", [False, True])
async def test_filter_allows_only_explicit_group_drawing(group_images):
    names = {"generate_image": "image", "generate_private_image": "image",
        "execute_python": "compute", "set_group_card": "write", "memory_write": "write",
        "send_group": "write", "send_private": "write", "read_context": "read",
        "unknown_image": "image", "unknown_write": "write"}
    tools = [AgentTool(name, name, {}, unused, effect) for name, effect in names.items()]
    expected = {"send_group", "send_private", "read_context"}
    if group_images:
        expected.add("generate_image")
    restricted = restrict_directed_tools(tools, owner=False, allow_group_images=group_images)
    assert [(t.name, t.description, t.parameters, t.effect) for t in restricted] == [
        (t.name, t.description, t.parameters, t.effect) for t in tools]
    for tool in restricted:
        assert (await tool.handler({})).status == ('ok' if tool.name in expected else 'blocked')
    assert restrict_directed_tools(tools, owner=True, allow_group_images=group_images) == tools
    assert (await next(t for t in restrict_directed_tools(tools, owner=False) if t.name == 'generate_image').handler({})).status == 'blocked'
    forged = AgentTool("generate_image", "forged write", {}, unused, "write")
    assert (await restrict_directed_tools([forged], owner=False, allow_group_images=True)[0].handler({})).status == 'blocked'


@pytest.mark.asyncio
@pytest.mark.parametrize("author", [FRIEND, FRIEND + 1])
@pytest.mark.parametrize("operation", ["generate", "edit", "foreign", "disabled"])
async def test_nonowner_native_group_drawing(sqlite_engine, tmp_path, author, operation):
    generated = []
    reference = []
    enabled = operation != "disabled"

    def script(body, number):
        tools = {tool["name"] for tool in body["tools"]}
        assert ("generate_image" in tools) is enabled
        assert 'generate_private_image' not in tools
        if 'Delegated result JSON:' in str(body):
            if not generated:
                import json
                for item in body['input']:
                    for part in item.get('content', []) if isinstance(item.get('content'), list) else []:
                        value = part.get('text', '')
                        if 'Delegated result JSON:\n' in value:
                            result = json.JSONDecoder().raw_decode(value.split('Delegated result JSON:\n', 1)[1])[0]
                            assert result['status'] == 'ok' and result['data']['ready']
                            generated.append(result['data']['artifact_id'])
                return responses_sse(name='send_group', arguments={'text': '合成作品', 'artifact_ids': generated},
                    call_id='media-complete', number=number)
            return responses_sse(number=number)
        if number == 1 and enabled:
            assert "本群所有成员可要求绘图" in str(body)
            return responses_sse(name="generate_image", arguments={"prompt": "合成猫",
                "reference_artifact_ids": reference, "reference_subjects": ["合成参考"] * len(reference)}, number=number)
        if number == 2 and enabled:
            output = latest(body, "generate_image")
            if operation == "foreign":
                assert output["status"] != "ok"
            else:
                assert output["status"] == "ok" and output["data"]["status"] == 'queued'
                assert not generated
        if number == (2 if enabled else 1):
            return responses_sse(name="send_group", arguments={"text": "合成回复",
                "artifact_ids": generated,'task_status':'progress' if enabled and operation!='foreign' else 'blocked'}, call_id="send", number=number)
        return responses_sse(number=number)

    client, requests, _ = model_client(script)
    async with composed(sqlite_engine, tmp_path, client, runtime="harness",
        options={"allow_images": enabled, "max_steps": 5}) as fixture:
        await fixture.agent.stop()
        if operation in {"edit", "foreign"}:
            reference.append(saved_reference(fixture, sqlite_engine,
                group=OTHER_GROUP if operation == "foreign" else GROUP))
        payload = {"post_type": "message", "message_type": "group", "group_id": GROUP,
            "message_id": "synthetic-drawing-source", "user_id": author, "time": NOW.timestamp(),
            "sender": {"nickname": "synthetic member"}, "message": [
                {"type": "at", "data": {"qq": str(BOT)}},
                {"type": "text", "data": {"text": "画一只猫"}}]}
        event = parse_group_message_event(payload, bot_qq=BOT, bot_name="小町")
        fixture.router.ingest_live_group_message(event)
        batch = AgentBatch(GROUP, (event,), "synthetic-drawing-turn")
        result = await fixture.executor.execute_group_turn(event, batch,
            AgentDecision("reply", event.platform_msg_id, "direct_trigger"),
            InteractiveIngressGuard(batch.turn_id))
        assert result.status == "sent" and len(writes(fixture)) == 1
        assert not fixture.image_client.calls and not fixture.image_client.edit_calls
        initial_requests = len(requests)
        assert initial_requests == (2 if enabled else 1)
        if operation in {'generate', 'edit'}:
            queue = fixture.executor.media_jobs
            queue.running = True
            assert await queue.tick()
            assert len(requests) == initial_requests
            assert await queue.tick()
            assert len(writes(fixture)) == 2
            assert requests[0]['tools'] == requests[-1]['tools']
        assert len(fixture.image_client.calls) == (operation == "generate")
        assert len(fixture.image_client.edit_calls) == (operation == "edit")
        if operation == "edit":
            assert fixture.image_client.edit_calls[0]["images"]
