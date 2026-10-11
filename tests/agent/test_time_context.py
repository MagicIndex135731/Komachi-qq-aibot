from datetime import UTC, datetime

import httpx
import pytest

from app.agent.contracts import AgentBatch
from app.agent.observation import GroupObservationService
from app.agent.social_contracts import SocialObservation
from app.storage.db import session_scope
from app.storage.models import Message
from test_observation import make_router
from test_autonomous_social import ScriptedModel, PolicyGuard, add_source, composed
from test_harness_runtime import bridge_factory, ModelHandler, request_body, responses_sse, chat_sse


CURRENT = datetime(2026, 10, 10, 19, 0, tzinfo=UTC)
OLD = datetime(2026, 10, 10, 14, 0, tzinfo=UTC)


def test_recent_group_messages_expose_source_date(sqlite_engine, tmp_path):
    router, event, _ = make_router(sqlite_engine, tmp_path)
    with session_scope(sqlite_engine) as session:
        row = session.query(Message).filter_by(platform_msg_id=event.platform_msg_id).one()
        row.timestamp = datetime(2026, 10, 10, 22, 0)
        row.plain_text = "今天看合成动画"
    value = GroupObservationService(router).build_agent_observation(AgentBatch(event.group_id, (event,), "synthetic-time"))
    assert "2026-10-10T22:00:00+08:00" in value["recent_lines"][0]
    assert "今天看合成动画" in value["recent_lines"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["group", "qzone_publish"])
async def test_turn_uses_execution_clock_not_old_observation(sqlite_engine, tmp_path, kind):
    model = ScriptedModel({"tool": "read_context", "arguments": {}}) if kind == "group" else ScriptedModel()
    async with composed(sqlite_engine, tmp_path, model, kind=kind) as fixture:
        fixture.clock[0] = CURRENT
        job = next(job for job in fixture.agent.jobs() if job.kind == kind)
        observation = SocialObservation(job, OLD, "合成人格")
        await fixture.executor.run_tools(observation, PolicyGuard(fixture.executor, observation))
        prompt = "\n".join(model.prompts[0])
        assert '"today": "2026-10-11"' in prompt
        assert '"yesterday": "2026-10-10"' in prompt
        assert '"observed_at": "2026-10-10T22:00:00+08:00"' in prompt


@pytest.mark.asyncio
async def test_history_lookup_reports_actual_source_time(sqlite_engine, tmp_path):
    source = add_source(sqlite_engine, platform_id="synthetic-old-history", text="今天看合成动画")
    with session_scope(sqlite_engine) as session:
        session.get(Message, source).timestamp = datetime(2026, 10, 10, 22)
    model = ScriptedModel({"tool": "search_history", "arguments": {"query": "合成动画"}})
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        fixture.clock[0] = CURRENT
        job = next(job for job in fixture.agent.jobs() if job.kind == "group")
        observation = SocialObservation(job, OLD, "合成人格", source_message_ids=(source,))
        await fixture.executor.run_tools(observation, PolicyGuard(fixture.executor, observation))
        assert '2026-10-10T22:00:00+08:00' in "\n".join(model.prompts[1])


@pytest.mark.parametrize("protocol", ["openai-responses", "openai-completions"])
def test_every_model_request_refreshes_shanghai_day(bridge_factory, protocol):
    bridge, requests, _ = bridge_factory(lambda *_: responses_sse() if protocol == "openai-responses" else chat_sse(), protocol=protocol)
    clock = [datetime(2026, 10, 10, 15, 59, 59, tzinfo=UTC)]
    bridge.owner.clock = lambda: clock[0]
    for _ in range(2):
        bridge.forward(ModelHandler(), request_body(protocol))
        clock[0] = datetime(2026, 10, 10, 16, 0, 0, tzinfo=UTC)
    def anchor(body):
        value = body["input"][-1]["content"][0]["text"] if protocol == "openai-responses" else body["messages"][-1]["content"]
        return value
    assert '"today":"2026-10-10"' in anchor(requests[0])
    assert '"today":"2026-10-11"' in anchor(requests[1])
    assert '"yesterday":"2026-10-10"' in anchor(requests[1])
    if protocol == "openai-responses":
        assert requests[0]["instructions"] == requests[1]["instructions"]
        assert requests[0]["input"][:-1] == requests[1]["input"][:-1]
        assert requests[1]["input"][-1]["role"] == "developer"
    else:
        assert requests[0]["messages"][:-1] == requests[1]["messages"][:-1]
        assert requests[1]["messages"][-1]["role"] == "user"


@pytest.mark.parametrize("protocol", ["openai-responses", "openai-completions"])
def test_transport_retry_crossing_midnight_refreshes_once(bridge_factory, protocol):
    clock = [datetime(2026, 10, 10, 15, 59, 59, tzinfo=UTC)]
    def transport(body, index):
        if index == 1:
            clock[0] = datetime(2026, 10, 10, 16, tzinfo=UTC)
            return httpx.Response(503, json={"error": "synthetic_retry"})
        return responses_sse() if protocol == "openai-responses" else chat_sse()
    bridge, requests, _ = bridge_factory(transport, protocol=protocol)
    bridge.owner.clock = lambda: clock[0]
    body = request_body(protocol)
    if protocol == "openai-responses":
        body["instructions"] = "synthetic stable persona"
    bridge.forward(ModelHandler(), body)
    assert len(requests) == 2
    anchors = [r["input"][-1]["content"][0]["text"] if protocol == "openai-responses" else r["messages"][-1]["content"] for r in requests]
    assert '"today":"2026-10-10"' in anchors[0]
    assert '"today":"2026-10-11"' in anchors[1]
    assert anchors[1].count("Current time JSON:") == 1
    if protocol == "openai-responses":
        assert requests[1]["instructions"].startswith("synthetic stable persona")
        assert requests[0]["instructions"] == requests[1]["instructions"]
        assert requests[0]["input"][:-1] == requests[1]["input"][:-1]
    else:
        assert len(requests[1]["messages"]) == 3
        assert requests[0]["messages"][:-1] == requests[1]["messages"][:-1]


@pytest.mark.parametrize("protocol", ["openai-responses", "openai-completions"])
def test_clock_does_not_split_or_rewrite_replayed_tool_pairs(bridge_factory, protocol):
    bridge, requests, _ = bridge_factory(lambda *_: responses_sse() if protocol == "openai-responses" else chat_sse(), protocol=protocol)
    body = request_body(protocol)
    if protocol == "openai-responses":
        body["input"].extend([
            {"type": "function_call", "call_id": "synthetic-call", "name": "read_context", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "synthetic-call", "output": "synthetic closed result"}])
        replay = list(body["input"])
    else:
        body["messages"].extend([
            {"role": "assistant", "content": "", "tool_calls": [{"id": "synthetic-call", "type": "function", "function": {"name": "read_context", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "synthetic-call", "content": "synthetic closed result"}])
        replay = list(body["messages"])
    bridge.forward(ModelHandler(), body)
    if protocol == "openai-responses":
        assert requests[0]["input"][:-1] == replay
    else:
        assert requests[0]["messages"][1:-1] == replay


@pytest.mark.parametrize("protocol", ["openai-responses", "openai-completions"])
def test_dynamic_persona_guidance_keeps_authority_after_replay(bridge_factory, protocol):
    bridge, requests, _ = bridge_factory(lambda *_: responses_sse() if protocol == "openai-responses" else chat_sse(), protocol=protocol)
    bridge.dynamic_instructions = "Current persona guidance:\nsynthetic live style"
    body = request_body(protocol)
    bridge.forward(ModelHandler(), body)
    if protocol == "openai-responses":
        guidance = requests[0]["input"][-2]
        assert guidance["role"] == "developer"
        assert guidance["content"][0]["text"] == bridge.dynamic_instructions
        assert bridge.dynamic_instructions not in requests[0]["instructions"]
    else:
        guidance = requests[0]["messages"][-2]
        assert guidance["role"] == "system"
        assert guidance["content"] == bridge.dynamic_instructions
    bridge.guard.allowed = False
    with pytest.raises(RuntimeError):
        bridge.forward(ModelHandler(), body)
    assert len(requests) == 1
