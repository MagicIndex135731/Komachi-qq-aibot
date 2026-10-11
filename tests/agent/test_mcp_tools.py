import json
from types import SimpleNamespace

import httpx
import pytest

from app.agent.mcp_tools import McpToolProvider
from app.agent.tool_agent import AutonomousToolAgent


class Guard:
    def __init__(self):
        self.dispatched = 0
        self.revoked = False
    def valid(self):
        return not self.revoked
    def require_valid(self):
        if self.revoked:
            raise RuntimeError("revoked")
    def on_dispatch(self):
        self.require_valid()
        self.dispatched += 1


def manifest(effect="read"):
    return {"enabled": True, "url": "https://synthetic.example.test/mcp", "tools": [{
        "name": "calculate", "description": "external computation", "effect": effect,
        "scopes": ["group"], "parameters": {"type": "object", "properties": {"expression": {"type": "string", "maxLength": 100}}, "required": ["expression"]},
        "acknowledgement_key": "receipt"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("sse", [False, True])
async def test_mcp_extension_negotiates_scope_and_returns_real_feedback(sse):
    calls = []
    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if payload["method"] == "notifications/initialized":
            assert request.headers["Mcp-Session-Id"] == "synthetic-session"
            return httpx.Response(202)
        result = {"protocolVersion": "2025-06-18"} if payload["method"] == "initialize" else {"content": [{"type": "text", "text": "42"}]}
        reply = {"jsonrpc": "2.0", "id": payload["id"], "result": result}
        return httpx.Response(200, headers={"Content-Type": "text/event-stream" if sse else "application/json", "Mcp-Session-Id": "synthetic-session"},
                              text="data: " + json.dumps(reply) + "\n\n" if sse else json.dumps(reply))
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = McpToolProvider(manifest(), client=client)
        guard = Guard()
        obs = SimpleNamespace(job=SimpleNamespace(kind="group"))
        tool, = provider.tools_for(obs, guard)
        outcome = await tool.handler({"expression": "6*7"})
        assert outcome.data["mcp_result"]["content"][0]["text"] == "42"
        assert guard.dispatched == 0 and len(calls) == 3
        assert provider.tools_for(SimpleNamespace(job=SimpleNamespace(kind="profile")), guard) == []


@pytest.mark.asyncio
async def test_mcp_write_unknown_never_replays_and_guard_precedes_submission():
    calls = []
    guard = Guard()
    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload["method"])
        if payload["method"] == "initialize":
            return httpx.Response(200, json={"id": payload["id"], "result": {"protocolVersion": "2025-06-18"}})
        if payload["method"] == "notifications/initialized":
            return httpx.Response(202)
        assert guard.dispatched == 1
        raise httpx.ReadTimeout("synthetic acknowledgement loss")
    class Model:
        calls = 0
        def generate_text(self, lines, **kwargs):
            self.calls += 1
            return '{"tool":"calculate","arguments":{"expression":"write"}}'
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = McpToolProvider(manifest("write"), client=client)
        model = Model()
        agent = AutonomousToolAgent(decision_client=model)
        try:
            result = await agent.run([], tuple(provider.tools_for(SimpleNamespace(job=SimpleNamespace(kind="group")), guard)), guard)
            assert result.status == "uncertain" and model.calls == 1
            assert calls.count("tools/call") == 1
        finally:
            agent.close()


@pytest.mark.asyncio
async def test_revoked_mcp_does_not_submit_and_unknown_schema_fields_fail_closed():
    guard = Guard()
    guard.revoked = True
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: pytest.fail("network after revocation"))) as client:
        provider = McpToolProvider(manifest(), client=client)
        tool, = provider.tools_for(SimpleNamespace(job=SimpleNamespace(kind="group")), guard)
        with pytest.raises(RuntimeError):
            await tool.handler({"expression": "x"})


@pytest.mark.parametrize("url", ["http://foreign.example.test/mcp", "file:///tmp/server", "https://name:secret@example.test/mcp", "https://example.test/mcp?token=secret"])
def test_mcp_untrusted_endpoint_shapes_rejected(url):
    config = manifest()
    config["url"] = url
    with pytest.raises(ValueError):
        McpToolProvider(config, client=object())
