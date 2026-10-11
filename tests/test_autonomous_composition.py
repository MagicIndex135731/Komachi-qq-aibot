"""Validate extension permissions and transport ownership at the factory boundary."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from app.core.router import InboundRouter
from app.providers.llm_client import LlmClient
from app.runtime import social
from app.runtime.blocking import AgentModelBusyError


GROUP = 900000001
BOT = 900000101


def reject_network(_request):
    raise AssertionError("composition must not issue a network request")


def model_client(name):
    return LlmClient(
        base_url="https://synthetic.example.test/v1", api_key="synthetic-key",
        model=name, http_client=httpx.Client(transport=httpx.MockTransport(reject_network)),
    )


@pytest.fixture
def configured(sqlite_engine, tmp_path):
    primary = model_client("synthetic-primary")
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine, sender=object(), llm_client=primary,
    )
    router.runtime.settings = router.runtime.settings.model_copy(update={
        "bot_qq": BOT, "data_dir": tmp_path / "data", "social_agent_enabled": True,
        "autonomous_sandbox_token": "synthetic-sandbox-token",
    })
    router.runtime.group_policy = {"groups": {str(GROUP): {
        "enabled": True, "speak": True, "proactive_reply": True,
    }}}
    router.runtime.social_policy = {"enabled": True, "groups": {str(GROUP): {
        "enabled": True, "autonomous_tools": True, "quiet_hours": None,
    }}}
    try:
        yield SimpleNamespace(router=router, primary=primary)
    finally:
        primary.http_client.close()


def compose(configured):
    return social.build_social_agent(
        router=configured.router, sender=object(), gateway=object(),
        llm_client=configured.primary,
    )


def test_configured_turn_deadline_and_account_lease_cover_mature_agent_execution(configured):
    settings = configured.router.runtime.settings
    settings.agent_runtime = "harness"
    settings.group_agent_turn_timeout_seconds = 600
    settings.agent_background_turn_timeout_seconds = 750
    agent, planner = compose(configured)
    try:
        assert agent.options.execution_timeout_seconds == 750
        assert agent.options.lease_seconds >= 750 + settings.social_agent_decision_timeout_seconds + 180
    finally:
        planner.close()


def manifest(*names, scopes=("group",)):
    return {
        "enabled": True, "url": "https://synthetic.example.test/mcp",
        "tools": [{"name": name, "scopes": list(scopes), "effect": "read",
                   "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}
                  for name in names],
    }


def forbid_allocations(monkeypatch):
    allocations = []

    def reject(*_args, **_kwargs):
        allocations.append("unexpected allocation")
        raise AssertionError("invalid extension permissions must reject before allocation")

    monkeypatch.setattr(social, "build_proactive_judge_client", reject)
    monkeypatch.setattr(social, "AutonomousSocialPlanner", reject)
    monkeypatch.setattr(social, "AutonomousSocialExecutor", reject)
    monkeypatch.setattr(httpx, "AsyncClient", reject)
    return allocations


@pytest.mark.parametrize("token", ["", "two words", "line\nbreak", "line\rbreak", "非ASCII", "x" * 257],
                         ids=["empty", "space", "newline", "carriage-return", "unicode", "oversize"])
def test_invalid_sandbox_token_rejects_before_judge_pool_or_transport(configured, monkeypatch, token):
    configured.router.runtime.settings = configured.router.runtime.settings.model_copy(
        update={"autonomous_sandbox_token": token},
    )
    configured.router.runtime.social_policy["compute"] = {
        "enabled": True, "endpoint": "http://127.0.0.1:8217",
    }
    allocations = forbid_allocations(monkeypatch)
    with pytest.raises(ValueError, match="authentication token"):
        compose(configured)
    assert allocations == []
    assert not configured.primary.http_client.is_closed


@pytest.mark.parametrize("manifests", [
    [manifest("extension_a", "extension_a")],
    [manifest("extension_a"), manifest("extension_a")],
    [manifest(*(f"extension_{i}" for i in range(8))), manifest("extension_8", "extension_9", "extension_10")],
], ids=["within-provider-duplicate", "between-provider-duplicate", "eleven-in-one-scope"])
def test_duplicate_or_overbudget_mcp_rejects_before_allocation(configured, monkeypatch, manifests):
    configured.router.runtime.social_policy["mcp"] = manifests
    allocations = forbid_allocations(monkeypatch)
    with pytest.raises(ValueError, match="duplicate MCP capability or scoped tool budget"):
        compose(configured)
    assert allocations == []


@pytest.mark.parametrize("name", [
    "remember_intention", "list_artifacts", "read_artifact", "read_context", "search_history",
    "generate_image", "send_group", "publish_qzone", "comment_qzone", "set_nickname",
    "set_group_card", "set_avatar", "execute_python", "search_web",
])
def test_mcp_cannot_shadow_builtin_agent_tool(configured, monkeypatch, name):
    configured.router.runtime.social_policy["mcp"] = [manifest(name)]
    allocations = forbid_allocations(monkeypatch)
    with pytest.raises(ValueError, match="built-in capability"):
        compose(configured)
    assert allocations == []


def trace_async_transports(monkeypatch):
    original = httpx.AsyncClient
    clients = []

    class TracedClient(original):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(reject_network)
            super().__init__(*args, **kwargs)
            self.closed_event = asyncio.Event()
            clients.append(self)

        async def aclose(self):
            try:
                await super().aclose()
            finally:
                self.closed_event.set()

    monkeypatch.setattr(httpx, "AsyncClient", TracedClient)
    return clients


def trace_pools(monkeypatch, *, failure=""):
    planners, executors = [], []
    original_planner, original_executor = social.AutonomousSocialPlanner, social.AutonomousSocialExecutor

    class TracedPlanner(original_planner):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            planners.append(self)

    class TracedExecutor(original_executor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            executors.append(self)

        def register_tool_factory(self, factory):
            if failure == "register":
                raise RuntimeError("synthetic registration failure")
            return super().register_tool_factory(factory)

    monkeypatch.setattr(social, "AutonomousSocialPlanner", TracedPlanner)
    monkeypatch.setattr(social, "AutonomousSocialExecutor", TracedExecutor)
    return planners, executors


async def assert_pools_closed(planners, executors):
    assert len(planners) == len(executors) == 1
    for resource in (*planners, *executors):
        with pytest.raises(AgentModelBusyError):
            await resource.blocking.run(lambda: "unexpected new task")


@pytest.mark.asyncio
@pytest.mark.parametrize("owned_judge", [True, False], ids=["owned-judge", "shared-primary"])
@pytest.mark.parametrize("failure", ["runtime", "register"], ids=["runtime-construction", "tool-registration"])
async def test_failed_composition_closes_all_owned_resources_without_closing_primary(
    configured, monkeypatch, owned_judge, failure,
):
    configured.router.runtime.social_policy.update({
        "compute": {"enabled": True, "endpoint": "http://127.0.0.1:8217"},
        "mcp": [manifest("extension_lookup")],
    })
    judge = model_client("synthetic-judge") if owned_judge else configured.primary
    monkeypatch.setattr(social, "build_proactive_judge_client", lambda **_kwargs: judge)
    clients = trace_async_transports(monkeypatch)
    planners, executors = trace_pools(monkeypatch, failure=failure)

    def reject_runtime(**_kwargs):
        raise RuntimeError("synthetic runtime construction failure")

    if failure == "runtime":
        monkeypatch.setattr(social, "SocialRuntime", reject_runtime)
    try:
        with pytest.raises(RuntimeError, match="synthetic"):
            compose(configured)
        await assert_pools_closed(planners, executors)
        assert len(clients) == 2  # Sandbox HTTP + MCP; pure prevalidation allocated neither.
        await asyncio.wait_for(asyncio.gather(*(client.closed_event.wait() for client in clients)), timeout=1)
        assert all(client.is_closed for client in clients)
        assert not configured.primary.http_client.is_closed
        if owned_judge:
            assert judge.http_client.is_closed
    finally:
        for resource in (*planners, *executors):
            resource.blocking.close()
        for client in clients:
            await client.aclose()
        if owned_judge:
            judge.http_client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("owned_judge", [True, False], ids=["owned-judge", "shared-primary"])
async def test_valid_composition_has_per_scope_mcp_budget_and_closes_owned_clients(
    configured, monkeypatch, owned_judge,
):
    # Ten group tools plus six publish tools are within each scope's budget.
    # A matching extension name in different scopes does not conflict.
    configured.router.runtime.social_policy.update({
        "compute": {"enabled": True, "endpoint": "http://127.0.0.1:8217"},
        "qzone": {"publish": {"enabled": True, "autonomous_tools": True, "topics": ["合成音乐"]}},
        "mcp": [manifest(*(f"extension_{i}" for i in range(8))), manifest("extension_8", "extension_9"),
                manifest(*(f"extension_{i}" for i in range(6)), scopes=("qzone_publish",))],
    })
    judge = model_client("synthetic-judge") if owned_judge else configured.primary
    monkeypatch.setattr(social, "build_proactive_judge_client", lambda **_kwargs: judge)
    clients = trace_async_transports(monkeypatch)
    planners, executors = trace_pools(monkeypatch)
    agent, planner = compose(configured)
    try:
        assert agent is not None and not agent.running
        assert planner.executor.decision_client is configured.primary
        assert len(clients) == 4  # One sandbox transport and three approved MCP providers.
        await planner.aclose()
        await assert_pools_closed(planners, executors)
        assert all(client.is_closed for client in clients)
        assert not configured.primary.http_client.is_closed
        if owned_judge:
            assert judge.http_client.is_closed
    finally:
        await planner.aclose()
        if owned_judge:
            judge.http_client.close()
