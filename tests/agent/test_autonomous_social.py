"""Exercise the scoped production tool composition with synthetic providers."""

import asyncio
import base64
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import httpx
from PIL import Image
import pytest
from sqlalchemy import select

from app.adapters.sender import Sender
from app.agent.artifacts import AutonomousArtifactStore
from app.agent.social_contracts import SocialDecision, SocialObservation, SocialPost
from app.agent.contracts import AgentBatch, AgentDecision
from app.agent.participation import GroupParticipationPlanner
from app.adapters.onebot_models import parse_group_message_event
from app.agent.social_policy import configured_social_jobs
from app.core.router import InboundRouter
from app.providers.image_adapter import ImageArtifact, ImageGenerationResult
from app.providers.llm_client import LlmClient
from app.runtime.social import build_social_agent
from app.runtime.blocking import AgentModelBusyError
from app.storage.autonomous_agent import AutonomousIntention, AutonomousUsage
from app.storage.agent_effects import AgentToolEffect
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository
from app.storage.social_agent import SocialActionRecord


BOT = 900000101
GROUP = 900000001
OTHER_GROUP = 900000002
FRIEND = 900000201
NOW = datetime(2026, 10, 9, 8, 0, tzinfo=UTC)


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (32, 24), "red").save(output, "PNG")
    return output.getvalue()


def feedback(lines):
    return json.loads(next(line for line in lines if line.startswith("Tool results JSON")).split("\n", 1)[1])


def observation_json(lines):
    return json.loads(next(line for line in lines if line.startswith("Current observation JSON")).split("\n", 1)[1])


def last_artifact(lines):
    return next(item["result"]["artifact_id"] for item in reversed(feedback(lines)) if "artifact_id" in item.get("result", {}))


def tool_result(lines, name):
    return next(item for item in reversed(feedback(lines)) if item.get("tool") == name)


def artifact_entries(lines, name="list_artifacts"):
    return tool_result(lines, name)["result"]["artifacts"]


def available_tools(lines):
    raw = next(line for line in lines if line.startswith("Available tools JSON")).split("\n", 1)[1]
    return {item["name"] for item in json.loads(raw)}


class ScriptedModel:
    def __init__(self, *commands):
        self.commands = list(commands)
        self.prompts, self.images = [], []

    def generate_text(self, lines, **kwargs):
        self.prompts.append(lines)
        self.images.append(tuple(kwargs.get("images", ())))
        command = self.commands.pop(0) if self.commands else {"finish": True}
        if callable(command):
            command = command(lines, kwargs)
        return json.dumps(command, ensure_ascii=False)


class SyntheticImageClient:
    def __init__(self, callback=None):
        self.calls = []
        self.edit_calls = []
        self.callback = callback

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        if self.callback:
            self.callback()
        return ImageGenerationResult(None, artifacts=[ImageArtifact(b64_json=base64.b64encode(image_bytes()).decode("ascii"))])

    def edit_image(self, **kwargs):
        self.edit_calls.append(kwargs)
        if self.callback:
            self.callback()
        return ImageGenerationResult(None, artifacts=[ImageArtifact(b64_json=base64.b64encode(image_bytes()).decode("ascii"))])


class SyntheticGateway:
    websocket = object()

    def __init__(self):
        self.calls = []
        self.write_response = None
        self.posts = [{"tid": "observed-photo", "content": "今天拍了猫", "time": int(NOW.timestamp()) - 30,
                       "comment_num": 0, "is_private": False, "images": []}]

    async def call_api(self, action, params):
        self.calls.append((action, params))
        if action == "get_qzone_msg_list":
            return {"status": "ok", "retcode": 0, "data": {"total": len(self.posts), "msglist": self.posts}}
        if action == "get_qzone_feeds":
            return {"status": "ok", "retcode": 0, "data": {"has_more": False, "feeds": [
                {"uin": FRIEND, "nickname": "合成好友", "appid": 311, "key": post["tid"], "time": post["time"],
                 "html": f'<li data-uin="{FRIEND}" data-tid="{post["tid"]}">公开动态</li>'} for post in self.posts]}}
        if self.write_response is not None:
            return self.write_response
        if action == "send_group_msg":
            data = {"message_id": "actual-group-receipt"}
        elif action == "send_qzone_msg":
            data = {"tid": "actual-published-tid", "time": int(NOW.timestamp())}
        elif action == "comment_qzone":
            data = {"comment_id": ""}
        elif action in {"set_qq_profile", "set_qq_avatar", "set_group_card"}:
            data = None
        else:
            raise AssertionError(action)
        return {"status": "ok", "retcode": 0, "data": data}


@asynccontextmanager
async def composed(engine, tmp_path, model, *, kind="group", options=None, image_callback=None, compute=False, runtime="legacy"):
    gateway = SyntheticGateway()
    sender = Sender(gateway)
    image_client = SyntheticImageClient(image_callback)
    router = InboundRouter.build_for_test(sqlite_engine=engine, sender=sender, llm_client=model)
    router.runtime.settings = router.runtime.settings.model_copy(update={
        "bot_qq": BOT, "social_agent_enabled": True, "data_dir": tmp_path / "isolated",
        "owner_qq": 900000999,
        "agent_runtime": runtime,
    })
    router.runtime.persona = {"name": "小町", "core_traits": ["好奇"], "facts": ["PRIVATE_PERSONA_FACT"]}
    router.runtime.group_policy = {"groups": {str(GROUP): {
        "enabled": True, "speak": True, "proactive_reply": True, "memory_enabled": False,
        "proactive_interval_seconds": "60-60",
    }}}
    common = {"enabled": True, "autonomous_tools": True, "allow_images": True,
              "interval_seconds": 60, "cooldown_seconds": 60, "max_daily": 10,
              "min_idle_seconds": 0, "quiet_hours": None, "topics": ["猫与绘画"], **(options or {})}
    if kind == "group":
        policy = {"groups": {str(GROUP): common}}
    elif kind == "profile":
        policy = {"profile": {**common, "identity_tools": ["nickname", "avatar"]}}
    else:
        name = "publish" if kind == "qzone_publish" else "comment"
        policy = {"qzone": {name: {**common, "topics": ["猫与绘画"], "friend_ids": [FRIEND]}}}
    router.runtime.social_policy = {"enabled": True, **policy}
    if compute:
        router.runtime.social_policy["compute"] = {"enabled": True}
    router.group_image_service = SimpleNamespace(llm_client=image_client, model="synthetic-image-model", size="1024x1024",
        quality="low", output_format="png", image_timeout_seconds=30)
    agent, planner = build_social_agent(router=router, sender=sender, gateway=gateway, llm_client=model)
    clock = [NOW]
    agent.clock = lambda: clock[0]
    planner.executor.clock = lambda: clock[0]
    planner.executor.legacy.clock = lambda: clock[0]
    planner.executor.repository.clock = lambda: clock[0]
    planner.executor.sessions.clock = lambda: clock[0]
    fixture = SimpleNamespace(agent=agent, planner=planner, executor=planner.executor, router=router,
                              gateway=gateway, model=model, image_client=image_client, clock=clock)
    try:
        await agent.start()
        agent.set_connected(True)
        await agent.tick(clock[0])
        yield fixture
    finally:
        await agent.stop()
        planner.close()


async def wake(fixture):
    fixture.clock[0] += timedelta(seconds=61)
    await fixture.agent.tick(fixture.clock[0])


def actions(engine):
    with session_scope(engine) as session:
        return [(row.status, row.reason, row.receipt_id) for row in session.scalars(select(SocialActionRecord))]


def writes(fixture):
    return [(action, args) for action, args in fixture.gateway.calls if not action.startswith("get_")]


async def test_group_native_web_body_open_find_and_send_are_wired(sqlite_engine, tmp_path):
    from test_harness_runtime import model_client, responses_sse
    from app.providers.web_search import WebSearchClient
    web_calls = []
    def web(request):
        commands = json.loads(request.content)["commands"]
        web_calls.append(commands)
        return httpx.Response(200, json={"output": "Source (https://example.com/page)\nciteturn0view0\nL0: DETAILED_GROUP_EVIDENCE",
            "results": [{"type": "text_result", "url": "https://example.com/page", "ref_id": "turn0view0", "snippet": "short"}]})
    commands = [("search_web", {"query": "public question", "max_results": 8}),
        ("open_web_page", {"ref_id": "https://example.com/page", "lineno": 0}),
        ("find_web_page", {"ref_id": "https://example.com/page", "pattern": "EVIDENCE"}),
        ("send_group", {"text": "已核对公开正文", "artifact_ids": []})]
    client, requests, _ = model_client(lambda body, n: responses_sse(name=commands[n-1][0], arguments=commands[n-1][1],
        call_id=f"web-{n}", number=n) if n <= len(commands) else responses_sse(number=n))
    async with composed(sqlite_engine, tmp_path, client, runtime="harness") as fixture:
        fixture.router.web_search_client = WebSearchClient(provider="wyzlab_alpha", base_url="https://synthetic.invalid/v1",
            api_key="synthetic", timeout_seconds=45, http_client=httpx.Client(transport=httpx.MockTransport(web)))
        source = add_source(sqlite_engine, platform_id='current-web-request', text='请联网查证公开问题')
        with session_scope(sqlite_engine) as session:
            session.get(Message, source).raw_json = {'message': [{'type': 'text', 'data': {'text': '请联网查证公开问题'}}]}
        job = configured_social_jobs(fixture.router.runtime)[0]
        observed = SocialObservation(job, NOW, '小町', source_message_ids=(source,))
        guard = SimpleNamespace(action_id='current-web-request', valid=lambda: True,
            require_valid=lambda: None, on_dispatch=lambda: None)
        await fixture.executor.run_tools(observed, guard, interactive=True,
            target_user_id=FRIEND, target_message_id='current-web-request', direct_group_reply=True)
        assert len(writes(fixture)) == 1
        assert len(web_calls) == 3 and len(requests) == 4
        assert "DETAILED_GROUP_EVIDENCE" in json.dumps(requests[1], ensure_ascii=False)
        assert {"search_web", "open_web_page", "find_web_page"} <= {v["name"] for v in requests[0]["tools"]}
        assert web_calls[1]["open"][0]["ref_id"] == "https://example.com/page"
        assert web_calls[2]["find"][0]["pattern"] == "EVIDENCE"


def add_source(engine, *, platform_id, group=GROUP, text="合成来源", images=()):
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(group_id=group, group_name="synthetic", enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=FRIEND, nickname="synthetic author", group_card="")
        row = MessageRepository(session).add_group_message(platform_msg_id=platform_id, group_id=group, user_id=FRIEND,
            timestamp=NOW - timedelta(hours=1), plain_text=text, msg_type="image" if images else "text",
            reply_to_msg_id=None, mentioned_bot=False,
            raw_json={"message": [{"type": "image", "data": image} for image in images]})
        session.flush()
        return row.id


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,tool,api,receipt", [
    ("group", "send_group", "send_group_msg", "actual-group-receipt"),
    ("qzone_publish", "publish_qzone", "send_qzone_msg", "actual-published-tid"),
])
async def test_composed_agent_generates_and_sends_owned_pixels_with_real_ack(sqlite_engine, tmp_path, kind, tool, api, receipt):
    model = ScriptedModel({"tool": "generate_image", "arguments": {"prompt": "画一只好奇的红色猫"}},
        lambda lines, _: {"tool": tool, "arguments": {"text": "突然想画一只猫", "artifact_ids": [last_artifact(lines)]}})
    async with composed(sqlite_engine, tmp_path, model, kind=kind) as fixture:
        await wake(fixture)
        assert len(fixture.image_client.calls) == 1
        assert fixture.image_client.calls[0]["max_attempts"] == 1
        assert len(writes(fixture)) == 1 and writes(fixture)[0][0] == api
        sent = writes(fixture)[0][1]
        encoded = next(item["data"]["file"] for item in sent["message"] if item["type"] == "image") if kind == "group" else sent["images"][0]
        assert encoded.startswith("base64://")
        assert Image.open(io.BytesIO(base64.b64decode(encoded[9:]))).getpixel((0, 0)) == (255, 0, 0)
        assert actions(sqlite_engine) == [("sent", "tool:" + tool, receipt)]
        assert len(model.prompts) == 2 and len(model.images[1]) == 1
        assert Image.open(model.images[1][0].local_path).getpixel((0, 0)) == (255, 0, 0)
        generated = list((fixture.router.runtime.settings.data_dir / "autonomous_artifacts").glob("art-*.meta.json"))
        assert len(generated) == 1
        metadata = json.loads(generated[0].read_text(encoding="utf-8"))
        assert metadata["origin"] == "autonomous_generated" and metadata["scope"] == ("group:" + str(GROUP) if kind == "group" else kind)
        assert metadata["source_message_ids"] == []
        if kind == "qzone_publish":
            assert "PRIVATE_PERSONA_FACT" not in "\n".join(model.prompts[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["set_nickname", "set_avatar"])
async def test_composed_profile_ack_is_honest_and_terminal(sqlite_engine, tmp_path, tool):
    if tool == "set_nickname":
        model = ScriptedModel({"tool": tool, "arguments": {"name": "小町的绘画时间"}})
        api = "set_qq_profile"
    else:
        model = ScriptedModel({"tool": "generate_image", "arguments": {"prompt": "红色猫头像"}},
                              lambda lines, _: {"tool": tool, "arguments": {"artifact_id": last_artifact(lines)}})
        api = "set_qq_avatar"
    async with composed(sqlite_engine, tmp_path, model, kind="profile") as fixture:
        await wake(fixture)
        assert [call[0] for call in writes(fixture)] == [api]
        assert actions(sqlite_engine) == [("sent", "profile_ack:" + tool, None)]
        assert len(model.prompts) == (1 if tool == "set_nickname" else 2)


@pytest.mark.asyncio
async def test_unknown_profile_ack_is_uncertain_with_no_retry(sqlite_engine, tmp_path):
    model = ScriptedModel({"tool": "set_nickname", "arguments": {"name": "小町"}},
                          {"tool": "set_nickname", "arguments": {"name": "重试"}})
    async with composed(sqlite_engine, tmp_path, model, kind="profile") as fixture:
        fixture.gateway.write_response = {"status": "ok", "data": {}}
        await wake(fixture)
        assert len(writes(fixture)) == 1
        assert actions(sqlite_engine)[0][0] == "uncertain"
        assert len(model.prompts) == 1


@pytest.mark.asyncio
async def test_image_budget_bounds_real_handler_across_wakes(sqlite_engine, tmp_path):
    model = ScriptedModel({"tool": "generate_image", "arguments": {"prompt": "红色猫"}}, {"finish": True},
                          {"tool": "generate_image", "arguments": {"prompt": "蓝色猫"}}, {"finish": True})
    async with composed(sqlite_engine, tmp_path, model, options={"max_image_calls_daily": 1}) as fixture:
        await wake(fixture)
        await wake(fixture)
        assert len(fixture.image_client.calls) == 1 and writes(fixture) == []
        assert "image_budget_exhausted" in "\n".join(model.prompts[-1])
        with session_scope(sqlite_engine) as session:
            assert len(list(session.scalars(select(AutonomousUsage).where(AutonomousUsage.kind == "image:group_autonomous")))) == 1


@pytest.mark.asyncio
async def test_model_budget_blocks_next_decision_before_any_send(sqlite_engine, tmp_path):
    model = ScriptedModel({"tool": "read_context", "arguments": {}},
                          {"tool": "send_group", "arguments": {"text": "超过预算", "artifact_ids": []}})
    async with composed(sqlite_engine, tmp_path, model, options={"max_model_calls_daily": 1}) as fixture:
        await wake(fixture)
        assert len(model.prompts) == 1 and writes(fixture) == []
        assert any(status == "blocked" and reason == "model_daily_budget" for status, reason, _ in actions(sqlite_engine))


@pytest.mark.asyncio
async def test_cross_group_and_foreign_artifact_arguments_never_reach_gateway(sqlite_engine, tmp_path):
    model = ScriptedModel({"tool": "send_group", "arguments": {"text": "跨群", "artifact_ids": [], "group_id": OTHER_GROUP}},
                          {"tool": "send_group", "arguments": {"text": "伪造图片", "artifact_ids": ["not-owned"]}},
                          {"tool": "send_group", "arguments": {"text": "重试", "artifact_ids": []}})
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        await wake(fixture)
        assert writes(fixture) == []
        assert actions(sqlite_engine)[0][0] == "blocked"
        assert len(model.prompts) == 2


@pytest.mark.asyncio
async def test_code_tool_is_not_available_without_operator_owned_sandbox(sqlite_engine, tmp_path):
    model = ScriptedModel({"tool": "execute_python", "arguments": {"code": "print(6*7)"}}, {"finish": True})
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        await wake(fixture)
        assert writes(fixture) == []
        advertised = json.loads(next(line for line in model.prompts[0] if line.startswith("Available tools JSON:")).split("\n", 1)[1])
        assert "execute_python" not in {tool["name"] for tool in advertised}
        assert "tool_or_arguments_not_allowed" in "\n".join(model.prompts[1])


@pytest.mark.asyncio
async def test_permission_revocation_after_generation_prevents_send(sqlite_engine, tmp_path):
    model = ScriptedModel({"tool": "generate_image", "arguments": {"prompt": "红色猫"}},
                          lambda lines, _: {"tool": "send_group", "arguments": {"text": "画好啦", "artifact_ids": [last_artifact(lines)]}})
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        fixture.image_client.callback = lambda: fixture.router.runtime.social_policy["groups"][str(GROUP)].update(allow_images=False)
        await wake(fixture)
        assert len(fixture.image_client.calls) == 1 and writes(fixture) == []
        assert actions(sqlite_engine)[0][0] != "sent"
        assert len(model.prompts) == 1


@pytest.mark.asyncio
async def test_pure_qzone_picture_without_pixels_is_not_commented(sqlite_engine, tmp_path):
    model = ScriptedModel({"tool": "comment_qzone", "arguments": {"text": "红色猫真可爱", "artifact_ids": []}})
    async with composed(sqlite_engine, tmp_path, model, kind="qzone_comment") as fixture:
        fixture.gateway.posts[0].update(content="", images=["https://invalid.example.test/unseen.png"])
        # A second textual post cannot lend visual evidence to the selected
        # first post, which is the actual durable comment target.
        fixture.gateway.posts.append({**fixture.gateway.posts[0], "tid": "textual-neighbor", "content": "今天散步"})
        await wake(fixture)
        assert writes(fixture) == []
        assert model.prompts == []
        assert actions(sqlite_engine)[0][0] == "silent"


class PolicyGuard:
    def __init__(self, executor, observation, *, scheduled=False):
        self.trace_trigger = {"trigger_reason_code": "intention_due"} if scheduled else {}
        self.executor, self.observation = executor, observation
        self.decision = SocialDecision("run_tools", "synthetic tool turn", source_message_ids=observation.source_message_ids)
        self.action_id = "synthetic-direct-turn"
        self.dispatched = False

    def valid(self):
        return self.executor.policy_check(self.observation, self.decision)

    def require_valid(self):
        if not self.valid():
            raise RuntimeError("synthetic authority revoked")

    def on_dispatch(self):
        self.require_valid()
        self.dispatched = True


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted", [False, True])
async def test_harness_fact_revision_rotates_native_and_suppresses_stale_intention_across_wakes(
    sqlite_engine, tmp_path, monkeypatch, interrupted,
):
    from app.agent.persona_integration import GroupPersonaContext
    from app.agent.tool_agent import ToolOutcome, ToolTurnResult

    captured = []
    revision = ['a' * 64]

    def valid(engine, *, group_id, dependencies, now):
        return all(digest == revision[0] for digest in dependencies.values())

    async def memory(context, **kwargs):
        if not captured:
            context.memory_dependencies["fact:1"] = revision[0]
        return ToolOutcome({"evidence": "", "memory_search_allowed": True})

    class Harness:
        def __init__(self, **kwargs):
            self.session_id = kwargs["session_id"]

        async def run(self, prompt, tools, guard, **kwargs):
            guard.require_valid()
            captured.append((self.session_id, prompt))
            if len(captured) == 1:
                remember = next(tool for tool in tools if tool.name == "remember_intention")
                await remember.handler({"goal": "依赖旧事实的目标", "progress": "", "next_step": "以后继续"})
                if interrupted:
                    raise RuntimeError("synthetic interruption after atomic goal write")
            else:
                listing = next(tool for tool in tools if tool.name == "list_wakeups")
                assert not (await listing.handler({})).data["wakeups"]
            return ToolTurnResult("silent", reason="model_finished")

    monkeypatch.setattr("app.agent.harness_runtime.HarnessToolAgent", Harness)
    monkeypatch.setattr("app.agent.memory_integration.build_agent_memory", memory)
    monkeypatch.setattr("app.agent.memory_integration.memory_dependencies_valid", valid)
    monkeypatch.setattr("app.agent.memory_integration.create_memory_dependency_receipt", lambda deps: dict(deps))
    monkeypatch.setattr("app.agent.memory_integration.restore_memory_dependency_receipt", lambda engine, *,
        group_id, receipt, now: dict(receipt) if valid(engine, group_id=group_id, dependencies=receipt, now=now) else None)
    monkeypatch.setattr("app.agent.persona_integration.build_group_persona_context", lambda *a, **k:
        GroupPersonaContext("default", "固定人格", "", "", "fixed-profile"))
    current = add_source(sqlite_engine, platform_id="fact-revision-current")
    async with composed(sqlite_engine, tmp_path, ScriptedModel(), runtime="harness") as fixture:
        job = configured_social_jobs(fixture.router.runtime)[0]
        observed = SocialObservation(job, NOW, "小町", source_message_ids=(current,))
        if interrupted:
            with pytest.raises(RuntimeError, match="synthetic interruption"):
                await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed, scheduled=True))
        else:
            await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed, scheduled=True))
        first = fixture.executor.sessions.read('background:' + job.key)
        assert first["checkpoint"]["memory_dependencies"] == {"fact:1": 'a' * 64}
        assert first["checkpoint"]["intention_memory_dependencies"] == ({} if interrupted else {"fact:1": 'a' * 64})
        assert fixture.executor.repository.read_intention(job.key)["working_intention"]["_memory_dependency_receipt"] == {"fact:1": 'a' * 64}
        revision[0] = 'b' * 64
        for _ in range(2):
            fixture.clock[0] += timedelta(minutes=31)
            await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed, scheduled=True))
        assert captured[0][0] != captured[1][0] == captured[2][0]
        assert all(observation_json(prompt)["working_intention"] == {} for _, prompt in captured[1:])
        assert all("依赖旧事实的目标" not in "\n".join(prompt) for _, prompt in captured[1:])
        assert fixture.executor.sessions.read('background:' + job.key)["checkpoint"]["memory_dependencies"] == {}


@pytest.mark.asyncio
async def test_harness_prepares_full_memory_and_persona_sources_and_rotates_changed_profile(
    sqlite_engine, tmp_path, monkeypatch,
):
    from app.agent.persona_integration import GroupPersonaContext
    from app.agent.tool_agent import ToolOutcome, ToolTurnResult

    captured = []

    class Harness:
        def __init__(self, **kwargs):
            self.session_id = kwargs["session_id"]

        async def run(self, prompt, tools, guard, **kwargs):
            guard.require_valid()
            captured.append((self.session_id, prompt, {tool.name for tool in tools}))
            return ToolTurnResult("silent", reason="model_finished")

    monkeypatch.setattr("app.agent.harness_runtime.HarnessToolAgent", Harness)
    historical = add_source(sqlite_engine, platform_id="profile-history", text="有效的历史事实")
    style = add_source(sqlite_engine, platform_id="profile-style", text="有效的模仿例句")
    current = add_source(sqlite_engine, platform_id="profile-current", text="当前请求必须可见")
    style_text = ["动态风格一"]

    async def memory(context, **kwargs):
        context.observed_sources.add(historical)
        return ToolOutcome({"evidence": "完整混合召回证据", "source_ids": [historical],
            "memory_search_allowed": False})

    def persona(*args, **kwargs):
        return GroupPersonaContext("imitated", "新固定人格", style_text[0], "寻址规则",
            "new-profile", (style,), FRIEND, True)

    monkeypatch.setattr("app.agent.memory_integration.build_agent_memory", memory)
    monkeypatch.setattr("app.agent.persona_integration.build_group_persona_context", persona)
    async with composed(sqlite_engine, tmp_path, ScriptedModel(), runtime="harness") as fixture:
        job = configured_social_jobs(fixture.router.runtime)[0]
        scope = fixture.executor.data_scope(job)
        old = fixture.executor.sessions.save(scope,
            [{"role": "user", "content": "OLD_PERSONA_CONTEXT"}, {"role": "assistant", "content": "old"}],
            {"session_id": "old-native", "persona_fingerprint": "old-profile",
             "receipts": [{"call_id": "prior-effect", "status": "uncertain"}]},
            (current,), expected_revision=0)
        observed = SocialObservation(job, NOW, "固定旧观察", ("近期消息未包含当前请求",), (current,))
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed),
            interactive=True, target_user_id=FRIEND, target_message_id="profile-current", direct_group_reply=True)
        first_id, prompt, names = captured[0]
        assert first_id != "old-native"
        assert "OLD_PERSONA_CONTEXT" not in "\n".join(prompt)
        # Persona guidance is a dynamic tail kept separate from the cached
        # static persona prefix, so it can change without invalidating it.
        assert "新固定人格" in "\n".join(prompt)
        assert "动态风格一寻址规则" in "\n".join(prompt)
        assert "完整混合召回证据" in "\n".join(prompt)
        assert observation_json(prompt)["current_target"]["text"] == "当前请求必须可见"
        assert "memory_search" not in names and "memory_read" not in names
        state = fixture.executor.sessions.read(scope)
        assert set(state["source_message_ids"]) == {current, historical, style}
        assert state["checkpoint"]["receipts"][0]["status"] == "uncertain"
        assert state["revision"] > old["revision"]
        style_text[0] = "动态风格二"
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed),
            interactive=True, target_user_id=FRIEND, target_message_id="profile-current", direct_group_reply=True)
        assert captured[1][0] == first_id
        assert "动态风格二" in "\n".join(captured[1][1])


@pytest.mark.asyncio
async def test_harness_current_target_remains_visible_when_memory_is_disabled(
    sqlite_engine, tmp_path, monkeypatch,
):
    from app.agent.tool_agent import ToolTurnResult
    captured = []

    class Harness:
        def __init__(self, **kwargs):
            pass

        async def run(self, prompt, tools, guard, **kwargs):
            guard.require_valid()
            captured.append(prompt)
            return ToolTurnResult("silent", reason="model_finished")

    monkeypatch.setattr("app.agent.harness_runtime.HarnessToolAgent", Harness)
    source = add_source(sqlite_engine, platform_id="older-current-request", text="请回答这条被批次积压的请求")
    recent = add_source(sqlite_engine, platform_id="newer-context-only", text="最近的无关消息")
    async with composed(sqlite_engine, tmp_path, ScriptedModel(), runtime="harness") as fixture:
        job = configured_social_jobs(fixture.router.runtime)[0]
        observed = SocialObservation(job, NOW, "小町", ("另外30条较新消息",), (recent,))
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed),
            interactive=True,direct_group_reply=True, target_user_id=FRIEND, target_message_id="older-current-request")
        payload = observation_json(captured[0])
        assert payload["current_target"]["text"] == "请回答这条被批次积压的请求"
        assert payload["current_target"]["source_id"] == source
        assert set(fixture.executor.sessions.read(job.key)["source_message_ids"]) == {source, recent}


@pytest.mark.asyncio
async def test_harness_old_conversation_quote_is_pinned_and_revoked_native_context_rotates(
    sqlite_engine, tmp_path, monkeypatch,
):
    from app.agent.tool_agent import ToolTurnResult
    captured = []

    class Harness:
        def __init__(self, **kwargs):
            self.session_id = kwargs["session_id"]

        async def run(self, prompt, tools, guard, **kwargs):
            guard.require_valid()
            captured.append((self.session_id, prompt))
            return ToolTurnResult("silent", reason="model_finished")

    monkeypatch.setattr("app.agent.harness_runtime.HarnessToolAgent", Harness)
    quoted = add_source(sqlite_engine, platform_id="old-conversation-quote", text="旧引用的独立正文")
    source = add_source(sqlite_engine, platform_id="quote-followup", text="这句是什么意思")
    with session_scope(sqlite_engine) as session:
        session.get(Message, source).reply_to_msg_id = "old-conversation-quote"
    async with composed(sqlite_engine, tmp_path, ScriptedModel(), runtime="harness") as fixture:
        job = configured_social_jobs(fixture.router.runtime)[0]
        observed = SocialObservation(job, NOW, "小町", ("近期窗口没有旧引用",), (source,))
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed),
            interactive=True,direct_group_reply=True, target_user_id=FRIEND, target_message_id="quote-followup")
        target = observation_json(captured[0][1])["current_target"]
        assert target["quoted_message"]["text"] == "旧引用的独立正文"
        assert target["quoted_message"]["source_id"] == quoted
        assert quoted in fixture.executor.sessions.read(job.key)["source_message_ids"]
        with session_scope(sqlite_engine) as session:
            session.get(Message, quoted).raw_json = {"delivery_state": "deleted"}
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed),
            interactive=True,direct_group_reply=True, target_user_id=FRIEND, target_message_id="quote-followup")
        assert captured[1][0] != captured[0][0]
        assert "旧引用的独立正文" not in "\n".join(captured[1][1])
        assert observation_json(captured[1][1])["current_target"]["quoted_message_unavailable"] is True
        assert quoted not in fixture.executor.sessions.read(job.key)["source_message_ids"]


@pytest.mark.asyncio
@pytest.mark.parametrize('text,expected', [('今晚吃什么',0),('这图怎么样',1)])
async def test_harness_picture_then_followup_selects_only_relevant_pixels(sqlite_engine,tmp_path,monkeypatch,text,expected):
    from app.agent.tool_agent import ToolTurnResult
    seen=[]
    class Harness:
        def __init__(self,**kwargs):pass
        async def run(self,prompt,tools,guard,**kwargs):
            seen.append(len(kwargs['images']))
            return ToolTurnResult('silent',reason='model_finished')
    monkeypatch.setattr('app.agent.harness_runtime.HarnessToolAgent',Harness)
    async with composed(sqlite_engine,tmp_path,ScriptedModel(),runtime='harness') as fixture:
        cache=fixture.router.runtime.settings.data_dir/'image_cache'/str(GROUP)
        cache.mkdir(parents=True,exist_ok=True);photo=cache/'photo.png';photo.write_bytes(image_bytes())
        picture=add_source(sqlite_engine,platform_id='followup-photo',images=({'local_path':str(photo)},))
        current=add_source(sqlite_engine,platform_id='followup-current',text=text)
        with session_scope(sqlite_engine) as session:
            session.get(Message,picture).timestamp=(NOW-timedelta(seconds=10)).replace(tzinfo=None)
            session.get(Message,current).timestamp=NOW.replace(tzinfo=None)
        job=configured_social_jobs(fixture.router.runtime)[0]
        observed=SocialObservation(job,NOW,'小町',(text,),(picture,current))
        await fixture.executor.run_tools(observed,PolicyGuard(fixture.executor,observed),interactive=True,direct_group_reply=True,
            target_user_id=FRIEND,target_message_id='followup-current')
        assert seen==[expected]


@pytest.mark.asyncio
async def test_group_input_rejection_rotates_and_keeps_uncertain_receipt(sqlite_engine,tmp_path,monkeypatch):
    from app.agent.tool_agent import ToolTurnResult
    from app.agent.harness_runtime import model_context_format_version
    seen=[]
    class Harness:
        def __init__(self,**kwargs):self.sid=kwargs['session_id']
        async def run(self,prompt,tools,guard,**kwargs):
            seen.append((self.sid,prompt))
            return ToolTurnResult('failed',reason='model_image_input_rejected') if len(seen)==1 else ToolTurnResult('silent',reason='model_finished')
    monkeypatch.setattr('app.agent.harness_runtime.HarnessToolAgent',Harness)
    rejected=add_source(sqlite_engine,platform_id='rejected-group-source',text='REJECTED_GROUP_INPUT')
    async with composed(sqlite_engine,tmp_path,ScriptedModel(),runtime='harness') as fixture:
        job=configured_social_jobs(fixture.router.runtime)[0]
        fixture.executor.sessions.save(job.key,[],{'session_id':'old','context_format_version':model_context_format_version(fixture.executor.decision_client),
            'receipts':[{'call_id':'unknown-old','status':'uncertain','replay_allowed':False}]},(),expected_revision=0)
        first=SocialObservation(job,NOW,'小町',('REJECTED_GROUP_INPUT',),(rejected,))
        await fixture.executor.run_tools(first,PolicyGuard(fixture.executor,first),interactive=True,direct_group_reply=True,target_user_id=FRIEND,target_message_id='rejected-group-source')
        state=fixture.executor.sessions.read(job.key)
        assert state['items']==[] and not state['checkpoint']['context_reusable']
        normal=add_source(sqlite_engine,platform_id='normal-group-source',text='普通聊天')
        second=SocialObservation(job,NOW,'小町',('普通聊天',),(normal,))
        await fixture.executor.run_tools(second,PolicyGuard(fixture.executor,second),interactive=True,direct_group_reply=True,target_user_id=FRIEND,target_message_id='normal-group-source')
        assert seen[0][0]!=seen[1][0]
        assert 'REJECTED_GROUP_INPUT' not in '\n'.join(seen[1][1])
        assert fixture.executor.sessions.read(job.key)['checkpoint']['receipts'][0]['status']=='uncertain'


@pytest.mark.asyncio
async def test_operator_owned_code_tool_can_be_chosen_and_its_result_drives_intention(sqlite_engine, tmp_path):
    calls = []

    class Sandbox:
        async def run(self, code, **kwargs):
            calls.append(code)
            return SimpleNamespace(stdout="42\n", status="ok", reason="", artifacts=())

    def save_computation(lines, kwargs):
        actual = next(item["result"] for item in feedback(lines) if item.get("tool") == "execute_python")
        assert actual["stdout"] == "42\n"
        return {"tool": "remember_intention", "arguments": {"goal": "研究一个数字话题", "progress": "实际计算结果是42", "next_step": "继续核对"}}

    model = ScriptedModel({"tool": "execute_python", "arguments": {"code": "print(6*7)"}}, save_computation, {"finish": True})
    async with composed(sqlite_engine, tmp_path, model, compute=True) as fixture:
        fixture.executor.sandbox = Sandbox()
        await wake(fixture)
        assert calls == ["print(6*7)"] and writes(fixture) == []
        intention = fixture.executor.repository.read_intention("group:" + str(GROUP))
        assert intention["working_intention"]["progress"] == "实际计算结果是42"


@pytest.mark.asyncio
async def test_runtime_permission_revocation_blocks_even_operator_owned_code(sqlite_engine, tmp_path):
    calls = []

    class Sandbox:
        async def run(self, code, **kwargs):
            calls.append(code)
            return SimpleNamespace(stdout="42\n", status="ok", reason="", artifacts=())

    fixture_ref = []

    def revoke(lines, kwargs):
        fixture_ref[0].router.runtime.social_policy["enabled"] = False
        return {"tool": "execute_python", "arguments": {"code": "print(6*7)"}}

    model = ScriptedModel(revoke)
    async with composed(sqlite_engine, tmp_path, model, compute=True) as fixture:
        fixture_ref.append(fixture)
        fixture.executor.sandbox = Sandbox()
        await wake(fixture)
        assert len(model.prompts) == 1 and calls == [] and writes(fixture) == []
        assert actions(sqlite_engine)[0][0] != "sent"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [{"enabled": False}, {"enabled": True, "endpoint": "https://replacement.invalid"}])
async def test_compute_manifest_revocation_blocks_the_code_call(sqlite_engine, tmp_path, change):
    calls, fixture_ref = [], []

    class Sandbox:
        async def run(self, code, **kwargs):
            calls.append(code)
            return SimpleNamespace(stdout="42\n", status="ok", reason="", artifacts=())

    def revoke(lines, kwargs):
        fixture_ref[0].router.runtime.social_policy["compute"] = change
        return {"tool": "execute_python", "arguments": {"code": "print(6*7)"}}

    def inspect_block(lines, kwargs):
        result = next(item for item in feedback(lines) if item.get("tool") == "execute_python")
        assert result["status"] == "blocked" and result["result"]["error"] == "compute_permission_revoked"
        return {"finish": True}

    model = ScriptedModel(revoke, inspect_block)
    async with composed(sqlite_engine, tmp_path, model, compute=True) as fixture:
        fixture_ref.append(fixture)
        fixture.executor.sandbox = Sandbox()
        await wake(fixture)
        assert calls == [] and writes(fixture) == []
        assert len(model.prompts) == 2


@pytest.mark.asyncio
async def test_native_responses_tools_preserve_reasoning_and_actual_tool_result(sqlite_engine, tmp_path):
    captured, usage = [], []
    reasoning = {"type": "reasoning", "id": "synthetic-reasoning", "summary": [], "encrypted_content": "opaque-synthetic"}

    def response(request):
        payload = json.loads(request.content)
        captured.append(payload)
        call_id = "actual-call-1" if len(captured) == 1 else "actual-call-2"
        tool = "read_context" if len(captured) == 1 else "send_group"
        args = {} if tool == "read_context" else {"text": "这是工具真实发送的内容", "artifact_ids": []}
        return httpx.Response(200, json={"id": "synthetic-response", "status": "completed", "output": [reasoning,
            {"type": "function_call", "name": tool, "call_id": call_id, "arguments": json.dumps(args)}],
            "usage": {"input_tokens": 50, "output_tokens": 10}})

    with httpx.Client(transport=httpx.MockTransport(response)) as transport:
        client = LlmClient(base_url="https://synthetic.example.test/v1", api_key="synthetic-key", model="primary-model",
                           responses_model="primary-model", responses_only=True, builtin_web_search=True,
                           http_client=transport, usage_recorder=usage.append)
        async with composed(sqlite_engine, tmp_path, client) as fixture:
            await wake(fixture)
            assert actions(sqlite_engine) == [("sent", "tool:send_group", "actual-group-receipt")]
            assert len(captured) == 2 and len(usage) == 2
            assert captured[0]["parallel_tool_calls"] is False
            assert all(tool["type"] == "function" for tool in captured[0]["tools"])
            continuation = captured[1]["input"][1:]
            assert continuation[0] == reasoning
            assert continuation[1] == {"type": "function_call", "call_id": "actual-call-1", "name": "read_context", "arguments": "{}"}
            assert continuation[2]["type"] == "function_call_output" and continuation[2]["call_id"] == "actual-call-1"
            output = json.loads(continuation[2]["output"])
            assert output["status"] == "ok" and output["tool"] == "read_context"
            assert "context" in output["result"]


@pytest.mark.asyncio
async def test_intention_continues_across_wakes_and_recalled_search_source_revokes_it(sqlite_engine, tmp_path):
    old = add_source(sqlite_engine, platform_id="old-agreement", text="旧约定：周五画猫")
    current = add_source(sqlite_engine, platform_id="current-chat", text="天气不错")
    model = ScriptedModel({"tool": "search_history", "arguments": {"query": "旧约定"}},
        {"tool": "remember_intention", "arguments": {"goal": "完成周五画猫约定", "progress": "读到了旧约定", "next_step": "构思画面"}},
        {"finish": True}, {"tool": "remember_intention", "arguments": {"goal": "完成周五画猫约定", "progress": "构思完成", "next_step": "开始绘图"}},
        {"finish": True}, {"finish": True})
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        job = configured_social_jobs(fixture.router.runtime)[0]
        observed = SocialObservation(job, NOW, "小町", ("天气不错",), (current,))
        guard = PolicyGuard(fixture.executor, observed)
        await fixture.executor.run_tools(observed, guard)
        note = fixture.executor.repository.read_intention(job.key)
        assert set(note["source_message_ids"]) == {old, current}
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed))
        assert observation_json(model.prompts[3])["working_intention"]["working_intention"]["goal"] == "完成周五画猫约定"
        assert set(fixture.executor.repository.read_intention(job.key)["source_message_ids"]) == {old, current}
        with session_scope(sqlite_engine) as session:
            row = session.get(Message, old)
            row.raw_json = {**row.raw_json, "delivery_state": "deleted"}
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed))
        assert observation_json(model.prompts[5])["working_intention"] == {}
        assert writes(fixture) == []


@pytest.mark.asyncio
async def test_recalled_history_source_during_turn_revokes_external_reply(sqlite_engine, tmp_path):
    old = add_source(sqlite_engine, platform_id="older-than-recent-window", text="旧约定：周五画猫")
    for index in range(31):
        add_source(sqlite_engine, platform_id=f"recent-{index}", text="天气不错")

    def revoke_then_send(lines, kwargs):
        assert any(item.get("tool") == "search_history" for item in feedback(lines))
        with session_scope(sqlite_engine) as session:
            row = session.get(Message, old)
            row.raw_json = {**row.raw_json, "delivery_state": "deleted"}
        return {"tool": "send_group", "arguments": {"text": "接着刚查到的周五画猫约定", "artifact_ids": []}}

    model = ScriptedModel({"tool": "search_history", "arguments": {"query": "旧约定"}}, revoke_then_send)
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        await wake(fixture)
        assert old not in observation_json(model.prompts[0])["source_message_ids"]
        assert writes(fixture) == []
        assert actions(sqlite_engine)[0][0] not in {"sent", "uncertain"}


@pytest.mark.asyncio
async def test_generated_artifact_records_the_history_sources_the_agent_actually_read(sqlite_engine, tmp_path):
    old = add_source(sqlite_engine, platform_id="old-drawing-interest", text="旧约定：想画红色猫")
    current = add_source(sqlite_engine, platform_id="current-context", text="天气不错")
    model = ScriptedModel({"tool": "search_history", "arguments": {"query": "旧约定"}},
                          {"tool": "generate_image", "arguments": {"prompt": "按照刚读到的约定画红色猫"}}, {"finish": True})
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        job = configured_social_jobs(fixture.router.runtime)[0]
        observed = SocialObservation(job, NOW, "小町", ("天气不错",), (current,))
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed))
        generated = list((fixture.router.runtime.settings.data_dir / "autonomous_artifacts").glob("art-*.meta.json"))
        assert len(generated) == 1
        metadata = json.loads(generated[0].read_text(encoding="utf-8"))
        assert set(metadata["source_message_ids"]) == {old, current}


@pytest.mark.asyncio
async def test_external_group_output_records_a_read_source_outside_recent_window(sqlite_engine, tmp_path):
    old = add_source(sqlite_engine, platform_id="older-read-source", text="旧约定：周五画猫")
    for index in range(31):
        add_source(sqlite_engine, platform_id=f"recent-source-{index}", text="天气不错")
    model = ScriptedModel({"tool": "search_history", "arguments": {"query": "旧约定"}},
                          {"tool": "send_group", "arguments": {"text": "记得之前聊过周五画猫", "artifact_ids": []}})
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        await wake(fixture)
        assert actions(sqlite_engine)[0][0] == "sent"
        with session_scope(sqlite_engine) as session:
            outgoing = session.scalar(select(Message).where(Message.user_id == BOT))
            assert outgoing is not None and old in outgoing.raw_json["source_message_ids"]


@pytest.mark.asyncio
async def test_cancelled_pixel_model_holds_source_snapshot_until_thread_finishes(sqlite_engine, tmp_path):
    started, release = threading.Event(), threading.Event()
    checked, snapshots = [], []

    def slow(lines, kwargs):
        snapshots.extend(Path(image.local_path) for image in kwargs["images"])
        started.set()
        release.wait(3)
        checked.append(all(path.exists() for path in snapshots))
        return {"finish": True}

    model = ScriptedModel(slow)
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        cache = fixture.router.runtime.settings.data_dir / "image_cache" / str(GROUP)
        cache.mkdir(parents=True)
        path = cache / "source.png"
        path.write_bytes(image_bytes())
        source = add_source(sqlite_engine, platform_id="visible-image", text="", images=({"local_path": str(path)},))
        job = configured_social_jobs(fixture.router.runtime)[0]
        observed = SocialObservation(job, NOW, "小町", ("分享一张图片",), (source,))
        task = asyncio.create_task(fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed)))
        try:
            for _ in range(300):
                if started.is_set():
                    break
                await asyncio.sleep(0.005)
            assert started.is_set() and snapshots
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert all(snapshot.exists() for snapshot in snapshots)
            release.set()
            for _ in range(300):
                if checked and all(not snapshot.exists() for snapshot in snapshots):
                    break
                await asyncio.sleep(0.005)
            assert checked == [True]
            assert all(not snapshot.exists() for snapshot in snapshots)
            assert writes(fixture) == []
        finally:
            release.set()


@pytest.mark.asyncio
async def test_cancelled_generation_holds_capacity_and_never_sends_late_result(sqlite_engine, tmp_path):
    started, release = threading.Event(), threading.Event()

    def slow_image():
        started.set()
        release.wait(3)

    model = ScriptedModel({"tool": "generate_image", "arguments": {"prompt": "红色猫"}},
                          lambda lines, _: {"tool": "send_group", "arguments": {"text": "画好啦", "artifact_ids": [last_artifact(lines)]}})
    async with composed(sqlite_engine, tmp_path, model, image_callback=slow_image) as fixture:
        task = asyncio.create_task(wake(fixture))
        try:
            for _ in range(300):
                if started.is_set():
                    break
                await asyncio.sleep(0.005)
            assert started.is_set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(AgentModelBusyError):
                await fixture.executor.blocking.run(lambda: None)
            assert actions(sqlite_engine)[0][0] == "blocked"
            release.set()
            for _ in range(300):
                try:
                    await fixture.executor.blocking.run(lambda: None)
                    break
                except AgentModelBusyError:
                    await asyncio.sleep(0.005)
            else:
                pytest.fail("cancelled generation never released its capacity")
            assert len(fixture.image_client.calls) == 1
            assert len(model.prompts) == 1 and writes(fixture) == []
        finally:
            release.set()


@pytest.mark.asyncio
async def test_scoped_history_tool_never_reads_another_groups_same_keyword(sqlite_engine, tmp_path):
    same = add_source(sqlite_engine, platform_id="this-group-evidence", text="旧约定：画猫")
    other = add_source(sqlite_engine, platform_id="other-group-evidence", group=OTHER_GROUP, text="旧约定：PRIVATE_OTHER_GROUP")

    def inspect(lines, kwargs):
        result = next(item["result"] for item in feedback(lines) if item.get("tool") == "search_history")
        assert {item["source_id"] for item in result["messages"]} == {same}
        assert other not in {item["source_id"] for item in result["messages"]}
        assert "PRIVATE_OTHER_GROUP" not in "\n".join(lines)
        return {"finish": True}

    model = ScriptedModel({"tool": "search_history", "arguments": {"query": "旧约定"}}, inspect)
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        await wake(fixture)
        assert len(model.prompts) == 2 and writes(fixture) == []


@pytest.mark.asyncio
async def test_generated_image_is_observed_and_sent_in_a_later_wake_with_old_provenance(sqlite_engine, tmp_path):
    old = add_source(sqlite_engine, platform_id="original-drawing-request", text="请画一只红猫")
    saved_id = []

    def finish_first(lines, kwargs):
        saved_id.append(last_artifact(lines))
        return {"finish": True}

    def read_owned(lines, kwargs):
        listed = artifact_entries(lines)
        assert {item["artifact_id"] if "artifact_id" in item else item["id"] for item in listed} == set(saved_id)
        assert old not in observation_json(lines)["source_message_ids"]
        return {"tool": "read_artifact", "arguments": {"artifact_id": saved_id[0]}}

    def send_observed(lines, kwargs):
        assert tool_result(lines, "read_artifact")["status"] == "ok"
        assert len(kwargs.get("images", ())) == 1
        with Image.open(kwargs["images"][0].local_path) as pixels:
            assert pixels.getpixel((0, 0))[:3] == (255, 0, 0)
        return {"tool": "send_group", "arguments": {"text": "昨天构思的红猫今天画好了", "artifact_ids": saved_id}}

    model = ScriptedModel({"tool": "generate_image", "arguments": {"prompt": "红色猫"}}, finish_first,
                          {"tool": "list_artifacts", "arguments": {}}, read_owned, send_observed)
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        await wake(fixture)
        assert writes(fixture) == []
        for index in range(31):
            add_source(sqlite_engine, platform_id=f"later-wake-{index}", text="另一个话题")
        await wake(fixture)
        assert len(fixture.image_client.calls) == 1
        assert [action for action, _ in writes(fixture)] == ["send_group_msg"]
        with session_scope(sqlite_engine) as session:
            output = session.scalar(select(Message).where(Message.user_id == BOT))
            assert old in output.raw_json["source_message_ids"]


@pytest.mark.asyncio
async def test_computed_dataset_can_be_read_and_used_to_derive_a_later_image(sqlite_engine, tmp_path):
    old = add_source(sqlite_engine, platform_id="first-computation", text="研究红色比例")
    dataset_ids, image_ids, input_calls, output_paths, fixture_ref = [], [], [], [], []

    class Sandbox:
        async def run(self, code, input_files=None):
            input_calls.append(dict(input_files or {}))
            root = fixture_ref[0].router.runtime.settings.data_dir / "autonomous_artifacts" / "sandbox" / f"run-{len(input_calls)}"
            root.mkdir(parents=True)
            if len(input_calls) == 1:
                assert not input_files
                path = root / "counts.csv"
                path.write_text("color,count\nred,42", encoding="utf-8")
            else:
                assert set(input_files) == set(dataset_ids)
                assert next(iter(input_files.values())).read_text(encoding="utf-8") == "color,count\nred,42"
                path = root / "derived.png"
                path.write_bytes(image_bytes())
            output_paths.append(path)
            return SimpleNamespace(stdout="实际计算完成", status="ok", reason="", artifacts=(path,))

    def keep_dataset(lines, kwargs):
        entries = artifact_entries(lines, "execute_python")
        assert len(entries) == 1 and entries[0]["type"] == "text/csv"
        dataset_ids.append(entries[0]["artifact_id"])
        return {"finish": True}

    def read_dataset(lines, kwargs):
        assert len(artifact_entries(lines)) == 1
        return {"tool": "read_artifact", "arguments": {"artifact_id": dataset_ids[0]}}

    def continue_computation(lines, kwargs):
        result = tool_result(lines, "read_artifact")
        assert result["status"] == "ok" and "color,count\nred,42" in json.dumps(result["result"], ensure_ascii=False).replace("\\n", "\n")
        return {"tool": "execute_python", "arguments": {"code": "构建图表", "input_artifact_ids": dataset_ids}}

    def send_computed_image(lines, kwargs):
        entries = artifact_entries(lines, "execute_python")
        assert len(entries) == 1 and entries[0]["type"] == "image/png"
        image_ids.append(entries[0]["artifact_id"])
        return {"tool": "send_group", "arguments": {"text": "根据真实数据制作了图", "artifact_ids": image_ids}}

    model = ScriptedModel({"tool": "execute_python", "arguments": {"code": "计算红色比例"}}, keep_dataset,
                          {"tool": "list_artifacts", "arguments": {}}, read_dataset, continue_computation, send_computed_image)
    async with composed(sqlite_engine, tmp_path, model, compute=True) as fixture:
        fixture_ref.append(fixture)
        fixture.executor.sandbox = Sandbox()
        job = configured_social_jobs(fixture.router.runtime)[0]
        initial = SocialObservation(job, NOW, "小町", ("研究红色比例",), (old,))
        await fixture.executor.run_tools(initial, PolicyGuard(fixture.executor, initial))
        output_paths[0].unlink()
        current = add_source(sqlite_engine, platform_id="second-computation", text="继续研究")
        later = replace(initial, context=("继续研究",), source_message_ids=(current,))
        await fixture.executor.run_tools(later, PolicyGuard(fixture.executor, later))
        assert [action for action, _ in writes(fixture)] == ["send_group_msg"]
        assert len(input_calls) == 2 and len(fixture.image_client.calls) == 0
        root = fixture.router.runtime.settings.data_dir / "autonomous_artifacts"
        manifest = json.loads((root / (image_ids[0] + ".meta.json")).read_text(encoding="utf-8"))
        assert set(manifest["source_message_ids"]) == {old, current}
        with session_scope(sqlite_engine) as session:
            output = session.scalar(select(Message).where(Message.user_id == BOT))
            assert set(output.raw_json["source_message_ids"]) == {old, current}


@pytest.mark.asyncio
async def test_recalled_artifact_cannot_supply_pixels_or_be_sent_in_a_new_turn(sqlite_engine, tmp_path):
    old = add_source(sqlite_engine, platform_id="recalled-artifact-origin", text="画红猫")
    saved_id = []

    def save(lines, kwargs):
        saved_id.append(last_artifact(lines))
        return {"finish": True}

    def refuse(lines, kwargs):
        assert tool_result(lines, "read_artifact")["status"] in {"blocked", "failed"}
        assert not kwargs.get("images")
        return {"tool": "send_group", "arguments": {"text": "复用已撤回来源的图片", "artifact_ids": saved_id}}

    model = ScriptedModel({"tool": "generate_image", "arguments": {"prompt": "红猫"}}, save,
                          lambda *_: {"tool": "read_artifact", "arguments": {"artifact_id": saved_id[0]}}, refuse)
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        job = configured_social_jobs(fixture.router.runtime)[0]
        initial = SocialObservation(job, NOW, "小町", ("画红猫",), (old,))
        await fixture.executor.run_tools(initial, PolicyGuard(fixture.executor, initial))
        with session_scope(sqlite_engine) as session:
            session.get(Message, old).raw_json = {"delivery_state": "deleted"}
        current = add_source(sqlite_engine, platform_id="safe-current", text="天气不错")
        later = replace(initial, context=("天气不错",), source_message_ids=(current,))
        await fixture.executor.run_tools(later, PolicyGuard(fixture.executor, later))
        assert writes(fixture) == []


@pytest.mark.asyncio
async def test_group_artifact_is_not_visible_in_public_scope(sqlite_engine, tmp_path):
    saved_id = []
    group_model = ScriptedModel({"tool": "generate_image", "arguments": {"prompt": "红猫"}},
        lambda lines, _: saved_id.append(last_artifact(lines)) or {"finish": True})
    async with composed(sqlite_engine, tmp_path, group_model) as fixture:
        await wake(fixture)

    def no_public_library(lines, kwargs):
        assert artifact_entries(lines) == []
        return {"tool": "read_artifact", "arguments": {"artifact_id": saved_id[0]}}

    def no_foreign_pixels(lines, kwargs):
        assert tool_result(lines, "read_artifact")["status"] in {"failed", "blocked"}
        assert not kwargs.get("images")
        return {"finish": True}

    public_model = ScriptedModel({"tool": "list_artifacts", "arguments": {}}, no_public_library, no_foreign_pixels)
    async with composed(sqlite_engine, tmp_path, public_model, kind="qzone_publish") as fixture:
        await wake(fixture)
        assert len(public_model.prompts) == 3 and writes(fixture) == []


@pytest.mark.asyncio
async def test_artifact_source_union_over_budget_is_blocked_without_dropping_provenance(sqlite_engine, tmp_path):
    origins = tuple(add_source(sqlite_engine, platform_id=f"library-source-{index}") for index in range(200))
    current = add_source(sqlite_engine, platform_id="new-source-201")
    chosen_id = []

    def blocked(lines, kwargs):
        result = tool_result(lines, "read_artifact")
        assert result["status"] == "blocked"
        assert result["result"]["error"] == "source_budget_exhausted"
        assert not kwargs.get("images")
        return {"finish": True}

    model = ScriptedModel(lambda *_: {"tool": "read_artifact", "arguments": {"artifact_id": chosen_id[0]}}, blocked)
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        root = fixture.router.runtime.settings.data_dir / "autonomous_artifacts"
        root.mkdir(parents=True)
        original = root / "trusted-provider.png"
        original.write_bytes(image_bytes())
        library = AutonomousArtifactStore(root, account_id=BOT, scope=f"group:{GROUP}", engine=sqlite_engine,
                                         source_message_ids=lambda: origins)
        chosen_id.append(library.add(original, origin="actual_generated_pixels"))
        job = configured_social_jobs(fixture.router.runtime)[0]
        observed = SocialObservation(job, NOW, "小町", ("新话题",), (current,))
        await fixture.executor.run_tools(observed, PolicyGuard(fixture.executor, observed))
        assert library.provenance(chosen_id[0]) == origins
        assert len(model.prompts) == 2 and writes(fixture) == []


@pytest.mark.asyncio
async def test_exhausted_write_quota_still_allows_reading_and_continuing_a_goal(sqlite_engine, tmp_path):
    def inspect_read_only(lines, kwargs):
        allowed = available_tools(lines)
        assert {"read_context", "list_artifacts", "read_artifact", "remember_intention"}.issubset(allowed)
        assert not allowed & {"send_group", "publish_qzone", "comment_qzone", "set_nickname", "set_avatar", "set_card"}
        return {"tool": "read_context", "arguments": {}}

    model = ScriptedModel({"tool": "send_group", "arguments": {"text": "今天的一次自主发言", "artifact_ids": []}},
                          inspect_read_only,
                          {"tool": "remember_intention", "arguments": {"goal": "继续研究画猫", "progress": "今天已经说过话了", "next_step": "明天再分享"}},
                          {"finish": True})
    async with composed(sqlite_engine, tmp_path, model, options={"max_daily": 1}) as fixture:
        await wake(fixture)
        assert len(writes(fixture)) == 1
        await wake(fixture)
        assert len(writes(fixture)) == 1 and len(model.prompts) == 4
        intention = fixture.executor.repository.read_intention(f"group:{GROUP}")
        assert intention["working_intention"]["next_step"] == "明天再分享"


@pytest.mark.asyncio
@pytest.mark.parametrize("uncertain", [False, True])
async def test_public_post_plan_can_continue_next_wake_and_dispatched_comment_is_never_replayed(sqlite_engine, tmp_path, uncertain):
    def comment_after_continuation(lines, kwargs):
        observed = observation_json(lines)
        assert observed["posts"][0]["post_id"] == "observed-photo"
        assert observed["working_intention"]["working_intention"]["next_step"] == "下次醒来想想怎样评价"
        return {"tool": "comment_qzone", "arguments": {"text": "你拍的猫很有趣", "artifact_ids": []}}

    model = ScriptedModel(
        {"tool": "remember_intention", "arguments": {"goal": "研究好友拍猫的动态", "progress": "读到了真实动态", "next_step": "下次醒来想想怎样评价"}},
        {"finish": True}, comment_after_continuation,
        {"tool": "comment_qzone", "arguments": {"text": "重复评论", "artifact_ids": []}},
    )
    async with composed(sqlite_engine, tmp_path, model, kind="qzone_comment", options={"max_daily": 1}) as fixture:
        if uncertain:
            fixture.gateway.write_response = {"status": "ok", "retcode": 0, "data": {"comment_id": []}}
        await wake(fixture)
        assert writes(fixture) == [] and len(model.prompts) == 2
        await wake(fixture)
        assert [action for action, _ in writes(fixture)] == ["comment_qzone"]
        assert len(model.prompts) == 3
        with session_scope(sqlite_engine) as session:
            records = list(session.scalars(select(SocialActionRecord).order_by(SocialActionRecord.created_at)))
            assert records[0].status == "silent" and records[0].dispatched_at is None
            assert records[1].status == ("uncertain" if uncertain else "sent")
            assert records[1].dispatched_at is not None
        await wake(fixture)
        assert len(writes(fixture)) == 1 and len(model.prompts) == 3


class InteractiveIngressGuard:
    def __init__(self, turn_id):
        self.turn_id = turn_id
        self.dispatched = False
        self.allowed = True

    def valid(self):
        return self.allowed

    def require_valid(self):
        if not self.valid():
            raise RuntimeError("synthetic ingress revoked")

    def on_dispatch(self):
        self.require_valid()
        self.dispatched = True


@pytest.mark.asyncio
async def test_native_proactive_send_uses_chat_interval_and_agent_can_be_silent(sqlite_engine, tmp_path):
    from test_harness_runtime import model_client, responses_sse
    from test_delegation_composition import observation
    judgments=[]
    def model(body,number):
        if not body['tools']:
            packet=observation(body,'Group observation JSON:\n')
            judgments.append(True)
            return responses_sse(text=json.dumps({'decision':'participate' if len(judgments)<=2 else 'silent',
                'source_id':packet['messages'][-1]['id'],'reason':'natural_hook'}),number=number)
        return responses_sse(name='send_group',arguments={'text':f'合成主动回复{number}','artifact_ids':[]},number=number)
    client, requests, _ = model_client(model)
    async with composed(sqlite_engine, tmp_path, client, runtime="harness",
                        options={"cooldown_seconds": 1800}) as fixture:
        await fixture.agent.stop()
        fixture.router.runtime.group_policy["groups"][str(GROUP)]["proactive_interval_seconds"] = "120-300"
        statuses = []
        for index in range(4):
            events = []
            for part in range(8):
                payload = {"post_type": "message", "message_type": "group", "group_id": GROUP,
                    "message_id": f"proactive-cadence-{index}-{part}", "user_id": FRIEND, "time": fixture.clock[0].timestamp(),
                    "sender": {"nickname": "synthetic author"},
                    "message": [{"type": "text", "data": {"text": f"大家在讨论夜空 {index}-{part}"}}]}
                event = parse_group_message_event(payload, bot_qq=BOT, bot_name="小町")
                fixture.router.ingest_live_group_message(event)
                events.append(event)
            batch = AgentBatch(GROUP, tuple(events), f"proactive-cadence-{index}")
            planner = GroupParticipationPlanner(router=fixture.router, decision_client=client,
                autonomous_executor=fixture.executor)
            try:
                decision = await planner.decide(batch)
                result = (await fixture.executor.execute_group_turn(event, batch, decision,
                    InteractiveIngressGuard(batch.turn_id)) if decision.action == 'reply' else
                    SimpleNamespace(status='silent'))
                statuses.append(result.status)
            finally:
                planner.close()
            fixture.clock[0] += timedelta(seconds=0 if index == 1 else 601)
        assert statuses == ["sent", "sent", "silent", "silent"]
        assert len(writes(fixture)) == 2
        assert len(requests) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("already_dispatched", [False, True])
async def test_native_model_failure_notice_once_and_never_after_dispatch(sqlite_engine, tmp_path, already_dispatched):
    from test_harness_runtime import model_client
    client, requests, _ = model_client(lambda *_: httpx.Response(503, json={"error": "synthetic failure"}))
    async with composed(sqlite_engine, tmp_path, client, runtime="harness") as fixture:
        await fixture.agent.stop()
        payload = {"post_type": "message", "message_type": "group", "group_id": GROUP,
            "message_id": "failure-notice", "user_id": FRIEND, "time": NOW.timestamp(),
            "sender": {"nickname": "synthetic author"}, "message": [
                {"type": "at", "data": {"qq": str(BOT)}}, {"type": "text", "data": {"text": "在吗"}}]}
        event = parse_group_message_event(payload, bot_qq=BOT, bot_name="小町")
        fixture.router.ingest_live_group_message(event)
        batch = AgentBatch(GROUP, (event,), "synthetic-failure-notice")
        guard = InteractiveIngressGuard(batch.turn_id)
        guard.dispatched = already_dispatched
        decision = AgentDecision("reply", event.platform_msg_id, "direct_trigger")
        await fixture.executor.execute_group_turn(event, batch, decision, guard)
        # Simulate repeat processing without the first guard's in-memory flag.
        again = InteractiveIngressGuard(batch.turn_id)
        again.dispatched = already_dispatched
        await fixture.executor.execute_group_turn(event, batch, decision, again)
        assert len(writes(fixture)) == (0 if already_dispatched else 1)
        assert len(requests) == (6 if already_dispatched else 3)


@pytest.mark.asyncio
@pytest.mark.parametrize('uncertain', [False, True])
async def test_post_send_rejection_drops_execution_history_and_recovery_does_not_send_notice(
        sqlite_engine, tmp_path, monkeypatch, uncertain):
    from app.agent.harness_runtime import HarnessToolAgent
    from app.agent.tool_agent import ToolTurnResult
    from app.agent.autonomous_social import configured_agent_jobs
    calls=[]
    async def send_then_reject(agent, prompt, tools, guard, **kwargs):
        calls.append(prompt)
        outcome=await agent.invoke_tool(next(t for t in tools if t.name=='send_group'),
            {'text':'synthetic confirmed reply', 'artifact_ids':[]}, f'synthetic-call-{len(calls)}')
        return ToolTurnResult(outcome.status, outcome.receipt_id, 'tool:send_group', (
            {'status':'model_request', 'purpose':'agent', 'outcome':'incomplete'},
            {'status':'terminal', 'reason':'model_input_rejected'}))
    monkeypatch.setattr(HarnessToolAgent, 'run', send_then_reject)
    async with composed(sqlite_engine, tmp_path, SimpleNamespace(), runtime='harness') as fixture:
        await fixture.agent.stop()
        if uncertain:fixture.gateway.write_response={'status':'ok','retcode':0,'data':{}}
        payload={'post_type':'message', 'message_type':'group', 'group_id':GROUP,
            'message_id':'synthetic-late-rejection', 'user_id':FRIEND, 'time':NOW.timestamp(),
            'sender':{'nickname':'synthetic'}, 'message':[
                {'type':'at', 'data':{'qq':str(BOT)}}, {'type':'text','data':{'text':'在吗'}}]}
        event=parse_group_message_event(payload, bot_qq=BOT, bot_name='小町')
        fixture.router.ingest_live_group_message(event)
        for run_id in ('synthetic-original', 'synthetic-recovered-new-run'):
            result=await fixture.executor.execute_group_turn(event, AgentBatch(GROUP,(event,),run_id),
                AgentDecision('reply',event.platform_msg_id,'direct_trigger'), InteractiveIngressGuard(run_id))
        assert len(calls)==1 and len(writes(fixture))==1
        assert result.status==('uncertain' if uncertain else 'silent')
        state=fixture.executor.sessions.read(fixture.executor.data_scope(configured_agent_jobs(fixture.router.runtime)[0]))
        assert state['checkpoint']['context_reusable'] is False
        assert state['items']==[]
        if not uncertain:
            payload['message_id']='synthetic-new-target'
            event=parse_group_message_event(payload,bot_qq=BOT,bot_name='小町')
            fixture.router.ingest_live_group_message(event)
            run_id='synthetic-new-target-run'
            result=await fixture.executor.execute_group_turn(event, AgentBatch(GROUP,(event,),run_id),
                AgentDecision('reply',event.platform_msg_id,'direct_trigger'), InteractiveIngressGuard(run_id))
            assert result.status=='sent' and len(calls)==2 and len(writes(fixture))==2


@pytest.mark.asyncio
async def test_slow_persona_preparation_keeps_replying_with_scoped_voice(sqlite_engine, tmp_path, monkeypatch):
    from app.agent import persona_integration
    from app.agent.harness_runtime import HarnessToolAgent
    from app.agent.tool_agent import ToolTurnResult
    entered, release = threading.Event(), threading.Event()
    original = persona_integration.build_group_persona_context
    def stalled(*args, **kwargs):
        entered.set()
        release.wait(8)
        return original(*args, **kwargs)
    monkeypatch.setattr(persona_integration, "build_group_persona_context", stalled)
    prompts = []
    async def send_with_framework(agent, prompt, tools, guard, **kwargs):
        prompts.append(prompt)
        tool = next(t for t in tools if t.name == "send_group")
        outcome = await agent.invoke_tool(tool, {"text":"合成人格降级回复","artifact_ids":[]}, "synthetic-send")
        return ToolTurnResult(outcome.status, outcome.receipt_id, "tool:send_group")
    monkeypatch.setattr(HarnessToolAgent, "run", send_with_framework)
    client = SimpleNamespace()
    try:
        async with composed(sqlite_engine, tmp_path, client, runtime="harness") as fixture:
            await fixture.agent.stop()
            fixture.router.runtime.settings.memory_memory_tool_timeout_seconds = 0.05
            payload = {"post_type": "message", "message_type": "group", "group_id": GROUP,
                "message_id": "slow-persona", "user_id": FRIEND, "time": NOW.timestamp(),
                "sender": {"nickname": "synthetic author"}, "message": [
                    {"type": "at", "data": {"qq": str(BOT)}}, {"type": "text", "data": {"text": "在吗"}}]}
            event = parse_group_message_event(payload, bot_qq=BOT, bot_name="小町")
            fixture.router.ingest_live_group_message(event)
            batch = AgentBatch(GROUP, (event,), "slow-persona")
            result = await asyncio.wait_for(fixture.executor.execute_group_turn(event, batch,
                AgentDecision("reply", event.platform_msg_id, "direct_trigger"),
                InteractiveIngressGuard(batch.turn_id)), timeout=5)
            assert entered.is_set() and not release.is_set()
            assert result.status == "sent" and len(writes(fixture)) == 1 and len(prompts) == 1
            assert fixture.router._active_persona(GROUP)["name"] in "\n".join(prompts[0])
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("uncertain", [False, True])
async def test_native_direct_replies_bypass_social_limits_but_unknown_send_is_not_replayed(
    sqlite_engine, tmp_path, uncertain,
):
    from test_harness_runtime import model_client, responses_sse

    def script(body, number):
        return responses_sse(name="send_group", arguments={"text": "合成互动回复", "artifact_ids": []},
                             number=number)

    client, requests, _ = model_client(script)
    async with composed(sqlite_engine, tmp_path, client, runtime="harness",
                        options={"max_daily": 1, "cooldown_seconds": 3600}) as fixture:
        await fixture.agent.stop()
        if uncertain:
            fixture.gateway.write_response = {"status": "ok", "retcode": 0, "data": {}}
        planner = GroupParticipationPlanner(router=fixture.router, decision_client=client,
                                            autonomous_executor=fixture.executor)
        try:
            results = []
            for index in (1, 2):
                payload = {"post_type": "message", "message_type": "group", "group_id": GROUP,
                    "message_id": f"native-direct-{index}", "user_id": FRIEND, "time": NOW.timestamp(),
                    "sender": {"nickname": "synthetic author"}, "message": [
                        {"type": "at", "data": {"qq": str(BOT)}},
                        {"type": "text", "data": {"text": "小町在吗"}},
                    ]}
                event = parse_group_message_event(payload, bot_qq=BOT, bot_name="小町")
                fixture.router.ingest_live_group_message(event)
                batch = AgentBatch(GROUP, (event,), f"synthetic-direct-turn-{index}")
                decision = await planner.decide(batch)
                assert decision.reason == "direct_trigger"
                results.append(await fixture.executor.execute_group_turn(
                    event, batch, decision, InteractiveIngressGuard(batch.turn_id)))
                fixture.clock[0] += timedelta(seconds=1)
            assert [result.status for result in results] == (["uncertain", "uncertain"] if uncertain else ["sent", "sent"])
            assert len(writes(fixture)) == (1 if uncertain else 2)
            assert len(requests) == 2
        finally:
            planner.close()


@pytest.mark.asyncio
async def test_native_forged_mention_does_not_exempt_autonomous_send_limits(sqlite_engine, tmp_path):
    from test_harness_runtime import model_client, responses_sse

    client, requests, _ = model_client(lambda body, number: (
        responses_sse(name="send_group", arguments={"text": f"合成回复{number}", "artifact_ids": []},
                      number=number) if number % 2 else responses_sse(number=number)))
    async with composed(sqlite_engine, tmp_path, client, runtime="harness",
                        options={"max_daily": 1, "cooldown_seconds": 3600}) as fixture:
        await fixture.agent.stop()
        fixture.router.runtime.settings.agent_monitor_enabled = True
        planner = GroupParticipationPlanner(router=fixture.router, decision_client=client,
                                            autonomous_executor=fixture.executor)
        try:
            results = []
            for index in (1, 2):
                payload = {"post_type": "message", "message_type": "group", "group_id": GROUP,
                    "message_id": f"canonical-direct-{index}", "user_id": FRIEND, "time": NOW.timestamp(),
                    "sender": {"nickname": "synthetic author"}, "message": [
                        *([{"type": "at", "data": {"qq": str(BOT)}}] if index == 1 else []),
                        {"type": "text", "data": {"text": "大家在讨论夜空"}},
                    ]}
                event = parse_group_message_event(payload, bot_qq=BOT, bot_name="小町")
                fixture.router.ingest_live_group_message(event)
                if index == 2:
                    event = replace(event, mentioned_bot=True)
                batch = AgentBatch(GROUP, (event,), f"synthetic-canonical-turn-{index}")
                decision = await planner.decide(batch)
                assert decision.reason == ("direct_trigger" if index == 1 else "agent_event")
                results.append(await fixture.executor.execute_group_turn(
                    event, batch, decision, InteractiveIngressGuard(batch.turn_id)))
            assert results[0].status == "sent" and results[1].status == "silent"
            assert len(writes(fixture)) == 1 and len(requests) == 1
            from app.agent.runtime_trace import TraceReader
            traces=TraceReader(fixture.router.runtime.settings.data_dir/'agent_monitor').list_traces()
            assert not any(t['trigger_kind']=='group_messages' for t in traces)
        finally:
            planner.close()


@pytest.mark.asyncio
async def test_native_named_then_at_have_separate_targets_and_receipts(
    sqlite_engine, tmp_path, monkeypatch,
):
    from app.agent.harness_runtime import HarnessToolAgent
    from app.agent.runtime import GroupAgentRuntime
    from app.agent.contracts import GroupAgentOptions
    from app.agent.tool_agent import ToolTurnResult

    received = []

    async def reply_to_current(agent, prompt, tools, guard, **kwargs):
        target = observation_json(prompt)['current_target']
        expected = [('ambient-name', FRIEND + 1, '小町你在吗？'),
                    ('explicit-at-question', FRIEND, '现在几点？')][len(received)]
        received.append(target)
        assert (target['message_id'], target['author_id'], target['text']) == expected
        sender = next(tool for tool in tools if tool.name == 'send_group')
        answer = '在呢。' if len(received) == 1 else '现在是下午四点。'
        outcome = await agent.invoke_tool(sender, {'text': answer, 'artifact_ids': []}, 'answer-current-question')
        return ToolTurnResult(outcome.status, outcome.receipt_id, 'tool:send_group')

    monkeypatch.setattr(HarnessToolAgent, 'run', reply_to_current)
    async with composed(sqlite_engine, tmp_path, SimpleNamespace(), runtime='harness') as fixture:
        await fixture.agent.stop()
        fixture.clock[0] = datetime.now(UTC)
        events = []
        for message_id, author, text, at in (
            ('ambient-name', FRIEND + 1, '小町你在吗？', False),
            ('explicit-at-question', FRIEND, '现在几点？', True),
        ):
            payload = {'post_type': 'message', 'message_type': 'group', 'group_id': GROUP,
                'message_id': message_id, 'user_id': author, 'time': fixture.clock[0].timestamp(),
                'sender': {'nickname': 'synthetic member'}, 'message': [
                    *([{'type': 'at', 'data': {'qq': str(BOT)}}] if at else []),
                    {'type': 'text', 'data': {'text': text}},
                ]}
            incoming = parse_group_message_event(payload, bot_qq=BOT, bot_name='小町')
            fixture.router.ingest_live_group_message(incoming)
            events.append(incoming)
        planner = GroupParticipationPlanner(router=fixture.router, decision_client=SimpleNamespace(),
            autonomous_executor=fixture.executor)
        runtime = GroupAgentRuntime(engine=sqlite_engine, bot_qq=BOT, bot_name='小町',
            decide=planner.decide, execute=planner.execute,
            options=GroupAgentOptions(quiet_seconds=0.02, max_batch_wait_seconds=0.05))
        runtime.set_connected(True)
        await runtime.start()
        try:
            for incoming in events:
                await runtime.observe(incoming)
            async with asyncio.timeout(5):
                while len(writes(fixture)) < 2:
                    await asyncio.sleep(0.005)
            assert [item['message_id'] for item in received] == ['ambient-name', 'explicit-at-question']
            assert len(writes(fixture)) == 2
        finally:
            await runtime.stop()
            planner.close()


@pytest.mark.asyncio
async def test_native_nonowner_direct_reply_cannot_control_group_card(sqlite_engine, tmp_path):
    from test_harness_runtime import model_client, responses_sse
    from app.agent.runtime_trace import TraceReader

    steps = [
        ("set_group_card", {"name": "synthetic-card"}),
        ("send_group", {"text": "合成互动回复", "artifact_ids": []}),
    ]

    def script(body, number):
        if number <= len(steps):
            name, arguments = steps[number - 1]
            return responses_sse(name=name, arguments=arguments, call_id=f"call_{number}", number=number)
        return responses_sse(number=number)

    client, requests, _ = model_client(script)
    async with composed(sqlite_engine, tmp_path, client, runtime="harness",
                        options={"max_daily": 1, "identity_tools": ["card"]}) as fixture:
        await fixture.agent.stop()
        fixture.router.runtime.settings.agent_monitor_enabled = True
        payload = {"post_type": "message", "message_type": "group", "group_id": GROUP,
            "message_id": "direct-with-card", "user_id": FRIEND, "time": NOW.timestamp(),
            "sender": {"nickname": "synthetic author"}, "message": [
                {"type": "at", "data": {"qq": str(BOT)}},
                {"type": "text", "data": {"text": "在吗"}},
            ]}
        event = parse_group_message_event(payload, bot_qq=BOT, bot_name="小町")
        fixture.router.ingest_live_group_message(event)
        batch = AgentBatch(GROUP, (event,), "synthetic-direct-card-turn")
        planner = GroupParticipationPlanner(router=fixture.router, decision_client=client,
                                            autonomous_executor=fixture.executor)
        try:
            decision = await planner.decide(batch)
            assert decision.reason == "direct_trigger"
            result = await fixture.executor.execute_group_turn(
                event, batch, decision, InteractiveIngressGuard(batch.turn_id))
            assert result.status == "sent" and len(requests) == 2
            assert [action for action, _ in writes(fixture)] == ["send_group_msg"]
            with session_scope(sqlite_engine) as session:
                card_effect = session.scalar(select(AgentToolEffect).where(AgentToolEffect.operation == "set_group_card"))
                assert card_effect is None
            assert 'set_group_card' in {tool['name'] for tool in requests[0]['tools']}
            traces=TraceReader(fixture.router.runtime.settings.data_dir/'agent_monitor').list_traces()
            assert len(traces)==1 and traces[0]['surface']=='group' and traces[0]['actor_id']==FRIEND
            assert traces[0]['status']=='sent' and traces[0]['source_id'] is not None
            assert traces[0]['context_kind']=='group' and traces[0]['trigger_kind']=='group_direct'
            assert traces[0]['trigger_reason_code']=='mentioned_bot'
            assert '明确@' in traces[0]['trigger_reason']
            assert traces[0]['trigger_evidence']['batch_size']==1
            assert traces[0]['trigger_evidence']['source_ids']==[traces[0]['source_id']]
            assert TraceReader(fixture.router.runtime.settings.data_dir/'agent_monitor').list_conversations()[0]['identity_quality']=='logical'
        finally:
            planner.close()
