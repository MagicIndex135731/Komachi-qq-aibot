"""Offline QQ/space scenarios through production composition and real Harness.

The synthetic model chooses fixture calls; this verifies execution and business
boundaries, not real-model intelligence. All platform ids and pixels are synthetic.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, UTC
import asyncio
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
from sqlalchemy import select

from app.agent.social_contracts import SocialObservation
from app.agent.social_policy import configured_social_jobs
from app.agent.contracts import GroupAgentOptions
from app.agent.runtime import GroupAgentRuntime
from app.agent.participation import GroupParticipationPlanner
from app.adapters.onebot_models import parse_group_message_event
from app.agent.mcp_tools import McpToolProvider
from app.agent.sandbox_gateway import HttpPythonSandbox, SandboxServer
from app.core.time_utils import ASIA_SHANGHAI
from app.runtime.memory_composition import build_memory_runtime
from app.storage.db import session_scope
from app.storage.models import Message
from app.storage.group_agent import GroupAgentTurn
from app.storage.agent_effects import AgentToolEffect

from test_autonomous_social import (BOT, FRIEND, GROUP, OTHER_GROUP, NOW, PolicyGuard,
                                    actions, add_source, composed, image_bytes, wake, writes)
from test_harness_runtime import model_client, responses_sse
from test_sandbox_gateway import FakeSandbox, TOKEN


def feedback(body):
    result = []
    calls = {item["call_id"]: item["name"] for item in body.get("input", []) if item.get("type") == "function_call"}
    for item in body.get("input", []):
        if item.get("type") != "function_call_output":
            continue
        output = item["output"]
        if isinstance(output, list):
            output = next((part["text"] for part in output if part.get("type") in {"input_text", "text"}), "{}")
        # pi-ai renders images beside a text result on providers whose native
        # tool-output field is text. Parse its leading structured receipt while
        # allowing the framework's image labels after that receipt.
        value, _ = json.JSONDecoder().raw_decode(output[output.find("{"):])
        result.append({"name": calls[item["call_id"]], **value})
    return result


def latest(body, name):
    return next(item for item in reversed(feedback(body)) if item["name"] == name)


def model_steps(*steps):
    def script(body, number):
        command = steps[number - 1] if number <= len(steps) else None
        command = command(body) if callable(command) else command
        return responses_sse(name=command[0], arguments=command[1], call_id=f"call_{number}", number=number) if command else responses_sse(number=number)
    return model_client(script)


@pytest.mark.asyncio
async def test_native_forward_discovery_read_pixels_and_reply(sqlite_engine, tmp_path, monkeypatch):
    import httpx
    from app.agent import group_forward
    origin = add_source(sqlite_engine, platform_id="native-forward-source", text="[合并转发消息]")
    calls = []
    monkeypatch.setattr(group_forward, "media_client", lambda: httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=blue_reference_pixels()))))
    locator = {"source_id": origin, "forward_index": 0, "path": []}
    def expand(body):
        discovery = latest(body, "read_context")
        assert discovery["data"]["forward_sources"] == [{"source_id": origin, "forward_count": 1}]
        return "read_forward_messages", {**locator, "offset": 0, "count": 20}
    def inspect(body):
        quoted = latest(body, "read_forward_messages")
        assert quoted["data"]["nodes"][0]["text"] == "合成转发图"
        assert quoted["data"]["trust"] == "unverified_forwarded_quotation"
        return "observe_forward_images", {**locator, "node_index": 0}
    def reply(body):
        pixels = latest(body, "observe_forward_images")
        assert pixels["status"] == "ok" and pixels["data"]["actual_pixels"]
        assert "data:image/png;base64," in json.dumps(body)
        calls.append(pixels["data"]["images"][0]["artifact_id"])
        return "send_group", {"text": "看到转发里的蓝色图片了", "artifact_ids": []}
    client, requests, _ = model_steps(("read_context", {}), expand, inspect, reply, None)
    async with composed(sqlite_engine, tmp_path, client, runtime="harness", options={"max_steps": 8}) as fixture:
        fixture.router.runtime.settings = fixture.router.runtime.settings.model_copy(update={"llm_supports_vision_input": True})
        with session_scope(sqlite_engine) as session:
            session.get(Message, origin).raw_json = {"message": [{"type": "forward", "data": {"id": "synthetic-native-forward"}}]}
        original = fixture.gateway.call_api
        async def api(action, args):
            if action == "get_forward_msg":
                fixture.gateway.calls.append((action, args))
                return {"status": "ok", "retcode": 0, "data": {"messages": [{"sender": {"nickname": "引用作者"}, "message": [
                    {"type": "text", "data": {"text": "合成转发图"}},
                    {"type": "image", "data": {"url": "https://example.qq.com/synthetic.png"}}]}]}}
            return await original(action, args)
        fixture.gateway.call_api = api
        await wake(fixture)
        assert len(requests) == 5 and len(writes(fixture)) == 1
        assert writes(fixture)[0][0] == "send_group_msg"
        assert effects(sqlite_engine)[0] == ("send_group", "sent", "actual-group-receipt")
        root = fixture.router.runtime.settings.data_dir / "autonomous_artifacts"
        meta = json.loads((root / (calls[0] + ".meta.json")).read_text())
        assert meta["origin"] == "observed_forward_pixels" and meta["source_message_ids"] == [origin]
        assert meta["scope"] == f"group:{GROUP}"


def effects(engine):
    with session_scope(engine) as session:
        return [(row.operation, row.status, row.receipt) for row in session.scalars(select(AgentToolEffect))]


def blue_reference_pixels():
    output = io.BytesIO()
    Image.new("RGB", (32, 24), "blue").save(output, "PNG")
    return output.getvalue()


@pytest.mark.asyncio
async def test_native_group_observed_reference_edits_owned_pixels_and_reuses_output_next_wake(sqlite_engine, tmp_path):
    origin = add_source(sqlite_engine, platform_id="native-edit-reference", text="看看这幅蓝色猫图")
    references, outputs = [], []
    label = "本群成员提供的蓝色猫参考"

    def edit_observed(body):
        observed = latest(body, "observe_group_images")
        assert observed["status"] == "ok" and observed["data"]["actual_pixels"]
        references.append(observed["data"]["images"][0]["artifact_id"])
        return "generate_image", {"prompt": "保留参考轮廓，改为红色星空中的猫",
            "reference_artifact_ids": references, "reference_subjects": [label]}

    def read_edited(body):
        output = latest(body, "generate_image")
        assert output["status"] == "ok" and output["data"]["ready"]
        outputs.append(output["data"]["artifact_id"])
        return "read_artifact", {"artifact_id": outputs[0]}

    client, requests, _ = model_steps(("observe_group_images", {"source_ids": [origin]}),
        edit_observed, read_edited,
        ("remember_intention", {"goal": "下次继续欣赏由真实参考编辑的猫画", "progress": "编辑产物已保存", "next_step": "再观察作品是否适合分享"}), None,
        lambda _: ("read_artifact", {"artifact_id": outputs[0]}),
        lambda _: ("send_group", {"text": "继续分享我保存的参考编辑作品", "artifact_ids": outputs}),
        ("set_group_card", {"name": "小町的猫画时间"}), None)
    async with composed(sqlite_engine, tmp_path, client, runtime="harness", options={"max_steps": 10, "identity_tools": ["card"]}) as fixture:
        cache = fixture.router.runtime.settings.data_dir / "image_cache" / str(GROUP)
        cache.mkdir(parents=True, exist_ok=True)
        reference_path = cache / "native-edit-reference.png"
        reference_bytes = blue_reference_pixels()
        reference_path.write_bytes(reference_bytes)
        with session_scope(sqlite_engine) as session:
            session.get(Message, origin).raw_json = {"message": [{"type": "image", "data": {"local_path": str(reference_path)}}]}
        fixture.router.runtime.settings = fixture.router.runtime.settings.model_copy(update={"llm_supports_vision_input": True})
        await wake(fixture)
        assert len(requests) == 5 and writes(fixture) == []
        assert fixture.image_client.calls == [] and len(fixture.image_client.edit_calls) == 1
        edit = fixture.image_client.edit_calls[0]
        assert edit["max_attempts"] == 1 and len(edit["images"]) == 1
        owned_reference = Path(edit["images"][0].local_path)
        artifact_root = fixture.router.runtime.settings.data_dir / "autonomous_artifacts"
        assert owned_reference.is_relative_to(artifact_root)
        assert owned_reference != reference_path and owned_reference.read_bytes() == reference_bytes
        assert edit["images"][0].reference_subject == label
        reference_metadata = json.loads((artifact_root / (references[0] + ".meta.json")).read_text(encoding="utf8"))
        output_metadata = json.loads((artifact_root / (outputs[0] + ".meta.json")).read_text(encoding="utf8"))
        assert reference_metadata["origin"] == "observed_group_pixels"
        assert reference_metadata["scope"] == output_metadata["scope"] == f"group:{GROUP}"
        assert reference_metadata["source_message_ids"] == output_metadata["source_message_ids"] == [origin]
        assert "data:image/png;base64," in json.dumps(requests[1])
        assert latest(requests[3], "read_artifact")["data"]["pixels_attached"]
        first_session = fixture.executor.sessions.read(f"group:{GROUP}")["checkpoint"]["session_id"]
        await wake(fixture)
        assert fixture.executor.sessions.read(f"group:{GROUP}")["checkpoint"]["session_id"] == first_session
        assert len(requests) == 9 and fixture.image_client.calls == [] and len(fixture.image_client.edit_calls) == 1
        assert "编辑产物已保存" in json.dumps(requests[5], ensure_ascii=False)
        assert [action for action, _ in writes(fixture)] == ["send_group_msg", "set_group_card"]
        encoded = next(part["data"]["file"] for part in writes(fixture)[0][1]["message"] if part["type"] == "image")
        assert base64.b64decode(encoded.removeprefix("base64://")) == (artifact_root / output_metadata["file"]).read_bytes()
        assert effects(sqlite_engine)[0] == ("send_group", "sent", "actual-group-receipt")
        assert writes(fixture)[1][1] == {"group_id": GROUP, "user_id": BOT, "card": "小町的猫画时间"}
        acknowledgement = latest(requests[-1], "set_group_card")
        assert acknowledgement["status"] == "sent" and acknowledgement["data"]["bridge_acknowledged"]
        assert acknowledgement["data"]["readback_verified"] is False


@pytest.mark.asyncio
async def test_native_public_agent_searches_reference_pixels_then_edits_and_publishes_owned_output(sqlite_engine, tmp_path, monkeypatch):
    import httpx
    from app.agent import autonomous_social
    from app.providers.llm_client import ImageAttachment

    reference_url, label = "https://reference.synthetic.invalid/blue-cat.png", "公开蓝色猫轮廓参考"
    reference_bytes = blue_reference_pixels()
    searches, downloads, references, outputs = [], [], [], []

    def search(*, query, max_results):
        searches.append((query, max_results))
        return [ImageAttachment(url=reference_url, reference_subject="合成搜索提供的公开图片")]

    def pixels(request):
        downloads.append(str(request.url))
        assert str(request.url) == reference_url
        return httpx.Response(200, content=reference_bytes, headers={"Content-Type": "image/png"})

    def edit_searched(body):
        result = latest(body, "search_image_references")
        assert result["status"] == "ok" and result["data"]["actual_pixels"]
        assert result["data"]["source"] == "public_image_search"
        reference = result["data"]["references"][0]
        assert reference["url"] == reference_url and reference["subject"] == label
        references.append(reference["artifact_id"])
        return "generate_image", {"prompt": "保持蓝猫的轮廓，把它画成红色星空猫",
            "reference_artifact_ids": references, "reference_subjects": [label]}

    def inspect_output(body):
        result = latest(body, "generate_image")
        assert result["status"] == "ok" and result["data"]["ready"]
        outputs.append(result["data"]["artifact_id"])
        return "read_artifact", {"artifact_id": outputs[0]}

    client, requests, _ = model_steps(("search_image_references", {"query": "公开蓝猫参考", "subject": label}),
        edit_searched, inspect_output,
        lambda _: ("publish_qzone", {"text": "由公开参考继续探索猫画", "artifact_ids": outputs}),
        lambda _: ("set_avatar", {"artifact_id": outputs[0]}), None)
    original_dns = autonomous_social.socket.getaddrinfo
    monkeypatch.setattr(autonomous_social.socket, "getaddrinfo", lambda host, port, *args, **kwargs:
        [(autonomous_social.socket.AF_INET, autonomous_social.socket.SOCK_STREAM, 6, "", ("8.8.8.8", port))]
        if host == "reference.synthetic.invalid" else original_dns(host, port, *args, **kwargs))
    with httpx.Client(transport=httpx.MockTransport(pixels)) as transport:
        async with composed(sqlite_engine, tmp_path, client, runtime="harness", kind="qzone_publish", options={"max_steps": 8}) as fixture:
            fixture.router.group_image_service.web_search_client = SimpleNamespace(image_search=search)
            fixture.router.group_image_service.http_client = transport
            fixture.router.runtime.social_policy["profile"] = {"enabled": True, "autonomous_tools": True,
                "identity_tools": ["avatar"], "interval_seconds": 600, "cooldown_seconds": 60,
                "max_daily": 10, "quiet_hours": None}
            await wake(fixture)
            assert searches == [("公开蓝猫参考", 4)] and downloads == [reference_url]
            assert len(requests) == 6 and fixture.image_client.calls == [] and len(fixture.image_client.edit_calls) == 1
            assert "search_image_references" in {tool["name"] for tool in requests[0]["tools"]}
            assert "data:image/png;base64," in json.dumps(requests[1])
            edit = fixture.image_client.edit_calls[0]
            assert edit["max_attempts"] == 1 and len(edit["images"]) == 1
            owned_reference = Path(edit["images"][0].local_path)
            artifact_root = fixture.router.runtime.settings.data_dir / "autonomous_artifacts"
            assert owned_reference.is_relative_to(artifact_root) and owned_reference.read_bytes() == reference_bytes
            assert edit["images"][0].reference_subject == label
            reference_metadata = json.loads((artifact_root / (references[0] + ".meta.json")).read_text(encoding="utf8"))
            output_metadata = json.loads((artifact_root / (outputs[0] + ".meta.json")).read_text(encoding="utf8"))
            assert reference_metadata["origin"] == "web_reference_pixels"
            assert reference_metadata["scope"] == output_metadata["scope"] == "public"
            assert reference_metadata["source_message_ids"] == output_metadata["source_message_ids"] == []
            assert latest(requests[3], "read_artifact")["data"]["pixels_attached"]
            assert "PRIVATE_PERSONA_FACT" not in json.dumps(requests, ensure_ascii=False)
            assert [action for action, _ in writes(fixture)] == ["send_qzone_msg", "set_qq_avatar"]
            encoded = writes(fixture)[0][1]["images"][0]
            assert base64.b64decode(encoded.removeprefix("base64://")) == (artifact_root / output_metadata["file"]).read_bytes()
            assert effects(sqlite_engine)[0] == ("publish_qzone", "sent", "actual-published-tid")
            avatar = writes(fixture)[1][1]["file"]
            assert base64.b64decode(avatar.removeprefix("base64://")) == (artifact_root / output_metadata["file"]).read_bytes()
            acknowledgement = latest(requests[-1], "set_avatar")
            assert acknowledgement["status"] == "sent" and acknowledgement["data"]["bridge_acknowledged"]
            assert acknowledgement["data"]["readback_verified"] is False


@pytest.mark.asyncio
async def test_native_production_group_pixels_history_memory_persona_goal_and_artifact_reuse(sqlite_engine, tmp_path):
    source = add_source(sqlite_engine, platform_id="native-source", text="我喜欢猫和星星")
    add_source(sqlite_engine, platform_id="foreign-source", group=OTHER_GROUP, text="我喜欢猫和 FOREIGN_SECRET")
    saved = []
    def use_generated(body):
        saved.append(latest(body, "generate_image")["data"]["artifact_id"])
        return "remember_intention", {"goal": "复用猫画探索夜空主题", "progress": "图片已保存在真实产物库", "next_step": "下次唤醒再观察"}
    def task_status(body):
        task = latest(body, "request_persona_refresh")["data"]
        assert task["status"] == "queued" and task["paid_execution"] == "paused_by_configuration"
        return "persona_task_status", {"task_id": task["task_id"]}
    client, requests, _ = model_steps(
        ("read_persona", {}), ("search_history", {"query": "猫"}),
        ("observe_group_images", {"source_ids": [source]}),
        ("memory_write", {"kind": "preference", "subject": str(FRIEND), "predicate": "likes", "object_text": "猫和星星",
                           "content": "喜欢猫和星星", "source_msg_ids": ["native-source"]}),
        ("memory_read", {"member": str(FRIEND)}),
        ("request_persona_refresh", {"persona_key": "synthetic", "request_key": "native-refresh"}), task_status,
        ("generate_image", {"prompt": "根据公开兴趣画猫和星星"}), use_generated, None,
        ("list_artifacts", {}), lambda body: ("read_artifact", {"artifact_id": saved[0]}),
        lambda body: ("send_group", {"text": "继续我上次画的猫", "artifact_ids": saved}), None)
    async with composed(sqlite_engine, tmp_path, client, runtime="harness", options={"max_steps": 16}) as fixture:
        cache = fixture.router.runtime.settings.data_dir / "image_cache" / str(GROUP)
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / "native-source.png"
        path.write_bytes(image_bytes())
        with session_scope(sqlite_engine) as session:
            row = session.get(Message, source)
            row.raw_json = {"message": [{"type": "image", "data": {"local_path": str(path)}}]}
        fixture.router.runtime.settings = fixture.router.runtime.settings.model_copy(update={
            "memory_memory_tools_enabled": True, "llm_supports_vision_input": True,
            "memory_orchestration_v2_enabled": True, "memory_orchestration_shadow_mode": False,
            "memory_layered_memory_enabled": True, "memory_raw_v3_enabled": True,
            "memory_embedding_provider": "disabled", "memory_query_rewrite_enabled": False})
        fixture.router.memory_orchestrator = build_memory_runtime(settings=fixture.router.runtime.settings,
            engine=sqlite_engine, llm_client=SimpleNamespace(), bot_display_name="小町").memory_orchestrator
        fixture.router.runtime.group_policy["groups"][str(GROUP)]["memory_enabled"] = True
        fixture.router.runtime.social_policy["services"] = {"memory_write": True, "persona_refresh": True}
        fixture.router.runtime.personas["synthetic"] = {"live_refresh": True, "source_group_id": GROUP, "source_user_id": FRIEND}
        await wake(fixture)
        assert len(requests) == 10 and writes(fixture) == []
        assert latest(requests[4], "memory_write")["status"] == "ok"
        assert "猫和星星" in latest(requests[5], "memory_read")["data"]["evidence"]
        assert latest(requests[3], "observe_group_images")["data"]["actual_pixels"]
        assert "data:image/png;base64," in json.dumps(requests[3])
        assert "FOREIGN_SECRET" not in json.dumps(requests, ensure_ascii=False)
        assert latest(requests[1], "read_persona")["data"]["expression_style"] in requests[0]["input"][0]["content"]
        state = fixture.executor.sessions.read(f"group:{GROUP}")
        assert source in state["source_message_ids"]
        first_session = state["checkpoint"]["session_id"]
        await wake(fixture)
        assert fixture.executor.sessions.read(f"group:{GROUP}")["checkpoint"]["session_id"] == first_session
        assert "复用猫画探索夜空主题" in json.dumps(requests[10], ensure_ascii=False)
        assert len(fixture.image_client.calls) == 1
        assert [a for a, _ in writes(fixture)] == ["send_group_msg"]
        recorded = effects(sqlite_engine)
        assert recorded[0][:2] == ("send_group", "sent")
        assert recorded[0][2] == "actual-group-receipt"
        assert len(requests) == 14
        assert fixture.executor.sessions.read(f"group:{GROUP}")["checkpoint"]["receipts"][0]["status"] == "sent"


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_execution", [False, True])
async def test_native_persona_refresh_receipt_tracks_current_execution_configuration(
        sqlite_engine, tmp_path, initial_execution):
    from app.agent.service_tools import PersonaTaskService

    add_source(sqlite_engine, platform_id="native-persona-refresh-source", text="合成的人格刷新来源")
    client, requests, _ = model_steps(
        ("request_persona_refresh", {"persona_key": "synthetic", "request_key": "native-refresh-before"}), None,
        ("request_persona_refresh", {"persona_key": "synthetic", "request_key": "native-refresh-after"}), None)
    async with composed(sqlite_engine, tmp_path, client, runtime="harness") as fixture:
        services = {"persona_refresh": True, "persona_task_execution": initial_execution}
        fixture.router.runtime.social_policy["services"] = services
        fixture.router.runtime.personas["synthetic"] = {
            "live_refresh": True, "source_group_id": GROUP, "source_user_id": FRIEND}
        tasks = PersonaTaskService(sqlite_engine, account_id=BOT, clock=lambda: fixture.clock[0])
        task_ids = []
        for execution_enabled in (initial_execution, not initial_execution):
            services["persona_task_execution"] = execution_enabled
            await wake(fixture)
            receipt = latest(requests[-1], "request_persona_refresh")
            assert receipt["status"] == "ok"
            task = receipt["data"]
            assert task["paid_execution"] == ("enabled" if execution_enabled else "paused_by_configuration")
            assert task["status"] == "queued"
            assert tasks.status(group_id=GROUP, task_id=task["task_id"])["status"] == "queued"
            task_ids.append(task["task_id"])
        assert len(set(task_ids)) == 2 and len(requests) == 4
        assert writes(fixture) == []


@pytest.mark.asyncio
async def test_native_public_agent_selects_friend_read_comment_publish_and_profile_by_permissions(sqlite_engine, tmp_path):
    saved = []
    def remember_generated(body):
        saved.append(latest(body, "generate_image")["data"]["artifact_id"])
        return "read_friend_posts", {"owner_id": FRIEND}
    client, requests, _ = model_steps(
        ("generate_image", {"prompt": "公开猫绘画"}), remember_generated,
        lambda body: ("comment_qzone", {"owner_id": FRIEND, "post_id": latest(body, "read_friend_posts")["data"]["posts"][0]["post_id"],
                                       "text": "猫的照片很有意思", "artifact_ids": saved}),
        lambda body: ("publish_qzone", {"text": "今天想继续探索猫与绘画", "artifact_ids": saved}),
        ("set_nickname", {"name": "小町的绘画时间"}), None)
    async with composed(sqlite_engine, tmp_path, client, runtime="harness", kind="qzone_publish", options={"max_steps": 10}) as fixture:
        fixture.router.runtime.settings = fixture.router.runtime.settings.model_copy(update={"agent_max_external_writes": 3})
        common = {"enabled": True, "autonomous_tools": True, "allow_images": True, "interval_seconds": 600,
                  "cooldown_seconds": 60, "max_daily": 10, "quiet_hours": None, "topics": ["猫与绘画"]}
        fixture.router.runtime.social_policy["qzone"]["comment"] = {**common, "friend_ids": [FRIEND]}
        fixture.router.runtime.social_policy["profile"] = {**common, "identity_tools": ["nickname"]}
        await wake(fixture)
        assert [a for a, _ in writes(fixture)] == ["comment_qzone", "send_qzone_msg", "set_qq_profile"]
        assert {row[0] for row in effects(sqlite_engine)} == {"comment_qzone", "publish_qzone", "set_nickname"}
        assert all(row[1] == "sent" for row in effects(sqlite_engine))
        assert "PRIVATE_PERSONA_FACT" not in json.dumps(requests, ensure_ascii=False)
        assert actions(sqlite_engine)[0][0] == "sent"


@pytest.mark.asyncio
async def test_native_comment_scope_defers_choice_to_framework_and_deduplicates_across_wakes(sqlite_engine, tmp_path):
    client, requests, _ = model_steps(
        ("read_friend_posts", {"owner_id": FRIEND}),
        ("comment_qzone", {"owner_id": FRIEND, "post_id": "observed-photo", "text": "今天的猫很有趣", "artifact_ids": []}), None,
        ("read_friend_posts", {"owner_id": FRIEND}),
        ("comment_qzone", {"owner_id": FRIEND, "post_id": "observed-photo", "text": "今天的猫很有趣", "artifact_ids": []}), None)
    async with composed(sqlite_engine, tmp_path, client, runtime="harness", kind="qzone_comment", options={"max_steps": 8}) as fixture:
        await wake(fixture)
        assert [a for a, _ in writes(fixture)] == ["comment_qzone"]
        await wake(fixture)
        assert [a for a, _ in writes(fixture)] == ["comment_qzone"]
        assert "effect_already_recorded" in json.dumps(requests[-1])


@pytest.mark.asyncio
async def test_native_public_friend_scope_and_actual_post_pixels_drive_comment(sqlite_engine, tmp_path, monkeypatch):
    import httpx
    from app.agent import public_tools
    from app.agent.social_images import post_image_inputs
    fetched = []
    def pixels(request):
        fetched.append(str(request.url))
        return httpx.Response(200, content=image_bytes(), headers={"Content-Type": "image/png"})
    client, requests, _ = model_steps(("read_friend_posts", {"owner_id": FRIEND + 1}),
        ("read_friend_posts", {"owner_id": FRIEND}),
        ("comment_qzone", {"owner_id": FRIEND, "post_id": "observed-photo", "text": "实际像素中的红色很鲜明", "artifact_ids": []}),
        ("observe_post_images", {"owner_id": FRIEND, "post_id": "observed-photo"}),
        ("comment_qzone", {"owner_id": FRIEND, "post_id": "observed-photo", "text": "实际像素中的红色很鲜明", "artifact_ids": []}), None)
    with httpx.Client(transport=httpx.MockTransport(pixels)) as transport:
        monkeypatch.setattr(public_tools, "post_image_inputs", lambda post, *, cache_dir:
            post_image_inputs(post, cache_dir=cache_dir, http_client=transport))
        async with composed(sqlite_engine, tmp_path, client, runtime="harness", kind="qzone_comment") as fixture:
            fixture.gateway.posts[0].update(content="", images=["https://synthetic.qpic.cn/offline.png"])
            await wake(fixture)
            assert latest(requests[1], "read_friend_posts")["status"] == "blocked"
            assert fetched == ["https://synthetic.qpic.cn/offline.png"]
            assert latest(requests[3], "comment_qzone")["status"] == "blocked"
            assert latest(requests[4], "observe_post_images")["data"]["pixels_attached"]
            assert "data:image/png;base64," in json.dumps(requests[4])
            assert [a for a, _ in writes(fixture)] == ["comment_qzone"]
            assert actions(sqlite_engine)[0][0] == "sent"


@pytest.mark.asyncio
async def test_native_scheduler_persisted_goal_wakes_without_fixed_social_interval(sqlite_engine, tmp_path):
    client, requests, _ = model_steps(("schedule_wakeup", {"goal": "稍后继续核对星空主题", "delay_seconds": 15}), None,
        ("read_work_note", {}), ("remember_intention", {"goal": "星空主题", "progress": "已在到期目标唤醒中继续", "next_step": "暂时等待"}), None)
    async with composed(sqlite_engine, tmp_path, client, runtime="harness", options={"max_steps": 8}) as fixture:
        await wake(fixture)
        initial_count = len(requests)
        fixture.clock[0] += timedelta(seconds=16)
        await fixture.agent.tick(fixture.clock[0])
        assert len(requests) > initial_count
        assert "稍后继续核对星空主题" in json.dumps(requests[initial_count], ensure_ascii=False)
        assert writes(fixture) == []
        assert "已在到期目标唤醒中继续" in json.dumps(fixture.executor.repository.read_intention(f"group:{GROUP}"), ensure_ascii=False)


@pytest.mark.asyncio
async def test_native_dormant_group_goals_wait_without_periodic_or_paid_polling(sqlite_engine, tmp_path):
    client, requests, _ = model_steps(("read_work_note", {}), None)
    async with composed(sqlite_engine, tmp_path, client, runtime="harness") as fixture:
        fixture.router.runtime.settings = fixture.router.runtime.settings.model_copy(update={
            "social_agent_enabled": False, "group_agent_enabled": True})
        fixture.router.runtime.social_policy["enabled"] = False
        fixture.router.runtime.group_policy["groups"][str(GROUP)]["proactive_reply"] = False
        fixture.clock[0] += timedelta(hours=1)
        await fixture.agent.tick(fixture.clock[0])
        assert requests == []
        fixture.executor.repository.write_intention(f"group:{GROUP}", {
            "goal": "由已授权事件保存的长期兴趣", "progress": "等待到期", "next_step": "读取笔记"}, (),
            next_due=fixture.clock[0] + timedelta(seconds=15), status="waiting")
        await fixture.agent.tick(fixture.clock[0])
        assert requests == []
        fixture.clock[0] += timedelta(seconds=16)
        await fixture.agent.tick(fixture.clock[0])
        assert len(requests) == 2
        assert "由已授权事件保存的长期兴趣" in json.dumps(requests[0], ensure_ascii=False)
        assert "send_group" not in {t["name"] for t in requests[0]["tools"]}
        fixture.clock[0] += timedelta(hours=1)
        await fixture.agent.tick(fixture.clock[0])
        assert len(requests) == 2 and writes(fixture) == []


@pytest.mark.asyncio
async def test_native_unknown_gateway_receipt_persists_uncertainty_and_never_auto_replays(sqlite_engine, tmp_path):
    client, requests, _ = model_steps(("send_group", {"text": "合成消息", "artifact_ids": []}),
        ("read_persona", {}), None)
    async with composed(sqlite_engine, tmp_path, client, runtime="harness", options={"max_steps": 8}) as fixture:
        fixture.gateway.write_response = {"status": "ok", "retcode": 0, "data": {}}
        await wake(fixture)
        assert len(requests) == 1
        assert len(writes(fixture)) == 1 and effects(sqlite_engine)[0][1] == "uncertain"
        state = fixture.executor.sessions.read(f"group:{GROUP}")
        assert state["checkpoint"]["receipts"][0]["status"] == "uncertain"
        await wake(fixture)
        assert len(writes(fixture)) == 1


@pytest.mark.asyncio
async def test_native_daily_budget_blocks_continuation_before_an_external_write(sqlite_engine, tmp_path):
    client, requests, _ = model_steps(("read_context", {}), ("send_group", {"text": "超预算消息", "artifact_ids": []}))
    async with composed(sqlite_engine, tmp_path, client, runtime="harness", options={"max_model_calls_daily": 1}) as fixture:
        await wake(fixture)
        assert len(requests) == 1 and writes(fixture) == []
        state = fixture.executor.sessions.read(f"group:{GROUP}")
        assert state["checkpoint"]["last_status"] == "blocked"
        assert "model_daily_budget" in json.dumps(state["items"])


@pytest.mark.asyncio
async def test_native_revoked_sources_rotate_framework_history_and_block_saved_pixels(sqlite_engine, tmp_path):
    origin = add_source(sqlite_engine, platform_id="native-revoked-origin", text="SOURCE_TO_REVOKE")
    saved = []
    def save(body):
        saved.append(latest(body, "generate_image")["data"]["artifact_id"])
        return "save_work_note", {"note": "由撤回来源派生的笔记"}
    client, requests, _ = model_steps(("generate_image", {"prompt": "合成猫"}), save, None,
        lambda _: ("read_artifact", {"artifact_id": saved[0]}), None)
    async with composed(sqlite_engine, tmp_path, client, runtime="harness", options={"max_steps": 8}) as fixture:
        job = configured_social_jobs(fixture.router.runtime)[0]
        observation = SocialObservation(job, NOW, "小町", ("SOURCE_TO_REVOKE",), (origin,))
        await fixture.executor.run_tools(observation, PolicyGuard(fixture.executor, observation))
        first = fixture.executor.sessions.read(job.key)["checkpoint"]["session_id"]
        with session_scope(sqlite_engine) as session:
            session.get(Message, origin).raw_json = {"delivery_state": "deleted"}
        current = add_source(sqlite_engine, platform_id="native-safe-current", text="SAFE_CURRENT")
        later = replace(observation, context=("SAFE_CURRENT",), source_message_ids=(current,))
        await fixture.executor.run_tools(later, PolicyGuard(fixture.executor, later))
        assert fixture.executor.sessions.read(job.key)["checkpoint"]["session_id"] != first
        assert "SOURCE_TO_REVOKE" not in json.dumps(requests[3], ensure_ascii=False)
        assert latest(requests[-1], "read_artifact")["status"] in {"failed", "blocked"}
        assert writes(fixture) == []


@pytest.mark.asyncio
async def test_native_inflight_checkpoint_rotates_uncommitted_framework_history(sqlite_engine, tmp_path):
    client, requests, _ = model_client(lambda _, n: responses_sse(
        text="UNCOMMITTED_NATIVE_HISTORY" if n == 1 else "safe resumed work", number=n))
    async with composed(sqlite_engine, tmp_path, client, runtime="harness") as fixture:
        job = configured_social_jobs(fixture.router.runtime)[0]
        observation = SocialObservation(job, NOW, "synthetic persona")
        await fixture.executor.run_tools(observation, PolicyGuard(fixture.executor, observation))
        state = fixture.executor.sessions.read(job.key)
        initial = state["checkpoint"]["session_id"]
        # Simulate a killed process after its native log was appended but the
        # project's source/effect checkpoint had not reached a closed turn.
        fixture.executor.sessions.begin_turn(job.key, state["checkpoint"], (),
                                             expected_revision=state["revision"])
        await fixture.executor.run_tools(observation, PolicyGuard(fixture.executor, observation))
        assert len(requests) == 2
        assert fixture.executor.sessions.read(job.key)["checkpoint"]["session_id"] != initial
        assert "UNCOMMITTED_NATIVE_HISTORY" not in json.dumps(requests[1])
        assert writes(fixture) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("send", [False, True])
async def test_native_unaddressed_group_event_owns_decision_silence_and_continuation(sqlite_engine, tmp_path, send):
    client, requests, _ = model_steps(*([
        ("send_group", {"text": "夜空的颜色很有趣", "artifact_ids": []}), ("read_persona", {}), None] if send else [None]))
    async with composed(sqlite_engine, tmp_path, client, runtime="harness") as fixture:
        await fixture.agent.stop()
        at = datetime.now(UTC)
        fixture.clock[0] = at
        row_id = add_source(sqlite_engine, platform_id="native-interactive-event", text="大家今天在聊夜空")
        payload = {"post_type": "message", "message_type": "group", "group_id": GROUP,
            "message_id": "native-interactive-event", "user_id": FRIEND, "time": at.timestamp(),
            "sender": {"nickname": "synthetic", "card": ""},
            "message": [{"type": "text", "data": {"text": "大家今天在聊夜空"}}]}
        event = parse_group_message_event(payload, bot_qq=BOT, bot_name="小町")
        assert not event.mentioned_bot
        with session_scope(sqlite_engine) as session:
            source = session.get(Message, row_id)
            source.timestamp, source.raw_json = at.astimezone(ASIA_SHANGHAI).replace(tzinfo=None), payload
        planner = GroupParticipationPlanner(router=fixture.router, decision_client=client,
                                             autonomous_executor=fixture.executor)
        runtime = GroupAgentRuntime(engine=sqlite_engine, bot_qq=BOT, bot_name="小町",
            decide=planner.decide, execute=planner.execute, options=GroupAgentOptions(
                quiet_seconds=0.01, max_batch_wait_seconds=0.03, decision_timeout_seconds=5,
                execution_timeout_seconds=30, shutdown_timeout_seconds=5))
        await runtime.start()
        runtime.set_connected(True)
        try:
            assert await runtime.observe(event)
            assert not await runtime.observe(event)
            async with asyncio.timeout(15):
                while True:
                    with session_scope(sqlite_engine) as session:
                        turn = session.scalars(select(GroupAgentTurn)).first()
                        status = turn.status if turn else None
                    if status in {"sent", "silent", "failed", "blocked", "uncertain"}:
                        break
                    await asyncio.sleep(0.02)
            assert status == ("sent" if send else "silent")
            assert len(requests) == (3 if send else 1)
            assert len(writes(fixture)) == int(send)
            checkpoint = fixture.executor.sessions.read(f"group:{GROUP}")["checkpoint"]
            assert checkpoint["last_status"] == status
            assert len(checkpoint["receipts"]) == int(send)
        finally:
            await runtime.stop()
            planner.close()


@pytest.mark.asyncio
async def test_native_registered_mcp_search_and_compute_feedback_plus_unknown_write_are_composable(sqlite_engine, tmp_path):
    import httpx
    mcp_calls, searches = [], []
    config = {"enabled": True, "url": "http://127.0.0.1:9001/mcp", "tools": [
        {"name": name, "description": "synthetic approved external service", "effect": effect, "scopes": scopes,
         "parameters": {"type": "object", "properties": {"expression": {"type": "string", "maxLength": 100}},
                        "required": ["expression"]}, "acknowledgement_key": "receipt"}
        for name, effect, scopes in (("calculate", "read", ["group"]), ("remote_write", "write", ["group"]),
                                    ("forbidden_public_action", "read", ["profile"]))]}
    def external(request):
        body = json.loads(request.content)
        mcp_calls.append(body)
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
        if body["method"] == "initialize":
            result = {"protocolVersion": "2025-06-18"}
        elif body["params"]["name"] == "remote_write":
            raise httpx.ReadTimeout("synthetic unknown external acknowledgement")
        else:
            result = {"content": [{"type": "text", "text": "42"}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})
    def search(query, *, max_results):
        searches.append((query, max_results))
        return [SimpleNamespace(title="合成天文资料", snippet="模拟数据提及42", source="https://example.com/source", date="2026-10-09")]
    def compute(body):
        assert latest(body, "calculate")["data"]["mcp_result"]["content"][0]["text"] == "42"
        assert latest(body, "search_web")["data"]["sources"][0]["title"] == "合成天文资料"
        return "execute_python", {"code": "print(6*7)"}
    client, requests, _ = model_steps(("calculate", {"expression": "6*7"}), ("search_web", {"query": "合成天文数字"}), compute,
        lambda body: ("read_artifact", {"artifact_id": latest(body, "execute_python")["data"]["artifacts"][0]["artifact_id"]}), None,
        ("remote_write", {"expression": "synthetic write"}), ("remote_write", {"expression": "synthetic retry must not run"}))
    async with composed(sqlite_engine, tmp_path, client, runtime="harness", compute=True) as fixture:
        fake = FakeSandbox(tmp_path / "synthetic-compute-service")
        server = SandboxServer(fake, token=TOKEN, host="127.0.0.1", port=0)
        await server.start()
        sandbox = HttpPythonSandbox(f"http://127.0.0.1:{server.port}", token=TOKEN,
            output_root=fixture.router.runtime.settings.data_dir / "autonomous_artifacts" / "sandbox-client")
        fixture.executor.sandbox = sandbox
        fixture.router.web_search_client = SimpleNamespace(search=search)
        fixture.router.runtime.social_policy["mcp"] = [config]
        async with httpx.AsyncClient(transport=httpx.MockTransport(external)) as transport:
            provider = McpToolProvider(config, client=transport,
                permission_check=lambda: config in fixture.router.runtime.social_policy["mcp"])
            fixture.executor.register_tool_factory(provider.tools_for)
            try:
                await wake(fixture)
                assert len(requests) == 5 and searches == [("合成天文数字", 8)]
                assert fake.calls == [("print(6*7)", {})]
                assert '"answer":42' in latest(requests[-1], "read_artifact")["data"]["text"]
                assert "forbidden_public_action" not in {t["name"] for t in requests[0]["tools"]}
                await wake(fixture)
                assert len(requests) == 6
                assert sum(c.get("params", {}).get("name") == "remote_write" for c in mcp_calls) == 1
                assert effects(sqlite_engine)[0][:2] == ("remote_write", "uncertain")
                assert writes(fixture) == []
            finally:
                await sandbox.aclose()
                await server.close()
