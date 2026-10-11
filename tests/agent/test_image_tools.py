"""Owned reference pixels, edit failures and optional public image search tools."""
import json
from types import SimpleNamespace

import httpx
from PIL import Image
import pytest

from app.agent.artifacts import AutonomousArtifactStore
from app.agent.social_contracts import SocialObservation
from app.agent.social_policy import configured_social_jobs
from app.core.message_content import ImageAttachment
from app.storage.db import session_scope
from app.storage.models import Message

from test_autonomous_social import (BOT, GROUP, OTHER_GROUP, NOW, PolicyGuard, ScriptedModel,
    add_source, composed, image_bytes, last_artifact, tool_result, writes)


def saved_reference(fixture, engine, *, group=GROUP, sources=(), content=None, suffix=".png"):
    root = fixture.router.runtime.settings.data_dir / "autonomous_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    path = root / (".reference" + suffix)
    path.write_bytes(content if content is not None else image_bytes())
    library = AutonomousArtifactStore(root, account_id=BOT, scope=f"group:{group}", engine=engine,
        source_message_ids=lambda: sources)
    return library.add(path, "synthetic_actual_reference")


@pytest.mark.asyncio
async def test_reference_tool_edits_real_owned_pixels_and_preserves_labels_size_and_provenance(sqlite_engine, tmp_path):
    reference_id, output_ids = [], []
    original = add_source(sqlite_engine, platform_id="reference-original")
    current = add_source(sqlite_engine, platform_id="reference-current")
    def inspect(lines, kwargs):
        output_ids.append(last_artifact(lines))
        assert tool_result(lines, "generate_image")["status"] == "ok"
        return {"finish": True}
    model = ScriptedModel(lambda *_: {"tool": "generate_image", "arguments": {"prompt": "保持人物画一张竖图",
        "reference_artifact_ids": reference_id, "reference_subjects": ["角色甲"]}}, inspect)
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        reference_id.append(saved_reference(fixture, sqlite_engine, sources=(original,)))
        service = fixture.router.group_image_service
        service._resolve_image_size = lambda **kwargs: "1024x1536"
        service.background, service.output_compression, service.moderation = "transparent", 70, "low"
        job = configured_social_jobs(fixture.router.runtime)[0]
        observation = SocialObservation(job, NOW, "synthetic", source_message_ids=(current,))
        await fixture.executor.run_tools(observation, PolicyGuard(fixture.executor, observation))
        assert fixture.image_client.calls == [] and len(fixture.image_client.edit_calls) == 1
        edit = fixture.image_client.edit_calls[0]
        assert edit["size"] == "1024x1536" and edit["background"] == "transparent"
        assert edit["output_compression"] == 70 and edit["moderation"] == "low" and edit["max_attempts"] == 1
        picture = edit["images"][0]
        assert picture.url == "" and picture.reference_subject == "角色甲"
        assert Image.open(picture.local_path).getpixel((0, 0)) == (255, 0, 0)
        root = fixture.router.runtime.settings.data_dir / "autonomous_artifacts"
        metadata = json.loads((root / (output_ids[0] + ".meta.json")).read_text())
        assert set(metadata["source_message_ids"]) == {original, current}
        assert writes(fixture) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["foreign_group", "recalled", "unsupported", "not_image", "labels"])
async def test_reference_tool_preconditions_fail_without_any_image_provider_call(sqlite_engine, tmp_path, failure):
    original = add_source(sqlite_engine, platform_id="bad-reference-origin", group=OTHER_GROUP if failure == "foreign_group" else GROUP)
    current = add_source(sqlite_engine, platform_id="valid-current-reference-turn")
    reference_id = []
    def inspect(lines, kwargs):
        outcome = tool_result(lines, "generate_image")
        assert outcome["status"] == "blocked"
        assert outcome["result"]["error"] == {
            "foreign_group": "reference_artifact_unavailable", "recalled": "reference_artifact_unavailable",
            "unsupported": "image_edit_not_supported", "not_image": "reference_artifact_not_image",
            "labels": "reference_ids_or_subjects_invalid"}[failure]
        return {"finish": True}
    model = ScriptedModel(lambda *_: {"tool": "generate_image", "arguments": {"prompt": "编辑参考",
        "reference_artifact_ids": reference_id, "reference_subjects": [] if failure == "labels" else ["人物"]}}, inspect)
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        reference_id.append(saved_reference(fixture, sqlite_engine, group=OTHER_GROUP if failure == "foreign_group" else GROUP,
            sources=(original,), content=b"text reference" if failure == "not_image" else None,
            suffix=".txt" if failure == "not_image" else ".png"))
        if failure == "recalled":
            with session_scope(sqlite_engine) as session:
                session.get(Message, original).raw_json = {"delivery_state": "deleted"}
        if failure == "unsupported":
            fixture.image_client.edit_image = None
        job = configured_social_jobs(fixture.router.runtime)[0]
        observation = SocialObservation(job, NOW, "synthetic", source_message_ids=(current,))
        await fixture.executor.run_tools(observation, PolicyGuard(fixture.executor, observation))
        assert fixture.image_client.calls == [] and fixture.image_client.edit_calls == []
        assert writes(fixture) == []


@pytest.mark.asyncio
async def test_edit_provider_failure_never_falls_back_to_text_generation(sqlite_engine, tmp_path):
    reference_id = []
    def inspect(lines, kwargs):
        outcome = tool_result(lines, "generate_image")
        assert outcome["status"] == "failed" and outcome["result"]["error_type"] == "RuntimeError"
        assert outcome["result"]["text_generation_fallback"] is False
        return {"finish": True}
    model = ScriptedModel(lambda *_: {"tool": "generate_image", "arguments": {"prompt": "编辑参考", "reference_artifact_ids": reference_id}}, inspect)
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        reference_id.append(saved_reference(fixture, sqlite_engine))
        def failed_edit(**kwargs):
            fixture.image_client.edit_calls.append(kwargs)
            raise RuntimeError("synthetic edit unavailable")
        fixture.image_client.edit_image = failed_edit
        job = configured_social_jobs(fixture.router.runtime)[0]
        observation = SocialObservation(job, NOW, "synthetic")
        await fixture.executor.run_tools(observation, PolicyGuard(fixture.executor, observation))
        assert len(fixture.image_client.edit_calls) == 1 and fixture.image_client.calls == []
        assert writes(fixture) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["private_address", "redirect", "invalid_pixels", "search_exception"])
async def test_image_reference_search_returns_real_failure_and_never_generates(sqlite_engine, tmp_path, monkeypatch, invalid):
    requests = []
    def http(request):
        requests.append(request)
        if invalid == "redirect":
            return httpx.Response(302, headers={"Location": "http://127.0.0.1/private"})
        return httpx.Response(200, content=b"not image" if invalid == "invalid_pixels" else image_bytes())
    def search(**kwargs):
        if invalid == "search_exception":
            raise httpx.ConnectError("synthetic search failure")
        return [ImageAttachment(url="https://reference.synthetic.invalid/ref.png")]
    model = ScriptedModel({"tool": "search_image_references", "arguments": {"query": "synthetic character", "subject": "人物"}},
        lambda lines, _: {"finish": True, "note": str(tool_result(lines, "search_image_references")["status"])})
    monkeypatch.setattr("app.agent.autonomous_social.socket.getaddrinfo", lambda *_: [(2, 1, 6, "", ("127.0.0.1" if invalid == "private_address" else "8.8.8.8", 443))])
    async with composed(sqlite_engine, tmp_path, model) as fixture:
        service = fixture.router.group_image_service
        service.web_search_client = SimpleNamespace(image_search=search)
        service.http_client = httpx.Client(transport=httpx.MockTransport(http))
        try:
            job = configured_social_jobs(fixture.router.runtime)[0]
            observation = SocialObservation(job, NOW, "synthetic")
            await fixture.executor.run_tools(observation, PolicyGuard(fixture.executor, observation))
            output = tool_result(model.prompts[-1], "search_image_references")
            assert output["status"] == "failed"
            assert output["result"]["error"] in {"image_reference_search_failed", "image_reference_search_no_usable_pixels"}
            assert len(requests) == (0 if invalid in {"private_address", "search_exception"} else 1)
            assert fixture.image_client.calls == [] and fixture.image_client.edit_calls == [] and writes(fixture) == []
        finally:
            service.http_client.close()
