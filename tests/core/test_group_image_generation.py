from __future__ import annotations

import asyncio
import base64
import io
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
import httpx

import pytest
from PIL import Image
from sqlalchemy import text

from app.core.group_image_generation import (
    GroupImageGenerationRequest,
    GroupImageGenerationService,
    PrivateImageGenerationRequest,
    PrivateImageGenerationService,
)
from app.adapters.outbound_guard import GroupDispatchScope, group_dispatch_scope
from app.adapters.sender import Sender
from app.core.message_content import ImageAttachment
from app.providers.llm_client import ImageGenerationResult
from app.storage.db import session_scope
from app.storage.repositories import GroupRepository, JobRepository, MessageRepository, UserRepository


class FakeGroupImageSender:
    def __init__(self) -> None:
        self.image_calls: list[dict] = []
        self.text_calls: list[dict] = []

    async def send_group_image(self, *, group_id: int, image_file: str) -> None:
        self.image_calls.append({"group_id": group_id, "image_file": image_file})

    async def send_group_text(self, outbound) -> None:
        self.text_calls.append({"group_id": outbound.group_id, "text": outbound.text})


class BlockingImageLlm:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = threading.Event()

    def generate_image(
        self,
        *,
        prompt: str,
        model: str,
        size=None,
        quality=None,
        background=None,
        output_format=None,
        output_compression=None,
        moderation=None,
        max_attempts=None,
        timeout_seconds=None,
    ):
        del prompt, model, size, quality, background, output_format, output_compression, moderation, max_attempts
        self.started.set()
        self.release.wait()
        return ImageGenerationResult(created=123, images=[{"b64_json": base64.b64encode(b"png-bytes").decode("ascii")}])


class StaticImageLlm:
    def __init__(self, *, image_b64: str) -> None:
        self.image_b64 = image_b64

    def generate_image(
        self,
        *,
        prompt: str,
        model: str,
        size=None,
        quality=None,
        background=None,
        output_format=None,
        output_compression=None,
        moderation=None,
        max_attempts=None,
        timeout_seconds=None,
    ):
        del prompt, model, size, quality, background, output_format, output_compression, moderation, max_attempts
        return ImageGenerationResult(created=123, images=[{"b64_json": self.image_b64}])


class ReferenceAwareImageLlm:
    def __init__(self, *, image_b64: str) -> None:
        self.image_b64 = image_b64
        self.generate_calls: list[dict] = []
        self.edit_calls: list[dict] = []

    def generate_image(
        self,
        *,
        prompt: str,
        model: str,
        size=None,
        quality=None,
        background=None,
        output_format=None,
        output_compression=None,
        moderation=None,
        max_attempts=None,
        timeout_seconds=None,
    ):
        self.generate_calls.append(
            {
                "prompt": prompt,
                "model": model,
                "size": size,
                "quality": quality,
                "background": background,
                "output_format": output_format,
                "output_compression": output_compression,
                "moderation": moderation,
                "max_attempts": max_attempts,
                "timeout_seconds": timeout_seconds,
            }
        )
        return ImageGenerationResult(created=123, images=[{"b64_json": self.image_b64}])

    def edit_image(
        self,
        *,
        prompt: str,
        model: str,
        images: list[ImageAttachment],
        size=None,
        quality=None,
        background=None,
        output_format=None,
        output_compression=None,
        moderation=None,
        max_attempts=None,
        timeout_seconds=None,
    ):
        self.edit_calls.append(
            {
                "prompt": prompt,
                "model": model,
                "images": images,
                "size": size,
                "quality": quality,
                "background": background,
                "output_format": output_format,
                "output_compression": output_compression,
                "moderation": moderation,
                "max_attempts": max_attempts,
                "timeout_seconds": timeout_seconds,
            }
        )
        return ImageGenerationResult(created=123, images=[{"b64_json": self.image_b64}])


class FakeImageSearchClient:
    def __init__(
        self,
        *,
        image_results: list[ImageAttachment],
        image_results_by_query: dict[str, list[ImageAttachment]] | None = None,
    ) -> None:
        self.image_results = list(image_results)
        self.image_results_by_query = dict(image_results_by_query or {})
        self.queries: list[tuple[str, int]] = []

    def image_search(self, query: str, max_results: int = 3) -> list[ImageAttachment]:
        self.queries.append((query, max_results))
        return list(self.image_results_by_query.get(query, self.image_results))


class FakeImageReferencePlanner:
    def __init__(self, result: str) -> None:
        self.result = result
        self.prompts: list[list[str]] = []

    def generate_text(self, prompt_lines: list[str]) -> str:
        self.prompts.append(list(prompt_lines))
        return self.result


class FakeAdapterImage:
    def __init__(self, *, b64_json: str | None = None, url: str | None = None, output_format: str | None = None) -> None:
        self.b64_json = b64_json
        self.url = url
        self.output_format = output_format


class UrlImageLlm:
    def __init__(self, *, image_url: str) -> None:
        self.image_url = image_url
        self.http_client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    content=b"url-png-bytes",
                    headers={"content-type": "image/png"},
                )
            )
        )

    def generate_image(
        self,
        *,
        prompt: str,
        model: str,
        size=None,
        quality=None,
        background=None,
        output_format=None,
        output_compression=None,
        moderation=None,
        max_attempts=None,
        timeout_seconds=None,
    ):
        del prompt, model, size, quality, background, output_format, output_compression, moderation, max_attempts, timeout_seconds
        return ImageGenerationResult(created=123, images=[{"url": self.image_url, "output_format": "png"}])


class FailingImageLlm:
    def __init__(self) -> None:
        self.calls = 0

    def generate_image(
        self,
        *,
        prompt: str,
        model: str,
        size=None,
        quality=None,
        background=None,
        output_format=None,
        output_compression=None,
        moderation=None,
        max_attempts=None,
        timeout_seconds=None,
    ):
        del prompt, model, size, quality, background, output_format, output_compression, moderation, max_attempts, timeout_seconds
        self.calls += 1
        raise RuntimeError("boom")


class FailingThenSucceedingImageLlm:
    def __init__(self) -> None:
        self.calls = 0

    def generate_image(
        self,
        *,
        prompt: str,
        model: str,
        size=None,
        quality=None,
        background=None,
        output_format=None,
        output_compression=None,
        moderation=None,
        max_attempts=None,
        timeout_seconds=None,
    ):
        del prompt, model, size, quality, background, output_format, output_compression, moderation, max_attempts, timeout_seconds
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return ImageGenerationResult(created=123, images=[{"b64_json": base64.b64encode(b"png-bytes").decode("ascii")}])


class TimeoutImageLlm:
    def __init__(self) -> None:
        self.calls = []

    def generate_image(
        self,
        *,
        prompt: str,
        model: str,
        size=None,
        quality=None,
        background=None,
        output_format=None,
        output_compression=None,
        moderation=None,
        max_attempts=None,
        timeout_seconds=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "size": size,
                "quality": quality,
                "background": background,
                "output_format": output_format,
                "output_compression": output_compression,
                "moderation": moderation,
                "max_attempts": max_attempts,
                "timeout_seconds": timeout_seconds,
            }
        )
        raise ValueError("images generations request failed after retries") from httpx.RemoteProtocolError(
            "Server disconnected without sending a response."
        )


class FailureTextFailsOnceSender(FakeGroupImageSender):
    def __init__(self) -> None:
        super().__init__()
        self.failure_attempts = 0

    async def send_group_text(self, outbound) -> None:
        self.failure_attempts += 1
        if self.failure_attempts == 1:
            raise RuntimeError("notify failed")
        await super().send_group_text(outbound)


def make_request(
    message_id: str,
    *,
    prompt: str = "rainy convenience store",
    requester_user_id: int = 20001,
    reference_images: list[ImageAttachment] | None = None,
    web_search_query: str | None = None,
) -> GroupImageGenerationRequest:
    return GroupImageGenerationRequest(
        group_id=10001,
        trigger_message_id=message_id,
        prompt=prompt,
        requester_user_id=requester_user_id,
        reference_images=reference_images or [],
        web_search_query=web_search_query,
    )


@pytest.mark.asyncio
async def test_group_image_service_rejects_fourth_global_job(tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = BlockingImageLlm()
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        size="1024x1024",
        quality="low",
        background="",
        output_format="jpeg",
        output_compression=70,
        moderation="low",
        max_slots=3,
    )

    first = await service.enqueue(make_request("draw-1"))
    second = await service.enqueue(make_request("draw-2"))
    third = await service.enqueue(make_request("draw-3"))
    fourth = await service.enqueue(make_request("draw-4"))

    assert first.accepted is True
    assert first.queue_position == 1
    assert second.accepted is True
    assert second.queue_position == 2
    assert third.accepted is True
    assert third.queue_position == 3
    assert fourth.accepted is False
    assert fourth.reason == "queue_full"
    llm.release.set()
    await service.wait_for_idle()


@pytest.mark.asyncio
async def test_group_image_service_sends_generated_file(tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = StaticImageLlm(image_b64=base64.b64encode(b"png-bytes").decode("ascii"))
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        size="1024x1024",
        quality="low",
        background="",
        output_format="jpeg",
        output_compression=70,
        moderation="low",
        max_slots=3,
    )

    result = await service.enqueue(make_request("draw-1"))
    await service.wait_for_idle()

    assert result.accepted is True
    assert sender.image_calls[0]["group_id"] == 10001
    assert sender.image_calls[0]["image_file"].endswith(".jpeg")
    assert sender.text_calls == [
        {
            "group_id": 10001,
            "text": "[CQ:at,qq=20001] \u56fe\u597d\u4e86",
        }
    ]


@pytest.mark.asyncio
async def test_group_image_service_uses_edit_path_when_reference_images_are_present(tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = ReferenceAwareImageLlm(image_b64=base64.b64encode(b"png-bytes").decode("ascii"))
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        size="1024x1024",
        quality="low",
        background="",
        output_format="jpeg",
        output_compression=70,
        moderation="low",
        max_slots=3,
    )

    result = await service.enqueue(
        make_request(
            "draw-edit-1",
            prompt="turn this into cyberpunk neon",
            reference_images=[ImageAttachment(url="https://img.example.test/source.png", file_id="source.png")],
        )
    )
    await service.wait_for_idle()

    assert result.accepted is True
    assert llm.generate_calls == []
    assert len(llm.edit_calls) == 1
    assert llm.edit_calls[0]["prompt"] == "turn this into cyberpunk neon"
    assert [image.url for image in llm.edit_calls[0]["images"]] == ["https://img.example.test/source.png"]
    assert len(sender.image_calls) == 1
    assert sender.text_calls[-1]["text"] == "[CQ:at,qq=20001] \u56fe\u597d\u4e86"


@pytest.mark.asyncio
async def test_group_image_service_uses_max_landscape_size_when_prompt_mentions_landscape(tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = ReferenceAwareImageLlm(image_b64=base64.b64encode(b"png-bytes").decode("ascii"))
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        size="1536x1024",
        quality="high",
        background="",
        output_format="png",
        moderation="low",
        max_slots=3,
    )

    result = await service.enqueue(make_request("draw-landscape-1", prompt="请画一张横图，机甲站在城市天台"))
    await service.wait_for_idle()

    assert result.accepted is True
    assert llm.generate_calls[0]["size"] == "1536x1024"
    assert llm.generate_calls[0]["quality"] == "high"


@pytest.mark.asyncio
async def test_group_image_service_uses_max_portrait_size_when_prompt_mentions_portrait(tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = ReferenceAwareImageLlm(image_b64=base64.b64encode(b"png-bytes").decode("ascii"))
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        size="auto",
        quality="high",
        background="",
        output_format="png",
        moderation="low",
        max_slots=3,
    )

    result = await service.enqueue(
        make_request(
            "draw-portrait-1",
            prompt="请按竖图出图，保留人物动作",
            reference_images=[ImageAttachment(url="https://img.example.test/source.png", file_id="source.png")],
        )
    )
    await service.wait_for_idle()

    assert result.accepted is True
    assert llm.generate_calls == []
    assert llm.edit_calls[0]["size"] == "1024x1536"
    assert llm.edit_calls[0]["quality"] == "high"


@pytest.mark.asyncio
async def test_group_image_service_uses_auto_size_when_prompt_does_not_specify_orientation(tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = ReferenceAwareImageLlm(image_b64=base64.b64encode(b"png-bytes").decode("ascii"))
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        size="auto",
        quality="high",
        background="",
        output_format="png",
        moderation="low",
        max_slots=3,
    )

    result = await service.enqueue(make_request("draw-auto-size-1", prompt="请画夜晚下雨的便利店门口"))
    await service.wait_for_idle()

    assert result.accepted is True
    assert llm.generate_calls[0]["size"] == "auto"


@pytest.mark.asyncio
async def test_group_image_service_combines_searched_character_refs_with_existing_reference_images(tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = ReferenceAwareImageLlm(image_b64=base64.b64encode(b"png-bytes").decode("ascii"))
    search_client = FakeImageSearchClient(
        image_results=[
            ImageAttachment(url="https://img.example.test/kaguya-a.png", file_id="kaguya-a.png"),
            ImageAttachment(url="https://img.example.test/kaguya-b.png", file_id="kaguya-b.png"),
        ]
    )
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        web_search_client=search_client,
        size="1024x1024",
        quality="low",
        background="",
        output_format="jpeg",
        output_compression=70,
        moderation="low",
        max_slots=3,
    )

    result = await service.enqueue(
        make_request(
            "draw-auto-ref-1",
            prompt="保留前图构图，只替换人物出图",
            reference_images=[ImageAttachment(url="https://img.example.test/layout.png", file_id="layout.png")],
            web_search_query="超时空辉夜姬 两个女主",
        )
    )
    await service.wait_for_idle()

    assert result.accepted is True
    assert search_client.queries == [("超时空辉夜姬 两个女主", 3)]
    assert llm.generate_calls == []
    assert len(llm.edit_calls) == 1
    assert [image.url for image in llm.edit_calls[0]["images"]] == [
        "https://img.example.test/layout.png",
        "https://img.example.test/kaguya-a.png",
        "https://img.example.test/kaguya-b.png",
    ]
    assert len(sender.image_calls) == 1
    assert sender.text_calls[-1]["text"] == "[CQ:at,qq=20001] \u56fe\u597d\u4e86"


@pytest.mark.asyncio
async def test_group_image_service_plans_independent_reference_queries(tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = ReferenceAwareImageLlm(image_b64=base64.b64encode(b"png-bytes").decode("ascii"))
    search_client = FakeImageSearchClient(
        image_results=[],
        image_results_by_query={
            "真绯瑠 官方角色立绘": [
                ImageAttachment(url="https://img.example.test/mahirus.png", file_id="mahirus.png")
            ],
            "弥希 官方角色立绘": [
                ImageAttachment(url="https://img.example.test/miki.png", file_id="miki.png")
            ],
        },
    )
    planner = FakeImageReferencePlanner(
        '{"should_search": true, "references": ['
        '{"subject": "真绯瑠", "queries": ["真绯瑠 官方角色立绘"]},'
        '{"subject": "弥希", "queries": ["弥希 官方角色立绘"]}'
        ']}'
    )
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-5.6-sol",
        web_search_client=search_client,
        image_reference_planner_client=planner,
    )

    result = await service.enqueue(make_request("draw-planned-1", web_search_query="真绯瑠和弥希"))
    await service.wait_for_idle()

    assert result.accepted is True
    assert search_client.queries == [
        ("真绯瑠 官方角色立绘", 2),
        ("弥希 官方角色立绘", 2),
    ]
    assert [image.reference_subject for image in llm.edit_calls[0]["images"]] == ["真绯瑠", "弥希"]
    assert len(planner.prompts) == 1
    assert len(llm.edit_calls) == 1


@pytest.mark.asyncio
async def test_group_image_service_accepts_legacy_flat_reference_queries(tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = ReferenceAwareImageLlm(image_b64=base64.b64encode(b"png-bytes").decode("ascii"))
    search_client = FakeImageSearchClient(
        image_results=[ImageAttachment(url="https://img.example.test/legacy.png", file_id="legacy.png")]
    )
    planner = FakeImageReferencePlanner(
        '{"should_search": true, "queries": ["旧格式 角色参考图"]}'
    )
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        web_search_client=search_client,
        image_reference_planner_client=planner,
    )

    result = await service.enqueue(make_request("draw-legacy-planned", web_search_query="旧格式角色"))
    await service.wait_for_idle()

    assert result.accepted is True
    assert search_client.queries == [("旧格式 角色参考图", 2)]
    assert llm.edit_calls[0]["images"][0].reference_subject is None


@pytest.mark.asyncio
async def test_group_image_service_falls_back_for_malformed_search_decision(tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = ReferenceAwareImageLlm(image_b64=base64.b64encode(b"png-bytes").decode("ascii"))
    search_client = FakeImageSearchClient(
        image_results=[ImageAttachment(url="https://img.example.test/fallback.png", file_id="fallback.png")]
    )
    planner = FakeImageReferencePlanner(
        '{"should_search": "yes", "references": [{"subject": "角色", "queries": ["角色 官方图"]}]}'
    )
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        web_search_client=search_client,
        image_reference_planner_client=planner,
    )

    result = await service.enqueue(make_request("draw-malformed-plan", web_search_query="明确搜索对象"))
    await service.wait_for_idle()

    assert result.accepted is True
    assert search_client.queries == [("明确搜索对象", 2)]
    assert llm.edit_calls[0]["images"][0].reference_subject is None


def test_group_image_service_round_trips_reference_subject_in_job_payload(tmp_path) -> None:
    service = GroupImageGenerationService(
        llm_client=StaticImageLlm(image_b64=base64.b64encode(b"png-bytes").decode("ascii")),
        sender=FakeGroupImageSender(),
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
    )
    request = make_request(
        "draw-persisted-reference",
        reference_images=[
            ImageAttachment(
                url="https://img.example.test/mahiru.png",
                file_id="mahiru.png",
                fallback_url="https://thumb.example.test/mahiru.png",
                reference_subject="真绯瑠",
            )
        ],
    )

    restored = service._deserialize_request(service._serialize_request(request))

    assert restored.reference_images == request.reference_images


@pytest.mark.asyncio
async def test_group_image_service_model_can_decline_search_for_ordinary_generation(tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = ReferenceAwareImageLlm(image_b64=base64.b64encode(b"png-bytes").decode("ascii"))
    search_client = FakeImageSearchClient(
        image_results=[ImageAttachment(url="https://img.example.test/should-not-use.png", file_id="no.png")]
    )
    planner = FakeImageReferencePlanner('{"should_search": false, "queries": []}')
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-5.6-sol",
        web_search_client=search_client,
        image_reference_planner_client=planner,
    )

    result = await service.enqueue(make_request("draw-no-search", prompt="生成一张未来城市夜景"))
    await service.wait_for_idle()

    assert result.accepted is True
    assert search_client.queries == []
    assert len(llm.generate_calls) == 1
    assert len(planner.prompts) == 1


@pytest.mark.asyncio
async def test_group_image_service_accepts_base64_without_padding(tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = StaticImageLlm(image_b64="YWI")
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        size="1024x1024",
        quality="low",
        background="",
        output_format="jpeg",
        output_compression=70,
        moderation="low",
        max_slots=3,
    )

    result = await service.enqueue(make_request("draw-missing-padding"))
    await service.wait_for_idle()

    assert result.accepted is True
    assert len(sender.image_calls) == 1
    written = tmp_path / "generated_images" / Path(sender.image_calls[0]["image_file"]).name
    assert written.read_bytes() == b"ab"


@pytest.mark.asyncio
async def test_group_image_service_accepts_typed_adapter_image_objects_without_extra_download(tmp_path) -> None:
    sender = FakeGroupImageSender()

    class TypedResultLlm:
        def generate_image(
            self,
            *,
            prompt: str,
            model: str,
            size=None,
            quality=None,
            background=None,
            output_format=None,
            output_compression=None,
            moderation=None,
            max_attempts=None,
            timeout_seconds=None,
        ):
            del prompt, model, size, quality, background, output_format, output_compression, moderation, max_attempts, timeout_seconds
            return type(
                "TypedImageResult",
                (),
                {
                    "images": [
                        FakeAdapterImage(
                            b64_json=base64.b64encode(b"typed-png-bytes").decode("ascii"),
                            output_format="png",
                        )
                    ]
                },
            )()

    service = GroupImageGenerationService(
        llm_client=TypedResultLlm(),
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        size="1024x1024",
        quality="low",
        background="",
        output_format="png",
        max_slots=3,
    )

    result = await service.enqueue(make_request("draw-typed-image-1"))
    await service.wait_for_idle()

    assert result.accepted is True
    assert len(sender.image_calls) == 1
    written = tmp_path / "generated_images" / Path(sender.image_calls[0]["image_file"]).name
    assert written.read_bytes() == b"typed-png-bytes"


@pytest.mark.asyncio
async def test_group_image_service_downloads_url_artifacts_when_provider_returns_url(tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = UrlImageLlm(image_url="https://img.example.test/generated.png")
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        size="1024x1024",
        quality="low",
        background="",
        output_format="png",
        max_slots=3,
    )

    result = await service.enqueue(make_request("draw-url-image-1"))
    await service.wait_for_idle()

    assert result.accepted is True
    assert len(sender.image_calls) == 1
    written = tmp_path / "generated_images" / Path(sender.image_calls[0]["image_file"]).name
    assert written.read_bytes() == b"url-png-bytes"


@pytest.mark.asyncio
async def test_group_image_service_decodes_large_data_url_artifacts_without_http_download(tmp_path) -> None:
    sender = FakeGroupImageSender()
    buffer = io.BytesIO()
    Image.new("RGB", (256, 128), color=(55, 105, 190)).save(buffer, format="PNG", compress_level=0)
    image_bytes = buffer.getvalue()
    assert len(image_bytes) > 8192
    encoded = base64.b64encode(image_bytes).decode("ascii")
    image_url = "data:image/png;base64," + encoded
    assert len(image_url) > 65536  # exceeds httpx's URL-length budget

    class DataUrlImageLlm(UrlImageLlm):
        def __init__(self) -> None:
            super().__init__(image_url=image_url)
            self.http_client.close()

            def reject_download(request):
                raise AssertionError("data URI was incorrectly sent to httpx")

            self.http_client = httpx.Client(transport=httpx.MockTransport(reject_download))

    service = GroupImageGenerationService(
        llm_client=DataUrlImageLlm(),
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        size="1024x1024",
        quality="low",
        background="",
        output_format="png",
        max_slots=3,
    )

    result = await service.enqueue(make_request("draw-data-url-image-1"))
    await service.wait_for_idle()

    assert result.accepted is True
    assert len(sender.image_calls) == 1
    written = tmp_path / "generated_images" / Path(sender.image_calls[0]["image_file"]).name
    assert written.read_bytes() == image_bytes
    with Image.open(written) as actual:
        actual.load()
        assert actual.format == "PNG"
        assert actual.size == (256, 128)
    service.http_client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_kind", [
    "not_image", "not_base64", "malformed_base64", "empty", "bad_flag", "non_ascii",
    "encoded_oversize", "decoded_oversize",
])
async def test_group_image_service_rejects_invalid_data_url_without_download_or_image_send(tmp_path, invalid_kind) -> None:
    values = {
        "not_image": "data:text/plain;base64,YWI=",
        "not_base64": "data:image/png,abc",
        "malformed_base64": "data:image/png;base64,YW!I=",
        "empty": "data:image/png;base64,",
        "bad_flag": "data:image/png;base64garbage,YWI=",
        "non_ascii": "data:image/png;base64,%FF",
    }
    if invalid_kind == "encoded_oversize":
        image_url = "data:image/png;base64," + "A" * (12 * 1024 * 1024 + 1)
    elif invalid_kind == "decoded_oversize":
        image_url = "data:image/png;base64," + base64.b64encode(b"x" * (8 * 1024 * 1024 + 1)).decode("ascii")
    else:
        image_url = values[invalid_kind]
    requests = []

    def reject_download(request):
        requests.append(request)
        raise AssertionError("data URI must never be downloaded")

    llm = UrlImageLlm(image_url=image_url)
    llm.http_client.close()
    llm.http_client = httpx.Client(transport=httpx.MockTransport(reject_download))
    sender = FakeGroupImageSender()
    service = GroupImageGenerationService(llm_client=llm, sender=sender,
        output_dir=tmp_path / "generated_images", model="test-image", failure_reply_text="invalid image result")

    result = await service.enqueue(make_request("draw-invalid-data-url"))
    await service.wait_for_idle()

    assert result.accepted is True
    assert requests == []
    assert sender.image_calls == []
    assert sender.text_calls == [{"group_id": 10001, "text": "[CQ:at,qq=20001] invalid image result"}]
    assert not list(service.output_dir.glob("*"))
    service.http_client.close()


@pytest.mark.asyncio
async def test_group_image_service_failure_releases_slot_and_notifies_group(tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = FailingImageLlm()
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        size="1024x1024",
        quality="",
        background="",
        output_format="png",
        max_slots=1,
        failure_reply_text="\u6ca1\u753b\u51fa\u6765",
    )

    first = await service.enqueue(make_request("draw-1"))
    await service.wait_for_idle()
    second = await service.enqueue(make_request("draw-2"))

    assert first.accepted is True
    assert sender.text_calls == [
        {
            "group_id": 10001,
            "text": "[CQ:at,qq=20001] \u6ca1\u753b\u51fa\u6765",
        }
    ]
    assert second.accepted is True
    await service.wait_for_idle()


@pytest.mark.asyncio
async def test_group_image_service_continues_after_failure_notification_send_error(tmp_path) -> None:
    sender = FailureTextFailsOnceSender()
    llm = FailingThenSucceedingImageLlm()
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        size="1024x1024",
        quality="low",
        background="",
        output_format="jpeg",
        output_compression=70,
        moderation="low",
        max_slots=3,
    )

    first = await service.enqueue(make_request("draw-1"))
    second = await service.enqueue(make_request("draw-2"))
    await asyncio.wait_for(service.wait_for_idle(), timeout=1.0)

    assert first.accepted is True
    assert second.accepted is True
    assert len(sender.image_calls) == 1


@pytest.mark.asyncio
async def test_group_image_service_start_recovers_running_job_from_persistent_queue(sqlite_engine, tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = StaticImageLlm(image_b64=base64.b64encode(b"png-bytes").decode("ascii"))
    service = GroupImageGenerationService(
        engine=sqlite_engine,
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        size="1024x1024",
        quality="low",
        background="",
        output_format="jpeg",
        output_compression=70,
        moderation="low",
        max_slots=3,
    )

    persist_image_source(sqlite_engine, make_request("draw-recover-1"))
    with session_scope(sqlite_engine) as session:
        JobRepository(session).add_job(
            job_type=service.job_type,
            payload_json=service._serialize_request(make_request("draw-recover-1")),
            run_at=datetime.now(UTC),
            status="running",
        )

    await service.start()
    await service.wait_for_idle()

    assert len(sender.image_calls) == 1
    assert sender.image_calls[0]["group_id"] == 10001
    assert sender.text_calls[-1]["text"] == "[CQ:at,qq=20001] 图好了"
    with session_scope(sqlite_engine) as session:
        completed = session.execute(
            text("select count(*) from jobs where job_type = :job_type and status = 'completed'"),
            {"job_type": service.job_type},
        ).scalar_one()
    assert completed == 1


@pytest.mark.asyncio
async def test_group_image_service_retries_transport_timeout_and_suggests_simpler_prompt(tmp_path) -> None:
    sender = FakeGroupImageSender()
    llm = TimeoutImageLlm()
    service = GroupImageGenerationService(
        llm_client=llm,
        sender=sender,
        output_dir=tmp_path / "generated_images",
        model="gpt-image-2",
        size="1024x1024",
        quality="low",
        background="",
        output_format="jpeg",
        output_compression=70,
        moderation="low",
        max_slots=3,
    )

    accepted = await service.enqueue(make_request("draw-timeout-1", prompt="two anime characters"))
    await service.wait_for_idle()

    assert accepted.accepted is True
    assert llm.calls[0]["quality"] == "low"
    assert llm.calls[0]["output_format"] == "jpeg"
    assert llm.calls[0]["output_compression"] == 70
    assert llm.calls[0]["moderation"] == "low"
    assert llm.calls[0]["max_attempts"] == 5
    assert llm.calls[0]["timeout_seconds"] is None
    assert "短" in sender.text_calls[-1]["text"]


def persist_image_source(engine, request, *, raw_json=None) -> None:
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(group_id=request.group_id, group_name="synthetic", enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=request.requester_user_id, nickname="synthetic", group_card="")
        MessageRepository(session).add_group_message(
            platform_msg_id=request.trigger_message_id, group_id=request.group_id,
            user_id=request.requester_user_id, timestamp=datetime.now(UTC),
            plain_text=request.prompt, raw_json=raw_json or {}, msg_type="text",
            reply_to_msg_id=None, mentioned_bot=True,
        )


def persistent_image_service(engine, tmp_path, sender, *, llm=None):
    return GroupImageGenerationService(
        engine=engine, llm_client=llm or StaticImageLlm(image_b64=base64.b64encode(b"png").decode()),
        sender=sender, output_dir=tmp_path / "images", model="test-image",
    )


class ImageReceiptGateway:
    def __init__(self, missing_receipt_at=None):
        self.calls = []
        self.missing_receipt_at = missing_receipt_at

    async def call_api(self, action, params):
        self.calls.append((action, params))
        data = {} if len(self.calls) == self.missing_receipt_at else {"message_id": f"image-receipt-{len(self.calls)}"}
        return {"status": "ok", "retcode": 0, "data": data}


@pytest.mark.asyncio
async def test_image_jobs_outlive_turn_scope_and_keep_each_group(sqlite_engine, tmp_path):
    gateway = ImageReceiptGateway()
    service = persistent_image_service(sqlite_engine, tmp_path, Sender(gateway))
    first = replace(make_request("independent-first"), group_id=900000001, requester_user_id=900000101)
    second = replace(first, group_id=900000002, trigger_message_id="independent-second")
    persist_image_source(sqlite_engine, first)
    persist_image_source(sqlite_engine, second)
    obsolete_turn = GroupDispatchScope(first.group_id, lambda: False, lambda: pytest.fail("old turn dispatched"))
    with group_dispatch_scope(obsolete_turn):
        assert (await service.enqueue(first)).accepted
        assert (await service.enqueue(second)).accepted
    await asyncio.wait_for(service.wait_for_idle(), 2)
    assert [params["group_id"] for _, params in gateway.calls] == [first.group_id, first.group_id, second.group_id, second.group_id]
    with session_scope(sqlite_engine) as session:
        completed = JobRepository(session).list_jobs(job_type=service.job_type, statuses=["completed"])
        assert len(completed) == 2
        assert completed[0].payload_json["image_receipt_ids"] == ["image-receipt-1"]
        assert completed[0].payload_json["notice_receipt_ids"] == ["image-receipt-2"]
    assert not obsolete_turn.dispatched


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_receipt_at", [1, 2])
async def test_unknown_image_or_notice_delivery_is_never_repeated(sqlite_engine, tmp_path, missing_receipt_at):
    gateway = ImageReceiptGateway(missing_receipt_at)
    service = persistent_image_service(sqlite_engine, tmp_path, Sender(gateway))
    request = replace(make_request("unknown-image"), group_id=900000001, requester_user_id=900000101)
    persist_image_source(sqlite_engine, request)
    assert (await service.enqueue(request)).accepted
    await asyncio.wait_for(service.wait_for_idle(), 2)
    assert len(gateway.calls) == missing_receipt_at
    with session_scope(sqlite_engine) as session:
        failed = JobRepository(session).list_jobs(job_type=service.job_type, statuses=["failed"])
        assert len(failed) == 1
        assert failed[0].payload_json["delivery_state"] == "uncertain"
    await service.start()
    await asyncio.wait_for(service.wait_for_idle(), 2)
    assert len(gateway.calls) == missing_receipt_at


@pytest.mark.asyncio
async def test_restart_does_not_repeat_dispatching_image_job(sqlite_engine, tmp_path):
    sender = FakeGroupImageSender()
    llm = ReferenceAwareImageLlm(image_b64=base64.b64encode(b"png").decode())
    service = persistent_image_service(sqlite_engine, tmp_path, sender, llm=llm)
    request = make_request("crash-during-image-dispatch")
    payload = service._serialize_request(request)
    payload["delivery_state"] = "dispatching"
    with session_scope(sqlite_engine) as session:
        JobRepository(session).add_job(job_type=service.job_type, payload_json=payload, run_at=datetime.now(UTC), status="running")
    await service.start()
    await service.wait_for_idle()
    assert llm.generate_calls == []
    assert sender.image_calls == sender.text_calls == []
    with session_scope(sqlite_engine) as session:
        job = JobRepository(session).list_jobs(job_type=service.job_type, statuses=["failed"])[0]
        assert job.payload_json["delivery_state"] == "uncertain"


@pytest.mark.asyncio
async def test_image_request_sources_must_match_group_author_and_live_source(sqlite_engine, tmp_path):
    service = persistent_image_service(sqlite_engine, tmp_path, FakeGroupImageSender())
    request = replace(make_request("source-scope"), group_id=900000001, requester_user_id=900000101)
    assert (await service.enqueue(request)).reason == "source_unusable"
    persist_image_source(sqlite_engine, request)
    assert (await service.enqueue(replace(request, group_id=900000002))).reason == "source_unusable"
    assert (await service.enqueue(replace(request, requester_user_id=900000102))).reason == "source_unusable"
    with session_scope(sqlite_engine) as session:
        source = MessageRepository(session).get_by_platform_msg_id(request.trigger_message_id)
        source.raw_json = {"delivery_state": "deleted"}
    assert (await service.enqueue(request)).reason == "source_unusable"


@pytest.mark.asyncio
async def test_recall_while_image_generates_prevents_all_qq_writes(sqlite_engine, tmp_path):
    sender = FakeGroupImageSender()
    llm = BlockingImageLlm()
    service = persistent_image_service(sqlite_engine, tmp_path, sender, llm=llm)
    request = make_request("recall-while-generating")
    persist_image_source(sqlite_engine, request)
    assert (await service.enqueue(request)).accepted
    try:
        for _ in range(200):
            if llm.started.is_set():
                break
            await asyncio.sleep(.005)
        assert llm.started.is_set()
        with session_scope(sqlite_engine) as session:
            source = MessageRepository(session).get_by_platform_msg_id(request.trigger_message_id)
            source.raw_json = {"delivery_state": "deleted"}
    finally:
        llm.release.set()
    await asyncio.wait_for(service.wait_for_idle(), 2)
    assert sender.image_calls == sender.text_calls == []


@pytest.mark.asyncio
async def test_stop_during_generation_revokes_late_image_and_notice(sqlite_engine, tmp_path):
    sender = FakeGroupImageSender()
    llm = BlockingImageLlm()
    service = persistent_image_service(sqlite_engine, tmp_path, sender, llm=llm)
    request = make_request("stop-while-generating")
    persist_image_source(sqlite_engine, request)
    assert (await service.enqueue(request)).accepted
    try:
        for _ in range(200):
            if llm.started.is_set():
                break
            await asyncio.sleep(.005)
        assert llm.started.is_set()
        await asyncio.wait_for(service.stop(), 1)
        assert (await service.enqueue(request)).reason == "service_stopping"
    finally:
        llm.release.set()
    await asyncio.sleep(.03)
    assert sender.image_calls == sender.text_calls == []


@pytest.mark.asyncio
async def test_image_services_reject_other_surface_requests(tmp_path):
    kwargs = dict(llm_client=StaticImageLlm(image_b64=base64.b64encode(b"png").decode()), sender=FakeGroupImageSender(), output_dir=tmp_path, model="test-image")
    group_service = GroupImageGenerationService(**kwargs)
    private_service = PrivateImageGenerationService(**kwargs)
    private_request = PrivateImageGenerationRequest(user_id=900000101, trigger_message_id="private-source", prompt="test")
    with pytest.raises(ValueError, match="chat surface"):
        await group_service.enqueue(private_request)
    with pytest.raises(ValueError, match="chat surface"):
        await private_service.enqueue(make_request("group-source"))
    group_payload = group_service._serialize_request(make_request("group-source"))
    private_payload = private_service._serialize_request(private_request)
    with pytest.raises(ValueError, match="private"):
        group_service._deserialize_request({**group_payload, **private_payload})
    with pytest.raises(ValueError, match="group"):
        private_service._deserialize_request({**private_payload, **group_payload})
