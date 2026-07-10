from __future__ import annotations

import asyncio
import base64
import threading
from datetime import UTC, datetime
from pathlib import Path
import httpx

import pytest
from sqlalchemy import text

from app.core.group_image_generation import (
    GroupImageGenerationRequest,
    GroupImageGenerationService,
)
from app.core.message_content import ImageAttachment
from app.providers.llm_client import ImageGenerationResult
from app.storage.db import session_scope
from app.storage.repositories import JobRepository


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
    def __init__(self, *, image_results: list[ImageAttachment]) -> None:
        self.image_results = list(image_results)
        self.queries: list[tuple[str, int]] = []

    def image_search(self, query: str, max_results: int = 3) -> list[ImageAttachment]:
        self.queries.append((query, max_results))
        return list(self.image_results)


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
async def test_group_image_service_uses_4k_landscape_size_when_prompt_mentions_landscape(tmp_path) -> None:
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
    assert llm.generate_calls[0]["size"] == "3840x2160"


@pytest.mark.asyncio
async def test_group_image_service_uses_4k_portrait_size_when_prompt_mentions_portrait(tmp_path) -> None:
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
    assert llm.edit_calls[0]["size"] == "2160x3840"


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
