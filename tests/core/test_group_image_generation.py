from __future__ import annotations

import asyncio
import base64
import threading
from pathlib import Path
import httpx

import pytest

from app.core.group_image_generation import (
    GroupImageGenerationRequest,
    GroupImageGenerationService,
)
from app.providers.llm_client import ImageGenerationResult


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
) -> GroupImageGenerationRequest:
    return GroupImageGenerationRequest(
        group_id=10001,
        trigger_message_id=message_id,
        prompt=prompt,
        requester_user_id=requester_user_id,
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
async def test_group_image_service_fast_fails_transport_timeout_and_suggests_simpler_prompt(tmp_path) -> None:
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
    assert llm.calls[0]["max_attempts"] == 1
    assert llm.calls[0]["timeout_seconds"] is None
    assert "短" in sender.text_calls[-1]["text"]
