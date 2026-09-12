"""Chat-only private service regression tests.

These cover the surface that survived the 2026-09-11 private-chat cleanup:
daily persona replies, private drawings with their follow-up windows, session
commands and outbound reply de-duplication.
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from types import SimpleNamespace

import pytest
from sqlalchemy import text

import app.private_chat.service as private_chat_module
from app.adapters.onebot_models import PrivateMessageEvent
from app.adapters.sender import QQMessageDeliveryUncertainError
from app.core.group_image_generation import PrivateImageGenerationRequest
from app.core.message_content import ImageAttachment
from app.private_chat.service import PrivateChatService
from app.storage.db import session_scope
from app.storage.repositories import (
    DevSessionRepository,
    DevTaskRepository,
    JobRepository,
    MessageRepository,
    UserRepository,
)


class FakeSender:
    def __init__(self) -> None:
        self.private_sent = []
        self.private_image_sent = []

    async def send_private_text(self, outbound) -> None:
        self.private_sent.append(outbound)

    async def send_private_image(self, *, user_id: int, image_file: str) -> None:
        self.private_image_sent.append({"user_id": user_id, "image_file": image_file})


class UncertainPrivateSender(FakeSender):
    async def send_private_text(self, outbound) -> None:
        self.private_sent.append(outbound)
        raise QQMessageDeliveryUncertainError("waitForSelfEcho timeout")


class FakeGateway:
    def __init__(self, *, get_msg_responses: dict[str, dict] | None = None) -> None:
        self.get_msg_responses = dict(get_msg_responses or {})
        self.calls: list[tuple[str, dict]] = []

    async def call_api(self, action: str, params: dict) -> dict:
        self.calls.append((action, dict(params)))
        if action != "get_msg":
            return {"status": "ok", "retcode": 0, "data": {"message_id": "sent-1"}}
        message_id = str(params.get("message_id"))
        payload = self.get_msg_responses.get(message_id)
        if payload is None:
            return {"status": "failed", "retcode": 1200, "data": None}
        return {"status": "ok", "retcode": 0, "data": payload}


class GatewayBackedSender(FakeSender):
    def __init__(self, *, gateway: FakeGateway) -> None:
        super().__init__()
        self.gateway = gateway


class FakeLlmClient:
    def __init__(self, reply_text: str = "daily reply") -> None:
        self.reply_text = reply_text
        self.prompts: list[list[str]] = []
        self.images_calls: list[list[ImageAttachment] | None] = []
        self.conversation_keys: list[str | None] = []

    def generate_text(self, prompt_lines, *, images=None, conversation_key=None, temperature=None):
        self.prompts.append(list(prompt_lines))
        self.images_calls.append(None if images is None else list(images))
        self.conversation_keys.append(conversation_key)
        return self.reply_text


class BuiltinSearchLlmClient(FakeLlmClient):
    """Chat client that can carry the provider's built-in ``web_search`` tool."""

    def __init__(self, reply_text: str = "builtin reply", *, web_search_model: str = "") -> None:
        super().__init__(reply_text=reply_text)
        self.supports_selective_web_search = True
        self.supports_forced_web_search = True
        self.builtin_web_search = True
        self.web_search_model = web_search_model
        self.generate_kwargs: list[dict] = []

    def generate_text(
        self,
        prompt_lines,
        *,
        images=None,
        conversation_key=None,
        temperature=None,
        **kwargs,
    ):
        self.generate_kwargs.append(dict(kwargs))
        return super().generate_text(
            prompt_lines,
            images=images,
            conversation_key=conversation_key,
            temperature=temperature,
        )


class TurnLabeledLlmClient(FakeLlmClient):
    """Replies ``reply-<n>`` to the owner message ``owner-message-<n>``.

    The private context window is asserted on message identity, so the fake
    echoes the turn number back instead of returning one fixed reply.
    """

    def generate_text(
        self,
        prompt_lines,
        *,
        images=None,
        conversation_key=None,
        temperature=None,
    ):
        current_message = next(
            (
                line.split("Current owner message: ", maxsplit=1)[1]
                for line in reversed(list(prompt_lines))
                if line.startswith("Current owner message: ")
            ),
            "",
        )
        if "owner-message-" in current_message:
            self.reply_text = f"reply-{current_message.split('owner-message-', maxsplit=1)[1]}"
        return super().generate_text(
            prompt_lines,
            images=images,
            conversation_key=conversation_key,
            temperature=temperature,
        )


class FakeSearchClient:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []
        self.page_reads: list[tuple[list[str], str | None, int]] = []

    def search(self, query: str, max_results: int = 3):
        self.queries.append((query, max_results))
        return [
            SimpleNamespace(
                title="Official site",
                snippet="Episode 1 aired and discussion focused on pacing.",
                source="https://official.example",
                date="2026-05-01",
            )
        ]

    def read_pages(self, results, *, query: str | None = None, max_pages: int = 3, skim_limit: int = 6):
        del skim_limit
        self.page_reads.append(([result.source for result in results], query, max_pages))
        return [
            SimpleNamespace(
                title="Detailed review",
                url="https://official.example/review",
                content="Episode 1 introduces the cast. Episode 2 deepens the conflict.",
            )
        ]


class FakeImageReferenceSearchClient:
    """Reference-image search client: only the private drawing path uses it."""

    def __init__(self, results: list[ImageAttachment] | None = None) -> None:
        self.results = list(results or [])
        self.image_queries: list[tuple[str, int]] = []

    def image_search(self, query: str, max_results: int = 3):
        self.image_queries.append((query, max_results))
        return list(self.results)


class FakeReferencePlanner:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[list[str]] = []

    def generate_text(self, prompt_lines, **_kwargs):
        self.prompts.append(list(prompt_lines))
        return self.response


class FakeImageGenerationLlm:
    def __init__(self) -> None:
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
        return SimpleNamespace(images=[{"b64_json": "cHJpdmF0ZS1pbWFnZS1ieXRlcw=="}])

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
                "images": list(images),
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
        return SimpleNamespace(images=[{"b64_json": "cHJpdmF0ZS1pbWFnZS1ieXRlcw=="}])


def make_private_event(
    *,
    message_id: str,
    user_id: int,
    text: str,
    raw_payload: dict | None = None,
    msg_type: str = "text",
    reply_to_msg_id: str | None = None,
    images: list[ImageAttachment] | None = None,
) -> PrivateMessageEvent:
    return PrivateMessageEvent(
        platform_msg_id=message_id,
        user_id=user_id,
        nickname="owner",
        plain_text=text,
        raw_payload=raw_payload or {},
        timestamp=datetime(2026, 5, 10, 12, 0, tzinfo=UTC),
        msg_type=msg_type,
        reply_to_msg_id=reply_to_msg_id,
        images=list(images or []),
    )


def build_service(sqlite_engine, tmp_path, **kwargs) -> PrivateChatService:
    return PrivateChatService(
        engine=sqlite_engine,
        data_dir=tmp_path / "data",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_owner_daily_chat_replies_inline_with_daily_prompt(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="This private chat is one continuous daily session.")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        assistant_name="比企谷小町",
        persona={
            "name": "比企谷小町",
            "identity": "A fixed AI persona modeled after Hikigaya Komachi.",
        },
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-1", user_id=10001, text="今天聊点什么")
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == [
        "This private chat is one continuous daily session."
    ]
    prompt = "\n".join(llm_client.prompts[0])
    assert "Current private daily session summary:" in prompt
    assert "Recent private daily turns:" in prompt
    assert "比企谷小町" in prompt
    # The private prompt runs the shared group work-style lines; only the
    # opening line names a direct chat instead of a group.
    assert "Talk like a real person chatting on QQ." in prompt
    assert "Do not use Markdown, headings, bullet lists, numbered lists" in prompt
    assert "Do not include URLs, website addresses, Markdown links" in prompt
    # The default persona keeps the mesugaki voice; ``chat_voice: clingy``
    # swaps only the edge lines (see the clingy test below).
    assert "mesugaki" in prompt
    # Memory retrieval and persona imitation stay out of the private prompt.
    assert "相关话题下他的原话示例" not in prompt
    assert "群历史" not in prompt
    # The Codex project channel is gone from the daily prompt.
    assert "local Xiaomachi repository" not in prompt
    assert "Relevant repository snippets:" not in prompt
    assert "启动管理员模式" not in prompt
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert len(completed) == 1
    assert completed[0].intent_type == "private_chat"
    assert completed[0].result_text == "This private chat is one continuous daily session."
    assert queued == []
    with sqlite_engine.connect() as connection:
        session_modes = [
            row[0]
            for row in connection.execute(text("select session_mode from dev_sessions order by id asc"))
        ]
    assert session_modes == ["daily"]


@pytest.mark.asyncio
async def test_private_context_window_keeps_the_last_twenty_messages(
    sqlite_engine, tmp_path
) -> None:
    """The daily context window counts both sides: 20 messages = ten exchanges."""

    sender = FakeSender()
    llm_client = TurnLabeledLlmClient()
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    for index in range(1, 16):
        handled = await service.handle_private_message(
            make_private_event(
                message_id=f"p-window-{index}",
                user_id=10001,
                text=f"owner-message-{index}",
            )
        )
        assert handled is True

    await service.handle_private_message(
        make_private_event(
            message_id="p-window-current",
            user_id=10001,
            text="owner-message-current",
        )
    )

    prompt_lines = llm_client.prompts[-1]
    summary_start = prompt_lines.index("Current private daily session summary:")
    turns_start = prompt_lines.index("Recent private daily turns:")
    summary_block = prompt_lines[summary_start + 1].splitlines()
    history_block = prompt_lines[turns_start + 1].splitlines()

    # Fifteen exchanges happened; the window keeps the newest 20 messages, so
    # the oldest survivors are #6 and its reply.  The in-flight turn is not
    # part of the window yet.
    expected_window = [
        line
        for index in range(6, 16)
        for line in (f"Owner: owner-message-{index}", f"Assistant: reply-{index}")
    ]
    assert len(expected_window) == private_chat_module.PRIVATE_CONTEXT_MESSAGE_LIMIT
    assert history_block == expected_window
    assert summary_block == expected_window
    assert sum(line.startswith("Owner: ") for line in history_block) == 10
    assert sum(line.startswith("Assistant: ") for line in history_block) == 10


@pytest.mark.asyncio
async def test_private_prompt_uses_the_persona_chat_voice(sqlite_engine, tmp_path) -> None:
    """``chat_voice: clingy`` swaps the mesugaki edge for the clingy voice."""

    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="在的在的，小町一直在等你说话呀。")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        assistant_name="比企谷小町",
        persona={
            "name": "比企谷小町",
            "identity": "A fixed AI persona modeled after Hikigaya Komachi.",
            "chat_voice": "clingy",
        },
    )

    await service.handle_private_message(
        make_private_event(message_id="p-chat-clingy", user_id=10001, text="在吗")
    )

    prompt = "\n".join(llm_client.prompts[0])
    assert "clingy little-sister warmth" in prompt
    assert "mesugaki" not in prompt


@pytest.mark.asyncio
async def test_owner_daily_chat_flattens_markdownish_reply(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="### 先说结论\n- 这个可以\n- 你现在就去改\n- 还有一堆实现细节后面再说")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-compact-1", user_id=10001, text="你刚才那个能不能简单说")
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == [
        "先说结论 这个可以。你现在就去改。还有一堆实现细节后面再说。"
    ]


@pytest.mark.asyncio
async def test_daily_chat_passes_daily_session_conversation_key(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="session keyed reply")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-key", user_id=10001, text="remember this chat")
    )

    assert handled is True
    with session_scope(sqlite_engine) as session:
        dev_session = DevSessionRepository(session).get_latest_owner_session(
            owner_qq=10001,
            session_mode=private_chat_module.SESSION_MODE_DAILY,
        )
    assert dev_session is not None
    assert llm_client.conversation_keys == [f"dev-session:{dev_session.id}"]


@pytest.mark.asyncio
async def test_daily_datetime_question_marks_runtime_facts_as_authoritative(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="现在是 2026 年。")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-year", user_id=10001, text="今年是几几年")
    )

    assert handled is True
    prompt = "\n".join(llm_client.prompts[0])
    assert "Runtime facts:" in prompt
    assert "Current local datetime:" in prompt
    assert "Treat runtime facts as authoritative for the current year, date, weekday, and clock time." in prompt


@pytest.mark.asyncio
async def test_daily_plain_question_still_receives_runtime_facts(sqlite_engine, tmp_path) -> None:
    """Every private turn carries the clock, not only date questions."""

    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我在的。")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-runtime-plain", user_id=10001, text="在吗")
    )

    assert handled is True
    prompt = "\n".join(llm_client.prompts[0])
    assert "Runtime facts:" in prompt
    assert "Current local date:" in prompt
    assert "Treat runtime facts as authoritative for the current year, date, weekday, and clock time." in prompt


@pytest.mark.asyncio
async def test_daily_chat_passes_current_images_to_llm(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我看到这张图了。")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-image",
            user_id=10001,
            text="看这个",
            msg_type="mixed",
            images=[
                ImageAttachment(
                    url="https://img.example.test/current-cat.png",
                    file_id="current-cat.png",
                    local_path=str(tmp_path / "current-cat.png"),
                )
            ],
        )
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == ["我看到这张图了。"]
    assert llm_client.images_calls[-1] is not None
    assert llm_client.images_calls[-1][0].file_id == "current-cat.png"
    assert llm_client.images_calls[-1][0].local_path == str(tmp_path / "current-cat.png")


@pytest.mark.asyncio
async def test_image_prompt_keeps_only_basic_vision_guidance(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="这是模型自己判断后的回复。")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-role-prompt",
            user_id=10001,
            text="这是什么动画的角色",
            msg_type="mixed",
            images=[
                ImageAttachment(
                    url="https://img.example.test/role.png",
                    file_id="role.png",
                    local_path=str(tmp_path / "role.png"),
                )
            ],
        )
    )

    assert handled is True
    assert sender.private_sent[-1].text == "这是模型自己判断后的回复。"
    prompt = "\n".join(llm_client.prompts[-1])
    assert "Vision task:" in prompt
    assert "attached image(s) belong to the current turn" in prompt
    assert "identify the most likely character name and franchise first" not in prompt
    assert "Do not pivot into generic art critique" not in prompt


@pytest.mark.asyncio
async def test_single_image_waits_silently_until_followup_text(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我看到的是你后面跟上的那张图。")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        private_image_followup_window_seconds=0.05,
    )

    handled_image = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-image-only-wait",
            user_id=10001,
            text="",
            msg_type="image",
            images=[
                ImageAttachment(
                    url="https://img.example.test/followup-cat.png",
                    file_id="followup-cat.png",
                    local_path=str(tmp_path / "followup-cat.png"),
                )
            ],
        )
    )
    await private_chat_module.asyncio.sleep(0.1)

    assert handled_image is True
    assert sender.private_sent == []
    assert llm_client.prompts == []

    handled_text = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-image-followup-wait",
            user_id=10001,
            text="这是谁",
        )
    )

    assert handled_text is True
    assert [outbound.text for outbound in sender.private_sent] == ["我看到的是你后面跟上的那张图。"]
    assert len(llm_client.prompts) == 1
    assert llm_client.images_calls[-1] is not None
    assert llm_client.images_calls[-1][0].file_id == "followup-cat.png"


@pytest.mark.asyncio
async def test_image_generation_request_uses_pinned_image_client(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    chat_llm = FakeLlmClient(reply_text="should not be used")
    image_llm = FakeImageGenerationLlm()
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=chat_llm,
        image_llm_client=image_llm,
        owner_qq=10001,
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-private-image-generate",
            user_id=10001,
            text="\u53c2\u8003\u8fd9\u5f20\u56fe\uff0c\u6539\u6210\u8d5b\u535a\u5e9f\u571f\u6d77\u62a5",
            msg_type="mixed",
            images=[
                ImageAttachment(
                    url="https://img.example.test/layout.png",
                    file_id="layout.png",
                    local_path=str(tmp_path / "layout.png"),
                )
            ],
        )
    )

    assert handled is True
    await service.private_image_service.wait_for_idle()
    assert chat_llm.prompts == []
    assert len(image_llm.edit_calls) == 1
    assert image_llm.edit_calls[0]["images"][0].file_id == "layout.png"
    assert image_llm.edit_calls[0]["max_attempts"] == 1
    assert image_llm.edit_calls[0]["timeout_seconds"] == 900.0
    assert sender.private_image_sent and sender.private_image_sent[0]["user_id"] == 10001
    assert [outbound.text for outbound in sender.private_sent] == ["图我接住了，开始画", "图好了"]


@pytest.mark.asyncio
async def test_reset_draw_clears_followup_image_context(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    chat_llm = FakeLlmClient(reply_text="现在不会再沿用上一张图。")
    image_llm = FakeImageGenerationLlm()
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=chat_llm,
        image_llm_client=image_llm,
        owner_qq=10001,
        private_image_followup_window_seconds=0.05,
    )

    with session_scope(sqlite_engine) as session:
        UserRepository(session).upsert_user(user_id=10001, nickname="owner", group_card="")
        MessageRepository(session).add_private_message(
            platform_msg_id="private-inbound-10001-p-private-layout-only",
            user_id=10001,
            timestamp=datetime(2026, 5, 10, 12, 0, tzinfo=UTC),
            plain_text="",
            raw_json={
                "message_id": "p-private-layout-only",
                "message": [
                    {
                        "type": "image",
                        "data": {
                            "file": "layout-followup.png",
                            "url": "https://img.example.test/layout-followup.png",
                            "local_path": str(tmp_path / "layout-followup.png"),
                        },
                    }
                ],
            },
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    handled_image = await service.handle_private_message(
        make_private_event(
            message_id="p-private-layout-only",
            user_id=10001,
            text="",
            msg_type="image",
            images=[
                ImageAttachment(
                    url="https://img.example.test/layout-followup.png",
                    file_id="layout-followup.png",
                    local_path=str(tmp_path / "layout-followup.png"),
                )
            ],
        )
    )
    handled_reset = await service.handle_private_message(
        make_private_event(message_id="p-private-draw-reset", user_id=10001, text="重置绘画")
    )
    handled_text = await service.handle_private_message(
        make_private_event(message_id="p-private-layout-followup", user_id=10001, text="这是谁")
    )
    await private_chat_module.asyncio.sleep(0.1)

    assert handled_image is True
    assert handled_reset is True
    assert handled_text is True
    assert image_llm.edit_calls == []
    assert image_llm.generate_calls == []
    assert sender.private_image_sent == []
    assert sender.private_sent[-1].text == "现在不会再沿用上一张图。"
    assert chat_llm.images_calls[-1] is None


@pytest.mark.asyncio
async def test_allowlisted_private_chat_replies_inline(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="可以正常私聊，但我不会替你改项目。")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        private_chat_qqs={10002},
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-guest-1", user_id=10002, text="你叫什么")
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == ["可以正常私聊，但我不会替你改项目。"]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert len(completed) == 1
    assert completed[0].intent_type == "private_chat"
    assert completed[0].requested_by_qq == 10002
    assert queued == []
    prompt = "\n".join(llm_client.prompts[0])
    assert "This user is not the owner" in prompt
    assert "Current user message:" in prompt


@pytest.mark.asyncio
async def test_unknown_private_sender_is_not_answered(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="should not be used")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        private_chat_qqs={10002},
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-stranger", user_id=10003, text="在吗")
    )

    assert handled is False
    assert sender.private_sent == []
    assert llm_client.prompts == []


@pytest.mark.asyncio
async def test_weather_location_followup_reuses_weather_context_for_new_search(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我按西安长安区重新查了一次天气。")
    search_client = FakeSearchClient()
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        web_search_client=search_client,
    )

    with session_scope(sqlite_engine) as session:
        sessions = DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001, session_mode="daily")
        first_task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="帮我上网搜一下今天西安西电南校区附近天气",
            intent_type="private_chat",
        )
        tasks.mark_completed(
            task_id=first_task.id,
            summary="weather lookup",
            result_text="我查了，但搜出来的地名不太对。",
            files_read=[],
            files_changed=[],
            commands_run=["llm_client.generate_text"],
            restart_required=False,
            restart_result="not-needed",
            checkpoint_dir="",
        )
        second_task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="西电南校区",
            intent_type="private_chat",
        )
        tasks.mark_completed(
            task_id=second_task.id,
            summary="weather lookup retry",
            result_text="这个词单独搜还是不准。",
            files_read=[],
            files_changed=[],
            commands_run=["llm_client.generate_text"],
            restart_required=False,
            restart_result="not-needed",
            checkpoint_dir="",
        )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-weather-followup", user_id=10001, text="那就西安长安区")
    )

    assert handled is True
    assert search_client.queries == [("西安长安区 今天天气", 3)]
    assert search_client.page_reads == [(["https://official.example"], "西安长安区 今天天气", 3)]
    assert [outbound.text for outbound in sender.private_sent] == ["我按西安长安区重新查了一次天气。"]


@pytest.mark.asyncio
async def test_new_session_command_creates_fresh_daily_session(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
    )

    await service.handle_private_message(
        make_private_event(message_id="p-1", user_id=10001, text="今天聊点什么")
    )
    await service.handle_private_message(make_private_event(message_id="p-2", user_id=10001, text="/bot new-session"))

    assert sender.private_sent[-1].text
    assert "会话" in sender.private_sent[-1].text
    with session_scope(sqlite_engine) as session:
        sessions = DevSessionRepository(session).list_recent_owner_sessions(owner_qq=10001, limit=10)
    assert len(sessions) == 2


@pytest.mark.asyncio
async def test_session_status_reports_active_session(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
    )

    await service.handle_private_message(
        make_private_event(message_id="p-1", user_id=10001, text="/bot session-status")
    )

    assert "当前日常对话" in sender.private_sent[-1].text


@pytest.mark.asyncio
async def test_send_private_text_deduplicates_same_context(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        bot_qq=200000003,
    )

    first = await service._send_private_text(
        user_id=10001,
        text="dedup reply",
        context="private_chat:123:completed",
    )
    second = await service._send_private_text(
        user_id=10001,
        text="dedup reply",
        context="private_chat:123:completed",
    )

    assert first is True
    assert second is True
    assert [outbound.text for outbound in sender.private_sent] == ["dedup reply"]

    with session_scope(sqlite_engine) as session:
        stored = MessageRepository(session).get_by_platform_msg_id(
            "private-outbound-private_chat:123:completed"
        )
    assert stored is not None
    assert stored.raw_json["delivery_state"] == "sent"


@pytest.mark.asyncio
async def test_send_private_text_preserves_reservation_when_delivery_is_uncertain(sqlite_engine, tmp_path) -> None:
    sender = UncertainPrivateSender()
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        bot_qq=200000003,
    )

    first = await service._send_private_text(
        user_id=10001,
        text="uncertain reply",
        context="private_chat:uncertain:completed",
    )
    second = await service._send_private_text(
        user_id=10001,
        text="uncertain reply",
        context="private_chat:uncertain:completed",
    )

    assert first is False
    assert second is True
    assert [outbound.text for outbound in sender.private_sent] == ["uncertain reply"]

    with session_scope(sqlite_engine) as session:
        stored = MessageRepository(session).get_by_platform_msg_id(
            "private-outbound-private_chat:uncertain:completed"
        )
    assert stored is not None
    assert stored.raw_json["delivery_state"] == "uncertain"
    assert stored.raw_json["failure_kind"] == "delivery_result_unknown"


@pytest.mark.asyncio
async def test_daily_chat_uses_quoted_private_images(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我看的是你引用的那张图。")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    with session_scope(sqlite_engine) as session:
        UserRepository(session).upsert_user(user_id=10001, nickname="owner", group_card="")
        MessageRepository(session).add_private_message(
            platform_msg_id="private-inbound-10001-p-prev-image",
            user_id=10001,
            timestamp=datetime(2026, 5, 10, 11, 59, tzinfo=UTC),
            plain_text="",
            raw_json={
                "message_id": "p-prev-image",
                "message": [
                    {
                        "type": "image",
                        "data": {
                            "file": "quoted-cat.png",
                            "url": "https://img.example.test/quoted-cat.png",
                            "local_path": str(tmp_path / "quoted-cat.png"),
                        },
                    }
                ],
            },
            msg_type="image",
            reply_to_msg_id=None,
        )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-quoted-image",
            user_id=10001,
            text="这张图怎么回事",
            reply_to_msg_id="p-prev-image",
        )
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == ["我看的是你引用的那张图。"]
    assert llm_client.images_calls[-1] is not None
    assert llm_client.images_calls[-1][0].file_id == "quoted-cat.png"
    assert llm_client.images_calls[-1][0].local_path == str(tmp_path / "quoted-cat.png")


@pytest.mark.asyncio
async def test_contextual_text_followup_reuses_recent_image(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我会继续按刚才那张图来判断。")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    with session_scope(sqlite_engine) as session:
        UserRepository(session).upsert_user(user_id=10001, nickname="owner", group_card="")
        MessageRepository(session).add_private_message(
            platform_msg_id="private-inbound-10001-p-chat-role-image",
            user_id=10001,
            timestamp=datetime(2026, 5, 10, 12, 0, tzinfo=UTC),
            plain_text="这是哪个角色",
            raw_json={
                "message_id": "p-chat-role-image",
                "message": [
                    {"type": "text", "data": {"text": "这是哪个角色"}},
                    {
                        "type": "image",
                        "data": {
                            "file": "witch-judge.png",
                            "url": "https://img.example.test/witch-judge.png",
                            "local_path": str(tmp_path / "witch-judge.png"),
                        },
                    },
                ],
            },
            msg_type="mixed",
            reply_to_msg_id=None,
        )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-role-followup",
            user_id=10001,
            text="这是魔女裁判游戏里的",
        )
    )

    assert handled is True
    assert llm_client.images_calls[-1] is not None
    assert llm_client.images_calls[-1][0].file_id == "witch-judge.png"


@pytest.mark.asyncio
async def test_reply_to_remote_private_image_runs_image_generation(sqlite_engine, tmp_path) -> None:
    gateway = FakeGateway(
        get_msg_responses={
            "quoted-bot-image-1": {
                "message_id": "quoted-bot-image-1",
                "message_type": "private",
                "message": [
                    {
                        "type": "image",
                        "data": {
                            "file": "quoted-bot-image.png",
                            "url": "https://img.example.test/quoted-bot-image.png",
                        },
                    }
                ],
            }
        }
    )
    sender = GatewayBackedSender(gateway=gateway)
    chat_llm = FakeLlmClient(reply_text="should not be used")
    image_llm = FakeImageGenerationLlm()
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=chat_llm,
        image_llm_client=image_llm,
        owner_qq=10001,
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-private-quoted-retouch",
            user_id=10001,
            text="在这张照片基础上进行轻度人像优化，保持人物身份特征和整体长相不变。",
            reply_to_msg_id="quoted-bot-image-1",
        )
    )

    assert handled is True
    await service.private_image_service.wait_for_idle()
    assert chat_llm.prompts == []
    assert len(image_llm.edit_calls) == 1
    assert image_llm.edit_calls[0]["images"][0].file_id == "quoted-bot-image.png"
    assert image_llm.edit_calls[0]["images"][0].url == "https://img.example.test/quoted-bot-image.png"
    assert len(sender.private_image_sent) == 1
    assert [outbound.text for outbound in sender.private_sent] == ["图我接住了，开始画", "图好了"]
    assert gateway.calls == [("get_msg", {"message_id": "quoted-bot-image-1"})]


@pytest.mark.asyncio
async def test_reply_to_remote_private_image_handles_simple_retouch_prompt(
    sqlite_engine, tmp_path
) -> None:
    gateway = FakeGateway(
        get_msg_responses={
            "quoted-bot-image-2": {
                "message_id": "quoted-bot-image-2",
                "message_type": "private",
                "message": [
                    {
                        "type": "image",
                        "data": {
                            "file": "quoted-bot-image-2.png",
                            "url": "https://img.example.test/quoted-bot-image-2.png",
                        },
                    }
                ],
            }
        }
    )
    sender = GatewayBackedSender(gateway=gateway)
    chat_llm = FakeLlmClient(reply_text="should not be used")
    image_llm = FakeImageGenerationLlm()
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=chat_llm,
        image_llm_client=image_llm,
        owner_qq=10001,
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-private-quoted-retouch-simple",
            user_id=10001,
            text="优化一下人脸，稍微调整一下五官修一下鼻毛和胡须，必须还要保持人脸的辨识度只能小修让人脸显的更好看",
            reply_to_msg_id="quoted-bot-image-2",
        )
    )

    assert handled is True
    await service.private_image_service.wait_for_idle()
    assert chat_llm.prompts == []
    assert len(image_llm.edit_calls) == 1
    assert image_llm.edit_calls[0]["images"][0].file_id == "quoted-bot-image-2.png"
    assert len(sender.private_image_sent) == 1
    assert [outbound.text for outbound in sender.private_sent] == ["图我接住了，开始画", "图好了"]


@pytest.mark.asyncio
async def test_service_start_recovers_running_private_image_task(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    image_llm = FakeImageGenerationLlm()
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=FakeLlmClient(reply_text="should not be used"),
        image_llm_client=image_llm,
        owner_qq=10001,
    )

    with session_scope(sqlite_engine) as session:
        sessions = DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(
            owner_qq=10001,
            session_mode=private_chat_module.SESSION_MODE_DAILY,
        )
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="参考这张图重新出图",
            intent_type="private_chat",
            status="running",
        )
        sessions.update_session(session_id=dev_session.id, last_task_id=task.id)
        JobRepository(session).add_job(
            job_type=service.private_image_service.job_type,
            payload_json=service.private_image_service._serialize_request(
                PrivateImageGenerationRequest(
                    user_id=10001,
                    trigger_message_id="recover-private-image-1",
                    prompt="参考这张图重新出图",
                    reference_images=[
                        ImageAttachment(
                            url="https://img.example.test/layout.png",
                            file_id="layout.png",
                            local_path=str(tmp_path / "layout.png"),
                        )
                    ],
                    dev_task_id=task.id,
                )
            ),
            run_at=datetime.now(UTC),
            status="running",
        )
        task_id = task.id

    await service.start()
    await service.private_image_service.wait_for_idle()
    await service.stop()

    assert len(image_llm.edit_calls) == 1
    assert len(sender.private_image_sent) == 1
    assert sender.private_image_sent[0]["user_id"] == 10001
    assert sender.private_sent[-1].text == "图好了"
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        failed = DevTaskRepository(session).list_tasks_by_status("failed")
    assert [task.id for task in completed] == [task_id]
    assert completed[0].result_text == "图好了"
    assert failed == []


@pytest.mark.asyncio
async def test_chinese_session_commands_reset_and_report_daily_session(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
    )

    await service.handle_private_message(
        make_private_event(message_id="p-cmd-1", user_id=10001, text="清空上下文")
    )
    assert "新的日常会话" in sender.private_sent[-1].text

    await service.handle_private_message(
        make_private_event(message_id="p-cmd-2", user_id=10001, text="会话状态")
    )
    assert "当前日常对话" in sender.private_sent[-1].text

    # The same command from two distinct messages must be answered twice.
    await service.handle_private_message(
        make_private_event(message_id="p-cmd-3", user_id=10001, text="重置会话")
    )
    await service.handle_private_message(
        make_private_event(message_id="p-cmd-4", user_id=10001, text="重置会话")
    )
    assert len(sender.private_sent) == 4
    assert sender.private_sent[2].text == sender.private_sent[3].text


def _burst_config(**overrides) -> dict:
    config = {
        "enabled": True,
        "separator": "|",
        "max_messages": 3,
        "max_chars": 64,
        "auto_split_long_segments": True,
        # Keep the delivery test fast; the zero-delay fallback is covered by
        # ``test_burst_delays_*`` in tests/core/test_chat_style.py.
        "min_delay_seconds": 0.01,
        "max_delay_seconds": 0.01,
    }
    config.update(overrides)
    return config


@pytest.mark.asyncio
async def test_private_chat_delivers_a_burst_reply_as_separate_messages(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="来了|几点|上号")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        bot_qq=200000003,
        reply_split_config=_burst_config(),
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-burst", user_id=10001, text="打游戏吗")
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == ["来了", "几点", "上号"]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        dev_session = DevSessionRepository(session).get_latest_owner_session(
            owner_qq=10001,
            session_mode=private_chat_module.SESSION_MODE_DAILY,
        )
        recent = DevTaskRepository(session).list_recent_tasks_for_session(
            session_id=dev_session.id,
            limit=1,
        )
    # The stored turn keeps the whole answer; delivery splits it.
    assert completed[0].result_text == "来了|几点|上号"
    task_id = recent[0].id
    with session_scope(sqlite_engine) as session:
        messages = MessageRepository(session)
        stored = [
            messages.get_by_platform_msg_id(
                f"private-outbound-private_chat:{task_id}:completed{suffix}"
            )
            for suffix in ("", "-b1", "-b2")
        ]
    assert all(message is not None for message in stored)
    assert [message.raw_json["delivery_state"] for message in stored] == [
        "sent",
        "sent",
        "sent",
    ]


@pytest.mark.asyncio
async def test_private_chat_strips_links_unless_the_user_asks_for_them(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="看这里 https://example.test/a 就明白了")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    await service.handle_private_message(
        make_private_event(message_id="p-chat-url-1", user_id=10001, text="这个是怎么做的")
    )

    assert "https://example.test/a" not in sender.private_sent[-1].text
    prompt = "\n".join(llm_client.prompts[0])
    assert "Do not include URLs, website addresses, Markdown links" in prompt

    requested_sender = FakeSender()
    requested_llm = FakeLlmClient(reply_text="看这里 https://example.test/a")
    requested_service = build_service(
        sqlite_engine,
        tmp_path,
        sender=requested_sender,
        llm_client=requested_llm,
        owner_qq=10001,
    )

    await requested_service.handle_private_message(
        make_private_event(message_id="p-chat-url-2", user_id=10001, text="把链接发我")
    )

    assert requested_sender.private_sent[-1].text == "看这里 https://example.test/a"
    requested_prompt = "\n".join(requested_llm.prompts[0])
    assert "The user explicitly requested links." in requested_prompt


@pytest.mark.asyncio
async def test_private_chat_puts_the_quoted_message_text_in_the_prompt(sqlite_engine, tmp_path) -> None:
    sender = GatewayBackedSender(
        gateway=FakeGateway(
            get_msg_responses={
                "q-quoted-text": {
                    "message": [{"type": "text", "data": {"text": "形式主义大国"}}],
                    "sender": {"nickname": "群友", "card": "阿渣"},
                    "user_id": 20002,
                }
            }
        )
    )
    llm_client = FakeLlmClient(reply_text="他是在说你刚才那句。")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        bot_qq=200000003,
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-quoted-text",
            user_id=10001,
            text="他在说谁",
            reply_to_msg_id="q-quoted-text",
        )
    )

    assert handled is True
    prompt = "\n".join(llm_client.prompts[0])
    assert "Quoted message: 阿渣（QQ昵称：群友）: 形式主义大国" in prompt
    assert "sender of the quoted message" in prompt


@pytest.mark.asyncio
async def test_private_chat_marks_a_searched_turn_as_search_priority(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我查了一下最近的新闻。")
    search_client = FakeSearchClient()
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        web_search_client=search_client,
    )

    await service.handle_private_message(
        make_private_event(message_id="p-chat-search", user_id=10001, text="上网搜一下最近的新闻")
    )

    assert search_client.queries
    prompt = "\n".join(llm_client.prompts[0])
    assert "Web search priority:" in prompt
    assert "Treat chat memory as background only." in prompt


@pytest.mark.asyncio
async def test_private_chat_forces_builtin_web_search_for_explicit_request(
    sqlite_engine, tmp_path, caplog
) -> None:
    """Production has no external client, so the provider tool carries search."""

    sender = FakeSender()
    llm_client = BuiltinSearchLlmClient(
        reply_text="我查了下现在的天气。",
        web_search_model="search-model",
    )
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    with caplog.at_level(logging.INFO, logger="app.private_chat.service"):
        await service.handle_private_message(
            make_private_event(
                message_id="p-chat-builtin-forced",
                user_id=10001,
                text="帮我查一下现在北京天气",
            )
        )

    assert service.web_search_client is None
    assert llm_client.generate_kwargs == [
        {"allow_web_search": True, "force_web_search": True}
    ]
    prompt = "\n".join(llm_client.prompts[0])
    assert "Web search priority:" in prompt
    # The turn carries no external evidence: the model has to search itself.
    assert "Web search results:" not in prompt
    assert any(
        record.message.startswith("private_web_search_builtin")
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_private_chat_allows_builtin_web_search_for_time_sensitive_turn(
    sqlite_engine, tmp_path
) -> None:
    """A dedicated search model scopes built-in search to fresh-info turns."""

    sender = FakeSender()
    llm_client = BuiltinSearchLlmClient(
        reply_text="我看看最近这条新闻。",
        web_search_model="search-model",
    )
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    await service.handle_private_message(
        make_private_event(
            message_id="p-chat-builtin-optional",
            user_id=10001,
            text="最近有什么新闻吗",
        )
    )

    assert llm_client.generate_kwargs == [{"allow_web_search": True}]
    prompt = "\n".join(llm_client.prompts[0])
    assert "Web search priority:" in prompt


@pytest.mark.asyncio
async def test_private_chat_keeps_plain_turns_out_of_builtin_web_search(
    sqlite_engine, tmp_path
) -> None:
    """A turn that needs no fresh facts must not attach the tool."""

    sender = FakeSender()
    llm_client = BuiltinSearchLlmClient(
        reply_text="我在的。",
        web_search_model="search-model",
    )
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    await service.handle_private_message(
        make_private_event(message_id="p-chat-builtin-plain", user_id=10001, text="在吗")
    )

    assert llm_client.generate_kwargs == [{"allow_web_search": False}]
    prompt = "\n".join(llm_client.prompts[0])
    assert "Web search priority:" not in prompt


@pytest.mark.asyncio
async def test_private_chat_excludes_search_verification_turns_from_builtin_search(
    sqlite_engine, tmp_path
) -> None:
    """Asking whether the bot searched must not trigger a fresh search."""

    sender = FakeSender()
    llm_client = BuiltinSearchLlmClient(
        reply_text="我刚刚没联网查。",
        web_search_model="search-model",
    )
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    await service.handle_private_message(
        make_private_event(
            message_id="p-chat-builtin-verification",
            user_id=10001,
            text="你刚刚上网查了吗",
        )
    )

    assert llm_client.generate_kwargs == [{"allow_web_search": False}]
    assert "Web search priority:" not in "\n".join(llm_client.prompts[0])


@pytest.mark.asyncio
async def test_private_chat_keeps_builtin_search_open_without_a_search_model(
    sqlite_engine, tmp_path
) -> None:
    """Without ``LLM_WEB_SEARCH_MODEL`` any turn may search, like the group."""

    sender = FakeSender()
    llm_client = BuiltinSearchLlmClient(reply_text="阿渣喜欢看动画。")
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    await service.handle_private_message(
        make_private_event(
            message_id="p-chat-builtin-unscoped",
            user_id=10001,
            text="阿渣喜欢什么动画",
        )
    )

    assert llm_client.generate_kwargs == [{"allow_web_search": True}]
    assert "Web search priority:" in "\n".join(llm_client.prompts[0])


@pytest.mark.asyncio
async def test_private_chat_keeps_external_search_turns_off_the_builtin_tool(
    sqlite_engine, tmp_path
) -> None:
    """With an external client the request stays on the old grounding path."""

    sender = FakeSender()
    llm_client = BuiltinSearchLlmClient(reply_text="我查了一下最近的新闻。")
    search_client = FakeSearchClient()
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        web_search_client=search_client,
    )

    await service.handle_private_message(
        make_private_event(
            message_id="p-chat-external-search",
            user_id=10001,
            text="上网搜一下最近的新闻",
        )
    )

    assert search_client.queries
    assert llm_client.generate_kwargs == [{"allow_web_search": False}]
    prompt = "\n".join(llm_client.prompts[0])
    assert "Web search results:" in prompt
    assert "Web search priority:" in prompt


@pytest.mark.asyncio
async def test_private_chat_forces_builtin_search_for_a_weather_followup(
    sqlite_engine, tmp_path
) -> None:
    """The private-only weather follow-up trigger also drives the provider tool."""

    sender = FakeSender()
    llm_client = BuiltinSearchLlmClient(
        reply_text="我按西安长安区重新查了一次天气。",
        web_search_model="search-model",
    )
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
    )

    with session_scope(sqlite_engine) as session:
        sessions = DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001, session_mode="daily")
        first_task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="帮我上网搜一下今天西安西电南校区附近天气",
            intent_type="private_chat",
        )
        tasks.mark_completed(
            task_id=first_task.id,
            summary="weather lookup",
            result_text="我查了，但搜出来的地名不太对。",
            files_read=[],
            files_changed=[],
            commands_run=["llm_client.generate_text"],
            restart_required=False,
            restart_result="not-needed",
            checkpoint_dir="",
        )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-builtin-weather-followup",
            user_id=10001,
            text="那就西安长安区",
        )
    )

    assert handled is True
    assert llm_client.generate_kwargs == [
        {"allow_web_search": True, "force_web_search": True}
    ]
    prompt = "\n".join(llm_client.prompts[0])
    assert "Web search priority:" in prompt
    # No external evidence and no decision call: the provider has to search.
    assert "Web search results:" not in prompt


def test_private_drawings_use_the_reference_search_and_planner_clients(sqlite_engine, tmp_path) -> None:
    chat_search = FakeSearchClient()
    reference_search = FakeImageReferenceSearchClient()
    planner = FakeReferencePlanner('{"should_search": false}')
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=FakeSender(),
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        web_search_client=chat_search,
        image_reference_search_client=reference_search,
        image_reference_planner_client=planner,
    )

    # The chat path keeps the chat search client; drawings use the reference one.
    assert service.web_search_client is chat_search
    assert service.private_image_service.web_search_client is reference_search
    assert service.private_image_service.image_reference_planner_client is planner


def test_private_drawings_fall_back_to_the_chat_search_client(sqlite_engine, tmp_path) -> None:
    chat_search = FakeSearchClient()
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=FakeSender(),
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        web_search_client=chat_search,
    )

    assert service.private_image_service.web_search_client is chat_search
    assert service.private_image_service.image_reference_planner_client is None


@pytest.mark.asyncio
async def test_private_reference_drawing_searches_images_with_the_planner(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    image_llm = FakeImageGenerationLlm()
    reference_search = FakeImageReferenceSearchClient(
        results=[
            ImageAttachment(
                url="https://img.example.test/azha-reference.png",
                file_id="azha-reference.png",
            )
        ]
    )
    planner = FakeReferencePlanner(
        '{"should_search": true, "references": ['
        '{"subject": "阿渣", "queries": ["阿渣 人设图"]}]}'
    )
    service = build_service(
        sqlite_engine,
        tmp_path,
        sender=sender,
        llm_client=FakeLlmClient(reply_text="should not be used"),
        image_llm_client=image_llm,
        owner_qq=10001,
        image_reference_search_client=reference_search,
        image_reference_planner_client=planner,
    )

    with session_scope(sqlite_engine) as session:
        sessions = DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(
            owner_qq=10001,
            session_mode=private_chat_module.SESSION_MODE_DAILY,
        )
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="参考阿渣的人设图生成一张",
            intent_type="private_chat",
        )
        dev_task_id = task.id

    result = await service.private_image_service.enqueue(
        PrivateImageGenerationRequest(
            user_id=10001,
            trigger_message_id="p-reference-image",
            prompt="参考阿渣的人设图生成一张",
            web_search_query="阿渣",
            dev_task_id=dev_task_id,
        )
    )

    await service.private_image_service.wait_for_idle()

    assert result.accepted is True
    assert len(planner.prompts) == 1
    assert reference_search.image_queries == [("阿渣 人设图", 2)]
    assert [image.url for image in image_llm.edit_calls[0]["images"]] == [
        "https://img.example.test/azha-reference.png"
    ]
    assert sender.private_image_sent and sender.private_image_sent[0]["user_id"] == 10001
