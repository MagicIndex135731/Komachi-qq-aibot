import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
import pytest
from sqlalchemy import select

from app.adapters.onebot_models import GroupMessageEvent
from app.adapters.sender import QQMessageBlockedError
from app.core.legacy_memory_context import LegacyMemoryPromptContext
from app.core.message_content import ImageAttachment
from app.core.memory_context_packer import PackedMemoryContext
from app.core.memory_orchestrator import MemoryContextResult, MemoryOrchestrator
import app.core.router as router_module
from app.core.router import InboundRouter
from app.core.reply_policy import ReplyDecision
from app.core.search_policy import AddressDecision
from app.storage.db import session_scope
from app.storage.models import MemoryItem, Message, Summary
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository


class FakeSender:
    def __init__(self) -> None:
        self.sent = []
        self.private_sent = []
        self.gateway = None

    async def send_group_text(self, outbound) -> None:
        self.sent.append(outbound)

    async def send_private_text(self, outbound) -> None:
        self.private_sent.append(outbound)


class FailingSenderOnce:
    def __init__(self) -> None:
        self.attempts = 0
        self.sent = []

    async def send_group_text(self, outbound) -> None:
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("send failed")
        self.sent.append(outbound)


class BlockingSender:
    def __init__(self) -> None:
        self.calls = []
        self.first_send_started = asyncio.Event()
        self.release_first_send = asyncio.Event()

    async def send_group_text(self, outbound) -> None:
        self.calls.append(outbound)
        if len(self.calls) == 1:
            self.first_send_started.set()
            await self.release_first_send.wait()


class QQBlockingSender:
    def __init__(self) -> None:
        self.sent = []

    async def send_group_text(self, outbound) -> None:
        self.sent.append(outbound)
        if len(self.sent) == 1:
            raise QQMessageBlockedError(
                "send_group_msg failed: status=failed retcode=1200 message=waitForSelfEcho timeout"
            )


class AlwaysQQBlockingSender:
    def __init__(self) -> None:
        self.sent = []

    async def send_group_text(self, outbound) -> None:
        self.sent.append(outbound)
        raise QQMessageBlockedError(
            "send_group_msg failed: status=failed retcode=1200 message=waitForSelfEcho timeout"
        )


class FakeLlm:
    def __init__(self) -> None:
        self.calls = []
        self.conversation_keys: list[str | None] = []

    def generate_text(self, prompt_lines: list[str], *, conversation_key=None) -> str:
        if any("Reply with exactly three lines in this grammar" in line for line in prompt_lines):
            return "SEARCH: no\nQUERY: \nREASON: answer-from-context"
        self.calls.append(prompt_lines)
        self.conversation_keys.append(conversation_key)
        return "I am here."


class LongReplyLlm:
    def __init__(self, reply_text: str) -> None:
        self.reply_text = reply_text
        self.calls = []

    def generate_text(self, prompt_lines: list[str], *, conversation_key=None) -> str:
        del conversation_key
        self.calls.append(prompt_lines)
        return self.reply_text


class MarkdownReplyLlm:
    def __init__(self) -> None:
        self.calls = []

    def generate_text(self, prompt_lines: list[str], *, conversation_key=None) -> str:
        self.calls.append(prompt_lines)
        return "### 先说结论\n- 这个可以\n- 你现在就去改"


class InspectingLlm:
    def __init__(self, *, sqlite_engine) -> None:
        self.sqlite_engine = sqlite_engine
        self.seen_messages_at_call: list[str] = []

    def generate_text(self, prompt_lines: list[str], *, conversation_key=None) -> str:
        del prompt_lines
        with session_scope(self.sqlite_engine) as session:
            messages = session.execute(select(Message).order_by(Message.id)).scalars().all()
        self.seen_messages_at_call = [message.plain_text for message in messages]
        return "I am here."


class InspectingReplyPolicy:
    def __init__(self, *, sqlite_engine) -> None:
        self.sqlite_engine = sqlite_engine
        self.seen_messages_at_decide: list[str] = []

    def decide(self, policy_input) -> ReplyDecision:
        del policy_input
        with session_scope(self.sqlite_engine) as session:
            messages = session.execute(select(Message).order_by(Message.id)).scalars().all()
        self.seen_messages_at_decide = [message.plain_text for message in messages]
        return ReplyDecision(True, "direct_trigger", 10)


class AlwaysProactiveReplyPolicy:
    def decide(self, policy_input) -> ReplyDecision:
        del policy_input
        return ReplyDecision(True, "proactive_score", 10)


class AlwaysDirectTriggerReplyPolicy:
    def decide(self, policy_input) -> ReplyDecision:
        del policy_input
        return ReplyDecision(True, "direct_trigger", 10)


class SearchAwareLlm:
    def __init__(self) -> None:
        self.calls = []
        self.search_decision_calls = 0
        self.reply_calls = 0

    def generate_text(self, prompt_lines: list[str], *, conversation_key=None) -> str:
        self.calls.append(prompt_lines)
        joined = "\n".join(prompt_lines)
        if "Reply with exactly three lines in this grammar" in joined:
            self.search_decision_calls += 1
            if "Target message: Alice: need-search" in joined or "Target message: Alice: 联网搜索aqua" in joined:
                return "SEARCH: yes\nQUERY: latest anime buzz\nREASON: current-facts-needed"
            return "SEARCH: no\nQUERY: \nREASON: answer-from-context"
        self.reply_calls += 1
        return "I checked just enough."


class FakeSearchClient:
    def __init__(self) -> None:
        self.queries = []
        self.page_reads = []

    def search(self, query: str, max_results: int = 3):
        self.queries.append((query, max_results))
        return [
            type(
                "Result",
                (),
                {
                    "title": "Official site",
                    "snippet": "Episode 1 aired and discussion focused on pacing.",
                    "source": "official.example",
                    "date": "2026-05-01",
                },
            )()
        ]

    def read_pages(self, results, *, query: str | None = None, max_pages: int = 3, skim_limit: int = 6):
        self.page_reads.append(([result.source for result in results], query, max_pages, skim_limit))
        return [
            type(
                "Page",
                (),
                {
                    "title": "Detailed review",
                    "url": "https://official.example/review",
                    "content": "Episode 1 introduces the cast. Episode 2 deepens the conflict.",
                },
            )()
        ]


class FailingSearchClient:
    def __init__(self) -> None:
        self.queries = []

    def search(self, query: str, max_results: int = 3):
        self.queries.append((query, max_results))
        raise RuntimeError("search backend unavailable")

    def read_pages(self, results, *, query: str | None = None, max_pages: int = 3, skim_limit: int = 6):
        raise RuntimeError("page reader unavailable")


class SearchDecisionFailingLlm:
    def __init__(self) -> None:
        self.search_decision_calls = 0
        self.reply_calls = 0

    def generate_text(self, prompt_lines: list[str], *, conversation_key=None) -> str:
        joined = "\n".join(prompt_lines)
        if "Reply with exactly three lines in this grammar" in joined:
            self.search_decision_calls += 1
            raise RuntimeError("search decision llm failed")
        self.reply_calls += 1
        return "Still replying."


class LocalLookupPromptInspectingLlm:
    def __init__(self) -> None:
        self.search_decision_calls = 0
        self.reply_calls = 0
        self.calls: list[list[str]] = []

    def generate_text(self, prompt_lines: list[str], *, conversation_key=None) -> str:
        joined = "\n".join(prompt_lines)
        if "Reply with exactly three lines in this grammar" in joined:
            self.search_decision_calls += 1
            return "SEARCH: yes\nQUERY: 西安电子科技大学 南校区 竹园牛肉拉面 具体位置\nREASON: local-facts-needed"
        self.reply_calls += 1
        self.calls.append(prompt_lines)
        return "I am here."


class RuntimeFactsInspectingLlm:
    def __init__(self) -> None:
        self.search_decision_calls = 0
        self.reply_calls = 0
        self.calls: list[list[str]] = []

    def generate_text(self, prompt_lines: list[str], *, conversation_key=None) -> str:
        self.calls.append(prompt_lines)
        joined = "\n".join(prompt_lines)
        if "Reply with exactly three lines in this grammar" in joined:
            self.search_decision_calls += 1
            return "SEARCH: yes\nQUERY: today date\nREASON: current-facts-needed"
        self.reply_calls += 1
        return "今天是 2026-05-09。"


class RelativeYearSearchLlm:
    def __init__(self) -> None:
        self.calls = []
        self.search_decision_calls = 0
        self.reply_calls = 0

    def generate_text(self, prompt_lines: list[str], *, conversation_key=None) -> str:
        self.calls.append(prompt_lines)
        joined = "\n".join(prompt_lines)
        if "Reply with exactly three lines in this grammar" in joined:
            self.search_decision_calls += 1
            return "SEARCH: yes\nQUERY: 今年欧冠冠军 当前结果\nREASON: current-facts-needed"
        self.reply_calls += 1
        return "I checked just enough."


class ImageCapturingLlm:
    def __init__(self) -> None:
        self.calls = []

    def generate_text(self, prompt_lines: list[str], *, images=None, conversation_key=None) -> str:
        self.calls.append({"prompt_lines": prompt_lines, "images": images})
        return "I can see it."


class WeeklyReportLlm:
    def __init__(self) -> None:
        self.calls = []
        self.conversation_keys: list[str | None] = []

    def generate_text(self, prompt_lines: list[str], *, conversation_key=None) -> str:
        self.calls.append(prompt_lines)
        self.conversation_keys.append(conversation_key)
        return "1|Alice|今天这波真离谱|火药味拉满\n2|Bob|这也太炸了吧|节目效果很强"


class FakeGateway:
    def __init__(self, *, responses=None, error: Exception | None = None) -> None:
        self.responses = responses or {}
        self.error = error
        self.calls = []

    async def call_api(self, action: str, params: dict) -> dict:
        self.calls.append((action, params))
        if self.error is not None:
            raise self.error
        message_id = str(params.get("message_id"))
        payload = self.responses.get(message_id)
        if payload is None:
            return {"status": "failed", "retcode": 1200, "data": None}
        return {"status": "ok", "retcode": 0, "data": payload}


class FailingReplyLlm:
    def __init__(self) -> None:
        self.calls = []

    def generate_text(self, prompt_lines: list[str], *, images=None, conversation_key=None) -> str:
        self.calls.append({"prompt_lines": prompt_lines, "images": images})
        raise RuntimeError("llm unavailable")


class StaticSearchClient:
    def __init__(self, *, results, pages) -> None:
        self.results = results
        self.pages = pages
        self.queries = []
        self.page_reads = []

    def search(self, query: str, max_results: int = 3):
        self.queries.append((query, max_results))
        return self.results

    def read_pages(self, results, *, query: str | None = None, max_pages: int = 3, skim_limit: int = 6):
        self.page_reads.append(([result.source for result in results], query, max_pages, skim_limit))
        return self.pages


class AlwaysReplyPolicy:
    def decide(self, policy_input) -> ReplyDecision:
        del policy_input
        return ReplyDecision(True, "proactive_score", 10)


def _fake_cache_local_paths(raw_payload, *, cache_dir) -> None:
    del cache_dir
    for index, segment in enumerate(raw_payload.get("message", []), start=1):
        if segment.get("type") == "image":
            segment.setdefault("data", {})["local_path"] = f"C:/tmp/cached-{index}.png"


class FakeGroupImageService:
    def __init__(self, *, accepted: bool, queue_position: int | None = None, reason: str = "") -> None:
        self.accepted = accepted
        self.queue_position = queue_position
        self.reason = reason
        self.requests = []

    async def enqueue(self, request):
        self.requests.append(request)
        return type(
            "GroupImageEnqueueResult",
            (),
            {
                "accepted": self.accepted,
                "queue_position": self.queue_position,
                "reason": self.reason,
            },
        )()


class FakeImageSearchClient:
    def __init__(self) -> None:
        self.image_queries: list[tuple[str, int]] = []

    def image_search(self, query: str, max_results: int = 3):
        self.image_queries.append((query, max_results))
        return []


def make_raw_payload(
    *,
    message_id: str,
    user_id: int,
    nickname: str,
    group_card: str,
    group_id: int,
    plain_text: str,
    mentioned_bot: bool,
    images: list[ImageAttachment],
    reply_to_msg_id: str | None,
    timestamp: datetime,
) -> dict:
    message_segments: list[dict] = []
    if reply_to_msg_id is not None:
        message_segments.append({"type": "reply", "data": {"id": reply_to_msg_id}})
    if mentioned_bot:
        message_segments.append({"type": "at", "data": {"qq": "123456789"}})
    if plain_text:
        message_segments.append({"type": "text", "data": {"text": plain_text}})
    for image in images:
        message_segments.append(
            {
                "type": "image",
                "data": {
                    "file": image.file_id or "image.png",
                    "url": image.url,
                },
            }
        )
    return {
        "post_type": "message",
        "message_id": message_id,
        "group_id": group_id,
        "user_id": user_id,
        "sender": {"nickname": nickname, "card": group_card},
        "time": int(timestamp.timestamp()),
        "message": message_segments,
    }


def make_event(
    *,
    group_id: int,
    mentioned_bot: bool,
    message_id: str = "m-1",
    plain_text: str | None = None,
    timestamp: datetime | None = None,
    user_id: int = 20001,
    nickname: str = "Alice",
    group_card: str = "",
    images: list[ImageAttachment] | None = None,
    reply_to_msg_id: str | None = None,
) -> GroupMessageEvent:
    resolved_plain_text = plain_text if plain_text is not None else ("@Mira hi" if mentioned_bot else "hello there")
    resolved_timestamp = timestamp or datetime(2026, 5, 9, 12, 0, tzinfo=UTC)
    resolved_images = images or []
    return GroupMessageEvent(
        platform_msg_id=message_id,
        group_id=group_id,
        user_id=user_id,
        nickname=nickname,
        group_card=group_card,
        plain_text=resolved_plain_text,
        raw_payload=make_raw_payload(
            message_id=message_id,
            user_id=user_id,
            nickname=nickname,
            group_card=group_card,
            group_id=group_id,
            plain_text=resolved_plain_text,
            mentioned_bot=mentioned_bot,
            images=resolved_images,
            reply_to_msg_id=reply_to_msg_id,
            timestamp=resolved_timestamp,
        ),
        timestamp=resolved_timestamp,
        msg_type="mixed" if resolved_images and resolved_plain_text else ("image" if resolved_images else "text"),
        images=resolved_images,
        mentioned_bot=mentioned_bot,
        reply_to_msg_id=reply_to_msg_id,
    )


@pytest.mark.asyncio
async def test_router_replies_to_direct_mention_in_allowlisted_group(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(make_event(group_id=10001, mentioned_bot=True))

    assert [outbound.text for outbound in sender.sent] == ["I am here."]
    assert len(llm.calls) == 1
    assert llm.conversation_keys == ["group:10001"]


@pytest.mark.asyncio
async def test_router_forwards_supported_bbot_command_in_target_group(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="bbot-1",
            plain_text="@Mira 帮我看看今天有什么新番",
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["[CQ:at,qq=20002] 今日新番"]
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_router_forwards_today_anime_command_for_alternate_phrasing(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="bbot-1b",
            plain_text="@Mira 告诉我今天有什么动画",
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["[CQ:at,qq=20002] 今日新番"]
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_router_forwards_bbot_command_with_extracted_argument(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="bbot-2",
            plain_text="@Mira 来个6657的烂梗",
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["[CQ:at,qq=20002] 随机烂梗 6657"]
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_router_does_not_forward_bbot_command_outside_target_group(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=30003,
            mentioned_bot=True,
            message_id="bbot-3",
            plain_text="@Mira 帮我看看今天有什么新番",
        )
    )

    assert sender.sent == []
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_blocks_bbot_admin_intent_without_bridge_admin_permission(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="bbot-4",
            plain_text="@Mira 把每日新番提醒打开",
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["这个需求对应 BBot 管理员命令，但我现在还没有它的管理员权限，先不能代你执行。"]
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_router_forwards_bbot_latest_dynamic_for_natural_language_query(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="bbot-dynamic-1",
            plain_text="@Mira 帮我看看老番茄的b站动态",
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["[CQ:at,qq=20002] 最新动态 老番茄"]
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_router_strips_latest_modifier_from_bbot_bilibili_dynamic_target(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="bbot-dynamic-2",
            plain_text="@Mira 看一下猫雷最新的b站动态",
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["[CQ:at,qq=20002] 最新动态 猫雷"]
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_router_uses_cached_bilibili_listener_uid_from_bbot_push_message(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            user_id=20002,
            nickname="BBot",
            message_id="bbot-cache-seed-1",
            plain_text=(
                "猫雷NyaRu_Official（UID: 697091119）于2026-05-13 21:05:50发布了一条新动态："
                "\n动态内容：\n【B限】可愛歌回！"
            ),
        )
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="bbot-cache-hit-1",
            plain_text="@Mira 看一下猫雷最新的b站动态",
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["[CQ:at,qq=20002] 最新动态 697091119"]
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_router_uses_cached_bilibili_listener_uid_from_list_response(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            user_id=20002,
            nickname="BBot",
            message_id="bbot-cache-seed-2",
            plain_text=(
                "B站监听列表：\n"
                "1. 猫雷NyaRu_Official（UID: 697091119）\n"
                "2. 阿森纳View（UID: 1473277782）"
            ),
        )
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="bbot-cache-hit-2",
            plain_text="@Mira 看一下猫雷最新的b站动态",
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["[CQ:at,qq=20002] 最新动态 697091119"]
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_router_forwards_bbot_latest_tweet_for_natural_language_query(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="bbot-tweet-1",
            plain_text="@Mira 看看 elonmusk 最近发了什么推文",
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["[CQ:at,qq=20002] 最新推文 elonmusk"]
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_router_blocks_natural_language_twitter_watch_admin_intent(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="bbot-admin-5",
            plain_text="@Mira 帮我监听一下 elonmusk 的推特",
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["这个需求对应 BBot 管理员命令，但我现在还没有它的管理员权限，先不能代你执行。"]
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_router_blocks_natural_language_bilibili_watch_admin_intent(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="bbot-admin-6",
            plain_text="@Mira 把 123456 加到b站监听里",
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["这个需求对应 BBot 管理员命令，但我现在还没有它的管理员权限，先不能代你执行。"]
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_router_enqueues_explicit_group_draw_request(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    image_service = FakeGroupImageService(accepted=True, queue_position=1)
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        group_image_service=image_service,
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            plain_text="@Mira 帮我画一张雨夜便利店",
            message_id="draw-1",
        )
    )

    assert len(image_service.requests) == 1
    assert image_service.requests[0].requester_user_id == 20001
    assert image_service.requests[0].prompt == "雨夜便利店"
    assert llm.calls == []
    assert sender.sent[-1].text


@pytest.mark.asyncio
async def test_router_enqueues_explicit_group_draw_request_with_current_image_reference(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    image_service = FakeGroupImageService(accepted=True, queue_position=1)
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        group_image_service=image_service,
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            plain_text="@Mira \u6539\u6210\u8d5b\u535a\u670b\u514b\u98ce",
            message_id="draw-with-image-1",
            images=[ImageAttachment(url="https://img.example.test/source.png", file_id="source.png")],
        )
    )

    assert len(image_service.requests) == 1
    assert image_service.requests[0].prompt == "\u6539\u6210\u8d5b\u535a\u670b\u514b\u98ce"
    assert [image.url for image in image_service.requests[0].reference_images] == [
        "https://img.example.test/source.png"
    ]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_enqueues_explicit_group_draw_request_with_recent_image_reference(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    image_service = FakeGroupImageService(accepted=True, queue_position=1)
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        group_image_service=image_service,
    )

    prior_timestamp = datetime(2026, 5, 9, 11, 59, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        messages.add_group_message(
            platform_msg_id="recent-image-1",
            group_id=10001,
            user_id=20001,
            timestamp=prior_timestamp,
            plain_text="",
            raw_json=make_raw_payload(
                message_id="recent-image-1",
                user_id=20001,
                nickname="Alice",
                group_card="",
                group_id=10001,
                plain_text="",
                mentioned_bot=False,
                images=[ImageAttachment(url="https://img.example.test/recent.png", file_id="recent.png")],
                reply_to_msg_id=None,
                timestamp=prior_timestamp,
            ),
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            plain_text="@Mira \u6539\u6210\u6cb9\u63cf\u52a8\u753b",
            message_id="draw-with-recent-image-1",
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        )
    )

    assert len(image_service.requests) == 1
    assert image_service.requests[0].prompt == "\u6539\u6210\u6cb9\u63cf\u52a8\u753b"
    assert [image.url for image in image_service.requests[0].reference_images] == [
        "https://img.example.test/recent.png"
    ]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_enqueues_recent_reference_request_with_multiple_images_from_same_prior_message(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    image_service = FakeGroupImageService(accepted=True, queue_position=1)
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        group_image_service=image_service,
    )

    prior_timestamp = datetime(2026, 5, 9, 11, 59, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        messages.add_group_message(
            platform_msg_id="recent-multi-image-1",
            group_id=10001,
            user_id=20001,
            timestamp=prior_timestamp,
            plain_text="",
            raw_json=make_raw_payload(
                message_id="recent-multi-image-1",
                user_id=20001,
                nickname="Alice",
                group_card="",
                group_id=10001,
                plain_text="",
                mentioned_bot=False,
                images=[
                    ImageAttachment(url="https://img.example.test/recent-a.png", file_id="recent-a.png"),
                    ImageAttachment(url="https://img.example.test/recent-b.png", file_id="recent-b.png"),
                ],
                reply_to_msg_id=None,
                timestamp=prior_timestamp,
            ),
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            plain_text="@Mira 改成赛博废土海报",
            message_id="draw-with-recent-multi-image-1",
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        )
    )

    assert len(image_service.requests) == 1
    assert image_service.requests[0].prompt == "改成赛博废土海报"
    assert [image.url for image in image_service.requests[0].reference_images] == [
        "https://img.example.test/recent-a.png",
        "https://img.example.test/recent-b.png",
    ]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_uses_only_latest_prior_image_message_when_images_are_split_across_messages(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    image_service = FakeGroupImageService(accepted=True, queue_position=1)
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        group_image_service=image_service,
    )

    first_timestamp = datetime(2026, 5, 9, 11, 58, 0, tzinfo=UTC)
    second_timestamp = datetime(2026, 5, 9, 11, 59, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        messages.add_group_message(
            platform_msg_id="split-image-1",
            group_id=10001,
            user_id=20001,
            timestamp=first_timestamp,
            plain_text="",
            raw_json=make_raw_payload(
                message_id="split-image-1",
                user_id=20001,
                nickname="Alice",
                group_card="",
                group_id=10001,
                plain_text="",
                mentioned_bot=False,
                images=[ImageAttachment(url="https://img.example.test/split-a.png", file_id="split-a.png")],
                reply_to_msg_id=None,
                timestamp=first_timestamp,
            ),
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="split-image-2",
            group_id=10001,
            user_id=20001,
            timestamp=second_timestamp,
            plain_text="",
            raw_json=make_raw_payload(
                message_id="split-image-2",
                user_id=20001,
                nickname="Alice",
                group_card="",
                group_id=10001,
                plain_text="",
                mentioned_bot=False,
                images=[ImageAttachment(url="https://img.example.test/split-b.png", file_id="split-b.png")],
                reply_to_msg_id=None,
                timestamp=second_timestamp,
            ),
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            plain_text="@Mira 改成蒸汽波封面",
            message_id="draw-with-split-images-1",
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        )
    )

    assert len(image_service.requests) == 1
    assert image_service.requests[0].prompt == "改成蒸汽波封面"
    assert [image.url for image in image_service.requests[0].reference_images] == [
        "https://img.example.test/split-b.png"
    ]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_enqueues_reference_image_request_with_embedded_draw_phrase(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    image_service = FakeGroupImageService(accepted=True, queue_position=1)
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        group_image_service=image_service,
    )

    quoted_timestamp = datetime(2026, 5, 9, 11, 59, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        messages.add_group_message(
            platform_msg_id="quoted-style-image-1",
            group_id=10001,
            user_id=20001,
            timestamp=quoted_timestamp,
            plain_text="",
            raw_json=make_raw_payload(
                message_id="quoted-style-image-1",
                user_id=20001,
                nickname="Alice",
                group_card="",
                group_id=10001,
                plain_text="",
                mentioned_bot=False,
                images=[ImageAttachment(url="https://img.example.test/quoted-style.png", file_id="quoted-style.png")],
                reply_to_msg_id=None,
                timestamp=quoted_timestamp,
            ),
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    prompt = "\u4eff\u7167\u8fd9\u4e24\u4e2a\u4eba\u7269\u52a8\u4f5c\u548c\u753b\u98ce\uff0c\u753b\u4e00\u5f20\u628a\u4eba\u7269\u6362\u6210\u8d85\u65f6\u7a7a\u8f89\u591c\u59ec\u7684\u4e24\u4e2a\u5973\u4e3b\u7684\u56fe"
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            plain_text=f"@Mira {prompt}",
            message_id="draw-with-embedded-phrase-1",
            reply_to_msg_id="quoted-style-image-1",
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        )
    )

    assert len(image_service.requests) == 1
    assert image_service.requests[0].prompt == prompt
    assert [image.url for image in image_service.requests[0].reference_images] == [
        "https://img.example.test/quoted-style.png"
    ]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_enqueues_reference_image_request_with_reference_generation_phrase(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    image_service = FakeGroupImageService(accepted=True, queue_position=1)
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        group_image_service=image_service,
    )

    prompt = "\u53c2\u8003\u8fd9\u5f20\u56fe\u751f\u6210\u4e00\u5f20\u50cf\u7d20\u98ce\u7248\u672c"
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            plain_text=f"@Mira {prompt}",
            message_id="draw-with-reference-generation-1",
            images=[ImageAttachment(url="https://img.example.test/source-pixel.png", file_id="source-pixel.png")],
        )
    )

    assert len(image_service.requests) == 1
    assert image_service.requests[0].prompt == prompt
    assert [image.url for image in image_service.requests[0].reference_images] == [
        "https://img.example.test/source-pixel.png"
    ]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_enqueues_reference_image_request_for_mimic_composition_phrase(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    image_service = FakeGroupImageService(accepted=True, queue_position=1)
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        group_image_service=image_service,
    )

    prompt = "模仿第二张图的构图，把第一张图六个人形成第二张图这种风格的图片，要求人脸必须保持第一张图的辨识度，其他可以灵活改变"
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            plain_text=f"@Mira {prompt}",
            message_id="draw-mimic-composition-1",
            images=[
                ImageAttachment(url="https://img.example.test/first.png", file_id="first.png"),
                ImageAttachment(url="https://img.example.test/second.png", file_id="second.png"),
            ],
        )
    )

    assert len(image_service.requests) == 1
    assert image_service.requests[0].prompt == prompt
    assert [image.url for image in image_service.requests[0].reference_images] == [
        "https://img.example.test/first.png",
        "https://img.example.test/second.png",
    ]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_enqueues_reference_image_request_for_on_this_image_basis_phrase(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    image_service = FakeGroupImageService(accepted=True, queue_position=1)
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        group_image_service=image_service,
    )

    quoted_timestamp = datetime(2026, 5, 9, 11, 59, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        messages.add_group_message(
            platform_msg_id="quoted-basis-image-1",
            group_id=10001,
            user_id=20001,
            timestamp=quoted_timestamp,
            plain_text="",
            raw_json=make_raw_payload(
                message_id="quoted-basis-image-1",
                user_id=20001,
                nickname="Alice",
                group_card="",
                group_id=10001,
                plain_text="",
                mentioned_bot=False,
                images=[ImageAttachment(url="https://img.example.test/quoted-basis.png", file_id="quoted-basis.png")],
                reply_to_msg_id=None,
                timestamp=quoted_timestamp,
            ),
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    prompt = "在这张图基础上画优化版横图，把这两个人弄到榻榻米上，修复手部异常"
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            plain_text=f"@Mira {prompt}",
            message_id="draw-on-this-image-basis-1",
            reply_to_msg_id="quoted-basis-image-1",
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        )
    )

    assert len(image_service.requests) == 1
    assert image_service.requests[0].prompt == prompt
    assert [image.url for image in image_service.requests[0].reference_images] == [
        "https://img.example.test/quoted-basis.png"
    ]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_enqueues_explicit_group_draw_request_when_replying_to_image_without_at(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    image_service = FakeGroupImageService(accepted=True, queue_position=1)
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        group_image_service=image_service,
    )
    router.reply_policy = AlwaysDirectTriggerReplyPolicy()

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(False, "none", 0),
    )

    quoted_timestamp = datetime(2026, 5, 9, 11, 59, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        messages.add_group_message(
            platform_msg_id="quoted-reply-image-1",
            group_id=10001,
            user_id=20001,
            timestamp=quoted_timestamp,
            plain_text="",
            raw_json=make_raw_payload(
                message_id="quoted-reply-image-1",
                user_id=20001,
                nickname="Alice",
                group_card="",
                group_id=10001,
                plain_text="",
                mentioned_bot=False,
                images=[ImageAttachment(url="https://img.example.test/reply-image.png", file_id="reply-image.png")],
                reply_to_msg_id=None,
                timestamp=quoted_timestamp,
            ),
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            plain_text="在这张图基础上画优化版横图，把这两个人弄到榻榻米上，修复手部异常",
            message_id="draw-reply-image-no-at-1",
            reply_to_msg_id="quoted-reply-image-1",
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        )
    )

    assert len(image_service.requests) == 1
    assert [image.url for image in image_service.requests[0].reference_images] == [
        "https://img.example.test/reply-image.png"
    ]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_enqueues_auto_web_character_reference_request_with_recent_layout_image(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    image_service = FakeGroupImageService(accepted=True, queue_position=1)
    search_client = FakeImageSearchClient()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
        group_image_service=image_service,
    )

    prior_timestamp = datetime(2026, 5, 9, 11, 59, 0, tzinfo=UTC)
    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        messages.add_group_message(
            platform_msg_id="auto-web-layout-1",
            group_id=10001,
            user_id=20001,
            timestamp=prior_timestamp,
            plain_text="",
            raw_json=make_raw_payload(
                message_id="auto-web-layout-1",
                user_id=20001,
                nickname="Alice",
                group_card="",
                group_id=10001,
                plain_text="",
                mentioned_bot=False,
                images=[ImageAttachment(url="https://img.example.test/layout.png", file_id="layout.png")],
                reply_to_msg_id=None,
                timestamp=prior_timestamp,
            ),
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            plain_text="@Mira 去网上找超时空辉夜姬两个女主的人设图，保留前图构图，只替换人物出图",
            message_id="draw-auto-web-ref-1",
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        )
    )

    assert len(image_service.requests) == 1
    assert image_service.requests[0].web_search_query == "超时空辉夜姬两个女主"
    assert "保留前图构图" in image_service.requests[0].prompt
    assert [image.url for image in image_service.requests[0].reference_images] == [
        "https://img.example.test/layout.png"
    ]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_enqueues_auto_web_character_reference_request_for_search_then_generate_phrase(
    sqlite_engine,
) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    image_service = FakeGroupImageService(accepted=True, queue_position=1)
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        group_image_service=image_service,
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            plain_text=(
                "@Mira 先去网上搜索超时空辉夜姬两个女主辉夜和酒寄彩叶的人设图 "
                "然后生成一张超时空辉夜姬动画的酒寄彩叶和辉夜近距离对视"
            ),
            message_id="draw-auto-web-ref-search-then-generate-1",
        )
    )

    assert len(image_service.requests) == 1
    assert image_service.requests[0].web_search_query == "超时空辉夜姬两个女主辉夜和酒寄彩叶"
    assert "生成一张超时空辉夜姬动画的酒寄彩叶和辉夜近距离对视" in image_service.requests[0].prompt
    assert llm.calls == []
    assert sender.sent[-1].text


@pytest.mark.asyncio
async def test_router_keeps_normal_image_question_on_recognition_path(sqlite_engine) -> None:
    sender = FakeSender()
    llm = ImageCapturingLlm()
    image_service = FakeGroupImageService(accepted=True, queue_position=1)
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        group_image_service=image_service,
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            plain_text="@Mira \u8fd9\u662f\u8c01",
            message_id="normal-image-question-1",
            images=[ImageAttachment(url="https://img.example.test/character.png", file_id="character.png")],
        )
    )

    assert image_service.requests == []
    assert len(llm.calls) == 1
    assert [image.url for image in llm.calls[0]["images"]] == ["https://img.example.test/character.png"]


@pytest.mark.asyncio
async def test_router_replies_when_group_image_queue_is_full(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    image_service = FakeGroupImageService(accepted=False, reason="queue_full")
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        group_image_service=image_service,
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            plain_text="@Mira 画个猫猫",
            message_id="draw-full-1",
        )
    )

    assert llm.calls == []
    assert sender.sent[-1].text


@pytest.mark.asyncio
async def test_router_passes_images_for_addressed_turn_without_putting_them_in_recent_history(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = ImageCapturingLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(True, "named_bot", 10),
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="image-addressed-1",
            plain_text="look at this",
            images=[ImageAttachment(url="https://img.example.test/cat.png", file_id="cat.png", local_path="C:/tmp/cat.png")],
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["I can see it."]
    assert len(llm.calls) == 1
    assert [image.url for image in llm.calls[0]["images"]] == ["https://img.example.test/cat.png"]
    assert not any("img.example.test" in line for line in llm.calls[0]["prompt_lines"])

    with session_scope(sqlite_engine) as session:
        stored = session.execute(
            select(Message).where(Message.platform_msg_id == "image-addressed-1")
        ).scalar_one()

    assert stored.msg_type == "mixed"


@pytest.mark.asyncio
async def test_router_caches_inbound_image_payload_before_persisting(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = ImageCapturingLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    def fake_cache(raw_payload, *, cache_dir) -> None:
        raw_payload["message"][0]["data"]["local_path"] = str(cache_dir / "10001" / "cached.png")

    monkeypatch.setattr(router_module, "cache_images_in_raw_payload", fake_cache)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="cache-persist-1",
            plain_text="",
            images=[ImageAttachment(url="https://img.example.test/cat.png", file_id="cat.png", local_path="C:/tmp/cat.png")],
        )
    )

    with session_scope(sqlite_engine) as session:
        stored = session.execute(
            select(Message).where(Message.platform_msg_id == "cache-persist-1")
        ).scalar_one()

    assert stored.raw_json["message"][0]["data"]["local_path"].endswith("cached.png")


@pytest.mark.asyncio
async def test_router_archives_inbound_and_outbound_messages_with_identity_snapshot(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    router.runtime.settings.data_dir = tmp_path

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="archive-turn-1",
            plain_text="@Mira 今晚吃什么",
            user_id=10001,
            nickname="不知道叫什么",
            group_card="群友甲",
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        )
    )

    archive_path = tmp_path / "history" / "group-10001" / "2026-05-09.jsonl"
    records = [json.loads(line) for line in archive_path.read_text(encoding="utf-8").splitlines()]

    assert records == [
        {
            "timestamp": "2026-05-09T12:00:00+00:00",
            "group_id": 10001,
            "platform_msg_id": "archive-turn-1",
            "user_id": 10001,
            "nickname": "不知道叫什么",
            "group_card": "群友甲",
            "plain_text": "@Mira 今晚吃什么",
            "msg_type": "text",
            "mentioned_bot": True,
            "reply_to_msg_id": None,
            "direction": "inbound",
            "image_local_paths": [],
        },
        {
            "timestamp": "2026-05-09T12:00:00+00:00",
            "group_id": 10001,
            "platform_msg_id": "bot-reply-archive-turn-1",
            "user_id": 123456789,
            "nickname": "Mira",
            "group_card": "",
            "plain_text": "I am here.",
            "msg_type": "text",
            "mentioned_bot": False,
            "reply_to_msg_id": "archive-turn-1",
            "direction": "outbound",
            "image_local_paths": [],
        },
    ]


@pytest.mark.asyncio
async def test_router_does_not_pass_images_for_unaddressed_turn_even_if_it_replies(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = ImageCapturingLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    router.reply_policy = AlwaysReplyPolicy()

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(False, "none", 0),
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="image-proactive-1",
            plain_text="random photo",
            images=[ImageAttachment(url="https://img.example.test/cat.png", file_id="cat.png", local_path="C:/tmp/cat.png")],
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["I can see it."]
    assert len(llm.calls) == 1
    assert llm.calls[0]["images"] is None


@pytest.mark.asyncio
async def test_router_adds_image_fallback_text_for_image_only_addressed_turn(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = ImageCapturingLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    monkeypatch.setattr(router_module, "cache_images_in_raw_payload", _fake_cache_local_paths)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="image-only-1",
            plain_text="",
            images=[ImageAttachment(url="https://img.example.test/cat.png", file_id="cat.png", local_path="C:/tmp/cat.png")],
        )
    )

    assert sender.sent == []
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_uses_referenced_image_when_addressed_turn_quotes_image(sqlite_engine) -> None:
    sender = FakeSender()
    llm = ImageCapturingLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20002, nickname="Bob", group_card="")
        messages.add_group_message(
            platform_msg_id="quoted-image-1",
            group_id=10001,
            user_id=20002,
            timestamp=datetime(2026, 5, 9, 11, 59, 0, tzinfo=UTC),
            plain_text="",
            raw_json=make_raw_payload(
                message_id="quoted-image-1",
                user_id=20002,
                nickname="Bob",
                group_card="",
                group_id=10001,
                plain_text="",
                mentioned_bot=False,
                images=[ImageAttachment(url="https://img.example.test/quoted.png", file_id="quoted.png")],
                reply_to_msg_id=None,
                timestamp=datetime(2026, 5, 9, 11, 59, 0, tzinfo=UTC),
            ),
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="quoted-trigger-1",
            plain_text="这张",
            reply_to_msg_id="quoted-image-1",
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["I can see it."]
    assert [image.url for image in llm.calls[0]["images"]] == ["https://img.example.test/quoted.png"]


@pytest.mark.asyncio
async def test_router_includes_quoted_text_in_group_prompt_when_replying_to_text_message(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    sender.gateway = FakeGateway(
        responses={
            "quoted-text-1": make_raw_payload(
                message_id="quoted-text-1",
                user_id=20002,
                nickname="Bob",
                group_card="",
                group_id=10001,
                plain_text="今天这句就按阴阳怪气那种语气说",
                mentioned_bot=False,
                images=[],
                reply_to_msg_id=None,
                timestamp=datetime(2026, 5, 9, 12, 1, 30, tzinfo=UTC),
            )
        }
    )
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="quoted-text-trigger-1",
            plain_text="@Mira 模仿上面的话",
            user_id=20001,
            nickname="Alice",
            reply_to_msg_id="quoted-text-1",
            timestamp=datetime(2026, 5, 9, 12, 3, 0, tzinfo=UTC),
        )
    )

    prompt_text = "\n".join(llm.calls[0])
    assert "今天这句就按阴阳怪气那种语气说" in prompt_text


@pytest.mark.asyncio
async def test_router_does_not_directly_reply_to_unaddressed_quote_of_non_bot_message(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20002, nickname="Bob", group_card="")
        messages.add_group_message(
            platform_msg_id="quoted-human-1",
            group_id=10001,
            user_id=20002,
            timestamp=datetime(2026, 5, 9, 11, 59, 0, tzinfo=UTC),
            plain_text="quoted human text",
            raw_json={"post_type": "message", "message_id": "quoted-human-1"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="quote-human-trigger-1",
            plain_text="确实",
            reply_to_msg_id="quoted-human-1",
        )
    )

    assert sender.sent == []
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_directly_replies_to_unaddressed_quote_of_bot_message(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=router.runtime.settings.bot_qq, nickname="Mira", group_card="")
        messages.add_group_message(
            platform_msg_id="quoted-bot-1",
            group_id=10001,
            user_id=router.runtime.settings.bot_qq,
            timestamp=datetime(2026, 5, 9, 11, 59, 0, tzinfo=UTC),
            plain_text="bot text",
            raw_json={"post_type": "message", "message_id": "quoted-bot-1"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="quote-bot-trigger-1",
            plain_text="这句呢",
            reply_to_msg_id="quoted-bot-1",
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["I am here."]
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_router_uses_gateway_resolved_quoted_image_when_local_quote_id_is_unresolved(
    sqlite_engine,
) -> None:
    sender = FakeSender()
    llm = ImageCapturingLlm()
    sender.gateway = FakeGateway(
        responses={
            "missing-quoted-id": make_raw_payload(
                message_id="missing-quoted-id",
                user_id=20002,
                nickname="Bob",
                group_card="",
                group_id=10001,
                plain_text="",
                mentioned_bot=False,
                images=[ImageAttachment(url="https://img.example.test/remote-quoted.png", file_id="remote-quoted.png")],
                reply_to_msg_id=None,
                timestamp=datetime(2026, 5, 9, 12, 1, 30, tzinfo=UTC),
            )
        }
    )
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        users.upsert_user(user_id=20002, nickname="Bob", group_card="")
        messages.add_group_message(
            platform_msg_id="stale-user-image-1",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
            plain_text="",
            raw_json=make_raw_payload(
                message_id="stale-user-image-1",
                user_id=20001,
                nickname="Alice",
                group_card="",
                group_id=10001,
                plain_text="",
                mentioned_bot=False,
                images=[ImageAttachment(url="https://img.example.test/stale-user.png", file_id="stale-user.png")],
                reply_to_msg_id=None,
                timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
            ),
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="other-user-image-1",
            group_id=10001,
            user_id=20002,
            timestamp=datetime(2026, 5, 9, 12, 1, 0, tzinfo=UTC),
            plain_text="",
            raw_json=make_raw_payload(
                message_id="other-user-image-1",
                user_id=20002,
                nickname="Bob",
                group_card="",
                group_id=10001,
                plain_text="",
                mentioned_bot=False,
                images=[ImageAttachment(url="https://img.example.test/other-user.png", file_id="other-user.png")],
                reply_to_msg_id=None,
                timestamp=datetime(2026, 5, 9, 12, 1, 0, tzinfo=UTC),
            ),
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="interruption-text-1",
            group_id=10001,
            user_id=20002,
            timestamp=datetime(2026, 5, 9, 12, 2, 0, tzinfo=UTC),
            plain_text="先别急，我插一句",
            raw_json={"post_type": "message", "message_id": "interruption-text-1"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="unresolved-quote-trigger-1",
            plain_text="谁最像日本偶像剧角色",
            user_id=20001,
            nickname="Alice",
            reply_to_msg_id="missing-quoted-id",
            timestamp=datetime(2026, 5, 9, 12, 3, 0, tzinfo=UTC),
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["I can see it."]
    assert [image.url for image in llm.calls[0]["images"]] == ["https://img.example.test/remote-quoted.png"]


@pytest.mark.asyncio
async def test_router_does_not_fall_back_to_recent_image_when_quoted_remote_message_is_text_only(sqlite_engine) -> None:
    sender = FakeSender()
    llm = ImageCapturingLlm()
    sender.gateway = FakeGateway(
        responses={
            "missing-quoted-id": make_raw_payload(
                message_id="missing-quoted-id",
                user_id=20002,
                nickname="Bob",
                group_card="",
                group_id=10001,
                plain_text="just a long text post",
                mentioned_bot=False,
                images=[],
                reply_to_msg_id=None,
                timestamp=datetime(2026, 5, 9, 12, 1, 30, tzinfo=UTC),
            )
        }
    )
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        users.upsert_user(user_id=20002, nickname="Bob", group_card="")
        messages.add_group_message(
            platform_msg_id="other-user-image-1",
            group_id=10001,
            user_id=20002,
            timestamp=datetime(2026, 5, 9, 12, 1, 0, tzinfo=UTC),
            plain_text="",
            raw_json=make_raw_payload(
                message_id="other-user-image-1",
                user_id=20002,
                nickname="Bob",
                group_card="",
                group_id=10001,
                plain_text="",
                mentioned_bot=False,
                images=[ImageAttachment(url="https://img.example.test/other-user.png", file_id="other-user.png")],
                reply_to_msg_id=None,
                timestamp=datetime(2026, 5, 9, 12, 1, 0, tzinfo=UTC),
            ),
            msg_type="image",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="unresolved-quote-trigger-2",
            plain_text="绠€鐭€荤粨涓€涓?",
            user_id=20001,
            nickname="Alice",
            reply_to_msg_id="missing-quoted-id",
            timestamp=datetime(2026, 5, 9, 12, 3, 0, tzinfo=UTC),
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["I can see it."]
    assert llm.calls[0]["images"] is None


@pytest.mark.asyncio
async def test_router_uses_followup_image_after_prior_addressed_turn_even_with_interruption(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = ImageCapturingLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    monkeypatch.setattr(router_module, "cache_images_in_raw_payload", _fake_cache_local_paths)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="image-session-open-1",
            plain_text="看这个",
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        )
    )
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="interruption-1",
            plain_text="我插一句",
            user_id=20002,
            nickname="Bob",
            timestamp=datetime(2026, 5, 9, 12, 0, 30, tzinfo=UTC),
        )
    )
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="image-followup-1",
            plain_text="",
            images=[ImageAttachment(url="https://img.example.test/followup.png", file_id="followup.png", local_path="C:/tmp/followup.png")],
            timestamp=datetime(2026, 5, 9, 12, 1, 0, tzinfo=UTC),
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["I can see it."]
    assert llm.calls[-1]["images"] is None


@pytest.mark.asyncio
async def test_router_uses_recent_image_when_user_addresses_bot_after_posting_image(sqlite_engine) -> None:
    sender = FakeSender()
    llm = ImageCapturingLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="recent-image-1",
            plain_text="",
            images=[ImageAttachment(url="https://img.example.test/recent.png", file_id="recent.png")],
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        )
    )
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="recent-image-trigger-1",
            plain_text="这张",
            timestamp=datetime(2026, 5, 9, 12, 1, 0, tzinfo=UTC),
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["I can see it."]
    assert [image.url for image in llm.calls[-1]["images"]] == ["https://img.example.test/recent.png"]


@pytest.mark.asyncio
async def test_router_waits_silently_on_single_image_until_followup_text(sqlite_engine) -> None:
    sender = FakeSender()
    llm = ImageCapturingLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="recent-image-hold-1",
            plain_text="",
            images=[ImageAttachment(url="https://img.example.test/recent-hold.png", file_id="recent-hold.png")],
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        )
    )

    assert sender.sent == []
    assert llm.calls == []

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="recent-image-hold-trigger-1",
            plain_text="她是谁",
            timestamp=datetime(2026, 5, 9, 12, 0, 8, tzinfo=UTC),
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["I can see it."]
    assert [image.url for image in llm.calls[-1]["images"]] == ["https://img.example.test/recent-hold.png"]


@pytest.mark.asyncio
async def test_router_uses_recent_image_for_natural_phrase_about_user_posted_image(sqlite_engine) -> None:
    sender = FakeSender()
    llm = ImageCapturingLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="recent-user-image-1",
            plain_text="",
            images=[ImageAttachment(url="https://img.example.test/user-posted.png", file_id="user-posted.png")],
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        )
    )
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="recent-user-image-trigger-1",
            plain_text="我发的图是谁",
            timestamp=datetime(2026, 5, 9, 12, 0, 8, tzinfo=UTC),
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["I can see it."]
    assert [image.url for image in llm.calls[-1]["images"]] == ["https://img.example.test/user-posted.png"]


@pytest.mark.asyncio
async def test_router_does_not_bind_stale_image_for_non_image_addressed_text(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = ImageCapturingLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    monkeypatch.setattr(router_module, "cache_images_in_raw_payload", _fake_cache_local_paths)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="stale-image-1",
            plain_text="",
            images=[ImageAttachment(url="https://img.example.test/stale.png", file_id="stale.png", local_path="C:/tmp/stale.png")],
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        )
    )
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="stale-image-trigger-1",
            plain_text="今天吃什么",
            timestamp=datetime(2026, 5, 9, 12, 1, 0, tzinfo=UTC),
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["I can see it."]
    assert [image.url for image in llm.calls[-1]["images"]] == ["https://img.example.test/stale.png"]


@pytest.mark.asyncio
async def test_router_does_not_reuse_already_consumed_recent_image_after_intervening_user_text(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = ImageCapturingLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    router.runtime.persona["name"] = "比企谷小町"
    monkeypatch.setattr(router_module, "cache_images_in_raw_payload", _fake_cache_local_paths)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="consumed-image-1",
            plain_text="",
            images=[ImageAttachment(url="https://img.example.test/consumed.png", file_id="consumed.png", local_path="C:/tmp/consumed.png")],
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        )
    )
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="consumed-image-trigger-1",
            plain_text="小町，这张图是谁",
            timestamp=datetime(2026, 5, 9, 12, 0, 9, tzinfo=UTC),
        )
    )
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="consumed-image-break-1",
            plain_text="ok",
            timestamp=datetime(2026, 5, 9, 12, 0, 33, tzinfo=UTC),
        )
    )
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="consumed-image-trigger-2",
            plain_text="这张图怎么样",
            timestamp=datetime(2026, 5, 9, 12, 2, 36, tzinfo=UTC),
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["I can see it.", "I can see it."]
    assert [image.url for image in llm.calls[0]["images"]] == ["https://img.example.test/consumed.png"]
    assert llm.calls[-1]["images"] is None


@pytest.mark.asyncio
async def test_router_normalizes_non_admin_markdownish_reply_before_sending(sqlite_engine) -> None:
    sender = FakeSender()
    llm = MarkdownReplyLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(make_event(group_id=10001, mentioned_bot=True, message_id="humanize-1"))

    assert [outbound.text for outbound in sender.sent] == ["先说结论 这个可以。你现在就去改。"]


@pytest.mark.asyncio
async def test_router_replies_to_short_persona_alias_without_at(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    router.runtime.persona["name"] = "\u6bd4\u4f01\u8c37\u5c0f\u753a"

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="alias-1",
            plain_text="\u5c0f\u753a\uff0c\u4f60\u597d\u53ef\u7231",
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["I am here."]
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_router_does_not_special_case_token_or_cost_keywords(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="token-keywords-1",
            user_id=987654321,
            plain_text="@Mira 查询 token 用量、消耗和花费",
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["I am here."]
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_router_ignores_duplicate_inbound_delivery(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    event = make_event(group_id=10001, mentioned_bot=True, message_id="dup-1")

    await router.handle_group_message(event)
    await router.handle_group_message(event)

    with session_scope(sqlite_engine) as session:
        stored_messages = session.execute(
            select(Message).where(Message.group_id == 10001).order_by(Message.id)
        ).scalars().all()

    assert [outbound.text for outbound in sender.sent] == ["I am here."]
    assert len(llm.calls) == 1
    assert [message.plain_text for message in stored_messages] == ["@Mira hi", "I am here."]


def test_router_ingests_historical_group_message_without_replying(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    event = make_event(group_id=10001, mentioned_bot=True, message_id="history-only-1")

    persisted = router.ingest_historical_group_message(event)

    with session_scope(sqlite_engine) as session:
        stored_messages = session.execute(
            select(Message).where(Message.group_id == 10001).order_by(Message.id)
        ).scalars().all()

    assert persisted is True
    assert sender.sent == []
    assert llm.calls == []
    assert [message.platform_msg_id for message in stored_messages] == ["history-only-1"]


def test_router_skips_image_cache_downloads_during_historical_ingest(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    event = make_event(
        group_id=10001,
        mentioned_bot=False,
        message_id="history-image-1",
        plain_text="",
        images=[ImageAttachment(url="https://img.example.test/history.png", file_id="history.png")],
    )
    cache_calls: list[str] = []

    def fake_cache(raw_payload, *, cache_dir) -> None:
        del raw_payload, cache_dir
        cache_calls.append("called")

    monkeypatch.setattr(router_module, "cache_images_in_raw_payload", fake_cache)

    persisted = router.ingest_historical_group_message(event)

    assert persisted is True
    assert cache_calls == []


@pytest.mark.asyncio
async def test_router_retries_duplicate_inbound_when_first_send_fails(sqlite_engine) -> None:
    sender = FailingSenderOnce()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    event = make_event(group_id=10001, mentioned_bot=True, message_id="dup-retry-1")

    with pytest.raises(RuntimeError, match="send failed"):
        await router.handle_group_message(event)

    await router.handle_group_message(event)

    with session_scope(sqlite_engine) as session:
        stored_messages = session.execute(
            select(Message).where(Message.group_id == 10001).order_by(Message.id)
        ).scalars().all()

    assert sender.attempts == 2
    assert [outbound.text for outbound in sender.sent] == ["I am here."]
    assert len(llm.calls) == 2
    assert [message.plain_text for message in stored_messages] == ["@Mira hi", "I am here."]


@pytest.mark.asyncio
async def test_router_persists_qq_blocked_reply_and_sends_safe_notice(sqlite_engine) -> None:
    sender = QQBlockingSender()
    llm = LongReplyLlm("sensitive generated detail")
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(group_id=10001, mentioned_bot=True, message_id="qq-blocked-1", plain_text="@Mira explain")
    )

    with session_scope(sqlite_engine) as session:
        stored_messages = session.execute(
            select(Message).where(Message.group_id == 10001).order_by(Message.id)
        ).scalars().all()

    assert [outbound.text for outbound in sender.sent] == [
        "sensitive generated detail",
        "刚刚的回复可能包含敏感信息，被 QQ 拦截了，无法发送。",
    ]
    assert [message.platform_msg_id for message in stored_messages] == [
        "qq-blocked-1",
        "bot-reply-qq-blocked-1",
        "bot-reply-notice-qq-blocked-1",
    ]
    blocked = stored_messages[1]
    assert blocked.raw_json["delivery_state"] == "blocked"
    assert blocked.raw_json["failure_kind"] == "qq_sensitive_content"
    assert blocked.raw_json["delivery_reason"] == "wait_for_self_echo_timeout"
    assert blocked.raw_json["delivery_attempts"] == 3
    assert blocked.plain_text.startswith("sensitive generated detail")
    assert "以上回复未在 QQ 群中送达" in blocked.plain_text
    assert "后续回答不得复述其中的敏感细节" in blocked.plain_text
    assert stored_messages[2].raw_json["delivery_state"] == "sent"

    router.runtime.settings.context_recent_limit = 1
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="qq-blocked-2",
            plain_text="@Mira sensitive continue",
        )
    )

    next_prompt = "\n".join(llm.calls[-1])
    assert "sensitive generated detail" in next_prompt
    assert "以上回复未在 QQ 群中送达" in next_prompt
    assert "Do not repeat sensitive details from replies marked as blocked by QQ" in next_prompt


@pytest.mark.asyncio
async def test_router_does_not_recurse_when_qq_block_notice_is_also_blocked(sqlite_engine) -> None:
    sender = AlwaysQQBlockingSender()
    llm = LongReplyLlm("sensitive generated detail")
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(group_id=10001, mentioned_bot=True, message_id="qq-blocked-twice", plain_text="@Mira explain")
    )

    with session_scope(sqlite_engine) as session:
        stored_messages = session.execute(
            select(Message).where(Message.group_id == 10001).order_by(Message.id)
        ).scalars().all()

    assert [outbound.text for outbound in sender.sent] == [
        "sensitive generated detail",
        "刚刚的回复可能包含敏感信息，被 QQ 拦截了，无法发送。",
    ]
    assert [message.platform_msg_id for message in stored_messages] == [
        "qq-blocked-twice",
        "bot-reply-qq-blocked-twice",
    ]
    assert stored_messages[1].raw_json["delivery_state"] == "blocked"


@pytest.mark.asyncio
async def test_router_sends_local_fallback_when_direct_reply_generation_fails(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FailingReplyLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(group_id=10001, mentioned_bot=True, message_id="llm-fallback-1", plain_text="@Mira 在吗")
    )

    assert len(llm.calls) == 1
    assert [outbound.text for outbound in sender.sent] == ["我这边刚刚卡了一下，结果没拿到。你再叫我一次，我马上接上。"]


@pytest.mark.asyncio
async def test_router_concurrent_duplicate_delivery_sends_once(sqlite_engine) -> None:
    sender = BlockingSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    event = make_event(group_id=10001, mentioned_bot=True, message_id="dup-concurrent-1")

    first_task = asyncio.create_task(router.handle_group_message(event))
    await sender.first_send_started.wait()

    second_task = asyncio.create_task(router.handle_group_message(event))
    await asyncio.sleep(0)

    sender.release_first_send.set()
    await asyncio.gather(first_task, second_task)

    with session_scope(sqlite_engine) as session:
        stored_messages = session.execute(
            select(Message).where(Message.group_id == 10001).order_by(Message.id)
        ).scalars().all()

    assert len(sender.calls) == 1
    assert len(llm.calls) == 1
    assert [message.plain_text for message in stored_messages] == ["@Mira hi", "I am here."]


@pytest.mark.asyncio
async def test_router_hides_reserved_outbound_from_recent_messages_and_cooldown(sqlite_engine) -> None:
    sender = BlockingSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        MessageRepository(session).add_group_message(
            platform_msg_id="inflight-seed-1",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 5, 9, 11, 59, 30, tzinfo=UTC),
            plain_text="recent setup",
            raw_json={"post_type": "message", "message_id": "inflight-seed-1"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    first_event = make_event(group_id=10001, mentioned_bot=True, message_id="inflight-1")
    second_event = make_event(
        group_id=10001,
        mentioned_bot=False,
        message_id="inflight-2",
        plain_text="How are you?",
    )

    first_task = asyncio.create_task(router.handle_group_message(first_event))
    await sender.first_send_started.wait()

    await router.handle_group_message(second_event)

    assert len(sender.calls) == 2
    assert len(llm.calls) == 2
    assert "I am here." not in "\n".join(llm.calls[1])
    assert llm.calls[1][3].startswith("Reply style: ")
    assert llm.calls[1][4] == "Recent messages:\nAlice: recent setup\nAlice: @Mira hi\nAlice: How are you?"

    sender.release_first_send.set()
    await first_task


@pytest.mark.asyncio
async def test_router_uses_complete_chronological_history_when_group_policy_enables_it(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    router.runtime.group_policy["groups"]["10001"]["long_context_history"] = True

    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        users.upsert_user(user_id=123456789, nickname="Mira", group_card="")
        messages.add_group_message(
            platform_msg_id="history-early",
            group_id=10001,
            user_id=20001,
            timestamp=datetime(2026, 5, 9, 11, 0, tzinfo=UTC),
            plain_text="earliest message",
            raw_json={"sender": {"nickname": "Alice", "card": "Group Alice"}},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="history-bot",
            group_id=10001,
            user_id=123456789,
            timestamp=datetime(2026, 5, 9, 11, 30, tzinfo=UTC),
            plain_text="middle bot reply",
            raw_json={"direction": "outbound", "delivery_state": "sent"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="history-latest",
            plain_text="@Mira latest message",
            timestamp=datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
        )
    )

    history_section = next(
        line for line in llm.calls[-1] if line.startswith("Full group conversation history (chronological")
    )
    assert history_section.index("20001: earliest message") < history_section.index("123456789: middle bot reply")
    assert "Participants (group-local display names; messages below remain in timestamp/id order):" in history_section
    assert "20001=Group Alice" in history_section
    assert "@Mira latest message" not in history_section


@pytest.mark.asyncio
async def test_router_marks_history_as_partial_when_it_exceeds_model_input_budget(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    router.runtime.group_policy["groups"]["10001"]["long_context_history"] = True
    router.runtime.settings.llm_context_window_tokens = 300
    router.runtime.settings.llm_max_output_tokens = 40

    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        for index in range(3):
            messages.add_group_message(
                platform_msg_id=f"long-history-{index}",
                group_id=10001,
                user_id=20001,
                timestamp=datetime(2026, 5, 9, 10, index, tzinfo=UTC),
                plain_text="old " + ("word " * 20),
                raw_json={},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="long-history-target",
            plain_text="@Mira latest",
            timestamp=datetime(2026, 5, 9, 12, tzinfo=UTC),
        )
    )

    history_section = next(line for line in llm.calls[-1] if line.startswith("Recent contiguous group history"))
    assert "older records exceed the configured model window" in history_section




@pytest.mark.asyncio
async def test_router_uses_bounded_relevant_older_history_when_full_history_is_disabled(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    router.runtime.group_policy["groups"]["10001"]["long_context_history"] = False

    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        messages.add_group_message(
            platform_msg_id="older-messi", group_id=10001, user_id=20001,
            timestamp=datetime(2026, 5, 9, 10, 0, tzinfo=UTC),
            plain_text="\u6885\u897f\u5728\u5df4\u8428\u65f6\u671f\u8e22\u5f97\u7279\u522b\u597d\u3002", raw_json={}, msg_type="text",
            reply_to_msg_id=None, mentioned_bot=False,
        )
        for index in range(65):
            messages.add_group_message(
                platform_msg_id=f"recent-unrelated-{index}", group_id=10001, user_id=20001,
                timestamp=datetime(2026, 5, 9, 11, index % 60, tzinfo=UTC),
                plain_text=f"unrelated chat {index}", raw_json={}, msg_type="text",
                reply_to_msg_id=None, mentioned_bot=False,
            )

    await router.handle_group_message(
        make_event(
            group_id=10001, mentioned_bot=True, message_id="bounded-history-target",
            plain_text="@Mira \u6885\u897f\u5728\u5df4\u8428\u600e\u4e48\u6837", timestamp=datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
        )
    )

    prompt = "\n".join(llm.calls[-1])
    assert "Full group conversation history" not in prompt
    assert "Relevant earlier group messages" in prompt
    assert "\u6885\u897f\u5728\u5df4\u8428\u65f6\u671f\u8e22\u5f97\u7279\u522b\u597d\u3002" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("v2_enabled", "shadow_mode", "expected_marker", "expected_calls"),
    [
        (False, False, "legacy-marker", ["legacy"]),
        (True, True, "legacy-marker", ["legacy", "shadow:orchestrated-memory-target"]),
        (True, False, "v2-marker", ["v2"]),
    ],
)
async def test_router_uses_memory_orchestrator_for_disabled_shadow_and_active_modes(
    sqlite_engine,
    v2_enabled,
    shadow_mode,
    expected_marker,
    expected_calls,
) -> None:
    calls: list[str] = []
    provider_requests: list[object] = []

    def result(request, marker: str, mode: str) -> MemoryContextResult:
        if mode == "v2":
            packed_context = PackedMemoryContext(
                mode="normal",
                budget=100,
                estimated_tokens=2,
                text=marker,
            )
        else:
            packed_context = LegacyMemoryPromptContext(
                recent_messages=[],
                full_history_messages=[],
                full_history_preamble=[],
                full_history_enabled=False,
                member_focus_lines=[],
                summaries=[],
                relevant_history_messages=[marker],
                memories=[],
                history_detail=False,
            )
        return MemoryContextResult(
            group_id=request.group_id,
            packed_context=packed_context,
            selected_source_msg_ids=(f"{mode}-source",),
            estimated_tokens=2,
            mode=mode,
        )

    def v2_provider(request):
        calls.append("v2")
        provider_requests.append(request)
        return result(request, "v2-marker", "v2")

    def legacy_provider(request):
        calls.append("legacy")
        provider_requests.append(request)
        return result(request, "legacy-marker", "v1")

    orchestrator = MemoryOrchestrator(
        v2_enabled=v2_enabled,
        shadow_mode=shadow_mode,
        v2_provider=v2_provider,
        legacy_provider=legacy_provider,
        recent_provider=lambda request: calls.append("recent") or result(request, "recent-marker", "recent"),
        shadow_enqueue=lambda request: calls.append(f"shadow:{request.current_msg_id}"),
    )
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        memory_orchestrator=orchestrator,
    )
    captured_context_builds: list[dict] = []
    original_build = router.context_builder.build

    def capture_context_build(**kwargs):
        captured_context_builds.append(kwargs)
        return original_build(**kwargs)

    router.context_builder.build = capture_context_build

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="orchestrated-memory-target",
            plain_text="@Mira remember this",
        )
    )

    prompt = "\n".join(llm.calls[-1])
    assert expected_marker in prompt
    assert ({"legacy-marker", "v2-marker", "recent-marker"} - {expected_marker}).isdisjoint(prompt.split())
    assert ("Packed memory context" in prompt) is (expected_marker == "v2-marker")
    assert isinstance(
        captured_context_builds[-1]["packed_memory_context"],
        PackedMemoryContext,
    ) is (expected_marker == "v2-marker")
    assert provider_requests[0].query == "@Mira remember this"
    assert provider_requests[0].target_message_id == "orchestrated-memory-target"
    assert all(
        message.group_id == 10001 for message in provider_requests[0].recent_messages
    )
    assert provider_requests[0].available_input > 0
    assert calls == expected_calls


@pytest.mark.asyncio
async def test_router_fallback_marks_delivered_reply_visible_for_cooldown(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    original_mark = InboundRouter._mark_outbound_reply_sent
    state = {"calls": 0}

    def fail_once(self, event, reply_text: str) -> None:
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("mark failed once")
        original_mark(self, event, reply_text)

    monkeypatch.setattr(InboundRouter, "_mark_outbound_reply_sent", fail_once)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="mark-fallback-1",
            timestamp=datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
        )
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="mark-fallback-2",
            plain_text="How are you?",
            timestamp=datetime(2026, 5, 9, 12, 0, 30, tzinfo=UTC),
        )
    )

    with session_scope(sqlite_engine) as session:
        stored_messages = session.execute(
            select(Message).where(Message.group_id == 10001).order_by(Message.id)
        ).scalars().all()
        last_reply_at = MessageRepository(session).last_bot_reply_at(
            group_id=10001,
            bot_user_id=123456789,
        )

    assert state["calls"] == 1
    assert [outbound.text for outbound in sender.sent] == ["I am here."]
    assert len(llm.calls) == 1
    assert [message.plain_text for message in stored_messages] == ["@Mira hi", "I am here.", "How are you?"]
    assert last_reply_at == datetime(2026, 5, 9, 12, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_router_persists_inbound_message_before_llm_call(sqlite_engine) -> None:
    sender = FakeSender()
    llm = InspectingLlm(sqlite_engine=sqlite_engine)
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(make_event(group_id=10001, mentioned_bot=True, message_id="m-before-llm"))

    assert llm.seen_messages_at_call == ["@Mira hi"]


@pytest.mark.asyncio
async def test_router_persists_inbound_message_before_reply_policy(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    router.reply_policy = InspectingReplyPolicy(sqlite_engine=sqlite_engine)

    await router.handle_group_message(make_event(group_id=10001, mentioned_bot=True, message_id="m-before-policy"))

    assert router.reply_policy.seen_messages_at_decide == ["@Mira hi"]


@pytest.mark.asyncio
async def test_router_applies_private_group_allow_command(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
    )

    await router.handle_private_command(sender_qq=987654321, raw_text="/bot group allow 30003")
    await router.handle_group_message(make_event(group_id=30003, mentioned_bot=True, message_id="m-allow"))

    with session_scope(sqlite_engine) as session:
        group = GroupRepository(session).get_group(30003)

    assert group is not None
    assert group.speak_enabled is True
    assert [outbound.text for outbound in sender.sent] == ["I am here."]
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_router_stays_silent_in_non_allowlisted_group(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(make_event(group_id=20002, mentioned_bot=True))

    with session_scope(sqlite_engine) as session:
        group = GroupRepository(session).get_group(20002)
        stored_messages = session.execute(
            select(Message).where(Message.group_id == 20002).order_by(Message.id)
        ).scalars().all()

    assert sender.sent == []
    assert llm.calls == []
    assert group is not None
    assert group.speak_enabled is False
    assert [message.plain_text for message in stored_messages] == ["@Mira hi"]


@pytest.mark.asyncio
async def test_router_honors_proactive_disabled_for_non_guaranteed_question(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    router.runtime.group_policy["groups"]["10001"]["proactive_reply"] = False

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="m-proactive-disabled",
            plain_text="How are you?",
        )
    )

    assert sender.sent == []
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_honors_quiet_hours_for_non_guaranteed_question(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    router.runtime.group_policy["groups"]["10001"]["quiet_hours"] = "01:00-23:00"

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="m-quiet-hours",
            plain_text="How are you?",
        )
    )

    assert sender.sent == []
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_uses_recent_minute_traffic_for_proactive_reply(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        for index in range(12):
            MessageRepository(session).add_group_message(
                platform_msg_id=f"old-traffic-{index}",
                group_id=10001,
                user_id=20001,
                timestamp=datetime(2026, 5, 9, 10, 0, index, tzinfo=UTC),
                plain_text=f"old message {index}",
                raw_json={"post_type": "message", "message_id": f"old-traffic-{index}"},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )
        for index, second in enumerate((20, 40), start=1):
            MessageRepository(session).add_group_message(
                platform_msg_id=f"recent-traffic-{index}",
                group_id=10001,
                user_id=20001,
                timestamp=datetime(2026, 5, 9, 11, 59, second, tzinfo=UTC),
                plain_text=f"recent message {index}",
                raw_json={"post_type": "message", "message_id": f"recent-traffic-{index}"},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(False, "none", 0),
    )
    monkeypatch.setattr(router_module, "is_time_sensitive_request", lambda text: text == "最近这动画更新了")

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="proactive-minute-1",
            plain_text="最近这动画更新了",
            timestamp=datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
        )
    )

    assert sender.sent == []
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_keeps_full_proactive_reply_content_in_one_message(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = LongReplyLlm("是啊，半小时制这个设定一出来，瞬间从小贵升级成抢钱。真打两小时的话，钱包先累趴了。")
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    router.reply_policy = AlwaysProactiveReplyPolicy()

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(False, "none", 0),
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="proactive-short-1",
            plain_text="这也太贵了吧",
        )
    )

    assert [outbound.text for outbound in sender.sent] == [
        "是啊，半小时制这个设定一出来，瞬间从小贵升级成抢钱。真打两小时的话，钱包先累趴了。"
    ]
    assert any("8-16 Chinese characters" in line for line in llm.calls[0] if line.startswith("Reply style: "))
    assert any("Do not rely on later truncation" in line for line in llm.calls[0] if line.startswith("Reply style: "))


@pytest.mark.asyncio
async def test_router_treats_unaddressed_direct_trigger_as_short_group_interjection(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = LongReplyLlm(
        "那日本人来一碗杨国福，算扯平了。本质上就是互相把对方的日常饭当异国体验项目。Komachi给饮食全球化但钱包一起受苦打高分。"
    )
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    router.reply_policy = AlwaysDirectTriggerReplyPolicy()

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(False, "none", 0),
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="direct-trigger-short-1",
            plain_text="日本人把杨国福当中餐体验项目了",
        )
    )

    assert [outbound.text for outbound in sender.sent] == [
        "那日本人来一碗杨国福，算扯平了。本质上就是互相把对方的日常饭当异国体验项目。Komachi给饮食全球化但钱包一起受苦打高分。"
    ]
    assert any("one complete short sentence" in line for line in llm.calls[0] if line.startswith("Reply style: "))
    assert any("Do not rely on later truncation" in line for line in llm.calls[0] if line.startswith("Reply style: "))


@pytest.mark.asyncio
async def test_router_persists_bot_reply_for_cooldown_state(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(make_event(group_id=10001, mentioned_bot=True, message_id="m-2"))

    with session_scope(sqlite_engine) as session:
        stored_messages = session.execute(
            select(Message).where(Message.group_id == 10001).order_by(Message.id)
        ).scalars().all()
        last_reply_at = MessageRepository(session).last_bot_reply_at(
            group_id=10001,
            bot_user_id=123456789,
        )

    assert [message.plain_text for message in stored_messages] == ["@Mira hi", "I am here."]
    assert last_reply_at == datetime(2026, 5, 9, 12, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_router_uses_search_for_addressed_time_sensitive_turn(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = SearchAwareLlm()
    search_client = FakeSearchClient()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
    )

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(True, "named_bot", 10),
    )
    monkeypatch.setattr(router_module, "is_time_sensitive_request", lambda text: text == "need-search")

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="search-1",
            plain_text="need-search",
        )
    )

    assert search_client.queries == [("latest anime buzz", 3)]
    assert any(line.startswith("Web results:\n") for line in llm.calls[-1])
    assert any(line.startswith("Web pages:\n") for line in llm.calls[-1])
    assert [outbound.text for outbound in sender.sent] == ["I checked just enough."]


@pytest.mark.asyncio
async def test_router_normalizes_relative_year_queries_before_search(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = RelativeYearSearchLlm()
    search_client = FakeSearchClient()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
    )

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(True, "named_bot", 10),
    )
    monkeypatch.setattr(router_module, "is_time_sensitive_request", lambda text: text == "今年欧冠冠军是谁")

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="search-relative-year-1",
            plain_text="今年欧冠冠军是谁",
            timestamp=datetime(2026, 5, 9, 15, 15, 48, tzinfo=UTC),
        )
    )

    assert llm.search_decision_calls == 1
    assert search_client.queries == [("2026年欧冠冠军 当前结果", 3)]
    assert any(
        line == "Current local date: 2026-05-09. Resolve relative time words like 今天、今年、明年、去年 against this date."
        for line in llm.calls[0]
    )
    assert [outbound.text for outbound in sender.sent] == ["I checked just enough."]


@pytest.mark.asyncio
async def test_router_logs_search_audit_trail(sqlite_engine, monkeypatch, caplog) -> None:
    sender = FakeSender()
    llm = SearchAwareLlm()
    search_client = FakeSearchClient()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
    )

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(True, "named_bot", 10),
    )
    monkeypatch.setattr(router_module, "is_time_sensitive_request", lambda text: text == "need-search")

    with caplog.at_level(logging.INFO):
        await router.handle_group_message(
            make_event(
                group_id=10001,
                mentioned_bot=False,
                message_id="search-log-1",
                plain_text="need-search",
            )
        )

    assert "reply_decision group_id=10001 msg_id=search-log-1 should_reply=True reason=direct_trigger" in caplog.text
    assert "web_search_decision group_id=10001 msg_id=search-log-1 should_search=True query=latest anime buzz" in caplog.text
    assert "web_search_execute group_id=10001 msg_id=search-log-1 query=latest anime buzz result_count=1" in caplog.text
    assert "web_page_fetch group_id=10001 msg_id=search-log-1 fetched_count=1" in caplog.text
    assert "reply_send_success group_id=10001 msg_id=search-log-1" in caplog.text


@pytest.mark.asyncio
async def test_router_skips_search_for_non_time_sensitive_addressed_turn(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = SearchAwareLlm()
    search_client = FakeSearchClient()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
    )

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(True, "named_bot", 10),
    )
    monkeypatch.setattr(router_module, "is_time_sensitive_request", lambda text: text == "need-search")

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="search-2",
            plain_text="not-time-sensitive",
        )
    )

    assert llm.search_decision_calls == 0
    assert search_client.queries == []
    assert not any(line.startswith("Web results:\n") for line in llm.calls[-1])
    assert [outbound.text for outbound in sender.sent] == ["I checked just enough."]


@pytest.mark.asyncio
async def test_router_uses_search_for_explicit_search_request_without_time_sensitive_topic(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = SearchAwareLlm()
    search_client = FakeSearchClient()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
    )

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(True, "named_bot", 10),
    )
    monkeypatch.setattr(router_module, "is_time_sensitive_request", lambda text: False)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="search-explicit-1",
            plain_text="联网搜索aqua",
        )
    )

    assert llm.search_decision_calls == 0
    assert search_client.queries == [("aqua", 3)]
    assert [outbound.text for outbound in sender.sent] == ["I checked just enough."]


@pytest.mark.asyncio
async def test_router_uses_search_for_reference_topic_without_explicit_search_words(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = SearchAwareLlm()
    search_client = FakeSearchClient()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
    )

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(True, "named_bot", 10),
    )
    monkeypatch.setattr(router_module, "is_time_sensitive_request", lambda text: False)
    monkeypatch.setattr(router_module, "needs_reference_search", lambda text: text == "need-search")

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="search-reference-1",
            plain_text="need-search",
        )
    )

    assert llm.search_decision_calls == 0
    assert search_client.queries == [("need-search", 5)]
    assert [outbound.text for outbound in sender.sent] == ["I checked just enough."]


@pytest.mark.asyncio
async def test_router_uses_search_for_location_recommendation_without_explicit_search_words(
    sqlite_engine, monkeypatch
) -> None:
    sender = FakeSender()
    llm = SearchAwareLlm()
    search_client = FakeSearchClient()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
    )

    monkeypatch.setattr(
        router_module,
        "is_time_sensitive_request",
        lambda text: False,
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="search-local-1",
            plain_text="西安电子科技大学南校区那家店最好吃",
        )
    )

    assert llm.search_decision_calls == 0
    assert search_client.queries == [("西安电子科技大学南校区那家店最好吃", 5)]
    assert [outbound.text for outbound in sender.sent] == ["I checked just enough."]


@pytest.mark.asyncio
async def test_router_allows_search_decision_for_general_knowledge_question(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = SearchAwareLlm()
    search_client = FakeSearchClient()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
    )

    monkeypatch.setattr(
        router_module,
        "is_time_sensitive_request",
        lambda text: False,
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="search-knowledge-1",
            plain_text="水的沸点是多少",
        )
    )

    assert llm.search_decision_calls == 1
    assert search_client.queries == []
    assert [outbound.text for outbound in sender.sent] == ["I checked just enough."]


@pytest.mark.asyncio
async def test_router_includes_grounding_note_for_local_lookup_without_exact_floor_evidence(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    search_client = StaticSearchClient(
        results=[
            type(
                "Result",
                (),
                {
                    "title": "西电南校区附近美食",
                    "snippet": "整理了附近常见吃饭去处。",
                    "source": "https://example.test/list",
                    "date": "2026-05-09",
                },
            )()
        ],
        pages=[
            type(
                "Page",
                (),
                {
                    "title": "南校区附近美食攻略",
                    "url": "https://example.test/list",
                    "content": "附近常见选择有面食、烧烤和简餐，适合晚饭和夜宵。",
                },
            )()
        ],
    )
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
    )

    monkeypatch.setattr(router_module, "is_time_sensitive_request", lambda text: False)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="grounding-local-1",
            plain_text="西安电子科技大学南校区那家店最好吃，我想不到晚上吃什么直接帮我推荐一个",
        )
    )

    assert any(line.startswith("Grounding notes:\n") for line in llm.calls[-1])
    assert any("do not claim an exact floor" in line.lower() for line in llm.calls[-1])


@pytest.mark.asyncio
async def test_router_includes_correction_grounding_note_when_new_evidence_conflicts_with_prior_bot_reply(
    sqlite_engine,
) -> None:
    sender = FakeSender()
    llm = LocalLookupPromptInspectingLlm()
    search_client = StaticSearchClient(
        results=[
            type(
                "Result",
                (),
                {
                    "title": "竹园餐厅二楼牛肉拉面",
                    "snippet": "档口位于竹园餐厅二楼。",
                    "source": "https://example.test/map",
                    "date": "2026-05-09",
                },
            )()
        ],
        pages=[
            type(
                "Page",
                (),
                {
                    "title": "地图详情",
                    "url": "https://example.test/map",
                    "content": "牛肉拉面位于竹园餐厅二楼，靠近楼梯口。",
                },
            )()
        ],
    )
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
    )

    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        users.upsert_user(user_id=123456789, nickname="Mira", group_card="")
        messages.add_group_message(
            platform_msg_id="prior-bot-floor-1",
            group_id=10001,
            user_id=123456789,
            timestamp=datetime(2026, 5, 9, 11, 59, 0, tzinfo=UTC),
            plain_text="去吃竹园三层的牛肉拉面。",
            raw_json={"direction": "outbound", "delivery_state": "sent"},
            msg_type="text",
            reply_to_msg_id="earlier-user-msg",
            mentioned_bot=False,
        )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="grounding-correction-1",
            plain_text="这个牛肉拉面具体在哪里",
            timestamp=datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
        )
    )

    assert llm.search_decision_calls == 1
    assert any(line.startswith("Grounding notes:\n") for line in llm.calls[-1])
    assert any("previous reply said 3层" in line for line in llm.calls[-1])
    assert any("current evidence points to 2层" in line for line in llm.calls[-1])


@pytest.mark.asyncio
async def test_router_forces_search_for_reference_topic_without_llm_opt_out(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = SearchAwareLlm()
    search_client = FakeSearchClient()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
    )

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(True, "named_bot", 10),
    )
    monkeypatch.setattr(router_module, "is_time_sensitive_request", lambda text: False)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="search-reference-force-1",
            plain_text="@Mira 重新客观评价一下上伊那牡丹，醉姿如百合到底是不是一部好作品",
        )
    )

    assert llm.search_decision_calls == 0
    assert llm.reply_calls == 1
    assert len(search_client.queries) == 1
    assert search_client.queries[0][1] == 5
    assert "上伊那牡丹" in search_client.queries[0][0]
    assert "醉姿如百合" in search_client.queries[0][0]
    assert [outbound.text for outbound in sender.sent] == ["I checked just enough."]


@pytest.mark.asyncio
async def test_router_logs_proactive_reply_decision(sqlite_engine, monkeypatch, caplog) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=None,
    )

    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        for index, second in enumerate((20, 40), start=1):
            MessageRepository(session).add_group_message(
                platform_msg_id=f"proactive-log-seed-{index}",
                group_id=10001,
                user_id=20001,
                timestamp=datetime(2026, 5, 9, 11, 59, second, tzinfo=UTC),
                plain_text=f"recent topic {index}",
                raw_json={"post_type": "message", "message_id": f"proactive-log-seed-{index}"},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(False, "none", 0),
    )
    monkeypatch.setattr(router_module, "is_time_sensitive_request", lambda text: text == "最新动画怎么样")

    with caplog.at_level(logging.INFO):
        await router.handle_group_message(
            make_event(
                group_id=10001,
                mentioned_bot=False,
                message_id="proactive-log-1",
                plain_text="最新动画怎么样",
            )
        )

    assert "reply_decision group_id=10001 msg_id=proactive-log-1 should_reply=False reason=below_threshold" in caplog.text
    assert "reply_send_success group_id=10001 msg_id=proactive-log-1" not in caplog.text


@pytest.mark.asyncio
async def test_router_replies_when_search_client_raises(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = SearchAwareLlm()
    search_client = FailingSearchClient()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
    )

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(True, "named_bot", 10),
    )
    monkeypatch.setattr(router_module, "is_time_sensitive_request", lambda text: text == "need-search")

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="search-failure-1",
            plain_text="need-search",
        )
    )

    assert search_client.queries == [("latest anime buzz", 3)]
    assert llm.search_decision_calls == 1
    assert llm.reply_calls == 1
    assert not any(line.startswith("Web results:\n") for line in llm.calls[-1])
    assert [outbound.text for outbound in sender.sent] == ["I checked just enough."]


@pytest.mark.asyncio
async def test_router_logs_reply_send_failure(sqlite_engine, caplog) -> None:
    sender = FailingSenderOnce()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError, match="send failed"):
            await router.handle_group_message(make_event(group_id=10001, mentioned_bot=True, message_id="send-log-1"))

    assert "reply_decision group_id=10001 msg_id=send-log-1 should_reply=True reason=direct_trigger" in caplog.text
    assert "reply_send_failed group_id=10001 msg_id=send-log-1" in caplog.text


@pytest.mark.asyncio
async def test_router_replies_when_search_decision_generation_raises(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = SearchDecisionFailingLlm()
    search_client = FakeSearchClient()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
    )

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(True, "named_bot", 10),
    )
    monkeypatch.setattr(router_module, "is_time_sensitive_request", lambda text: text == "need-search")

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="search-decision-failure-1",
            plain_text="need-search",
        )
    )

    assert llm.search_decision_calls == 1
    assert llm.reply_calls == 1
    assert search_client.queries == []
    assert [outbound.text for outbound in sender.sent] == ["Still replying."]


@pytest.mark.asyncio
async def test_router_injects_runtime_clock_facts_for_date_question_without_search_decision(
    sqlite_engine, monkeypatch
) -> None:
    sender = FakeSender()
    llm = RuntimeFactsInspectingLlm()
    search_client = FakeSearchClient()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
    )

    monkeypatch.setattr(
        router_module,
        "build_current_datetime_facts",
        lambda now: [
            "Current local datetime: 2026-05-09 13:26:00 +08:00",
            "Current local date: 2026-05-09",
            "Current local weekday: Saturday",
        ],
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="clock-1",
            plain_text="@Mira 今天几号",
        )
    )

    assert llm.search_decision_calls == 0
    assert llm.reply_calls == 1
    assert search_client.queries == []
    assert any(line.startswith("Runtime facts:\n") for line in llm.calls[-1])
    assert [outbound.text for outbound in sender.sent] == ["今天是 2026-05-09。"]


@pytest.mark.asyncio
async def test_router_injects_runtime_clock_facts_for_current_year_question_without_search(
    sqlite_engine, monkeypatch
) -> None:
    sender = FakeSender()
    llm = RuntimeFactsInspectingLlm()
    search_client = FakeSearchClient()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=search_client,
    )

    monkeypatch.setattr(
        router_module,
        "build_current_datetime_facts",
        lambda now: [
            "Current local datetime: 2026-05-09 15:52:50 +08:00",
            "Current local date: 2026-05-09",
            "Current local weekday: Saturday",
        ],
    )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="clock-year-1",
            plain_text="@Mira 今年是几几年",
            timestamp=datetime(2026, 5, 9, 15, 52, 50, tzinfo=UTC),
        )
    )

    assert llm.search_decision_calls == 0
    assert llm.reply_calls == 1
    assert search_client.queries == []
    assert any(line.startswith("Runtime facts:\n") for line in llm.calls[-1])
    assert [outbound.text for outbound in sender.sent] == ["今天是 2026-05-09。"]


@pytest.mark.asyncio
async def test_router_skips_search_decision_call_when_no_search_client(sqlite_engine, monkeypatch) -> None:
    sender = FakeSender()
    llm = SearchAwareLlm()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=sender,
        llm_client=llm,
        web_search_client=None,
    )

    monkeypatch.setattr(
        router_module,
        "detect_address_intent",
        lambda **kwargs: AddressDecision(True, "named_bot", 10),
    )
    monkeypatch.setattr(router_module, "is_time_sensitive_request", lambda text: text == "need-search")

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="search-none-1",
            plain_text="need-search",
        )
    )

    assert llm.search_decision_calls == 0
    assert llm.reply_calls == 1
    assert [outbound.text for outbound in sender.sent] == ["I checked just enough."]


@pytest.mark.asyncio
async def test_router_persists_inline_summary_and_memories_without_breaking_reply_flow(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=20001, nickname="Alice", group_card="")
        for index in range(1, 25):
            MessageRepository(session).add_group_message(
                platform_msg_id=f"seed-{index}",
                group_id=10001,
                user_id=20001,
                timestamp=datetime(2026, 5, 9, 11, 0, index, tzinfo=UTC),
                plain_text=f"seed message {index}",
                raw_json={"post_type": "message", "message_id": f"seed-{index}"},
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="summary-25",
            plain_text="I like hotpot.",
        )
    )

    with session_scope(sqlite_engine) as session:
        summaries = session.execute(select(Summary).order_by(Summary.id)).scalars().all()
        memories = session.execute(select(MemoryItem).order_by(MemoryItem.id)).scalars().all()

    assert [outbound.text for outbound in sender.sent] == ["I am here."]
    assert len(llm.calls) == 1
    assert [summary.summary_level for summary in summaries] == ["window", "daily"]
    assert summaries[0].source_count == 25
    assert summaries[0].content.startswith("Recent chat summary:")
    assert summaries[0].source_start_msg_id == "seed-1"
    assert summaries[0].source_end_msg_id == "summary-25"
    assert summaries[1].summary_key == "daily:2026-05-09"
    assert summaries[1].content.startswith("Rolling group memory:")
    assert [memory.subject_id for memory in memories] == ["20001"]
    assert [memory.content for memory in memories] == ["Alice likes hotpot."]


@pytest.mark.asyncio
async def test_router_persists_explicit_plan_as_source_backed_current_memory(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="plan-memory-1",
            plain_text="I plan to visit Shanghai next week.",
        )
    )

    with session_scope(sqlite_engine) as session:
        memory = session.execute(
            select(MemoryItem).where(MemoryItem.source_msg_id == "plan-memory-1", MemoryItem.memory_kind == "plan")
        ).scalar_one()

    assert memory.status == "active"
    assert memory.subject_id == "20001"
    assert memory.content == "Alice: I plan to visit Shanghai next week."
    assert memory.valid_from is not None


@pytest.mark.asyncio
async def test_router_supersedes_cancelled_plan_instead_of_prompting_both_versions(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="plan-before-cancel",
            plain_text="I plan to visit Shanghai next week.",
        )
    )
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="plan-cancelled",
            plain_text="I cancelled my plan to visit Shanghai next week.",
        )
    )

    with session_scope(sqlite_engine) as session:
        plans = session.execute(
            select(MemoryItem)
            .where(MemoryItem.memory_kind == "plan")
            .order_by(MemoryItem.id)
        ).scalars().all()

    assert [plan.status for plan in plans] == ["superseded", "active"]
    assert plans[0].superseded_by_id == plans[1].id
    assert plans[1].supersedes_id == plans[0].id


@pytest.mark.asyncio
async def test_router_handles_group_weekly_report_command(sqlite_engine) -> None:
    sender = FakeSender()
    llm = WeeklyReportLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="weekly-seed-1",
            plain_text="今天这波真离谱",
            timestamp=datetime(2026, 5, 14, 12, 0, tzinfo=UTC),
            user_id=20001,
            nickname="Alice",
            group_card="Alice",
        )
    )
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="weekly-seed-2",
            plain_text="这也太炸了吧",
            timestamp=datetime(2026, 5, 14, 12, 5, tzinfo=UTC),
            user_id=20002,
            nickname="Bob",
            group_card="",
        )
    )
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="weekly-command",
            plain_text="@Mira 周报",
            timestamp=datetime(2026, 5, 15, 12, 0, tzinfo=UTC),
            user_id=20003,
            nickname="Carol",
            group_card="",
        )
    )

    assert len(sender.sent) == 1
    assert "本群近一周高能雷霆发言周报" in sender.sent[0].text
    assert "第1名" in sender.sent[0].text
    assert llm.conversation_keys == ["group-weekly-report:10001"]


@pytest.mark.asyncio
async def test_router_keeps_full_addressed_reply_content_for_normal_group_reply(sqlite_engine) -> None:
    sender = FakeSender()
    llm = LongReplyLlm("是啊，半小时制这个设定一出来，瞬间从小贵升级成抢钱。真打两小时的话，钱包先累趴了。")
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="addressed-keep-full-1",
            plain_text="@Mira 怎么突然变这么贵",
        )
    )

    assert [outbound.text for outbound in sender.sent] == [
        "是啊，半小时制这个设定一出来，瞬间从小贵升级成抢钱。真打两小时的话，钱包先累趴了。"
    ]


@pytest.mark.asyncio
async def test_router_includes_group_card_and_qq_nickname_in_recent_message_labels(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="label-1",
            plain_text="@Mira 看看我是谁",
            user_id=20001,
            nickname="熟人A",
            group_card="送外卖去了",
        )
    )

    assert any(
        line == "Recent messages:\n送外卖去了（QQ昵称：熟人A）: @Mira 看看我是谁"
        for line in llm.calls[-1]
    )
    assert any(
        line == "Target message: 送外卖去了（QQ昵称：熟人A）: @Mira 看看我是谁"
        for line in llm.calls[-1]
    )


def test_router_format_member_label_keeps_group_card_and_qq_nickname_even_when_same(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)

    assert router._format_member_label(
        nickname="Alice",
        group_card="Alice",
        fallback="20001",
    ) == "Alice\uFF08QQ\u6635\u79F0\uFF1AAlice\uFF09"


@pytest.mark.asyncio
async def test_router_includes_named_member_history_when_asked_to_evaluate_group_member(sqlite_engine) -> None:
    sender = FakeSender()
    llm = FakeLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    router.runtime.settings.context_recent_limit = 2

    with session_scope(sqlite_engine) as session:
        groups = GroupRepository(session)
        users = UserRepository(session)
        messages = MessageRepository(session)
        groups.upsert_group(group_id=10001, group_name="10001", enabled=True, speak_enabled=True)
        users.upsert_user(user_id=10002, nickname="熟人A", group_card="送外卖去了")
        users.upsert_user(user_id=10001, nickname="不知道叫什么", group_card="群友甲")
        users.upsert_user(user_id=20002, nickname="Bob", group_card="")
        messages.add_group_message(
            platform_msg_id="target-old-1",
            group_id=10001,
            user_id=10002,
            timestamp=datetime(2026, 5, 9, 11, 30, tzinfo=UTC),
            plain_text="今天又加班，累麻了。",
            raw_json={"post_type": "message", "message_id": "target-old-1"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="target-old-2",
            group_id=10001,
            user_id=10002,
            timestamp=datetime(2026, 5, 9, 11, 31, tzinfo=UTC),
            plain_text="刚送完最后一单。",
            raw_json={"post_type": "message", "message_id": "target-old-2"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="other-recent-1",
            group_id=10001,
            user_id=20002,
            timestamp=datetime(2026, 5, 9, 11, 58, tzinfo=UTC),
            plain_text="最近动画真不错",
            raw_json={"post_type": "message", "message_id": "other-recent-1"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )
        messages.add_group_message(
            platform_msg_id="other-recent-2",
            group_id=10001,
            user_id=10001,
            timestamp=datetime(2026, 5, 9, 11, 59, tzinfo=UTC),
            plain_text="还没改好吗",
            raw_json={"post_type": "message", "message_id": "other-recent-2"},
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="member-eval-1",
            plain_text="@Mira 如何评价群里叫送外卖去了这个人",
            user_id=10001,
            nickname="不知道叫什么",
            group_card="群友甲",
            timestamp=datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
        )
    )

    assert any(
        line.startswith("Member focus:\nReferenced member: 送外卖去了（QQ昵称：熟人A）")
        for line in llm.calls[-1]
    )
    assert any("送外卖去了（QQ昵称：熟人A）: 今天又加班，累麻了。" in line for line in llm.calls[-1])
    assert any("送外卖去了（QQ昵称：熟人A）: 刚送完最后一单。" in line for line in llm.calls[-1])
@pytest.mark.asyncio
async def test_router_sends_local_fallback_when_image_context_exists_but_vision_is_disabled(sqlite_engine) -> None:
    sender = FakeSender()
    llm = ImageCapturingLlm()
    router = InboundRouter.build_for_test(sqlite_engine=sqlite_engine, sender=sender, llm_client=llm)
    router.runtime.settings.llm_supports_vision_input = False

    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=False,
            message_id="vision-disabled-image-1",
            plain_text="",
            images=[ImageAttachment(url="https://img.example.test/recent.png", file_id="recent.png")],
            timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        )
    )
    await router.handle_group_message(
        make_event(
            group_id=10001,
            mentioned_bot=True,
            message_id="vision-disabled-trigger-1",
            plain_text="这张图在说什么",
            timestamp=datetime(2026, 5, 9, 12, 0, 8, tzinfo=UTC),
        )
    )

    assert [outbound.text for outbound in sender.sent] == ["我这边这路模型现在还看不了图，得换支持识图的模型才行。"]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_router_enqueues_persisted_inbound_message_for_episode_allocation(sqlite_engine) -> None:
    class BackgroundAwareCompaction:
        def __init__(self) -> None:
            self.enqueued: list[dict] = []
            self.wake_count = 0

        def enqueue_episode_allocation(self, **kwargs):
            self.enqueued.append(kwargs)

        async def wake(self) -> None:
            self.wake_count += 1

    service = BackgroundAwareCompaction()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=FakeSender(),
        llm_client=FakeLlm(),
        memory_compaction_service=service,
    )
    event = make_event(
        group_id=10001,
        mentioned_bot=False,
        message_id="episode-enqueue-inbound-1",
        plain_text="普通群消息",
        timestamp=datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
    )

    await router.handle_group_message(event)

    with session_scope(sqlite_engine) as session:
        row = MessageRepository(session).get_by_platform_msg_id(event.platform_msg_id)
        assert row is not None
        expected_id = row.id
    assert service.enqueued == [
        {
                "group_id": 10001,
                "message_id": expected_id,
                "now": event.timestamp,
                "late_arrival": False,
            }
        ]
    assert service.wake_count == 1


@pytest.mark.asyncio
async def test_router_marks_out_of_order_timestamp_as_late_arrival(
    sqlite_engine,
) -> None:
    class BackgroundAwareCompaction:
        def __init__(self) -> None:
            self.enqueued: list[dict] = []

        def enqueue_episode_allocation(self, **kwargs):
            self.enqueued.append(kwargs)

        async def wake(self) -> None:
            return None

    service = BackgroundAwareCompaction()
    router = InboundRouter.build_for_test(
        sqlite_engine=sqlite_engine,
        sender=FakeSender(),
        llm_client=FakeLlm(),
        memory_compaction_service=service,
    )
    newer = make_event(
        group_id=10001,
        mentioned_bot=False,
        message_id="episode-newer",
        plain_text="newer",
        timestamp=datetime(2026, 5, 9, 12, 30, tzinfo=UTC),
    )
    late = make_event(
        group_id=10001,
        mentioned_bot=False,
        message_id="episode-late",
        plain_text="late",
        timestamp=datetime(2026, 5, 9, 12, 10, tzinfo=UTC),
    )

    await router.handle_group_message(newer)
    await router.handle_group_message(late)

    assert service.enqueued[0]["late_arrival"] is False
    assert service.enqueued[1]["late_arrival"] is True
