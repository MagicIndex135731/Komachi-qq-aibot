from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from app.adapters.onebot_models import PrivateMessageEvent
from app.core.message_content import ImageAttachment
import app.dev_control.service as dev_service_module
from app.dev_control.codex_bridge import CodexTaskResult
from app.dev_control.service import DevControlService
from app.storage.db import session_scope
from app.storage.repositories import DevSessionRepository, DevTaskRepository


class FakeSender:
    def __init__(self) -> None:
        self.private_sent = []

    async def send_private_text(self, outbound) -> None:
        self.private_sent.append(outbound)


class SlowSender(FakeSender):
    def __init__(self, delay_seconds: float) -> None:
        super().__init__()
        self.delay_seconds = delay_seconds

    async def send_private_text(self, outbound) -> None:
        await dev_service_module.asyncio.sleep(self.delay_seconds)
        await super().send_private_text(outbound)


class SelectiveFailSender(FakeSender):
    def __init__(self, *, failing_user_ids: set[int]) -> None:
        super().__init__()
        self.failing_user_ids = set(failing_user_ids)

    async def send_private_text(self, outbound) -> None:
        if outbound.user_id in self.failing_user_ids:
            raise RuntimeError(f"send_private_msg failed: target={outbound.user_id}")
        await super().send_private_text(outbound)


class FakeLlmClient:
    def __init__(self, reply_text: str = "fast project reply") -> None:
        self.reply_text = reply_text
        self.prompts: list[list[str]] = []
        self.images_calls: list[list[ImageAttachment] | None] = []
        self.conversation_keys: list[str | None] = []

    def generate_text(self, prompt_lines, *, images=None, conversation_key=None):
        self.prompts.append(list(prompt_lines))
        self.images_calls.append(None if images is None else list(images))
        self.conversation_keys.append(conversation_key)
        return self.reply_text


class SearchAwareFakeLlmClient:
    def __init__(self) -> None:
        self.prompts: list[list[str]] = []
        self.search_decision_calls = 0

    def generate_text(self, prompt_lines, *, images=None, conversation_key=None):
        del images
        del conversation_key
        self.prompts.append(list(prompt_lines))
        joined = "\n".join(prompt_lines)
        if "Reply with exactly three lines in this grammar" in joined:
            self.search_decision_calls += 1
            return "SEARCH: yes\nQUERY: latest anime buzz\nREASON: current-facts-needed"
        return "### 先说结论\n- 我查了下，确实有新消息。"


class IntentRoutingFakeLlmClient:
    def __init__(self, *, intent_reply: str, chat_reply: str = "chat reply") -> None:
        self.intent_reply = intent_reply
        self.chat_reply = chat_reply
        self.prompts: list[list[str]] = []
        self.intent_calls = 0

    def generate_text(self, prompt_lines, *, images=None, conversation_key=None):
        del images
        del conversation_key
        self.prompts.append(list(prompt_lines))
        joined = "\n".join(prompt_lines)
        if 'Reply with exactly one label: EXECUTE, INSPECT, or CHAT.' in joined:
            self.intent_calls += 1
            return self.intent_reply
        return self.chat_reply


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


class FakeCodexBridge:
    def __init__(self, *, result: CodexTaskResult) -> None:
        self.result = result
        self.prompts: list[str] = []
        self.resume_thread_ids: list[str | None] = []

    def run_task(
        self,
        *,
        prompt: str,
        repo_root: Path,
        artifact_dir: Path,
        resume_thread_id: str | None = None,
    ) -> CodexTaskResult:
        self.prompts.append(prompt)
        self.resume_thread_ids.append(resume_thread_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return self.result


class MutatingCodexBridge(FakeCodexBridge):
    def __init__(self, *, result: CodexTaskResult, relative_path: str, content: str) -> None:
        super().__init__(result=result)
        self.relative_path = relative_path
        self.content = content

    def run_task(
        self,
        *,
        prompt: str,
        repo_root: Path,
        artifact_dir: Path,
        resume_thread_id: str | None = None,
    ) -> CodexTaskResult:
        target_path = repo_root / self.relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(self.content, encoding="utf-8")
        return super().run_task(
            prompt=prompt,
            repo_root=repo_root,
            artifact_dir=artifact_dir,
            resume_thread_id=resume_thread_id,
        )


class RaisingCodexBridge:
    def __init__(self, message: str) -> None:
        self.message = message

    def run_task(
        self,
        *,
        prompt: str,
        repo_root: Path,
        artifact_dir: Path,
        resume_thread_id: str | None = None,
    ) -> CodexTaskResult:
        raise FileNotFoundError(self.message)


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


def owner_admin_text(text: str) -> str:
    return f"管理员权限 {text}"


@pytest.mark.asyncio
async def test_owner_private_log_inspection_replies_inline_and_stores_completed_turn(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="I checked the latest logs. There is no new crash.")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "runtime.stderr.log").write_text("line 1\nline 2\nlatest error line\n", encoding="utf-8")
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-1", user_id=987654321, text=owner_admin_text("check logs"))
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == ["I checked the latest logs. There is no new crash."]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert len(completed) == 1
    assert completed[0].intent_type == "project_inspect"
    assert completed[0].result_text == "I checked the latest logs. There is no new crash."
    assert queued == []
    assert llm_client.prompts
    with session_scope(sqlite_engine) as session:
        dev_session = DevSessionRepository(session).get_latest_owner_session(
            owner_qq=987654321,
            session_mode=dev_service_module.SESSION_MODE_PROJECT,
        )
    assert dev_session is not None
    assert llm_client.conversation_keys == [f"dev-session:{dev_session.id}"]


@pytest.mark.asyncio
async def test_owner_private_restart_status_question_uses_project_inspect_instead_of_triggering_restart(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我这边看到进程已经拉起来了，但还要看最新日志才能确认是否完全生效。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    (data_dir / "logs").mkdir(parents=True)
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-restart-status",
            user_id=987654321,
            text=owner_admin_text("已经重启生效了吗"),
        )
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == ["我这边看到进程已经拉起来了，但还要看最新日志才能确认是否完全生效。"]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert len(completed) == 1
    assert completed[0].intent_type == "project_inspect"
    assert queued == []


@pytest.mark.asyncio
async def test_owner_private_restart_status_question_without_admin_prefix_stays_project_chat(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我先按普通私聊理解这句，不直接走仓库检查。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        assistant_name="比企谷小町",
        persona={
            "name": "比企谷小町",
            "identity": "A fixed AI persona modeled after Hikigaya Komachi.",
        },
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-restart-status-chat", user_id=987654321, text="已经重启生效了吗")
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == ["我先按普通私聊理解这句，不直接走仓库检查。"]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert len(completed) == 1
    assert completed[0].intent_type == "project_chat"
    assert completed[0].result_text == "我先按普通私聊理解这句，不直接走仓库检查。"
    assert queued == []


@pytest.mark.asyncio
async def test_owner_private_execute_like_text_without_admin_prefix_stays_project_chat(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="先按普通聊天处理，这条不会直接进仓库执行。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        assistant_name="比企谷小町",
        persona={
            "name": "比企谷小町",
            "identity": "A fixed AI persona modeled after Hikigaya Komachi.",
        },
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-fix", user_id=987654321, text="fix this")
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == ["先按普通聊天处理，这条不会直接进仓库执行。"]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert len(completed) == 1
    assert completed[0].intent_type == "project_chat"
    assert completed[0].result_text == "先按普通聊天处理，这条不会直接进仓库执行。"
    assert queued == []


@pytest.mark.asyncio
async def test_owner_private_admin_execute_request_still_creates_dev_task_and_sends_ack(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    bridge = FakeCodexBridge(result=CodexTaskResult(summary="done", reply_text="ok", restart_required=False))
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
    )
    (tmp_path / "repo").mkdir()

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-1",
            user_id=987654321,
            text=owner_admin_text("把这个功能整体修一下并上线"),
        )
    )

    assert handled is True
    assert sender.private_sent == []
    with session_scope(sqlite_engine) as session:
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert [task.raw_request_text for task in queued] == ["把这个功能整体修一下并上线"]


@pytest.mark.asyncio
async def test_owner_private_featureword_增加_is_classified_as_execute_request(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir()

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-exec-increase",
            user_id=987654321,
            text=owner_admin_text("你去确认，发现能@就直接@他，不行就增加这个功能然后@他什么都不说，并上线"),
        )
    )

    assert handled is True
    assert sender.private_sent == []
    with session_scope(sqlite_engine) as session:
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert len(queued) == 1
    assert queued[0].intent_type == "feature_work"
    assert "增加这个功能" in queued[0].raw_request_text


@pytest.mark.asyncio
async def test_process_next_task_restart_only_restarts_directly_without_codex(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    bridge = FakeCodexBridge(result=CodexTaskResult(summary="done", reply_text="should not run", restart_required=False))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    command_calls: list[list[str]] = []

    def fake_command_runner(command: list[str], cwd: Path):
        del cwd
        command_calls.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
        command_runner=fake_command_runner,
    )

    await service.handle_private_message(
        make_private_event(message_id="p-restart-only", user_id=987654321, text=owner_admin_text("重启一下"))
    )
    processed = await service.process_next_task_once()

    assert processed is True
    assert bridge.prompts == []
    assert len(command_calls) == 2
    assert command_calls[0][-1].endswith("stop_xiaomachi_runtime.ps1")
    assert command_calls[1][-1].endswith("start_xiaomachi_runtime.ps1")
    assert [outbound.text for outbound in sender.private_sent] == [
        "我开始处理这条了。",
        "我现在重启小町，让改动生效。",
        "已经重启完了。",
    ]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 1
    assert completed[0].intent_type == "restart_only"
    assert completed[0].restart_required is True
    assert completed[0].restart_result == "success"
    assert completed[0].result_text == "已经重启完了。"


@pytest.mark.asyncio
async def test_owner_private_ambiguous_action_request_can_be_routed_to_execute_by_llm_classifier(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    llm_client = IntentRoutingFakeLlmClient(intent_reply="EXECUTE")
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir()

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-exec-ambiguous",
            user_id=987654321,
            text=owner_admin_text("去群里@熟人A什么话都不说，持续推进到完成"),
        )
    )

    assert handled is True
    assert llm_client.intent_calls == 1
    assert sender.private_sent == []
    with session_scope(sqlite_engine) as session:
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert len(queued) == 1
    assert queued[0].intent_type == "feature_work"
    assert queued[0].raw_request_text == "去群里@熟人A什么话都不说，持续推进到完成"


@pytest.mark.asyncio
async def test_owner_private_execute_ack_is_not_deduplicated_across_different_tasks(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir()

    handled_first = await service.handle_private_message(
        make_private_event(
            message_id="p-exec-1",
            user_id=987654321,
            text=owner_admin_text("把这个功能整体修一下并上线"),
        )
    )
    handled_second = await service.handle_private_message(
        make_private_event(
            message_id="p-exec-2",
            user_id=987654321,
            text=owner_admin_text("把这个功能整体再修一下并上线"),
        )
    )

    assert handled_first is True
    assert handled_second is True
    assert sender.private_sent == []
    with session_scope(sqlite_engine) as session:
        messages = dev_service_module.MessageRepository(session)
        first = messages.get_by_platform_msg_id("private-outbound-queued_execute_task:1")
        second = messages.get_by_platform_msg_id("private-outbound-queued_execute_task:2")
    assert first is None
    assert second is None


@pytest.mark.asyncio
async def test_owner_private_daily_chat_replies_inline_and_uses_daily_prompt(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="This private chat is now one continuous daily session.")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("project session", encoding="utf-8")
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        assistant_name="比企谷小町",
        persona={
            "name": "比企谷小町",
            "identity": "A fixed AI persona modeled after Hikigaya Komachi.",
        },
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-1", user_id=987654321, text="how does the private project session work")
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == [
        "This private chat is now one continuous daily session."
    ]
    prompt = "\n".join(llm_client.prompts[0])
    assert "Current private daily session summary:" in prompt
    assert "Recent private daily turns:" in prompt
    assert "比企谷小町" in prompt
    assert "Markdown is allowed" in prompt
    assert "local Xiaomachi repository" not in prompt
    assert "Relevant repository snippets:" not in prompt
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert len(completed) == 1
    assert completed[0].intent_type == "project_chat"
    assert completed[0].result_text == "This private chat is now one continuous daily session."
    with sqlite_engine.connect() as connection:
        session_modes = [row[0] for row in connection.execute(text("select session_mode from dev_sessions order by id asc"))]
    assert session_modes == ["daily"]


@pytest.mark.asyncio
async def test_owner_private_daily_chat_passes_dev_session_conversation_key(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="session keyed reply")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-key", user_id=987654321, text="remember this chat")
    )

    assert handled is True
    with session_scope(sqlite_engine) as session:
        dev_session = DevSessionRepository(session).get_latest_owner_session(
            owner_qq=987654321,
            session_mode=dev_service_module.SESSION_MODE_DAILY,
        )
    assert dev_session is not None
    assert llm_client.conversation_keys == [f"dev-session:{dev_session.id}"]


@pytest.mark.asyncio
async def test_owner_private_daily_datetime_question_marks_runtime_facts_as_authoritative(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="现在是 2026 年。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-year", user_id=987654321, text="今年是几几年")
    )

    assert handled is True
    prompt = "\n".join(llm_client.prompts[0])
    assert "Runtime facts:" in prompt
    assert "Current local datetime:" in prompt
    assert "Treat runtime facts as authoritative for the current year, date, weekday, and clock time." in prompt


@pytest.mark.asyncio
async def test_owner_private_daily_chat_passes_current_images_to_llm(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我看到这张图了。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-image",
            user_id=987654321,
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
async def test_owner_private_daily_chat_uses_quoted_private_images(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我看的是你引用的那张图。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    with session_scope(sqlite_engine) as session:
        dev_service_module.UserRepository(session).upsert_user(user_id=987654321, nickname="owner", group_card="")
        dev_service_module.MessageRepository(session).add_private_message(
            platform_msg_id="private-inbound-987654321-p-prev-image",
            user_id=987654321,
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
            user_id=987654321,
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
async def test_owner_private_image_then_immediate_followup_text_only_replies_once(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我看的是你后面接着问的那张图。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        private_image_followup_window_seconds=0.05,
    )

    with session_scope(sqlite_engine) as session:
        dev_service_module.UserRepository(session).upsert_user(user_id=987654321, nickname="owner", group_card="")
        dev_service_module.MessageRepository(session).add_private_message(
            platform_msg_id="private-inbound-987654321-p-chat-image-only",
            user_id=987654321,
            timestamp=datetime(2026, 5, 10, 12, 0, tzinfo=UTC),
            plain_text="",
            raw_json={
                "message_id": "p-chat-image-only",
                "message": [
                    {
                        "type": "image",
                        "data": {
                            "file": "followup-cat.png",
                            "url": "https://img.example.test/followup-cat.png",
                            "local_path": str(tmp_path / "followup-cat.png"),
                        },
                    }
                ],
            },
            msg_type="image",
            reply_to_msg_id=None,
        )

    handled_image = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-image-only",
            user_id=987654321,
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
    handled_text = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-image-followup",
            user_id=987654321,
            text="这是谁",
        )
    )
    await dev_service_module.asyncio.sleep(0.1)

    assert handled_image is True
    assert handled_text is True
    assert [outbound.text for outbound in sender.private_sent] == ["我看的是你后面接着问的那张图。"]
    assert len(llm_client.prompts) == 1
    assert llm_client.images_calls[-1] is not None
    assert llm_client.images_calls[-1][0].file_id == "followup-cat.png"


@pytest.mark.asyncio
async def test_owner_private_contextual_text_followup_reuses_recent_image_without_explicit_image_keyword(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我会继续按刚才那张图来判断。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    with session_scope(sqlite_engine) as session:
        dev_service_module.UserRepository(session).upsert_user(user_id=987654321, nickname="owner", group_card="")
        dev_service_module.MessageRepository(session).add_private_message(
            platform_msg_id="private-inbound-987654321-p-chat-role-image",
            user_id=987654321,
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
            user_id=987654321,
            text="这是魔女裁判游戏里的",
        )
    )

    assert handled is True
    assert llm_client.images_calls[-1] is not None
    assert llm_client.images_calls[-1][0].file_id == "witch-judge.png"


@pytest.mark.asyncio
async def test_owner_private_character_id_image_prompt_adds_vision_guidance(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="这是角色识别回复。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-role-prompt",
            user_id=987654321,
            text="这是哪个角色",
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
    prompt = "\n".join(llm_client.prompts[-1])
    assert "Vision task:" in prompt
    assert "identify the most likely character name and franchise first" in prompt
    assert "If the image is too small, blurry, cropped, stylized, or fan-art-like" in prompt


@pytest.mark.asyncio
async def test_owner_private_character_followup_with_work_hint_adds_non_fabrication_guardrails(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="继续判断")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    with session_scope(sqlite_engine) as session:
        dev_service_module.UserRepository(session).upsert_user(user_id=987654321, nickname="owner", group_card="")
        dev_service_module.MessageRepository(session).add_private_message(
            platform_msg_id="private-inbound-987654321-p-chat-role-image",
            user_id=987654321,
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

    await service.handle_private_message(
        make_private_event(
            message_id="p-chat-work-hint",
            user_id=987654321,
            text="是魔女裁判这个游戏里的人物",
        )
    )
    await service.handle_private_message(
        make_private_event(
            message_id="p-chat-guess-name",
            user_id=987654321,
            text="你直接猜角色名",
        )
    )

    prompt = "\n".join(llm_client.prompts[-1])
    assert "《魔女裁判》" in prompt
    assert "Only name characters that actually belong to 《魔女裁判》." in prompt
    assert "Do not borrow names from other works or invent a new character name." in prompt


@pytest.mark.asyncio
async def test_owner_private_character_followup_with_work_hint_searches_character_list(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="继续判断")
    search_client = FakeSearchClient()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        web_search_client=search_client,
    )

    with session_scope(sqlite_engine) as session:
        dev_service_module.UserRepository(session).upsert_user(user_id=987654321, nickname="owner", group_card="")
        dev_service_module.MessageRepository(session).add_private_message(
            platform_msg_id="private-inbound-987654321-p-chat-role-image",
            user_id=987654321,
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

    await service.handle_private_message(
        make_private_event(
            message_id="p-chat-work-hint",
            user_id=987654321,
            text="是魔女裁判这个游戏里的人物",
        )
    )
    await service.handle_private_message(
        make_private_event(
            message_id="p-chat-guess-name",
            user_id=987654321,
            text="你直接猜角色名",
        )
    )

    assert any("魔女裁判" in query and "角色" in query for query, _max_results in search_client.queries)
    prompt = "\n".join(llm_client.prompts[-1])
    assert "Web search results:" in prompt


@pytest.mark.asyncio
async def test_owner_private_daily_and_admin_turns_use_separate_sessions(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="ok")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    (data_dir / "logs").mkdir(parents=True)
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await service.handle_private_message(
        make_private_event(message_id="p-chat-daily", user_id=987654321, text="今天有点困")
    )
    await service.handle_private_message(
        make_private_event(message_id="p-chat-admin", user_id=987654321, text=owner_admin_text("检查日志"))
    )

    with sqlite_engine.connect() as connection:
        rows = connection.execute(
            text(
                "select owner_qq, session_mode, count(*) "
                "from dev_sessions group by owner_qq, session_mode order by session_mode asc"
            )
        ).fetchall()
    assert rows == [
        (987654321, "daily", 1),
        (987654321, "project", 1),
    ]


@pytest.mark.asyncio
async def test_owner_admin_new_session_command_resets_project_session_only(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="ok")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    (data_dir / "logs").mkdir(parents=True)
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await service.handle_private_message(
        make_private_event(message_id="p-chat-daily", user_id=987654321, text="今天天气不错")
    )
    await service.handle_private_message(
        make_private_event(message_id="p-chat-admin", user_id=987654321, text=owner_admin_text("检查日志"))
    )
    await service.handle_private_message(
        make_private_event(message_id="p-chat-admin-reset", user_id=987654321, text=owner_admin_text("清空上下文"))
    )

    assert sender.private_sent[-1].text
    with sqlite_engine.connect() as connection:
        rows = connection.execute(
            text("select session_mode, count(*) from dev_sessions group by session_mode order by session_mode asc")
        ).fetchall()
    assert rows == [("daily", 1), ("project", 2)]


@pytest.mark.asyncio
async def test_owner_start_admin_mode_routes_plain_messages_into_project_session(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="project mode reply")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await service.handle_private_message(
        make_private_event(message_id="mode-on", user_id=987654321, text="启动管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="project-chat", user_id=987654321, text="what is the project structure")
    )

    assert "管理员模式" in sender.private_sent[0].text
    prompt = "\n".join(llm_client.prompts[-1])
    assert "Current project session summary:" in prompt
    assert "Relevant repository snippets:" in prompt
    mode_payload = json.loads((data_dir / "dev_control" / "owner_private_mode.json").read_text(encoding="utf-8"))
    assert mode_payload["session_mode"] == "project"


@pytest.mark.asyncio
async def test_owner_end_admin_mode_routes_plain_messages_back_to_daily_session(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="ok")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await service.handle_private_message(
        make_private_event(message_id="mode-on", user_id=987654321, text="启动管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="project-chat", user_id=987654321, text="what is the project structure")
    )
    await service.handle_private_message(
        make_private_event(message_id="mode-off", user_id=987654321, text="结束管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="daily-chat", user_id=987654321, text="你叫什么")
    )

    assert any("退出管理员模式" in outbound.text for outbound in sender.private_sent)
    prompt = "\n".join(llm_client.prompts[-1])
    assert "Current private daily session summary:" in prompt
    assert "Relevant repository snippets:" not in prompt
    mode_payload = json.loads((data_dir / "dev_control" / "owner_private_mode.json").read_text(encoding="utf-8"))
    assert mode_payload["session_mode"] == "daily"


@pytest.mark.asyncio
async def test_owner_exit_admin_mode_alias_routes_back_to_daily_session(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="ok")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await service.handle_private_message(
        make_private_event(message_id="mode-on", user_id=987654321, text="启动管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="mode-off-alias", user_id=987654321, text="退出管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="daily-chat", user_id=987654321, text="你叫什么")
    )

    prompt = "\n".join(llm_client.prompts[-1])
    assert "Current private daily session summary:" in prompt
    mode_payload = json.loads((data_dir / "dev_control" / "owner_private_mode.json").read_text(encoding="utf-8"))
    assert mode_payload["session_mode"] == "daily"


@pytest.mark.asyncio
async def test_owner_admin_mode_state_persists_across_service_recreation(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    first_llm = FakeLlmClient(reply_text="mode set")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    first_service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=first_llm,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await first_service.handle_private_message(
        make_private_event(message_id="mode-on", user_id=987654321, text="启动管理员模式")
    )

    second_llm = FakeLlmClient(reply_text="still project mode")
    second_service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=second_llm,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
    )
    await second_service.handle_private_message(
        make_private_event(message_id="project-chat", user_id=987654321, text="what is the project structure")
    )

    prompt = "\n".join(second_llm.prompts[-1])
    assert "Current project session summary:" in prompt
    assert "Relevant repository snippets:" in prompt


@pytest.mark.asyncio
async def test_reset_session_in_admin_mode_only_resets_project_context(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="ok")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await service.handle_private_message(
        make_private_event(message_id="daily-chat", user_id=987654321, text="今天天气不错")
    )
    await service.handle_private_message(
        make_private_event(message_id="mode-on", user_id=987654321, text="启动管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="project-chat", user_id=987654321, text="检查日志")
    )
    await service.handle_private_message(
        make_private_event(message_id="project-reset", user_id=987654321, text="清空上下文")
    )

    with sqlite_engine.connect() as connection:
        rows = connection.execute(
            text("select session_mode, count(*) from dev_sessions group by session_mode order by session_mode asc")
        ).fetchall()
    assert rows == [("daily", 1), ("project", 2)]


@pytest.mark.asyncio
async def test_reset_session_in_daily_mode_only_resets_daily_context(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="ok")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await service.handle_private_message(
        make_private_event(message_id="mode-on", user_id=987654321, text="启动管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="project-chat", user_id=987654321, text="检查日志")
    )
    await service.handle_private_message(
        make_private_event(message_id="mode-off", user_id=987654321, text="结束管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="daily-chat", user_id=987654321, text="你叫什么")
    )
    await service.handle_private_message(
        make_private_event(message_id="daily-reset", user_id=987654321, text="清空上下文")
    )

    with sqlite_engine.connect() as connection:
        rows = connection.execute(
            text("select session_mode, count(*) from dev_sessions group by session_mode order by session_mode asc")
        ).fetchall()
    assert rows == [("daily", 2), ("project", 1)]


@pytest.mark.asyncio
async def test_allowlisted_private_chat_replies_inline_without_dev_control(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="可以正常私聊，但我不会替你改项目。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("project session", encoding="utf-8")
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        private_chat_qqs={20002},
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-guest-1", user_id=20002, text="你叫什么")
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == ["可以正常私聊，但我不会替你改项目。"]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert len(completed) == 1
    assert completed[0].intent_type == "project_chat"
    assert completed[0].requested_by_qq == 20002
    assert queued == []


@pytest.mark.asyncio
async def test_allowlisted_private_chat_does_not_queue_execute_task(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="这个我不能帮你改项目。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        private_chat_qqs={20002},
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-guest-2", user_id=20002, text="帮我改一下项目配置")
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == ["这个我不能帮你改项目。"]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert len(completed) == 1
    assert completed[0].intent_type == "project_chat"
    assert queued == []
    assert queued == []
    assert llm_client.prompts


@pytest.mark.asyncio
async def test_owner_private_project_chat_preserves_markdownish_reply(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="### 先说结论\n- 确实有点怪。\n- 你再等等看。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-markdown", user_id=987654321, text="why did private chat drift")
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == ["### 先说结论\n- 确实有点怪。\n- 你再等等看。"]


@pytest.mark.asyncio
async def test_owner_private_meta_question_does_not_send_lookup_progress(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="你可以叫我小町。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-name", user_id=987654321, text="你叫什么")
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == ["你可以叫我小町。"]


@pytest.mark.asyncio
async def test_owner_private_project_chat_uses_web_search_and_pages_when_needed(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = SearchAwareFakeLlmClient()
    search_client = FakeSearchClient()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        web_search_client=search_client,
        assistant_name="Codex",
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-search", user_id=987654321, text="水的沸点是多少")
    )

    assert handled is True
    assert llm_client.search_decision_calls == 1
    assert search_client.queries == [("latest anime buzz", 3)]
    assert search_client.page_reads == [(["https://official.example"], "latest anime buzz", 3)]
    assert [outbound.text for outbound in sender.private_sent] == ["### 先说结论\n- 我查了下，确实有新消息。"]
    assert any("Web search results:" in line for line in llm_client.prompts[-1])
    assert any("Detailed review | https://official.example/review | Episode 1 introduces the cast." in line for line in llm_client.prompts[-1])


@pytest.mark.asyncio
async def test_owner_private_search_verification_followup_does_not_trigger_new_web_search(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="搜了，刚刚那轮只是结果不靠谱。")
    search_client = FakeSearchClient()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        web_search_client=search_client,
    )

    with session_scope(sqlite_engine) as session:
        sessions = DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=987654321, session_mode="daily")
        prior_task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=987654321,
            raw_request_text="帮我上网搜一下今天西安西电南校区附近天气",
            intent_type="project_chat",
        )
        tasks.mark_completed(
            task_id=prior_task.id,
            summary="weather lookup",
            result_text="我刚搜了，但搜到的结果有点偏。",
            files_read=[],
            files_changed=[],
            commands_run=["llm_client.generate_text"],
            restart_required=False,
            restart_result="not-needed",
            checkpoint_dir="",
        )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-search-meta", user_id=987654321, text="你真的上网搜了吗")
    )

    assert handled is True
    assert search_client.queries == []
    assert search_client.page_reads == []
    assert [outbound.text for outbound in sender.private_sent] == ["搜了，刚刚那轮只是结果不靠谱。"]


@pytest.mark.asyncio
async def test_owner_private_weather_location_followup_reuses_weather_context_for_new_search(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我按西安长安区重新查了一次天气。")
    search_client = FakeSearchClient()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        web_search_client=search_client,
    )

    with session_scope(sqlite_engine) as session:
        sessions = DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=987654321, session_mode="daily")
        first_task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=987654321,
            raw_request_text="帮我上网搜一下今天西安西电南校区附近天气",
            intent_type="project_chat",
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
            requested_by_qq=987654321,
            raw_request_text="西电南校区",
            intent_type="project_chat",
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
        make_private_event(message_id="p-chat-weather-followup", user_id=987654321, text="那就西安长安区")
    )

    assert handled is True
    assert search_client.queries == [("西安长安区 今天天气", 3)]
    assert search_client.page_reads == [(["https://official.example"], "西安长安区 今天天气", 3)]
    assert [outbound.text for outbound in sender.private_sent] == ["我按西安长安区重新查了一次天气。"]


@pytest.mark.asyncio
async def test_owner_private_runtime_health_check_uses_project_inspect_and_skips_web_search(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = SearchAwareFakeLlmClient()
    search_client = FakeSearchClient()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "app").mkdir()
    (repo_root / "app" / "group_runtime.py").write_text(
        "def handle_group_reply():\n    return 'group reply runtime'\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    (data_dir / "logs").mkdir(parents=True)
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
        web_search_client=search_client,
        assistant_name="Codex",
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-runtime",
            user_id=987654321,
            text=owner_admin_text("检查一下小町群聊回复是否正常运作"),
        )
    )

    assert handled is True
    assert llm_client.search_decision_calls == 0
    assert search_client.queries == []
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert completed[0].intent_type == "project_inspect"
    assert any("Local inspection facts:" in line for line in llm_client.prompts[-1])
    assert any("Relevant repository snippets:" in line for line in llm_client.prompts[-1])


@pytest.mark.asyncio
async def test_owner_private_permission_lookup_uses_project_inspect_and_reads_local_whitelist(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="已经查到了。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "app").mkdir()
    (repo_root / "app" / "config.py").write_text(
        "class AppSettings:\n    private_chat_qqs = ''\n    def private_chat_whitelist(self):\n        return self.private_chat_qqs\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
    )
    (repo_root / ".env").write_text("PRIVATE_CHAT_QQS=20002,20002\n", encoding="utf-8")

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-whitelist",
            user_id=987654321,
            text=owner_admin_text("帮我查一下到底给没给20002私聊权限"),
        )
    )

    assert handled is True
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert completed[0].intent_type == "project_inspect"
    assert any("Local inspection facts:" in line for line in llm_client.prompts[-1])
    assert any("PRIVATE_CHAT_QQS includes 20002: yes" in line for line in llm_client.prompts[-1])
    assert any("app/config.py" in line for line in llm_client.prompts[-1])


@pytest.mark.asyncio
async def test_owner_private_effectiveness_question_uses_project_inspect_even_if_llm_would_bias_execute(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    llm_client = IntentRoutingFakeLlmClient(
        intent_reply="EXECUTE",
        chat_reply="现在还不能确认运行中已经生效，而且就算生效了也只是偶尔@熟人A。",
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    (data_dir / "logs").mkdir(parents=True)
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-effectiveness",
            user_id=987654321,
            text=owner_admin_text("现在稍微针对熟人A的改动生效了吗，会@熟人A吗"),
        )
    )

    assert handled is True
    assert llm_client.intent_calls == 0
    assert [outbound.text for outbound in sender.private_sent] == ["现在还不能确认运行中已经生效，而且就算生效了也只是偶尔@熟人A。"]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert len(completed) == 1
    assert completed[0].intent_type == "project_inspect"
    assert queued == []


@pytest.mark.asyncio
async def test_owner_private_admin_confirmation_continues_recent_inspect_offer(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(
        reply_text=(
            "现在还不能确认已经生效。"
            "如果你要，我下一步就直接去查 configs/persona.yaml 和触发这段人格注入的代码，给你一个明确结论。"
        )
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    (data_dir / "logs").mkdir(parents=True)
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    handled_first = await service.handle_private_message(
        make_private_event(
            message_id="p-offer-inspect",
            user_id=987654321,
            text=owner_admin_text("稍微针对熟人A的改动生效了吗"),
        )
    )
    llm_client.reply_text = "我已经直接去查了相关配置和代码，现在给你明确结论。"
    handled_second = await service.handle_private_message(
        make_private_event(
            message_id="p-confirm-inspect",
            user_id=987654321,
            text=owner_admin_text("好的"),
        )
    )

    assert handled_first is True
    assert handled_second is True
    assert [outbound.text for outbound in sender.private_sent[-2:]] == [
        "现在还不能确认已经生效。如果你要，我下一步就直接去查 configs/persona.yaml 和触发这段人格注入的代码，给你一个明确结论。",
        "我已经直接去查了相关配置和代码，现在给你明确结论。",
    ]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert [task.intent_type for task in completed] == ["project_inspect", "project_inspect"]
    assert "继续上一条" in completed[-1].raw_request_text


@pytest.mark.asyncio
async def test_owner_private_admin_confirmation_continues_recent_project_chat_offer(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我已经直接进仓库核对实际配置和代码了。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    (data_dir / "logs").mkdir(parents=True)
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=987654321)
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=987654321,
            raw_request_text="好的",
            intent_type="project_chat",
        )
        tasks.mark_completed(
            task_id=task.id,
            summary="offered next step",
            result_text="嗯。你要我继续的话，我下一步就直接进仓库核对实际配置和代码，确认那段设定到底有没有写进去。",
            files_read=[],
            files_changed=[],
            commands_run=[],
            restart_required=False,
            restart_result="not-needed",
            checkpoint_dir="",
        )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-confirm-chat-offer",
            user_id=987654321,
            text=owner_admin_text("好的"),
        )
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == ["我已经直接进仓库核对实际配置和代码了。"]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 2
    assert completed[-1].intent_type == "project_inspect"
    assert "继续上一条" in completed[-1].raw_request_text


@pytest.mark.asyncio
async def test_owner_private_admin_confirmation_like_好的就这样_continues_recent_feature_work(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    codex_bridge = FakeCodexBridge(
        result=CodexTaskResult(
            summary="follow-up applied",
            reply_text="我已经把这条后续修改接着做完了。",
            restart_required=False,
        )
    )
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=codex_bridge,
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=987654321)
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=987654321,
            raw_request_text="给 20002 加一个今天 8 点的定时私聊提醒",
            intent_type="feature_work",
        )
        tasks.mark_completed(
            task_id=task.id,
            summary="need final confirmation",
            result_text=(
                "收到，目标 QQ 和时间我都记下了。"
                "还差最后一个确认：文案就用我写的这句，还是你自己提供？"
                "你直接回我一句就行。"
            ),
            files_read=[],
            files_changed=[],
            commands_run=[],
            restart_required=False,
            restart_result="not-needed",
            checkpoint_dir="",
        )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-confirm-feature-followup",
            user_id=987654321,
            text=owner_admin_text("好的就这样"),
        )
    )

    assert handled is True
    assert codex_bridge.prompts
    assert sender.private_sent[-1].text == "我已经把这条后续修改接着做完了。"
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 2
    assert completed[-1].intent_type == "feature_work"
    assert "继续上一条" in completed[-1].raw_request_text


@pytest.mark.asyncio
async def test_owner_private_admin_simple_feature_request_runs_inline_fast_path(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("before\n", encoding="utf-8")
    codex_bridge = MutatingCodexBridge(
        result=CodexTaskResult(
            summary="updated readme",
            reply_text="README 我已经直接改好了。",
            restart_required=False,
        ),
        relative_path="README.md",
        content="after\n",
    )
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=codex_bridge,
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-fast-feature",
            user_id=987654321,
            text=owner_admin_text("把 README 第一行改成 after"),
        )
    )

    assert handled is True
    assert (repo_root / "README.md").read_text(encoding="utf-8") == "after\n"
    assert codex_bridge.prompts
    assert [outbound.text for outbound in sender.private_sent] == [
        "我开始处理这条了。",
        "README 我已经直接改好了。",
    ]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert [task.intent_type for task in completed] == ["feature_work"]
    assert queued == []


@pytest.mark.asyncio
async def test_owner_private_admin_complex_feature_request_queues_worker_workflow(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    codex_bridge = FakeCodexBridge(
        result=CodexTaskResult(
            summary="should not run inline",
            reply_text="不该直接执行",
            restart_required=False,
        )
    )
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=codex_bridge,
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-worker-feature",
            user_id=987654321,
            text=owner_admin_text("把整个私聊开发通道重构一下，持续推进到完成并上线"),
        )
    )

    assert handled is True
    assert sender.private_sent == []
    assert codex_bridge.prompts == []
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert completed == []
    assert len(queued) == 1
    assert queued[0].intent_type == "feature_work"


@pytest.mark.asyncio
async def test_owner_private_send_private_message_request_is_refused_consistently(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="已经发了。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-send", user_id=987654321, text='给熟人A私聊发送“你好”')
    )

    assert handled is True
    assert len(sender.private_sent) == 1
    assert "不能直接替你给别人发私聊消息" in sender.private_sent[0].text
    assert "管理员权限 开发这个功能" in sender.private_sent[0].text
    assert llm_client.prompts == []
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert completed[0].intent_type == "project_chat"
    assert "不能直接替你给别人发私聊消息" in completed[0].result_text


@pytest.mark.asyncio
async def test_owner_project_private_send_request_delivers_to_allowlisted_target_and_replies_with_receipt(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        private_chat_qqs={20002},
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-project-send-ok",
            user_id=987654321,
            text=owner_admin_text('给 20002 私聊发送“管理员模式使用方法：先发启动管理员模式，再直接说要查什么、改什么。”'),
        )
    )

    assert handled is True
    assert [outbound.user_id for outbound in sender.private_sent] == [20002, 987654321]
    assert sender.private_sent[0].text == "管理员模式使用方法：先发启动管理员模式，再直接说要查什么、改什么。"
    assert "已经给 20002 私聊发过去了" in sender.private_sent[1].text
    assert "管理员模式使用方法" in sender.private_sent[1].text
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert completed[-1].intent_type == "project_chat"
    assert "已经给 20002 私聊发过去了" in completed[-1].result_text


@pytest.mark.asyncio
async def test_owner_project_private_send_request_refuses_non_allowlisted_target_with_visible_receipt(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        private_chat_qqs={20003},
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-project-send-denied",
            user_id=987654321,
            text=owner_admin_text('给 20002 私聊发送“你好”'),
        )
    )

    assert handled is True
    assert len(sender.private_sent) == 1
    assert sender.private_sent[0].user_id == 987654321
    assert "20002" in sender.private_sent[0].text
    assert "不在私聊白名单" in sender.private_sent[0].text
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert completed[-1].intent_type == "project_chat"
    assert "不在私聊白名单" in completed[-1].result_text


@pytest.mark.asyncio
async def test_owner_project_private_send_request_reports_target_delivery_failure(
    sqlite_engine, tmp_path
) -> None:
    sender = SelectiveFailSender(failing_user_ids={20002})
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        private_chat_qqs={20002},
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-project-send-failed",
            user_id=987654321,
            text=owner_admin_text('给 20002 私聊发送“你好”'),
        )
    )

    assert handled is True
    assert [outbound.user_id for outbound in sender.private_sent] == [987654321]
    assert "20002" in sender.private_sent[0].text
    assert "没发出去" in sender.private_sent[0].text
    assert "send_private_msg failed: target=20002" in sender.private_sent[0].text
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert completed[-1].intent_type == "project_chat"
    assert "没发出去" in completed[-1].result_text


@pytest.mark.asyncio
async def test_owner_private_new_session_command_creates_fresh_session(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    await service.handle_private_message(
        make_private_event(message_id="p-1", user_id=987654321, text="what is the project structure")
    )
    await service.handle_private_message(make_private_event(message_id="p-2", user_id=987654321, text="/bot new-session"))

    assert sender.private_sent[-1].text
    assert "session" in sender.private_sent[-1].text.lower() or "会话" in sender.private_sent[-1].text


@pytest.mark.asyncio
async def test_owner_private_new_session_chinese_command_creates_fresh_session(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    await service.handle_private_message(
        make_private_event(message_id="p-1", user_id=987654321, text="what is the project structure")
    )
    await service.handle_private_message(make_private_event(message_id="p-2", user_id=987654321, text="清空上下文"))

    assert sender.private_sent[-1].text
    assert "session" in sender.private_sent[-1].text.lower() or "会话" in sender.private_sent[-1].text


@pytest.mark.asyncio
async def test_owner_private_session_command_is_not_deduplicated_across_distinct_messages(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    await service.handle_private_message(
        make_private_event(message_id="p-reset-1", user_id=987654321, text="重置会话")
    )
    await service.handle_private_message(
        make_private_event(message_id="p-reset-2", user_id=987654321, text="重置会话")
    )

    assert len(sender.private_sent) == 2
    assert sender.private_sent[0].text == sender.private_sent[1].text


@pytest.mark.asyncio
async def test_owner_private_session_status_reports_active_session(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    await service.handle_private_message(make_private_event(message_id="p-1", user_id=987654321, text="/bot session-status"))

    assert sender.private_sent[-1].text


@pytest.mark.asyncio
async def test_owner_private_session_status_chinese_command_reports_active_session(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    await service.handle_private_message(make_private_event(message_id="p-1", user_id=987654321, text="会话状态"))

    assert sender.private_sent[-1].text


@pytest.mark.asyncio
async def test_process_next_task_completes_queued_task_and_sends_result(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    bridge = FakeCodexBridge(
        result=CodexTaskResult(
            summary="checked latest logs",
            reply_text="I checked it. The last problem was a timeout, not a missing message.",
            restart_required=False,
        )
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("repo", encoding="utf-8")
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
    )

    await service.handle_private_message(
        make_private_event(
            message_id="p-1",
            user_id=987654321,
            text=owner_admin_text("把这个功能整体修一下并上线"),
        )
    )

    processed = await service.process_next_task_once()

    assert processed is True
    assert [outbound.text for outbound in sender.private_sent] == [
        "我开始处理这条了。",
        "I checked it. The last problem was a timeout, not a missing message.",
    ]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 1
    assert completed[0].summary == "checked latest logs"
    assert bridge.prompts


@pytest.mark.asyncio
async def test_process_next_task_once_ignores_non_execute_project_chat_tasks(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    bridge = FakeCodexBridge(result=CodexTaskResult(summary="done", reply_text="should not run", restart_required=False))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=987654321)
        tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=987654321,
            raw_request_text="你好",
            intent_type="project_chat",
        )

    processed = await service.process_next_task_once()

    assert processed is False
    assert bridge.prompts == []
    assert sender.private_sent == []


@pytest.mark.asyncio
async def test_owner_private_can_confirm_capability_upgrade_after_missing_feature_reply(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled_first = await service.handle_private_message(
        make_private_event(message_id="p-send-missing", user_id=987654321, text='给熟人A私聊发送“你好”')
    )
    handled_second = await service.handle_private_message(
        make_private_event(
            message_id="p-send-upgrade",
            user_id=987654321,
            text=owner_admin_text("开发这个功能"),
        )
    )

    assert handled_first is True
    assert handled_second is True
    assert len(sender.private_sent) == 1
    assert "管理员权限 开发这个功能" in sender.private_sent[0].text
    with session_scope(sqlite_engine) as session:
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert len(queued) == 1
    assert queued[0].intent_type == "feature_work"
    assert "实现一个新能力" in queued[0].raw_request_text
    assert "给熟人A私聊发送“你好”" in queued[0].raw_request_text


@pytest.mark.asyncio
async def test_process_next_task_normalizes_codex_reply_text(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    bridge = FakeCodexBridge(
        result=CodexTaskResult(
            summary="done",
            reply_text="### 先说结论\n- 这个已经改好了\n- 我顺手测过了",
            restart_required=False,
        )
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("repo", encoding="utf-8")
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
    )

    await service.handle_private_message(
        make_private_event(
            message_id="p-exec-1",
            user_id=987654321,
            text=owner_admin_text("把这个功能整体修一下并上线"),
        )
    )
    processed = await service.process_next_task_once()

    assert processed is True
    assert sender.private_sent[-1].text == "### 先说结论\n- 这个已经改好了\n- 我顺手测过了"
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert completed[0].result_text == "### 先说结论\n- 这个已经改好了\n- 我顺手测过了"


@pytest.mark.asyncio
async def test_process_next_task_reuses_codex_thread_for_same_project_session(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    bridge = FakeCodexBridge(
        result=CodexTaskResult(
            summary="done",
            reply_text="changed",
            restart_required=False,
            thread_id="thread-1",
        )
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("repo", encoding="utf-8")
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
    )

    await service.handle_private_message(
        make_private_event(
            message_id="p-1",
            user_id=987654321,
            text=owner_admin_text("把这个功能整体修一下并上线"),
        )
    )
    await service.process_next_task_once()
    await service.handle_private_message(
        make_private_event(
            message_id="p-2",
            user_id=987654321,
            text=owner_admin_text("把这个功能整体再修一下并上线"),
        )
    )
    await service.process_next_task_once()

    assert bridge.resume_thread_ids == [None, "thread-1"]


@pytest.mark.asyncio
async def test_runtime_file_change_triggers_restart_even_when_bridge_does_not_request_it(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    bridge = MutatingCodexBridge(
        result=CodexTaskResult(summary="patched", reply_text="patched it", restart_required=False),
        relative_path="app/runtime_flag.py",
        content="patched",
    )
    repo_root = tmp_path / "repo"
    (repo_root / "app").mkdir(parents=True)
    (repo_root / "app/runtime_flag.py").write_text("before", encoding="utf-8")
    command_calls: list[list[str]] = []

    def fake_command_runner(command: list[str], cwd: Path):
        command_calls.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
        command_runner=fake_command_runner,
    )

    await service.handle_private_message(
        make_private_event(
            message_id="p-1",
            user_id=987654321,
            text=owner_admin_text("把这个功能整体修一下并上线"),
        )
    )
    await service.process_next_task_once()

    assert len(command_calls) == 2
    assert command_calls[0][-1].endswith("stop_xiaomachi_runtime.ps1")
    assert command_calls[1][-1].endswith("start_xiaomachi_runtime.ps1")
    assert [outbound.text for outbound in sender.private_sent] == [
        "我开始处理这条了。",
        "我现在重启小町，让改动生效。",
        "patched it",
    ]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 1
    assert completed[0].restart_required is True


@pytest.mark.asyncio
async def test_process_next_task_does_not_restart_when_codex_only_requests_restart_without_changes(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    bridge = FakeCodexBridge(
        result=CodexTaskResult(
            summary="checked",
            reply_text="我看完了，现在还不能确认已经生效。",
            restart_required=True,
        )
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("repo", encoding="utf-8")
    command_calls: list[list[str]] = []

    def fake_command_runner(command: list[str], cwd: Path):
        command_calls.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
        command_runner=fake_command_runner,
    )

    await service.handle_private_message(
        make_private_event(
            message_id="p-1",
            user_id=987654321,
            text=owner_admin_text("把这个功能整体修一下并上线"),
        )
    )
    processed = await service.process_next_task_once()

    assert processed is True
    assert command_calls == []
    assert [outbound.text for outbound in sender.private_sent] == [
        "我开始处理这条了。",
        "我看完了，现在还不能确认已经生效。",
    ]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 1
    assert completed[0].restart_required is False
    assert completed[0].restart_result == "not-needed"


@pytest.mark.asyncio
async def test_process_next_task_rewrites_final_reply_after_successful_restart(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    bridge = FakeCodexBridge(
        result=CodexTaskResult(
            summary="checked",
            reply_text="代码里已经加进去了，但我不能直接替你执行重启。",
            restart_required=False,
        )
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("repo", encoding="utf-8")
    command_calls: list[list[str]] = []

    def fake_command_runner(command: list[str], cwd: Path):
        command_calls.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
        command_runner=fake_command_runner,
    )

    await service.handle_private_message(
        make_private_event(
            message_id="p-1",
            user_id=987654321,
            text=owner_admin_text("你确保这个改动加入进去了，如果确定不了保险起见你就重启确保真的加入进去了，并上线"),
        )
    )
    processed = await service.process_next_task_once()

    assert processed is True
    assert len(command_calls) == 2
    assert sender.private_sent[0].text == "我开始处理这条了。"
    assert sender.private_sent[-1].text != "代码里已经加进去了，但我不能直接替你执行重启。"
    assert "不能直接替你执行重启" not in sender.private_sent[-1].text
    assert "重启" in sender.private_sent[-1].text
    assert "运行中的实例" in sender.private_sent[-1].text
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 1
    assert "不能直接替你执行重启" not in completed[0].result_text
    assert completed[0].restart_required is True
    assert completed[0].restart_result == "success"


@pytest.mark.asyncio
async def test_service_start_can_disable_local_worker_loop(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
        enable_local_worker=False,
    )

    await service.start()
    await service.stop()

    assert service._worker_task is None


def test_default_command_runner_detaches_stdio_for_runtime_restart_scripts(tmp_path, monkeypatch) -> None:
    sender = FakeSender()
    service = DevControlService(
        engine=object(),
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
    )
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(dev_service_module.subprocess, "run", fake_run)

    service._default_command_runner(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(tmp_path / "start_xiaomachi_runtime.ps1"),
        ],
        tmp_path,
    )

    kwargs = captured["kwargs"]
    assert kwargs["stdout"] is dev_service_module.subprocess.DEVNULL
    assert kwargs["stderr"] is dev_service_module.subprocess.DEVNULL
    assert "capture_output" not in kwargs


def test_default_command_runner_keeps_capturing_regular_commands(tmp_path, monkeypatch) -> None:
    sender = FakeSender()
    service = DevControlService(
        engine=object(),
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
    )
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return type("Completed", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(dev_service_module.subprocess, "run", fake_run)

    service._default_command_runner(["git", "status", "--short", "--branch"], tmp_path)

    kwargs = captured["kwargs"]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"


@pytest.mark.asyncio
async def test_process_next_task_rolls_back_when_restart_fails(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    bridge = MutatingCodexBridge(
        result=CodexTaskResult(summary="patched", reply_text="fixed it", restart_required=True),
        relative_path="app/runtime_flag.py",
        content="after",
    )
    repo_root = tmp_path / "repo"
    (repo_root / "app").mkdir(parents=True)
    (repo_root / "app/runtime_flag.py").write_text("before", encoding="utf-8")
    call_index = {"value": 0}

    def fake_command_runner(command: list[str], cwd: Path):
        call_index["value"] += 1
        if call_index["value"] == 2:
            return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "start failed"})()
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
        command_runner=fake_command_runner,
    )

    await service.handle_private_message(
        make_private_event(
            message_id="p-1",
            user_id=987654321,
            text=owner_admin_text("把整个运行时修一下然后重启上线"),
        )
    )
    await service.process_next_task_once()

    assert (repo_root / "app/runtime_flag.py").read_text(encoding="utf-8") == "before"
    with session_scope(sqlite_engine) as session:
        rolled_back = DevTaskRepository(session).list_tasks_by_status("rolled_back")
    assert len(rolled_back) == 1
    assert len(sender.private_sent) == 3
    assert sender.private_sent[-1].text


@pytest.mark.asyncio
async def test_process_next_task_reports_codex_unavailable_without_crashing_service(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=RaisingCodexBridge("codex executable not found on PATH"),
    )

    await service.handle_private_message(
        make_private_event(
            message_id="p-1",
            user_id=987654321,
            text=owner_admin_text("把这个功能整体修一下并上线"),
        )
    )
    processed = await service.process_next_task_once()

    assert processed is True
    with session_scope(sqlite_engine) as session:
        failed = DevTaskRepository(session).list_tasks_by_status("failed")
    assert len(failed) == 1
    assert "codex executable not found on PATH" in failed[0].failure_reason
    assert len(sender.private_sent) == 2
    assert sender.private_sent[-1].text


@pytest.mark.asyncio
async def test_service_start_does_not_construct_codex_bridge_eagerly(sqlite_engine, tmp_path, monkeypatch) -> None:
    sender = FakeSender()

    class ExplodingBridge:
        def __init__(self) -> None:
            raise AssertionError("should not be constructed during service start")

    monkeypatch.setattr(dev_service_module, "CodexBridge", ExplodingBridge)

    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
    )

    await service.start()
    await service.stop()


@pytest.mark.asyncio
async def test_process_next_task_reports_checkpoint_failure_without_stalling_queue(
    sqlite_engine, tmp_path, monkeypatch
) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    def exploding_checkpoint(*, repo_root: Path, checkpoint_dir: Path):
        raise OSError("checkpoint failed")

    monkeypatch.setattr(dev_service_module, "create_repo_checkpoint", exploding_checkpoint)

    await service.handle_private_message(
        make_private_event(
            message_id="p-1",
            user_id=987654321,
            text=owner_admin_text("把这个功能整体修一下并上线"),
        )
    )
    processed = await service.process_next_task_once()

    assert processed is True
    with session_scope(sqlite_engine) as session:
        failed = DevTaskRepository(session).list_tasks_by_status("failed")
    assert len(failed) == 1
    assert "checkpoint failed" in failed[0].failure_reason
    assert len(sender.private_sent) == 2
    assert sender.private_sent[-1].text


@pytest.mark.asyncio
async def test_process_next_task_continues_when_artifact_recording_fails(
    sqlite_engine, tmp_path, monkeypatch
) -> None:
    sender = FakeSender()
    bridge = FakeCodexBridge(result=CodexTaskResult(summary="done", reply_text="all good", restart_required=False))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("repo", encoding="utf-8")
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
    )

    class ExplodingArtifactRepository:
        def __init__(self, session) -> None:
            pass

        def add_artifact(self, **kwargs) -> None:
            raise RuntimeError("artifact table unavailable")

    monkeypatch.setattr(dev_service_module, "DevTaskArtifactRepository", ExplodingArtifactRepository)

    await service.handle_private_message(
        make_private_event(
            message_id="p-1",
            user_id=987654321,
            text=owner_admin_text("把这个功能整体修一下并上线"),
        )
    )
    processed = await service.process_next_task_once()

    assert processed is True
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 1
    assert completed[0].result_text == "all good"
    assert len(sender.private_sent) == 2
    assert sender.private_sent[-1].text == "all good"


@pytest.mark.asyncio
async def test_service_start_recovers_running_task_from_saved_codex_result(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("repo", encoding="utf-8")
    command_calls: list[list[str]] = []

    def fake_command_runner(command: list[str], cwd: Path):
        del cwd
        command_calls.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        command_runner=fake_command_runner,
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=987654321)
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=987654321,
            raw_request_text="fix this",
            intent_type="feature_work",
        )
        tasks.mark_status(task_id=task.id, status="running")
        task_id = task.id

    artifact_dir = tmp_path / "data" / "dev_control" / "tasks" / f"task-{task_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "codex.last_message.json").write_text(
        '{"summary":"recovered","reply_text":"补发结果","restart_required":true}',
        encoding="utf-8",
    )

    await service.start()
    await service.stop()

    assert command_calls == []
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 1
    assert completed[0].result_text == "补发结果"
    assert completed[0].restart_required is False
    assert completed[0].restart_result == "not-needed"
    assert sender.private_sent[-1].text == "刚才那条任务其实已经跑完了，我把结果补发给你：补发结果"


@pytest.mark.asyncio
async def test_service_start_marks_stale_running_non_execute_task_failed(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=987654321)
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=987654321,
            raw_request_text="评价一下最近刚出的低智商犯罪这个电视剧",
            intent_type="project_chat",
        )
        tasks.mark_status(task_id=task.id, status="running")

    await service.start()
    await service.stop()

    with session_scope(sqlite_engine) as session:
        failed = DevTaskRepository(session).list_tasks_by_status("failed")
    assert len(failed) == 1
    assert failed[0].intent_type == "project_chat"
    assert "stale non-execute task" in failed[0].failure_reason
    assert sender.private_sent == []


@pytest.mark.asyncio
async def test_service_start_marks_stale_queued_non_execute_task_failed(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=987654321)
        tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=987654321,
            raw_request_text="你真的上网搜了吗",
            intent_type="project_chat",
        )

    await service.start()
    await service.stop()

    with session_scope(sqlite_engine) as session:
        failed = DevTaskRepository(session).list_tasks_by_status("failed")
    assert len(failed) == 1
    assert failed[0].intent_type == "project_chat"
    assert "stale non-execute task" in failed[0].failure_reason
    assert sender.private_sent == []


@pytest.mark.asyncio
async def test_send_private_text_does_not_cancel_slow_delivery_early(sqlite_engine, tmp_path) -> None:
    sender = SlowSender(delay_seconds=0.05)
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir()

    delivered = await service._send_private_text(
        user_id=987654321,
        text="slow reply",
        context="test-slow-send",
        timeout_seconds=0.01,
    )

    assert delivered is True
    assert [outbound.text for outbound in sender.private_sent] == ["slow reply"]


@pytest.mark.asyncio
async def test_send_private_text_deduplicates_same_context(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=987654321,
        bot_qq=123456789,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir()

    first = await service._send_private_text(
        user_id=987654321,
        text="dedup reply",
        context="project_chat:123:completed",
    )
    second = await service._send_private_text(
        user_id=987654321,
        text="dedup reply",
        context="project_chat:123:completed",
    )

    assert first is True
    assert second is True
    assert [outbound.text for outbound in sender.private_sent] == ["dedup reply"]

    with session_scope(sqlite_engine) as session:
        messages = dev_service_module.MessageRepository(session)
        stored = messages.get_by_platform_msg_id("private-outbound-project_chat:123:completed")
    assert stored is not None
    assert stored.raw_json["delivery_state"] == "sent"
