from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from app.adapters.onebot_models import PrivateMessageEvent
from app.core.group_image_generation import PrivateImageGenerationRequest
from app.core.message_content import ImageAttachment
import app.dev_control.service as dev_service_module
from app.dev_control.codex_bridge import CodexTaskResult
from app.dev_control.service import DevControlService
from app.storage.db import session_scope
from app.storage.repositories import DevSessionRepository, DevTaskRepository, JobRepository


class FakeSender:
    def __init__(self) -> None:
        self.private_sent = []
        self.private_image_sent = []

    async def send_private_text(self, outbound) -> None:
        self.private_sent.append(outbound)

    async def send_private_image(self, *, user_id: int, image_file: str) -> None:
        self.private_image_sent.append({"user_id": user_id, "image_file": image_file})


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


class FakeImageSearchClient:
    def __init__(self, *, image_results: list[ImageAttachment]) -> None:
        self.image_results = list(image_results)
        self.queries: list[tuple[str, int]] = []

    def image_search(self, query: str, max_results: int = 3) -> list[ImageAttachment]:
        self.queries.append((query, max_results))
        return list(self.image_results)


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


class SequenceCodexBridge:
    def __init__(self, *, results: list[CodexTaskResult]) -> None:
        self.results = list(results)
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
        del repo_root
        self.prompts.append(prompt)
        self.resume_thread_ids.append(resume_thread_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return self.results.pop(0)


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


class ThreadCapturingCodexBridge(FakeCodexBridge):
    def __init__(self, *, result: CodexTaskResult) -> None:
        super().__init__(result=result)
        self.call_thread_ids: list[int] = []

    def run_task(
        self,
        *,
        prompt: str,
        repo_root: Path,
        artifact_dir: Path,
        resume_thread_id: str | None = None,
    ) -> CodexTaskResult:
        self.call_thread_ids.append(threading.get_ident())
        return super().run_task(
            prompt=prompt,
            repo_root=repo_root,
            artifact_dir=artifact_dir,
            resume_thread_id=resume_thread_id,
        )


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


async def confirm_owner_feature_request(
    service: DevControlService,
    *,
    request_text: str,
    request_message_id: str = "p-feature-plan",
    confirmation_text: str = "好",
    confirmation_message_id: str = "p-feature-confirm",
) -> tuple[bool, bool]:
    handled_first = await service.handle_private_message(
        make_private_event(
            message_id=request_message_id,
            user_id=10001,
            text=owner_admin_text(request_text),
        )
    )
    handled_second = await service.handle_private_message(
        make_private_event(
            message_id=confirmation_message_id,
            user_id=10001,
            text=owner_admin_text(confirmation_text),
        )
    )
    return handled_first, handled_second


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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-1", user_id=10001, text=owner_admin_text("check logs"))
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
            owner_qq=10001,
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-restart-status",
            user_id=10001,
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        assistant_name="比企谷小町",
        persona={
            "name": "比企谷小町",
            "identity": "A fixed AI persona modeled after Hikigaya Komachi.",
        },
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-restart-status-chat", user_id=10001, text="已经重启生效了吗")
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        assistant_name="比企谷小町",
        persona={
            "name": "比企谷小町",
            "identity": "A fixed AI persona modeled after Hikigaya Komachi.",
        },
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-fix", user_id=10001, text="fix this")
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
async def test_owner_private_admin_execute_request_creates_feature_plan_and_sends_confirmation(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    bridge = FakeCodexBridge(result=CodexTaskResult(summary="done", reply_text="ok", restart_required=False))
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=IntentRoutingFakeLlmClient(intent_reply="EXECUTE"),
        owner_qq=10001,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
    )
    (tmp_path / "repo").mkdir()

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-1",
            user_id=10001,
            text=owner_admin_text("把这个功能整体修一下并上线"),
        )
    )

    assert handled is True
    assert len(sender.private_sent) == 1
    assert "把这个功能整体修一下并上线" in sender.private_sent[0].text
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert [task.intent_type for task in completed] == ["feature_plan"]
    assert [task.raw_request_text for task in completed] == ["把这个功能整体修一下并上线"]
    assert queued == []


@pytest.mark.asyncio
async def test_owner_private_featureword_增加_is_classified_as_execute_request(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=IntentRoutingFakeLlmClient(intent_reply="EXECUTE"),
        owner_qq=10001,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir()

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-exec-increase",
            user_id=10001,
            text=owner_admin_text("你去确认，发现能@就直接@他，不行就增加这个功能然后@他什么都不说，并上线"),
        )
    )

    assert handled is True
    assert len(sender.private_sent) == 1
    assert "增加这个功能" in sender.private_sent[0].text
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert len(completed) == 1
    assert completed[0].intent_type == "feature_plan"
    assert "增加这个功能" in completed[0].raw_request_text
    assert queued == []


def test_owner_private_api_probe_request_is_classified_as_execute_request(sqlite_engine, tmp_path) -> None:
    service = DevControlService(
        engine=sqlite_engine,
        sender=FakeSender(),
        llm_client=IntentRoutingFakeLlmClient(intent_reply="EXECUTE"),
        owner_qq=10001,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir()

    assert service._classify_intent(
        "go check whether gpt-image-2 is available on our api and run a real probe"
    ) == "feature_work"


def test_build_execute_prompt_requires_current_evidence_before_claiming_machine_wide_network_block(
    sqlite_engine, tmp_path
) -> None:
    service = DevControlService(
        engine=sqlite_engine,
        sender=FakeSender(),
        llm_client=IntentRoutingFakeLlmClient(intent_reply="EXECUTE"),
        owner_qq=10001,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir()

    prompt = service._build_execute_prompt(
        session_id=1,
        task_id=1,
        request_text="check whether gpt-image-2 is available on our api and diagnose failures",
    )

    assert "Do not claim a machine-wide network or HTTPS block" in prompt
    assert "If current verification shows any successful API call or probe result" in prompt


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
        llm_client=IntentRoutingFakeLlmClient(intent_reply="EXECUTE"),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
        command_runner=fake_command_runner,
    )

    await service.handle_private_message(
        make_private_event(message_id="p-restart-only", user_id=10001, text=owner_admin_text("重启一下"))
    )
    processed = await service.process_next_task_once()

    assert processed is True
    assert bridge.prompts == []
    assert command_calls == [
        ["wsl.exe", "bash", "/usr/local/bin/xiaomachi-wsl-entry", "install"]
    ]
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
        owner_qq=10001,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir()

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-exec-ambiguous",
            user_id=10001,
            text=owner_admin_text("去群里@熟人A什么话都不说，持续推进到完成"),
        )
    )

    assert handled is True
    assert llm_client.intent_calls == 1
    assert len(sender.private_sent) == 1
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert len(completed) == 1
    assert completed[0].intent_type == "feature_plan"
    assert completed[0].raw_request_text == "去群里@熟人A什么话都不说，持续推进到完成"
    assert queued == []


@pytest.mark.asyncio
async def test_owner_private_execute_ack_is_not_deduplicated_across_different_tasks(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir()

    handled_first = await service.handle_private_message(
        make_private_event(
            message_id="p-exec-1",
            user_id=10001,
            text=owner_admin_text("把这个功能整体修一下并上线"),
        )
    )
    handled_second = await service.handle_private_message(
        make_private_event(
            message_id="p-exec-2",
            user_id=10001,
            text=owner_admin_text("把这个功能整体再修一下并上线"),
        )
    )

    assert handled_first is True
    assert handled_second is True
    assert len(sender.private_sent) == 2
    with session_scope(sqlite_engine) as session:
        messages = dev_service_module.MessageRepository(session)
        first = messages.get_by_platform_msg_id("private-outbound-feature_plan:1:completed")
        second = messages.get_by_platform_msg_id("private-outbound-feature_plan:2:completed")
    assert first is not None
    assert second is not None


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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        assistant_name="比企谷小町",
        persona={
            "name": "比企谷小町",
            "identity": "A fixed AI persona modeled after Hikigaya Komachi.",
        },
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-1", user_id=10001, text="how does the private project session work")
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == [
        "This private chat is now one continuous daily session."
    ]
    prompt = "\n".join(llm_client.prompts[0])
    assert "Current private daily session summary:" in prompt
    assert "Recent private daily turns:" in prompt
    assert "比企谷小町" in prompt
    assert "Do not default to Markdown" in prompt
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
async def test_owner_private_daily_chat_flattens_markdownish_reply_without_cutting_content(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="### 先说结论\n- 这个可以\n- 你现在就去改\n- 还有一堆实现细节后面再说")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-compact-1", user_id=10001, text="你刚才那个能不能简单说")
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == ["先说结论 这个可以。你现在就去改。还有一堆实现细节后面再说。"]


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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-key", user_id=10001, text="remember this chat")
    )

    assert handled is True
    with session_scope(sqlite_engine) as session:
        dev_session = DevSessionRepository(session).get_latest_owner_session(
            owner_qq=10001,
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
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
async def test_owner_private_daily_chat_passes_current_images_to_llm(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我看到这张图了。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
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
async def test_owner_private_daily_chat_uses_quoted_private_images(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我看的是你引用的那张图。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    with session_scope(sqlite_engine) as session:
        dev_service_module.UserRepository(session).upsert_user(user_id=10001, nickname="owner", group_card="")
        dev_service_module.MessageRepository(session).add_private_message(
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        private_image_followup_window_seconds=0.05,
    )

    with session_scope(sqlite_engine) as session:
        dev_service_module.UserRepository(session).upsert_user(user_id=10001, nickname="owner", group_card="")
        dev_service_module.MessageRepository(session).add_private_message(
            platform_msg_id="private-inbound-10001-p-chat-image-only",
            user_id=10001,
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
    handled_text = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-image-followup",
            user_id=10001,
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
async def test_owner_private_single_image_waits_silently_until_followup_text(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="我看到的是你后面跟上的那张图。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
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
    await dev_service_module.asyncio.sleep(0.1)

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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    with session_scope(sqlite_engine) as session:
        dev_service_module.UserRepository(session).upsert_user(user_id=10001, nickname="owner", group_card="")
        dev_service_module.MessageRepository(session).add_private_message(
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
async def test_owner_private_daily_image_generation_request_sends_private_image(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    chat_llm = FakeLlmClient(reply_text="should not be used")
    image_llm = FakeImageGenerationLlm()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=chat_llm,
        image_llm_client=image_llm,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
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
async def test_owner_private_daily_positive_negative_prompt_template_sends_private_image(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    chat_llm = FakeLlmClient(reply_text="should not be used")
    image_llm = FakeImageGenerationLlm()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=chat_llm,
        image_llm_client=image_llm,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-private-image-prompt-template",
            user_id=10001,
            text=(
                "画一张图，正面提示词为:masterpiece, best quality, 1girl\n"
                "负面提示词为：worst quality, low quality"
            ),
        )
    )

    assert handled is True
    await service.private_image_service.wait_for_idle()
    assert chat_llm.prompts == []
    assert len(image_llm.generate_calls) == 1
    assert "masterpiece, best quality, 1girl" in image_llm.generate_calls[0]["prompt"]
    assert sender.private_image_sent and sender.private_image_sent[0]["user_id"] == 10001
    assert [outbound.text for outbound in sender.private_sent] == ["图我接住了，开始画", "图好了"]


@pytest.mark.asyncio
async def test_owner_private_image_then_followup_generation_only_replies_once_with_private_image(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    chat_llm = FakeLlmClient(reply_text="should not be used")
    image_llm = FakeImageGenerationLlm()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=chat_llm,
        image_llm_client=image_llm,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        private_image_followup_window_seconds=0.05,
    )

    with session_scope(sqlite_engine) as session:
        dev_service_module.UserRepository(session).upsert_user(user_id=10001, nickname="owner", group_card="")
        dev_service_module.MessageRepository(session).add_private_message(
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
    handled_text = await service.handle_private_message(
        make_private_event(
            message_id="p-private-layout-followup",
            user_id=10001,
            text="\u4fdd\u7559\u524d\u56fe\u6784\u56fe\uff0c\u53ea\u66ff\u6362\u4eba\u7269\u51fa\u56fe",
        )
    )
    await dev_service_module.asyncio.sleep(0.1)

    assert handled_image is True
    assert handled_text is True
    await service.private_image_service.wait_for_idle()
    assert chat_llm.prompts == []
    assert len(image_llm.edit_calls) == 1
    assert image_llm.edit_calls[0]["images"][0].file_id == "layout-followup.png"
    assert len(sender.private_image_sent) == 1
    assert [outbound.text for outbound in sender.private_sent] == ["图我接住了，开始画", "图好了"]


@pytest.mark.asyncio
async def test_owner_private_reset_draw_clears_followup_image_context(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    chat_llm = FakeLlmClient(reply_text="现在不会再沿用上一张图。")
    image_llm = FakeImageGenerationLlm()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=chat_llm,
        image_llm_client=image_llm,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        private_image_followup_window_seconds=0.05,
    )

    with session_scope(sqlite_engine) as session:
        dev_service_module.UserRepository(session).upsert_user(user_id=10001, nickname="owner", group_card="")
        dev_service_module.MessageRepository(session).add_private_message(
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
        make_private_event(
            message_id="p-reset-draw",
            user_id=10001,
            text="重置绘画",
        )
    )
    handled_text = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-after-reset",
            user_id=10001,
            text="这是谁",
        )
    )
    await dev_service_module.asyncio.sleep(0.1)

    assert handled_image is True
    assert handled_reset is True
    assert handled_text is True
    assert image_llm.edit_calls == []
    assert image_llm.generate_calls == []
    assert sender.private_image_sent == []
    assert sender.private_sent[-1].text == "现在不会再沿用上一张图。"
    assert chat_llm.images_calls[-1] is None


@pytest.mark.asyncio
async def test_owner_private_auto_web_reference_generation_combines_search_results_and_private_layout(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    chat_llm = FakeLlmClient(reply_text="should not be used")
    image_llm = FakeImageGenerationLlm()
    web_search_client = FakeImageSearchClient(
        image_results=[
            ImageAttachment(
                url="https://img.example.test/hero-a.png",
                file_id="hero-a.png",
                local_path=str(tmp_path / "hero-a.png"),
            ),
            ImageAttachment(
                url="https://img.example.test/hero-b.png",
                file_id="hero-b.png",
                local_path=str(tmp_path / "hero-b.png"),
            ),
        ]
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=chat_llm,
        image_llm_client=image_llm,
        web_search_client=web_search_client,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-private-auto-web-ref",
            user_id=10001,
            text="\u53bb\u7f51\u4e0a\u627e\u8d85\u65f6\u7a7a\u8f89\u591c\u59ec\u4e24\u4e2a\u5973\u4e3b\u7684\u4eba\u8bbe\u56fe\uff0c\u4fdd\u7559\u524d\u56fe\u6784\u56fe\uff0c\u53ea\u66ff\u6362\u4eba\u7269\u51fa\u56fe",
            msg_type="mixed",
            images=[
                ImageAttachment(
                    url="https://img.example.test/private-layout.png",
                    file_id="private-layout.png",
                    local_path=str(tmp_path / "private-layout.png"),
                )
            ],
        )
    )

    assert handled is True
    await service.private_image_service.wait_for_idle()
    assert chat_llm.prompts == []
    assert web_search_client.queries == [("\u8d85\u65f6\u7a7a\u8f89\u591c\u59ec\u4e24\u4e2a\u5973\u4e3b", 3)]
    assert len(image_llm.edit_calls) == 1
    assert [image.file_id for image in image_llm.edit_calls[0]["images"]] == [
        "private-layout.png",
        "hero-a.png",
        "hero-b.png",
    ]
    assert len(sender.private_image_sent) == 1
    assert [outbound.text for outbound in sender.private_sent] == ["图我接住了，开始画", "图好了"]


@pytest.mark.asyncio
async def test_owner_private_reply_to_remote_private_image_runs_image_generation(sqlite_engine, tmp_path) -> None:
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
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=chat_llm,
        image_llm_client=image_llm,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
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
async def test_owner_private_reply_to_remote_private_image_handles_simple_retouch_prompt(sqlite_engine, tmp_path) -> None:
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
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=chat_llm,
        image_llm_client=image_llm,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
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
async def test_service_start_recovers_running_private_image_task_from_persistent_queue(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    image_llm = FakeImageGenerationLlm()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(reply_text="should not be used"),
        image_llm_client=image_llm,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        enable_local_worker=False,
    )

    with session_scope(sqlite_engine) as session:
        sessions = DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(
            owner_qq=10001,
            session_mode=dev_service_module.SESSION_MODE_PROJECT,
        )
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="参考这张图重新出图",
            intent_type="project_chat",
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
async def test_owner_private_image_prompt_keeps_only_basic_vision_guidance(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="这是模型自己判断后的回复。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
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
    assert "Only name characters that actually belong to" not in prompt


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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await service.handle_private_message(
        make_private_event(message_id="p-chat-daily", user_id=10001, text="今天有点困")
    )
    await service.handle_private_message(
        make_private_event(message_id="p-chat-admin", user_id=10001, text=owner_admin_text("检查日志"))
    )

    with sqlite_engine.connect() as connection:
        rows = connection.execute(
            text(
                "select owner_qq, session_mode, count(*) "
                "from dev_sessions group by owner_qq, session_mode order by session_mode asc"
            )
        ).fetchall()
    assert rows == [
        (10001, "daily", 1),
        (10001, "project", 1),
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await service.handle_private_message(
        make_private_event(message_id="p-chat-daily", user_id=10001, text="今天天气不错")
    )
    await service.handle_private_message(
        make_private_event(message_id="p-chat-admin", user_id=10001, text=owner_admin_text("检查日志"))
    )
    await service.handle_private_message(
        make_private_event(message_id="p-chat-admin-reset", user_id=10001, text=owner_admin_text("清空上下文"))
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await service.handle_private_message(
        make_private_event(message_id="mode-on", user_id=10001, text="启动管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="project-chat", user_id=10001, text="what is the project structure")
    )

    assert "管理员模式" in sender.private_sent[0].text
    assert "完了直接回你结果" in sender.private_sent[-1].text
    assert llm_client.prompts == []
    with session_scope(sqlite_engine) as session:
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert [task.intent_type for task in queued] == ["admin_agent_turn"]
    mode_payload = json.loads((data_dir / "dev_control" / "owner_private_mode.json").read_text(encoding="utf-8"))
    assert mode_payload["session_mode"] == "project"


@pytest.mark.asyncio
async def test_allowlisted_admin_is_blocked_when_another_admin_repo_task_is_active(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        private_chat_qqs={10002},
        admin_qqs={10002},
        repo_root=repo_root,
        data_dir=data_dir,
        enable_local_worker=False,
    )

    with session_scope(sqlite_engine) as session:
        sessions = DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(
            owner_qq=10001,
            session_mode=dev_service_module.SESSION_MODE_PROJECT,
        )
        running_task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="把 README 改掉并跑测试",
            intent_type="admin_agent_turn",
            status="running",
        )
        sessions.update_session(session_id=dev_session.id, last_task_id=running_task.id)

    await service.handle_private_message(
        make_private_event(message_id="guest-mode-on-busy", user_id=10002, text="启动管理员模式")
    )
    handled = await service.handle_private_message(
        make_private_event(message_id="guest-project-chat-busy", user_id=10002, text="把 README 改掉并跑测试")
    )

    assert handled is True
    assert sender.private_sent[-1].text == "当前已有管理员任务正在执行，请稍后重试。"
    with session_scope(sqlite_engine) as session:
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
        running = DevTaskRepository(session).list_tasks_by_status("running")
    assert queued == []
    assert len(running) == 1
    assert running[0].requested_by_qq == 10001


@pytest.mark.asyncio
async def test_allowlisted_admin_can_start_admin_mode_and_route_plain_messages_into_project_session(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="admin project mode reply")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        private_chat_qqs={10002},
        admin_qqs={10002},
        repo_root=repo_root,
        data_dir=data_dir,
        enable_local_worker=False,
    )

    await service.handle_private_message(
        make_private_event(message_id="guest-mode-on", user_id=10002, text="启动管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="guest-project-chat", user_id=10002, text="what is the project structure")
    )

    assert any("管理员模式" in outbound.text for outbound in sender.private_sent)
    assert "完了直接回你结果" in sender.private_sent[-1].text
    assert llm_client.prompts == []
    with session_scope(sqlite_engine) as session:
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert [task.intent_type for task in queued] == ["admin_agent_turn"]
    assert queued[0].requested_by_qq == 10002
    mode_payload = json.loads((data_dir / "dev_control" / "owner_private_mode.json").read_text(encoding="utf-8"))
    assert mode_payload["user_modes"]["10002"]["session_mode"] == "project"


@pytest.mark.asyncio
async def test_admin_mode_state_is_isolated_per_user(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="owner daily reply")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        private_chat_qqs={10002},
        admin_qqs={10002},
        repo_root=repo_root,
        data_dir=data_dir,
        enable_local_worker=False,
    )

    await service.handle_private_message(
        make_private_event(message_id="guest-mode-on-isolated", user_id=10002, text="启动管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="owner-daily-after-guest-mode", user_id=10001, text="你叫什么")
    )

    prompt = "\n".join(llm_client.prompts[-1])
    assert "Current private daily session summary:" in prompt
    assert "Relevant repository snippets:" not in prompt


@pytest.mark.asyncio
async def test_service_start_sends_private_admin_intro_once(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        private_chat_qqs={10002},
        admin_qqs={10002},
        repo_root=repo_root,
        data_dir=data_dir,
        enable_local_worker=False,
    )

    await service.start()
    await service.stop()

    assert [outbound.user_id for outbound in sender.private_sent] == [10002]
    assert "启动管理员模式" in sender.private_sent[0].text
    assert "结束管理员模式" in sender.private_sent[0].text

    second_service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        private_chat_qqs={10002},
        admin_qqs={10002},
        repo_root=repo_root,
        data_dir=data_dir,
        enable_local_worker=False,
    )
    await second_service.start()
    await second_service.stop()

    assert len(sender.private_sent) == 1


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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await service.handle_private_message(
        make_private_event(message_id="mode-on", user_id=10001, text="启动管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="project-chat", user_id=10001, text="what is the project structure")
    )
    await service.handle_private_message(
        make_private_event(message_id="mode-off", user_id=10001, text="结束管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="daily-chat", user_id=10001, text="你叫什么")
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await service.handle_private_message(
        make_private_event(message_id="mode-on", user_id=10001, text="启动管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="mode-off-alias", user_id=10001, text="退出管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="daily-chat", user_id=10001, text="你叫什么")
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await first_service.handle_private_message(
        make_private_event(message_id="mode-on", user_id=10001, text="启动管理员模式")
    )

    second_llm = FakeLlmClient(reply_text="still project mode")
    second_service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=second_llm,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
    )
    await second_service.handle_private_message(
        make_private_event(message_id="project-chat", user_id=10001, text="what is the project structure")
    )

    assert second_llm.prompts == []
    with session_scope(sqlite_engine) as session:
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert [task.intent_type for task in queued] == ["admin_agent_turn"]


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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await service.handle_private_message(
        make_private_event(message_id="daily-chat", user_id=10001, text="今天天气不错")
    )
    await service.handle_private_message(
        make_private_event(message_id="mode-on", user_id=10001, text="启动管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="project-chat", user_id=10001, text="检查日志")
    )
    await service.handle_private_message(
        make_private_event(message_id="project-reset", user_id=10001, text="清空上下文")
    )

    with sqlite_engine.connect() as connection:
        rows = connection.execute(
            text("select session_mode, count(*) from dev_sessions group by session_mode order by session_mode asc")
        ).fetchall()
    assert rows == [("daily", 1), ("project", 1)]
    assert "正在处理的项目对话任务" in sender.private_sent[-1].text


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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    await service.handle_private_message(
        make_private_event(message_id="mode-on", user_id=10001, text="启动管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="project-chat", user_id=10001, text="检查日志")
    )
    await service.handle_private_message(
        make_private_event(message_id="mode-off", user_id=10001, text="结束管理员模式")
    )
    await service.handle_private_message(
        make_private_event(message_id="daily-chat", user_id=10001, text="你叫什么")
    )
    await service.handle_private_message(
        make_private_event(message_id="daily-reset", user_id=10001, text="清空上下文")
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
        owner_qq=10001,
        private_chat_qqs={10002},
        repo_root=repo_root,
        data_dir=tmp_path / "data",
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
    assert completed[0].intent_type == "project_chat"
    assert completed[0].requested_by_qq == 10002
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
        owner_qq=10001,
        private_chat_qqs={10002},
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-guest-2", user_id=10002, text="帮我改一下项目配置")
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
async def test_owner_private_project_chat_flattens_markdownish_reply(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(reply_text="### 先说结论\n- 确实有点怪。\n- 你再等等看。")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-markdown", user_id=10001, text="why did private chat drift")
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == ["先说结论 确实有点怪。你再等等看。"]


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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-name", user_id=10001, text="你叫什么")
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        web_search_client=search_client,
        assistant_name="Codex",
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-search", user_id=10001, text="水的沸点是多少")
    )

    assert handled is True
    assert llm_client.search_decision_calls == 1
    assert search_client.queries == [("latest anime buzz", 3)]
    assert search_client.page_reads == [(["https://official.example"], "latest anime buzz", 3)]
    assert [outbound.text for outbound in sender.private_sent] == ["先说结论 我查了下，确实有新消息。"]
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        web_search_client=search_client,
    )

    with session_scope(sqlite_engine) as session:
        sessions = DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001, session_mode="daily")
        prior_task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
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
        make_private_event(message_id="p-chat-search-meta", user_id=10001, text="你真的上网搜了吗")
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
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
            requested_by_qq=10001,
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
        make_private_event(message_id="p-chat-weather-followup", user_id=10001, text="那就西安长安区")
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
        web_search_client=search_client,
        assistant_name="Codex",
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-runtime",
            user_id=10001,
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
    )
    (repo_root / ".env").write_text("PRIVATE_CHAT_QQS=10002,20002\n", encoding="utf-8")

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-whitelist",
            user_id=10001,
            text=owner_admin_text("帮我查一下到底给没给10002私聊权限"),
        )
    )

    assert handled is True
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert completed[0].intent_type == "project_inspect"
    assert any("Local inspection facts:" in line for line in llm_client.prompts[-1])
    assert any("PRIVATE_CHAT_QQS includes 10002: yes" in line for line in llm_client.prompts[-1])
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-chat-effectiveness",
            user_id=10001,
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
async def test_owner_private_admin_confirmation_continues_recent_inspect_offer_into_feature_work(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    llm_client = FakeLlmClient(
        reply_text=(
            "现在还不能确认已经生效。"
            "如果你要，我下一步就直接去查 configs/persona.yaml 和触发这段人格注入的代码，给你一个明确结论。"
        )
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "configs").mkdir()
    (repo_root / "configs" / "persona.yaml").write_text("name: Komachi\n", encoding="utf-8")
    data_dir = tmp_path / "data"
    (data_dir / "logs").mkdir(parents=True)
    codex_bridge = FakeCodexBridge(
        result=CodexTaskResult(
            summary="checked actual runtime state",
            reply_text="我已经直接进项目里核对并验证过了，现在给你明确结论。",
            restart_required=False,
        )
    )
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=llm_client,
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
        codex_bridge=codex_bridge,
    )

    handled_first = await service.handle_private_message(
        make_private_event(
            message_id="p-offer-inspect",
            user_id=10001,
            text=owner_admin_text("稍微针对熟人A的改动生效了吗"),
        )
    )
    handled_second = await service.handle_private_message(
        make_private_event(
            message_id="p-confirm-inspect",
            user_id=10001,
            text=owner_admin_text("好的"),
        )
    )

    assert handled_first is True
    assert handled_second is True
    assert [outbound.text for outbound in sender.private_sent[-3:]] == [
        "现在还不能确认已经生效。如果你要，我下一步就直接去查 configs/persona.yaml 和触发这段人格注入的代码，给你一个明确结论。",
        "我开始处理这条了。",
        "我已经直接进项目里核对并验证过了，现在给你明确结论。",
    ]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert [task.intent_type for task in completed] == ["project_inspect", "feature_work"]
    assert "继续上一条" in completed[-1].raw_request_text
    assert codex_bridge.prompts


@pytest.mark.asyncio
async def test_owner_private_admin_confirmation_continues_recent_project_chat_offer_into_feature_work(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = tmp_path / "data"
    (data_dir / "logs").mkdir(parents=True)
    codex_bridge = FakeCodexBridge(
        result=CodexTaskResult(
            summary="checked repo state",
            reply_text="我已经直接进项目里查完并给你对上了。",
            restart_required=False,
        )
    )
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=data_dir,
        codex_bridge=codex_bridge,
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001)
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
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
            user_id=10001,
            text=owner_admin_text("好的"),
        )
    )

    assert handled is True
    assert [outbound.text for outbound in sender.private_sent] == [
        "我开始处理这条了。",
        "我已经直接进项目里查完并给你对上了。",
    ]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 2
    assert completed[-1].intent_type == "feature_work"
    assert "继续上一条" in completed[-1].raw_request_text
    assert codex_bridge.prompts


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
        llm_client=IntentRoutingFakeLlmClient(intent_reply="EXECUTE"),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=codex_bridge,
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001)
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="给 10002 加一个今天 8 点的定时私聊提醒",
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
            user_id=10001,
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
        llm_client=IntentRoutingFakeLlmClient(intent_reply="EXECUTE"),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=codex_bridge,
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-fast-feature",
            user_id=10001,
            text=owner_admin_text("把 README 第一行改成 after"),
        )
    )

    assert handled is True
    assert (repo_root / "README.md").read_text(encoding="utf-8") == "before\n"
    assert codex_bridge.prompts == []
    assert len(sender.private_sent) == 1
    assert "把 README 第一行改成 after" in sender.private_sent[0].text
    assert "你回我“好 / 就这样 / 按这个”" in sender.private_sent[0].text
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert [task.intent_type for task in completed] == ["feature_plan"]
    assert queued == []


@pytest.mark.asyncio
async def test_owner_project_mode_plain_feature_request_queues_admin_agent_turn_and_executes_through_codex(
    sqlite_engine, tmp_path
) -> None:
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
        llm_client=IntentRoutingFakeLlmClient(intent_reply="EXECUTE"),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=codex_bridge,
    )
    service._set_owner_private_session_mode(dev_service_module.SESSION_MODE_PROJECT)

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-project-inline-auto",
            user_id=10001,
            text="把 README 第一行改成 after",
        )
    )
    processed = await service.process_next_task_once()

    assert handled is True
    assert processed is True
    assert (repo_root / "README.md").read_text(encoding="utf-8") == "after\n"
    assert codex_bridge.prompts
    assert len(sender.private_sent) == 2
    assert "完了直接回你结果" in sender.private_sent[0].text
    assert sender.private_sent[1].text == "README 我已经直接改好了。"
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert [task.intent_type for task in completed] == ["admin_agent_turn"]
    assert queued == []


@pytest.mark.asyncio
async def test_owner_project_mode_plain_complex_feature_request_queues_admin_agent_turn_without_feature_plan(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=FakeCodexBridge(
            result=CodexTaskResult(
                summary="queued work",
                reply_text="不该直接执行",
                restart_required=False,
            )
        ),
    )
    service._set_owner_private_session_mode(dev_service_module.SESSION_MODE_PROJECT)

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-project-worker-auto",
            user_id=10001,
            text="把整个私聊开发通道重构一下，持续推进到完成并上线",
        )
    )

    assert handled is True
    assert len(sender.private_sent) == 1
    assert "完了直接回你结果" in sender.private_sent[0].text
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert completed == []
    assert len(queued) == 1
    assert queued[0].intent_type == "admin_agent_turn"


@pytest.mark.asyncio
async def test_owner_project_mode_plain_messages_reuse_admin_agent_codex_thread(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    bridge = SequenceCodexBridge(
        results=[
            CodexTaskResult(
                summary="first turn",
                reply_text="第一条处理完了。",
                restart_required=False,
                thread_id="thread-1",
            ),
            CodexTaskResult(
                summary="second turn",
                reply_text="第二条也接着处理完了。",
                restart_required=False,
                thread_id="thread-1",
            ),
        ]
    )
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
    )
    service._set_owner_private_session_mode(dev_service_module.SESSION_MODE_PROJECT)

    await service.handle_private_message(
        make_private_event(message_id="p-project-thread-1", user_id=10001, text="先看一下 README")
    )
    await service.process_next_task_once()
    await service.handle_private_message(
        make_private_event(message_id="p-project-thread-2", user_id=10001, text="然后顺着上一条继续改")
    )
    await service.process_next_task_once()

    assert bridge.resume_thread_ids == [None, "thread-1"]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert [task.intent_type for task in completed] == ["admin_agent_turn", "admin_agent_turn"]


@pytest.mark.asyncio
async def test_owner_project_mode_followup_plain_message_queues_behind_running_admin_agent_turn(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=IntentRoutingFakeLlmClient(intent_reply="EXECUTE"),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=FakeCodexBridge(
            result=CodexTaskResult(
                summary="should not run",
                reply_text="不该重复执行",
                restart_required=False,
            )
        ),
    )
    service._set_owner_private_session_mode(dev_service_module.SESSION_MODE_PROJECT)

    with session_scope(sqlite_engine) as session:
        sessions = DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(
            owner_qq=10001,
            session_mode=dev_service_module.SESSION_MODE_PROJECT,
        )
        running_task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="把整个私聊开发通道重构一下，持续推进到完成并上线",
            intent_type="admin_agent_turn",
            status="running",
        )
        sessions.update_session(session_id=dev_session.id, last_task_id=running_task.id)
        running_task_id = running_task.id

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-project-followup-running",
            user_id=10001,
            text="那你开始完成我说的功能吧",
        )
    )

    assert handled is True
    assert len(sender.private_sent) == 1
    assert "完了直接回你结果" in sender.private_sent[0].text
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
        running = DevTaskRepository(session).list_tasks_by_status("running")
    assert completed == []
    assert [task.intent_type for task in queued] == ["admin_agent_turn"]
    assert [task.id for task in running] == [running_task_id]
    assert [task.intent_type for task in running] == ["admin_agent_turn"]


@pytest.mark.asyncio
async def test_owner_project_mode_followup_plain_message_queues_after_existing_admin_agent_turn(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=IntentRoutingFakeLlmClient(intent_reply="EXECUTE"),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=FakeCodexBridge(
            result=CodexTaskResult(
                summary="should not run",
                reply_text="不该重复排队",
                restart_required=False,
            )
        ),
    )
    service._set_owner_private_session_mode(dev_service_module.SESSION_MODE_PROJECT)

    with session_scope(sqlite_engine) as session:
        sessions = DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(
            owner_qq=10001,
            session_mode=dev_service_module.SESSION_MODE_PROJECT,
        )
        queued_task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="把整个私聊开发通道重构一下，持续推进到完成并上线",
            intent_type="admin_agent_turn",
            status="queued",
        )
        sessions.update_session(session_id=dev_session.id, last_task_id=queued_task.id)
        queued_task_id = queued_task.id

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-project-followup-queued",
            user_id=10001,
            text="那你开始完成我说的功能吧",
        )
    )

    assert handled is True
    assert len(sender.private_sent) == 1
    assert "完了直接回你结果" in sender.private_sent[0].text
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
        running = DevTaskRepository(session).list_tasks_by_status("running")
    assert completed == []
    assert len(queued) == 2
    assert queued[0].id == queued_task_id
    assert [task.intent_type for task in queued] == ["admin_agent_turn", "admin_agent_turn"]
    assert running == []


@pytest.mark.asyncio
async def test_owner_private_admin_confirmation_executes_simple_feature_request_after_plan(sqlite_engine, tmp_path) -> None:
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=codex_bridge,
    )

    await service.handle_private_message(
        make_private_event(
            message_id="p-fast-feature-plan",
            user_id=10001,
            text=owner_admin_text("把 README 第一行改成 after"),
        )
    )
    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-fast-feature-confirm",
            user_id=10001,
            text=owner_admin_text("好"),
        )
    )

    assert handled is True
    assert (repo_root / "README.md").read_text(encoding="utf-8") == "after\n"
    assert codex_bridge.prompts
    assert [outbound.text for outbound in sender.private_sent[-2:]] == [
        "我开始处理这条了。",
        "README 我已经直接改好了。",
    ]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert [task.intent_type for task in completed] == ["feature_plan", "feature_work"]
    assert queued == []


@pytest.mark.asyncio
async def test_owner_private_inline_runtime_change_hands_off_restart_until_service_start(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "app").mkdir()
    (repo_root / "app" / "main.py").write_text("before\n", encoding="utf-8")
    command_calls: list[list[str]] = []

    def fake_command_runner(command: list[str], cwd: Path):
        del cwd
        command_calls.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    codex_bridge = MutatingCodexBridge(
        result=CodexTaskResult(
            summary="updated runtime",
            reply_text="runtime fix applied",
            restart_required=False,
        ),
        relative_path="app/main.py",
        content="after\n",
    )
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=codex_bridge,
        command_runner=fake_command_runner,
        enable_local_worker=False,
    )

    await service.handle_private_message(
        make_private_event(
            message_id="p-runtime-feature-plan",
            user_id=10001,
            text=owner_admin_text("把 app main 改成 after"),
        )
    )
    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-runtime-feature-confirm",
            user_id=10001,
            text=owner_admin_text("好"),
        )
    )

    assert handled is True
    assert (repo_root / "app" / "main.py").read_text(encoding="utf-8") == "after\n"
    assert len(command_calls) == 1
    assert command_calls[0][:3] == ["wsl.exe", "bash", "-lc"]
    assert "xiaomachi-wsl-entry install" in command_calls[0][-1]
    assert [outbound.text for outbound in sender.private_sent[-2:]] == [
        "我开始处理这条了。",
        "我现在重启小町，让改动生效。",
    ]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        running = DevTaskRepository(session).list_tasks_by_status("running")
    assert [task.intent_type for task in completed] == ["feature_plan"]
    assert [task.intent_type for task in running] == ["feature_work"]

    recovery_service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        command_runner=fake_command_runner,
        enable_local_worker=False,
    )
    await recovery_service.start()
    await recovery_service.stop()

    assert len(command_calls) == 1
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        running = DevTaskRepository(session).list_tasks_by_status("running")
    assert [task.intent_type for task in completed] == ["feature_plan", "feature_work"]
    assert running == []
    assert completed[-1].result_text == "runtime fix applied"
    assert completed[-1].restart_required is True
    assert completed[-1].restart_result == "recovered-on-start"
    assert sender.private_sent[-1].text.endswith("runtime fix applied")


@pytest.mark.asyncio
async def test_owner_private_admin_complex_feature_request_asks_for_confirmation_before_queueing_worker_workflow(
    sqlite_engine, tmp_path
) -> None:
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=codex_bridge,
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-worker-feature",
            user_id=10001,
            text=owner_admin_text("把整个私聊开发通道重构一下，持续推进到完成并上线"),
        )
    )

    assert handled is True
    assert len(sender.private_sent) == 1
    assert "持续推进到完成并上线" in sender.private_sent[0].text
    assert codex_bridge.prompts == []
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert [task.intent_type for task in completed] == ["feature_plan"]
    assert queued == []


@pytest.mark.asyncio
async def test_owner_private_admin_confirmation_queues_complex_feature_request_after_plan(sqlite_engine, tmp_path) -> None:
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=codex_bridge,
    )

    await service.handle_private_message(
        make_private_event(
            message_id="p-worker-feature-plan",
            user_id=10001,
            text=owner_admin_text("把整个私聊开发通道重构一下，持续推进到完成并上线"),
        )
    )
    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-worker-feature-confirm",
            user_id=10001,
            text=owner_admin_text("就这样"),
        )
    )

    assert handled is True
    assert codex_bridge.prompts == []
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert [task.intent_type for task in completed] == ["feature_plan"]
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(message_id="p-chat-send", user_id=10001, text='给熟人A私聊发送“你好”')
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
        owner_qq=10001,
        private_chat_qqs={10002},
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-project-send-ok",
            user_id=10001,
            text=owner_admin_text('给 10002 私聊发送“管理员模式使用方法：先发启动管理员模式，再直接说要查什么、改什么。”'),
        )
    )

    assert handled is True
    assert [outbound.user_id for outbound in sender.private_sent] == [10002, 10001]
    assert sender.private_sent[0].text == "管理员模式使用方法：先发启动管理员模式，再直接说要查什么、改什么。"
    assert "已经给 10002 私聊发过去了" in sender.private_sent[1].text
    assert "管理员模式使用方法" in sender.private_sent[1].text
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert completed[-1].intent_type == "project_chat"
    assert "已经给 10002 私聊发过去了" in completed[-1].result_text


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
        owner_qq=10001,
        private_chat_qqs={20002},
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-project-send-denied",
            user_id=10001,
            text=owner_admin_text('给 10002 私聊发送“你好”'),
        )
    )

    assert handled is True
    assert len(sender.private_sent) == 1
    assert sender.private_sent[0].user_id == 10001
    assert "10002" in sender.private_sent[0].text
    assert "不在私聊白名单" in sender.private_sent[0].text
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert completed[-1].intent_type == "project_chat"
    assert "不在私聊白名单" in completed[-1].result_text


@pytest.mark.asyncio
async def test_owner_project_private_send_request_reports_target_delivery_failure(
    sqlite_engine, tmp_path
) -> None:
    sender = SelectiveFailSender(failing_user_ids={10002})
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        private_chat_qqs={10002},
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-project-send-failed",
            user_id=10001,
            text=owner_admin_text('给 10002 私聊发送“你好”'),
        )
    )

    assert handled is True
    assert [outbound.user_id for outbound in sender.private_sent] == [10001]
    assert "10002" in sender.private_sent[0].text
    assert "没发出去" in sender.private_sent[0].text
    assert "send_private_msg failed: target=10002" in sender.private_sent[0].text
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    await service.handle_private_message(
        make_private_event(message_id="p-1", user_id=10001, text="what is the project structure")
    )
    await service.handle_private_message(make_private_event(message_id="p-2", user_id=10001, text="/bot new-session"))

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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    await service.handle_private_message(
        make_private_event(message_id="p-1", user_id=10001, text="what is the project structure")
    )
    await service.handle_private_message(make_private_event(message_id="p-2", user_id=10001, text="清空上下文"))

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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    await service.handle_private_message(
        make_private_event(message_id="p-reset-1", user_id=10001, text="重置会话")
    )
    await service.handle_private_message(
        make_private_event(message_id="p-reset-2", user_id=10001, text="重置会话")
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    await service.handle_private_message(make_private_event(message_id="p-1", user_id=10001, text="/bot session-status"))

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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    await service.handle_private_message(make_private_event(message_id="p-1", user_id=10001, text="会话状态"))

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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
    )

    handled_plan, handled_confirm = await confirm_owner_feature_request(
        service,
        request_text="把这个功能整体修一下并上线",
        request_message_id="p-1",
        confirmation_message_id="p-1-confirm",
    )

    processed = await service.process_next_task_once()

    assert handled_plan is True
    assert handled_confirm is True
    assert processed is True
    assert [outbound.text for outbound in sender.private_sent[-2:]] == [
        "我开始处理这条了。",
        "I checked it. The last problem was a timeout, not a missing message.",
    ]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 2
    assert completed[-1].summary == "checked latest logs"
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001)
        tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    handled_first = await service.handle_private_message(
        make_private_event(message_id="p-send-missing", user_id=10001, text='给熟人A私聊发送“你好”')
    )
    handled_second = await service.handle_private_message(
        make_private_event(
            message_id="p-send-upgrade",
            user_id=10001,
            text=owner_admin_text("开发这个功能"),
        )
    )

    assert handled_first is True
    assert handled_second is True
    assert len(sender.private_sent) == 2
    assert "管理员权限 开发这个功能" in sender.private_sent[0].text
    assert "实现一个新能力" in sender.private_sent[1].text
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        queued = DevTaskRepository(session).list_tasks_by_status("queued")
    assert [task.intent_type for task in completed] == ["project_chat", "feature_plan"]
    assert "实现一个新能力" in completed[-1].raw_request_text
    assert "给熟人A私聊发送“你好”" in completed[-1].raw_request_text
    assert queued == []


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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
    )

    await confirm_owner_feature_request(
        service,
        request_text="把这个功能整体修一下并上线",
        request_message_id="p-exec-1",
        confirmation_message_id="p-exec-1-confirm",
    )
    processed = await service.process_next_task_once()

    assert processed is True
    assert sender.private_sent[-1].text == "先说结论 这个已经改好了。我顺手测过了。"
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert completed[-1].result_text == "先说结论 这个已经改好了。我顺手测过了。"


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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
    )

    await confirm_owner_feature_request(
        service,
        request_text="把这个功能整体修一下并上线",
        request_message_id="p-1",
        confirmation_message_id="p-1-confirm",
    )
    await service.process_next_task_once()
    await confirm_owner_feature_request(
        service,
        request_text="把这个功能整体再修一下并上线",
        request_message_id="p-2",
        confirmation_message_id="p-2-confirm",
    )
    await service.process_next_task_once()

    assert bridge.resume_thread_ids == [None, "thread-1"]


@pytest.mark.asyncio
async def test_admin_agent_turn_runs_codex_bridge_off_event_loop_thread(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    bridge = ThreadCapturingCodexBridge(
        result=CodexTaskResult(summary="done", reply_text="ok", restart_required=False)
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("before\n", encoding="utf-8")
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=IntentRoutingFakeLlmClient(intent_reply="EXECUTE"),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
    )
    service._set_owner_private_session_mode(dev_service_module.SESSION_MODE_PROJECT)

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-admin-thread-1",
            user_id=10001,
            text="把 README 第一行改成 after",
        )
    )
    processed = await service.process_next_task_once()

    assert handled is True
    assert processed is True
    assert bridge.call_thread_ids
    assert bridge.call_thread_ids[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_admin_agent_turn_runs_checkpoint_creation_off_event_loop_thread(sqlite_engine, tmp_path, monkeypatch) -> None:
    sender = FakeSender()
    bridge = FakeCodexBridge(
        result=CodexTaskResult(summary="done", reply_text="ok", restart_required=False)
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("before\n", encoding="utf-8")
    checkpoint_thread_ids: list[int] = []

    def fake_create_repo_checkpoint(*, repo_root: Path, checkpoint_dir: Path):
        del repo_root, checkpoint_dir
        checkpoint_thread_ids.append(threading.get_ident())
        return {"files": []}

    monkeypatch.setattr(dev_service_module, "create_repo_checkpoint", fake_create_repo_checkpoint)

    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=IntentRoutingFakeLlmClient(intent_reply="EXECUTE"),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
    )
    service._set_owner_private_session_mode(dev_service_module.SESSION_MODE_PROJECT)

    handled = await service.handle_private_message(
        make_private_event(
            message_id="p-admin-checkpoint-thread-1",
            user_id=10001,
            text="把 README 第一行改成 after",
        )
    )
    processed = await service.process_next_task_once()

    assert handled is True
    assert processed is True
    assert checkpoint_thread_ids
    assert checkpoint_thread_ids[0] != threading.get_ident()


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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
        command_runner=fake_command_runner,
    )

    await confirm_owner_feature_request(
        service,
        request_text="把这个功能整体修一下并上线",
        request_message_id="p-1",
        confirmation_message_id="p-1-confirm",
    )
    await service.process_next_task_once()

    assert command_calls == [
        ["wsl.exe", "bash", "/usr/local/bin/xiaomachi-wsl-entry", "install"]
    ]
    assert [outbound.text for outbound in sender.private_sent[-3:]] == [
        "我开始处理这条了。",
        "我现在重启小町，让改动生效。",
        "patched it",
    ]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 2
    assert completed[-1].restart_required is True


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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
        command_runner=fake_command_runner,
    )

    await confirm_owner_feature_request(
        service,
        request_text="把这个功能整体修一下并上线",
        request_message_id="p-1",
        confirmation_message_id="p-1-confirm",
    )
    processed = await service.process_next_task_once()

    assert processed is True
    assert command_calls == []
    assert [outbound.text for outbound in sender.private_sent[-2:]] == [
        "我开始处理这条了。",
        "我看完了，现在还不能确认已经生效。",
    ]
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 2
    assert completed[-1].restart_required is False
    assert completed[-1].restart_result == "not-needed"


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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
        command_runner=fake_command_runner,
    )

    await confirm_owner_feature_request(
        service,
        request_text="你确保这个改动加入进去了，如果确定不了保险起见你就重启确保真的加入进去了，并上线",
        request_message_id="p-1",
        confirmation_message_id="p-1-confirm",
    )
    processed = await service.process_next_task_once()

    assert processed is True
    assert command_calls == [
        ["wsl.exe", "bash", "/usr/local/bin/xiaomachi-wsl-entry", "install"]
    ]
    assert sender.private_sent[-3].text == "我开始处理这条了。"
    assert sender.private_sent[-1].text != "代码里已经加进去了，但我不能直接替你执行重启。"
    assert "不能直接替你执行重启" not in sender.private_sent[-1].text
    assert "重启" in sender.private_sent[-1].text
    assert "运行中的实例" in sender.private_sent[-1].text
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 2
    assert "不能直接替你执行重启" not in completed[-1].result_text
    assert completed[-1].restart_required is True
    assert completed[-1].restart_result == "success"


@pytest.mark.asyncio
async def test_service_start_can_disable_local_worker_loop(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
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
        owner_qq=10001,
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
            "wsl.exe",
            "bash",
            "/usr/local/bin/xiaomachi-wsl-entry",
            "start",
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
        owner_qq=10001,
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
        if call_index["value"] == 1:
            return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "install failed"})()
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=bridge,
        command_runner=fake_command_runner,
    )

    await confirm_owner_feature_request(
        service,
        request_text="把整个运行时修一下然后重启上线",
        request_message_id="p-1",
        confirmation_message_id="p-1-confirm",
    )
    await service.process_next_task_once()

    assert (repo_root / "app/runtime_flag.py").read_text(encoding="utf-8") == "before"
    with session_scope(sqlite_engine) as session:
        rolled_back = DevTaskRepository(session).list_tasks_by_status("rolled_back")
    assert len(rolled_back) == 1
    assert len(sender.private_sent) == 4
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        codex_bridge=RaisingCodexBridge("codex executable not found on PATH"),
    )

    await confirm_owner_feature_request(
        service,
        request_text="把这个功能整体修一下并上线",
        request_message_id="p-1",
        confirmation_message_id="p-1-confirm",
    )
    processed = await service.process_next_task_once()

    assert processed is True
    with session_scope(sqlite_engine) as session:
        failed = DevTaskRepository(session).list_tasks_by_status("failed")
    assert len(failed) == 1
    assert "codex executable not found on PATH" in failed[0].failure_reason
    assert len(sender.private_sent) == 3
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
        owner_qq=10001,
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    def exploding_checkpoint(*, repo_root: Path, checkpoint_dir: Path):
        raise OSError("checkpoint failed")

    monkeypatch.setattr(dev_service_module, "create_repo_checkpoint", exploding_checkpoint)

    await confirm_owner_feature_request(
        service,
        request_text="把这个功能整体修一下并上线",
        request_message_id="p-1",
        confirmation_message_id="p-1-confirm",
    )
    processed = await service.process_next_task_once()

    assert processed is True
    with session_scope(sqlite_engine) as session:
        failed = DevTaskRepository(session).list_tasks_by_status("failed")
    assert len(failed) == 1
    assert "checkpoint failed" in failed[0].failure_reason
    assert len(sender.private_sent) == 3
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
        owner_qq=10001,
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

    await confirm_owner_feature_request(
        service,
        request_text="把这个功能整体修一下并上线",
        request_message_id="p-1",
        confirmation_message_id="p-1-confirm",
    )
    processed = await service.process_next_task_once()

    assert processed is True
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 2
    assert completed[-1].result_text == "all good"
    assert len(sender.private_sent) == 3
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        command_runner=fake_command_runner,
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001)
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
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
async def test_service_start_recovers_saved_result_without_restart_for_probe_and_test_scripts(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "scripts").mkdir()
    (repo_root / "tests").mkdir()
    (repo_root / "scripts" / "probe_gpt_image2.py").write_text("print('probe')\n", encoding="utf-8")
    (repo_root / "tests" / "test_probe_gpt_image2.py").write_text("def test_probe():\n    assert True\n", encoding="utf-8")
    command_calls: list[list[str]] = []

    def fake_command_runner(command: list[str], cwd: Path):
        del cwd
        command_calls.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        command_runner=fake_command_runner,
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001)
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="add a dedicated probe script and run it",
            intent_type="feature_work",
        )
        tasks.mark_status(task_id=task.id, status="running")
        task_id = task.id

    artifact_dir = tmp_path / "data" / "dev_control" / "tasks" / f"task-{task_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "codex.last_message.json").write_text(
        '{"summary":"probe ready","reply_text":"probe finished","restart_required":false}',
        encoding="utf-8",
    )

    checkpoint_dir = tmp_path / "data" / "dev_control" / "checkpoints" / f"task-{task_id}"
    (checkpoint_dir / "snapshot").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "manifest.json").write_text('{"files":[]}', encoding="utf-8")

    await service.start()
    await service.stop()

    assert command_calls == []
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 1
    assert completed[0].result_text == "probe finished"
    assert completed[0].restart_required is False
    assert completed[0].restart_result == "not-needed"
    assert sender.private_sent[-1].text.endswith("probe finished")


@pytest.mark.asyncio
async def test_service_start_recovers_saved_result_even_when_local_worker_is_disabled(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        enable_local_worker=False,
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001)
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="recover this inline task",
            intent_type="feature_work",
        )
        tasks.mark_status(task_id=task.id, status="running")
        task_id = task.id

    artifact_dir = tmp_path / "data" / "dev_control" / "tasks" / f"task-{task_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "codex.last_message.json").write_text(
        '{"summary":"recovered","reply_text":"inline recovery works","restart_required":false}',
        encoding="utf-8",
    )

    await service.start()
    await service.stop()

    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 1
    assert completed[0].result_text == "inline recovery works"
    assert sender.private_sent[-1].text.endswith("inline recovery works")


@pytest.mark.asyncio
async def test_service_start_recovery_does_not_restart_again_for_app_changes(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "app").mkdir()
    (repo_root / "app" / "main.py").write_text("print('updated runtime')\n", encoding="utf-8")
    command_calls: list[list[str]] = []

    def fake_command_runner(command: list[str], cwd: Path):
        del cwd
        command_calls.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        command_runner=fake_command_runner,
        enable_local_worker=False,
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001)
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="apply the runtime fix",
            intent_type="feature_work",
        )
        tasks.mark_status(task_id=task.id, status="running")
        task_id = task.id

    artifact_dir = tmp_path / "data" / "dev_control" / "tasks" / f"task-{task_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "codex.last_message.json").write_text(
        '{"summary":"runtime fix applied","reply_text":"runtime fix applied","restart_required":false}',
        encoding="utf-8",
    )

    checkpoint_dir = tmp_path / "data" / "dev_control" / "checkpoints" / f"task-{task_id}"
    (checkpoint_dir / "snapshot").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "manifest.json").write_text('{"files":[]}', encoding="utf-8")

    await service.start()
    await service.stop()

    assert command_calls == []
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 1
    assert completed[0].result_text == "runtime fix applied"
    assert completed[0].restart_required is True
    assert completed[0].restart_result == "recovered-on-start"
    assert sender.private_sent[-1].text.endswith("runtime fix applied")


@pytest.mark.asyncio
async def test_service_start_recovery_rewrites_reply_after_recovered_restart(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "app").mkdir()
    (repo_root / "app" / "main.py").write_text("print('updated runtime')\n", encoding="utf-8")
    command_calls: list[list[str]] = []

    def fake_command_runner(command: list[str], cwd: Path):
        del cwd
        command_calls.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        command_runner=fake_command_runner,
        enable_local_worker=False,
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001)
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="把运行时改动直接上线",
            intent_type="feature_work",
        )
        tasks.mark_status(task_id=task.id, status="running")
        task_id = task.id

    artifact_dir = tmp_path / "data" / "dev_control" / "tasks" / f"task-{task_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "codex.last_message.json").write_text(
        json.dumps(
            {
                "summary": "runtime fix applied",
                "reply_text": "代码里已经加进去了，但我不能直接替你执行重启。",
                "restart_required": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    checkpoint_dir = tmp_path / "data" / "dev_control" / "checkpoints" / f"task-{task_id}"
    (checkpoint_dir / "snapshot").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "manifest.json").write_text('{"files":[]}', encoding="utf-8")

    await service.start()
    await service.stop()

    assert command_calls == []
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 1
    assert completed[0].restart_required is True
    assert completed[0].restart_result == "recovered-on-start"
    assert "不能直接替你执行重启" not in completed[0].result_text
    assert "重启" in completed[0].result_text
    assert "运行中的实例" in completed[0].result_text
    assert "不能直接替你执行重启" not in sender.private_sent[-1].text
    assert "重启" in sender.private_sent[-1].text
    assert "运行中的实例" in sender.private_sent[-1].text


@pytest.mark.asyncio
async def test_service_start_recovery_rewrites_restart_denial_reply_with_rule_based_phrase(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "app").mkdir()
    (repo_root / "app" / "main.py").write_text("print('updated runtime')\n", encoding="utf-8")
    command_calls: list[list[str]] = []

    def fake_command_runner(command: list[str], cwd: Path):
        del cwd
        command_calls.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        command_runner=fake_command_runner,
        enable_local_worker=False,
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001)
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="把私聊生图功能直接上线",
            intent_type="feature_work",
        )
        tasks.mark_status(task_id=task.id, status="running")
        task_id = task.id

    artifact_dir = tmp_path / "data" / "dev_control" / "tasks" / f"task-{task_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "codex.last_message.json").write_text(
        json.dumps(
            {
                "summary": "private image runtime fix applied",
                "reply_text": "已经接上了。要真正生效还得重启私聊运行时；我这边按规则没有替你重启。",
                "restart_required": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    checkpoint_dir = tmp_path / "data" / "dev_control" / "checkpoints" / f"task-{task_id}"
    (checkpoint_dir / "snapshot").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "manifest.json").write_text('{"files":[]}', encoding="utf-8")

    await service.start()
    await service.stop()

    assert command_calls == []
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
    assert len(completed) == 1
    assert completed[0].restart_required is True
    assert completed[0].restart_result == "recovered-on-start"
    assert "按规则没有替你重启" not in completed[0].result_text
    assert "要真正生效还得重启" not in completed[0].result_text
    assert "重启" in completed[0].result_text
    assert "运行中的实例" in completed[0].result_text
    assert "按规则没有替你重启" not in sender.private_sent[-1].text
    assert "要真正生效还得重启" not in sender.private_sent[-1].text
    assert "重启" in sender.private_sent[-1].text
    assert "运行中的实例" in sender.private_sent[-1].text


@pytest.mark.asyncio
async def test_service_start_recovery_rewrites_manual_restart_only_reply_after_restart(
    sqlite_engine, tmp_path
) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "app").mkdir()
    (repo_root / "app" / "main.py").write_text("print('updated runtime')\n", encoding="utf-8")

    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        enable_local_worker=False,
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001)
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="把高清生图默认配置上线",
            intent_type="feature_work",
        )
        tasks.mark_status(task_id=task.id, status="running")
        task_id = task.id

    artifact_dir = tmp_path / "data" / "dev_control" / "tasks" / f"task-{task_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "codex.last_message.json").write_text(
        json.dumps(
            {
                "summary": "high quality image defaults applied",
                "reply_text": (
                    "我不能替你直接重启。前面改的高清生图默认配置要生效，确实需要重启运行时；"
                    "但这条会话的规则就是“不要自己重启运行时”。所以我这边不会假装已经重启，也不会说它已经生效。"
                    "当前准确状态是：\n"
                    "- 代码和默认参数已经改了\n"
                    "- 还需要你那边实际重启运行时\n"
                    "- 重启之后，新生成的图才会吃到 `high + 1536x1024 + png` 这套配置"
                ),
                "restart_required": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    checkpoint_dir = tmp_path / "data" / "dev_control" / "checkpoints" / f"task-{task_id}"
    (checkpoint_dir / "snapshot").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "manifest.json").write_text('{"files":[]}', encoding="utf-8")

    await service.start()
    await service.stop()

    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")

    assert len(completed) == 1
    assert completed[0].restart_required is True
    assert completed[0].restart_result == "recovered-on-start"
    assert "我不能替你直接重启" not in completed[0].result_text
    assert "还需要你那边实际重启运行时" not in completed[0].result_text
    assert "不会假装已经重启" not in completed[0].result_text
    assert "不要自己重启运行时" not in completed[0].result_text
    assert "重启" in completed[0].result_text
    assert "运行中的实例" in completed[0].result_text
    assert "我不能替你直接重启" not in sender.private_sent[-1].text
    assert "还需要你那边实际重启运行时" not in sender.private_sent[-1].text
    assert "不会假装已经重启" not in sender.private_sent[-1].text
    assert "不要自己重启运行时" not in sender.private_sent[-1].text
    assert "重启" in sender.private_sent[-1].text
    assert "运行中的实例" in sender.private_sent[-1].text


@pytest.mark.asyncio
async def test_service_start_recovery_skips_duplicate_reply_when_completed_message_was_already_sent(
    sqlite_engine, tmp_path
) -> None:
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
        command_runner=fake_command_runner,
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        messages = dev_service_module.MessageRepository(session)
        users = dev_service_module.UserRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001)
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
            raw_request_text="fix this",
            intent_type="feature_work",
        )
        tasks.mark_status(task_id=task.id, status="running")
        task_id = task.id
        users.upsert_user(user_id=10001, nickname="bot", group_card="")
        messages.add_private_message(
            platform_msg_id=f"private-outbound-dev_task:{task_id}:completed",
            user_id=10001,
            timestamp=datetime.now().astimezone(),
            plain_text="补发结果",
            raw_json={
                "direction": "outbound",
                "recipient_user_id": 10001,
                "delivery_state": "sent",
                "context": f"dev_task:{task_id}:completed",
            },
            msg_type="text",
            reply_to_msg_id=None,
            mentioned_bot=False,
        )

    artifact_dir = tmp_path / "data" / "dev_control" / "tasks" / f"task-{task_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "codex.last_message.json").write_text(
        json.dumps(
            {
                "summary": "recovered",
                "reply_text": "补发结果",
                "restart_required": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    await service.start()
    await service.stop()

    assert command_calls == []
    assert sender.private_sent == []
    with session_scope(sqlite_engine) as session:
        completed = DevTaskRepository(session).list_tasks_by_status("completed")
        messages = dev_service_module.MessageRepository(session)
        completed_message = messages.get_by_platform_msg_id(f"private-outbound-dev_task:{task_id}:completed")
        recovered_message = messages.get_by_platform_msg_id(f"private-outbound-dev_task:{task_id}:recovered_on_start")
    assert len(completed) == 1
    assert completed[0].result_text == "补发结果"
    assert completed_message is not None
    assert recovered_message is None


@pytest.mark.asyncio
async def test_service_start_marks_stale_running_non_execute_task_failed(sqlite_engine, tmp_path) -> None:
    sender = FakeSender()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    service = DevControlService(
        engine=sqlite_engine,
        sender=sender,
        llm_client=FakeLlmClient(),
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001)
        task = tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
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
        owner_qq=10001,
        repo_root=repo_root,
        data_dir=tmp_path / "data",
    )

    with session_scope(sqlite_engine) as session:
        sessions = dev_service_module.DevSessionRepository(session)
        tasks = DevTaskRepository(session)
        dev_session = sessions.get_or_create_owner_session(owner_qq=10001)
        tasks.add_task(
            session_id=dev_session.id,
            requested_by_qq=10001,
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
        owner_qq=10001,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir()

    delivered = await service._send_private_text(
        user_id=10001,
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
        owner_qq=10001,
        bot_qq=1807533371,
        repo_root=tmp_path / "repo",
        data_dir=tmp_path / "data",
    )
    (tmp_path / "repo").mkdir()

    first = await service._send_private_text(
        user_id=10001,
        text="dedup reply",
        context="project_chat:123:completed",
    )
    second = await service._send_private_text(
        user_id=10001,
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
