from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import re
import subprocess
from typing import Callable

from app.adapters.onebot_models import PrivateMessageEvent
from app.adapters.sender import OutboundPrivateMessage
from app.core.chat_style import normalize_chat_reply
from app.core.group_image_generation import ImageJobResult, PrivateImageGenerationRequest, PrivateImageGenerationService
from app.core.image_turn_resolver import resolve_private_images_for_turn
from app.core.persona_engine import render_persona, render_safety_lines
from app.core.router import (
    AUTO_WEB_REFERENCE_LEADING_CONNECTOR_PATTERN,
    AUTO_WEB_REFERENCE_QUERY_PATTERN,
    GROUP_IMAGE_NEGATIVE_PATTERNS,
    GROUP_IMAGE_REFERENCE_CONTEXT_KEYWORDS,
    GROUP_IMAGE_REFERENCE_GENERATION_KEYWORDS,
    GROUP_IMAGE_REFERENCE_INTENT_KEYWORDS,
    GROUP_IMAGE_REQUEST_PATTERNS,
    LOOKUP_NORMALIZER,
)
from app.core.search_policy import (
    SearchDecision,
    build_current_datetime_facts,
    build_forced_search_query,
    build_search_decision_prompt,
    is_explicit_search_request,
    is_general_search_decision_candidate,
    is_search_verification_query,
    is_time_sensitive_request,
    needs_current_datetime_context,
    needs_external_lookup_search,
    needs_reference_search,
    normalize_relative_time_query,
    parse_search_decision,
)
from app.core.web_grounding import build_grounding_notes
from app.dev_control.admin_agent_runtime import ADMIN_AGENT_INTENT, ADMIN_AGENT_ACK_TEXT, AdminAgentRuntime
from app.dev_control.checkpoints import create_repo_checkpoint, restore_repo_checkpoint
from app.dev_control.codex_bridge import CodexBridge, CodexTaskResult
from app.dev_control.repo_context import build_repo_context_snippets
from app.storage.db import session_scope
from app.storage.models import DevSession
from app.storage.repositories import (
    DevSessionRepository,
    DevTaskArtifactRepository,
    DevTaskRepository,
    MessageRepository,
    UserRepository,
)

logger = logging.getLogger(__name__)

SESSION_NEW_COMMANDS = {
    "/bot new-session",
    "/bot reset-session",
    "清空上下文",
    "开新对话",
    "开启新对话",
    "新建对话",
    "重开对话",
    "重置会话",
}
SESSION_STATUS_COMMANDS = {
    "/bot session-status",
    "会话状态",
    "查看会话状态",
}
PROJECT_SESSION_NEW_COMMANDS = SESSION_NEW_COMMANDS | {
    "/bot new-project-session",
    "清空项目上下文",
    "重置项目会话",
    "重开项目对话",
    "新建项目对话",
}
PROJECT_SESSION_STATUS_COMMANDS = SESSION_STATUS_COMMANDS | {
    "/bot project-session-status",
    "项目会话状态",
    "查看项目会话状态",
}
OWNER_ADMIN_PREFIX = "管理员权限"
OWNER_ADMIN_CONTINUE_CONFIRMATIONS = {
    "好",
    "好的",
    "好的呀",
    "行",
    "可以",
    "继续",
    "继续吧",
    "嗯",
    "嗯嗯",
    "开始吧",
    "去吧",
    "查吧",
}
PRIVATE_IMAGE_RETOUCH_INTENT_KEYWORDS = (
    "轻度人像优化",
    "人像优化",
    "优化一下人脸",
    "优化人脸",
    "稍微调整一下五官",
    "修一下鼻毛",
    "修下鼻毛",
    "修一下胡须",
    "修下胡须",
    "修图",
    "精修",
    "小修",
    "润色",
)

SESSION_NEW_COMMANDS = {
    "/bot new-session",
    "/bot reset-session",
    "清空上下文",
    "开新对话",
    "开启新对话",
    "新建对话",
    "重开对话",
    "重置会话",
}
SESSION_STATUS_COMMANDS = {
    "/bot session-status",
    "会话状态",
    "查看会话状态",
}
PROJECT_SESSION_NEW_COMMANDS = SESSION_NEW_COMMANDS | {
    "/bot new-project-session",
    "清空项目上下文",
    "重置项目会话",
    "重开项目对话",
    "新建项目对话",
}
PROJECT_SESSION_STATUS_COMMANDS = SESSION_STATUS_COMMANDS | {
    "/bot project-session-status",
    "项目会话状态",
    "查看项目会话状态",
}
PRIVATE_DRAW_RESET_COMMANDS = {
    "/bot reset-draw",
    "重置绘画",
    "清空绘画",
    "清空画图",
    "重置画图",
}
OWNER_MODE_ENABLE_COMMANDS = {"启动管理员模式"}
OWNER_MODE_DISABLE_COMMANDS = {
    "结束管理员模式",
    "退出管理员模式",
    "关闭管理员模式",
}
OWNER_ADMIN_PREFIX = "管理员权限"
OWNER_ADMIN_CONTINUE_CONFIRMATIONS = {
    "好",
    "好的",
    "好的呀",
    "行",
    "可以",
    "继续",
    "继续吧",
    "嗯",
    "嗯嗯",
    "开始吧",
    "去吧",
    "查吧",
}

EXECUTE_KEYWORDS = (
    "fix",
    "patch",
    "implement",
    "deploy",
    "release",
    "commit",
    "push",
    "refactor",
    "rewrite",
    "update",
    "remove",
    "delete",
    "pytest",
    "test",
    "tests",
    "修",
    "改",
    "实现",
    "新增",
    "增加",
    "删除",
    "上线",
    "提交",
    "推送",
    "重构",
    "测试",
)
RESTART_KEYWORDS = ("restart", "重启")
RESTART_STATUS_HINTS = (
    "成功了吗",
    "生效了吗",
    "好了没",
    "好了吗",
    "有没有",
    "是否",
    "是不是",
    "能不能确认",
    "确认一下",
    "确认下",
    "结果怎么样",
)
CHANGE_STATUS_HINTS = (
    "生效",
    "改动",
    "改好",
    "加上",
    "加入",
    "会@",
    "会不会@",
    "会不会艾特",
    "会艾特",
    "会不会提到",
    "会提到",
)
CHANGE_STATUS_TARGET_HINTS = (
    "设置",
    "设定",
    "功能",
    "逻辑",
    "配置",
    "代码",
    "人格",
    "仓库",
    "规则",
    "熟人A",
)
INSPECT_KEYWORDS = (
    "log",
    "logs",
    "debug",
    "error",
    "traceback",
    "stderr",
    "stdout",
    "status",
    "state",
    "process",
    "pid",
    "port",
    "git status",
    "git diff",
    "diff",
    "grep",
    "search",
    "code",
    "source",
    "file",
    "worktree",
    "未提交",
    "改动",
    "日志",
    "报错",
    "卡住",
    "排查",
    "状态",
    "进程",
    "端口",
    "代码",
    "源码",
    "文件",
    "路径",
    "没反应",
    "不回",
)
LOG_KEYWORDS = (
    "log",
    "logs",
    "debug",
    "error",
    "traceback",
    "stderr",
    "stdout",
    "日志",
    "报错",
    "卡住",
    "排查",
    "没反应",
    "不回",
)
GIT_STATUS_KEYWORDS = ("git", "diff", "worktree", "未提交", "改动")
CODE_LOOKUP_KEYWORDS = ("code", "source", "file", "grep", "search", "代码", "源码", "文件", "路径")
CONFIG_LOOKUP_KEYWORDS = (
    "config",
    "permission",
    "permissions",
    "whitelist",
    "allowlist",
    "private_chat_qqs",
    "private_chat_whitelist",
    "配置",
    "权限",
    "白名单",
    "名单",
    "给没给",
    "有没有",
    "生效",
)
CAPABILITY_PROBE_TARGET_HINTS = (
    "api",
    "endpoint",
    "model",
    "models",
    "image",
    "gpt-image-2",
    "gpt image 2",
    "接口",
    "模型",
    "出图",
    "绘图",
)
CAPABILITY_PROBE_ACTION_HINTS = (
    "probe",
    "test",
    "run",
    "verify",
    "check",
    "可用",
    "能用",
    "跑",
    "测试",
    "探测",
    "验证",
    "看看",
    "查查",
    "刚加",
)
PROJECT_RUNTIME_HINTS = (
    "小町",
    "群聊",
    "私聊",
    "回复",
    "消息",
    "bot",
    "runtime",
    "router",
    "日志",
    "进程",
    "配置",
    "启动",
    "运行",
    "运作",
    "仓库",
    "项目",
    "工作区",
    "代码",
)
PROJECT_HEALTH_HINTS = (
    "检查",
    "看一下",
    "看看",
    "正常",
    "运作",
    "运行",
    "工作",
    "可用",
    "异常",
    "为什么",
    "没反应",
    "不回",
    "状态",
)
OWNER_FAST_CHAT_HINTS = (
    "你好",
    "您好",
    "哈喽",
    "在吗",
    "你是谁",
    "你叫什么",
    "你叫啥",
    "谢谢",
    "辛苦了",
    "我喜欢你",
    "我爱你",
)
LEGACY_ASYNC_EXECUTE_INTENTS = ("feature_work", "restart_only")
ASYNC_EXECUTE_INTENTS = LEGACY_ASYNC_EXECUTE_INTENTS + (ADMIN_AGENT_INTENT,)
GLOBAL_ADMIN_MUTEX_INTENTS = ASYNC_EXECUTE_INTENTS
FEATURE_PLAN_INTENT = "feature_plan"
SESSION_MODE_DAILY = "daily"
SESSION_MODE_PROJECT = "project"
PRIVATE_SCOPE_OWNER_DAILY = "owner_daily"
PRIVATE_SCOPE_OWNER_PROJECT = "owner_project"
PRIVATE_SCOPE_ALLOWLIST_DAILY = "allowlist_daily"
RESTART_DENIAL_REPLY_FRAGMENTS = (
    "不能直接替你执行重启",
    "不能替你执行重启",
    "不能替你直接重启",
    "我不能替你直接重启",
    "不能直接重启",
    "不能自己重启",
    "没有重启权限",
    "没权限重启",
    "我不能重启",
    "我不能自己重启",
    "我这边按规则没有替你重启",
    "按规则没有替你重启",
    "没有替你重启",
    "要真正生效还得重启",
    "要生效还需要重启",
    "确实需要重启运行时",
    "还需要重启运行时",
    "还需要你那边实际重启",
    "还需要你那边实际重启运行时",
    "如果运行时还没重启",
    "如果还没重启",
    "还没重启的话",
    "不要自己重启运行时",
    "不会假装已经重启",
    "不会说它已经生效",
)
SUMMARY_LINE_LIMIT = 14
RECENT_TURN_LIMIT = 8
LOG_TAIL_LINE_LIMIT = 80
LOG_TAIL_CHAR_LIMIT = 4000


@dataclass(slots=True)
class ScriptRunResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(slots=True)
class ExplicitPrivateSendRequest:
    target_user_id: int
    message_text: str


@dataclass(slots=True)
class InspectionSnapshot:
    prompt_block: str
    files_read: list[str]
    commands_run: list[str]


@dataclass(slots=True)
class PrivateWebContext:
    runtime_facts: list[str]
    web_results: list[str]
    web_pages: list[str]
    grounding_notes: list[str]


class DevControlService:
    def __init__(
        self,
        *,
        engine,
        sender,
        llm_client,
        image_llm_client=None,
        owner_qq: int,
        bot_qq: int | None = None,
        private_chat_qqs: set[int] | None = None,
        admin_qqs: set[int] | None = None,
        repo_root: Path,
        data_dir: Path,
        codex_bridge: CodexBridge | None = None,
        command_runner: Callable[[list[str], Path], ScriptRunResult] | None = None,
        poll_interval_seconds: float = 1.0,
        enable_local_worker: bool = True,
        private_image_followup_window_seconds: float = 1.2,
        web_search_client=None,
        image_model: str = "gpt-image-2",
        image_size: str | None = "auto",
        image_quality: str | None = "high",
        image_background: str | None = None,
        image_output_format: str | None = "png",
        image_output_compression: int | None = 100,
        image_moderation: str | None = "low",
        image_queue_capacity: int = 3,
        image_max_attempts: int = 1,
        image_timeout_seconds: float = 900.0,
        assistant_name: str = "Codex",
        persona: dict | None = None,
        safety: dict | None = None,
    ) -> None:
        self.engine = engine
        self.sender = sender
        self.llm_client = llm_client
        self.image_llm_client = image_llm_client or llm_client
        self.owner_qq = owner_qq
        self.bot_qq = bot_qq
        self.admin_qqs = {qq for qq in (admin_qqs or set()) if qq != owner_qq}
        self.private_chat_qqs = {qq for qq in (private_chat_qqs or set()) if qq != owner_qq}
        self.repo_root = repo_root.resolve()
        self.data_dir = data_dir.resolve()
        self.admin_agent_runtime = AdminAgentRuntime(engine=engine, repo_root=self.repo_root)
        self.codex_bridge = codex_bridge
        self.command_runner = command_runner or self._default_command_runner
        self.poll_interval_seconds = poll_interval_seconds
        self.enable_local_worker = enable_local_worker
        self.private_image_followup_window_seconds = max(0.0, float(private_image_followup_window_seconds))
        self.web_search_client = web_search_client
        self.private_image_service = PrivateImageGenerationService(
            engine=engine,
            llm_client=self.image_llm_client,
            sender=sender,
            web_search_client=web_search_client,
            output_dir=self.data_dir / "generated_private_images",
            model=image_model,
            size=image_size,
            quality=image_quality,
            background=image_background,
            output_format=image_output_format,
            output_compression=image_output_compression,
            moderation=image_moderation,
            max_slots=image_queue_capacity,
            image_max_attempts=image_max_attempts,
            image_timeout_seconds=image_timeout_seconds,
            task_result_callback=self._finalize_private_image_task,
        )
        self.assistant_name = assistant_name.strip() or "Codex"
        self.persona = dict(persona or {})
        self.safety = dict(safety or {})
        self._stop_event = asyncio.Event()
        self._worker_task: asyncio.Task | None = None
        self._pending_private_image_turns: dict[tuple[int, str], tuple[asyncio.Task, PrivateMessageEvent]] = {}
        self._private_image_turn_overrides: dict[int, list] = {}
        self._private_draw_context_reset_users: set[int] = set()

    @property
    def control_dir(self) -> Path:
        return self.data_dir / "dev_control"

    @property
    def task_dir(self) -> Path:
        return self.control_dir / "tasks"

    @property
    def checkpoint_root(self) -> Path:
        return self.control_dir / "checkpoints"

    @property
    def codex_thread_state_path(self) -> Path:
        return self.control_dir / "codex_threads.json"

    @property
    def owner_private_mode_state_path(self) -> Path:
        return self.control_dir / "owner_private_mode.json"

    @property
    def private_admin_intro_state_path(self) -> Path:
        return self.control_dir / "private_admin_intro_state.json"

    async def start(self) -> None:
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        await self.private_image_service.start()
        await self._recover_running_tasks()
        await self._maybe_send_private_admin_intro_messages()
        if not self.enable_local_worker:
            return
        self._stop_event.clear()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        self._cancel_pending_private_image_turns_for_user(None, preserve_for_next_turn=False)
        if not self.enable_local_worker or self._worker_task is None:
            await self.private_image_service.stop()
            return
        self._stop_event.set()
        await self._worker_task
        self._worker_task = None
        await self.private_image_service.stop()

    async def handle_private_message(self, event: PrivateMessageEvent) -> bool:
        if getattr(event, "reply_to_msg_id", None) is not None or bool(list(getattr(event, "images", []) or [])):
            self._private_draw_context_reset_users.discard(event.user_id)
        if self._should_hold_private_image_for_followup(event):
            self._cancel_pending_private_image_turns_for_user(
                event.user_id,
                preserve_for_next_turn=False,
            )
            self._private_image_turn_overrides[event.user_id] = list(event.images)
            return True
        if self._should_defer_private_image_turn(event):
            self._private_image_turn_overrides.pop(event.user_id, None)
        self._cancel_pending_private_image_turns_for_user(
            event.user_id,
            preserve_for_next_turn=not self._should_defer_private_image_turn(event),
        )
        if self._is_private_admin_user(event.user_id):
            mode_switch_reply = self._handle_owner_mode_switch_command(
                raw_text=event.plain_text,
                user_id=event.user_id,
            )
            if mode_switch_reply is not None:
                await self._send_private_text(
                    user_id=event.user_id,
                    text=mode_switch_reply,
                    context=f"private_mode_switch:{event.platform_msg_id}",
                )
                return True

            admin_event = self._build_owner_admin_event(event)
            active_session_mode = (
                SESSION_MODE_PROJECT
                if admin_event is not None
                else self._get_owner_private_session_mode(user_id=event.user_id)
            )
            target_event = admin_event or event
            draw_reset_reply = self._handle_private_draw_reset_command(
                raw_text=target_event.plain_text,
                user_id=event.user_id,
            )
            if draw_reset_reply is not None:
                await self._send_private_text(
                    user_id=event.user_id,
                    text=draw_reset_reply,
                    context=f"private_draw_reset:{target_event.platform_msg_id}",
                )
                return True
            command_reply = self._handle_private_session_command(
                raw_text=target_event.plain_text,
                user_id=event.user_id,
                session_mode=active_session_mode,
                session_label="项目对话" if active_session_mode == SESSION_MODE_PROJECT else "日常对话",
                new_commands=(
                    PROJECT_SESSION_NEW_COMMANDS
                    if active_session_mode == SESSION_MODE_PROJECT
                    else SESSION_NEW_COMMANDS
                ),
                status_commands=(
                    PROJECT_SESSION_STATUS_COMMANDS
                    if active_session_mode == SESSION_MODE_PROJECT
                    else SESSION_STATUS_COMMANDS
                ),
            )
            if command_reply is not None:
                await self._send_private_text(
                    user_id=event.user_id,
                    text=command_reply,
                    context=f"private_session_command:{active_session_mode}:{target_event.platform_msg_id}",
                )
                return True

            if active_session_mode == SESSION_MODE_DAILY and admin_event is None:
                if await self._schedule_private_image_followup_if_needed(
                    event,
                    private_scope=PRIVATE_SCOPE_OWNER_DAILY,
                ):
                    return True
                await self._handle_private_chat_turn(event, private_scope=PRIVATE_SCOPE_OWNER_DAILY)
                return True

            project_event = target_event
            if active_session_mode == SESSION_MODE_PROJECT and admin_event is None:
                await self._handle_admin_agent_turn(project_event)
                return True

            if admin_event is not None and not project_event.plain_text.strip():
                await self._send_private_text(
                    user_id=event.user_id,
                    text="你在“管理员权限”后面直接说要我做什么就行。",
                    context="owner_admin_prefix_empty",
                )
                return True

            capability_upgrade_request = self._build_capability_upgrade_request_from_recent_context(
                owner_qq=project_event.user_id,
                raw_text=project_event.plain_text,
            )
            if capability_upgrade_request is not None:
                await self._dispatch_feature_work_request(
                    event=project_event,
                    request_text=capability_upgrade_request,
                    routing_text=capability_upgrade_request,
                )
                return True

            followup_request = self._build_owner_admin_followup_from_recent_context(
                owner_qq=project_event.user_id,
                raw_text=project_event.plain_text,
            )
            if followup_request is not None:
                followup_intent, followup_text, routing_text = followup_request
                followup_event = replace(project_event, plain_text=followup_text)
                if followup_intent == "project_inspect":
                    await self._handle_project_inspect_turn(followup_event)
                    return True
                if followup_intent == "project_chat":
                    if await self._schedule_private_image_followup_if_needed(
                        followup_event,
                        private_scope=PRIVATE_SCOPE_OWNER_PROJECT,
                    ):
                        return True
                    await self._handle_private_chat_turn(followup_event, private_scope=PRIVATE_SCOPE_OWNER_PROJECT)
                    return True
                await self._dispatch_feature_work_request(
                    event=followup_event,
                    request_text=followup_text,
                    routing_text=routing_text,
                    confirmed=True,
                )
                return True

            if self._parse_explicit_private_send_request(project_event.plain_text) is not None:
                if await self._schedule_private_image_followup_if_needed(
                    project_event,
                    private_scope=PRIVATE_SCOPE_OWNER_PROJECT,
                ):
                    return True
                await self._handle_private_chat_turn(project_event, private_scope=PRIVATE_SCOPE_OWNER_PROJECT)
                return True

            intent_type = self._classify_intent(project_event.plain_text)
            if intent_type == "project_chat":
                if await self._schedule_private_image_followup_if_needed(
                    project_event,
                    private_scope=PRIVATE_SCOPE_OWNER_PROJECT,
                ):
                    return True
                await self._handle_private_chat_turn(project_event, private_scope=PRIVATE_SCOPE_OWNER_PROJECT)
                return True
            if intent_type == "project_inspect":
                await self._handle_project_inspect_turn(project_event)
                return True
            if intent_type == "feature_work":
                await self._dispatch_feature_work_request(
                    event=project_event,
                    request_text=project_event.plain_text,
                    routing_text=project_event.plain_text,
                )
                return True

            task_id = self._queue_execute_task(event=project_event, intent_type=intent_type)
            return True

            command_reply = self._handle_private_session_command(
                raw_text=event.plain_text,
                session_mode=SESSION_MODE_DAILY,
                session_label="日常对话",
                new_commands=SESSION_NEW_COMMANDS,
                status_commands=SESSION_STATUS_COMMANDS,
            )
            if command_reply is not None:
                await self._send_private_text(
                    user_id=event.user_id,
                    text=command_reply,
                    context=f"private_session_command:daily:{event.platform_msg_id}",
                )
                return True

            admin_event = self._build_owner_admin_event(event)
            if admin_event is None:
                await self._handle_private_chat_turn(
                    event,
                    private_scope=PRIVATE_SCOPE_OWNER_DAILY,
                )
                return True
            project_command_reply = self._handle_private_session_command(
                raw_text=admin_event.plain_text,
                session_mode=SESSION_MODE_PROJECT,
                session_label="项目对话",
                new_commands=PROJECT_SESSION_NEW_COMMANDS,
                status_commands=PROJECT_SESSION_STATUS_COMMANDS,
            )
            if project_command_reply is not None:
                await self._send_private_text(
                    user_id=event.user_id,
                    text=project_command_reply,
                    context=f"private_session_command:project:{admin_event.platform_msg_id}",
                )
                return True
            if not admin_event.plain_text.strip():
                await self._send_private_text(
                    user_id=event.user_id,
                    text="你在“管理员权限”后面直接说要我做什么就行。",
                    context="owner_admin_prefix_empty",
                )
                return True

            capability_upgrade_request = self._build_capability_upgrade_request_from_recent_context(
                owner_qq=admin_event.user_id,
                raw_text=admin_event.plain_text,
            )
            if capability_upgrade_request is not None:
                task_id = self._queue_execute_task(
                    event=admin_event,
                    intent_type="feature_work",
                    request_text=capability_upgrade_request,
                )
                return True

            followup_request = self._build_owner_admin_followup_from_recent_context(
                owner_qq=admin_event.user_id,
                raw_text=admin_event.plain_text,
            )
            if followup_request is not None:
                followup_intent, followup_text = followup_request
                followup_event = replace(admin_event, plain_text=followup_text)
                if followup_intent == "project_inspect":
                    await self._handle_project_inspect_turn(followup_event)
                    return True
                if followup_intent == "project_chat":
                    await self._handle_private_chat_turn(
                        followup_event,
                        private_scope=PRIVATE_SCOPE_OWNER_PROJECT,
                    )
                    return True
                task_id = self._queue_execute_task(
                    event=followup_event,
                    intent_type=followup_intent,
                    request_text=followup_text,
                )
                return True

            if self._parse_explicit_private_send_request(admin_event.plain_text) is not None:
                await self._handle_private_chat_turn(
                    admin_event,
                    private_scope=PRIVATE_SCOPE_OWNER_PROJECT,
                )
                return True

            intent_type = self._classify_intent(admin_event.plain_text)
            if intent_type == "project_chat":
                await self._handle_private_chat_turn(
                    admin_event,
                    private_scope=PRIVATE_SCOPE_OWNER_PROJECT,
                )
                return True
            if intent_type == "project_inspect":
                await self._handle_project_inspect_turn(admin_event)
                return True

            task_id = self._queue_execute_task(event=admin_event, intent_type=intent_type)
            return True

        if event.user_id not in self.private_chat_qqs:
            return False

        if await self._schedule_private_image_followup_if_needed(
            event,
            private_scope=PRIVATE_SCOPE_ALLOWLIST_DAILY,
        ):
            return True
        await self._handle_private_chat_turn(event, private_scope=PRIVATE_SCOPE_ALLOWLIST_DAILY)
        return True

    async def process_next_task_once(self) -> bool:
        admin_work_item = self.admin_agent_runtime.claim_next_turn()
        if admin_work_item is not None:
            return await self._run_admin_agent_turn(
                task_id=admin_work_item.task_id,
                session_id=admin_work_item.session_id,
                owner_qq=admin_work_item.user_id,
                request_text=admin_work_item.request_text,
            )

        with session_scope(self.engine) as session:
            tasks = DevTaskRepository(session)
            task = tasks.claim_oldest_queued_task(intent_types=list(LEGACY_ASYNC_EXECUTE_INTENTS))
            if task is None:
                return False
            session_id = task.session_id
            task_id = task.id
            owner_qq = task.requested_by_qq
            request_text = task.raw_request_text
            intent_type = task.intent_type

        return await self._run_execute_task(
            task_id=task_id,
            session_id=session_id,
            owner_qq=owner_qq,
            request_text=request_text,
            intent_type=intent_type,
        )

    async def _run_execute_task(
        self,
        *,
        task_id: int,
        session_id: int,
        owner_qq: int,
        request_text: str,
        intent_type: str,
    ) -> bool:
        artifact_dir = self.task_dir / f"task-{task_id}"
        checkpoint_dir = self.checkpoint_root / f"task-{task_id}"
        checkpoint_manifest: dict[str, list[str]] | None = None
        resume_thread_id = self._get_codex_thread_id(session_id=session_id)

        try:
            logger.info("dev_task_stage task_id=%s stage=claimed", task_id)
            await self._send_private_text(
                user_id=owner_qq,
                text="我开始处理这条了。",
                context=f"dev_task:{task_id}:claimed",
                timeout_seconds=5.0,
            )
            if intent_type == "restart_only":
                logger.info("dev_task_stage task_id=%s stage=restarting_runtime", task_id)
                await self._send_private_text(
                    user_id=owner_qq,
                    text="我现在重启小町，让改动生效。",
                    context=f"dev_task:{task_id}:restarting_runtime",
                    timeout_seconds=5.0,
                )
                if not self.enable_local_worker:
                    final_reply_text = "已经重启完了。"
                    self._write_saved_task_result(
                        task_id=task_id,
                        result=CodexTaskResult(
                            summary=self._build_turn_summary(request_text, final_reply_text),
                            reply_text=final_reply_text,
                            restart_required=True,
                        ),
                    )
                    restart_ok, restart_result_text = self._handoff_inline_runtime_restart()
                    if not restart_ok:
                        with session_scope(self.engine) as session:
                            tasks = DevTaskRepository(session)
                            tasks.mark_failed(
                                task_id=task_id,
                                failure_reason=restart_result_text,
                                checkpoint_dir="",
                            )
                            self._append_session_summary(
                                session_id=session_id,
                                owner_text=request_text,
                                assistant_text=f"失败：{restart_result_text}",
                                sessions=DevSessionRepository(session),
                            )
                        await self._send_private_text(
                            user_id=owner_qq,
                            text=f"这次重启没成功：{restart_result_text}",
                            context=f"dev_task:{task_id}:failed",
                            timeout_seconds=8.0,
                        )
                    else:
                        logger.info("dev_task_stage task_id=%s stage=restart_handed_off", task_id)
                    return True
                restart_ok, restart_result_text = self._restart_runtime()
                commands_run = ["xiaomachi-wsl-entry.sh stop", "xiaomachi-wsl-entry.sh start"]
                if not restart_ok:
                    with session_scope(self.engine) as session:
                        tasks = DevTaskRepository(session)
                        tasks.mark_failed(
                            task_id=task_id,
                            failure_reason=restart_result_text,
                            checkpoint_dir="",
                        )
                        self._append_session_summary(
                            session_id=session_id,
                            owner_text=request_text,
                            assistant_text=f"失败：{restart_result_text}",
                            sessions=DevSessionRepository(session),
                        )
                    await self._send_private_text(
                        user_id=owner_qq,
                        text=f"这次重启没成功：{restart_result_text}",
                        context=f"dev_task:{task_id}:failed",
                        timeout_seconds=8.0,
                    )
                    return True

                final_reply_text = "已经重启完了。"
                with session_scope(self.engine) as session:
                    tasks = DevTaskRepository(session)
                    tasks.mark_completed(
                        task_id=task_id,
                        summary=self._build_turn_summary(request_text, final_reply_text),
                        result_text=final_reply_text,
                        files_read=[],
                        files_changed=[],
                        commands_run=commands_run,
                        restart_required=True,
                        restart_result=restart_result_text,
                        checkpoint_dir="",
                    )
                    self._append_session_summary(
                        session_id=session_id,
                        owner_text=request_text,
                        assistant_text=final_reply_text,
                        sessions=DevSessionRepository(session),
                    )
                logger.info("dev_task_stage task_id=%s stage=completed", task_id)
                await self._send_private_text(
                    user_id=owner_qq,
                    text=final_reply_text,
                    context=f"dev_task:{task_id}:completed",
                    timeout_seconds=8.0,
                )
                return True

            prompt = self._build_execute_prompt(session_id=session_id, task_id=task_id, request_text=request_text)
            return await self._run_codex_task_with_prompt(
                task_id=task_id,
                session_id=session_id,
                owner_qq=owner_qq,
                request_text=request_text,
                prompt=prompt,
            )
        except Exception as exc:
            logger.exception("dev_task_failed task_id=%s", task_id)
            failure_reason = str(exc)
            with session_scope(self.engine) as session:
                DevTaskRepository(session).mark_failed(
                    task_id=task_id,
                    failure_reason=failure_reason,
                    checkpoint_dir="",
                )
                self._append_session_summary(
                    session_id=session_id,
                    owner_text=request_text,
                    assistant_text=f"失败：{failure_reason}",
                    sessions=DevSessionRepository(session),
                )
            await self._send_private_text(
                user_id=owner_qq,
                text=f"这次没跑通，我先记下来：{failure_reason}",
                context=f"dev_task:{task_id}:failed",
                timeout_seconds=8.0,
            )
            return True

    async def _run_admin_agent_turn(
        self,
        *,
        task_id: int,
        session_id: int,
        owner_qq: int,
        request_text: str,
    ) -> bool:
        logger.info("admin_agent_turn_stage task_id=%s stage=claimed", task_id)
        prompt = self.admin_agent_runtime.build_prompt(
            session_summary=self._session_summary(session_id=session_id),
            recent_turns=self._recent_turn_lines(session_id=session_id, exclude_task_id=task_id),
            request_text=request_text,
        )
        return await self._run_codex_task_with_prompt(
            task_id=task_id,
            session_id=session_id,
            owner_qq=owner_qq,
            request_text=request_text,
            prompt=prompt,
        )

    async def _run_codex_task_with_prompt(
        self,
        *,
        task_id: int,
        session_id: int,
        owner_qq: int,
        request_text: str,
        prompt: str,
    ) -> bool:
        artifact_dir = self.task_dir / f"task-{task_id}"
        checkpoint_dir = self.checkpoint_root / f"task-{task_id}"
        checkpoint_manifest: dict[str, list[str]] | None = None
        resume_thread_id = self._get_codex_thread_id(session_id=session_id)

        try:
            checkpoint_manifest = await asyncio.to_thread(
                create_repo_checkpoint,
                repo_root=self.repo_root,
                checkpoint_dir=checkpoint_dir,
            )
            self._write_checkpoint_manifest(checkpoint_dir=checkpoint_dir, manifest=checkpoint_manifest)
            try:
                with session_scope(self.engine) as session:
                    DevTaskArtifactRepository(session).add_artifact(
                        task_id=task_id,
                        artifact_type="checkpoint_manifest",
                        artifact_path=str(checkpoint_dir),
                        metadata_json=checkpoint_manifest,
                    )
            except Exception:
                logger.exception("dev_task_artifact_record_failed task_id=%s", task_id)

            result = await asyncio.to_thread(
                self._get_codex_bridge().run_task,
                prompt=prompt,
                repo_root=self.repo_root,
                artifact_dir=artifact_dir,
                resume_thread_id=resume_thread_id,
            )
            logger.info("dev_task_stage task_id=%s stage=codex_finished", task_id)
            normalized_reply_text = self._normalize_private_reply(result.reply_text)
            if result.thread_id:
                self._set_codex_thread_id(session_id=session_id, thread_id=result.thread_id)

            files_changed = await asyncio.to_thread(
                self._detect_changed_files,
                checkpoint_dir=checkpoint_dir,
                manifest=checkpoint_manifest,
            )
            commands_run = ["codex exec resume" if resume_thread_id else "codex exec"]
            restart_required = self._should_restart_after_task(
                request_text=request_text,
                files_changed=files_changed,
                model_restart_required=result.restart_required,
            )
            self._write_saved_task_result(
                task_id=task_id,
                result=CodexTaskResult(
                    summary=result.summary,
                    reply_text=normalized_reply_text,
                    restart_required=restart_required,
                    raw_last_message=result.raw_last_message,
                    thread_id=result.thread_id,
                ),
            )
            restart_result = "not-needed"
            if restart_required:
                logger.info("dev_task_stage task_id=%s stage=restarting_runtime", task_id)
                await self._send_private_text(
                    user_id=owner_qq,
                    text="我现在重启小町，让改动生效。",
                    context=f"dev_task:{task_id}:restarting_runtime",
                    timeout_seconds=5.0,
                )
                if not self.enable_local_worker:
                    restart_ok, restart_result_text = self._handoff_inline_runtime_restart()
                    if not restart_ok:
                        await asyncio.to_thread(
                            restore_repo_checkpoint,
                            repo_root=self.repo_root,
                            checkpoint_dir=checkpoint_dir,
                            manifest=checkpoint_manifest,
                        )
                        with session_scope(self.engine) as session:
                            tasks = DevTaskRepository(session)
                            tasks.mark_failed(
                                task_id=task_id,
                                failure_reason=restart_result_text,
                                checkpoint_dir=str(checkpoint_dir),
                            )
                            self._append_session_summary(
                                session_id=session_id,
                                owner_text=request_text,
                                assistant_text=f"失败：{restart_result_text}",
                                sessions=DevSessionRepository(session),
                            )
                        await self._send_private_text(
                            user_id=owner_qq,
                            text=f"这次改完以后，重启接管没成功：{restart_result_text}",
                            context=f"dev_task:{task_id}:restart_failed",
                            timeout_seconds=8.0,
                        )
                    else:
                        logger.info("dev_task_stage task_id=%s stage=restart_handed_off", task_id)
                    return True
                restart_ok, restart_result_text = self._restart_runtime()
                commands_run.extend(["xiaomachi-wsl-entry.sh stop", "xiaomachi-wsl-entry.sh start"])
                if not restart_ok:
                    await asyncio.to_thread(
                        restore_repo_checkpoint,
                        repo_root=self.repo_root,
                        checkpoint_dir=checkpoint_dir,
                        manifest=checkpoint_manifest,
                    )
                    rollback_ok, rollback_result_text = self._restart_runtime()
                    commands_run.extend(["xiaomachi-wsl-entry.sh stop", "xiaomachi-wsl-entry.sh start"])
                    with session_scope(self.engine) as session:
                        tasks = DevTaskRepository(session)
                        if rollback_ok:
                            tasks.mark_status(task_id=task_id, status="rolled_back")
                            rolled_back = tasks.get_task(task_id)
                            if rolled_back is not None:
                                rolled_back.failure_reason = restart_result_text
                                rolled_back.restart_result = "rolled_back"
                        else:
                            tasks.mark_failed(
                                task_id=task_id,
                                failure_reason=rollback_result_text,
                                checkpoint_dir=str(checkpoint_dir),
                            )
                    await self._send_private_text(
                        user_id=owner_qq,
                        text=(
                            "这次改完以后启动失败了，我已经先回滚并拉起来了。"
                            if rollback_ok
                            else f"这次改完以后启动失败了，而且回滚后也没恢复：{rollback_result_text}"
                        ),
                        context=f"dev_task:{task_id}:restart_failed",
                        timeout_seconds=8.0,
                    )
                    return True
                restart_result = restart_result_text

            normalized_reply_text = self._rewrite_reply_after_successful_restart(
                normalized_reply_text,
                restart_result=restart_result,
            )

            with session_scope(self.engine) as session:
                tasks = DevTaskRepository(session)
                tasks.mark_completed(
                    task_id=task_id,
                    summary=result.summary or self._build_turn_summary(request_text, normalized_reply_text),
                    result_text=normalized_reply_text,
                    files_read=[],
                    files_changed=files_changed,
                    commands_run=commands_run,
                    restart_required=restart_required,
                    restart_result=restart_result,
                    checkpoint_dir=str(checkpoint_dir),
                )
                self._append_session_summary(
                    session_id=session_id,
                    owner_text=request_text,
                    assistant_text=normalized_reply_text,
                    sessions=DevSessionRepository(session),
                )

            logger.info("dev_task_stage task_id=%s stage=completed", task_id)
            await self._send_private_text(
                user_id=owner_qq,
                text=normalized_reply_text,
                context=f"dev_task:{task_id}:completed",
                timeout_seconds=8.0,
            )
            return True
        except Exception as exc:
            logger.exception("dev_task_failed task_id=%s", task_id)
            recovered_reply = self._recover_task_from_saved_result(
                task_id=task_id,
                session_id=session_id,
                owner_qq=owner_qq,
                request_text=request_text,
                checkpoint_manifest=checkpoint_manifest,
                notification_prefix="这条任务其实已经跑完了，我直接把结果给你：",
            )
            if recovered_reply is not None:
                if self._should_send_private_recovery_reply(task_id=task_id, recovery_text=recovered_reply):
                    await self._send_private_text(
                        user_id=owner_qq,
                        text=recovered_reply,
                        context=f"dev_task:{task_id}:recovered_after_exception",
                        timeout_seconds=8.0,
                    )
                return True
            failure_reason = str(exc)
            if checkpoint_manifest is not None:
                try:
                    await asyncio.to_thread(
                        restore_repo_checkpoint,
                        repo_root=self.repo_root,
                        checkpoint_dir=checkpoint_dir,
                        manifest=checkpoint_manifest,
                    )
                except Exception as restore_exc:
                    logger.exception("dev_task_restore_failed task_id=%s", task_id)
                    failure_reason = f"{failure_reason}; rollback failed: {restore_exc}"
            with session_scope(self.engine) as session:
                DevTaskRepository(session).mark_failed(
                    task_id=task_id,
                    failure_reason=failure_reason,
                    checkpoint_dir=str(checkpoint_dir),
                )
                self._append_session_summary(
                    session_id=session_id,
                    owner_text=request_text,
                    assistant_text=f"失败：{failure_reason}",
                    sessions=DevSessionRepository(session),
                )
            await self._send_private_text(
                user_id=owner_qq,
                text=f"这次没跑通，我先记下来：{failure_reason}",
                context=f"dev_task:{task_id}:failed",
                timeout_seconds=8.0,
            )
            return True

    async def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            processed = await self.process_next_task_once()
            if processed:
                continue
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval_seconds)
            except TimeoutError:
                continue

    async def _send_private_text(
        self,
        *,
        user_id: int,
        text: str,
        context: str,
        timeout_seconds: float = 8.0,
    ) -> bool:
        del timeout_seconds
        delivered, _failure_reason = await self._deliver_private_text(
            user_id=user_id,
            text=text,
            context=context,
        )
        return delivered

    async def _deliver_private_text(
        self,
        *,
        user_id: int,
        text: str,
        context: str,
    ) -> tuple[bool, str | None]:
        if not self._reserve_private_outbound_reply(user_id=user_id, reply_text=text, context=context):
            logger.info("private_reply_dedup_skip context=%s user_id=%s", context, user_id)
            return True, None
        try:
            await self.sender.send_private_text(OutboundPrivateMessage(user_id=user_id, text=text))
            self._mark_private_outbound_reply_sent(user_id=user_id, reply_text=text, context=context)
            return True, None
        except Exception as exc:
            self._clear_private_outbound_reply_reservation(context=context)
            logger.exception("private_reply_send_failed context=%s user_id=%s", context, user_id)
            failure_reason = str(exc).strip() or exc.__class__.__name__
            return False, failure_reason

    async def _fetch_private_quoted_message_payload(self, *, reply_to_msg_id: str | None) -> dict | None:
        if not reply_to_msg_id:
            return None
        gateway = getattr(self.sender, "gateway", None)
        if gateway is None or not hasattr(gateway, "call_api"):
            return None
        message_id: int | str = int(reply_to_msg_id) if str(reply_to_msg_id).isdigit() else str(reply_to_msg_id)
        try:
            response = await gateway.call_api("get_msg", {"message_id": message_id})
        except Exception:
            logger.exception("private_quoted_message_fetch_failed reply_to_msg_id=%s", reply_to_msg_id)
            return None
        if not isinstance(response, dict):
            return None
        payload = response.get("data")
        if not isinstance(payload, dict):
            return None
        return payload

    def _private_image_turn_key(self, *, user_id: int, private_scope: str) -> tuple[int, str]:
        return (user_id, private_scope)

    def _should_defer_private_image_turn(self, event: PrivateMessageEvent) -> bool:
        if self.private_image_followup_window_seconds <= 0:
            return False
        if str(event.plain_text or "").strip():
            return False
        if getattr(event, "reply_to_msg_id", None) is not None:
            return False
        return bool(list(getattr(event, "images", []) or []))

    def _should_hold_private_image_for_followup(self, event: PrivateMessageEvent) -> bool:
        return (
            getattr(event, "reply_to_msg_id", None) is None
            and not str(event.plain_text or "").strip()
            and len(list(getattr(event, "images", []) or [])) == 1
        )

    def _cancel_pending_private_image_turns_for_user(
        self,
        user_id: int | None,
        *,
        preserve_for_next_turn: bool,
    ) -> None:
        for key, pending in list(self._pending_private_image_turns.items()):
            if user_id is not None and key[0] != user_id:
                continue
            task, pending_event = pending
            if preserve_for_next_turn:
                pending_images = list(getattr(pending_event, "images", []) or [])
                if pending_images:
                    self._private_image_turn_overrides[key[0]] = pending_images
            task.cancel()
            self._pending_private_image_turns.pop(key, None)

    def _consume_private_image_turn_override(self, *, user_id: int) -> list | None:
        override_images = self._private_image_turn_overrides.pop(user_id, None)
        if not override_images:
            return None
        return list(override_images)

    def _reset_private_draw_state(self, *, user_id: int) -> None:
        self._cancel_pending_private_image_turns_for_user(user_id, preserve_for_next_turn=False)
        self._private_image_turn_overrides.pop(user_id, None)
        self._private_draw_context_reset_users.add(user_id)

    async def _schedule_private_image_followup_if_needed(
        self,
        event: PrivateMessageEvent,
        *,
        private_scope: str,
    ) -> bool:
        if not self._should_defer_private_image_turn(event):
            return False

        key = self._private_image_turn_key(user_id=event.user_id, private_scope=private_scope)

        async def _run_deferred_turn() -> None:
            try:
                await asyncio.sleep(self.private_image_followup_window_seconds)
                await self._handle_private_chat_turn(event, private_scope=private_scope)
            except asyncio.CancelledError:
                return
            finally:
                self._pending_private_image_turns.pop(key, None)

        self._pending_private_image_turns[key] = (asyncio.create_task(_run_deferred_turn()), event)
        return True

    def _private_turn_text_for_prompt(self, event: PrivateMessageEvent, *, resolved_image_count: int = 0) -> str:
        plain_text = str(event.plain_text or "").strip()
        if plain_text:
            return plain_text

        current_images = list(getattr(event, "images", []) or [])
        image_count = len(current_images) or resolved_image_count
        if image_count <= 0:
            return plain_text
        if image_count == 1:
            return "[sent 1 image]" if current_images else "[asked about 1 image]"
        return f"[sent {image_count} images]" if current_images else f"[asked about {image_count} images]"

    def _normalize_private_lookup_text(self, value: str) -> str:
        return LOOKUP_NORMALIZER.sub("", value).lower()

    def _extract_auto_web_reference_query(self, *, stripped_text: str) -> str | None:
        match = AUTO_WEB_REFERENCE_QUERY_PATTERN.search(stripped_text)
        if match is not None:
            query = str(match.group("query") or "").strip(" \t,\uFF0C\u3002.!?\uFF1F\uFF1B;:\uFF1A")
            if query:
                return query
        marker_positions = [
            stripped_text.find(marker)
            for marker in ("\u7684\u4eba\u8bbe\u56fe", "\u4eba\u8bbe\u56fe", "\u8bbe\u5b9a\u56fe", "\u53c2\u8003\u56fe")
            if stripped_text.find(marker) >= 0
        ]
        if not marker_positions:
            return None
        end = min(marker_positions)
        action_positions = [
            idx
            for idx in (stripped_text.find("\u627e"), stripped_text.find("\u641c"))
            if 0 <= idx < end
        ]
        if not action_positions:
            return None
        start = min(action_positions) + 1
        query = stripped_text[start:end].strip(" \t,\uFF0C\u3002.!?\uFF1F\uFF1B;:\uFF1A")
        while query.startswith(("\u7f51\u4e0a", "\u4e0a\u7f51", "\u8054\u7f51")):
            query = query[2:].strip(" \t,\uFF0C\u3002.!?\uFF1F\uFF1B;:\uFF1A")
        return query or None

    def _build_auto_web_reference_prompt(self, *, stripped_text: str, query: str) -> str:
        prompt = AUTO_WEB_REFERENCE_QUERY_PATTERN.sub("", stripped_text, count=1)
        prompt = AUTO_WEB_REFERENCE_LEADING_CONNECTOR_PATTERN.sub("", prompt).strip(" \t,\uFF0C\u3002.!?\uFF1F\uFF1B;:\uFF1A")
        if not prompt:
            return f"\u53c2\u8003\u641c\u7d22\u5230\u7684{query}\u4eba\u8bbe\u56fe\u751f\u6210\u4e00\u5f20\u56fe"
        return f"\u53c2\u8003\u641c\u7d22\u5230\u7684{query}\u4eba\u8bbe\u56fe\uff0c{prompt}"

    def _looks_like_private_reference_image_generation_request(
        self,
        *,
        stripped_text: str,
        target_images: list | None,
    ) -> bool:
        if not target_images:
            return False
        normalized_text = self._normalize_private_lookup_text(stripped_text)
        if not normalized_text:
            return False
        has_transform_intent = any(
            self._normalize_private_lookup_text(keyword) in normalized_text
            for keyword in GROUP_IMAGE_REFERENCE_INTENT_KEYWORDS
        )
        has_reference_context = any(
            self._normalize_private_lookup_text(keyword) in normalized_text
            for keyword in GROUP_IMAGE_REFERENCE_CONTEXT_KEYWORDS
        )
        has_generation_intent = any(
            self._normalize_private_lookup_text(keyword) in normalized_text
            for keyword in GROUP_IMAGE_REFERENCE_GENERATION_KEYWORDS
        )
        has_retouch_intent = any(
            self._normalize_private_lookup_text(keyword) in normalized_text
            for keyword in PRIVATE_IMAGE_RETOUCH_INTENT_KEYWORDS
        )
        if has_transform_intent or (has_reference_context and has_generation_intent):
            return True
        if has_retouch_intent:
            return True
        direct_text = str(stripped_text or "")
        return (
            ("\u6784\u56fe" in direct_text and ("\u66ff\u6362\u4eba\u7269" in direct_text or "\u6362\u6210\u4eba\u7269" in direct_text) and "\u51fa\u56fe" in direct_text)
            or ("\u53c2\u8003" in direct_text and "\u51fa\u56fe" in direct_text)
        )

    def _build_private_image_request(
        self,
        *,
        event: PrivateMessageEvent,
        target_images: list | None,
    ) -> PrivateImageGenerationRequest | None:
        stripped = str(event.plain_text or "").strip()
        if not stripped:
            return None
        if any(pattern.search(stripped) for pattern in GROUP_IMAGE_NEGATIVE_PATTERNS):
            return None
        reference_images = list(target_images or [])
        auto_web_reference_query = self._extract_auto_web_reference_query(stripped_text=stripped)
        if auto_web_reference_query is not None:
            return PrivateImageGenerationRequest(
                user_id=event.user_id,
                trigger_message_id=event.platform_msg_id,
                prompt=self._build_auto_web_reference_prompt(
                    stripped_text=stripped,
                    query=auto_web_reference_query,
                ),
                reference_images=reference_images,
                web_search_query=auto_web_reference_query,
            )
        if self._looks_like_private_reference_image_generation_request(
            stripped_text=stripped,
            target_images=reference_images,
        ):
            return PrivateImageGenerationRequest(
                user_id=event.user_id,
                trigger_message_id=event.platform_msg_id,
                prompt=stripped,
                reference_images=reference_images,
            )
        for pattern in GROUP_IMAGE_REQUEST_PATTERNS:
            match = pattern.match(stripped)
            if match is None:
                continue
            prompt = match.group("prompt").strip(" \t,\uFF0C\u3002.!?\uFF1F\uFF1B;:\uFF1A")
            if not prompt:
                return None
            return PrivateImageGenerationRequest(
                user_id=event.user_id,
                trigger_message_id=event.platform_msg_id,
                prompt=prompt,
                reference_images=reference_images,
            )
        if reference_images and any(
            keyword in stripped
            for keyword in ("\u51fa\u56fe", "\u753b\u56fe", "\u7ed8\u56fe", "\u751f\u6210")
        ):
            return PrivateImageGenerationRequest(
                user_id=event.user_id,
                trigger_message_id=event.platform_msg_id,
                prompt=stripped,
                reference_images=reference_images,
            )
        return None

    async def _handle_admin_agent_turn(self, event: PrivateMessageEvent) -> None:
        override_images = self._consume_private_image_turn_override(user_id=event.user_id)
        request_text = self._private_turn_text_for_prompt(
            event,
            resolved_image_count=len(override_images or []),
        )
        conflicting_task = self._find_conflicting_admin_repo_task(requested_by_qq=event.user_id)
        if conflicting_task is not None:
            await self._send_admin_repo_busy_reply(user_id=event.user_id, platform_msg_id=event.platform_msg_id)
            return
        _session_id, task_id = self.admin_agent_runtime.enqueue_turn(
            event=event,
            request_text=request_text,
        )
        await self._send_private_text(
            user_id=event.user_id,
            text=ADMIN_AGENT_ACK_TEXT,
            context=f"admin_agent:{task_id}:queued",
        )

    def _find_conflicting_admin_repo_task(self, *, requested_by_qq: int):
        with session_scope(self.engine) as session:
            tasks = DevTaskRepository(session).list_tasks_by_statuses(
                statuses=["running", "queued"],
                intent_types=list(GLOBAL_ADMIN_MUTEX_INTENTS),
            )
        for task in tasks:
            if task.requested_by_qq == requested_by_qq:
                continue
            return task
        return None

    async def _send_admin_repo_busy_reply(self, *, user_id: int, platform_msg_id: str) -> None:
        await self._send_private_text(
            user_id=user_id,
            text="当前已有管理员任务正在执行，请稍后重试。",
            context=f"admin_busy:{platform_msg_id}",
        )

    def _finalize_private_image_task(self, task_id: int, result: ImageJobResult) -> None:
        with session_scope(self.engine) as session:
            tasks = DevTaskRepository(session)
            task = tasks.get_task(task_id)
            if task is None or task.status in {"completed", "failed", "rolled_back"}:
                return
            if result.success:
                tasks.mark_completed(
                    task_id=task_id,
                    summary=self._build_turn_summary(task.raw_request_text, result.notice_text),
                    result_text=result.notice_text,
                    files_read=[],
                    files_changed=[],
                    commands_run=["private_image_service.completed"],
                    restart_required=False,
                    restart_result="not-needed",
                    checkpoint_dir="",
                )
                if result.image_path is not None:
                    DevTaskArtifactRepository(session).add_artifact(
                        task_id=task_id,
                        artifact_type="private_image",
                        artifact_path=str(result.image_path),
                        metadata_json={"notice_text": result.notice_text},
                    )
                self._append_session_summary(
                    session_id=task.session_id,
                    owner_text=task.raw_request_text,
                    assistant_text=result.notice_text,
                    sessions=DevSessionRepository(session),
                )
                return
            tasks.mark_failed(task_id=task_id, failure_reason=result.failure_reason or result.notice_text)
            self._append_session_summary(
                session_id=task.session_id,
                owner_text=task.raw_request_text,
                assistant_text=f"失败：{result.notice_text}",
                sessions=DevSessionRepository(session),
            )

    async def _handle_private_chat_turn(self, event: PrivateMessageEvent, *, private_scope: str) -> None:
        session_mode = SESSION_MODE_PROJECT if private_scope == PRIVATE_SCOPE_OWNER_PROJECT else SESSION_MODE_DAILY
        initial_request_text = self._private_turn_text_for_prompt(event)
        with session_scope(self.engine) as session:
            sessions = DevSessionRepository(session)
            tasks = DevTaskRepository(session)
            dev_session = sessions.get_or_create_owner_session(
                owner_qq=event.user_id,
                session_mode=session_mode,
            )
            task = tasks.add_task(
                session_id=dev_session.id,
                requested_by_qq=event.user_id,
                raw_request_text=initial_request_text,
                intent_type="project_chat",
            )
            sessions.update_session(session_id=dev_session.id, last_task_id=task.id)
            session_id = dev_session.id
            task_id = task.id

        try:
            override_images = self._consume_private_image_turn_override(user_id=event.user_id)
            if override_images:
                target_images = override_images
            else:
                quoted_raw_payload = await self._fetch_private_quoted_message_payload(
                    reply_to_msg_id=getattr(event, "reply_to_msg_id", None)
                )
                with session_scope(self.engine) as session:
                    target_images_turn = resolve_private_images_for_turn(
                        event=event,
                        messages=MessageRepository(session),
                        quoted_raw_payload=quoted_raw_payload,
                    )
                if (
                    target_images_turn is not None
                    and target_images_turn.source_kind == "recent"
                    and event.user_id in self._private_draw_context_reset_users
                ):
                    target_images = None
                else:
                    target_images = target_images_turn.images if target_images_turn and target_images_turn.images else None
            request_text = self._private_turn_text_for_prompt(
                event,
                resolved_image_count=len(target_images or []),
            )
            progress_text = self._build_inline_progress_text(
                request_text=request_text,
                private_scope=private_scope,
                intent_type="project_chat",
            )
            if progress_text:
                await self._send_private_text(
                    user_id=event.user_id,
                    text=progress_text,
                    context=f"private_chat:{task_id}:progress",
                )
            conversation_key = f"dev-session:{session_id}"
            reply_text = None
            if private_scope == PRIVATE_SCOPE_OWNER_PROJECT:
                explicit_private_send = self._parse_explicit_private_send_request(request_text)
                if explicit_private_send is not None:
                    reply_text = await self._handle_owner_project_private_send_request(
                        task_id=task_id,
                        private_send=explicit_private_send,
                    )
            private_image_request = None
            if reply_text is None:
                private_image_request = self._build_private_image_request(
                    event=event,
                    target_images=target_images,
                )
                if private_image_request is not None:
                    private_image_request.dev_task_id = task_id
            if private_image_request is not None:
                enqueue_result = await self.private_image_service.enqueue(private_image_request)
                if enqueue_result.accepted:
                    reply_text = "图我接住了，开始画"
                    with session_scope(self.engine) as session:
                        DevTaskRepository(session).mark_status(task_id=task_id, status="running")
                else:
                    reply_text = "现在排队的图太多了，你等一下再发"
            if reply_text is None:
                reply_text = self._build_unsupported_private_action_reply(
                    request_text=request_text,
                    private_scope=private_scope,
                )
            if reply_text is None:
                prompt_lines = self._build_private_chat_prompt(
                    session_id=session_id,
                    task_id=task_id,
                    request_text=request_text,
                    request_time=event.timestamp,
                    private_scope=private_scope,
                    image_count=len(target_images or []),
                )
                reply_text = self._normalize_private_reply(
                    self.llm_client.generate_text(
                        prompt_lines,
                        images=target_images,
                        conversation_key=conversation_key,
                    )
                )
                if not reply_text:
                    raise ValueError("empty private project chat reply")

            if private_image_request is None:
                with session_scope(self.engine) as session:
                    tasks = DevTaskRepository(session)
                    tasks.mark_completed(
                        task_id=task_id,
                        summary=self._build_turn_summary(request_text, reply_text),
                        result_text=reply_text,
                        files_read=[],
                        files_changed=[],
                        commands_run=["llm_client.generate_text"],
                        restart_required=False,
                        restart_result="not-needed",
                        checkpoint_dir="",
                    )
                    self._append_session_summary(
                        session_id=session_id,
                        owner_text=request_text,
                        assistant_text=reply_text,
                        sessions=DevSessionRepository(session),
                    )

            await self._send_private_text(
                user_id=event.user_id,
                text=reply_text,
                context=(
                    f"private_chat:{task_id}:accepted"
                    if private_image_request is not None
                    else f"private_chat:{task_id}:completed"
                ),
            )
        except Exception as exc:
            logger.exception("private_chat_failed user_id=%s scope=%s", event.user_id, private_scope)
            failure_reason = str(exc)
            with session_scope(self.engine) as session:
                tasks = DevTaskRepository(session)
                tasks.mark_failed(task_id=task_id, failure_reason=failure_reason)
                self._append_session_summary(
                    session_id=session_id,
                    owner_text=initial_request_text,
                    assistant_text=f"失败：{failure_reason}",
                    sessions=DevSessionRepository(session),
                )
            await self._send_private_text(
                user_id=event.user_id,
                text="我这边刚刚卡了一下，你再问我一句。",
                context=f"private_chat:{task_id}:failed",
            )

    async def _handle_project_inspect_turn(self, event: PrivateMessageEvent) -> None:
        initial_request_text = self._private_turn_text_for_prompt(event)
        with session_scope(self.engine) as session:
            sessions = DevSessionRepository(session)
            tasks = DevTaskRepository(session)
            dev_session = sessions.get_or_create_owner_session(
                owner_qq=event.user_id,
                session_mode=SESSION_MODE_PROJECT,
            )
            task = tasks.add_task(
                session_id=dev_session.id,
                requested_by_qq=event.user_id,
                raw_request_text=initial_request_text,
                intent_type="project_inspect",
            )
            sessions.update_session(session_id=dev_session.id, last_task_id=task.id)
            session_id = dev_session.id
            task_id = task.id

        try:
            override_images = self._consume_private_image_turn_override(user_id=event.user_id)
            if override_images:
                target_images = override_images
            else:
                quoted_raw_payload = await self._fetch_private_quoted_message_payload(
                    reply_to_msg_id=getattr(event, "reply_to_msg_id", None)
                )
                with session_scope(self.engine) as session:
                    target_images_turn = resolve_private_images_for_turn(
                        event=event,
                        messages=MessageRepository(session),
                        quoted_raw_payload=quoted_raw_payload,
                    )
                if (
                    target_images_turn is not None
                    and target_images_turn.source_kind == "recent"
                    and event.user_id in self._private_draw_context_reset_users
                ):
                    target_images = None
                else:
                    target_images = target_images_turn.images if target_images_turn and target_images_turn.images else None
            request_text = self._private_turn_text_for_prompt(
                event,
                resolved_image_count=len(target_images or []),
            )
            progress_text = self._build_inline_progress_text(
                request_text=request_text,
                private_scope=PRIVATE_SCOPE_OWNER_PROJECT,
                intent_type="project_inspect",
            )
            if progress_text:
                await self._send_private_text(
                    user_id=event.user_id,
                    text=progress_text,
                    context=f"project_inspect:{task_id}:progress",
                )
            snapshot = self._collect_inspection_snapshot(request_text=request_text)
            prompt_lines = self._build_project_inspect_prompt(
                session_id=session_id,
                task_id=task_id,
                request_text=request_text,
                request_time=event.timestamp,
                snapshot=snapshot,
                image_count=len(target_images or []),
            )
            reply_text = self._normalize_private_reply(
                self.llm_client.generate_text(
                    prompt_lines,
                    images=target_images,
                    conversation_key=f"dev-session:{session_id}",
                )
            )
            if not reply_text:
                raise ValueError("empty private inspection reply")

            with session_scope(self.engine) as session:
                tasks = DevTaskRepository(session)
                tasks.mark_completed(
                    task_id=task_id,
                    summary=self._build_turn_summary(request_text, reply_text),
                    result_text=reply_text,
                    files_read=snapshot.files_read,
                    files_changed=[],
                    commands_run=snapshot.commands_run + ["llm_client.generate_text"],
                    restart_required=False,
                    restart_result="not-needed",
                    checkpoint_dir="",
                )
                self._append_session_summary(
                    session_id=session_id,
                    owner_text=request_text,
                    assistant_text=reply_text,
                    sessions=DevSessionRepository(session),
                )

            await self._send_private_text(
                user_id=event.user_id,
                text=reply_text,
                context=f"project_inspect:{task_id}:completed",
            )
        except Exception as exc:
            logger.exception("project_inspect_failed user_id=%s", event.user_id)
            failure_reason = str(exc)
            with session_scope(self.engine) as session:
                tasks = DevTaskRepository(session)
                tasks.mark_failed(task_id=task_id, failure_reason=failure_reason)
                self._append_session_summary(
                    session_id=session_id,
                    owner_text=initial_request_text,
                    assistant_text=f"失败：{failure_reason}",
                    sessions=DevSessionRepository(session),
                )
            await self._send_private_text(
                user_id=event.user_id,
                text="我刚刚查到一半卡住了，你再丢我一句。",
                context=f"project_inspect:{task_id}:failed",
            )

    async def _dispatch_feature_work_request(
        self,
        *,
        event: PrivateMessageEvent,
        request_text: str,
        routing_text: str | None = None,
        confirmed: bool = False,
        auto_execute_after_plan: bool = False,
    ) -> None:
        conflicting_task = self._find_conflicting_admin_repo_task(requested_by_qq=event.user_id)
        if conflicting_task is not None:
            await self._send_admin_repo_busy_reply(user_id=event.user_id, platform_msg_id=event.platform_msg_id)
            return
        active_task = self._find_reusable_active_feature_work_task(
            owner_qq=event.user_id,
            request_text=request_text,
        )
        if active_task is not None:
            await self._reply_with_active_feature_work_status(
                event=event,
                request_text=request_text,
                active_task=active_task,
            )
            return

        if not confirmed:
            await self._offer_feature_work_confirmation(
                event=event,
                request_text=request_text,
                routing_text=routing_text,
                require_confirmation=not auto_execute_after_plan,
            )
            if not auto_execute_after_plan:
                return

        if self._should_use_inline_feature_workflow(request_text=request_text, routing_text=routing_text):
            await self._start_inline_execute_task(event=event, request_text=request_text)
            return

        task_id = self._queue_execute_task(
            event=event,
            intent_type="feature_work",
            request_text=request_text,
        )
        logger.info("private_project_execute_waiting task_id=%s", task_id)

    def _find_reusable_active_feature_work_task(self, *, owner_qq: int, request_text: str):
        with session_scope(self.engine) as session:
            sessions = DevSessionRepository(session)
            tasks = DevTaskRepository(session)
            dev_session = sessions.get_latest_owner_session(
                owner_qq=owner_qq,
                session_mode=SESSION_MODE_PROJECT,
            )
            if dev_session is None:
                return None
            active_tasks = tasks.list_tasks_for_session_by_status(
                session_id=dev_session.id,
                statuses=["running", "queued"],
            )

        for task in reversed(active_tasks):
            if task.intent_type != "feature_work":
                continue
            if self._should_reuse_active_feature_work(
                request_text=request_text,
                active_request_text=task.raw_request_text,
            ):
                return task
        return None

    def _should_reuse_active_feature_work(self, *, request_text: str, active_request_text: str) -> bool:
        normalized_request = self._normalize_private_command_text(request_text)
        normalized_active = self._normalize_private_command_text(active_request_text)
        if not normalized_request or not normalized_active:
            return False
        if normalized_request == normalized_active:
            return True
        return self._looks_like_feature_work_continue_request(normalized_request)

    def _looks_like_feature_work_continue_request(self, normalized_text: str) -> bool:
        continue_hints = (
            "那你开始完成我说的功能吧",
            "开始完成我说的功能",
            "开始做吧",
            "你开始做吧",
            "你开始做",
            "那你开始做",
            "继续做吧",
            "继续做",
            "继续处理",
            "继续推进",
            "按这个做",
            "就按这个",
            "照这个做",
            "去做吧",
            "开做吧",
        )
        return any(hint in normalized_text for hint in continue_hints)

    async def _reply_with_active_feature_work_status(
        self,
        *,
        event: PrivateMessageEvent,
        request_text: str,
        active_task,
    ) -> None:
        reply_text = self._build_active_feature_work_status_reply(active_task=active_task)
        with session_scope(self.engine) as session:
            sessions = DevSessionRepository(session)
            tasks = DevTaskRepository(session)
            dev_session = sessions.get_or_create_owner_session(
                owner_qq=event.user_id,
                session_mode=SESSION_MODE_PROJECT,
            )
            task = tasks.add_task(
                session_id=dev_session.id,
                requested_by_qq=event.user_id,
                raw_request_text=request_text.strip() or event.plain_text.strip(),
                intent_type="project_chat",
            )
            sessions.update_session(session_id=dev_session.id, last_task_id=task.id)
            tasks.mark_completed(
                task_id=task.id,
                summary=self._build_turn_summary(request_text, reply_text),
                result_text=reply_text,
                files_read=[],
                files_changed=[],
                commands_run=["active_feature_work_status"],
                restart_required=False,
                restart_result="not-needed",
                checkpoint_dir="",
            )
            self._append_session_summary(
                session_id=dev_session.id,
                owner_text=request_text,
                assistant_text=reply_text,
                sessions=sessions,
            )
        await self._send_private_text(
            user_id=event.user_id,
            text=reply_text,
            context=f"active_feature_work:{active_task.id}:status",
        )

    def _build_active_feature_work_status_reply(self, *, active_task) -> str:
        request_brief = self._truncate_text(" ".join(active_task.raw_request_text.strip().split()), limit=80)
        if active_task.status == "running":
            return (
                "上一条开发任务还在处理，我会继续沿着那条推进，不再重复开一条。\n\n"
                f"当前在做的是：{request_brief}\n"
                "这条跑完后我再给你结果。"
            )
        return (
            "上一条开发任务已经在队列里了，我会继续沿着那条推进，不再重复开一条。\n\n"
            f"当前排队的是：{request_brief}\n"
            "轮到它开始处理后，我会继续按那条往下做。"
        )

    async def _offer_feature_work_confirmation(
        self,
        *,
        event: PrivateMessageEvent,
        request_text: str,
        routing_text: str | None = None,
        require_confirmation: bool = True,
    ) -> None:
        session_id, task_id, normalized_request_text = self._create_execute_task(
            event=event,
            intent_type=FEATURE_PLAN_INTENT,
            request_text=request_text,
            status="running",
        )
        confirmation_text, referenced_paths = self._build_feature_work_confirmation_reply(
            request_text=normalized_request_text,
            routing_text=routing_text,
            require_confirmation=require_confirmation,
        )
        with session_scope(self.engine) as session:
            tasks = DevTaskRepository(session)
            tasks.mark_completed(
                task_id=task_id,
                summary=self._build_turn_summary(normalized_request_text, confirmation_text),
                result_text=confirmation_text,
                files_read=referenced_paths,
                files_changed=[],
                commands_run=["feature_work_confirmation"],
                restart_required=False,
                restart_result="not-needed",
                checkpoint_dir="",
            )
            self._append_session_summary(
                session_id=session_id,
                owner_text=normalized_request_text,
                assistant_text=confirmation_text,
                sessions=DevSessionRepository(session),
            )
        await self._send_private_text(
            user_id=event.user_id,
            text=confirmation_text,
            context=f"feature_plan:{task_id}:completed",
        )

    def _build_feature_work_confirmation_reply(
        self,
        *,
        request_text: str,
        routing_text: str | None = None,
        require_confirmation: bool = True,
    ) -> tuple[str, list[str]]:
        normalized_request = " ".join(request_text.strip().split())
        repo_snippets = build_repo_context_snippets(
            repo_root=self.repo_root,
            query=routing_text or request_text,
            max_files=2,
            max_lines_per_file=2,
        )
        referenced_paths: list[str] = []
        for snippet in repo_snippets:
            path_line = snippet.splitlines()[0].strip()
            if path_line and path_line not in referenced_paths:
                referenced_paths.append(path_line)
        if referenced_paths:
            focus_text = "、".join(referenced_paths[:2])
        else:
            focus_text = "相关代码、配置和日志"

        if self._looks_like_capability_probe_request(request_text):
            action_text = f"我会先直接实测这条能力，重点看 {focus_text} 和真实返回。"
        elif (
            self._looks_like_restart_status_question(request_text)
            or self._looks_like_change_status_question(request_text)
            or self._looks_like_local_project_question(request_text)
        ):
            action_text = (
                f"我会先直接核对 {focus_text} 和当前运行情况；"
                "如果只是没接上、没生效或者差一小段逻辑，我会顺手改掉再验证。"
            )
        else:
            action_text = f"我会先从 {focus_text} 入手，必要时直接改代码并跑定向验证。"

        if require_confirmation:
            return (
                (
                    f"我理解你的目标是：{self._truncate_text(normalized_request, limit=120)}。"
                    f"{action_text}"
                    "如果动到运行链路，我会在收尾时一起重启确认是否生效。"
                    "如果你要，我下一步就直接进项目里执行；你回我“好 / 就这样 / 按这个”就行。"
                ),
                referenced_paths,
            )

        execution_text = (
            "我现在就直接开始。"
            if self._should_use_inline_feature_workflow(
                request_text=request_text,
                routing_text=routing_text,
            )
            else "我先按持续执行的方式推进，先把它排进项目处理队列。"
        )
        return (
            (
                f"我理解的是：{self._truncate_text(normalized_request, limit=120)}。"
                f"我会这样处理：{action_text}"
                "如果动到运行链路，我会在收尾时一起重启确认是否生效。"
                f"{execution_text}"
            ),
            referenced_paths,
        )

    def _create_execute_task(
        self,
        *,
        event: PrivateMessageEvent,
        intent_type: str,
        request_text: str | None = None,
        status: str = "queued",
    ) -> tuple[int, int, str]:
        normalized_request_text = (request_text or event.plain_text).strip()
        with session_scope(self.engine) as session:
            sessions = DevSessionRepository(session)
            tasks = DevTaskRepository(session)
            dev_session = sessions.get_or_create_owner_session(
                owner_qq=event.user_id,
                session_mode=SESSION_MODE_PROJECT,
            )
            task = tasks.add_task(
                session_id=dev_session.id,
                requested_by_qq=event.user_id,
                raw_request_text=normalized_request_text,
                intent_type=intent_type,
                status=status,
            )
            sessions.update_session(session_id=dev_session.id, last_task_id=task.id)
            return dev_session.id, task.id, normalized_request_text

    async def _start_inline_execute_task(self, *, event: PrivateMessageEvent, request_text: str) -> None:
        session_id, task_id, normalized_request_text = self._create_execute_task(
            event=event,
            intent_type="feature_work",
            request_text=request_text,
            status="running",
        )
        logger.info(
            "private_project_execute_inline session_id=%s task_id=%s",
            session_id,
            task_id,
        )
        await self._run_execute_task(
            task_id=task_id,
            session_id=session_id,
            owner_qq=event.user_id,
            request_text=normalized_request_text,
            intent_type="feature_work",
        )

    def _queue_execute_task(
        self,
        *,
        event: PrivateMessageEvent,
        intent_type: str,
        request_text: str | None = None,
    ) -> int:
        session_id, task_id, _ = self._create_execute_task(
            event=event,
            intent_type=intent_type,
            request_text=request_text,
            status="queued",
        )
        logger.info(
            "private_project_execute_queued session_id=%s task_id=%s intent=%s",
            session_id,
            task_id,
            intent_type,
        )
        return task_id

    def _should_use_inline_feature_workflow(self, *, request_text: str, routing_text: str | None = None) -> bool:
        normalized_request = self._normalize_private_command_text(request_text)
        normalized_route = self._normalize_private_command_text(routing_text or request_text)
        if not normalized_route:
            return True

        worker_hints = (
            "整个",
            "全部",
            "持续推进",
            "直到完成",
            "上线",
            "部署",
            "重构",
            "架构",
            "工作流",
            "多步",
            "复杂",
            "彻底",
            "系统性",
            "长期",
            "批量",
        )
        if any(hint in normalized_route for hint in worker_hints):
            return False

        if normalized_request.startswith(self._normalize_private_command_text("继续上一条")):
            return True

        inline_hints = (
            "改",
            "修",
            "加",
            "删",
            "调整",
            "改成",
            "就这样",
            "按这个",
            "照这个",
            "用你写的",
            "继续上一条",
        )
        if len(normalized_request) <= 24 and any(hint in normalized_request for hint in inline_hints):
            return True

        llm_decision = self._classify_feature_workflow_with_llm(request_text=request_text)
        if llm_decision is not None:
            return llm_decision == "inline"
        return len(normalized_request) <= 48

    def _classify_feature_workflow_with_llm(self, *, request_text: str) -> str | None:
        prompt_lines = [
            "You are routing the owner's private repository-change request.",
            "Both routes already have full repository access and may inspect or modify local code.",
            "Route labels:",
            "- INLINE: small, focused, likely one short pass, and suitable to execute immediately in the live private chat flow.",
            "- WORKER: broader, riskier, multi-step, or likely to take longer and should go through the queued worker workflow.",
            "Bias toward INLINE for narrow edits and simple follow-up confirmations.",
            f"Owner request: {request_text.strip()}",
            "Reply with exactly one label: INLINE or WORKER.",
        ]
        try:
            raw_label = str(self.llm_client.generate_text(prompt_lines)).strip().upper()
        except Exception:
            logger.exception("feature_workflow_llm_classification_failed owner_qq=%s", self.owner_qq)
            return None
        if "INLINE" in raw_label:
            return "inline"
        if "WORKER" in raw_label:
            return "worker"
        return None

    def _build_private_chat_prompt(
        self,
        *,
        session_id: int,
        task_id: int,
        request_text: str,
        request_time,
        private_scope: str,
        image_count: int = 0,
    ) -> list[str]:
        session_summary = self._session_summary(session_id=session_id)
        recent_turns = self._recent_turn_lines(session_id=session_id, exclude_task_id=task_id)
        history_block = "\n".join(recent_turns) if recent_turns else "(none)"
        web_context = self._build_private_web_context(
            request_text=request_text,
            request_time=request_time,
            recent_turns=recent_turns,
        )
        prompt_lines: list[str]
        if private_scope == PRIVATE_SCOPE_OWNER_PROJECT:
            repo_snippets = build_repo_context_snippets(repo_root=self.repo_root, query=request_text)
            snippet_block = "\n\n".join(repo_snippets) if repo_snippets else "(none)"
            prompt_lines = [
                "System persona: You are the same assistant style as Codex in this workspace, replying in one persistent private owner conversation about the local Xiaomachi repository.",
                "Safety rules: Stay grounded in the provided session history, repository snippets, runtime facts, and web evidence. Do not invent current file state or reveal secret values.",
                *self._private_reply_style_lines(private_scope=private_scope),
                "Current project session summary:",
                session_summary or "(none)",
                "Recent private project turns:",
                history_block,
                *self._private_web_context_lines(web_context),
                *self._private_image_reasoning_lines(
                    request_text=request_text,
                    recent_turns=recent_turns,
                    image_count=image_count,
                ),
                "Relevant repository snippets:",
                snippet_block,
                f"Current owner message: {request_text}",
            ]
        elif private_scope == PRIVATE_SCOPE_ALLOWLIST_DAILY:
            prompt_lines = [
                f"System persona: {self._private_daily_persona_text()}",
                self._private_daily_safety_line(
                    extra_rule="If the owner wants repository changes or local runtime inspection, tell them to send “启动管理员模式” first."
                ),
                self._private_daily_safety_line(
                    extra_rule="This user is not the owner: do not offer or imply code changes, admin actions, runtime restarts, or project control."
                ),
                *self._private_reply_style_lines(private_scope=private_scope),
                "Current private daily session summary:",
                session_summary or "(none)",
                "Recent private daily turns:",
                history_block,
                *self._private_web_context_lines(web_context),
                *self._private_image_reasoning_lines(
                    request_text=request_text,
                    recent_turns=recent_turns,
                    image_count=image_count,
                ),
                f"Current user message: {request_text}",
            ]
        else:
            prompt_lines = [
                f"System persona: {self._private_daily_persona_text()}",
                "Safety rules: Stay grounded in the provided session history, runtime facts, and web evidence. Do not invent repo or runtime state. If the owner wants repository changes or local runtime inspection, tell them to send “启动管理员模式” first.",
                *self._private_reply_style_lines(private_scope=private_scope),
                "Current private daily session summary:",
                session_summary or "(none)",
                "Recent private daily turns:",
                history_block,
                *self._private_web_context_lines(web_context),
                *self._private_image_reasoning_lines(
                    request_text=request_text,
                    recent_turns=recent_turns,
                    image_count=image_count,
                ),
                f"Current owner message: {request_text}",
            ]
        return prompt_lines

    def _build_project_inspect_prompt(
        self,
        *,
        session_id: int,
        task_id: int,
        request_text: str,
        request_time,
        snapshot: InspectionSnapshot,
        image_count: int = 0,
    ) -> list[str]:
        session_summary = self._session_summary(session_id=session_id)
        recent_turns = self._recent_turn_lines(session_id=session_id, exclude_task_id=task_id)
        repo_snippets = build_repo_context_snippets(repo_root=self.repo_root, query=request_text)
        history_block = "\n".join(recent_turns) if recent_turns else "(none)"
        snippet_block = "\n\n".join(repo_snippets) if repo_snippets else "(none)"
        web_context = self._build_private_web_context(
            request_text=request_text,
            request_time=request_time,
            recent_turns=recent_turns,
        )
        prompt_lines = [
            f"System persona: You are the same assistant style as Codex in this workspace, replying in one persistent private owner conversation about the local Xiaomachi repository.",
            "Safety rules: Use only the provided session history, local inspection facts, repository snippets, runtime facts, and web evidence. If evidence is missing, say so plainly. Do not invent current file state or reveal secret values.",
            *self._private_reply_style_lines(private_scope=PRIVATE_SCOPE_OWNER_PROJECT),
            "Current project session summary:",
            session_summary or "(none)",
            "Recent private project turns:",
            history_block,
            "Local inspection facts:",
            snapshot.prompt_block or "(none)",
            *self._private_web_context_lines(web_context),
            *self._private_image_reasoning_lines(
                request_text=request_text,
                recent_turns=recent_turns,
                image_count=image_count,
            ),
            "Relevant repository snippets:",
            snippet_block,
            f"Current owner message: {request_text}",
        ]
        return prompt_lines

    def _build_execute_prompt(self, *, session_id: int, task_id: int, request_text: str) -> str:
        session_summary = self._session_summary(session_id=session_id)
        history_lines = self._recent_turn_lines(session_id=session_id, exclude_task_id=task_id)
        history_block = "\n".join(history_lines) if history_lines else "(none)"
        repo_snippets = build_repo_context_snippets(repo_root=self.repo_root, query=request_text)
        snippet_block = "\n\n".join(repo_snippets) if repo_snippets else "(none)"
        return "\n".join(
            [
                "You are operating on the Xiaomachi repository for the owner's persistent private project session.",
                f"Repository root: {self.repo_root}",
                "Hard rules:",
                "- Stay inside the repository root.",
                "- Do not reveal secret values or .env contents.",
                "- Do not modify unrelated machine files or settings.",
                "- If code changes are unnecessary, leave files unchanged.",
                "- Do not restart the runtime yourself. Only inspect, edit, and run focused verification.",
                "- If a runtime restart seems needed, set restart_required accordingly. The outer controller may perform that restart after you return.",
                "- Do not tell the user that you lack permission to restart or cannot restart. Describe whether a restart is needed, and leave the actual restart to the outer controller.",
                "- Do not claim a machine-wide network or HTTPS block unless the current run shows multiple independent failures and no contradictory successful API call, probe result, or control request.",
                "- If current verification shows any successful API call or probe result, report the outcome as mixed, endpoint-specific, or transport-specific instead of escalating it into a whole-machine conclusion.",
                "Project session summary:",
                session_summary or "(none)",
                "Recent private project turns:",
                history_block,
                "Relevant repository snippets:",
                snippet_block,
                f"Current owner request: {request_text}",
                'Your final response must be strict JSON with keys "summary", "reply_text", and "restart_required".',
                'The "reply_text" value must sound like the same assistant style as Codex in this workspace: direct, concise, factual, natural Chinese. In private chat, Markdown is allowed when it helps clarity.',
            ]
        )

    def _private_daily_persona_text(self) -> str:
        if self.persona:
            return render_persona(self.persona)
        return (
            f"You are {self.assistant_name}. Identity: a private-chat AI assistant. "
            "Speaking tone: natural. Keep replies concise unless asked to expand."
        )

    def _private_daily_safety_line(self, *, extra_rule: str | None = None) -> str:
        rules = render_safety_lines(self.safety)
        rules.append("Stay grounded in the provided session history, runtime facts, and web evidence. Do not invent repo or runtime state.")
        if extra_rule:
            rules.append(extra_rule)
        return "Safety rules: " + " ".join(rule for rule in rules if rule)

    def _private_reply_style_lines(self, *, private_scope: str) -> list[str]:
        if private_scope == PRIVATE_SCOPE_OWNER_PROJECT:
            return [
                "Reply style: Sound like the same assistant style as Codex in this workspace.",
                "Reply style: Be direct, concise, factual, calm, and practical.",
                "Reply style: Do not default to Markdown, headings, bullet lists, or multi-paragraph formatting in private chat replies.",
                "Reply style: Prefer one compact message in one or two short paragraphs. Keep the information, but avoid splitting it into many blocks.",
                "Reply style: Only use light structure when the user explicitly asks for steps or when collapsing it would make the answer harder to follow.",
                "Reply style: For small acknowledgements or simple answers, do not force structure.",
                "Reply style: If evidence is missing or conflicting, say so plainly instead of smoothing it over.",
            ]
        return [
            f"Reply style: Stay in {str(self.persona.get('name', self.assistant_name) or self.assistant_name)}'s daily-chat persona.",
            "Reply style: Keep the tone natural, lively, and human, not stiff customer-service wording.",
            "Reply style: Do not default to Markdown, headings, bullet lists, or multi-paragraph formatting in private chat replies.",
            "Reply style: Prefer one compact message in one or two short paragraphs. Keep the information, but avoid splitting it into many blocks.",
            "Reply style: For casual back-and-forth, stay natural and do not over-structure tiny replies.",
            "Reply style: If evidence is missing or conflicting, say so plainly instead of smoothing it over.",
        ]

    def _looks_like_private_model_question(self, text: str) -> bool:
        normalized = self._normalize_private_command_text(text)
        if not normalized or "模型" not in text:
            return False
        return any(keyword in normalized for keyword in ("现在", "对话", "用", "哪个", "什么", "啥"))

    def _build_private_model_info_reply(self) -> str:
        primary_model = str(getattr(self.llm_client, "model", "") or "").strip()
        fallback_model = str(getattr(self.llm_client, "fallback_model", "") or "").strip()
        vision_model = str(getattr(self.llm_client, "vision_model", "") or "").strip()
        if not primary_model:
            try:
                settings = AppSettings()
            except Exception:
                settings = None
            if settings is not None:
                primary_model = str(settings.llm_model or "").strip()
                fallback_model = fallback_model or str(settings.llm_fallback_model or "").strip()
                vision_model = vision_model or str(settings.llm_vision_model or "").strip()
        primary_model = primary_model or "unknown"
        parts = [f"现在主模型是 {primary_model}"]
        if fallback_model:
            parts.append(f"副模型是 {fallback_model}")
        if vision_model:
            parts.append(f"看图时会单独走 {vision_model}")
        else:
            parts.append("看图时就直接走这套主副模型")
        return "，".join(parts) + "。"

    def _looks_like_generic_art_critique_reply(self, text: str) -> bool:
        normalized = str(text or "").strip().lower()
        if not normalized:
            return False
        critique_keywords = ("构图", "色彩", "光影", "层次感", "细节", "风格", "配色", "插画练习", "角色设定")
        identity_signals = ("没法确定", "不确定", "认不出来", "不知道", "出自", "来自", "原创", "同人", "动画", "游戏", "画师")
        return any(keyword in normalized for keyword in critique_keywords) and not any(
            signal in normalized for signal in identity_signals
        )

    def _build_private_identity_uncertain_reply(self) -> str:
        return "我现在没法可靠认出来，不想瞎猜。你要是愿意，我可以继续根据外观特征帮你缩小范围。"

    def _postprocess_private_character_reply(
        self,
        *,
        request_text: str,
        reply_text: str,
        has_images: bool,
    ) -> str:
        if not has_images or not self._is_private_character_identification_request(request_text):
            return reply_text
        if self._looks_like_generic_art_critique_reply(reply_text):
            return self._build_private_identity_uncertain_reply()
        return reply_text

    def _is_private_character_identification_request(self, text: str) -> bool:
        normalized_text = re.sub(r"\s+", "", str(text or "").strip().lower())
        if not normalized_text:
            return False
        hints = (
            "谁",
            "哪个角色",
            "什么角色",
            "哪位",
            "人物",
            "角色",
            "名字",
            "出自",
            "哪部",
            "哪作",
        )
        return any(hint in normalized_text for hint in hints)

    def _private_image_reasoning_lines(
        self,
        *,
        request_text: str,
        recent_turns: list[str],
        image_count: int,
    ) -> list[str]:
        del request_text
        del recent_turns
        if image_count <= 0:
            return []
        return [
            f"Vision task: {image_count} attached image(s) belong to the current turn. Inspect them directly before replying.",
            "Vision task: Base claims about identity, source, or scene details on visible evidence in the image, not on chat memory alone.",
        ]

    def _private_web_context_lines(self, web_context: PrivateWebContext) -> list[str]:
        lines: list[str] = []
        if web_context.runtime_facts:
            lines.append(
                "Treat runtime facts as authoritative for the current year, date, weekday, and clock time."
            )
            lines.append("Runtime facts:")
            lines.extend(web_context.runtime_facts)
        if web_context.grounding_notes:
            lines.append("Grounding notes:")
            lines.extend(web_context.grounding_notes)
        if web_context.web_results:
            lines.append("Web search results:")
            lines.extend(web_context.web_results)
        if web_context.web_pages:
            lines.append("Web page extracts:")
            lines.extend(web_context.web_pages)
        return lines

    def _private_search_names(self) -> set[str]:
        candidates = {
            self.assistant_name,
            self.assistant_name.lower(),
            "codex",
            "Codex",
            "小町",
            "komachi",
            "xiaomachi",
        }
        return {candidate for candidate in candidates if candidate}

    def _looks_like_private_send_request(self, text: str) -> bool:
        if any(keyword in text for keyword in ("权限", "白名单", "allowlist", "whitelist")):
            return False
        lowered = text.lower()
        has_send_hint = any(
            phrase in text or phrase in lowered
            for phrase in (
                "私聊发送",
                "发送",
                "发消息",
                "发一句",
                "发一条",
                "发个",
                "私信",
                "转达",
                "带句话",
            )
        )
        has_private_hint = any(phrase in text or phrase in lowered for phrase in ("私聊", "私信", "qq", "QQ"))
        has_target_hint = "给" in text or re.search(r"\d{5,16}", text) is not None
        return has_send_hint and has_private_hint and has_target_hint

    def _build_unsupported_private_action_reply(self, *, request_text: str, owner_mode: bool) -> str | None:
        if not owner_mode:
            return None
        if not self._looks_like_private_send_request(request_text):
            return None
        return (
            "我现在不能直接替你给别人发私聊消息。当前这条私聊开发通道只有看仓库、查配置、改项目和重启本体的能力，"
            "没有做“代替机器人给指定 QQ 发私聊内容”的执行接口，也没有发送成功回执链路，所以我不能假装已经发出去了。"
            "如果你要开发这个功能，直接回我“管理员权限 开发这个功能”就行。"
            "如果你只是想先确认目标 QQ 或私聊权限，我也可以先帮你查。"
        )

    def _build_inline_progress_text(self, *, request_text: str, owner_mode: bool, intent_type: str) -> str | None:
        del request_text, owner_mode, intent_type
        return None

    def _looks_like_capability_upgrade_confirmation(self, text: str) -> bool:
        normalized = "".join(text.strip().lower().split())
        if not normalized:
            return False
        direct_matches = {
            "开发这个功能",
            "开发下这个功能",
            "把这个功能开发出来",
            "开发这个能力",
            "做这个功能",
            "实现这个功能",
            "加上这个功能",
            "加这个功能",
            "把它做出来",
            "把它开发出来",
            "优化出这个功能",
            "继续开发这个功能",
        }
        if normalized in direct_matches:
            return True
        has_build_hint = any(keyword in text for keyword in ("开发", "实现", "做", "加上", "加个", "支持", "优化"))
        has_target_hint = any(keyword in text for keyword in ("功能", "能力", "这个", "它"))
        return has_build_hint and has_target_hint

    def _is_missing_private_send_capability_reply(self, reply_text: str) -> bool:
        normalized = self._normalize_private_reply(reply_text)
        return (
            "不能直接替你给别人发私聊消息" in normalized
            and "开发这个功能" in normalized
        )

    def _build_capability_upgrade_request_from_recent_context(self, *, owner_qq: int, raw_text: str) -> str | None:
        if not self._looks_like_capability_upgrade_confirmation(raw_text):
            return None

        with session_scope(self.engine) as session:
            sessions = DevSessionRepository(session)
            tasks = DevTaskRepository(session)
            dev_session = sessions.get_latest_owner_session(owner_qq=owner_qq)
            if dev_session is None:
                return None
            recent_tasks = tasks.list_recent_tasks_for_session(session_id=dev_session.id, limit=4)

        if not recent_tasks:
            return None

        previous_task = recent_tasks[-1]
        if previous_task.requested_by_qq != owner_qq:
            return None
        if previous_task.intent_type != "project_chat" or previous_task.status != "completed":
            return None
        if not self._looks_like_private_send_request(previous_task.raw_request_text):
            return None
        if not self._is_missing_private_send_capability_reply(previous_task.result_text):
            return None

        return self._build_capability_upgrade_request(
            original_request_text=previous_task.raw_request_text,
            confirmation_text=raw_text,
        )

    def _build_owner_admin_event(self, event: PrivateMessageEvent) -> PrivateMessageEvent | None:
        admin_text = self._extract_owner_admin_request_text(event.plain_text)
        if admin_text is None:
            return None
        return replace(event, plain_text=admin_text)

    def _extract_owner_admin_request_text(self, text: str) -> str | None:
        normalized = text.lstrip()
        if not normalized.startswith(OWNER_ADMIN_PREFIX):
            return None
        suffix = normalized[len(OWNER_ADMIN_PREFIX) :].lstrip(" \t:：,，;；、")
        return suffix.strip()

    def _build_capability_upgrade_request(self, *, original_request_text: str, confirmation_text: str) -> str:
        return "\n".join(
            [
                "实现一个新能力：让小町支持在权限允许时，按明确指令给指定 QQ 用户发送私聊消息。",
                f"原始需求：{original_request_text.strip()}",
                f"用户确认：{confirmation_text.strip()}",
                "要求：",
                "- 功能真正做完前，继续明确告诉用户当前还不能直接发送。",
                "- 发送链路要有权限校验、目标校验和失败可见的回执，不能伪装成已经发出。",
                "- 完成后要能从私聊开发通道稳定调用这个新能力。",
            ]
        )

    def _looks_like_owner_admin_continue_confirmation(self, text: str) -> bool:
        normalized = "".join(text.strip().lower().split())
        if not normalized:
            return False
        if normalized in OWNER_ADMIN_CONTINUE_CONFIRMATIONS:
            return True
        confirmation_prefixes = (
            "好",
            "好的",
            "行",
            "可以",
            "继续",
            "嗯",
            "就这样",
            "按这个",
            "照这个",
            "用你写的",
        )
        return len(normalized) <= 16 and any(normalized.startswith(prefix) for prefix in confirmation_prefixes)

    def _build_owner_admin_followup_from_recent_context(
        self,
        *,
        owner_qq: int,
        raw_text: str,
    ) -> tuple[str, str] | None:
        if not self._looks_like_owner_admin_continue_confirmation(raw_text):
            return None

        with session_scope(self.engine) as session:
            sessions = DevSessionRepository(session)
            tasks = DevTaskRepository(session)
            dev_session = sessions.get_latest_owner_session(owner_qq=owner_qq)
            if dev_session is None:
                return None
            recent_tasks = tasks.list_recent_tasks_for_session(session_id=dev_session.id, limit=4)

        if not recent_tasks:
            return None

        previous_task = recent_tasks[-1]
        if previous_task.requested_by_qq != owner_qq or previous_task.status != "completed":
            return None

        followup_intent = self._infer_owner_followup_intent_from_offer(previous_task.result_text)
        if followup_intent is None:
            return None

        return followup_intent, self._build_owner_followup_request(
            previous_request_text=previous_task.raw_request_text,
            previous_reply_text=previous_task.result_text,
            confirmation_text=raw_text,
            intent_type=followup_intent,
        )

    def _infer_owner_followup_intent_from_offer(self, reply_text: str) -> str | None:
        normalized = self._normalize_private_reply(reply_text)
        if not normalized:
            return None
        has_offer_shape = any(
            phrase in normalized
            for phrase in (
                "如果你要",
                "你要我继续",
                "你要的话",
                "要我继续的话",
                "下一步",
                "继续的话",
                "我可以继续",
            )
        )
        if not has_offer_shape:
            return None
        if any(
            phrase in normalized
            for phrase in (
                "下一步就直接重启",
                "下一步就重启",
                "直接执行重启",
                "先重启",
                "去重启",
                "我现在重启",
                "我下一步就重启",
            )
        ):
            return "restart_only"
        if any(
            phrase in normalized
            for phrase in (
                "去查",
                "核对",
                "确认",
                "去看",
                "进仓库",
                "查配置",
                "查代码",
                "查日志",
                "看代码",
                "看配置",
            )
        ):
            return "project_inspect"
        if any(
            phrase in normalized
            for phrase in (
                "去改",
                "帮你改",
                "实现",
                "开发",
                "加进去",
                "加上",
                "修掉",
                "修好",
            )
        ):
            return "feature_work"
        return None

    def _build_owner_followup_request(
        self,
        *,
        previous_request_text: str,
        previous_reply_text: str,
        confirmation_text: str,
        intent_type: str,
    ) -> str:
        action_line = {
            "project_inspect": "继续上一条你刚才主动提出的本地核对步骤，直接检查仓库、配置、代码或日志，并给我明确结论。",
            "feature_work": "继续上一条你刚才确认的本地执行步骤，直接进项目里检查、修改并验证。",
            "restart_only": "继续上一条你刚才主动提出的重启步骤，直接执行重启并告诉我结果。",
            "project_chat": "继续上一条。",
        }.get(intent_type, "继续上一条。")
        return "\n".join(
            [
                action_line,
                f"上一条用户问题：{previous_request_text.strip()}",
                f"上一条助手回复：{self._normalize_private_reply(previous_reply_text)}",
                f"用户确认继续：{confirmation_text.strip()}",
            ]
        )

    def _normalize_private_timestamp(self, value):
        if getattr(value, "tzinfo", None) is None:
            return value.replace(tzinfo=UTC)
        return value

    def _recent_private_owner_turns(self, recent_turns: list[str]) -> list[str]:
        return [
            line.split("Owner: ", maxsplit=1)[1].strip()
            for line in recent_turns
            if line.startswith("Owner: ")
        ]

    def _extract_private_character_work_hint_from_text(self, text: str) -> str | None:
        raw_text = str(text or "").strip()
        if not raw_text:
            return None

        quoted_match = re.search(r"《\s*([^》]{1,40})\s*》", raw_text)
        if quoted_match:
            title = quoted_match.group(1).strip()
            if title:
                return title

        patterns = (
            r"(?:这是|这是个|这是部|这是款|这个是|那是|这是|是|在|来自|出自)?\s*([0-9A-Za-z\u4e00-\u9fff·・ー]{1,40}?)(?:这个)?(?:游戏|作品|番|动画|动漫|galgame|gal|视觉小说)",
            r"(?:这是|这是个|这是部|这是款|这个是|那是|这是|是|在|来自|出自)?\s*([0-9A-Za-z\u4e00-\u9fff·・ー]{1,40}?)(?:里|中的)(?:人物|角色)",
        )
        for pattern in patterns:
            match = re.search(pattern, raw_text, flags=re.IGNORECASE)
            if match is None:
                continue
            title = match.group(1).strip(" \t\r\n`'\"“”‘’.,，。！？!?:：；;()（）[]【】")
            if title:
                return title
        return None

    def _find_private_character_work_hint(self, *, request_text: str, recent_turns: list[str]) -> str | None:
        current_hint = self._extract_private_character_work_hint_from_text(request_text)
        if current_hint:
            return current_hint
        for owner_turn in reversed(self._recent_private_owner_turns(recent_turns)):
            hint = self._extract_private_character_work_hint_from_text(owner_turn)
            if hint:
                return hint
        return None

    def _build_private_character_reference_search(
        self,
        *,
        request_text: str,
        recent_turns: list[str],
    ) -> SearchDecision | None:
        if not self._is_private_character_identification_request(request_text):
            return None
        work_hint = self._find_private_character_work_hint(
            request_text=request_text,
            recent_turns=recent_turns,
        )
        if not work_hint:
            return None
        return SearchDecision(True, f"{work_hint} 角色列表", "character-work-candidate-list")

    def _find_recent_private_weather_turn(self, recent_owner_turns: list[str]) -> str | None:
        for turn in reversed(recent_owner_turns):
            normalized = turn.strip()
            if "天气" in normalized:
                return normalized
        return None

    def _normalize_private_weather_followup_location(self, text: str) -> str:
        normalized = re.sub(r"[\n\r\t,，。！？!?:;；、“”‘’（）()《》【】\[\]<>]+", " ", text.strip())
        normalized = re.sub(
            r"^(那就|那|就|改成|换成|那换|那改成|那就按|那就查|那就搜|改查|改搜|查一下|查下|搜一下|搜下|查|搜)\s*",
            "",
            normalized,
        )
        normalized = re.sub(r"(吧|呢|呀|啊|可以吗|行吗)$", "", normalized).strip()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _looks_like_private_location_fragment(self, text: str) -> bool:
        normalized = text.strip()
        if not normalized or len(normalized) < 2 or len(normalized) > 24:
            return False
        if any(token in normalized for token in ("天气", "联网", "上网", "搜索", "搜", "查", "评价", "口碑", "新闻")):
            return False
        return any(
            token in normalized
            for token in (
                "省",
                "市",
                "区",
                "县",
                "镇",
                "乡",
                "村",
                "校区",
                "大学",
                "学校",
                "路",
                "街",
                "巷",
                "站",
                "机场",
                "广场",
                "商场",
                "医院",
                "酒店",
                "景区",
                "公园",
            )
        )

    def _extract_private_weather_time_hint(self, *texts: str) -> str:
        for text in texts:
            normalized = text.strip()
            for hint in ("今天", "明天", "后天", "今晚", "今早", "今晨", "今夜", "现在", "当前", "本周", "这周", "周末"):
                if hint in normalized:
                    return hint
        return "今天"

    def _build_contextual_private_search(
        self,
        *,
        request_text: str,
        recent_turns: list[str],
        search_reference_time,
    ) -> SearchDecision | None:
        normalized_request = request_text.strip()
        if not normalized_request or "天气" in normalized_request:
            return None
        if (
            is_explicit_search_request(normalized_request)
            or needs_reference_search(normalized_request)
            or needs_external_lookup_search(normalized_request)
            or is_general_search_decision_candidate(normalized_request)
        ):
            return None

        recent_owner_turns = self._recent_private_owner_turns(recent_turns)
        weather_anchor = self._find_recent_private_weather_turn(recent_owner_turns)
        if weather_anchor is None:
            return None

        location_text = self._normalize_private_weather_followup_location(normalized_request)
        if not self._looks_like_private_location_fragment(location_text):
            return None

        time_hint = self._extract_private_weather_time_hint(location_text, weather_anchor)
        query = normalize_relative_time_query(f"{location_text} {time_hint}天气", now=search_reference_time)
        return SearchDecision(True, query, "weather-followup-context")

    def _looks_like_local_project_question(self, text: str) -> bool:
        lowered = text.lower()
        has_runtime_target = any(keyword in lowered or keyword in text for keyword in PROJECT_RUNTIME_HINTS)
        if not has_runtime_target:
            return False
        has_health_or_lookup_intent = any(keyword in lowered or keyword in text for keyword in PROJECT_HEALTH_HINTS)
        has_existing_inspect_keyword = any(keyword in lowered or keyword in text for keyword in INSPECT_KEYWORDS)
        has_config_lookup_intent = self._looks_like_local_config_question(text)
        return has_health_or_lookup_intent or has_existing_inspect_keyword or has_config_lookup_intent

    def _looks_like_local_config_question(self, text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered or keyword in text for keyword in CONFIG_LOOKUP_KEYWORDS)

    def _build_private_web_context(
        self,
        *,
        request_text: str,
        request_time,
        recent_turns: list[str],
    ) -> PrivateWebContext:
        runtime_facts: list[str] = []
        web_results: list[str] = []
        web_pages: list[str] = []
        grounding_notes: list[str] = []

        search_reference_time = self._normalize_private_timestamp(request_time).astimezone()
        if needs_current_datetime_context(request_text):
            runtime_facts = build_current_datetime_facts(search_reference_time)

        if self.web_search_client is None or runtime_facts or self._looks_like_local_project_question(request_text):
            return PrivateWebContext(
                runtime_facts=runtime_facts,
                web_results=web_results,
                web_pages=web_pages,
                grounding_notes=grounding_notes,
            )

        if is_search_verification_query(request_text):
            return PrivateWebContext(
                runtime_facts=runtime_facts,
                web_results=web_results,
                web_pages=web_pages,
                grounding_notes=grounding_notes,
            )

        explicit_search_request = is_explicit_search_request(request_text)
        reference_search_request = needs_reference_search(request_text)
        external_lookup_search_request = needs_external_lookup_search(request_text)
        general_search_candidate = is_general_search_decision_candidate(request_text)
        time_sensitive = is_time_sensitive_request(request_text)
        contextual_followup_search = self._build_contextual_private_search(
            request_text=request_text,
            recent_turns=recent_turns,
            search_reference_time=search_reference_time,
        )

        forced_search_request = (
            explicit_search_request
            or reference_search_request
            or external_lookup_search_request
            or contextual_followup_search is not None
        )
        optional_search_eligible = (time_sensitive or general_search_candidate) and not forced_search_request
        if not forced_search_request and not optional_search_eligible:
            return PrivateWebContext(
                runtime_facts=runtime_facts,
                web_results=web_results,
                web_pages=web_pages,
                grounding_notes=grounding_notes,
            )

        if contextual_followup_search is not None:
            parsed_search = contextual_followup_search
        elif forced_search_request:
            parsed_search = SearchDecision(
                True,
                normalize_relative_time_query(
                    build_forced_search_query(request_text, bot_names=self._private_search_names()),
                    now=search_reference_time,
                ),
                (
                    "reference-topic-required"
                    if reference_search_request
                    else "local-lookup-required"
                    if external_lookup_search_request
                    else "explicit-search-request"
                ),
            )
        else:
            search_prompt = build_search_decision_prompt(
                bot_name=self.assistant_name,
                target_message=f"Owner: {request_text}",
                recent_messages=recent_turns,
                proactive_turn=False,
                now=search_reference_time,
            )
            try:
                parsed_search = parse_search_decision(self.llm_client.generate_text(search_prompt))
                if parsed_search.should_search:
                    parsed_search = SearchDecision(
                        True,
                        normalize_relative_time_query(parsed_search.query, now=search_reference_time),
                        parsed_search.reason,
                    )
            except Exception:
                logger.exception("private_web_search_decision_failed owner_qq=%s", self.owner_qq)
                parsed_search = SearchDecision(False, "", "search-decision-error")

        if not parsed_search.should_search:
            return PrivateWebContext(
                runtime_facts=runtime_facts,
                web_results=web_results,
                web_pages=web_pages,
                grounding_notes=grounding_notes,
            )

        search_result_limit = 5 if reference_search_request or external_lookup_search_request else 3
        try:
            search_hits = self.web_search_client.search(parsed_search.query, max_results=search_result_limit)
        except Exception:
            logger.exception("private_web_search_execute_failed owner_qq=%s query=%s", self.owner_qq, parsed_search.query)
            search_hits = []
        web_results = [
            f"{hit.title} | {hit.snippet} | {hit.source} | {hit.date}"
            for hit in search_hits
        ]

        try:
            page_reads = self.web_search_client.read_pages(
                search_hits,
                query=parsed_search.query,
                max_pages=3,
            )
        except Exception:
            logger.exception("private_web_page_fetch_failed owner_qq=%s query=%s", self.owner_qq, parsed_search.query)
            page_reads = []
        web_pages = [
            f"{page.title} | {page.url} | {page.content}"
            for page in page_reads
        ]
        recent_assistant_replies = [
            line.split("Assistant: ", maxsplit=1)[1]
            for line in recent_turns
            if line.startswith("Assistant: ")
        ]
        grounding_notes = build_grounding_notes(
            target_text=request_text,
            external_lookup=external_lookup_search_request,
            web_results=search_hits,
            web_pages=page_reads,
            recent_bot_replies=recent_assistant_replies,
        )
        return PrivateWebContext(
            runtime_facts=runtime_facts,
            web_results=web_results,
            web_pages=web_pages,
            grounding_notes=grounding_notes,
        )

    def _normalize_private_reply(self, text: str) -> str:
        return normalize_chat_reply(text).strip()

    def _rewrite_reply_after_successful_restart(self, text: str, *, restart_result: str) -> str:
        normalized = self._normalize_private_reply(text)
        if restart_result not in {"success", "recovered-on-start"} or not normalized:
            return normalized

        cleaned = normalized
        stripped_any = False
        for fragment in RESTART_DENIAL_REPLY_FRAGMENTS:
            pattern = rf"[^。！？!?]*{re.escape(fragment)}[^。！？!?]*[。！？!?]?"
            updated = re.sub(pattern, "", cleaned)
            if updated != cleaned:
                stripped_any = True
                cleaned = updated

        cleaned = re.sub(r"\s+", " ", cleaned).strip(" \n\t，,。；;")
        if not stripped_any:
            return cleaned
        if cleaned:
            return f"{cleaned} 另外，我已经替你把小町重启过了，现在运行中的实例就是按这次重启重新拉起来的。"
        return "我已经替你把小町重启过了，现在运行中的实例就是按这次重启重新拉起来的。"

    def _handle_private_session_command(self, *, raw_text: str) -> str | None:
        normalized = raw_text.strip().lower()
        if normalized not in SESSION_NEW_COMMANDS and normalized not in SESSION_STATUS_COMMANDS:
            return None

        with session_scope(self.engine) as session:
            sessions = DevSessionRepository(session)
            tasks = DevTaskRepository(session)
            active_session = sessions.get_or_create_owner_session(owner_qq=self.owner_qq)
            open_tasks = tasks.list_tasks_for_session_by_status(
                session_id=active_session.id,
                statuses=["queued", "running"],
            )

            if normalized in SESSION_NEW_COMMANDS:
                if open_tasks:
                    return "现在还有正在处理的执行任务，等它跑完我再给你开新会话。"
                sessions.create_owner_session(owner_qq=self.owner_qq)
                return "好，这里给你重新开了一条新的项目会话。"

            queued_count = len([task for task in open_tasks if task.status == "queued"])
            running_count = len([task for task in open_tasks if task.status == "running"])
            return (
                f"当前项目会话 #{active_session.id}，排队 {queued_count}，进行中 {running_count}。"
                "要重开就发 /bot new-session，也可以直接发“清空上下文”或“开新对话”。"
            )

    def _classify_intent(self, text: str) -> str:
        lowered = text.lower()
        if self._looks_like_restart_status_question(text):
            return "project_inspect"
        if self._looks_like_change_status_question(text):
            return "project_inspect"
        if self._looks_like_capability_probe_request(text):
            return "feature_work"
        if any(keyword in lowered or keyword in text for keyword in EXECUTE_KEYWORDS):
            return "feature_work"
        if any(keyword in lowered or keyword in text for keyword in RESTART_KEYWORDS):
            return "restart_only"
        if self._looks_like_private_send_request(text):
            return "project_chat"
        if self._looks_like_local_project_question(text):
            return "project_inspect"
        if any(keyword in lowered or keyword in text for keyword in INSPECT_KEYWORDS):
            return "project_inspect"
        if self._looks_like_fast_chat_request(text):
            return "project_chat"
        llm_intent = self._classify_owner_intent_with_llm(text)
        if llm_intent is not None:
            return llm_intent
        return "project_chat"

    def _looks_like_restart_status_question(self, text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        lowered = normalized.lower()
        if not any(keyword in lowered or keyword in normalized for keyword in RESTART_KEYWORDS):
            return False
        if any(hint in normalized for hint in RESTART_STATUS_HINTS):
            return True
        if "重启" in normalized and any(hint in normalized for hint in ("已经", "是否", "是不是", "确认")):
            return True
        if ("吗" in normalized or "？" in normalized or "?" in normalized) and any(
            hint in normalized for hint in ("生效", "成功", "好了", "完成")
        ):
            return True
        return False

    def _looks_like_change_status_question(self, text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        has_question_shape = (
            any(token in normalized for token in ("吗", "？", "?"))
            or "会不会" in normalized
            or "是不是" in normalized
            or "有没有" in normalized
            or "确认" in normalized
        )
        if not has_question_shape:
            return False
        matched_hints = [hint for hint in CHANGE_STATUS_HINTS if hint in normalized]
        if not matched_hints:
            return False
        if any(hint in normalized for hint in ("生效", "改动", "会@", "会不会@", "会艾特", "会不会艾特")):
            return True
        if any(hint in normalized for hint in ("加上", "加入", "改好", "会提到", "会不会提到")):
            return any(hint in normalized for hint in CHANGE_STATUS_TARGET_HINTS)
        return False

    def _looks_like_capability_probe_request(self, text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        lowered = normalized.lower()
        has_target = any(keyword in lowered or keyword in normalized for keyword in CAPABILITY_PROBE_TARGET_HINTS)
        if not has_target:
            return False
        return any(keyword in lowered or keyword in normalized for keyword in CAPABILITY_PROBE_ACTION_HINTS)

    def _looks_like_fast_chat_request(self, text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return True
        return any(hint in normalized for hint in OWNER_FAST_CHAT_HINTS)

    def _classify_owner_intent_with_llm(self, text: str) -> str | None:
        prompt_lines = [
            "You are classifying the owner's private development-channel request for a local coding agent.",
            "The owner has full authority over this repository and bot runtime.",
            "Intent labels:",
            "- EXECUTE: enter repo work mode, inspect/edit code, verify capability, add or change functionality, or perform an operational task that may require code or runtime changes.",
            "- INSPECT: only inspect local logs/config/code/runtime state and answer inline, with no code changes.",
            "- CHAT: normal conversation, identity questions, or pure general-knowledge discussion with no local project work.",
            "Bias toward EXECUTE when the owner is asking you to do something in the project or bot, even if the wording is informal or imperative.",
            f"Owner message: {text.strip()}",
            "Reply with exactly one label: EXECUTE, INSPECT, or CHAT.",
        ]
        try:
            raw_label = str(self.llm_client.generate_text(prompt_lines)).strip().upper()
        except Exception:
            logger.exception("owner_intent_llm_classification_failed owner_qq=%s", self.owner_qq)
            return None
        if "EXECUTE" in raw_label:
            return "feature_work"
        if "INSPECT" in raw_label:
            return "project_inspect"
        if "CHAT" in raw_label:
            return "project_chat"
        return None

    def _collect_inspection_snapshot(self, *, request_text: str) -> InspectionSnapshot:
        lowered = request_text.lower()
        sections: list[str] = [self._build_runtime_overview()]
        files_read: list[str] = []
        commands_run: list[str] = []

        if any(keyword in lowered or keyword in request_text for keyword in LOG_KEYWORDS) or self._looks_like_local_project_question(request_text):
            log_sections, log_files = self._build_log_sections()
            sections.extend(log_sections)
            files_read.extend(log_files)

        if any(keyword in lowered or keyword in request_text for keyword in GIT_STATUS_KEYWORDS):
            git_block = self._run_git_status_snapshot()
            sections.append(git_block)
            commands_run.append("git status --short --branch")

        if self._looks_like_local_config_question(request_text):
            config_sections, config_files = self._build_config_lookup_sections(request_text=request_text)
            sections.extend(config_sections)
            files_read.extend(config_files)

        if any(keyword in lowered or keyword in request_text for keyword in CODE_LOOKUP_KEYWORDS):
            sections.append("Code lookup hint: relevant repository snippets are attached below.")

        return InspectionSnapshot(
            prompt_block="\n\n".join(section for section in sections if section.strip()),
            files_read=files_read,
            commands_run=commands_run or ["local inspection"],
        )

    def _build_runtime_overview(self) -> str:
        log_dir = self.data_dir / "logs"
        lines = [
            self._describe_pid_file(log_dir / "group.pid", label="group process"),
            self._describe_pid_file(log_dir / "private.pid", label="private process"),
            self._describe_pid_file(log_dir / "worker.pid", label="worker process"),
            self._describe_file_state(log_dir / "group.stderr.log", label="group stderr log"),
            self._describe_file_state(log_dir / "private.stderr.log", label="private stderr log"),
            self._describe_file_state(log_dir / "worker.stderr.log", label="worker stderr log"),
        ]
        return "Runtime overview:\n" + "\n".join(f"- {line}" for line in lines)

    def _build_log_sections(self) -> tuple[list[str], list[str]]:
        sections: list[str] = []
        files_read: list[str] = []
        for path, label in (
            (self.data_dir / "logs" / "group.stderr.log", "Latest group stderr tail"),
            (self.data_dir / "logs" / "group.stdout.log", "Latest group stdout tail"),
            (self.data_dir / "logs" / "private.stderr.log", "Latest private stderr tail"),
            (self.data_dir / "logs" / "private.stdout.log", "Latest private stdout tail"),
            (self.data_dir / "logs" / "worker.stderr.log", "Latest worker stderr tail"),
            (self.data_dir / "logs" / "worker.stdout.log", "Latest worker stdout tail"),
        ):
            excerpt = self._tail_text_file(path)
            if excerpt is None:
                continue
            sections.append(f"{label} ({path.as_posix()}):\n{excerpt}")
            files_read.append(str(path))
        if not sections:
            sections.append("Latest runtime logs: no bot process log file is available yet.")
        return sections, files_read

    def _run_git_status_snapshot(self) -> str:
        result = self.command_runner(["git", "status", "--short", "--branch"], self.repo_root)
        output = (result.stdout or result.stderr or "").strip()
        if result.returncode != 0:
            return f"Git status:\n(command failed) {output or 'git status returned a non-zero exit code'}"
        if not output:
            output = "working tree clean"
        return f"Git status:\n{output}"

    def _build_config_lookup_sections(self, *, request_text: str) -> tuple[list[str], list[str]]:
        sections: list[str] = []
        files_read: list[str] = []
        qq_ids = re.findall(r"\d{5,16}", request_text)

        env_path = self.repo_root / ".env"
        if env_path.exists():
            env_text = env_path.read_text(encoding="utf-8", errors="ignore")
            files_read.append(str(env_path))
            private_chat_value = ""
            for raw_line in env_text.splitlines():
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key.strip() != "PRIVATE_CHAT_QQS":
                    continue
                private_chat_value = value.strip().strip('"').strip("'")
                break

            env_lines = [
                f"PRIVATE_CHAT_QQS configured: {'yes' if private_chat_value else 'no'}",
            ]
            if private_chat_value:
                env_lines.append(f"PRIVATE_CHAT_QQS current value: {private_chat_value}")
                current_values = {item.strip() for item in private_chat_value.split(",") if item.strip()}
                for qq_id in qq_ids:
                    env_lines.append(
                        f"PRIVATE_CHAT_QQS includes {qq_id}: {'yes' if qq_id in current_values else 'no'}"
                    )
            sections.append("Local config facts:\n" + "\n".join(f"- {line}" for line in env_lines))

        for relative_path in ("app/config.py", "app/main.py", "app/private_main.py", "app/dev_control/service.py"):
            excerpt = self._match_file_excerpt(
                self.repo_root / relative_path,
                terms=("private_chat", "whitelist", "private", "qq"),
            )
            if excerpt is None:
                continue
            sections.append(f"Relevant config code ({relative_path}):\n{excerpt}")
            files_read.append(str(self.repo_root / relative_path))

        return sections, files_read

    def _match_file_excerpt(self, path: Path, *, terms: tuple[str, ...], max_lines: int = 6) -> str | None:
        if not path.exists() or not path.is_file():
            return None
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return None
        lowered_terms = tuple(term.lower() for term in terms if term)
        excerpts: list[str] = []
        for index, line in enumerate(lines, 1):
            lowered_line = line.lower()
            if not any(term in lowered_line for term in lowered_terms):
                continue
            excerpts.append(f"{index}: {line.strip()}")
            if len(excerpts) >= max_lines:
                break
        if not excerpts:
            return None
        return "\n".join(excerpts)

    def _describe_file_state(self, path: Path, *, label: str) -> str:
        if not path.exists():
            return f"{label}: missing"
        stat = path.stat()
        return f"{label}: present, {stat.st_size} bytes, modified {stat.st_mtime:.0f}"

    def _describe_pid_file(self, path: Path, *, label: str) -> str:
        pid_text = ""
        if path.exists():
            pid_text = path.read_text(encoding="utf-8", errors="ignore").strip()

        pid_state = "missing"
        if pid_text.isdigit():
            running = self._is_process_running(int(pid_text))
            if running is True:
                pid_state = f"running:{pid_text}"
            elif running is False:
                pid_state = f"stale:{pid_text}"
            else:
                pid_state = f"unknown:{pid_text}"
        return f"{label}: {pid_state}"

    def _tail_text_file(self, path: Path) -> str | None:
        if not path.exists() or not path.is_file():
            return None
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            return "(empty)"
        lines = text.splitlines()
        excerpt = "\n".join(lines[-LOG_TAIL_LINE_LIMIT:])
        if len(excerpt) > LOG_TAIL_CHAR_LIMIT:
            excerpt = excerpt[-LOG_TAIL_CHAR_LIMIT:]
        return excerpt

    def _is_process_running(self, pid: int) -> bool | None:
        result = self.command_runner(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ 'running' }} else {{ 'stopped' }}",
            ],
            self.repo_root,
        )
        if result.returncode != 0:
            return None
        output = (result.stdout or "").strip().lower()
        if output == "running":
            return True
        if output == "stopped":
            return False
        return None

    def _session_summary(self, *, session_id: int) -> str:
        with session_scope(self.engine) as session:
            dev_session = session.get(DevSession, session_id)
            if dev_session is None:
                return ""
            return str(getattr(dev_session, "summary", "") or "")

    def _private_outbound_platform_msg_id(self, *, context: str) -> str:
        return f"private-outbound-{context}"

    def _private_sender_user_id(self) -> int:
        return self.bot_qq if isinstance(self.bot_qq, int) and self.bot_qq > 0 else self.owner_qq

    def _private_outbound_sent_text(self, *, context: str) -> str | None:
        platform_msg_id = self._private_outbound_platform_msg_id(context=context)
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            outbound_message = messages.get_by_platform_msg_id(platform_msg_id)
            if outbound_message is None:
                return None
            raw_json = outbound_message.raw_json
            if not isinstance(raw_json, dict) or raw_json.get("delivery_state") != "sent":
                return None
            return str(outbound_message.plain_text or "")

    def _should_send_private_recovery_reply(
        self,
        *,
        task_id: int,
        recovery_text: str,
    ) -> bool:
        completed_text = self._private_outbound_sent_text(context=f"dev_task:{task_id}:completed")
        if completed_text is None:
            return True

        normalized_completed = self._normalize_private_reply(completed_text)
        normalized_recovery = self._normalize_private_reply(recovery_text)
        if not normalized_completed:
            return True
        if normalized_recovery != normalized_completed and not normalized_recovery.endswith(normalized_completed):
            return True

        logger.info("private_reply_recovery_skip_duplicate task_id=%s", task_id)
        return False

    def _reserve_private_outbound_reply(self, *, user_id: int, reply_text: str, context: str) -> bool:
        platform_msg_id = self._private_outbound_platform_msg_id(context=context)
        sender_user_id = self._private_sender_user_id()
        with session_scope(self.engine) as session:
            users = UserRepository(session)
            messages = MessageRepository(session)
            existing = messages.get_by_platform_msg_id(platform_msg_id)
            if existing is not None:
                return False

            users.upsert_user(
                user_id=sender_user_id,
                nickname=self.assistant_name,
                group_card="",
            )
            messages.add_private_message(
                platform_msg_id=platform_msg_id,
                user_id=sender_user_id,
                timestamp=datetime.now().astimezone(),
                plain_text=reply_text,
                raw_json={
                    "direction": "outbound",
                    "recipient_user_id": user_id,
                    "delivery_state": "reserved",
                    "context": context,
                },
                msg_type="text",
                reply_to_msg_id=None,
                mentioned_bot=False,
            )
            return True

    def _mark_private_outbound_reply_sent(self, *, user_id: int, reply_text: str, context: str) -> None:
        platform_msg_id = self._private_outbound_platform_msg_id(context=context)
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            outbound_message = messages.get_by_platform_msg_id(platform_msg_id)
            if outbound_message is None:
                return
            outbound_message.plain_text = reply_text
            outbound_message.raw_json = {
                "direction": "outbound",
                "recipient_user_id": user_id,
                "delivery_state": "sent",
                "context": context,
            }
            session.add(outbound_message)

    def _clear_private_outbound_reply_reservation(self, *, context: str) -> None:
        platform_msg_id = self._private_outbound_platform_msg_id(context=context)
        with session_scope(self.engine) as session:
            messages = MessageRepository(session)
            outbound_message = messages.get_by_platform_msg_id(platform_msg_id)
            if outbound_message is None:
                return
            session.delete(outbound_message)

    def _recent_turn_lines(self, *, session_id: int, exclude_task_id: int | None = None) -> list[str]:
        with session_scope(self.engine) as session:
            recent_tasks = DevTaskRepository(session).list_recent_tasks_for_session(
                session_id=session_id,
                limit=RECENT_TURN_LIMIT + 2,
            )
        lines: list[str] = []
        for task in recent_tasks:
            if exclude_task_id is not None and task.id == exclude_task_id:
                continue
            lines.append(f"Owner: {task.raw_request_text}")
            if task.result_text:
                lines.append(f"Assistant: {task.result_text}")
            elif task.failure_reason:
                lines.append(f"Assistant: {task.failure_reason}")
        return lines[-SUMMARY_LINE_LIMIT:]

    def _append_session_summary(
        self,
        *,
        session_id: int,
        owner_text: str,
        assistant_text: str,
        sessions: DevSessionRepository,
    ) -> None:
        current_session = sessions.session.get(DevSession, session_id)
        current_summary = ""
        if current_session is not None:
            current_summary = str(getattr(current_session, "summary", "") or "")
        lines = current_summary.splitlines() if current_summary else []
        lines.extend(
            [
                f"Owner: {self._truncate_text(owner_text, limit=120)}",
                f"Assistant: {self._truncate_text(assistant_text, limit=180)}",
            ]
        )
        sessions.update_session(session_id=session_id, summary="\n".join(lines[-SUMMARY_LINE_LIMIT:]))

    def _build_turn_summary(self, owner_text: str, assistant_text: str) -> str:
        return (
            f"Owner asked: {self._truncate_text(owner_text, limit=80)} | "
            f"Assistant replied: {self._truncate_text(assistant_text, limit=120)}"
        )

    def _truncate_text(self, text: str, *, limit: int) -> str:
        value = " ".join(text.strip().split())
        if len(value) <= limit:
            return value
        return f"{value[: limit - 3]}..."

    def _get_codex_bridge(self) -> CodexBridge:
        if self.codex_bridge is None:
            self.codex_bridge = CodexBridge()
        return self.codex_bridge

    def _restart_relevant_request_text(self, text: str) -> str:
        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        if not lines or not lines[0].startswith("继续上一条"):
            return str(text or "")
        relevant_lines: list[str] = []
        for line in lines:
            if line.startswith("上一条助手回复：") or line.startswith("用户确认继续："):
                continue
            if line.startswith("上一条用户问题："):
                relevant_lines.append(line.split("：", 1)[1].strip())
                continue
            relevant_lines.append(line)
        return "\n".join(relevant_lines)

    def _request_wants_restart(self, text: str) -> bool:
        relevant_text = self._restart_relevant_request_text(text)
        lowered = relevant_text.lower()
        return any(keyword in lowered or keyword in relevant_text for keyword in RESTART_KEYWORDS)

    def _change_requires_restart(self, files_changed: list[str]) -> bool:
        restart_prefixes = ("app/", "configs/")
        restart_files = {
            "start-xiaomachi-wsl.bat",
            "stop-xiaomachi-wsl.bat",
            "infra/wsl/docker-compose.yml",
            "infra/wsl/scripts/start.sh",
            "infra/wsl/scripts/stop.sh",
            "infra/wsl/.env",
        }
        for relative_path in files_changed:
            normalized = relative_path.replace("\\", "/")
            if normalized.startswith(restart_prefixes):
                return True
            if normalized in restart_files:
                return True
        return False

    def _should_restart_after_task(
        self,
        *,
        request_text: str,
        files_changed: list[str],
        model_restart_required: bool,
    ) -> bool:
        if self._request_wants_restart(request_text):
            return True
        if self._change_requires_restart(files_changed):
            return True
        if model_restart_required and files_changed:
            return True
        return False

    def _detect_changed_files(self, *, checkpoint_dir: Path, manifest: dict[str, list[str]]) -> list[str]:
        snapshot_dir = checkpoint_dir / "snapshot"
        known_files = {Path(item) for item in manifest.get("files", [])}
        current_files = set()
        for current_path in self.repo_root.rglob("*"):
            if not current_path.is_file():
                continue
            if checkpoint_dir.resolve() in current_path.resolve().parents:
                continue
            if any(
                part in {"data", ".git", ".venv", "__pycache__", ".pytest_cache"}
                for part in current_path.relative_to(self.repo_root).parts
            ):
                continue
            current_files.add(current_path.relative_to(self.repo_root))
        changed: list[str] = []
        for relative_path in sorted(known_files | current_files):
            before_path = snapshot_dir / relative_path
            after_path = self.repo_root / relative_path
            if not before_path.exists() or not after_path.exists():
                changed.append(relative_path.as_posix())
                continue
            if before_path.read_bytes() != after_path.read_bytes():
                changed.append(relative_path.as_posix())
        return changed

    async def _recover_running_tasks(self) -> None:
        with session_scope(self.engine) as session:
            running_tasks = list(DevTaskRepository(session).list_tasks_by_status("running"))
            queued_tasks = list(DevTaskRepository(session).list_tasks_by_status("queued"))

        pending_private_image_task_ids = self.private_image_service.pending_dev_task_ids()
        stale_non_execute_tasks = [
            task
            for task in [*running_tasks, *queued_tasks]
            if task.intent_type not in ASYNC_EXECUTE_INTENTS and task.id not in pending_private_image_task_ids
        ]

        for task in stale_non_execute_tasks:
            logger.info("dev_task_stage task_id=%s stage=mark_stale_non_execute intent=%s status=%s", task.id, task.intent_type, task.status)
            failure_reason = f"stale non-execute task left in {task.status} state"
            with session_scope(self.engine) as session:
                tasks = DevTaskRepository(session)
                tasks.mark_failed(
                    task_id=task.id,
                    failure_reason=failure_reason,
                )
                self._append_session_summary(
                    session_id=task.session_id,
                    owner_text=task.raw_request_text,
                    assistant_text=f"失败：{failure_reason}",
                    sessions=DevSessionRepository(session),
                )

        for task in running_tasks:
            if task.intent_type not in ASYNC_EXECUTE_INTENTS:
                continue
            logger.info("dev_task_stage task_id=%s stage=recovering_running_task", task.id)
            recovered_reply = self._recover_task_from_saved_result(
                task_id=task.id,
                session_id=task.session_id,
                owner_qq=task.requested_by_qq,
                request_text=task.raw_request_text,
                notification_prefix="刚才那条任务其实已经跑完了，我把结果补发给你：",
                allow_restart=False,
            )
            if recovered_reply is not None:
                if self._should_send_private_recovery_reply(task_id=task.id, recovery_text=recovered_reply):
                    await self._send_private_text(
                        user_id=task.requested_by_qq,
                        text=recovered_reply,
                        context=f"dev_task:{task.id}:recovered_on_start",
                        timeout_seconds=8.0,
                    )
                continue

            with session_scope(self.engine) as session:
                tasks = DevTaskRepository(session)
                tasks.mark_failed(
                    task_id=task.id,
                    failure_reason="service restarted before task completion",
                    checkpoint_dir=str(self.checkpoint_root / f"task-{task.id}"),
                )
                self._append_session_summary(
                    session_id=task.session_id,
                    owner_text=task.raw_request_text,
                    assistant_text="失败：service restarted before task completion",
                    sessions=DevSessionRepository(session),
                )
            await self._send_private_text(
                user_id=task.requested_by_qq,
                text="上一条任务在收尾前断掉了，我先记成失败了。你再丢我一句，我继续接着弄。",
                context=f"dev_task:{task.id}:failed_on_start",
                timeout_seconds=8.0,
            )

    def _checkpoint_manifest_path(self, checkpoint_dir: Path) -> Path:
        return checkpoint_dir / "manifest.json"

    def _write_checkpoint_manifest(self, *, checkpoint_dir: Path, manifest: dict[str, list[str]]) -> None:
        self._checkpoint_manifest_path(checkpoint_dir).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_checkpoint_manifest(self, *, checkpoint_dir: Path) -> dict[str, list[str]] | None:
        manifest_path = self._checkpoint_manifest_path(checkpoint_dir)
        if not manifest_path.exists():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("checkpoint_manifest_read_failed path=%s", manifest_path)
            return None
        if not isinstance(payload, dict):
            return None
        files = payload.get("files")
        if not isinstance(files, list):
            return None
        normalized_files = [str(item) for item in files if item]
        return {"files": normalized_files}

    def _task_last_message_path(self, *, task_id: int) -> Path:
        return self.task_dir / f"task-{task_id}" / "codex.last_message.json"

    def _write_saved_task_result(self, *, task_id: int, result: CodexTaskResult, overwrite: bool = False) -> None:
        last_message_path = self._task_last_message_path(task_id=task_id)
        if last_message_path.exists() and not overwrite:
            return
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        raw_last_message = str(result.raw_last_message or "").strip()
        if raw_last_message:
            try:
                json.loads(raw_last_message)
            except Exception:
                logger.exception("dev_task_saved_result_invalid_json task_id=%s", task_id)
            else:
                last_message_path.write_text(raw_last_message, encoding="utf-8")
                return
        payload = {
            "summary": result.summary,
            "reply_text": result.reply_text,
            "restart_required": bool(result.restart_required),
        }
        if result.thread_id:
            payload["thread_id"] = result.thread_id
        last_message_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _load_saved_task_result(self, *, task_id: int) -> CodexTaskResult | None:
        last_message_path = self._task_last_message_path(task_id=task_id)
        if not last_message_path.exists():
            return None
        try:
            payload = json.loads(last_message_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("dev_task_saved_result_read_failed task_id=%s", task_id)
            return None
        summary = str(payload.get("summary", "") or "").strip()
        reply_text = str(payload.get("reply_text", "") or "").strip()
        if not reply_text:
            return None
        return CodexTaskResult(
            summary=summary,
            reply_text=reply_text,
            restart_required=bool(payload.get("restart_required", False)),
            raw_last_message=json.dumps(payload, ensure_ascii=False),
        )

    def _recover_task_from_saved_result(
        self,
        *,
        task_id: int,
        session_id: int,
        owner_qq: int,
        request_text: str,
        notification_prefix: str,
        checkpoint_manifest: dict[str, list[str]] | None = None,
        allow_restart: bool = True,
    ) -> str | None:
        saved_result = self._load_saved_task_result(task_id=task_id)
        if saved_result is None:
            return None

        checkpoint_dir = self.checkpoint_root / f"task-{task_id}"
        manifest = checkpoint_manifest or self._load_checkpoint_manifest(checkpoint_dir=checkpoint_dir)
        files_changed: list[str] = []
        if manifest is not None and (checkpoint_dir / "snapshot").exists():
            files_changed = self._detect_changed_files(checkpoint_dir=checkpoint_dir, manifest=manifest)

        restart_required = self._should_restart_after_task(
            request_text=request_text,
            files_changed=files_changed,
            model_restart_required=saved_result.restart_required,
        )
        commands_run = ["artifact recovery"]
        restart_result = "not-needed"

        if restart_required and allow_restart:
            restart_ok, restart_result_text = self._restart_runtime()
            commands_run.extend(["xiaomachi-wsl-entry.sh stop", "xiaomachi-wsl-entry.sh start"])
            if not restart_ok:
                if manifest is not None and (checkpoint_dir / "snapshot").exists():
                    restore_repo_checkpoint(
                        repo_root=self.repo_root,
                        checkpoint_dir=checkpoint_dir,
                        manifest=manifest,
                    )
                    rollback_ok, rollback_result_text = self._restart_runtime()
                    commands_run.extend(["xiaomachi-wsl-entry.sh stop", "xiaomachi-wsl-entry.sh start"])
                    with session_scope(self.engine) as session:
                        tasks = DevTaskRepository(session)
                        if rollback_ok:
                            tasks.mark_status(task_id=task_id, status="rolled_back")
                            rolled_back = tasks.get_task(task_id)
                            if rolled_back is not None:
                                rolled_back.failure_reason = restart_result_text
                                rolled_back.restart_result = "rolled_back"
                            self._append_session_summary(
                                session_id=session_id,
                                owner_text=request_text,
                                assistant_text="启动失败，已自动回滚。",
                                sessions=DevSessionRepository(session),
                            )
                            return "刚才那条任务收尾时重启失败了，我已经先回滚并拉起来了。"

                        tasks.mark_failed(
                            task_id=task_id,
                            failure_reason=rollback_result_text,
                            checkpoint_dir=str(checkpoint_dir),
                        )
                        self._append_session_summary(
                            session_id=session_id,
                            owner_text=request_text,
                            assistant_text=f"失败：{rollback_result_text}",
                            sessions=DevSessionRepository(session),
                        )
                    return f"刚才那条任务收尾时重启失败了，而且回滚后也没恢复：{rollback_result_text}"

                with session_scope(self.engine) as session:
                    tasks = DevTaskRepository(session)
                    tasks.mark_failed(
                        task_id=task_id,
                        failure_reason=restart_result_text,
                        checkpoint_dir=str(checkpoint_dir),
                    )
                    self._append_session_summary(
                        session_id=session_id,
                        owner_text=request_text,
                        assistant_text=f"失败：{restart_result_text}",
                        sessions=DevSessionRepository(session),
                    )
                return f"刚才那条任务代码已经跑完了，但补重启失败了：{restart_result_text}"
            restart_result = "success"
        elif restart_required:
            restart_result = "recovered-on-start"

        normalized_reply_text = self._rewrite_reply_after_successful_restart(
            saved_result.reply_text,
            restart_result=restart_result,
        )
        summary = saved_result.summary or self._build_turn_summary(request_text, normalized_reply_text)
        with session_scope(self.engine) as session:
            tasks = DevTaskRepository(session)
            tasks.mark_completed(
                task_id=task_id,
                summary=summary,
                result_text=normalized_reply_text,
                files_read=[],
                files_changed=files_changed,
                commands_run=commands_run,
                restart_required=restart_required,
                restart_result=restart_result,
                checkpoint_dir=str(checkpoint_dir),
            )
            self._append_session_summary(
                session_id=session_id,
                owner_text=request_text,
                assistant_text=normalized_reply_text,
                sessions=DevSessionRepository(session),
            )
        logger.info("dev_task_stage task_id=%s stage=recovered_from_saved_result", task_id)
        return f"{notification_prefix}{normalized_reply_text}"

    def _handoff_inline_runtime_restart(self) -> tuple[bool, str]:
        restart_result = self.command_runner(
            [
                "wsl.exe",
                "bash",
                "-lc",
                "bash /mnt/d/xiaomachi-wsl-entry.sh stop && bash /mnt/d/xiaomachi-wsl-entry.sh start",
            ],
            self.repo_root,
        )
        if restart_result.returncode != 0:
            return False, restart_result.stderr or restart_result.stdout or "restart handoff script failed"
        return True, restart_result.stdout or "restart handoff launched"

    def _restart_runtime(self) -> tuple[bool, str]:
        stop_result = self.command_runner(
            ["wsl.exe", "bash", "/mnt/d/xiaomachi-wsl-entry.sh", "stop"],
            self.repo_root,
        )
        start_result = self.command_runner(
            ["wsl.exe", "bash", "/mnt/d/xiaomachi-wsl-entry.sh", "start"],
            self.repo_root,
        )
        if stop_result.returncode != 0:
            return False, stop_result.stderr or stop_result.stdout or "stop script failed"
        if start_result.returncode != 0:
            return False, start_result.stderr or start_result.stdout or "start script failed"
        return True, "success"

    def _get_codex_thread_id(self, *, session_id: int) -> str | None:
        thread_map = self._load_codex_thread_map()
        return thread_map.get(str(session_id)) or None

    def _set_codex_thread_id(self, *, session_id: int, thread_id: str) -> None:
        self.control_dir.mkdir(parents=True, exist_ok=True)
        thread_map = self._load_codex_thread_map()
        thread_map[str(session_id)] = thread_id
        self.codex_thread_state_path.write_text(
            json.dumps(thread_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_codex_thread_map(self) -> dict[str, str]:
        if not self.codex_thread_state_path.exists():
            return {}
        try:
            payload = json.loads(self.codex_thread_state_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("codex_thread_map_read_failed path=%s", self.codex_thread_state_path)
            return {}
        if not isinstance(payload, dict):
            return {}
        normalized: dict[str, str] = {}
        for key, value in payload.items():
            if not key or not value:
                continue
            normalized[str(key)] = str(value)
        return normalized

    def _normalize_private_command_text(self, text: str) -> str:
        return "".join(text.strip().lower().split())

    def _is_private_admin_user(self, user_id: int) -> bool:
        return user_id == self.owner_qq or user_id in self.admin_qqs

    async def _maybe_send_private_admin_intro_messages(self) -> None:
        pending_user_ids = sorted(self.admin_qqs - self._load_private_admin_intro_state())
        for user_id in pending_user_ids:
            delivered = await self._send_private_text(
                user_id=user_id,
                text=self._build_private_admin_intro_text(),
                context=f"private_admin_intro:{user_id}",
            )
            if delivered:
                self._mark_private_admin_intro_sent(user_id)

    def _build_private_admin_intro_text(self) -> str:
        return (
            "你现在已经可以用小町的私聊管理员模式了。\n\n"
            "用法：\n"
            "1. 先发“启动管理员模式”进入项目对话\n"
            "2. 直接说你要查什么、改什么、排查什么\n"
            "3. 发“重置会话”可以开一条新的项目会话\n"
            "4. 发“结束管理员模式”或“退出管理员模式”回到普通私聊"
        )

    def _load_private_admin_intro_state(self) -> set[int]:
        if not self.private_admin_intro_state_path.exists():
            return set()
        try:
            payload = json.loads(self.private_admin_intro_state_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("private_admin_intro_state_read_failed path=%s", self.private_admin_intro_state_path)
            return set()
        if not isinstance(payload, dict):
            return set()
        raw_user_ids = payload.get("sent_user_ids", [])
        if not isinstance(raw_user_ids, list):
            return set()
        sent_user_ids: set[int] = set()
        for item in raw_user_ids:
            try:
                sent_user_ids.add(int(item))
            except (TypeError, ValueError):
                continue
        return sent_user_ids

    def _mark_private_admin_intro_sent(self, user_id: int) -> None:
        sent_user_ids = self._load_private_admin_intro_state()
        sent_user_ids.add(user_id)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.private_admin_intro_state_path.write_text(
            json.dumps(
                {
                    "sent_user_ids": sorted(sent_user_ids),
                    "updated_at": datetime.now().astimezone().isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _handle_owner_mode_switch_command(self, *, raw_text: str, user_id: int | None = None) -> str | None:
        target_user_id = self.owner_qq if user_id is None else user_id
        normalized = self._normalize_private_command_text(raw_text)
        enable_commands = {self._normalize_private_command_text(item) for item in OWNER_MODE_ENABLE_COMMANDS}
        disable_commands = {self._normalize_private_command_text(item) for item in OWNER_MODE_DISABLE_COMMANDS}
        if normalized not in enable_commands and normalized not in disable_commands:
            return None

        current_mode = self._get_owner_private_session_mode(user_id=target_user_id)
        if normalized in enable_commands:
            if current_mode == SESSION_MODE_PROJECT:
                return "现在已经在管理员模式里了。接下来你直接说要查什么、改什么就行。"
            self._set_owner_private_session_mode(SESSION_MODE_PROJECT, user_id=target_user_id)
            return "好，已经切到管理员模式了。接下来这条私聊会进入项目对话，上下文和普通聊天分开记。"

        if current_mode == SESSION_MODE_DAILY:
            return "现在本来就是普通对话模式。"
        self._set_owner_private_session_mode(SESSION_MODE_DAILY, user_id=target_user_id)
        return "好，已经退出管理员模式了。后面这条私聊会回到普通聊天，上下文也和项目对话分开。"

    def _get_owner_private_session_mode(self, *, user_id: int | None = None) -> str:
        target_user_id = self.owner_qq if user_id is None else user_id
        payload = self._load_owner_private_mode_state()
        if target_user_id == self.owner_qq:
            session_mode = str(payload.get("session_mode", SESSION_MODE_DAILY) or SESSION_MODE_DAILY)
        else:
            session_mode = SESSION_MODE_DAILY
            user_modes = payload.get("user_modes", {})
            if isinstance(user_modes, dict):
                user_payload = user_modes.get(str(target_user_id), {})
                if isinstance(user_payload, dict):
                    session_mode = str(user_payload.get("session_mode", SESSION_MODE_DAILY) or SESSION_MODE_DAILY)
        if session_mode not in {SESSION_MODE_DAILY, SESSION_MODE_PROJECT}:
            return SESSION_MODE_DAILY
        return session_mode

    def _set_owner_private_session_mode(self, session_mode: str, *, user_id: int | None = None) -> None:
        target_user_id = self.owner_qq if user_id is None else user_id
        normalized_mode = session_mode if session_mode in {SESSION_MODE_DAILY, SESSION_MODE_PROJECT} else SESSION_MODE_DAILY
        payload = self._load_owner_private_mode_state()
        if not isinstance(payload, dict):
            payload = {}
        now_text = datetime.now().astimezone().isoformat()
        if target_user_id == self.owner_qq:
            payload["owner_qq"] = self.owner_qq
            payload["session_mode"] = normalized_mode
            payload["updated_at"] = now_text
        else:
            user_modes = payload.get("user_modes", {})
            if not isinstance(user_modes, dict):
                user_modes = {}
            user_modes[str(target_user_id)] = {
                "session_mode": normalized_mode,
                "updated_at": now_text,
            }
            payload["user_modes"] = user_modes
            payload.setdefault("owner_qq", self.owner_qq)
            payload.setdefault("session_mode", SESSION_MODE_DAILY)
            payload.setdefault("updated_at", now_text)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.owner_private_mode_state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_owner_private_mode_state(self) -> dict[str, object]:
        if not self.owner_private_mode_state_path.exists():
            return {}
        try:
            payload = json.loads(self.owner_private_mode_state_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("owner_private_mode_state_read_failed path=%s", self.owner_private_mode_state_path)
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): value for key, value in payload.items() if key}

    def _parse_explicit_private_send_request(self, text: str) -> ExplicitPrivateSendRequest | None:
        normalized = str(text or "").strip()
        if not normalized:
            return None
        if not self._looks_like_private_send_request(normalized):
            return None
        target_match = re.search(r"\d{5,16}", normalized)
        if target_match is None:
            return None
        remainder = normalized[target_match.end() :].strip()
        if not remainder:
            return None

        send_markers = (
            "私聊发送",
            "发送私聊",
            "发私聊",
            "发私信",
            "私信发送",
            "发送私信",
            "发送消息",
            "发消息",
            "发送",
            "发",
        )
        message_text = ""
        for marker in send_markers:
            marker_index = remainder.find(marker)
            if marker_index < 0:
                continue
            message_text = remainder[marker_index + len(marker) :].strip()
            break
        if not message_text:
            return None

        message_text = self._strip_balanced_wrapping_quotes(message_text.lstrip(" :：,，"))
        if not message_text:
            return None
        return ExplicitPrivateSendRequest(
            target_user_id=int(target_match.group(0)),
            message_text=message_text,
        )

    def _strip_balanced_wrapping_quotes(self, text: str) -> str:
        normalized = text.strip()
        quote_pairs = (("“", "”"), ('"', '"'), ("'", "'"), ("‘", "’"))
        for left, right in quote_pairs:
            if normalized.startswith(left) and normalized.endswith(right):
                return normalized[len(left) : len(normalized) - len(right)].strip()
        return normalized

    async def _handle_owner_project_private_send_request(
        self,
        *,
        task_id: int,
        private_send: ExplicitPrivateSendRequest,
    ) -> str:
        allowed_targets = {self.owner_qq, *self.private_chat_qqs, *self.admin_qqs}
        if private_send.target_user_id not in allowed_targets:
            return (
                f"QQ {private_send.target_user_id} 现在不在私聊白名单里，这条不能直接发。"
                "先把它加入 PRIVATE_CHAT_QQS，再重试。"
            )

        delivered, failure_reason = await self._deliver_private_text(
            user_id=private_send.target_user_id,
            text=private_send.message_text,
            context=f"private_send:{task_id}:target:{private_send.target_user_id}",
        )
        if not delivered:
            return (
                f"给 {private_send.target_user_id} 的这条私聊没发出去：{failure_reason or 'unknown error'}"
            )
        return (
            f"已经给 {private_send.target_user_id} 私聊发过去了。"
            f"\n内容：{private_send.message_text}"
        )

    def _looks_like_private_send_request(self, text: str) -> bool:
        lowered = text.lower()
        if any(keyword in text or keyword in lowered for keyword in ("权限", "白名单", "allowlist", "whitelist")):
            return False
        has_send_hint = any(
            phrase in text or phrase in lowered
            for phrase in (
                "私聊发送",
                "发送",
                "发消息",
                "发一句",
                "发一条",
                "发个",
                "私信",
                "转达",
                "带句话",
            )
        )
        has_private_hint = any(phrase in text or phrase in lowered for phrase in ("私聊", "私信", "qq"))
        has_target_hint = "给" in text or re.search(r"\d{5,16}", text) is not None
        return has_send_hint and has_private_hint and has_target_hint

    def _build_unsupported_private_action_reply(self, *, request_text: str, private_scope: str) -> str | None:
        if not self._looks_like_private_send_request(request_text):
            return None

        if private_scope == PRIVATE_SCOPE_ALLOWLIST_DAILY:
            return "我不能替你给别人发私聊消息，也不能代你做项目控制。"

        upgrade_hint = (
            "如果你要我开发这个功能，先发“启动管理员模式”，然后直接回我“开发这个功能”就行。"
            if private_scope == PRIVATE_SCOPE_OWNER_DAILY
            else "如果你要我开发这个功能，直接回我“开发这个功能”就行。"
        )
        return (
            "我现在还不能直接替你给别人发私聊消息。当前这条私聊通道只有看仓库、查配置、改项目和重启本体的能力，"
            "没有“代替机器人给指定 QQ 发私聊内容”的执行接口，也没有发送成功回执链路，所以我不能假装已经发出去了。"
            f"{upgrade_hint} 兼容旧用法的话，发“管理员权限 开发这个功能”也可以。"
            "如果你只是想先确认目标 QQ 或私聊权限，我也可以先帮你查。"
        )

    def _build_inline_progress_text(self, *, request_text: str, private_scope: str, intent_type: str) -> str | None:
        del request_text, private_scope, intent_type
        return None

    def _looks_like_capability_upgrade_confirmation(self, text: str) -> bool:
        normalized = self._normalize_private_command_text(text)
        if not normalized:
            return False
        direct_matches = {
            "开发这个功能",
            "开发下这个功能",
            "把这个功能开发出来",
            "开发这个能力",
            "做这个功能",
            "实现这个功能",
            "加上这个功能",
            "加这个功能",
            "把它做出来",
            "把它开发出来",
            "优化出这个功能",
            "继续开发这个功能",
        }
        if normalized in direct_matches:
            return True
        has_build_hint = any(keyword in text for keyword in ("开发", "实现", "做", "加上", "加个", "支持", "优化"))
        has_target_hint = any(keyword in text for keyword in ("功能", "能力", "这个", "它"))
        return has_build_hint and has_target_hint

    def _is_missing_private_send_capability_reply(self, reply_text: str) -> bool:
        normalized = self._normalize_private_reply(reply_text)
        return "不能直接替你给别人发私聊消息" in normalized and "开发这个功能" in normalized

    def _find_recent_owner_task(
        self,
        *,
        owner_qq: int,
        session_modes: list[str],
        predicate: Callable,
        session_limit: int = 4,
        task_limit: int = 4,
    ):
        with session_scope(self.engine) as session:
            sessions = DevSessionRepository(session)
            tasks = DevTaskRepository(session)
            recent_sessions = sessions.list_recent_owner_sessions(
                owner_qq=owner_qq,
                limit=session_limit,
                session_modes=session_modes,
            )
            candidates = []
            for dev_session in recent_sessions:
                candidates.extend(tasks.list_recent_tasks_for_session(session_id=dev_session.id, limit=task_limit))

        candidates.sort(key=lambda task: task.id)
        for task in reversed(candidates):
            if predicate(task):
                return task
        return None

    def _build_capability_upgrade_request_from_recent_context(self, *, owner_qq: int, raw_text: str) -> str | None:
        if not self._looks_like_capability_upgrade_confirmation(raw_text):
            return None

        previous_task = self._find_recent_owner_task(
            owner_qq=owner_qq,
            session_modes=[SESSION_MODE_DAILY, SESSION_MODE_PROJECT],
            predicate=lambda task: (
                task.requested_by_qq == owner_qq
                and task.intent_type == "project_chat"
                and task.status == "completed"
                and self._looks_like_private_send_request(task.raw_request_text)
                and self._is_missing_private_send_capability_reply(task.result_text)
            ),
        )
        if previous_task is None:
            return None

        return self._build_capability_upgrade_request(
            original_request_text=previous_task.raw_request_text,
            confirmation_text=raw_text,
        )

    def _infer_owner_followup_intent_from_offer(self, reply_text: str) -> str | None:
        normalized = self._normalize_private_reply(reply_text)
        if not normalized:
            return None
        has_offer_shape = any(
            phrase in normalized
            for phrase in (
                "如果你要",
                "你要我继续",
                "你要的话",
                "要我继续的话",
                "下一步",
                "继续的话",
                "我可以继续",
            )
        )
        if not has_offer_shape:
            return None
        if any(
            phrase in normalized
            for phrase in (
                "下一步就直接重启",
                "下一步就重启",
                "直接执行重启",
                "先重启",
                "去重启",
                "我现在重启",
                "我下一步就重启",
            )
        ):
            return "restart_only"
        if any(
            phrase in normalized
            for phrase in ("去查", "核对", "确认", "去看", "进仓库", "查配置", "查代码", "查日志", "看代码", "看配置")
        ):
            return "feature_work"
        if any(
            phrase in normalized
            for phrase in ("去改", "帮你改", "实现", "开发", "加进去", "加上", "修掉", "修好")
        ):
            return "feature_work"
        return None

    def _is_feature_followup_confirmation_prompt(self, reply_text: str) -> bool:
        normalized = self._normalize_private_reply(reply_text)
        if not normalized:
            return False
        cue_phrases = (
            "还差最后一个确认",
            "最后一个确认",
            "直接回我一句就行",
            "直接回我就行",
            "你直接回我",
            "二选一",
            "用我写的",
            "你自己提供",
            "按这个",
            "就这样",
        )
        return any(phrase in normalized for phrase in cue_phrases)

    def _build_owner_admin_followup_from_recent_context(
        self,
        *,
        owner_qq: int,
        raw_text: str,
    ) -> tuple[str, str, str] | None:
        if not self._looks_like_owner_admin_continue_confirmation(raw_text):
            return None

        previous_task = self._find_recent_owner_task(
            owner_qq=owner_qq,
            session_modes=[SESSION_MODE_PROJECT],
            predicate=lambda task: task.requested_by_qq == owner_qq and task.status == "completed",
        )
        if previous_task is None:
            return None

        followup_intent = self._infer_owner_followup_intent_from_offer(previous_task.result_text)
        if followup_intent is None:
            if previous_task.intent_type in {FEATURE_PLAN_INTENT, "feature_work"} and self._is_feature_followup_confirmation_prompt(
                previous_task.result_text
            ):
                followup_intent = "feature_work"
            else:
                return None
        followup_text = self._build_owner_followup_request(
            previous_request_text=previous_task.raw_request_text,
            previous_reply_text=previous_task.result_text,
            confirmation_text=raw_text,
            intent_type=followup_intent,
        )
        routing_text = "\n".join(
            [
                previous_task.raw_request_text.strip(),
                previous_task.result_text.strip(),
                raw_text.strip(),
            ]
        )
        return followup_intent, followup_text, routing_text

    def _handle_private_session_command(
        self,
        *,
        raw_text: str,
        user_id: int | None = None,
        session_mode: str,
        session_label: str,
        new_commands: set[str],
        status_commands: set[str],
    ) -> str | None:
        target_user_id = self.owner_qq if user_id is None else user_id
        normalized = self._normalize_private_command_text(raw_text)
        normalized_new_commands = {self._normalize_private_command_text(item) for item in new_commands}
        normalized_status_commands = {self._normalize_private_command_text(item) for item in status_commands}
        if normalized not in normalized_new_commands and normalized not in normalized_status_commands:
            return None

        with session_scope(self.engine) as session:
            sessions = DevSessionRepository(session)
            tasks = DevTaskRepository(session)
            active_session = sessions.get_or_create_owner_session(
                owner_qq=target_user_id,
                session_mode=session_mode,
            )
            open_tasks = tasks.list_tasks_for_session_by_status(
                session_id=active_session.id,
                statuses=["queued", "running"],
            )

            if normalized in normalized_new_commands:
                if open_tasks:
                    return f"现在还有正在处理的{session_label}任务，等它跑完我再给你开新的{session_label}。"
                sessions.create_owner_session(owner_qq=target_user_id, session_mode=session_mode)
                opened_label = "项目会话" if session_mode == SESSION_MODE_PROJECT else "日常会话"
                return f"好，这里给你重新开了一条新的{opened_label}。"

            queued_count = len([task for task in open_tasks if task.status == "queued"])
            running_count = len([task for task in open_tasks if task.status == "running"])
            return (
                f"当前{session_label} #{active_session.id}，排队 {queued_count}，进行中 {running_count}。"
                "要重开就发“清空上下文”或者“开新对话”。"
            )

    def _handle_private_draw_reset_command(
        self,
        *,
        raw_text: str,
        user_id: int | None = None,
    ) -> str | None:
        target_user_id = self.owner_qq if user_id is None else user_id
        normalized = self._normalize_private_command_text(raw_text)
        normalized_commands = {self._normalize_private_command_text(item) for item in PRIVATE_DRAW_RESET_COMMANDS}
        if normalized not in normalized_commands:
            return None
        self._reset_private_draw_state(user_id=target_user_id)
        return "好，绘画上下文已经清空。要重新参考哪张图，直接再发图或重新说。"

    def _build_private_chat_prompt(
        self,
        *,
        session_id: int,
        task_id: int,
        request_text: str,
        request_time,
        private_scope: str,
        image_count: int = 0,
    ) -> list[str]:
        session_summary = self._session_summary(session_id=session_id)
        recent_turns = self._recent_turn_lines(session_id=session_id, exclude_task_id=task_id)
        history_block = "\n".join(recent_turns) if recent_turns else "(none)"
        web_context = self._build_private_web_context(
            request_text=request_text,
            request_time=request_time,
            recent_turns=recent_turns,
        )

        if private_scope == PRIVATE_SCOPE_OWNER_PROJECT:
            repo_snippets = build_repo_context_snippets(repo_root=self.repo_root, query=request_text)
            snippet_block = "\n\n".join(repo_snippets) if repo_snippets else "(none)"
            return [
                "System persona: You are the same assistant style as Codex in this workspace, replying in one persistent private owner conversation about the local Xiaomachi repository.",
                "Safety rules: Stay grounded in the provided session history, repository snippets, runtime facts, and web evidence. Do not invent current file state or reveal secret values.",
                "Admin-mode fact: this private project channel already has repository access through local tooling when needed.",
                "Admin-mode fact: never claim that you lack permission, cannot access the repository, or need the owner to paste project files manually. If more evidence is needed, say you will inspect or modify it through this project channel.",
                *self._private_reply_style_lines(private_scope=private_scope),
                "Current project session summary:",
                session_summary or "(none)",
                "Recent private project turns:",
                history_block,
                *self._private_web_context_lines(web_context),
                *self._private_image_reasoning_lines(
                    request_text=request_text,
                    recent_turns=recent_turns,
                    image_count=image_count,
                ),
                "Relevant repository snippets:",
                snippet_block,
                f"Current owner message: {request_text}",
            ]

        daily_safety_extra = (
            "This user is not the owner: do not offer or imply code changes, admin actions, runtime restarts, or project control."
            if private_scope == PRIVATE_SCOPE_ALLOWLIST_DAILY
            else "If the owner wants repository changes or local runtime inspection, tell them to send “启动管理员模式” first."
        )
        current_message_label = "Current user message:" if private_scope == PRIVATE_SCOPE_ALLOWLIST_DAILY else "Current owner message:"
        return [
            f"System persona: {self._private_daily_persona_text()}",
            self._private_daily_safety_line(extra_rule=daily_safety_extra),
            *self._private_reply_style_lines(private_scope=private_scope),
            "Current private daily session summary:",
            session_summary or "(none)",
            "Recent private daily turns:",
            history_block,
            *self._private_web_context_lines(web_context),
            *self._private_image_reasoning_lines(
                request_text=request_text,
                recent_turns=recent_turns,
                image_count=image_count,
            ),
            f"{current_message_label} {request_text}",
        ]

    def _normalize_private_reply(self, text: str) -> str:
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            return ""

        normalized = normalize_chat_reply(raw).strip()
        return normalized

    def _default_command_runner(self, command: list[str], cwd: Path) -> ScriptRunResult:
        run_kwargs = {
            "cwd": cwd,
            "text": True,
            "encoding": "utf-8",
            "check": False,
        }
        if self._should_detach_command_stdio(command):
            run_kwargs["stdout"] = subprocess.DEVNULL
            run_kwargs["stderr"] = subprocess.DEVNULL
        else:
            run_kwargs["capture_output"] = True

        completed = subprocess.run(command, **run_kwargs)
        return ScriptRunResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )

    def _should_detach_command_stdio(self, command: list[str]) -> bool:
        launch_script_names = {
            "start-xiaomachi-wsl.bat",
        }
        normalized_parts = [str(part).replace("\\", "/").lower() for part in command]
        if any(part.endswith(script_name) for part in normalized_parts for script_name in launch_script_names):
            return True
        command_text = " ".join(normalized_parts)
        return "xiaomachi-wsl-entry.sh" in command_text and "start" in command_text
