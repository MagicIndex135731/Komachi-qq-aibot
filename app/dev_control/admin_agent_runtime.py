from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.adapters.onebot_models import PrivateMessageEvent
from app.dev_control.repo_context import build_repo_context_snippets
from app.storage.db import session_scope
from app.storage.repositories import DevSessionRepository, DevTaskRepository

ADMIN_AGENT_INTENT = "admin_agent_turn"
ADMIN_AGENT_ACK_TEXT = "我接着在这条管理员会话里处理，完了直接回你结果。"


@dataclass(slots=True)
class AdminAgentWorkItem:
    task_id: int
    session_id: int
    user_id: int
    request_text: str


class AdminAgentRuntime:
    def __init__(self, *, engine, repo_root: Path) -> None:
        self.engine = engine
        self.repo_root = repo_root.resolve()

    def enqueue_turn(self, *, event: PrivateMessageEvent, request_text: str) -> tuple[int, int]:
        normalized_request_text = request_text.strip() or event.plain_text.strip()
        with session_scope(self.engine) as session:
            sessions = DevSessionRepository(session)
            tasks = DevTaskRepository(session)
            dev_session = sessions.get_or_create_owner_session(
                owner_qq=event.user_id,
                session_mode="project",
            )
            task = tasks.add_task(
                session_id=dev_session.id,
                requested_by_qq=event.user_id,
                raw_request_text=normalized_request_text,
                intent_type=ADMIN_AGENT_INTENT,
            )
            sessions.update_session(session_id=dev_session.id, last_task_id=task.id)
            return dev_session.id, task.id

    def claim_next_turn(self) -> AdminAgentWorkItem | None:
        with session_scope(self.engine) as session:
            task = DevTaskRepository(session).claim_oldest_queued_task(intent_types=[ADMIN_AGENT_INTENT])
            if task is None:
                return None
            return AdminAgentWorkItem(
                task_id=task.id,
                session_id=task.session_id,
                user_id=task.requested_by_qq,
                request_text=task.raw_request_text,
            )

    def build_prompt(
        self,
        *,
        session_summary: str,
        recent_turns: list[str],
        request_text: str,
    ) -> str:
        history_block = "\n".join(recent_turns) if recent_turns else "(none)"
        repo_snippets = build_repo_context_snippets(repo_root=self.repo_root, query=request_text)
        snippet_block = "\n\n".join(repo_snippets) if repo_snippets else "(none)"
        return "\n".join(
            [
                "You are operating on the Xiaomachi repository for one persistent private admin session.",
                f"Repository root: {self.repo_root}",
                "Admin-session rules:",
                "- Treat this as the same continuous Codex-style private admin conversation for this QQ admin.",
                "- Default to taking action directly. Do not ask for permission before inspecting files, editing repository code, or running focused verification.",
                "- You may inspect and modify any repository files when needed to finish the request.",
                "- If the request only needs explanation or inspection, do not force code changes.",
                "- Stay inside the repository root.",
                "- Do not reveal secret values or dump .env contents.",
                "- Do not modify unrelated machine files or settings.",
                "- Do not restart the runtime yourself. If a restart is needed after your work, set restart_required=true and explain why.",
                "- Keep the final user-facing reply concise, direct, factual, and natural Chinese.",
                "- In the reply, mention concrete files, commands, and verification results when they matter.",
                "Project session summary:",
                session_summary or "(none)",
                "Recent admin turns:",
                history_block,
                "Relevant repository snippets:",
                snippet_block,
                f"Current admin message: {request_text}",
                'Your final response must be strict JSON with keys "summary", "reply_text", and "restart_required".',
            ]
        )
