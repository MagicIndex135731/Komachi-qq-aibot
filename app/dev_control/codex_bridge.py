from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess


@dataclass(slots=True)
class CodexTaskResult:
    summary: str
    reply_text: str
    restart_required: bool
    raw_last_message: str = ""
    thread_id: str = ""


def _parse_last_message(path: Path, *, thread_id: str = "") -> CodexTaskResult:
    raw_last_message = path.read_text(encoding="utf-8")
    payload = json.loads(raw_last_message)
    return CodexTaskResult(
        summary=str(payload["summary"]),
        reply_text=str(payload["reply_text"]),
        restart_required=bool(payload["restart_required"]),
        raw_last_message=raw_last_message,
        thread_id=thread_id,
    )


def _extract_thread_id(stdout_text: str) -> str:
    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("type") != "thread.started":
            continue
        thread_id = payload.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            return thread_id
    return ""


class CodexBridge:
    def __init__(self, *, timeout_seconds: int = 1800) -> None:
        self.timeout_seconds = timeout_seconds
        self.codex_executable = self._resolve_codex_executable()

    def _fallback_candidates(self) -> list[Path]:
        home = Path.home()
        candidates: list[Path] = []
        explicit = os.environ.get("CODEX_EXECUTABLE", "").strip()
        if explicit:
            candidates.append(Path(explicit))
        candidates.extend(
            sorted(
                home.glob(".vscode/extensions/openai.chatgpt-*/bin/windows-x86_64/codex.exe"),
                reverse=True,
            )
        )
        return candidates

    def _resolve_codex_executable(self) -> str:
        candidate = shutil.which("codex.exe") or shutil.which("codex")
        if candidate is not None:
            return candidate
        for fallback in self._fallback_candidates():
            if fallback.exists():
                return str(fallback)
        raise FileNotFoundError("codex executable not found on PATH")

    def run_task(
        self,
        *,
        prompt: str,
        repo_root: Path,
        artifact_dir: Path,
        resume_thread_id: str | None = None,
    ) -> CodexTaskResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = artifact_dir / "codex.stdout.log"
        stderr_path = artifact_dir / "codex.stderr.log"
        last_message_path = artifact_dir / "codex.last_message.json"

        command = self._build_command(
            last_message_path=last_message_path,
            resume_thread_id=resume_thread_id,
        )
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env={**os.environ},
            capture_output=True,
            text=True,
            encoding="utf-8",
            input=prompt,
            timeout=self.timeout_seconds,
            check=False,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"codex exec failed with exit code {completed.returncode}: {completed.stderr.strip()}"
            )
        if not last_message_path.exists():
            raise RuntimeError("codex exec did not write a last message")

        thread_id = resume_thread_id or _extract_thread_id(completed.stdout)
        return _parse_last_message(last_message_path, thread_id=thread_id)

    def _build_command(self, *, last_message_path: Path, resume_thread_id: str | None) -> list[str]:
        base = [
            self.codex_executable,
            "exec",
        ]
        if resume_thread_id:
            return base + [
                "resume",
                "--skip-git-repo-check",
                "--ignore-rules",
                "--config",
                'model_reasoning_effort="medium"',
                "--json",
                "--output-last-message",
                str(last_message_path),
                resume_thread_id,
                "-",
            ]

        return base + [
            "--skip-git-repo-check",
            "--ignore-rules",
            "--config",
            'model_reasoning_effort="medium"',
            "--color",
            "never",
            "--json",
            "--output-last-message",
            str(last_message_path),
            "-",
        ]
