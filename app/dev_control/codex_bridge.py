from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


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
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env={**os.environ},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        stdout_chunks, stdout_thread = self._start_stream_reader(getattr(process, "stdout", None))
        stderr_chunks, stderr_thread = self._start_stream_reader(getattr(process, "stderr", None))
        stdout_text = ""
        stderr_text = ""
        parsed_result: CodexTaskResult | None = None
        try:
            if process.stdin is None:
                raise RuntimeError("codex exec did not expose stdin")
            process.stdin.write(prompt)
            process.stdin.close()

            deadline = time.monotonic() + self.timeout_seconds
            while True:
                if process.poll() is not None:
                    stdout_text, stderr_text = self._collect_process_output(
                        process=process,
                        stdout_chunks=stdout_chunks,
                        stdout_thread=stdout_thread,
                        stderr_chunks=stderr_chunks,
                        stderr_thread=stderr_thread,
                    )
                    break

                if parsed_result is None and last_message_path.exists():
                    try:
                        parsed_result = _parse_last_message(last_message_path)
                    except Exception:
                        parsed_result = None
                    else:
                        grace_deadline = min(deadline, time.monotonic() + 2.0)
                        while process.poll() is None and time.monotonic() < grace_deadline:
                            time.sleep(0.1)
                        if process.poll() is None:
                            process.terminate()
                            try:
                                stdout_text, stderr_text = self._collect_process_output(
                                    process=process,
                                    stdout_chunks=stdout_chunks,
                                    stdout_thread=stdout_thread,
                                    stderr_chunks=stderr_chunks,
                                    stderr_thread=stderr_thread,
                                    wait_timeout=5,
                                )
                            except subprocess.TimeoutExpired:
                                process.kill()
                                stdout_text, stderr_text = self._collect_process_output(
                                    process=process,
                                    stdout_chunks=stdout_chunks,
                                    stdout_thread=stdout_thread,
                                    stderr_chunks=stderr_chunks,
                                    stderr_thread=stderr_thread,
                                    wait_timeout=5,
                                )
                            break

                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, self.timeout_seconds)
                time.sleep(0.1)
        except Exception:
            if process.poll() is None:
                process.kill()
                try:
                    stdout_text, stderr_text = self._collect_process_output(
                        process=process,
                        stdout_chunks=stdout_chunks,
                        stdout_thread=stdout_thread,
                        stderr_chunks=stderr_chunks,
                        stderr_thread=stderr_thread,
                        wait_timeout=5,
                    )
                except Exception:
                    stdout_text = stdout_text or "".join(stdout_chunks)
                    stderr_text = stderr_text or "".join(stderr_chunks)
            stdout_path.write_text(stdout_text, encoding="utf-8")
            stderr_path.write_text(stderr_text, encoding="utf-8")
            raise

        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")

        if parsed_result is None:
            if process.returncode != 0:
                raise RuntimeError(
                    f"codex exec failed with exit code {process.returncode}: {stderr_text.strip()}"
                )
            if not last_message_path.exists():
                raise RuntimeError("codex exec did not write a last message")
            parsed_result = _parse_last_message(last_message_path)

        thread_id = resume_thread_id or _extract_thread_id(stdout_text)
        parsed_result.thread_id = thread_id
        return parsed_result

    def _start_stream_reader(self, stream) -> tuple[list[str], threading.Thread | None]:
        if stream is None or not hasattr(stream, "readline"):
            return [], None

        chunks: list[str] = []

        def _reader() -> None:
            try:
                while True:
                    chunk = stream.readline()
                    if chunk == "":
                        break
                    chunks.append(chunk)
            finally:
                try:
                    stream.close()
                except Exception:
                    return

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        return chunks, thread

    def _collect_process_output(
        self,
        *,
        process,
        stdout_chunks: list[str],
        stdout_thread: threading.Thread | None,
        stderr_chunks: list[str],
        stderr_thread: threading.Thread | None,
        wait_timeout: float | None = None,
    ) -> tuple[str, str]:
        if stdout_thread is None and stderr_thread is None:
            out, err = process.communicate(timeout=wait_timeout)
            return out or "", err or ""

        wait = getattr(process, "wait", None)
        if callable(wait):
            wait(timeout=wait_timeout)

        if stdout_thread is not None:
            stdout_thread.join(timeout=wait_timeout)
        if stderr_thread is not None:
            stderr_thread.join(timeout=wait_timeout)
        return "".join(stdout_chunks), "".join(stderr_chunks)

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
