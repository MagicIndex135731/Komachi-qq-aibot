import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import websockets


ACTION_NONE = "none"
ACTION_RESTART = "restart"
ACTION_NOTIFY = "notify"
OFFLINE_THRESHOLD = 3
RECOVERY_GRACE_SECONDS = 120


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _open_without_redirects(request: Request, timeout: float) -> Any:
    return build_opener(_NoRedirectHandler()).open(request, timeout=timeout)


def _is_allowed_webui_url(webui_url: str) -> bool:
    try:
        parsed = urlsplit(webui_url)
        return (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            and parsed.port == 6099
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


@dataclass
class WatchdogState:
    offline_checks: int = 0
    restart_used: bool = False
    restart_requested_at: float = 0.0
    alerted: bool = False
    webui_alerted: bool = False
    isLogin: bool | None = None
    isOffline: bool | None = None
    webui_login_error: bool = False
    webui_login_error_kind: str = ""


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError:
                pass
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
        yield acquired
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def evaluate_state(
    state: WatchdogState,
    *,
    online: bool | None,
    active_session_ok: bool | None = True,
    webui_login_error: bool = False,
    now: float,
    offline_threshold: int = OFFLINE_THRESHOLD,
    recovery_grace_seconds: int = RECOVERY_GRACE_SECONDS,
) -> tuple[WatchdogState, str]:
    unhealthy = online is False or active_session_ok is False
    recovered = online is True and active_session_ok is True
    if recovered:
        next_state = WatchdogState(webui_alerted=state.webui_alerted if webui_login_error else False)
    elif unhealthy:
        next_state = replace(state, offline_checks=state.offline_checks + 1)
    else:
        next_state = state

    if (
        unhealthy
        and not next_state.restart_used
        and next_state.offline_checks >= offline_threshold
    ):
        return (
            replace(next_state, restart_used=True, restart_requested_at=now),
            ACTION_RESTART,
        )
    if next_state.alerted:
        return next_state, ACTION_NONE
    if webui_login_error and not next_state.webui_alerted:
        return replace(next_state, alerted=True, webui_alerted=True), ACTION_NOTIFY
    if (
        next_state.restart_used
        and now - next_state.restart_requested_at >= recovery_grace_seconds
    ):
        return replace(next_state, alerted=True), ACTION_NOTIFY
    return next_state, ACTION_NONE


def load_state(path: Path) -> WatchdogState:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return WatchdogState(
            offline_checks=int(payload.get("offline_checks", 0)),
            restart_used=bool(payload.get("restart_used", False)),
            restart_requested_at=float(payload.get("restart_requested_at", 0.0)),
            alerted=bool(payload.get("alerted", False)),
            webui_alerted=bool(payload.get("webui_alerted", False)),
            isLogin=payload.get("isLogin") if isinstance(payload.get("isLogin"), bool) else None,
            isOffline=payload.get("isOffline") if isinstance(payload.get("isOffline"), bool) else None,
            webui_login_error=bool(payload.get("webui_login_error", False)),
            webui_login_error_kind=(
                "reported" if payload.get("webui_login_error_kind") == "reported" else ""
            ),
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return WatchdogState()


def save_state(path: Path, state: WatchdogState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(state), sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def append_log(path: Path, event: str, detail: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    suffix = f" detail={detail}" if detail else ""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} event={event}{suffix}\n")


async def _onebot_call(
    ws: Any, action: str, params: dict[str, Any], timeout: float = 8
) -> dict[str, Any]:
    echo = f"watchdog_{action}"
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    remaining = deadline - loop.time()
    if remaining <= 0:
        raise asyncio.TimeoutError()
    await asyncio.wait_for(
        ws.send(json.dumps({"action": action, "params": params, "echo": echo})),
        timeout=remaining,
    )
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise asyncio.TimeoutError()
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        payload: dict[str, Any] = json.loads(raw)
        if payload.get("echo") == echo:
            return payload


async def probe_onebot(ws_url: str) -> tuple[bool | None, bool | None, str]:
    try:
        async with websockets.connect(ws_url, open_timeout=8, close_timeout=3) as ws:
            status = await _onebot_call(ws, "get_status", {})
            if status.get("status") != "ok":
                return None, None, "get_status_not_ok"
            online = bool(status.get("data", {}).get("online"))
            try:
                group_list = await _onebot_call(ws, "get_group_list", {"no_cache": True})
            except Exception as exc:
                return online, False, f"get_group_list_{type(exc).__name__}"
            if group_list.get("status") != "ok":
                return online, False, "get_group_list_not_ok"
            return online, True, "get_group_list"
    except Exception as exc:
        return None, None, type(exc).__name__


def _webui_status_payload(payload: Any) -> dict[str, bool | None | str]:
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        data = {}
    return {
        "isLogin": data.get("isLogin") if isinstance(data.get("isLogin"), bool) else None,
        "isOffline": data.get("isOffline") if isinstance(data.get("isOffline"), bool) else None,
        "loginError": "reported" if data.get("loginError") else "",
    }


def probe_webui(
    config_path: Path,
    webui_url: str,
    *,
    opener: Callable[..., Any] | None = None,
) -> dict[str, bool | None | str]:
    unknown: dict[str, bool | None | str] = {
        "isLogin": None,
        "isOffline": None,
        "loginError": "",
    }
    if not _is_allowed_webui_url(webui_url):
        return unknown
    try:
        request_opener = opener or _open_without_redirects
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        token = payload.get("token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            return unknown
        login_body = json.dumps(
            {"hash": hashlib.sha256((token + ".napcat").encode("utf-8")).hexdigest()}
        ).encode("utf-8")
        login_request = Request(
            webui_url.rstrip("/") + "/api/auth/login",
            data=login_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request_opener(login_request, timeout=8) as response:
            login_response = json.loads(response.read().decode("utf-8"))
        login_data = login_response.get("data", {}) if isinstance(login_response, dict) else {}
        credential = login_data.get("Credential") if isinstance(login_data, dict) else None
        if not isinstance(credential, str) or not credential:
            return unknown
        status_request = Request(
            webui_url.rstrip("/") + "/api/QQLogin/CheckLoginStatus",
            data=b"{}",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {credential}"},
            method="POST",
        )
        with request_opener(status_request, timeout=8) as response:
            return _webui_status_payload(json.loads(response.read().decode("utf-8")))
    except Exception:
        return unknown


def is_explicit_webui_login_error(status: dict[str, bool | None | str]) -> bool:
    return status["isLogin"] is False and bool(status["loginError"])


def restart_napcat(compose_file: Path) -> tuple[bool, str]:
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "restart", "napcat"],
        cwd=compose_file.parent,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    detail = (result.stderr or result.stdout or "").strip().replace("\n", " ")
    return result.returncode == 0, detail[:500]


def notify_windows(script_path: Path, reason: str) -> tuple[bool, str]:
    converted = subprocess.run(
        ["wslpath", "-w", str(script_path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if converted.returncode != 0 or not converted.stdout.strip():
        return False, "wslpath_failed"
    try:
        subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-STA",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                converted.stdout.strip(),
                "-Reason",
                reason,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return False, type(exc).__name__
    return True, "started"


def run_once(
    *,
    ws_url: str,
    state_file: Path,
    log_file: Path,
    compose_file: Path,
    notifier: Path,
    webui_config: Path | None = None,
    webui_url: str = "http://127.0.0.1:6099",
) -> int:
    state = load_state(state_file)
    online, active_session_ok, probe_detail = asyncio.run(probe_onebot(ws_url))
    webui_status = probe_webui(
        webui_config or compose_file.parent / "runtime/napcat/config/webui.json", webui_url
    )
    webui_login_error = is_explicit_webui_login_error(webui_status)
    next_state, action = evaluate_state(
        state,
        online=online,
        active_session_ok=active_session_ok,
        webui_login_error=webui_login_error,
        now=time.time(),
    )
    next_state = replace(
        next_state,
        isLogin=webui_status["isLogin"],
        isOffline=webui_status["isOffline"],
        webui_login_error=webui_login_error,
        webui_login_error_kind="reported" if webui_login_error else "",
    )
    save_state(state_file, next_state)

    probe_event = "probe_online" if online is True else "probe_offline" if online is False else "probe_unknown"
    append_log(log_file, probe_event, probe_detail)

    if action == ACTION_RESTART:
        ok, detail = restart_napcat(compose_file)
        append_log(log_file, "napcat_restart_requested" if ok else "napcat_restart_failed", detail)
        if not ok:
            failed_state = replace(next_state, alerted=True)
            save_state(state_file, failed_state)
            notified, notify_detail = notify_windows(notifier, "napcat_restart_failed")
            append_log(log_file, "windows_alert_started" if notified else "windows_alert_failed", notify_detail)
            if not notified:
                save_state(state_file, replace(failed_state, alerted=False))
    elif action == ACTION_NOTIFY:
        reason = (
            "webui_login_error"
            if webui_login_error and not state.webui_alerted and next_state.webui_alerted
            else "onebot_session_unhealthy"
        )
        notified, detail = notify_windows(notifier, reason)
        append_log(log_file, "windows_alert_started" if notified else "windows_alert_failed", detail)
        if not notified:
            save_state(
                state_file,
                replace(
                    next_state,
                    alerted=False,
                    webui_alerted=False if reason == "webui_login_error" else next_state.webui_alerted,
                ),
            )
    return 0


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    wsl_dir = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one watchdog check.")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:3001")
    parser.add_argument("--state-file", type=Path, default=wsl_dir / "runtime/onebot-watchdog.json")
    parser.add_argument("--log-file", type=Path, default=wsl_dir / "runtime/logs/onebot-watchdog.log")
    parser.add_argument("--compose-file", type=Path, default=wsl_dir / "docker-compose.yml")
    parser.add_argument("--notifier", type=Path, default=script_dir / "notify_windows.ps1")
    parser.add_argument("--webui-config", type=Path, default=wsl_dir / "runtime/napcat/config/webui.json")
    parser.add_argument("--webui-url", default="http://127.0.0.1:6099")
    args = parser.parse_args()
    if not args.once:
        parser.error("--once is required; keepalive.sh owns the scheduling loop")
    lock_file = args.state_file.with_suffix(args.state_file.suffix + ".lock")
    with exclusive_lock(lock_file) as acquired:
        if not acquired:
            append_log(args.log_file, "watchdog_run_skipped", "lock_busy")
            return 0
        return run_once(
            ws_url=args.ws_url,
            state_file=args.state_file,
            log_file=args.log_file,
            compose_file=args.compose_file,
            notifier=args.notifier,
            webui_config=args.webui_config,
            webui_url=args.webui_url,
        )


if __name__ == "__main__":
    raise SystemExit(main())
