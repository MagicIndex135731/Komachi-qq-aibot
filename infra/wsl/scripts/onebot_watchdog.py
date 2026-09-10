import argparse
import asyncio
import hashlib
import ipaddress
import json
import os
import shutil
import signal
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
LLBOT_MAX_RECOVERY_RESTARTS = 2
LLBOT_FORENSICS_FILENAME = "llbot-kick-forensics.jsonl"
LLBOT_FORENSICS_MAX_BYTES = 262_144
LLBOT_FORENSICS_KEEP_LINES = 200
# Several public echo services, tried in order.  api.ipify.org is first for
# most networks but is refused on some Chinese ISP paths, so a single hard-coded
# endpoint would silently disable the egress fingerprint.
FORENSIC_EGRESS_URLS = (
    "https://icanhazip.com",
    "https://ifconfig.me/ip",
    "https://ipinfo.io/ip",
    "https://api.ipify.org",
)
# The watchdog shares one event loop with the 60-second probe cadence, so the
# whole egress lookup is capped instead of costing one timeout per endpoint.
FORENSIC_EGRESS_BUDGET_SECONDS = 12.0

# One read-only probe collects every device/session fact we need without ever
# reading the credential itself.  Only fingerprints and timestamps leave the
# container.
_LLBOT_FORENSICS_NODE_SCRIPT = """
const fs = require('fs');
const crypto = require('crypto');
const out = {
  guid_fingerprint: '',
  guid_mtime: '',
  session_present: false,
  session_saved_at: '',
  signer_set_machine_guid: null,
};
const dataDir = '/app/llbot/data';
try {
  const guidPath = dataDir + '/machine_guid.bin';
  const guid = fs.readFileSync(guidPath);
  out.guid_fingerprint = crypto.createHash('sha256').update(guid).digest('hex').slice(0, 16);
  out.guid_mtime = fs.statSync(guidPath).mtime.toISOString();
} catch (error) {
  out.guid_fingerprint = '';
}
try {
  const sessions = fs
    .readdirSync(dataDir)
    .filter((name) => /^qq-session-[0-9]+\\.json$/.test(name))
    .sort();
  if (sessions.length > 0) {
    out.session_present = true;
    const payload = JSON.parse(fs.readFileSync(dataDir + '/' + sessions[sessions.length - 1], 'utf8'));
    const savedAt = Number(payload.savedAt);
    if (Number.isFinite(savedAt) && savedAt > 0) {
      out.session_saved_at = new Date(savedAt > 1e12 ? savedAt : savedAt * 1000).toISOString();
    }
  }
} catch (error) {
  out.session_present = false;
}
try {
  const candidates = fs
    .readdirSync('/app/llbot')
    .filter((name) => name.startsWith('sign-proxy.') && name.endsWith('.node'))
    .sort();
  const selected =
    candidates.find((name) => name.includes('-' + process.arch + '-')) || candidates[0];
  if (selected) {
    const signer = require('/app/llbot/' + selected);
    out.signer_set_machine_guid = Object.prototype.hasOwnProperty.call(signer, 'setMachineGuid');
  }
} catch (error) {
  out.signer_set_machine_guid = null;
}
process.stdout.write(JSON.stringify(out));
"""


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
    restart_attempts: int = 0
    restart_requested_at: float = 0.0
    llbot_guid_resync_restart_used: bool = False
    llbot_kick_signature: str = ""
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
    max_recovery_restarts: int = 1,
    llbot_guid_resync_required: bool = False,
) -> tuple[WatchdogState, str]:
    unhealthy = online is False or active_session_ok is False
    recovered = online is True and active_session_ok is True
    if recovered:
        # The kick signature survives recovery so a kick line that is still in
        # the log tail cannot produce a duplicate forensic record.
        next_state = WatchdogState(
            webui_alerted=state.webui_alerted if webui_login_error else False,
            llbot_kick_signature=state.llbot_kick_signature,
        )
    elif unhealthy:
        next_state = replace(state, offline_checks=state.offline_checks + 1)
    else:
        next_state = state

    restart_attempts = max(int(next_state.restart_attempts), int(next_state.restart_used))
    if (
        unhealthy
        and llbot_guid_resync_required
        and not next_state.llbot_guid_resync_restart_used
        and restart_attempts < max(1, int(max_recovery_restarts))
    ):
        return (
            replace(
                next_state,
                restart_used=True,
                restart_attempts=restart_attempts + 1,
                restart_requested_at=now,
                llbot_guid_resync_restart_used=True,
            ),
            ACTION_RESTART,
        )
    if (
        unhealthy
        and next_state.offline_checks >= offline_threshold
        and restart_attempts < max(1, int(max_recovery_restarts))
        and (
            restart_attempts == 0
            or now - next_state.restart_requested_at >= recovery_grace_seconds
        )
    ):
        return (
            replace(
                next_state,
                restart_used=True,
                restart_attempts=restart_attempts + 1,
                restart_requested_at=now,
            ),
            ACTION_RESTART,
        )
    if next_state.alerted:
        return next_state, ACTION_NONE
    if webui_login_error and not next_state.webui_alerted:
        return replace(next_state, alerted=True, webui_alerted=True), ACTION_NOTIFY
    if (
        restart_attempts >= max(1, int(max_recovery_restarts))
        and now - next_state.restart_requested_at >= recovery_grace_seconds
    ):
        return replace(next_state, alerted=True), ACTION_NOTIFY
    return next_state, ACTION_NONE


def load_state(path: Path) -> WatchdogState:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        restart_used = bool(payload.get("restart_used", False))
        restart_attempts = int(payload.get("restart_attempts", 1 if restart_used else 0))
        return WatchdogState(
            offline_checks=int(payload.get("offline_checks", 0)),
            restart_used=restart_used or restart_attempts > 0,
            restart_attempts=max(0, restart_attempts),
            restart_requested_at=float(payload.get("restart_requested_at", 0.0)),
            llbot_guid_resync_restart_used=bool(
                payload.get("llbot_guid_resync_restart_used", False)
            ),
            llbot_kick_signature=(
                payload.get("llbot_kick_signature")
                if isinstance(payload.get("llbot_kick_signature"), str)
                else ""
            ),
            alerted=bool(payload.get("alerted", False)),
            webui_alerted=bool(payload.get("webui_alerted", False)),
            isLogin=payload.get("isLogin") if isinstance(payload.get("isLogin"), bool) else None,
            isOffline=payload.get("isOffline") if isinstance(payload.get("isOffline"), bool) else None,
            webui_login_error=bool(payload.get("webui_login_error", False)),
            webui_login_error_kind=(
                payload.get("webui_login_error_kind")
                if payload.get("webui_login_error_kind")
                in {"reported", "llbot_signing_backend_unavailable"}
                else ""
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
    except (
        ConnectionRefusedError,
        OSError,
        asyncio.TimeoutError,
        websockets.WebSocketException,
    ) as exc:
        # A transport that cannot be reached is a real service outage.  Count
        # it toward the debounced restart threshold instead of treating it as
        # an unknowable probe result forever.
        return False, False, type(exc).__name__
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


def restart_service(compose_file: Path, service_name: str) -> tuple[bool, str]:
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "restart", service_name],
        cwd=compose_file.parent,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    detail = (result.stderr or result.stdout or "").strip().replace("\n", " ")
    return result.returncode == 0, detail[:500]


def restart_napcat(compose_file: Path) -> tuple[bool, str]:
    return restart_service(compose_file, "napcat")


def _llbot_recent_logs(
    container_name: str = "xiaomachi-llbot", *, timestamps: bool = False
) -> str:
    """Read the small diagnostic tail without persisting raw gateway logs."""
    command = ["docker", "logs", "--tail", "200"]
    if timestamps:
        # ``docker logs`` prints the application line on its own, so the log
        # driver timestamp is the only stable per-incident identity a repeated
        # QQ kick text has.
        command.append("--timestamps")
    command.append(container_name)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return f"{result.stdout}\n{result.stderr}".lower()


def llbot_signing_backend_unavailable(container_name: str = "xiaomachi-llbot") -> bool:
    """Detect the known LLBot signing outage without persisting raw logs."""
    output = _llbot_recent_logs(container_name)
    return (
        "replay protection unavailable" in output
        or "sign 未初始化" in output
    )


def llbot_guid_resync_required(container_name: str = "xiaomachi-llbot") -> bool:
    """Detect a 1001-kick GUID change that an old native signer cannot absorb."""
    return "setmachineguid" in _llbot_recent_logs(container_name).lower()


def llbot_kick_signature(container_name: str = "xiaomachi-llbot") -> str:
    """Fingerprint the newest QQ 1001 kick line without persisting it.

    The kick line carries no credentials, but the watchdog contract forbids
    storing raw gateway logs.  A hash over the newest matching line is stable
    for a given incident and changes as soon as QQ reports another kick.
    QQ reuses one fixed kick text, so the docker log timestamp is part of the
    hashed line; without it two kicks days apart collapse into one signature
    and the later incident is dropped as a duplicate.
    """
    lines = [
        line.strip()
        for line in _llbot_recent_logs(container_name, timestamps=True).splitlines()
        if "code=1001" in line
    ]
    if not lines:
        return ""
    return hashlib.sha256(lines[-1].encode("utf-8")).hexdigest()[:16]


def _forensic_salt(salt_file: Path) -> bytes:
    """Return a per-installation salt so egress fingerprints stay private."""
    try:
        salt = salt_file.read_bytes()
        if len(salt) >= 16:
            return salt
    except OSError:
        pass
    salt = os.urandom(32)
    salt_file.parent.mkdir(parents=True, exist_ok=True)
    salt_file.write_bytes(salt)
    try:
        salt_file.chmod(0o600)
    except OSError:
        pass
    return salt


def _forensic_container_info(container_name: str) -> dict[str, Any]:
    """Read container identity plus the proxy class, never the proxy value."""
    info: dict[str, Any] = {
        "id_prefix": "",
        "started_at": "",
        "restart_count": None,
        "image": "",
        "proxy_class": "unknown",
    }
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                container_name,
                "--format",
                "{{.Id}}|{{.State.StartedAt}}|{{.RestartCount}}|{{.Config.Image}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return info
    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split("|")
        if len(parts) >= 4:
            info["id_prefix"] = parts[0][:12]
            info["started_at"] = parts[1]
            info["restart_count"] = int(parts[2]) if parts[2].isdigit() else None
            info["image"] = parts[3]
    try:
        env_result = subprocess.run(
            ["docker", "inspect", container_name, "--format", "{{json .Config.Env}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if env_result.returncode == 0 and env_result.stdout.strip():
            entries = json.loads(env_result.stdout)
            if isinstance(entries, list):
                info["proxy_class"] = _proxy_class(entries)
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return info
    return info


def _proxy_class(env_entries: list[Any]) -> str:
    """Classify QQ egress routing without persisting any proxy address."""
    values: dict[str, str] = {}
    for entry in env_entries:
        if isinstance(entry, str) and "=" in entry:
            key, _, value = entry.partition("=")
            values[key.strip()] = value.strip()
    configured = [values.get("HTTP_PROXY", ""), values.get("HTTPS_PROXY", "")]
    if not any(configured):
        return "direct"
    for value in configured:
        if not value:
            continue
        if urlsplit(value).hostname not in {"127.0.0.1", "localhost", "::1"}:
            return "remote"
    return "loopback"


def _forensic_device_info(container_name: str) -> dict[str, Any]:
    """Probe device/session fingerprints and signer capability read-only."""
    info: dict[str, Any] = {
        "guid_fingerprint": "",
        "guid_mtime": "",
        "session_present": None,
        "session_saved_at": "",
        "signer_set_machine_guid": None,
    }
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "node", "-e", _LLBOT_FORENSICS_NODE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return info
    if result.returncode != 0 or not result.stdout.strip():
        return info
    try:
        payload = json.loads(result.stdout)
    except (ValueError, json.JSONDecodeError):
        return info
    if not isinstance(payload, dict):
        return info
    for key in info:
        if key in payload:
            info[key] = payload[key]
    return info


def _forensic_egress_fingerprint(salt_file: Path) -> dict[str, str]:
    """Fingerprint the public egress address with a per-installation salt."""
    salt = _forensic_salt(salt_file)
    address = ""
    deadline = time.monotonic() + FORENSIC_EGRESS_BUDGET_SECONDS
    for url in FORENSIC_EGRESS_URLS:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            with _open_without_redirects(Request(url), timeout=min(8.0, remaining)) as response:
                candidate = response.read().decode("utf-8", "replace").strip()
        except Exception:
            continue
        if _looks_like_address(candidate):
            address = candidate
            break
    if not address:
        return {"fingerprint": "", "probe": "unavailable"}
    digest = hashlib.sha256(salt + b"\0" + address.encode("utf-8")).hexdigest()
    return {"fingerprint": digest[:16], "probe": "ok"}


def _looks_like_address(value: str) -> bool:
    """Accept only a bare IP literal so no provider text is ever hashed."""
    if not value or len(value) > 45:
        return False
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def build_llbot_kick_forensics(
    *,
    container_name: str = "xiaomachi-llbot",
    kick_signature: str = "",
    state: WatchdogState | None = None,
    forensics_file: Path = Path("llbot-kick-forensics.jsonl"),
    now: float | None = None,
    container_probe: Callable[..., dict[str, Any]] | None = None,
    device_probe: Callable[..., dict[str, Any]] | None = None,
    egress_probe: Callable[..., dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Compose one credential-free snapshot for a QQ 1001 kick incident."""
    timestamp = time.time() if now is None else now
    salt_file = forensics_file.parent / "llbot-forensics-salt.bin"
    container = (container_probe or _forensic_container_info)(container_name)
    device = (device_probe or _forensic_device_info)(container_name)
    egress = (egress_probe or _forensic_egress_fingerprint)(salt_file)
    return {
        "event": "llbot_kick_1001",
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(timestamp)),
        "observed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
        "kick_signature": kick_signature,
        "container": container,
        "device": {
            "guid_fingerprint": device.get("guid_fingerprint", ""),
            "guid_mtime": device.get("guid_mtime", ""),
            "signer_set_machine_guid": device.get("signer_set_machine_guid"),
        },
        "session": {
            "present": device.get("session_present"),
            "saved_at": device.get("session_saved_at", ""),
        },
        "egress": egress,
        "watchdog": {
            "offline_checks": getattr(state, "offline_checks", 0),
            "restart_attempts": getattr(state, "restart_attempts", 0),
            "guid_resync_restart_used": getattr(state, "llbot_guid_resync_restart_used", False),
        },
    }


def append_jsonl_bounded(
    path: Path,
    payload: dict[str, Any],
    *,
    max_bytes: int = LLBOT_FORENSICS_MAX_BYTES,
    keep_lines: int = LLBOT_FORENSICS_KEEP_LINES,
) -> None:
    """Append one record and keep the forensic file strictly bounded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
    try:
        if path.stat().st_size <= max_bytes:
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > keep_lines:
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text("\n".join(lines[-keep_lines:]) + "\n", encoding="utf-8")
            temporary.replace(path)
    except OSError:
        return


def _windows_powershell_path() -> str | None:
    discovered = shutil.which("powershell.exe")
    if discovered:
        return discovered
    # systemd's default PATH excludes Windows interop directories even though
    # WSL interop itself is available.
    fallback = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    return fallback if Path(fallback).is_file() else None


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
    powershell = _windows_powershell_path()
    if powershell is None:
        return False, "powershell_not_found"
    try:
        subprocess.Popen(
            [
                powershell,
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


async def run_check(
    *,
    ws_url: str,
    state_file: Path,
    log_file: Path,
    compose_file: Path,
    notifier: Path,
    service_name: str = "napcat",
    platform: str = "napcat",
    webui_config: Path | None = None,
    webui_url: str = "http://127.0.0.1:6099",
    forensics_file: Path | None = None,
) -> int:
    state = load_state(state_file)
    online, active_session_ok, probe_detail = await probe_onebot(ws_url)
    kick_signature = llbot_kick_signature() if platform == "llbot" else ""
    if kick_signature and kick_signature != state.llbot_kick_signature:
        record_path = forensics_file or (log_file.parent / LLBOT_FORENSICS_FILENAME)
        try:
            # Container inspection and the egress lookup block on subprocess
            # and network I/O; keep the shared probe cadence responsive.
            record = await asyncio.to_thread(
                build_llbot_kick_forensics,
                kick_signature=kick_signature,
                state=state,
                forensics_file=record_path,
            )
            append_jsonl_bounded(record_path, record)
            append_log(
                log_file,
                "llbot_kick_1001_recorded",
                f"signature={kick_signature}",
            )
        except Exception as exc:
            # Forensics must never change restart or notification behaviour.
            append_log(log_file, "llbot_kick_1001_record_failed", type(exc).__name__)
    webui_status = (
        probe_webui(
            webui_config or compose_file.parent / "runtime/napcat/config/webui.json", webui_url
        )
        if platform == "napcat"
        else {"isLogin": None, "isOffline": None, "loginError": ""}
    )
    llbot_signing_error = (
        platform == "llbot"
        and online is not True
        and llbot_signing_backend_unavailable()
    )
    llbot_guid_resync_needed = (
        platform == "llbot"
        and online is not True
        and not llbot_signing_error
        and llbot_guid_resync_required()
    )
    webui_login_error = is_explicit_webui_login_error(webui_status) or llbot_signing_error
    # Restarting LLBot cannot repair an unavailable external signing service.
    # Treat it as an explicit login error so the user is notified immediately,
    # while leaving the one controlled restart available for a real session fault.
    evaluated_online = None if llbot_signing_error else online
    evaluated_active_session = None if llbot_signing_error else active_session_ok
    next_state, action = evaluate_state(
        state,
        online=evaluated_online,
        active_session_ok=evaluated_active_session,
        webui_login_error=webui_login_error,
        now=time.time(),
        max_recovery_restarts=(LLBOT_MAX_RECOVERY_RESTARTS if platform == "llbot" else 1),
        llbot_guid_resync_required=llbot_guid_resync_needed,
    )
    next_state = replace(
        next_state,
        isLogin=webui_status["isLogin"],
        isOffline=webui_status["isOffline"],
        llbot_kick_signature=kick_signature or next_state.llbot_kick_signature,
        webui_login_error=webui_login_error,
        webui_login_error_kind=(
            "llbot_signing_backend_unavailable"
            if llbot_signing_error
            else "reported" if webui_login_error else ""
        ),
    )
    save_state(state_file, next_state)

    probe_event = "probe_online" if online is True else "probe_offline" if online is False else "probe_unknown"
    append_log(log_file, probe_event, probe_detail)

    if action == ACTION_RESTART:
        ok, detail = (
            restart_napcat(compose_file)
            if service_name == "napcat"
            else restart_service(compose_file, service_name)
        )
        restart_event = (
            f"{service_name}_guid_resync_restart_requested"
            if llbot_guid_resync_needed
            else f"{service_name}_restart_requested"
        )
        failed_event = (
            f"{service_name}_guid_resync_restart_failed"
            if llbot_guid_resync_needed
            else f"{service_name}_restart_failed"
        )
        append_log(log_file, restart_event if ok else failed_event, detail)
        if not ok:
            failed_state = replace(next_state, alerted=True)
            save_state(state_file, failed_state)
            notified, notify_detail = notify_windows(notifier, "qq_platform_restart_failed")
            append_log(log_file, "windows_alert_started" if notified else "windows_alert_failed", notify_detail)
            if not notified:
                save_state(state_file, replace(failed_state, alerted=False))
    elif action == ACTION_NOTIFY:
        if llbot_signing_error:
            reason = "llbot_signing_backend_unavailable"
        elif webui_login_error and not state.webui_alerted and next_state.webui_alerted:
            reason = "webui_login_error"
        else:
            reason = "onebot_session_unhealthy"
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


def run_once(**kwargs: Any) -> int:
    """Run one check for manual diagnostics and tests.

    The normal service path uses ``run_daemon`` so all periodic checks share a
    single asyncio event loop.  Keeping this small wrapper makes ``--once`` a
    useful, deterministic operational command.
    """
    return asyncio.run(run_check(**kwargs))


async def _wait_for_stop(stop_event: asyncio.Event, interval: float) -> bool:
    """Wait for either a shutdown signal or the next scheduled check."""
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=interval)
        return True
    except asyncio.TimeoutError:
        return False


async def run_daemon(*, interval: float, **kwargs: Any) -> int:
    """Run checks in one long-lived process and one asyncio event loop.

    The first probe deliberately waits one full interval: platform containers
    need time to finish their own startup/login work, and an immediate probe
    would create false incidents during normal boot.
    """
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        stop_event.set()

    installed_signals: list[signal.Signals] = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(signum, request_stop)
            installed_signals.append(signum)
        except (NotImplementedError, RuntimeError):
            # ``--daemon`` normally runs in WSL/Linux.  This fallback keeps
            # the function testable on platforms without asyncio signal hooks.
            signal.signal(signum, lambda *_: request_stop())

    append_log(kwargs["log_file"], "watchdog_daemon_started", f"interval={interval:g}")
    try:
        while not await _wait_for_stop(stop_event, interval):
            try:
                await run_check(**kwargs)
            except Exception as exc:
                # A transient probe implementation failure must not make the
                # supervisor restart the daemon or reset its persistent state.
                append_log(kwargs["log_file"], "watchdog_check_failed", type(exc).__name__)
    finally:
        for signum in installed_signals:
            loop.remove_signal_handler(signum)
        append_log(kwargs["log_file"], "watchdog_daemon_stopped")
    return 0


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    wsl_dir = script_dir.parent
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Run one watchdog check.")
    mode.add_argument("--daemon", action="store_true", help="Run checks in one long-lived process.")
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between daemon checks, including the initial startup delay (default: 60).",
    )
    parser.add_argument("--ws-url", default="ws://127.0.0.1:3001")
    parser.add_argument("--state-file", type=Path, default=wsl_dir / "runtime/onebot-watchdog.json")
    parser.add_argument("--log-file", type=Path, default=wsl_dir / "runtime/logs/onebot-watchdog.log")
    parser.add_argument("--compose-file", type=Path, default=wsl_dir / "docker-compose.yml")
    parser.add_argument("--service-name", default="napcat")
    parser.add_argument("--platform", choices=("napcat", "llbot"), default="napcat")
    parser.add_argument("--notifier", type=Path, default=script_dir / "notify_windows.ps1")
    parser.add_argument("--webui-config", type=Path, default=wsl_dir / "runtime/napcat/config/webui.json")
    parser.add_argument("--webui-url", default="http://127.0.0.1:6099")
    parser.add_argument(
        "--forensics-file",
        type=Path,
        default=wsl_dir / "runtime/logs" / LLBOT_FORENSICS_FILENAME,
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    lock_file = args.state_file.with_suffix(args.state_file.suffix + ".lock")
    with exclusive_lock(lock_file) as acquired:
        if not acquired:
            append_log(args.log_file, "watchdog_run_skipped", "lock_busy")
            return 0
        run_kwargs = {
            "ws_url": args.ws_url,
            "state_file": args.state_file,
            "log_file": args.log_file,
            "compose_file": args.compose_file,
            "notifier": args.notifier,
            "service_name": args.service_name,
            "platform": args.platform,
            "webui_config": args.webui_config,
            "webui_url": args.webui_url,
            "forensics_file": args.forensics_file,
        }
        if args.daemon:
            return asyncio.run(run_daemon(interval=args.interval, **run_kwargs))
        return run_once(**run_kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
