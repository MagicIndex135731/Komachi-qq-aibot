from __future__ import annotations

import importlib.util
import asyncio
import hashlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.request import Request

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WATCHDOG_PATH = REPO_ROOT / "infra/wsl/scripts/onebot_watchdog.py"


def load_watchdog() -> ModuleType:
    spec = importlib.util.spec_from_file_location("onebot_watchdog", WATCHDOG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_three_consecutive_offline_checks_request_one_restart() -> None:
    watchdog = load_watchdog()
    state = watchdog.WatchdogState()

    state, action = watchdog.evaluate_state(state, online=False, now=10.0)
    assert action == watchdog.ACTION_NONE
    state, action = watchdog.evaluate_state(state, online=False, now=20.0)
    assert action == watchdog.ACTION_NONE
    state, action = watchdog.evaluate_state(state, online=False, now=30.0)

    assert action == watchdog.ACTION_RESTART
    assert state.restart_used is True
    assert state.restart_requested_at == 30.0


def test_offline_after_restart_grace_notifies_once_without_restart_loop() -> None:
    watchdog = load_watchdog()
    state = watchdog.WatchdogState(
        offline_checks=3,
        restart_used=True,
        restart_requested_at=30.0,
    )

    state, action = watchdog.evaluate_state(state, online=False, now=149.0)
    assert action == watchdog.ACTION_NONE
    state, action = watchdog.evaluate_state(state, online=False, now=150.0)
    assert action == watchdog.ACTION_NOTIFY
    assert state.alerted is True

    state, action = watchdog.evaluate_state(state, online=False, now=600.0)
    assert action == watchdog.ACTION_NONE
    assert state.restart_used is True


def test_online_probe_resets_incident_state() -> None:
    watchdog = load_watchdog()
    incident = watchdog.WatchdogState(
        offline_checks=8,
        restart_used=True,
        restart_requested_at=30.0,
        alerted=True,
    )

    state, action = watchdog.evaluate_state(incident, online=True, now=900.0)

    assert action == watchdog.ACTION_NONE
    assert state == watchdog.WatchdogState()


def test_unknown_probe_result_does_not_count_as_account_offline() -> None:
    watchdog = load_watchdog()
    state = watchdog.WatchdogState(offline_checks=2)

    state, action = watchdog.evaluate_state(state, online=None, now=30.0)

    assert action == watchdog.ACTION_NONE
    assert state.offline_checks == 2
    assert state.restart_used is False


def test_unknown_probe_after_restart_grace_notifies_once() -> None:
    watchdog = load_watchdog()
    state = watchdog.WatchdogState(
        offline_checks=3,
        restart_used=True,
        restart_requested_at=30.0,
    )

    state, action = watchdog.evaluate_state(state, online=None, now=150.0)

    assert action == watchdog.ACTION_NOTIFY
    assert state.alerted is True


def test_active_session_failure_requests_one_restart_after_three_checks() -> None:
    watchdog = load_watchdog()
    state = watchdog.WatchdogState()

    for now in (10.0, 20.0):
        state, action = watchdog.evaluate_state(
            state, online=True, active_session_ok=False, now=now
        )
        assert action == watchdog.ACTION_NONE

    state, action = watchdog.evaluate_state(
        state, online=True, active_session_ok=False, now=30.0
    )

    assert action == watchdog.ACTION_RESTART
    assert state.restart_used is True


def test_unknown_active_session_does_not_increment_offline_counter() -> None:
    watchdog = load_watchdog()
    state = watchdog.WatchdogState(offline_checks=2)

    state, action = watchdog.evaluate_state(
        state, online=True, active_session_ok=None, now=30.0
    )

    assert action == watchdog.ACTION_NONE
    assert state.offline_checks == 2


def test_group_list_failure_after_get_status_counts_as_active_session_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watchdog = load_watchdog()

    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []

        async def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))

        async def recv(self) -> str:
            if len(self.sent) == 1:
                return json.dumps({"status": "ok", "data": {"online": True}, "echo": "watchdog_get_status"})
            raise TimeoutError("local fake timeout")

    class FakeConnection:
        async def __aenter__(self) -> FakeWebSocket:
            return fake_ws

        async def __aexit__(self, *args: object) -> None:
            return None

    fake_ws = FakeWebSocket()
    monkeypatch.setattr(watchdog.websockets, "connect", lambda *args, **kwargs: FakeConnection())

    online, active_session_ok, _ = asyncio.run(watchdog.probe_onebot("ws://fake"))

    assert (online, active_session_ok) == (True, False)


def test_explicit_webui_login_error_notifies_once_without_sensitive_state() -> None:
    watchdog = load_watchdog()
    state = watchdog.WatchdogState()

    state, action = watchdog.evaluate_state(
        state, online=None, webui_login_error=True, now=30.0
    )

    assert action == watchdog.ACTION_NOTIFY
    assert state.webui_alerted is True
    assert state.alerted is True
    assert "token" not in json.dumps(watchdog.asdict(state)).lower()
    state, action = watchdog.evaluate_state(
        state, online=None, webui_login_error=True, now=31.0
    )
    assert action == watchdog.ACTION_NONE


def test_webui_alerted_incident_can_restart_but_never_notifies_again_after_grace() -> None:
    watchdog = load_watchdog()
    state = watchdog.WatchdogState()

    state, action = watchdog.evaluate_state(
        state, online=None, webui_login_error=True, now=1.0
    )
    assert action == watchdog.ACTION_NOTIFY
    assert state.alerted is True

    for now in (2.0, 3.0):
        state, action = watchdog.evaluate_state(
            state, online=True, active_session_ok=False, now=now
        )
        assert action == watchdog.ACTION_NONE
    state, action = watchdog.evaluate_state(
        state, online=True, active_session_ok=False, now=4.0
    )
    assert action == watchdog.ACTION_RESTART

    state, action = watchdog.evaluate_state(
        state, online=False, active_session_ok=False, now=124.0
    )
    assert action == watchdog.ACTION_NONE


def test_restart_failure_alert_blocks_a_later_webui_alert() -> None:
    watchdog = load_watchdog()

    state, action = watchdog.evaluate_state(
        watchdog.WatchdogState(alerted=True),
        online=None,
        webui_login_error=True,
        now=30.0,
    )

    assert action == watchdog.ACTION_NONE
    assert state.alerted is True


def test_webui_login_error_requires_false_login_but_not_offline_true() -> None:
    watchdog = load_watchdog()

    assert watchdog.is_explicit_webui_login_error(
        {"isLogin": False, "isOffline": False, "loginError": "captcha required"}
    ) is True
    assert watchdog.is_explicit_webui_login_error(
        {"isLogin": None, "isOffline": False, "loginError": "captcha required"}
    ) is False


def test_healthy_recovery_clears_webui_alert_for_a_future_incident() -> None:
    watchdog = load_watchdog()

    state, action = watchdog.evaluate_state(
        watchdog.WatchdogState(webui_alerted=True),
        online=True,
        active_session_ok=True,
        now=30.0,
    )

    assert action == watchdog.ACTION_NONE
    assert state.webui_alerted is False


def test_continuing_webui_login_error_does_not_repeat_alert_after_onebot_recovers() -> None:
    watchdog = load_watchdog()

    state, action = watchdog.evaluate_state(
        watchdog.WatchdogState(webui_alerted=True),
        online=True,
        active_session_ok=True,
        webui_login_error=True,
        now=30.0,
    )

    assert action == watchdog.ACTION_NONE
    assert state.webui_alerted is True


def test_state_json_persists_only_safe_webui_booleans_and_error_category(tmp_path: Path) -> None:
    watchdog = load_watchdog()
    state_file = tmp_path / "onebot-watchdog.json"
    state = watchdog.WatchdogState(
        isLogin=False,
        isOffline=False,
        webui_login_error=True,
        webui_login_error_kind="reported",
    )

    watchdog.save_state(state_file, state)
    payload = json.loads(state_file.read_text(encoding="utf-8"))

    assert payload["isLogin"] is False
    assert payload["isOffline"] is False
    assert payload["webui_login_error"] is True
    assert payload["webui_login_error_kind"] == "reported"
    assert "loginError" not in payload


def test_probe_onebot_uses_read_only_group_list_to_confirm_active_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watchdog = load_watchdog()

    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []

        async def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))

        async def recv(self) -> str:
            request = self.sent[-1]
            if request["action"] == "get_status":
                return json.dumps({"status": "ok", "data": {"online": True}, "echo": request["echo"]})
            return json.dumps({"status": "ok", "data": [], "echo": request["echo"]})

    class FakeConnection:
        def __init__(self, ws: FakeWebSocket) -> None:
            self.ws = ws

        async def __aenter__(self) -> FakeWebSocket:
            return self.ws

        async def __aexit__(self, *args: object) -> None:
            return None

    fake_ws = FakeWebSocket()
    monkeypatch.setattr(
        watchdog.websockets, "connect", lambda *args, **kwargs: FakeConnection(fake_ws)
    )

    online, active_session_ok, detail = asyncio.run(watchdog.probe_onebot("ws://fake"))

    assert (online, active_session_ok, detail) == (True, True, "get_group_list")
    assert fake_ws.sent == [
        {"action": "get_status", "params": {}, "echo": "watchdog_get_status"},
        {"action": "get_group_list", "params": {"no_cache": True}, "echo": "watchdog_get_group_list"},
    ]


def test_probe_webui_hashes_token_and_returns_only_safe_login_fields(tmp_path: Path) -> None:
    watchdog = load_watchdog()
    config = tmp_path / "webui.json"
    token = "local-test-token"
    config.write_text(json.dumps({"token": token}), encoding="utf-8")
    requests: list[Request] = []

    def opener(request: Request, timeout: float) -> Any:
        requests.append(request)
        if request.full_url.endswith("/api/auth/login"):
            assert json.loads(request.data or b"{}") == {
                "hash": hashlib.sha256((token + ".napcat").encode()).hexdigest()
            }
            return FakeHttpResponse({"data": {"Credential": "test-credential"}})
        assert request.get_header("Authorization") == "Bearer test-credential"
        return FakeHttpResponse(
            {
                "data": {
                    "isLogin": False,
                    "isOffline": False,
                    "loginError": "captcha required: https://captcha.example/?sid=secret-sid",
                }
            }
        )

    result = watchdog.probe_webui(config, "http://127.0.0.1:6099", opener=opener)

    assert result == {"isLogin": False, "isOffline": False, "loginError": "reported"}
    assert len(requests) == 2
    assert token not in json.dumps(result)
    assert "test-credential" not in json.dumps(result)
    assert "secret-sid" not in json.dumps(result)


def test_probe_webui_rejects_non_loopback_or_wrong_port_urls(tmp_path: Path) -> None:
    watchdog = load_watchdog()
    config = tmp_path / "webui.json"
    config.write_text(json.dumps({"token": "local-test-token"}), encoding="utf-8")
    calls: list[Request] = []

    def opener(request: Request, timeout: float) -> Any:
        calls.append(request)
        raise AssertionError("external URL must be rejected before opening")

    result = watchdog.probe_webui(config, "http://example.com:6099", opener=opener)

    assert result == {"isLogin": None, "isOffline": None, "loginError": ""}
    assert calls == []


def test_default_webui_opener_does_not_follow_redirect_with_bearer_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watchdog = load_watchdog()
    config = tmp_path / "webui.json"
    config.write_text(json.dumps({"token": "local-test-token"}), encoding="utf-8")
    requests: list[Request] = []

    def no_redirect_opener(request: Request, timeout: float) -> Any:
        requests.append(request)
        if request.full_url.endswith("/api/auth/login"):
            return FakeHttpResponse({"data": {"Credential": "test-credential"}})
        raise watchdog.HTTPError(
            request.full_url,
            302,
            "Found",
            {"Location": "http://example.com/steal"},
            None,
        )

    monkeypatch.setattr(watchdog, "_open_without_redirects", no_redirect_opener)

    result = watchdog.probe_webui(config, "http://127.0.0.1:6099")

    assert result == {"isLogin": None, "isOffline": None, "loginError": ""}
    assert len(requests) == 2
    assert requests[-1].get_header("Authorization") == "Bearer test-credential"
    assert all("example.com" not in request.full_url for request in requests)


def test_onebot_call_uses_a_single_deadline_for_unrelated_echoes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watchdog = load_watchdog()
    time_values = iter((0.0, 0.0, 3.0, 6.0))

    class FakeLoop:
        def time(self) -> float:
            return next(time_values)

    class FakeWebSocket:
        async def send(self, raw: str) -> None:
            return None

        async def recv(self) -> str:
            if not hasattr(self, "seen"):
                self.seen = True
                return json.dumps({"echo": "unrelated"})
            return json.dumps({"echo": "watchdog_get_status", "status": "ok"})

    timeouts: list[float] = []

    async def wait_for(awaitable: Any, timeout: float) -> Any:
        timeouts.append(timeout)
        return await awaitable

    monkeypatch.setattr(watchdog.asyncio, "get_running_loop", lambda: FakeLoop())
    monkeypatch.setattr(watchdog.asyncio, "wait_for", wait_for)

    payload = asyncio.run(watchdog._onebot_call(FakeWebSocket(), "get_status", {}, timeout=8))

    assert payload["status"] == "ok"
    assert timeouts == [8.0, 5.0, 2.0]


def test_webui_notification_failure_rolls_back_alerts_for_one_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watchdog = load_watchdog()
    state_file = tmp_path / "state.json"
    log_file = tmp_path / "watchdog.log"
    raw_error = "captcha https://captcha.example/?sid=secret-sid"
    notifications: list[str] = []

    async def probe_onebot(_: str) -> tuple[bool | None, bool | None, str]:
        return None, None, "local"

    monkeypatch.setattr(watchdog, "probe_onebot", probe_onebot)
    monkeypatch.setattr(
        watchdog,
        "probe_webui",
        lambda *args, **kwargs: {"isLogin": False, "isOffline": False, "loginError": raw_error},
    )
    monkeypatch.setattr(watchdog.time, "time", lambda: 10.0)
    monkeypatch.setattr(
        watchdog,
        "notify_windows",
        lambda *args: (notifications.append("attempt") is not None, "failed"),
    )

    watchdog.run_once(
        ws_url="ws://fake",
        state_file=state_file,
        log_file=log_file,
        compose_file=tmp_path / "docker-compose.yml",
        notifier=tmp_path / "notify.ps1",
    )

    state = watchdog.load_state(state_file)
    assert state.alerted is False
    assert state.webui_alerted is False
    assert raw_error not in state_file.read_text(encoding="utf-8")
    assert raw_error not in log_file.read_text(encoding="utf-8")

    def succeeding_notify(*args: Any) -> tuple[bool, str]:
        notifications.append("attempt")
        return True, "started"

    monkeypatch.setattr(watchdog, "notify_windows", succeeding_notify)
    watchdog.run_once(
        ws_url="ws://fake",
        state_file=state_file,
        log_file=log_file,
        compose_file=tmp_path / "docker-compose.yml",
        notifier=tmp_path / "notify.ps1",
    )
    watchdog.run_once(
        ws_url="ws://fake",
        state_file=state_file,
        log_file=log_file,
        compose_file=tmp_path / "docker-compose.yml",
        notifier=tmp_path / "notify.ps1",
    )

    assert notifications == ["attempt", "attempt"]


def test_restart_failure_notification_failure_is_retried_after_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watchdog = load_watchdog()
    state_file = tmp_path / "state.json"
    notifications: list[str] = []
    now_values = iter((1.0, 2.0, 3.0, 123.0))

    async def probe_onebot(_: str) -> tuple[bool | None, bool | None, str]:
        return True, False, "local"

    monkeypatch.setattr(watchdog, "probe_onebot", probe_onebot)
    monkeypatch.setattr(
        watchdog,
        "probe_webui",
        lambda *args, **kwargs: {"isLogin": None, "isOffline": None, "loginError": ""},
    )
    monkeypatch.setattr(watchdog, "restart_napcat", lambda *args: (False, "failed"))
    monkeypatch.setattr(watchdog.time, "time", lambda: next(now_values))

    def notify(*args: Any) -> tuple[bool, str]:
        notifications.append("attempt")
        return len(notifications) > 1, "started"

    monkeypatch.setattr(watchdog, "notify_windows", notify)
    kwargs = {
        "ws_url": "ws://fake",
        "state_file": state_file,
        "log_file": tmp_path / "watchdog.log",
        "compose_file": tmp_path / "docker-compose.yml",
        "notifier": tmp_path / "notify.ps1",
    }
    watchdog.run_once(**kwargs)
    watchdog.run_once(**kwargs)
    watchdog.run_once(**kwargs)
    assert watchdog.load_state(state_file).alerted is False
    watchdog.run_once(**kwargs)

    assert notifications == ["attempt", "attempt"]


class FakeHttpResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_watchdog_lock_rejects_a_parallel_run(tmp_path: Path) -> None:
    watchdog = load_watchdog()
    lock_file = tmp_path / "watchdog.lock"

    with watchdog.exclusive_lock(lock_file) as first_acquired:
        assert first_acquired is True
        with watchdog.exclusive_lock(lock_file) as second_acquired:
            assert second_acquired is False


def test_keepalive_runs_watchdog_and_windows_notifier_is_present() -> None:
    keepalive = (REPO_ROOT / "infra/wsl/scripts/keepalive.sh").read_text(encoding="utf-8")
    notifier = REPO_ROOT / "infra/wsl/scripts/notify_windows.ps1"

    assert "onebot_watchdog.py" in keepalive
    assert "--once" in keepalive
    assert notifier.exists()
    notifier_text = notifier.read_text(encoding="utf-8")
    assert "System.Windows.Forms.MessageBox" in notifier_text
    assert "ServiceNotification" in notifier_text


def test_stop_cleans_watchdog_state_even_when_compose_down_fails() -> None:
    stop_script = (REPO_ROOT / "infra/wsl/scripts/stop.sh").read_text(encoding="utf-8")

    assert "compose_exit=0" in stop_script
    assert "docker compose -f docker-compose.yml down --remove-orphans || compose_exit=$?" in stop_script
    assert "docker compose -f docker-compose.llbot.yml down --remove-orphans || compose_exit=$?" in stop_script
    assert stop_script.index('rm -f "${flag_file}"') > stop_script.index(
        "docker compose -f docker-compose.llbot.yml down"
    )
    assert 'exit "${compose_exit}"' in stop_script
