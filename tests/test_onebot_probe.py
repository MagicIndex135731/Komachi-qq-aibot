from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "infra/wsl/scripts/onebot_probe.py"
SPEC = importlib.util.spec_from_file_location("onebot_probe", PROBE_PATH)
assert SPEC and SPEC.loader
onebot_probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(onebot_probe)


class NeverRespondingWebSocket:
    async def send(self, _: str) -> None:
        return None

    async def recv(self) -> str:
        await asyncio.Future()
        raise AssertionError("unreachable")


class NeverSendingWebSocket:
    async def send(self, _: str) -> None:
        await asyncio.Future()

    async def recv(self) -> str:
        raise AssertionError("recv must not be called after send times out")


class WrongEchoWebSocket:
    async def send(self, _: str) -> None:
        return None

    async def recv(self) -> str:
        await asyncio.sleep(0)
        return '{"echo": "another_action"}'


def test_call_times_out_when_onebot_never_returns_the_matching_echo() -> None:
    with pytest.raises(TimeoutError, match="get_status"):
        asyncio.run(onebot_probe.call(NeverRespondingWebSocket(), "get_status", timeout=0.01))


def test_call_times_out_when_onebot_send_blocks() -> None:
    with pytest.raises(TimeoutError, match="get_status"):
        asyncio.run(onebot_probe.call(NeverSendingWebSocket(), "get_status", timeout=0.01))


def test_call_deadline_is_not_extended_by_nonmatching_echoes() -> None:
    with pytest.raises(TimeoutError, match="get_status"):
        asyncio.run(onebot_probe.call(WrongEchoWebSocket(), "get_status", timeout=0.01))
