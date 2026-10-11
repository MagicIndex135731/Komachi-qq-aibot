from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

import websockets


async def call(
    ws: Any,
    action: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 8,
) -> dict[str, Any]:
    payload = {"action": action, "params": params or {}, "echo": action}
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    def remaining() -> float:
        value = deadline - loop.time()
        if value <= 0:
            raise TimeoutError(f"OneBot {action} response timed out")
        return value

    try:
        await asyncio.wait_for(ws.send(json.dumps(payload, ensure_ascii=True)), timeout=remaining())
    except TimeoutError as exc:
        raise TimeoutError(f"OneBot {action} response timed out") from exc
    while True:
        try:
            receive_timeout = remaining()
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=receive_timeout))
        except TimeoutError as exc:
            raise TimeoutError(f"OneBot {action} response timed out") from exc
        if message.get("echo") == action:
            return message


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-url", default="ws://127.0.0.1:3001")
    parser.add_argument("--group-id", type=int, default=0)
    parser.add_argument("--history-count", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=8)
    args = parser.parse_args()

    async with websockets.connect(args.ws_url, open_timeout=8, close_timeout=3) as ws:
        status = await call(ws, "get_status", timeout=args.request_timeout)
        login = await call(ws, "get_login_info", timeout=args.request_timeout)
        print("get_status=" + json.dumps(status, ensure_ascii=False))
        print("get_login_info=" + json.dumps(login, ensure_ascii=False))

        online = bool(status.get("data", {}).get("online"))
        if not online:
            print("OneBot account is offline.", file=sys.stderr)
            return 2

        if args.group_id:
            history = await call(
                ws,
                "get_group_msg_history",
                {"group_id": args.group_id, "count": args.history_count},
                timeout=args.request_timeout,
            )
            print("get_group_msg_history=" + json.dumps(history, ensure_ascii=False))
            if history.get("status") != "ok":
                return 3

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except (OSError, TimeoutError, websockets.WebSocketException) as exc:
        print(f"OneBot probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
