from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

import websockets


async def call(ws: Any, action: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"action": action, "params": params or {}, "echo": action}
    await ws.send(json.dumps(payload, ensure_ascii=True))
    while True:
        message = json.loads(await ws.recv())
        if message.get("echo") == action:
            return message


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-url", default="ws://127.0.0.1:3001")
    parser.add_argument("--group-id", type=int, default=0)
    parser.add_argument("--history-count", type=int, default=5)
    args = parser.parse_args()

    async with websockets.connect(args.ws_url, open_timeout=8, close_timeout=3) as ws:
        status = await call(ws, "get_status")
        login = await call(ws, "get_login_info")
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
            )
            print("get_group_msg_history=" + json.dumps(history, ensure_ascii=False))
            if history.get("status") != "ok":
                return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
