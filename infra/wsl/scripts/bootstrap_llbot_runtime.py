from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def write_private(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsl-dir", type=Path, required=True)
    args = parser.parse_args()

    env = read_env(args.wsl_dir / ".env")
    qq = env.get("BOT_QQ", "").strip()
    if not qq.isdigit():
        raise SystemExit("BOT_QQ must be configured in infra/wsl/.env")
    try:
        onebot_port = int(env.get("LLBOT_WS_PORT", "3002"))
    except ValueError as exc:
        raise SystemExit("LLBOT_WS_PORT must be an integer") from exc
    if not 1 <= onebot_port <= 65535:
        raise SystemExit("LLBOT_WS_PORT must be between 1 and 65535")

    data_dir = args.wsl_dir / "runtime" / "llbot" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    webui_token = data_dir / "webui_token.txt"
    if not webui_token.exists():
        write_private(webui_token, secrets.token_urlsafe(24))

    auth_token = data_dir / "auth_token.txt"
    if not auth_token.exists():
        write_private(auth_token, "")

    config_path = data_dir / f"config_{qq}.json"
    if not config_path.exists():
        payload = {
            "milky": {"enable": False},
            "satori": {"enable": False},
            "ob11": {
                "enable": True,
                "connect": [
                    {
                        "type": "ws",
                        "enable": True,
                        "host": "",
                        "port": onebot_port,
                        "token": "",
                        "reportSelfMessage": False,
                        "reportOfflineMessage": False,
                        "messageFormat": "array",
                        "debug": False,
                        "heartInterval": 30000,
                    }
                ],
            },
            "webui": {"enable": True, "host": "", "port": 3080},
        }
        write_private(config_path, json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
    else:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        connections = payload.get("ob11", {}).get("connect", [])
        for connection in connections:
            if connection.get("type") == "ws" and connection.get("enable") is True:
                connection["port"] = onebot_port
        webui = payload.setdefault("webui", {})
        webui.update({"enable": True, "host": "", "port": 3080})
        write_private(config_path, json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
