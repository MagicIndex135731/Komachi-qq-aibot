# WSL NapCat Isolation Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把小町的 NapCat、QQ 登录态、Python bot 和 watchdog 迁移到 WSL2/Docker 中运行，彻底隔离 Windows 本机个人 QQ。

**Architecture:** Windows 只保留启动/停止入口 BAT/PowerShell，实际服务全部在 WSL2 Ubuntu 内运行。WSL 内优先使用 Docker Compose 启动 NapCat Linux/Docker 版与小町 Python bot；用 healthcheck、restart policy 和日志探针替代当前 Windows 进程树清理逻辑。旧 Windows 版在验收前保留，不作为默认入口。

**Tech Stack:** Windows PowerShell/BAT, WSL2 Ubuntu, Docker Compose, NapCat Linux/Docker deployment, Python virtualenv, pytest, OneBot websocket.

---

## 背景与根因

当前 Windows 版已经确认存在三类问题：

1. NapCat/QQ 层会在运行一段时间后出现 OneBot 群消息流 stale，表现为群里新消息不进入 bot。
2. 旧 watchdog 对 `onebot_group_stream_stale` 的处理是停掉 NapCat/QQ/worker，然后等待用户手动重启，导致“卡死在那里”。
3. 即使 launcher 给 NapCat 传了 `--user-data-dir=D:\qq群ai小人\data\napcat\qq-userdata-9.9.26`，实测小町 QQ 子进程仍出现 `--user-data-dir="C:\Users\13573\AppData\Roaming\QQ"`，说明 Windows QQ/NapCat 这条链路不能可靠隔离个人 QQ。

所以迁移目标不是“把 Python 搬进 WSL 但继续启动 Windows QQ”，而是“NapCat/QQ 协议入口也在 WSL/Docker 内运行”。

## 文件结构

Create:

- `infra/wsl/README.md`：中文部署与排障文档。
- `infra/wsl/docker-compose.yml`：WSL 内 Docker Compose 编排 NapCat 与小町服务。
- `infra/wsl/.env.example`：WSL 部署专用非敏感配置模板。
- `infra/wsl/scripts/bootstrap_wsl.sh`：WSL 内初始化 Python venv、安装依赖、生成目录。
- `infra/wsl/scripts/start.sh`：WSL 内启动服务。
- `infra/wsl/scripts/stop.sh`：WSL 内停止服务。
- `infra/wsl/scripts/status.sh`：WSL 内状态检查。
- `infra/wsl/scripts/onebot_probe.py`：OneBot websocket 在线、登录、群消息流探针。
- `infra/wsl/scripts/sync_from_windows.ps1`：从 Windows 现有仓库同步代码和非敏感配置到 WSL 工作目录。
- `infra/wsl/scripts/redact_env.ps1`：生成脱敏配置快照，避免把密钥带入文档或 git。
- `启动小町-WSL.bat`：Windows 双击启动 WSL 版。
- `关闭小町-WSL.bat`：Windows 双击停止 WSL 版。
- `查看小町状态-WSL.bat`：Windows 双击查看 WSL 版状态。
- `tests/test_wsl_deployment_artifacts.py`：验证部署脚本、Compose 文件和敏感信息防护。

Modify:

- `.gitignore`：忽略 WSL 运行态、NapCat 数据、二维码、日志、`.env`。
- `README.md`：增加 WSL 隔离部署说明，保留中文。

Do not modify in this migration plan:

- `app/**` 主业务逻辑，除非 WSL 路径兼容测试发现硬编码 Windows 路径。
- `configs/**` 群策略内容，迁移时只复制现有配置。
- 现有 Windows 启动/停止脚本，直到 WSL 版验收通过。

## 敏感信息规则

- 不提交 `.env`、API key、QQ 账号态、二维码、NapCat cache、NapCat logs。
- 迁移时从 Windows `.env` 复制到 WSL 的实际 `.env`，但 git 只提交 `.env.example`。
- 提交前必须运行：

```powershell
git status --short
git diff --cached --name-only
git grep -n "sk-|OPENAI_API_KEY|API_KEY|Authorization|Bearer|3983010865|1807533371" -- .
```

如果命中真实密钥或 QQ 账号态文件，停止提交并清理。

## Task 1: 建立 WSL 部署骨架与安全测试

**Files:**

- Create: `tests/test_wsl_deployment_artifacts.py`
- Create: `infra/wsl/.env.example`
- Create: `infra/wsl/README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing artifact tests**

Create `tests/test_wsl_deployment_artifacts.py`:

```python
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_wsl_required_files_exist() -> None:
    required = [
        "infra/wsl/README.md",
        "infra/wsl/.env.example",
        "infra/wsl/docker-compose.yml",
        "infra/wsl/scripts/bootstrap_wsl.sh",
        "infra/wsl/scripts/start.sh",
        "infra/wsl/scripts/stop.sh",
        "infra/wsl/scripts/status.sh",
        "infra/wsl/scripts/onebot_probe.py",
        "infra/wsl/scripts/sync_from_windows.ps1",
        "infra/wsl/scripts/redact_env.ps1",
        "启动小町-WSL.bat",
        "关闭小町-WSL.bat",
        "查看小町状态-WSL.bat",
    ]
    missing = [path for path in required if not (REPO_ROOT / path).exists()]
    assert missing == []


def test_wsl_env_example_has_no_real_secrets() -> None:
    env_example = (REPO_ROOT / "infra/wsl/.env.example").read_text(encoding="utf-8")
    forbidden = ["sk-", "Bearer ", "OPENAI_API_KEY=", "3983010865", "1807533371"]
    assert not any(token in env_example for token in forbidden)
    assert "NAPCAT_WS_URL=ws://napcat:3001" in env_example
    assert "GROUP_STREAM_WATCH_GROUP_ID=" in env_example


def test_gitignore_excludes_wsl_runtime_state() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    required_patterns = [
        "infra/wsl/runtime/",
        "infra/wsl/.env",
        "data/napcat/",
        "data/logs/",
    ]
    for pattern in required_patterns:
        assert pattern in gitignore
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_wsl_deployment_artifacts.py -q
```

Expected: FAIL because WSL artifact files do not exist yet.

- [ ] **Step 3: Create `.env.example`**

Create `infra/wsl/.env.example`:

```dotenv
# Copy this file to infra/wsl/.env inside WSL or generate it with sync_from_windows.ps1.
# Do not commit infra/wsl/.env.

TZ=Asia/Shanghai

# OneBot websocket inside docker network.
NAPCAT_WS_URL=ws://napcat:3001
NAPCAT_HTTP_PORT=6099
NAPCAT_WS_PORT=3001

# Fill with the watched group id used by Xiaomachi's group stream probe.
GROUP_STREAM_WATCH_GROUP_ID=
GROUP_STREAM_MAX_LAG_SECONDS=1800

# Existing Xiaomachi runtime options should be copied from the Windows .env.
ONEBOT_PROBE_STARTUP_GRACE_SECONDS=180
XIAOMACHI_FULL_RESTART_COOLDOWN_SECONDS=600

# API keys and model settings must be provided in infra/wsl/.env only.
OPENAI_API_KEY=
OPENAI_BASE_URL=
```

- [ ] **Step 4: Create initial README**

Create `infra/wsl/README.md`:

```markdown
# 小町 WSL 隔离部署

目标：让 NapCat、QQ 登录态、小町 Python 进程都运行在 WSL2/Docker 内，Windows 只保留启动入口，从而避免小町 QQ 与 Windows 本机个人 QQ 共用数据目录。

禁止方案：不要从 WSL 中启动 Windows 的 `QQ.exe`。如果看到小町 QQ 进程路径仍是 `C:\app\QQ\QQ.exe` 或命令行包含 `C:\Users\13573\AppData\Roaming\QQ`，说明隔离失败。

验收标准：

- `查看小町状态-WSL.bat` 显示 NapCat 容器 healthy。
- OneBot `get_status` 返回 `online=true`。
- `get_login_info` 返回小町 QQ，而不是个人 QQ。
- 群消息流探针能读取目标群最新消息。
- Windows 任务管理器里没有由小町启动的 Windows `QQ.exe`。
- Windows 个人 QQ 可以独立登录，不影响小町。
```

- [ ] **Step 5: Update `.gitignore`**

Append these lines to `.gitignore`:

```gitignore

# WSL/Docker deployment runtime state
infra/wsl/.env
infra/wsl/runtime/
infra/wsl/.cache/

# Local bot runtime state
data/logs/
data/napcat/
*.qrcode.png
qrcode.png
```

- [ ] **Step 6: Run test to verify current partial state**

Run:

```powershell
python -m pytest tests/test_wsl_deployment_artifacts.py -q
```

Expected: still FAIL because Compose, scripts and BAT files are not created yet.

- [ ] **Step 7: Commit**

```powershell
git add tests/test_wsl_deployment_artifacts.py infra/wsl/.env.example infra/wsl/README.md .gitignore
git commit -m "test: add wsl isolation deployment artifact checks"
```

## Task 2: 添加 WSL Docker Compose 与 Linux 启停脚本

**Files:**

- Create: `infra/wsl/docker-compose.yml`
- Create: `infra/wsl/scripts/bootstrap_wsl.sh`
- Create: `infra/wsl/scripts/start.sh`
- Create: `infra/wsl/scripts/stop.sh`
- Create: `infra/wsl/scripts/status.sh`

- [ ] **Step 1: Create Docker Compose**

Create `infra/wsl/docker-compose.yml`:

```yaml
services:
  napcat:
    image: napcat/napcat:latest
    container_name: xiaomachi-napcat
    restart: unless-stopped
    environment:
      - TZ=${TZ:-Asia/Shanghai}
      - NAPCAT_GID=${NAPCAT_GID:-1000}
      - NAPCAT_UID=${NAPCAT_UID:-1000}
    ports:
      - "127.0.0.1:${NAPCAT_HTTP_PORT:-6099}:6099"
      - "127.0.0.1:${NAPCAT_WS_PORT:-3001}:3001"
    volumes:
      - ./runtime/napcat/config:/app/napcat/config
      - ./runtime/napcat/logs:/app/napcat/logs
      - ./runtime/napcat/cache:/app/napcat/cache
    healthcheck:
      test: ["CMD-SHELL", "node -e \"require('net').connect(3001,'127.0.0.1').on('connect',()=>process.exit(0)).on('error',()=>process.exit(1))\""]
      interval: 20s
      timeout: 5s
      retries: 6
      start_period: 60s

  xiaomachi:
    image: python:3.11-slim
    container_name: xiaomachi-bot
    restart: unless-stopped
    working_dir: /workspace
    env_file:
      - ./.env
    environment:
      - NAPCAT_WS_URL=ws://napcat:3001
      - PYTHONUNBUFFERED=1
      - TZ=${TZ:-Asia/Shanghai}
    volumes:
      - ../../:/workspace
      - ./runtime/logs:/workspace/data/logs
      - ./runtime/cache:/workspace/data/cache
    depends_on:
      napcat:
        condition: service_healthy
    command: >
      bash -lc "
      python -m pip install -U pip &&
      python -m pip install -e . &&
      python -m app.group_main
      "
```

If the official NapCat image name differs in the installed environment, inspect the current NapCat documentation and replace only the `image:` line. Do not switch back to Windows QQ.

- [ ] **Step 2: Create bootstrap script**

Create `infra/wsl/scripts/bootstrap_wsl.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${WSL_DIR}/../.." && pwd)"

cd "${WSL_DIR}"
mkdir -p runtime/napcat/config runtime/napcat/logs runtime/napcat/cache runtime/logs runtime/cache

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created ${WSL_DIR}/.env from .env.example. Fill secrets before starting."
fi

cd "${REPO_ROOT}"
python3 -m venv .venv-wsl
./.venv-wsl/bin/python -m pip install -U pip
./.venv-wsl/bin/python -m pip install -e .

echo "WSL bootstrap complete."
```

- [ ] **Step 3: Create start script**

Create `infra/wsl/scripts/start.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${WSL_DIR}"
if [[ ! -f .env ]]; then
  echo "Missing ${WSL_DIR}/.env. Run bootstrap_wsl.sh and fill required secrets."
  exit 1
fi

docker compose up -d
docker compose ps
```

- [ ] **Step 4: Create stop script**

Create `infra/wsl/scripts/stop.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${WSL_DIR}"
docker compose down
```

- [ ] **Step 5: Create status script**

Create `infra/wsl/scripts/status.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${WSL_DIR}"
docker compose ps
echo
docker compose logs --tail=80 napcat
echo
docker compose logs --tail=80 xiaomachi
```

- [ ] **Step 6: Mark scripts executable in WSL**

Run inside WSL:

```bash
chmod +x infra/wsl/scripts/*.sh
```

- [ ] **Step 7: Run artifact test**

Run on Windows:

```powershell
python -m pytest tests/test_wsl_deployment_artifacts.py -q
```

Expected: still FAIL because OneBot probe, PowerShell sync scripts, and BAT wrappers are missing.

- [ ] **Step 8: Commit**

```powershell
git add infra/wsl/docker-compose.yml infra/wsl/scripts/bootstrap_wsl.sh infra/wsl/scripts/start.sh infra/wsl/scripts/stop.sh infra/wsl/scripts/status.sh
git commit -m "feat: add wsl docker deployment scripts"
```

## Task 3: 添加 OneBot 探针和 Windows 包装入口

**Files:**

- Create: `infra/wsl/scripts/onebot_probe.py`
- Create: `启动小町-WSL.bat`
- Create: `关闭小町-WSL.bat`
- Create: `查看小町状态-WSL.bat`

- [ ] **Step 1: Create OneBot probe**

Create `infra/wsl/scripts/onebot_probe.py`:

```python
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
```

- [ ] **Step 2: Create Windows start BAT**

Create `启动小町-WSL.bat`:

```bat
@echo off
setlocal
cd /d "%~dp0"
wsl.exe bash -lc "cd '/mnt/d/qq群ai小人' && bash infra/wsl/scripts/start.sh"
pause
```

- [ ] **Step 3: Create Windows stop BAT**

Create `关闭小町-WSL.bat`:

```bat
@echo off
setlocal
cd /d "%~dp0"
wsl.exe bash -lc "cd '/mnt/d/qq群ai小人' && bash infra/wsl/scripts/stop.sh"
pause
```

- [ ] **Step 4: Create Windows status BAT**

Create `查看小町状态-WSL.bat`:

```bat
@echo off
setlocal
cd /d "%~dp0"
wsl.exe bash -lc "cd '/mnt/d/qq群ai小人' && bash infra/wsl/scripts/status.sh"
pause
```

- [ ] **Step 5: Run artifact test**

Run:

```powershell
python -m pytest tests/test_wsl_deployment_artifacts.py -q
```

Expected: still FAIL because sync/redact scripts are missing.

- [ ] **Step 6: Commit**

```powershell
git add infra/wsl/scripts/onebot_probe.py 启动小町-WSL.bat 关闭小町-WSL.bat 查看小町状态-WSL.bat
git commit -m "feat: add wsl onebot probe and windows launchers"
```

## Task 4: 添加配置同步和脱敏脚本

**Files:**

- Create: `infra/wsl/scripts/sync_from_windows.ps1`
- Create: `infra/wsl/scripts/redact_env.ps1`

- [ ] **Step 1: Create redaction script**

Create `infra/wsl/scripts/redact_env.ps1`:

```powershell
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

$workdir = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$source = Join-Path $workdir ".env"
$target = Join-Path $workdir "infra\wsl\runtime\redacted-env.snapshot"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null

if (-not (Test-Path $source)) {
    throw "Missing .env: $source"
}

$secretPattern = "(?i)(KEY|TOKEN|SECRET|PASSWORD|BASE_URL|API)"
$lines = foreach ($line in Get-Content -Path $source -Encoding utf8) {
    if ($line -match "^\s*#" -or $line -notmatch "=") {
        $line
        continue
    }
    $key = $line.Substring(0, $line.IndexOf("=")).Trim()
    if ($key -match $secretPattern) {
        "$key=<redacted>"
    } else {
        $line
    }
}

$lines | Set-Content -Path $target -Encoding utf8
Write-Host "Wrote redacted env snapshot: $target"
```

- [ ] **Step 2: Create sync script**

Create `infra/wsl/scripts/sync_from_windows.ps1`:

```powershell
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8
$OutputEncoding = [Console]::OutputEncoding

$workdir = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$wslDir = Join-Path $workdir "infra\wsl"
$sourceEnv = Join-Path $workdir ".env"
$targetEnv = Join-Path $wslDir ".env"

if (-not (Test-Path $sourceEnv)) {
    throw "Missing Windows .env: $sourceEnv"
}

New-Item -ItemType Directory -Force -Path $wslDir | Out-Null
Copy-Item -LiteralPath $sourceEnv -Destination $targetEnv -Force

$content = Get-Content -Path $targetEnv -Raw -Encoding utf8
$content = $content -replace "NAPCAT_WS_URL=.*", "NAPCAT_WS_URL=ws://napcat:3001"
$content = $content -replace "QQ_EXE_PATH=.*", "# QQ_EXE_PATH is intentionally unused in WSL deployment"
$content = $content -replace "QQ_EXTRA_ARGS=.*", "# QQ_EXTRA_ARGS is intentionally unused in WSL deployment"
$content = $content -replace "NAPCAT_SHELL_DIR=.*", "# NAPCAT_SHELL_DIR is intentionally unused in WSL deployment"
$content | Set-Content -Path $targetEnv -Encoding utf8

Write-Host "Copied sanitized runtime env to: $targetEnv"
Write-Host "Do not commit infra/wsl/.env."
```

- [ ] **Step 3: Run artifact test**

Run:

```powershell
python -m pytest tests/test_wsl_deployment_artifacts.py -q
```

Expected: PASS.

- [ ] **Step 4: Run sync script**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\infra\wsl\scripts\sync_from_windows.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\infra\wsl\scripts\redact_env.ps1
```

Expected:

- `infra/wsl/.env` exists but is ignored by git.
- `infra/wsl/runtime/redacted-env.snapshot` exists but is ignored by git.
- `infra/wsl/.env` contains `NAPCAT_WS_URL=ws://napcat:3001`.

- [ ] **Step 5: Commit**

```powershell
git add infra/wsl/scripts/sync_from_windows.ps1 infra/wsl/scripts/redact_env.ps1
git commit -m "feat: add wsl config sync and redaction scripts"
```

## Task 5: WSL 环境实机验收

**Files:**

- Runtime only: `infra/wsl/.env`
- Runtime only: `infra/wsl/runtime/**`

- [ ] **Step 1: Confirm WSL and Docker**

Run on Windows:

```powershell
wsl.exe --status
wsl.exe bash -lc "uname -a && docker --version && docker compose version"
```

Expected:

- WSL default distro is Ubuntu or another Linux distro.
- Docker and Docker Compose are available from WSL.

If Docker is missing, install Docker Desktop with WSL integration or install Docker Engine inside WSL, then rerun this step.

- [ ] **Step 2: Bootstrap**

Run:

```powershell
wsl.exe bash -lc "cd '/mnt/d/qq群ai小人' && bash infra/wsl/scripts/bootstrap_wsl.sh"
```

Expected:

- `.venv-wsl` created.
- `infra/wsl/.env` exists.
- runtime directories exist.

- [ ] **Step 3: Sync config**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\infra\wsl\scripts\sync_from_windows.ps1
```

Expected: `infra/wsl/.env` uses `NAPCAT_WS_URL=ws://napcat:3001` and does not contain active `QQ_EXE_PATH`.

- [ ] **Step 4: Start WSL deployment**

Run:

```powershell
wsl.exe bash -lc "cd '/mnt/d/qq群ai小人' && bash infra/wsl/scripts/start.sh"
```

Expected:

- `xiaomachi-napcat` container starts.
- `xiaomachi-bot` container starts after NapCat is healthy.
- If QR login is required, find QR in NapCat container logs or NapCat web UI, scan once.

- [ ] **Step 5: Probe OneBot**

Run:

```powershell
wsl.exe bash -lc "cd '/mnt/d/qq群ai小人' && .venv-wsl/bin/python infra/wsl/scripts/onebot_probe.py --ws-url ws://127.0.0.1:3001 --group-id ${GROUP_STREAM_WATCH_GROUP_ID:-0}"
```

If PowerShell does not expand `GROUP_STREAM_WATCH_GROUP_ID`, manually replace it with the group id from `.env`.

Expected:

- `get_status` prints `online=true`.
- `get_login_info` is 小町 QQ.
- Group history returns `status=ok` when group id is provided.

- [ ] **Step 6: Verify Windows QQ isolation**

Run:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -eq "QQ.exe" } |
  Select-Object ProcessId,ExecutablePath,CommandLine |
  Format-List
```

Expected:

- No Xiaomachi-launched Windows QQ exists.
- If Windows personal QQ exists, it should be `C:\app\QQ\QQ.exe` or your personal install path only.
- No Xiaomachi QQ command line should include `D:\qq群ai小人\data\napcat\qq-runtime`.

- [ ] **Step 7: Send real group test**

In the target group, send:

```text
@比企谷小町 1
```

Expected:

- Xiaomachi replies once.
- `docker compose logs -f xiaomachi` shows the incoming message and response path.
- No Windows QQ login prompt appears.

- [ ] **Step 8: Stop WSL deployment**

Run:

```powershell
wsl.exe bash -lc "cd '/mnt/d/qq群ai小人' && bash infra/wsl/scripts/stop.sh"
```

Expected:

- Containers stop.
- Windows personal QQ remains open.

## Task 6: 切换 Windows 入口与文档

**Files:**

- Modify: `README.md`
- Optional modify after approval: `启动小町.bat`, `关闭小町.bat`

- [ ] **Step 1: Update README**

Add a Chinese section to `README.md`:

```markdown
## WSL 隔离部署

推荐长期运行使用 WSL 版入口：

- `启动小町-WSL.bat`
- `关闭小町-WSL.bat`
- `查看小町状态-WSL.bat`

WSL 版会把 NapCat、QQ 登录态、小町 Python 进程放在 WSL/Docker 内运行，避免与 Windows 本机个人 QQ 共用数据目录。

首次使用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\infra\wsl\scripts\sync_from_windows.ps1
wsl.exe bash -lc "cd '/mnt/d/qq群ai小人' && bash infra/wsl/scripts/bootstrap_wsl.sh"
wsl.exe bash -lc "cd '/mnt/d/qq群ai小人' && bash infra/wsl/scripts/start.sh"
```

验收：

```powershell
wsl.exe bash -lc "cd '/mnt/d/qq群ai小人' && .venv-wsl/bin/python infra/wsl/scripts/onebot_probe.py --ws-url ws://127.0.0.1:3001"
```
```

- [ ] **Step 2: Decide whether to replace old BAT files**

Only after WSL real group test passes, ask the user:

```text
WSL 版已通过验收。是否把原来的 启动小町.bat / 关闭小町.bat 改成调用 WSL 版？
```

If user says yes, replace:

`启动小町.bat`:

```bat
@echo off
call "%~dp0启动小町-WSL.bat"
```

`关闭小町.bat`:

```bat
@echo off
call "%~dp0关闭小町-WSL.bat"
```

- [ ] **Step 3: Run tests**

Run:

```powershell
python -m pytest tests/test_wsl_deployment_artifacts.py tests/test_runtime_watchdog.py -q
```

Expected: PASS.

- [ ] **Step 4: Secret scan**

Run:

```powershell
git status --short
git grep -n "sk-|OPENAI_API_KEY=sk-|Authorization: Bearer|3983010865|1807533371" -- .
```

Expected:

- No real API keys.
- If QQ numbers appear only in old local logs or ignored runtime state, do not add those files.

- [ ] **Step 5: Commit**

```powershell
git add README.md tests/test_wsl_deployment_artifacts.py infra/wsl .gitignore 启动小町-WSL.bat 关闭小町-WSL.bat 查看小町状态-WSL.bat
git commit -m "docs: add wsl isolated deployment workflow"
```

## Rollback

If WSL deployment fails before default BAT replacement:

```powershell
wsl.exe bash -lc "cd '/mnt/d/qq群ai小人' && bash infra/wsl/scripts/stop.sh"
```

Then continue using existing Windows entry:

```powershell
.\启动小町.bat
```

If old `启动小町.bat` and `关闭小町.bat` were already replaced, restore them from git:

```powershell
git checkout HEAD~1 -- 启动小町.bat 关闭小町.bat
```

Do not delete `infra/wsl/runtime/napcat` until QQ login and message reception are confirmed stable in the replacement setup.

## Verification Summary Required Before Claiming Complete

Final report must include:

- `python -m pytest tests/test_wsl_deployment_artifacts.py tests/test_runtime_watchdog.py -q` result.
- `docker compose ps` output summary.
- OneBot probe result: `online=true` and login QQ nickname/user id.
- Windows process check proving no Xiaomachi Windows QQ process.
- Whether real target group reply test passed.
- Whether original BAT files were replaced or WSL-specific BAT files were left separate.

## New Conversation Prompt

Copy the following prompt into a new Codex conversation:

```text
中文交互。你在 Windows 机器上，仓库路径是 D:\qq群ai小人。请按仓库里的计划文档 D:\qq群ai小人\docs\superpowers\plans\2026-07-09-wsl-napcat-migration.md 执行，把小町迁移到 WSL2/Docker 隔离运行。

目标：
1. 不要让小町再启动或依赖 Windows QQ.exe。
2. NapCat、QQ 登录态、小町 Python bot 都必须在 WSL2/Docker 内运行。
3. Windows 只保留启动/停止/查看状态 BAT 入口。
4. 保留现有 Windows 版作为回滚，WSL 版验收通过前不要替换原来的 启动小町.bat / 关闭小町.bat。
5. 迁移现有 .env、API、群配置、人格配置，但不要把任何敏感信息提交到 git。
6. 验收必须证明：OneBot online=true、get_login_info 是小町账号、目标群能收到并回复、Windows 进程列表里没有小町启动的 QQ.exe。

执行要求：
- 使用 superpowers:subagent-driven-development 或 superpowers:executing-plans 按计划逐项执行。
- 先读完整计划文档，再动文件。
- 每个阶段先写/跑测试，再实现。
- 不要盲目联网测试消耗真实 token；只做必要的 OneBot 状态和一次真实群聊验收。
- 若发现 NapCat 官方 Docker 镜像名或路径与计划不同，查官方文档并只调整 NapCat 容器配置，不能退回 Windows QQ 方案。
- 任何时候都不要提交 .env、NapCat 登录态、二维码、日志、cache、API key。
- 完成后给出测试结果、docker compose ps 摘要、OneBot 探针结果、Windows QQ 进程隔离证明、是否替换默认 BAT 的状态。
```
