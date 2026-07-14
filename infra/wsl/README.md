# WSL/Docker 运行目录

这里是小町当前唯一受支持的运行栈。默认 `QQ_PLATFORM=llbot` 使用 LLBot；`QQ_PLATFORM=napcat` 是本地回退选项。两者共享同一套小町业务、数据库和模型配置，但登录态各自独立保存。

## 启动链路

```text
start-xiaomachi-wsl.bat
  -> D:\xiaomachi-wsl-entry.sh
  -> infra/wsl/scripts/start.sh
  -> 启动当前 QQ 平台容器
  -> 条件打开当前 QQ 平台 WebUI
  -> 无依赖启动小町（OneBot 未就绪时自动重连）
  -> OneBot 与小町心跳检查
```

停止和状态入口使用同一个固定脚本，分别调用 `stop.sh` 和 `status.sh`。

## 初始化

在 WSL 中执行：

```bash
cd "/mnt/d/qq群ai小人"
bash infra/wsl/scripts/bootstrap_wsl.sh
```

脚本会创建：

- `infra/wsl/.env`：从 `.env.example` 生成，需要手工填入本地密钥。
- `infra/wsl/runtime/napcat/config/onebot11.json`：本机 OneBot WebSocket 配置。
- `.venv-wsl`：watchdog 和探针环境。

## 操作命令

推荐从 Windows 使用仓库根目录的 BAT。WSL 内也可以直接运行：

```bash
cd "/mnt/d/qq群ai小人"
bash infra/wsl/scripts/start.sh
bash infra/wsl/scripts/status.sh
bash infra/wsl/scripts/stop.sh
```

`start.sh` 先启动 QQ 平台并尝试打开 WebUI，再启动小町，避免 Compose 的健康依赖阻塞登录页面。LLBot WebUI 为 `http://127.0.0.1:3080/`，OneBot 为 `ws://127.0.0.1:3002`；NapCat 回退平台仍使用 `6099` 与 `3001`。浏览器启动失败不会阻断容器。

文本模型使用 Responses 端点时，可在 `.env` 设置 `LLM_BUILTIN_WEB_SEARCH=true` 启用主模型内置联网检索。明确“联网/搜索/查资料”的群请求会强制检索；工具事件保存到 `runtime/logs/responses-tool-events.jsonl`，不进入 Git。

## 运行态保护

不要删除：

- `.env`
- `runtime/llbot/data`
- `runtime/napcat/ntqq`
- `runtime/napcat/config`
- `runtime/logs`
- `runtime/onebot-watchdog.json`

`runtime/pip-cache` 可以重建，但保留它能显著缩短容器重建时间。

## 验收

```bash
docker compose config --quiet
docker compose ps
bash scripts/status.sh
```

正常在线时应看到当前 QQ 平台 healthy、OneBot `online=true`、主动群列表探针成功，以及新鲜的小町心跳。若 QQ 本身已离线，先完成对应 WebUI 登录，再重复状态检查。
