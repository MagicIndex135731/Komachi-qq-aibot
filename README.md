# 小町 QQ AI Bot（WSL/Docker）

当前支持的部署方式是 WSL2 + Docker：NapCat、QQ 登录态和小町 Python 进程都运行在 WSL/Docker 中，Windows 只保留四个操作入口。

## 日常操作

在资源管理器中双击：

- `start-xiaomachi-wsl.bat`：启动 NapCat、小町和 watchdog。QQ 未登录时会自动打开 NapCat WebUI。
- `stop-xiaomachi-wsl.bat`：停止当前 WSL 小町栈。
- `status-xiaomachi-wsl.bat`：检查容器、OneBot 会话和小町心跳。
- `open-napcat-webui.bat`：手动打开 NapCat 登录页面，不启动或重启容器。

不要删除 `D:\xiaomachi-wsl-entry.sh`。三个 WSL BAT 通过这个固定 ASCII 路径查找仓库，避免中文路径经过 CMD/WSL 参数传递时乱码。

## 首次配置

要求：Windows 11、WSL2、Docker，以及一个可用的 Ubuntu WSL 发行版。

1. 在 WSL 中初始化目录和探针环境：

   ```bash
   cd "/mnt/d/qq群ai小人"
   bash infra/wsl/scripts/bootstrap_wsl.sh
   ```

2. 编辑本地文件 `infra/wsl/.env`。至少填写：

   ```dotenv
   BOT_QQ=
   OWNER_QQ=
   LLM_BASE_URL=
   LLM_API_KEY=
   LLM_MODEL=gpt-5.6-terra
   LLM_TEXT_ENDPOINT=responses
   LLM_REASONING_EFFORT=medium
   ```

3. 如果固定入口丢失，从仓库恢复：

   ```powershell
   Copy-Item .\infra\wsl\scripts\xiaomachi-wsl-entry.sh D:\xiaomachi-wsl-entry.sh
   ```

`.env`、API key、QQ 密码、WebUI token 和验证码链接不得提交 Git。

## 配置

### 群和人格

- `configs/groups.yaml`：控制群是否接收、发言、主动回复、归档和生图。
- `configs/persona.yaml`：人格、称呼和回复风格。
- `configs/safety.yaml`：安全限制。

群配置只有同时设置 `enabled: true` 和 `speak: true` 才允许小町在该群回复。

### 文本、搜索和上下文

常用环境变量位于 `infra/wsl/.env.example`：

- `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`
- `LLM_TEXT_ENDPOINT`、`LLM_REASONING_EFFORT`
- `SEARCH_API_KEY`、`SEARCH_PROVIDER`、`SEARCH_TIMEOUT_SECONDS`
- `CONTEXT_RECENT_LIMIT`、`CONTEXT_SUMMARY_LIMIT`、`CONTEXT_HISTORY_LIMIT`
- 生图直接复用主模型的 Responses `image_generation` 工具；横图使用 `1536x1024`，竖图使用 `1024x1536`，质量固定为 `high`；仅单独配置 `GROUP_IMAGE_QUEUE_CAPACITY` 和 `GROUP_IMAGE_TIMEOUT_SECONDS`

修改 `infra/wsl/.env` 后，需要重建小町容器才能加载新环境变量：

```bash
cd "/mnt/d/qq群ai小人/infra/wsl"
docker compose up -d --force-recreate xiaomachi
```

## 运行结构

- `xiaomachi-napcat`：NapCat 和 NTQQ，端口仅映射到本机 `127.0.0.1:6099`、`127.0.0.1:3001`。
- `xiaomachi-bot`：运行 `python -m app.group_main`。
- `.venv-wsl`：供 keepalive、OneBot 探针和登录 watchdog 使用，不是旧 Windows 虚拟环境。
- `infra/wsl/scripts/onebot_watchdog.py`：主动调用 `get_status` 和 `get_group_list(no_cache=true)`；连续异常时只重启 NapCat 一次，仍需登录时通知 Windows。

## 不能删除的数据

以下内容不进入 Git，但属于当前运行态：

- `infra/wsl/.env`
- `infra/wsl/runtime/napcat/ntqq`：QQ 登录态
- `infra/wsl/runtime/napcat/config`：NapCat/OneBot 配置
- `infra/wsl/runtime/logs` 和 watchdog 状态
- `.venv-wsl`
- `data/bot.db*`：聊天数据库
- `data/history`：群消息归档
- `data/image_cache`：收到的图片缓存
- `data/generated_images`：生成图片

Git 只能恢复已跟踪源码，不能恢复这些本地状态。

## 故障排查

先运行 `status-xiaomachi-wsl.bat`。常见情况：

- 容器 healthy 但 OneBot 离线：运行 `open-napcat-webui.bat` 完成验证码或重新登录。
- WebSocket 持续握手失败：通常是 QQ 未登录，不代表模型配置失败。
- 修改模型后未生效：重建 `xiaomachi` 容器，并从容器环境确认非敏感变量。
- 登录反复失效：保留 `infra/wsl/runtime/napcat/ntqq`，查看 NapCat 日志和 watchdog 状态，不要删除登录态目录。

## 开发与测试

本地开发环境可随时重建：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

提交前至少运行受影响测试、`docker compose config`、PowerShell/Bash 语法检查和 `git diff --check`。

## Git 回退

旧 Windows 运行栈清理前的回退点是 `f63efe1`。查看或恢复已跟踪文件：

```powershell
git show --stat f63efe1
git restore --source f63efe1 -- path\to\file
```

不要用 `git reset --hard` 处理包含本地运行数据的工作区。
