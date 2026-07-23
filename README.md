# 小町 QQ AI Bot（WSL/Docker）

当前唯一受支持的部署方式是 WSL2 + Docker。默认 QQ 平台为 LLBot；NapCat 保留为本地回退选项。两者共享小町 Python 进程、数据库、模型配置和业务代码，但 QQ 登录态彼此独立。

## 日常操作

在资源管理器中双击：

- `start-xiaomachi-wsl.bat`：启动 `infra/wsl/.env` 中 `QQ_PLATFORM` 选择的平台、小町和 watchdog；启动前会关闭另一平台，避免同号并行。
- `stop-xiaomachi-wsl.bat`：停止当前 WSL 小町栈。
- `status-xiaomachi-wsl.bat`：检查容器、OneBot 会话和小町心跳。
- `open-napcat-webui.bat`：手动打开 NapCat 登录页面，不启动或重启容器。
- `open-llbot-webui.bat`：手动打开 LLBot WebUI，并把本地 WebUI 密码复制到剪贴板。

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
   QQ_PLATFORM=llbot
   LLM_BASE_URL=
   LLM_API_KEY=
   LLM_MODEL=gpt-5.6-terra
   LLM_TEXT_ENDPOINT=responses
   LLM_REASONING_EFFORT=medium
   LLM_BUILTIN_WEB_SEARCH=true
   LLM_BUILTIN_WEB_SEARCH_CONTEXT_SIZE=high
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
- `LLM_BUILTIN_WEB_SEARCH`、`LLM_BUILTIN_WEB_SEARCH_CONTEXT_SIZE`
- `SEARCH_PROVIDER`、`SEARCH_API_KEY`、`SEARCH_BASE_URL`
- `CONTEXT_RECENT_LIMIT`（默认 100 条近期实际消息）、`CONTEXT_SUMMARY_LIMIT`、`CONTEXT_HISTORY_LIMIT`；带有旧话题承接、人物指代或时间回顾语义的问题会自动扩大详细历史回溯。
- `MEMORY_COMPACTION_ENABLED`、`MEMORY_COMPACTION_BATCH_SIZE`、`MEMORY_COMPACTION_BACKFILL_WINDOWS`
- 生图直接复用主模型的 Responses `image_generation` 工具；横图使用 `1536x1024`，竖图使用 `1024x1536`，质量固定为 `high`；仅单独配置 `GROUP_IMAGE_QUEUE_CAPACITY` 和 `GROUP_IMAGE_TIMEOUT_SECONDS`

`LLM_TEXT_ENDPOINT=responses` 且 `LLM_BUILTIN_WEB_SEARCH=true` 时，文本请求可使用主模型的内置 `web_search` 工具。明确写出“联网”“搜索”“查资料”等请求会强制调用检索；普通聊天则由模型自行决定是否检索。实际工具调用会记录在未纳入 Git 的 `infra/wsl/runtime/logs/responses-tool-events.jsonl`，用于核验是否真的联网。

当 LLBot 连续返回 `retcode=1200 / waitForSelfEcho timeout` 时，小町会把原回复标记为“QQ 拦截、未送达”并保留在本地上下文，同时在群内改发固定的失败提示。后续模型可以理解原回复，但会收到不得复述其中敏感细节的明确约束；这类记录不会进入自动摘要或长期记忆压缩。

修改 `infra/wsl/.env` 后，需要重建小町容器才能加载新环境变量：

```bash
cd "/mnt/d/qq群ai小人/infra/wsl"
docker compose up -d --force-recreate xiaomachi
```

### 群聊记忆编排 V2 灰度与回滚

`infra/wsl/.env.example` 给出了全部 `MEMORY_*` 配置的无秘密示例。初始值保持
`MEMORY_ORCHESTRATION_V2_ENABLED=true` 与
`MEMORY_ORCHESTRATION_SHADOW_MODE=true`：V1 继续生成真实提示词，V2 仅异步记录
安全的 IDs、计数、分数、token、耗时和错误类别，不能增加群聊回复延迟。

The required rollout order is: **shadow -> backfill -> evaluate -> active**.
FastEmbed 模型缓存位于持久数据卷中的 `/workspace/data/models`；镜像构建期安装依赖，
运行期不会重新安装。模型或 provider 不可用时，保持 shadow/V1 或只走 FTS，不能直接
启用 V2。

启用前，先从在线 SQLite 数据库通过 backup API 生成并验证备份（`integrity_check=ok`），
再运行可恢复的回填并记录 backfill run、每群 snapshot watermark、episode/文档/embedding
覆盖率以及 pending/running/failed job 数。使用 `data/memory_eval/` 中不纳入 Git 的人工确认
JSONL 对比 V1/V2；只有冻结 run 内 mandatory jobs 全部清空、embedding ready 且评测达标后，
才把 `MEMORY_ORCHESTRATION_SHADOW_MODE=false`，进入 active V2 阶段。

发布前后记录 `xiaomachi-llbot` 的 container ID 与 `StartedAt`。发布只允许重建
`xiaomachi` service（容器名 `xiaomachi-bot`）：

```bash
cd "/mnt/d/qq群ai小人/infra/wsl"
docker compose build xiaomachi
docker compose up -d --no-deps --force-recreate xiaomachi
```

**must not restart xiaomachi-llbot**：不得重建或重启 LLBot，也不得删除其登录态。若 V2
出现故障，立即设置 `MEMORY_ORCHESTRATION_V2_ENABLED=false` 回到 V1；若仅向量通道有问题，
设置 `MEMORY_EMBEDDING_PROVIDER=disabled` 保留 FTS。正常回滚不恢复数据库；只有确认数据
损坏并获得单独授权时，才可从已验证 backup 恢复。

### 群聊记忆 V2 迁移与评测

迁移必须严格按“在线备份 → 水位内回填 → 真实数据集评测 → 启用 V2”执行。以下命令只针对
`xiaomachi-bot` 的数据库；不要把 LLBot 数据卷或 `.env` 作为参数。

```bash
python scripts/backup_memory_v2.py \
  --database /workspace/data/bot.db \
  --backup-dir /workspace/data/backups \
  --tag pre-memory-v2-YYYYMMDDTHHMMSSZ

python scripts/backfill_memory_v2.py \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-pre-memory-v2-YYYYMMDDTHHMMSSZ.manifest.json \
  --run-key pre-memory-v2-YYYYMMDDTHHMMSSZ \
  --output /workspace/data/memory_eval/backfill-report.json

python scripts/build_memory_eval_dataset.py \
  --database /workspace/data/backups/bot-pre-memory-v2-YYYYMMDDTHHMMSSZ.db \
  --manifest /workspace/data/backups/bot-pre-memory-v2-YYYYMMDDTHHMMSSZ.manifest.json \
  --output /workspace/data/memory_eval/cases.jsonl \
  --review-output /workspace/data/memory_eval/cases-review.json

python scripts/run_memory_recall_eval.py \
  --database /workspace/data/bot.db \
  --dataset /workspace/data/memory_eval/cases.jsonl \
  --review /workspace/data/memory_eval/cases-review.json \
  --backfill-run-key pre-memory-v2-YYYYMMDDTHHMMSSZ \
  --results-output /workspace/data/memory_eval/results.jsonl \
  --report-output /workspace/data/memory_eval/report.json \
  --benchmark-output /workspace/data/memory_eval/benchmark.json \
  --warmup 20 --benchmark-runs 250 --enforce-real-dataset
```

生产镜像使用 CUDA 12.8、cuDNN 和 `fastembed-gpu`。`MEMORY_EMBEDDING_DEVICE=auto`
会优先使用 `CUDAExecutionProvider`，CUDA 初始化或推理失败时回退 CPU；Docker 通过
`nvidia.com/gpu=all` CDI 设备把 GPU 仅分配给 `xiaomachi`，不会分配给 LLBot。
WSL 主机需安装 NVIDIA Container Toolkit 并生成 `/etc/cdi/nvidia.yaml`。
模型首次下载完成后可设置 `MEMORY_EMBEDDING_LOCAL_FILES_ONLY=true`，让后续启动严格
使用持久化缓存，不再依赖 Hugging Face 网络。

回填命令会验证备份账本，固定逐群 snapshot watermark，并要求 mandatory jobs
`queued/running/failed=0`、无 orphan、无 blocked 派生物和无 embedding failure 才标记完成。
评测数据及报告位于 gitignored 的 `data/memory_eval/`；不要提交真实聊天内容。

## 运行结构

- `xiaomachi-llbot`：当前默认平台；WebUI 为 `http://127.0.0.1:3080/`，OneBot WebSocket 为 `ws://127.0.0.1:3002`。
- `xiaomachi-napcat`：保留的回退平台，使用 `ws://127.0.0.1:3001`；不与 LLBot 使用同一个 QQ 号并行运行。
- `xiaomachi-bot`：运行 `python -m app.group_main`。
- `.venv-wsl`：供 keepalive、OneBot 探针和登录 watchdog 使用，不是旧 Windows 虚拟环境。
- `infra/wsl/scripts/onebot_watchdog.py`：主动调用 `get_status` 和 `get_group_list(no_cache=true)`；连续异常时只重启当前 QQ 平台一次，仍需登录时通知 Windows。

## 不能删除的数据

以下内容不进入 Git，但属于当前运行态：

- `infra/wsl/.env`
- `infra/wsl/runtime/llbot/data`：LLBot WebUI 密码、签名令牌、QQ 会话和 OneBot 配置
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

- LLBot 登录页无法完成快速登录或二维码登录：先看 `xiaomachi-llbot` 日志；签名服务不可用时，重试扫码不会恢复，需等待或修复 LLBot 签名服务。
- 容器 healthy 但 OneBot 离线：运行当前平台对应的 WebUI 快捷方式完成登录，再运行 `status-xiaomachi-wsl.bat`。
- WebSocket 持续握手失败：通常是 QQ 未登录或 OneBot 尚未就绪，不代表模型配置失败。
- 修改模型后未生效：重建 `xiaomachi` 容器，并从容器环境确认非敏感变量。
- 登录反复失效：保留 `infra/wsl/runtime/llbot/data`（或 NapCat 的 `runtime/napcat/ntqq`），查看当前平台日志和 watchdog 状态，不要删除登录态目录。

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
