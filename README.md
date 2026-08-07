# 小町 QQ AI Bot（WSL/Docker）

当前唯一受支持的部署方式是 WSL2 + Docker。默认 QQ 平台为 LLBot；NapCat 保留为本地回退选项。两者共享小町 Python 进程、数据库、模型配置和业务代码，但 QQ 登录态彼此独立。

完整的系统架构、消息工作流、商用 API、QQ/OneBot 服务和 Memory V3 原理见 [小町工程设计与运行原理](docs/ARCHITECTURE.md)。

## 功能亮点

- **人格化群聊**：小町人格、@/主动回复、群策略、安全规则与 QQ 拦截降级，普通群聊与开发控制台共用一套运行时。
- **Memory V3 分层记忆**：最近消息、episode 摘要、结构化事实、用户画像、群公共上下文分层编排；模型可调用原生记忆工具
  `memory_search` / `memory_read` / `memory_write` 主动取上下文，而不是依赖单一 RAG 召回碰运气。
- **语义排序 + 持久向量**：本地 bge-small-zh 在 CUDA 上运行，事实向量预计算并持久化，冷启动零重算、排序稳定。
- **问法鲁棒性**：首人称（我喜欢/我喜欢看/我想看）、自称原话（我什么时候说过/哪条）、评价（觉得/怎么看/如何评价）、
  引用消息代词（他/她=被引用消息发送人）、成员昵称 vs QQ 号等问法族统一处理，口语与错字变体可命中。
- **按群记忆策略**：默认群只使用最近 100 条消息作为上下文；完整分层记忆仅在明确开启的群生效（真实群号由本地的 `configs/groups.local.yaml` 配置，不提交仓库）。
- **数据治理**：每条事实绑定真实消息 source、可纠正/撤回；历史噪音清理、向量回填、按群数据清理脚本均幂等且先备份。
- **可观测与安全**：`memory_runtime`/指标/心跳日志，跨群隔离 fail-closed，敏感投递自动降级不泄露。

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

- `configs/groups.yaml`：控制群是否接收、发言、主动回复、归档、生图和**记忆系统**。
- `configs/persona.yaml`：人格、称呼和回复风格。
- `configs/safety.yaml`：安全限制。

群配置只有同时设置 `enabled: true` 和 `speak: true` 才允许小町在该群回复。
记忆按群开关：`memory_enabled: false` 的群（默认）只使用最近
`recent_context_limit`（默认 100）条消息作为上下文，不生成、不检索、不产生任何记忆数据；
`memory_enabled: true` 的群启用完整分层记忆；真实群号在本地
`configs/groups.local.yaml` 中配置（该文件不提交仓库），仓库中的
`configs/groups.yaml` 仅保留占位群号。

### 记忆系统（Memory V3）

- `MEMORY_LAYERED_MEMORY_ENABLED=true`：查询侧接通 episode 摘要、结构化事实与用户画像；
  `MEMORY_MEMORY_TOOLS_ENABLED=true`：启用模型原生记忆工具（带 source 约束与审计）。
- `MEMORY_FACT_SEMANTIC_RANKING_ENABLED=true`：事实按“问法意图类型 + 语义相似度 + 字面匹配 +
  重要性/置信度/时效”排序，目标类型事实进入上下文；向量持久化于
  `memory_item_semantic_vectors`，由后台与回填脚本写入。
- `MEMORY_QUERY_REWRITE_ENABLED=true`：确定性绑定失败时做一次受约束改写，不改宽时间范围、
  不虚构成员。
- `MEMORY_EMBEDDING_DEVICE=cuda`：生产使用 CUDA 推理，Docker 通过 CDI 只把 GPU 分配给
  `xiaomachi`。

维护命令（均在容器内执行，先备份）：

```bash
# 事实向量全量/增量回填
python -m scripts.backfill_memory_item_semantic_vectors --database /workspace/data/bot.db --batch-size 100
# 历史噪音清理（plan 先看候选，run 再执行，可恢复）
python -m scripts.cleanup_memory_noise plan --database /workspace/data/bot.db
python -m scripts.cleanup_memory_noise run --database /workspace/data/bot.db
# 关闭记忆的群：删除全部记忆派生数据（原始消息保留）
python -m scripts.purge_group_memory --database /workspace/data/bot.db --group-id 100000002 --dry-run
```

### 文本、搜索和上下文

常用环境变量位于 `infra/wsl/.env.example`：

- `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`
- `LLM_TEXT_ENDPOINT`、`LLM_REASONING_EFFORT`
- `LLM_BUILTIN_WEB_SEARCH`、`LLM_BUILTIN_WEB_SEARCH_CONTEXT_SIZE`
- `SEARCH_PROVIDER`、`SEARCH_API_KEY`、`SEARCH_BASE_URL`
- `CONTEXT_RECENT_LIMIT`（默认 60 条近期实际消息）、`CONTEXT_SUMMARY_LIMIT`、`CONTEXT_HISTORY_LIMIT`；带有旧话题承接、人物指代或时间回顾语义的问题会自动扩大详细历史回溯。
- `MEMORY_ADAPTIVE_CONTEXT_ENABLED=true` 让 Memory V3 在有效输入 token 总预算内动态分配近期与历史空间；`MEMORY_ADAPTIVE_MAX_RECENT_MESSAGES=120` 和 `MEMORY_ADAPTIVE_MAX_HISTORY_MESSAGES=300` 只是应急行数上限，不代表每次固定注入这些消息。
- `MEMORY_COMPACTION_ENABLED`、`MEMORY_COMPACTION_BATCH_SIZE`、`MEMORY_COMPACTION_BACKFILL_WINDOWS`
- `MEMORY_EPISODE_IDLE_MINUTES`（默认 10 分钟）控制普通语义聊天停止多久后封闭 episode 并进入长期记忆提炼；它与 `CONTEXT_RECENT_LIMIT` 相互独立。
- 群聊生图使用独立的 OpenAI 兼容图片服务配置；模型、接口、队列和超时见 `GROUP_IMAGE_*` 环境变量

`LLM_TEXT_ENDPOINT=responses` 且 `LLM_BUILTIN_WEB_SEARCH=true` 时，文本请求可使用主模型的内置 `web_search` 工具。明确写出“联网”“搜索”“查资料”等请求会强制调用检索；普通聊天则由模型自行决定是否检索。实际工具调用会记录在未纳入 Git 的 `infra/wsl/runtime/logs/responses-tool-events.jsonl`，用于核验是否真的联网。

当 LLBot 连续返回 `retcode=1200 / waitForSelfEcho timeout` 时，小町会把原回复标记为“QQ 拦截、未送达”并保留在本地上下文，同时在群内改发固定的失败提示。后续模型可以理解原回复，但会收到不得复述其中敏感细节的明确约束；这类记录不会进入自动摘要或长期记忆压缩。

修改 `infra/wsl/.env` 后，需要重建小町容器才能加载新环境变量：

```bash
cd "/mnt/d/qq群ai小人/infra/wsl"
docker compose -f docker-compose.llbot.yml up -d --no-deps --force-recreate xiaomachi
```

### 群聊记忆编排 V2 灰度与回滚

> Memory V3 是生产启用的历史查询路径（当前生产 `MEMORY_RAW_V3_ENABLED=true`，
> 分层/记忆工具/语义排序/改写均开启，运行时日志 `route=raw_v3`）。
> `.env.example` 已按生产模板全部开启；代码默认值保持安全关闭，避免未配置环境误启用。
> 生产部署在 `.env` 中显式打开，需先完成发布门禁（备份、回填、评测、激活）。以下 V2 内容用于理解兼容路径和底层
> generation；新的生产发布、评测与回滚以 [Memory V3 运维清单](infra/wsl/README.md#memory-v3-prepare-evaluate-activate-and-rollback) 为准。V3 运行仍要求
> `MEMORY_ORCHESTRATION_V2_ENABLED=true`，不要把它作为 V3 回滚开关。

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
docker compose -f docker-compose.llbot.yml build xiaomachi
docker compose -f docker-compose.llbot.yml up -d --no-deps --force-recreate xiaomachi
```

**must not restart xiaomachi-llbot**：不得重建或重启 LLBot，也不得删除其登录态。若 V2
出现故障，立即设置 `MEMORY_ORCHESTRATION_V2_ENABLED=false` 回到 V1；若仅向量通道有问题，
设置 `MEMORY_EMBEDDING_PROVIDER=disabled` 保留 FTS。正常回滚不恢复数据库；只有确认数据
损坏并获得单独授权时，才可从已验证 backup 恢复。

### 群聊记忆 V2 迁移与评测

> 本节保留用于 legacy V2 数据维护，不是 Memory V3 的发布入口。

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
  --paraphrase-overrides /workspace/data/memory_eval/paraphrase-overrides.json \
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
