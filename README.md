# 小町 QQ AI Bot（WSL/Docker）

当前唯一受支持的部署方式是 WSL2 + Docker。默认 QQ 平台为 LLBot；NapCat 保留为本地回退选项。两者共享小町 Python 进程、数据库、模型配置和业务代码，但 QQ 登录态彼此独立。

完整的系统架构、消息工作流、商用 API、QQ/OneBot 服务和 Memory V3 原理见 [小町工程设计与运行原理](docs/ARCHITECTURE.md)。

## 核心优势

### 一个真正“记得住”的群聊 AI

- **Memory V3 分层记忆**：最近消息、episode 摘要、结构化事实、用户画像、群公共上下文
  分层编排。不是“把聊天记录塞进向量库再碰运气”，而是让模型带着工具主动找上下文。
- **原生记忆工具**：`memory_search` / `memory_read` / `memory_write`——模型可以按需检索
  原文、读取某成员画像、写入有出处的记忆；写入必须绑定当前群、当前会话的真实消息，
  全量审计。
- **每条记忆都有出处**：事实、画像、摘要全部绑定真实消息 source，可纠正、可撤回、
  可追溯；跨群内容严格隔离，答不出就明确说证据不足，绝不编造。

### 问什么都能接住

- **问法族统一处理**：首人称（我喜欢/我喜欢看/我想看）、自称原话（我什么时候说过/哪条）、
  评价（觉得/怎么看/如何评价）、引用消息代词（他/她=被引用消息发送人）、成员昵称 vs
  QQ 号等，口语、错字、倒装变体都能命中同一份事实。
- **意图类型 + 语义 + 字面 + 时效的综合排序**：按问题意图优先命中正确类型的事实
  （喜欢、讨厌、梗、关系、计划、决定、近况、画像），语义相似度兜底，重要性/置信度/
  时效决胜。
- **持久向量 + CUDA**：事实向量预计算并持久化，冷启动零重算、排序稳定；本地模型在
  GPU 上推理，不依赖外部付费 reranker。

### 按群定制，隐私可控

- 默认群只使用最近 100 条消息作为上下文，不生成、不检索、不保存任何记忆；
  完整分层记忆只在显式开启的群生效。
- 真实群号等部署配置放在 gitignore 的本地覆盖文件中，公开仓库只保留占位符。

### 安全、可靠、可运维

- QQ 投递被拦截时自动降级，不泄露敏感细节；联网检索工具事件留痕可核验。
- 心跳、指标、`memory_runtime` 日志、LLBot 登录态保护：发布只重建 `xiaomachi`，
  绝不重启 LLBot。
- 备份、向量回填、噪音清理、按群数据清理全部幂等且先备份，原始消息永不删除。
- WSL2 + Docker 一键部署，GPU 只分配给 bot 容器；开发与运维命令齐全。

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
   LLM_MODEL=gpt-5.6-luna
   LLM_TEXT_ENDPOINT=responses
   LLM_REASONING_EFFORT=high
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
  `xiaomachi`。模板默认 `auto`：有 NVIDIA GPU 自动用 CUDA，没有则自动回退 CPU，
  无 CUDA 机器可直接部署运行。

无 NVIDIA 环境部署：保持 `.env` 中 `ENABLE_GPU=0`、`MEMORY_EMBEDDING_DEVICE=auto`、
`MEMORY_EMBEDDING_LOCAL_FILES_ONLY=false`（首次运行需联网下载嵌入模型）即可；
不需要安装 NVIDIA Container Toolkit，也不需要改动 Compose。有 NVIDIA 的机器设
`ENABLE_GPU=1` 启用 `docker-compose.gpu.yml`，并把
`MEMORY_EMBEDDING_DEVICE=cuda`、模型缓存后 `MEMORY_EMBEDDING_LOCAL_FILES_ONLY=true`。

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

### 记忆系统（Memory V3）发布与回滚

Memory V3 是生产启用的历史查询路径（当前生产 `MEMORY_RAW_V3_ENABLED=true`，
分层/记忆工具/语义排序/改写均开启，运行时日志 `route=raw_v3`）。
`.env.example` 已按生产模板全部开启；代码默认值保持安全关闭，避免未配置环境误启用。
完整的 V3 准备、评测、激活与回滚流程见
[Memory V3 运维清单](infra/wsl/README.md#memory-v3-prepare-evaluate-activate-and-rollback)。

发布只允许重建 `xiaomachi` service（容器名 `xiaomachi-bot`）：

```bash
cd "/mnt/d/qq群ai小人/infra/wsl"
# 无 NVIDIA 机器（默认）
docker compose -f docker-compose.llbot.yml up -d --no-deps --force-recreate xiaomachi
# 有 NVIDIA 机器（ENABLE_GPU=1）
docker compose -f docker-compose.llbot.yml -f docker-compose.gpu.yml up -d --no-deps --force-recreate xiaomachi
```

**不得重建或重启 `xiaomachi-llbot`**，也不得删除其登录态。
V3 运行仍要求 `MEMORY_ORCHESTRATION_V2_ENABLED=true`，但它只是 V2 兼容开关，
不是 V3 回滚开关；
若向量通道异常，设置 `MEMORY_EMBEDDING_PROVIDER=disabled` 保留 FTS。
生产镜像使用 CUDA 12.8、cuDNN 与 `fastembed-gpu`；`ENABLE_GPU=1` 时通过
`docker-compose.gpu.yml` 挂载 `nvidia.com/gpu=all` CDI 给 `xiaomachi`，
`MEMORY_EMBEDDING_DEVICE=cuda` 仅在启用 GPU 的生产使用；无 GPU 机器保持
`auto` 自动回退 CPU。模型缓存后设置 `MEMORY_EMBEDDING_LOCAL_FILES_ONLY=true`
可离线启动。V2 为已退役兼容路径，旧迁移与评测命令不再需要。

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

## 记忆测试平台（Memory Test Platform）

对 Memory V3 做全链条评估：解析 → 检索 → 打包 → 上游模型真实请求
（生成回答 + 引用校验 + 模型判定）。统一驱动：

```bash
# 0) 前置：容器内生产快照（只读副本）+ 上游模型 API key（环境变量）
# 1) 一次性全流程（本地冒烟：--count 小值 + --dry-run）
python -m scripts.run_memory_test_suite --database data/bot.db \
  --count 100 --fullchain-limit 20 --dry-run

# 2) 正式运行（离线 ≥3000 例 + 全链条 300 例，先 dry-run 看预算）
python -m scripts.run_memory_test_suite --database /tmp/snapshot.db \
  --count 3000 --fullchain-limit 300 --dry-run
python -m scripts.run_memory_test_suite --database /tmp/snapshot.db --all

# 模型配置：最终回答 Luna medium，judge/修复/改写 Luna low（默认）
python -m scripts.run_memory_test_suite --database /tmp/snapshot.db \
  --answer-model gpt-5.6-luna --answer-effort medium \
  --aux-model gpt-5.6-luna --aux-effort low --stage fullchain

# 3) 分阶段 + 断点 + 缓存
python -m scripts.run_memory_test_suite --database /tmp/snapshot.db --stage fullchain --resume

# 4) 基线对比与门禁（--baseline-dir 指向旧报告目录）
python -m scripts.run_memory_test_suite --database /tmp/snapshot.db --stage report \
  --baseline-dir data/test-platform-baseline \
  --gate-grounded-accuracy 0.7 --gate-recall 0.6 --gate-protocol-failures 5
```

阶段：`prepare`（只读快照复制 + integrity/FTS 校验）、`dataset`（分层生成
≥3000 例）、`offline`（全量离线全链路，零模型成本）、`fullchain`
（300 例真实模型，响应按 prompt 哈希缓存）、`stress`（复用 300 例压力）、
`report`（指标聚合/基线 diff/门禁）。产物在 `data/test-platform/`
（已 gitignore）：`cases.jsonl`、`offline-results.jsonl`、
`fullchain-results.jsonl`、`report.json/md`、私有明细与缓存。

隐私：公共报告只有聚合数字；prompt、模型原文、judge 原文只进本地私有明细，
不提交 Git。运行前请确认 API key 与成本预算（`--dry-run` 会先给估算）。
