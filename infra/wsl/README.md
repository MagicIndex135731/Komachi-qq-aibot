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

LLBot 返回 `retcode=1200 / waitForSelfEcho timeout`、等待回执超时或发送过程中断线时，
系统会将本次投递标记为“结果未确认”。由于 QQ 可能已经收到消息，机器人不会自动重试、
切片补发或发送额外失败提示，以避免同一回复重复出现。该记录保留在近期上下文中以维持
对话连续性，但不参与自动摘要和长期记忆压缩；同一入站消息重放时也不会再次生成回复。

## 运行态保护

不要删除：

- `.env`
- `runtime/llbot/data`
- `runtime/napcat/ntqq`
- `runtime/napcat/config`
- `runtime/logs`
- `runtime/onebot-watchdog.json`

`runtime/pip-cache` 可以重建，但保留它能显著缩短容器重建时间。

## 群聊记忆编排 V2 发布清单

> Memory V3 是通过发布门禁后启用的生产历史查询路径；仓库模板仍安全默认关闭。本节保留为 legacy V2 兼容与底层 generation 资料；新发布应直接使用下方的 Memory V3 流程。V3 运行仍要求 `MEMORY_ORCHESTRATION_V2_ENABLED=true`，不要把它作为 V3 回滚开关。

`.env.example` 的 `MEMORY_*` 示例保持
`MEMORY_ORCHESTRATION_V2_ENABLED=true` 和
`MEMORY_ORCHESTRATION_SHADOW_MODE=true`。The required rollout order is:
**shadow -> backfill -> evaluate -> active**。本地 FastEmbed 缓存使用既有持久卷的
`/workspace/data/models`；模型初始化或向量服务失败时只保留 FTS/V1，不能阻塞回复或直接启用 V2。

发布前使用 SQLite backup API 创建并验证 `integrity_check=ok` 的备份，随后完成幂等回填，
记录 run ID、每群 watermark、episode/document/embedding 覆盖率和失败/待处理 job。仅当回填
边界内 mandatory jobs 已清空且 V1/V2 评测通过，才将
`MEMORY_ORCHESTRATION_SHADOW_MODE=false` 用于 active 灰度。

部署只构建和重建 `xiaomachi` service（容器名 `xiaomachi-bot`）：

```bash
docker compose -f docker-compose.llbot.yml build xiaomachi
docker compose -f docker-compose.llbot.yml up -d --no-deps --force-recreate xiaomachi
```

Before and after this operation, record the `xiaomachi-llbot` container ID and
`StartedAt`; **must not restart xiaomachi-llbot**. V2 的即时行为回滚为
`MEMORY_ORCHESTRATION_V2_ENABLED=false`；仅向量回滚为
`MEMORY_EMBEDDING_PROVIDER=disabled`，保留 FTS。普通回滚不恢复数据库，也不得删除
LLBot 登录态。

### Memory V2 数据操作

所有命令都在新镜像或仓库根目录执行，目标只能是 bot 数据卷中的
`/workspace/data/bot.db`。先运行 `scripts/backup_memory_v2.py` 创建在线备份和逐群水位账本，
再运行 `scripts/backfill_memory_v2.py`；回填报告必须显示 mandatory jobs 全部终态、无
orphan、无 blocked 派生物、无 embedding failure。随后用
`scripts/build_memory_eval_dataset.py` 生成 gitignored 的 64 题真实数据集。逐题核验
evidence 后，将 hash 绑定的 review sidecar 标记为 approved，再以
`scripts/run_memory_recall_eval.py --review PATH --backfill-run-key ID
--warmup 20 --benchmark-runs 250
--enforce-real-dataset` 完成 V1/V2 指标与本地检索 p95。完整参数示例见仓库根
`README.md`。

只有上述检查通过后才把 shadow 切换为 active；任何阶段均不得重建
`xiaomachi-llbot`，也不得读取、打印或覆盖 `.env` 全文。

### CUDA 向量加速

`xiaomachi` 镜像使用 CUDA 12.8、cuDNN 与 `fastembed-gpu`，Compose 只向 bot 服务挂载
`nvidia.com/gpu=all` CDI 设备。设置 `MEMORY_EMBEDDING_DEVICE=auto` 后优先使用 NVIDIA
GPU，并在 CUDA 推理异常时回退 CPU；LLBot 不申请 GPU。主机需安装 NVIDIA Container
Toolkit，并确保 `nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` 已生成设备规范。
可用 `docker run --rm --device nvidia.com/gpu=all ...` 验证透传。
确认模型已经缓存后设置 `MEMORY_EMBEDDING_LOCAL_FILES_ONLY=true`，可保证离线重启不会
等待模型站点超时。

## 验收

```bash
docker compose config --quiet
docker compose ps
bash scripts/status.sh
```

正常在线时应看到当前 QQ 平台 healthy、OneBot `online=true`、主动群列表探针成功，以及新鲜的小町心跳。若 QQ 本身已离线，先完成对应 WebUI 登录，再重复状态检查。

### Memory V3 prepare, evaluate, activate, and rollback

V3 rollout is deliberately split into separate fail-closed phases. Preparing a
generation never changes the active vector generation:

```bash
python -m scripts.backfill_memory_v3_raw \
  --phase prepare \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --output /workspace/data/backups/memory-v3-prepared.json
```

Build the real snapshot dataset and review every frozen evidence source. The
generated review sidecar starts with `approved=false`; do not activate until a
human reviewer has approved every case. The review bundle contains private chat
content: keep it under `/workspace/data/backups`, never commit or upload it.

```bash
python -m scripts.build_memory_eval_dataset \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --output /workspace/data/backups/memory-v3-cases.jsonl \
  --review-output /workspace/data/backups/memory-v3-review.json

python -m scripts.export_memory_eval_review_bundle \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --dataset /workspace/data/backups/memory-v3-cases.jsonl \
  --output /workspace/data/backups/memory-v3-review-bundle.json
```

After the review sidecar is approved, run the evaluator once without a quality
sidecar to freeze retrieval results and generate a retrieval-bound quality
template. This command is expected to exit with the missing-quality gate:

```bash
python -m scripts.run_memory_recall_eval \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --prepared-report /workspace/data/backups/memory-v3-prepared.json \
  --dataset /workspace/data/backups/memory-v3-cases.jsonl \
  --review /workspace/data/backups/memory-v3-review.json \
  --quality-template-output /workspace/data/backups/memory-v3-quality.json \
  --results-output /workspace/data/backups/memory-v3-results.jsonl \
  --report-output /workspace/data/backups/memory-v3-gate-draft.json \
  --benchmark-output /workspace/data/backups/memory-v3-benchmark-draft.json \
  --warmup 20 --benchmark-runs 250
```

Fill the template only from a controlled GPT answer replay and at least 20 real
index-visibility samples. Then rerun the V3 evaluator against the prepared,
non-active generation. Its passing report is bound to the manifest, dataset,
retrieval fingerprint, exact quality-sidecar digest, and `vector_generation`:

```bash
python -m scripts.run_memory_v3_quality_replay \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --prepared-report /workspace/data/backups/memory-v3-prepared.json \
  --dataset /workspace/data/backups/memory-v3-cases.jsonl \
  --review /workspace/data/backups/memory-v3-review.json \
  --quality-output /workspace/data/backups/memory-v3-quality.json \
  --private-replay-output /workspace/data/backups/memory-v3-quality-private.json \
  --visibility-output /workspace/data/backups/memory-v3-quality-visibility.json \
  --visibility-samples 20
```

After that controlled replay completes, run the final evaluator:

```bash
python -m scripts.run_memory_recall_eval \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --prepared-report /workspace/data/backups/memory-v3-prepared.json \
  --dataset /workspace/data/backups/memory-v3-cases.jsonl \
  --review /workspace/data/backups/memory-v3-review.json \
  --quality-sidecar /workspace/data/backups/memory-v3-quality.json \
  --quality-private-replay /workspace/data/backups/memory-v3-quality-private.json \
  --quality-visibility-artifact /workspace/data/backups/memory-v3-quality-visibility.json \
  --results-output /workspace/data/backups/memory-v3-results.jsonl \
  --report-output /workspace/data/backups/memory-v3-gate.json \
  --benchmark-output /workspace/data/backups/memory-v3-benchmark.json \
  --warmup 20 --benchmark-runs 320
```

Activation requires both the original prepared report and a passing gate
report. It performs final live catch-up and a locked manifest check before the
generation CAS. Immediately after this command succeeds, set
`MEMORY_RAW_V3_ENABLED=true` in `infra/wsl/.env` and recreate only
`xiaomachi`; never recreate LLBot. Production retrieval resolves the active
generation per query, so it does not keep reading the deactivated legacy table
between the CAS and this bounded restart:

```bash
python -m scripts.backfill_memory_v3_raw \
  --phase activate \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --prepared-report /workspace/data/backups/memory-v3-prepared.json \
  --gate-report /workspace/data/backups/memory-v3-gate.json \
  --dataset /workspace/data/backups/memory-v3-cases.jsonl \
  --quality-sidecar /workspace/data/backups/memory-v3-quality.json \
  --quality-private-replay /workspace/data/backups/memory-v3-quality-private.json \
  --quality-visibility-artifact /workspace/data/backups/memory-v3-quality-visibility.json \
  --results /workspace/data/backups/memory-v3-results.jsonl \
  --benchmark-report /workspace/data/backups/memory-v3-benchmark.json \
  --output /workspace/data/backups/memory-v3-activated.json
```

Emergency rollback preserves all raw messages and vector tables and switches
only the active vector generation back to the legacy generation recorded by
prepare. After rollback succeeds, set `MEMORY_RAW_V3_ENABLED=false` and
recreate only `xiaomachi`:

```bash
python -m scripts.backfill_memory_v3_raw \
  --phase rollback \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --prepared-report /workspace/data/backups/memory-v3-prepared.json \
  --output /workspace/data/backups/memory-v3-rollback.json
```
