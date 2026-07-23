# 群聊记忆编排 V2 实施计划

## 执行原则

- 所有新行为先添加失败测试，再实现。
- 每个阶段保持 V1 可运行；schema 只做 additive migration。
- 写入型子 agent 只拥有明确且互不重叠的文件范围；主 agent 负责集成、
  router/main wiring、冲突处理和最终验证。
- 任何真实数据操作先在线一致性备份并验证；任何部署命令只指定
  `xiaomachi` 服务。

## 阶段 0：基线和规划门

- [x] 初始化 Trellis 并完成 backend spec bootstrap。
- [x] 核对 main/origin/main、最近提交、dirty state 和目标分支。
- [x] 建立 `codex/memory-orchestration-v2`。
- [x] 运行基线完整 pytest：602 passed。
- [x] 完成 PRD、技术研究、design、implement。
- [ ] 配置 implement/check context 并 `task.py start`。

## 阶段 1：存储、迁移和 Episode（提交 1）

### 测试先行

- [ ] 新/旧库创建 `conversation_episodes`、`episode_messages`、
  `retrieval_documents`。
- [ ] 重复与并发 `create_all` 幂等，原始消息不变。
- [ ] 一条消息只能属于一个 episode；一个群只能有一个 open episode。
- [ ] 组合 FK 拒绝跨群 episode/message/document/source 关联；各召回通道
  在其他群存在更高分同向量/同昵称时仍泄漏 0。
- [ ] coalescing job 覆盖 completed 后新消息、running 中新消息、完成 CAS
  竞态、两 worker claim、崩溃/lease 恢复，最终 orphan=0。
- [ ] episode idle/day/count/token 边界、引用链/连续 @bot 延续。
- [ ] 12～24 条、token 上限和约 5 条重叠窗口，引用/回复尽量同窗。
- [ ] retrieval FTS group scope、派生索引不可用降级。

### 实现

- [ ] 扩展 `models.py`、`db.py`、repositories。
- [ ] 新增 `episode_segmenter.py`。
- [ ] 入站/出站原始消息落库后只原子 enqueue/rearm coalescing allocator job；
  worker 完成 episode 分配。
- [ ] 添加 retrieval document、document-message provenance 和 FTS 初始化；
  vec generation 在阶段 2 provider 契约完成后实现。

### 定向验证

```powershell
python -m pytest tests/storage tests/core/test_episode_segmenter.py -q
python -m compileall -q app
```

提交：`feat: add memory episode and retrieval document storage`

## 阶段 2：真实 Embedding 和混合召回（提交 2）

### 测试先行

- [ ] fake local/openai-compatible/disabled provider、维度校验、初始化失败。
- [ ] V2 不调用 hashed embedding；vector unavailable 时只走 FTS。
- [ ] 版本化 vec generation 在并发查询/写入和 build/swap 故障时保留旧
  active；维度/coverage 验证后 CAS 切换。
- [ ] 中文无关键词重合语义候选仍保留。
- [ ] BM25/vector/time/entity/fact/reply 多路 RRF。
- [ ] 引用、人物昵称、时间、多跳、事实更新、群隔离。
- [ ] 命中周边 5～10 条、reply 上游和 bot 回复、source ID 去重。

### 实现

- [ ] 新增 `semantic_embeddings.py`。
- [ ] 新增 `memory_query_resolver.py`、`hybrid_memory_retriever.py`。
- [ ] 实现 query rewrite strict JSON、短超时和回退。
- [ ] worker 批量生成真实 embedding，模型/维度/版本变化安全重建 vec。
- [ ] 同步 FastEmbed 依赖与镜像测试。

### 定向验证

```powershell
python -m pytest tests/providers/test_semantic_embeddings.py tests/core/test_memory_query_resolver.py tests/core/test_hybrid_memory_retriever.py tests/storage -q
```

提交：`feat: add semantic hybrid memory retrieval`

## 阶段 3：后台处理、编排和自适应上下文（提交 3）

### 测试先行

- [ ] worker episode process 成功、失败重试、stale running 恢复和优雅停止。
- [ ] 所有 fact source IDs 真实且旧事实保留/superseded。
- [ ] backfill 中断续跑和逐群覆盖率。
- [ ] normal/detail token 预算、recent 连续后缀、证据段完整顺序、来源去重。
- [ ] QQ blocked note 保留且敏感原文不进入 V2 evidence。
- [ ] blocked 原文仅保留 raw/recent；document/FTS/vector/summary/fact/event、
  provider payload 和 shadow log 全链路不存在，并清理错误旧派生物。
- [ ] shadow 只记安全指标；V2 异常回退 V1 且仍回复。
- [ ] group/private/dev/图片/搜索/OneBot wiring 回归。

### 实现

- [ ] 新增 `memory_context_packer.py`、`memory_orchestrator.py`、
  `memory_backfill.py`。
- [ ] 重构 `MemoryCompactionService` 支持 episode jobs。
- [ ] Router 只通过 `MemoryOrchestrator` 获取记忆上下文。
- [ ] main/group_main 构建 provider/orchestrator/worker，保持 private/dev 隔离。
- [ ] 接入全部配置和 shadow/V1/V2 开关。

### 定向验证

```powershell
python -m pytest tests/core/test_memory_context_packer.py tests/core/test_memory_orchestrator.py tests/core/test_memory_compaction_service.py tests/core/test_router.py tests/test_process_split_smoke.py tests/test_service_smoke.py -q
```

提交：`feat: orchestrate adaptive historical context`

## 阶段 4：回填工具、真实评测和迁移覆盖（提交 4）

- [ ] 新增 `scripts/evaluate_memory_recall.py` 和离线 fake/fixture 测试。
- [ ] JSONL schema 校验；固定 64～100 题分类最低配额、schema/version/SHA-256，
  逐题人工确认 expected evidence，敏感数据集不提交。
- [ ] 输出 V1/V2 recall@10、packed hit、MRR/NDCG、token、latency、rewrite rate。
- [ ] 增加 `memory_backfill` CLI/覆盖率报告：总消息、assigned、documents、
  embeddings、pending/failed、orphans、per-group。
- [ ] 增加 SQLite online backup CLI 和完整性测试。
- [ ] ledger manifest 测试固定 backup 来源、字段顺序、raw text、UTF-8 JSON row、
  8-byte 长度前缀、SHA-256、private bucket 和逐群 watermark。
- [ ] `data/memory_eval/`、模型缓存、备份显式 gitignore。

验证：

```powershell
python -m pytest tests/core/test_memory_backfill.py tests/test_evaluate_memory_recall.py tests/storage -q
python -m compileall -q scripts
```

提交：`test: add memory recall evaluation and migration coverage`

## 阶段 5：配置、文档和部署契约（提交 5）

- [ ] 更新 `.env.example`、README、`infra/wsl/README.md`。
- [ ] 更新 pyproject/requirements，并确保依赖清单测试。
- [ ] Compose 模型缓存仍在现有 `/workspace/data` 卷。
- [ ] 记录 shadow -> backfill/eval -> V2 enable 和即时 V1 回退步骤。

验证：

```powershell
python -m pytest tests/test_config.py tests/test_llbot_deployment.py tests/test_wsl_deployment_artifacts.py -q
docker compose -f infra/wsl/docker-compose.llbot.yml config
```

提交：`docs: document memory orchestration v2`

## 阶段 6：离线总体验证

- [ ] 全量 `python -m pytest -q`。
- [ ] `python -m compileall -q app scripts` 和 import smoke。
- [ ] Compose config 和 Docker image build。
- [ ] 模拟历史库迁移、并发初始化、integrity check。
- [ ] 敏感信息扫描、`git diff --check`、git status。
- [ ] 独立 Sol xhigh reviewer 基于实际 diff/tests 做高风险最终审查；修复后重跑。

## 阶段 7：真实数据备份、迁移、回填和评测

- [ ] 目标 WSL/Docker 只读预检并记录 LLBot container ID/StartedAt。
- [ ] 在运行数据卷用 SQLite backup API 创建 pre-v2 备份并
  `integrity_check=ok`。
- [ ] 备份事务记录每群 `max(message.id)` 水位和水位内规范化不可变行哈希；
  迁移后逐群复核，新消息只计入水位外增量。
- [ ] 以 shadow mode 运行新镜像，仅重建 `xiaomachi-bot`。
- [ ] 完成全量回填；验证原始消息 count/hash、不跨群、无 orphan。
- [ ] 验证 late-arrival 会局部版本化重分段、失效旧派生物且不修改 message。
- [ ] 生成并人工检查 64～100 题真实评测集。
- [ ] V1/V2 指标达到 PRD 阈值；不达标则迭代召回，不能直接启用 V2。
- [ ] 指定 backfill run + snapshot watermark + segmentation/compaction/index
  generation 内 mandatory jobs `pending=running=failed=0`、embedding
  ready/failed=0；水位外实时/shadow job 单独报告。
- [ ] 实际 warm 容器单并发、20 次预热、≥250 次本地检索统计 p95<500ms。

## 阶段 8：启用、实机验收和 Git 交付

- [ ] 启用 V2 并只重建 `xiaomachi-bot`。
- [ ] LLBot ID/StartedAt 不变、登录/healthy、OneBot `get_status` 成功。
- [ ] 授权群消息入库/@回复/“详细讲讲”/“后来呢”/跨天问题通过。
- [ ] 审计日志只含安全 IDs/metrics。
- [ ] 最终完整测试、运行状态、数据库完整性和 git diff/status。
- [ ] 合并分支到 main，无强推；推送 `origin/main`。
- [ ] 归档 Trellis task、更新规范和 session journal。

## 回滚点

- 任何实现阶段：V2 默认关闭或 shadow，V1 继续工作。
- 真实迁移前：必须有已验证 backup 文件名。
- 运行失败：关闭 V2 或回滚旧 bot 镜像，仅重建 `xiaomachi`。
- 不自动恢复数据库；如确需从备份恢复，停止 writer 并另行取得破坏性恢复授权。
# Completion Record (2026-07-23)

- [x] Phases 0-6: implementation, offline tests, migration tools, deployment
  contracts, and independent review completed.
- [x] Phase 7: verified online backup, immutable-watermark backfill, approved
  64-case real evaluation, and 320-run warm benchmark completed.
- [x] Phase 8: V2 enabled in production with CUDA auto selection; only the bot
  container recreated; LLBot identity/login preserved; production health and
  database invariants passed.
- [x] Late-arrival sequential-generation and concurrent stale-episode defects
  fixed with generation-base compatibility, guarded SQL append, full-batch
  chronological retry, and real SQLite concurrency tests.
- [x] Final quality gate: 773 tests passed, compileall passed, diff check passed.
