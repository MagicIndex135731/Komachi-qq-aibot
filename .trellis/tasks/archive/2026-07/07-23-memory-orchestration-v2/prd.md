# 群聊记忆编排 V2

## Goal

将当前依赖“最近消息、扁平摘要、关键词和哈希词特征”的群聊历史上下文，
升级为可灰度、可回填、可回退的群聊记忆编排系统：

永久原始消息账本 + 会话 episode + 重叠证据窗口 + 结构化事实/事件 +
真实语义向量 + FTS/BM25 混合召回 + 时间/人物/引用检索 +
命中周边展开 + 动态 token 装箱。

最终成果必须在真实 WSL/LLBot 运行环境完成数据备份、迁移、评测、部署和
消息链路验收，并提交、合并到 `main`、推送到 `origin/main`。

## Background

当前实现存在以下用户可见缺陷：

- 压缩固定按 50 条处理，不理解会话边界。
- 历史检索依赖词面和中文双字词，语义改写命中差。
- 所谓向量是哈希词特征，且语义候选会被关键词硬过滤丢弃。
- 历史命中是零散消息，缺少引用链和前后语境。
- 上下文虽有 token 预算，但候选仍主要由固定条数控制。

基线为 `main == origin/main` 的 `b5b9dbe`，本任务分支为
`codex/memory-orchestration-v2`。

## Requirements

### R1. 数据安全与兼容性

- 永久保留所有现有 `messages` 原始记录；保留 `memory_items`、
  `summaries`、`jobs` 和 V1 行为。
- SQLite 仍是主数据库；派生摘要、FTS 和向量索引不得成为事实源。
- 所有查询、回填、索引和上下文严格按群隔离。
- 保持 LLBot、OneBot、图片、联网、私聊、开发控制和 QQ 拦截回复逻辑。
- 现有配置继续可用；V2 可通过开关立即回退到 V1。
- 不提交或输出 `.env`、密钥、token、数据库和模型缓存。

### R2. Episode 与检索文档存储

- 新增 `conversation_episodes`，记录群、起止时间/消息、状态、边界原因、
  标题、摘要、消息/token 数、内容哈希和压缩版本。
- 新增 `episode_messages`，记录 episode、消息和 ordinal；一条群消息只能
  属于一个主 episode。
- 新增 `retrieval_documents`，记录 scope、文档类型、来源、时间范围、内容、
  元数据、内容哈希、embedding 模型/版本和状态；来源与版本组合唯一。
- 建立 FTS5 trigram 与 sqlite-vec 派生索引。
- 迁移幂等并支持多个入口并发启动；不得重建或删除 `bot.db`。

### R3. 真实 Embedding

- 提供 `EmbeddingProvider` 抽象，支持 `local`、
  `openai_compatible`、`disabled`。
- 默认本地模型为 `BAAI/bge-small-zh-v1.5`，FastEmbed/ONNX，
  512 维，缓存目录 `/workspace/data/models` 且位于持久卷。
- 依赖在镜像构建期安装；运行时不重复安装。模型只在首次缺失时下载。
- 模型下载/初始化/维度不兼容时禁用向量通道并降级 FTS，不阻止启动或回复。
- V2 不得使用 `hashed_text_embedding` 冒充语义向量；模型、维度或版本变化
  可安全重建派生向量索引。

### R4. Episode 边界与重叠窗口

- 按群维护开放 episode；默认边界为静默 30 分钟、跨天、50 条消息、
  8000 估算 token。
- 强引用/回复链与连续 @小町 问答尽量保持完整。
- episode 关闭后异步生成 12～24 条、最多 1800 token、约 5 条重叠的窗口。
- 引用与回复尽量同窗；窗口按时间排序并保存完整 `source_msg_ids`。
- 每条消息主路径不得调用 LLM 或执行批量 embedding。

### R5. 后台记忆处理

- 重构 `MemoryCompactionService`/`jobs`：消息主路径只持久化并唤醒/排队。
- episode 关闭后异步生成窗口 embedding、episode 摘要、事实、事件和每日
  主题摘要。
- 每个事实必须有真实 `source_msg_ids`；支持 `valid_from`、
  `valid_until`、`superseded_by_id`；旧事实保留。
- 任务幂等、有限重试、可断点回填；worker 故障不影响回复；关闭时优雅停止。

### R6. 查询理解

- 提供 `ResolvedMemoryQuery`：原问题、解析后问题、实体、说话人、时间范围、
  检索模式、是否需要历史、是否需要细节。
- 结合当前消息、引用/被回复的小町消息、最近 6～12 条和最近明确实体解析
  “详细讲讲/后来呢/之前那个/那个人/他说了什么/最后怎么样”等追问。
- 明确问题走确定性规则；仅模糊历史追问允许一次低输出 LLM 改写。
- 改写必须严格 JSON、有限超时、失败/坏 JSON 回退原问题且不阻塞回复。

### R7. 混合检索与周边展开

- 并行召回 FTS/BM25、真实向量、时间、人物/实体、结构化事实、引用关系，
  每路默认 20～40 候选。
- 使用 RRF 或等价方法融合；不得用“关键词重合为 0”硬过滤语义候选。
- 精确引用/实体权重最高，语义和 BM25 互补，时间/近期性只加权。
- 命中后定位 episode/窗口，展开前后 5～10 条，并加入引用上游和小町回复。
- 与近期消息按来源消息 ID 去重；episode 内按时间排序；禁止跨群。

### R8. 动态上下文装箱

- 正常历史预算默认约 32000 token：最近连续消息 8k～12k、事实 2k～4k、
  episode 证据 8k～16k、上层摘要 2k～4k。
- 详细回溯默认约 64000 token，最多 4～6 个完整证据片段，不常态填满
  258k 上下文。
- 系统/安全/目标消息优先；最近消息必须是连续后缀；证据按相关性选片段、
  片段内按时间排序并按来源 ID 去重。
- 历史证据带时间、说话人、episode、消息来源，并标记为不可信引用数据。
- 保留 `QQ_BLOCKED_CONTEXT_NOTE` 和禁止复述敏感细节规则。
- 无足够证据时不得用无关最新摘要伪装历史依据。

### R9. 编排边界

- `router.py` 仅通过 `MemoryOrchestrator` 获取最终记忆上下文。
- 查询解析、混合召回、周边展开、装箱、回填、embedding provider 各有清晰
  模块边界并可离线测试。
- V2 任意异常均回退 V1/近期上下文，不阻止正常群聊回复。

### R10. 配置与运行文档

至少新增并接入以下向后兼容设置：

`MEMORY_ORCHESTRATION_V2_ENABLED`,
`MEMORY_ORCHESTRATION_SHADOW_MODE`,
`MEMORY_EMBEDDING_PROVIDER`,
`MEMORY_EMBEDDING_MODEL`,
`MEMORY_EMBEDDING_DIMENSIONS`,
`MEMORY_EMBEDDING_CACHE_DIR`,
`MEMORY_EPISODE_IDLE_MINUTES`,
`MEMORY_EPISODE_MAX_MESSAGES`,
`MEMORY_EPISODE_MAX_TOKENS`,
`MEMORY_CHUNK_MAX_TOKENS`,
`MEMORY_CHUNK_OVERLAP_MESSAGES`,
`MEMORY_QUERY_REWRITE_ENABLED`,
`MEMORY_LLM_RERANK_ENABLED`,
`MEMORY_NORMAL_CONTEXT_BUDGET_TOKENS`,
`MEMORY_DETAIL_CONTEXT_BUDGET_TOKENS`,
`MEMORY_RECENT_CONTEXT_BUDGET_TOKENS`,
`MEMORY_FTS_CANDIDATE_LIMIT`,
`MEMORY_VECTOR_CANDIDATE_LIMIT`,
`MEMORY_FINAL_EPISODE_LIMIT`。

同步更新 `AppSettings`、`infra/wsl/.env.example`、README 和 WSL 运行文档。

### R11. 灰度、回填与备份

- 初始启用 shadow mode：V1 构造实际提示，V2 执行分段/索引/召回，仅记录
  证据 ID、分数、token、耗时和错误类别。
- 对全部现有群消息幂等回填，并输出总消息、episode 已分配消息、索引文档、
  embedding 覆盖率、failed/pending jobs、孤立消息和逐群统计。
- 回填和评测达标后启用 V2；保留 V1 开关与旧表。
- 迁移真实库前使用 SQLite 一致性 backup API 备份有效 WAL 状态，记录位置，
  并验证备份可打开且 `PRAGMA integrity_check = ok`。
- 部署只重建/重启 `xiaomachi-bot`，不得重启 `xiaomachi-llbot` 或破坏 QQ 登录。

### R12. 测试

- 新行为先写失败测试；所有外部 embedding/LLM/OneBot 使用 fake/stub，
  单元测试离线。
- 覆盖 episode 边界/回复链、重叠窗口、中文关键词/语义改写、模糊追问、
  引用、人物/昵称、时间、多跳、事实失效、周边展开、QQ 拦截、群隔离、
  向量降级、改写超时/坏 JSON、worker 重试、回填恢复、幂等/并发迁移、
  token 预算、来源 ID 去重和原始消息不变。

### R13. 真实评测

- 提供 JSONL 格式：`group_id`、`query`、`recent_context_message_ids`、
  `expected_evidence_message_ids`、`category`、可选时间范围。
- 从真实数据库人工/半自动构建至少 50～100 题，覆盖 exact、paraphrase、
  vague_reference、temporal、multi_hop、update、abstention。
- 对比 V1/V2 的 evidence recall@k、最终装入证据命中率、context token、
  检索耗时、查询改写调用率；MRR/NDCG 可选。

### R14. Git 与交付

- 分阶段提交存储、混合检索、上下文编排、测试评测和文档。
- 完成全部自动化与真实运行验收后合并回 `main` 并推送
  `https://github.com/MagicIndex135731/Komachi-qq-aibot` 的
  `origin/main`；不得强推。

## Acceptance Criteria

- [ ] AC1：从已验证 backup 文件生成 versioned ledger manifest；按
  `(id, platform_msg_id, group_id, user_id, timestamp raw text, raw_json raw text,
  plain_text, msg_type, reply_to_msg_id, mentioned_bot)`、ID 顺序、长度前缀 UTF-8
  canonical JSON row 计算逐群/private bucket SHA-256。各 snapshot 水位内
  count/digest 迁移前后完全一致，丢失/修改 0；水位外新增单独统计；跨群泄漏 0。
- [ ] AC2：新 schema 在新库、历史库、重复初始化和并发初始化下安全通过。
- [ ] AC3：全部既有测试与新增测试通过，Python 语法/导入检查通过。
- [ ] AC4：关键词问题 evidence recall@10 ≥ 95%。
- [ ] AC5：模糊指代/改写 recall@10 ≥ 80%，或较 V1 提升 ≥ 25 个百分点。
- [ ] AC6：普通历史上下文约 35k token 内，详细回溯约 70k token 内。
- [ ] AC7：真实 warm 容器/真实回填库、单并发、20 次预热、至少 250 次
  resolve+local retrieval+expansion+pack（不含 LLM/network）p95 < 500 ms。
- [ ] AC8：向量不可用、worker/查询改写异常均不阻止正常群聊回复。
- [ ] AC9：真实库在线一致性备份和备份/部署后 `integrity_check` 均为 `ok`。
- [ ] AC10：启用 V2 前，指定 backfill run + 逐群 snapshot watermark +
  segmentation/compaction/index generation 边界内无孤立群消息，mandatory jobs
  pending/running/failed 均为 0，embedding ready 且 failed=0；水位外实时 job
  单独报告，逐群冻结统计一致。
- [ ] AC11：Compose config、Docker 镜像构建和完整 pytest 通过。
- [ ] AC12：部署后 `xiaomachi-bot` 正常，`xiaomachi-llbot` 未重启且保持
  登录/healthy，OneBot `get_status` 成功。
- [ ] AC13：真实群消息正常入库，@小町正常回复；“详细讲讲”“后来呢”和
  跨天历史问题通过手工验收。
- [ ] AC14：V2 日志只含 episode/证据 ID、分数、token、耗时和安全错误类别，
  无完整敏感聊天、prompt 或密钥。
- [ ] AC15：V1 开关可即时回退，QQ 拦截、图片、联网、私聊和开发控制回归通过。
- [ ] AC16：分阶段提交完成，合并到 `main` 并成功推送 `origin/main`，无强推。

## Out of Scope

- 删除 V1 表、旧摘要或原始聊天记录。
- 将主数据库迁移到 SQLite 以外的服务。
- 在单元测试中依赖公网模型、真实 LLM 或真实 QQ。
- 重启或重建 `xiaomachi-llbot`。
