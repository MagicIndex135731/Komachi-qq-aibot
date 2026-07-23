# 群聊记忆编排 V2 技术设计

## 1. 边界与总览

V2 是现有群聊链路旁路可灰度的记忆子系统，不替换原始 `messages`，也不改变
OneBot、LLBot、图片、联网、私聊或开发控制协议。

```text
OneBot message
  -> Router persists raw Message and idempotently queues episode allocation
  -> reply decision
  -> MemoryOrchestrator
       -> MemoryQueryResolver
       -> HybridMemoryRetriever
       -> episode/reply-neighborhood expansion
       -> MemoryContextPacker
  -> ContextBuilder combines instructions + V1 or V2 memory context
  -> LLM -> Sender -> persist outbound Message -> attach to open episode

MemoryCompactionService (background)
  -> close idle episodes
  -> process episode jobs
  -> build overlap documents + embeddings
  -> update summary/facts/events/daily summary
  -> resumable backfill jobs
```

### 模块职责

- `app/providers/semantic_embeddings.py`：provider 协议、本地 FastEmbed、
  OpenAI-compatible 和 disabled 实现；维度/版本校验与安全降级。
- `app/core/episode_segmenter.py`：纯边界判断、token 估算、重叠窗口切分。
- `app/core/memory_query_resolver.py`：确定性实体/时间/模糊追问解析，可选单次
  严格 JSON 改写。
- `app/core/hybrid_memory_retriever.py`：各路候选、RRF 融合、群隔离、周边展开。
- `app/core/memory_context_packer.py`：连续近期后缀、事实、证据片段、摘要的
  动态预算和来源 ID 去重。
- `app/core/memory_orchestrator.py`：V2 查询侧门面、shadow 指标和异常回退。
- `app/core/legacy_memory_context.py`：从 router 原样迁出的 V1 检索/格式化，
  保证 disabled/fallback 行为等价。
- `app/core/memory_backfill.py`：一致性备份、幂等回填、覆盖率统计。
- `MemoryCompactionService`：保留现有 V1 job 能力，并增加 episode/background
  processing；查询编排不放入 worker。

## 2. 数据模型

### `conversation_episodes`

- `id` INTEGER PK
- `group_id` FK `groups.group_id`，索引
- `started_at`, `ended_at`
- `start_message_id`, `end_message_id`：FK `messages.id`
- `status`：`open|closed|processed`
- `boundary_reason`, `title`, `summary`
- `message_count`, `token_count`
- `content_hash`, `compaction_version`

索引：

- `(group_id, started_at)`
- `(group_id, status, ended_at)`
- partial unique `(group_id) WHERE status='open'`

### `episode_messages`

- `episode_id` FK `conversation_episodes.id`
- `group_id` FK `groups.group_id`
- `message_id` FK `messages.id`
- `ordinal`
- composite PK `(episode_id, message_id)`
- unique `message_id`，保证每条群消息只属于一个主 episode
- unique `(episode_id, ordinal)`
- `conversation_episodes` 和 `messages` 均增加/声明 `(id, group_id)` 唯一键；
  membership 使用 `(episode_id, group_id)`、`(message_id, group_id)` 组合外键，
  从数据库层禁止跨群挂接。

### `retrieval_documents`

- `id` INTEGER PK
- `scope_type`, `scope_id`
- `group_id`：group document 必填，FK `groups.group_id`
- `episode_id` 可空；非空时以 `(episode_id, group_id)` 组合外键绑定本群 episode
- `document_kind`：
  `message|episode|episode_summary|memory|daily_summary`
- `source_table`, `source_id`
- `start_at`, `end_at`
- `content`, `metadata_json`, `content_hash`
- `embedding_model`, `embedding_version`, `status`

唯一键为
`(scope_type, scope_id, document_kind, source_table, source_id, content_hash)`。
内容变化产生新派生版本，旧版本标记 inactive；相同内容重跑不重复。

新增 `retrieval_document_messages(document_id, group_id, message_id, ordinal,
role)` 保存规范化完整来源：

- 组合主键/ordinal 唯一；
- `(document_id,group_id)` 和 `(message_id,group_id)` 组合外键；
- repository 不从任意 `metadata_json` 猜 provenance；
- 所有 FTS/vector/time/entity/fact/reply 查询在 SQL 第一层限制 group，返回
  orchestrator 前再次校验 group/source；不一致时整批 V2 fail-closed 并走独立 V1。

### 派生索引

- `retrieval_documents_fts`：
  FTS5 trigram，列含 `content`、unindexed scope/document ID。
- `retrieval_documents_vec_<generation>`：
  sqlite-vec `document_id` + `group_id PARTITION KEY` + 由配置维度决定的 vector。
- `retrieval_index_state` 记录 channel、generation、物理表、model/provider、
  dimensions/version、status、coverage 和唯一 active generation。

## 3. Schema 迁移与并发

继续采用 `create_all` + `_run_schema_migrations`：

1. ORM `create_all` 创建新普通表。
2. `CREATE INDEX IF NOT EXISTS` 创建普通索引/partial unique。
3. FTS/vec 初始化在独立 best-effort 函数中完成。
4. 多进程竞争只忽略已明确等价于“另一进程已完成”的
   already-exists/duplicate-column 类错误；其余异常抛出。
5. 启动迁移不得 drop 当前 active vec 表。模型/维度/版本变化时创建新的
   inactive generation，后台构建并验证维度/行数/逐群 coverage，然后以 metadata
   CAS 原子切 active pointer；失败保留旧 active 或禁用 vector。旧 generation
   只在观察期后清理。
6. 迁移测试同时启动多个线程/进程调用 `create_all`，并验证表、约束和原始行。

真实库变更前调用 Python `sqlite3.Connection.backup()`，目标使用临时文件写完、
`integrity_check` 通过后原子改名。禁止对活跃 WAL 数据库直接 `copy`。

备份验证后直接从**已验证 backup 文件**生成 `message-ledger-manifest.json`，
而不是从继续写入的 live DB 生成。manifest 包含总计、每个 group 及
`group_id IS NULL` 私聊 bucket：

- watermark：该 bucket 的 `max(messages.id)`；
- count；
- SHA-256 digest。

digest 固定查询：

```sql
SELECT id, platform_msg_id, group_id, user_id, CAST(timestamp AS TEXT),
       CAST(raw_json AS TEXT), plain_text, msg_type, reply_to_msg_id, mentioned_bot
FROM messages
WHERE <bucket predicate> AND id <= :watermark
ORDER BY id
```

每行转为 UTF-8 JSON array：
`json.dumps(list(row), ensure_ascii=False, separators=(',', ':'))`；hash 输入为
8-byte big-endian 行字节长度 + 行字节，按查询顺序流式送入 SHA-256。manifest
记录 `format_version=1`、算法、精确列名/顺序、backup 文件名和生成 UTC 时间。
迁移后 live DB 只比较各 bucket `id<=watermark` 的 count/digest；水位外新增和
late-arrival 单独统计。manifest/backup 都是敏感运行产物，不进入 Git。

## 4. Episode 状态机

### 在线排队与后台分配

每次群消息事务只保存原始消息，并以
`episode_allocate:<group_id>:<segmentation_version>` 稳定 key
原子 upsert 一个 coalescing job；提交后唤醒 worker。主路径不做批量读取、
分段、LLM 或 embedding。

`jobs` additive 增加 `requested_generation`、`processed_generation`、
`locked_at/lease_until`、`backfill_run_id`、`target_generation` 和安全
`last_error_code`。enqueue 原子递增
`requested_generation`；completed job 重新置 queued，running job保持 running
但 dirty generation 增加。worker claim 时记录水位，drain 后用 CAS 完成：
generation 未变才 completed；已变化则 requeue/继续。这样 completed 后新消息、
running 中新消息和完成竞态都不会丢唤醒。

allocator 对单群按 `(timestamp,id)` 稳定顺序处理尚未分配的消息：

1. 按群读取 open episode。
2. 基于前一条消息判断：
   - 日期变化；
   - 静默超过配置分钟；
   - 消息数达到上限；
   - token 达到上限。
3. 若当前消息引用 open episode 内消息，或构成连续 @小町问答，则优先延续；
   为避免无限 episode，只允许延续到软上限（默认硬上限为数量/token 的
   1.25 倍），之后以 `hard_limit` 关闭。
4. 关闭旧 episode，写入结束字段/哈希/边界原因，幂等入队
   `memory_episode_process:<episode_id>:<compaction_version>`。
5. 新建 episode 并写 `episode_messages`。bot 出站消息总是优先附着当前 open
   episode，以保持问答完整。

所有 membership 与 episode 更新由 worker 完成。`episode_messages.message_id`
唯一约束、coalescing generation 和原子 job claim 共同保证重复唤醒/并发 worker
幂等。

### 空闲关闭与回填

worker 周期性关闭末消息超过 idle 阈值的 open episode。回填按
`group_id, timestamp, id` 重放相同分段器；`episode_messages.message_id`
唯一约束使中断重跑安全。每处理一个 episode/job 就提交检查点。

初次回填为每群记录 `max(messages.id)` snapshot 水位。水位内按规范化不可变字段
哈希验证完全不变；水位后实时新增消息单独计数。若新插入消息 ID 高于水位但
timestamp 落入已处理区间，标记 late-arrival：从受影响 episode 前一个稳定边界
开始局部重分段，先把旧 retrieval documents 标 inactive，再只重建派生
episode/membership/documents；绝不更新/删除 message。局部重建也使用版本与
coalescing job，失败时 V1 仍可用。

### 重叠窗口

纯函数按 12～24 条/1800 token 切块，优先在自然边界断开；下一窗口从前一窗口
末尾回退约 5 条。引用上游若在邻近范围内则同窗，`metadata_json` 保存有序完整
`source_msg_ids`、message DB IDs、speaker IDs、reply edges 和 episode ID。

## 5. Embedding Provider

统一协议：

```python
class EmbeddingProvider(Protocol):
    model_name: str
    dimensions: int
    version: str
    available: bool
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float] | None: ...
```

- `LocalFastEmbedProvider` 延迟构造 `TextEmbedding`，显式 cache_dir，
  `BAAI/bge-small-zh-v1.5`/512 维。返回值转为普通 float list 并校验维度。
- `OpenAICompatibleEmbeddingProvider` 使用专用有限超时 httpx client，
  默认复用 LLM base URL/key，也允许额外 embedding base URL/key 设置；验证
  response index 和维度。
- `DisabledEmbeddingProvider` 永远不可用。
- provider 初始化不在 import 期发生。失败只记录模型/错误类别并返回 disabled。
- 批量 embedding 只在 worker/backfill；查询只计算一个 query embedding。
- V2 永不调用 `hashed_text_embedding`；V1 保留原实现和旧测试。

## 6. Episode 后台处理

episode job 顺序：

1. 读取 episode 和有序消息，验证 content hash。
2. 在唯一 derivation boundary 过滤 QQ blocked、reserved/failed outbound；
   它们保留 episode membership 和原始近期上下文，但不得进入 document、FTS、
   vector、LLM summary/fact/event/provider payload。
3. 生成 overlap retrieval documents（无 LLM）。
4. 批量生成文档 embedding；不可用则留下 disabled 状态，FTS 仍可用。
5. 用现有 compaction prompt/parser 生成 episode 摘要和 source-backed facts/events。
6. upsert `summaries`/`memory_items`，保留 supersession 链。
7. 创建 episode summary/memory/daily summary retrieval documents。
8. 标记 episode processed、job completed。

outbound reservation 只有在 delivered/blocked/failed 最终状态落库后才 enqueue
episode allocator；derivation query 在执行时再次检查状态，消除 reservation 与
worker 的竞态。若历史错误派生文档含 blocked source，reconciliation 将其标 inactive，
从 FTS/active vec generation 移除并重建相关 episode 派生物。安全 note 只渲染到
近期上下文，不写入 fact/event/embedding。

每一步写入采用稳定 source/version key；失败 payload 记录 attempts/phase，
有限退避后重试。进程启动 requeue 超时 running job，关闭 set event 并 await。
LLM 不可用可完成纯窗口/FTS 阶段，摘要阶段留待重试，不影响回复。

## 7. 查询解析

`ResolvedMemoryQuery` 为不可变 dataclass，另有结构化 `TimeRange`。

确定性解析：

- 历史/细节触发词决定 `needs_history/needs_detail`。
- `今天/昨天/前天/上周/某月某日/后来/最后` 解析为绝对时间范围或 temporal
  retrieval mode。
- 当前引用 ID 直接加入精确引用目标。
- 从最近 6～12 条的 nickname/group_card/user_id 和最近明确实体解析
  “他/她/那个人/之前那个”。
- 明确昵称/@/引用不调用 LLM。

仅仍含模糊代词且需要历史时调用一次低 token rewriter。rewriter 通过独立有限
超时 client，要求严格 JSON；schema/超时/网络失败返回原问题。

## 8. 混合召回与融合

候选路：

1. FTS5 `bm25()`，scope 在查询内限制。
2. query embedding + sqlite-vec，相同 scope 限制。
3. 时间范围内 documents/messages。
4. speaker/entity 对 metadata、users、facts 精确匹配。
5. `memory_items` 当前事实及必要的 superseded/update 链。
6. reply ID/被回复 bot ID/上下游 message edge。

每路返回 `RetrievalCandidate(document_id/source_msg_ids/channel/rank/score)`。
用 weighted RRF：

`sum(channel_weight / (60 + rank)) + exact/entity/time bonuses`

引用 > 实体 > BM25/semantic > temporal/recency。任何候选都不因关键词 overlap=0
被删除。最终以 document/source provenance 去重，限制 episode 数。

命中后：

- 解析 source message IDs 和 episode；
- 在同一 group/episode 扩展前后配置范围（默认 8）；
- 加入 reply 上游和对应 bot 回复；
- 组合为按 episode 分组的 `EvidenceSegment`，内部按 `(timestamp,id)`；
- 不跨群，和 recent source IDs 去重。

## 9. 动态上下文装箱

`MemoryContextPacker` 输入结构化 recent messages、facts、evidence segments、
summaries，输出 `PackedMemoryContext`：

- normal：总 32000，recent 默认 10000，facts 3000，evidence 15000，
  summaries 4000；
- detail：总 64000，最多 6 个证据段；
- 全局 settings 可覆盖。

算法：

1. recent 只取在预算内的连续最新后缀；
2. facts 按相关性/置信度选；
3. evidence 按融合分数选段，段内保持完整时间顺序，不拆单条消息；
4. summary 最后补充，且只有与命中 episode/实体/时间有关才加入；
5. 所有来源按 `source_msg_id` 去重；
6. 输出每段带 group/time/speaker/episode/source 标签，并统一加
   “untrusted quoted data” 标记；
7. QQ-blocked 消息只保留安全 note，不复述被拦截原文。

ContextBuilder 仍负责系统/persona/safety/web/target 的总 prompt 优先级。

## 10. Router、灰度与回退

`InboundRouter` 接受可选 `memory_orchestrator`。

- `shadow=false, v2=false`：完全 V1。
- `shadow=true`：V1 作为真实 prompt；以当前 message ID 幂等提交有界持久
  shadow job，worker 异步执行 V2，只记录 IDs/counts/scores/token/ms；队列满或
  worker 失败不增加回复延迟。
- `v2=true`：V2 memory package 进入 ContextBuilder。
- V2 抛出任意异常：`logger.exception` 仅含 group/msg/error class，立即走 V1。

V1 检索 SQL/排序逻辑从 router 抽到 orchestrator 的 V1 fallback helper，避免
router 继续膨胀；但 V1 表和行为不删除。

## 11. 配置、依赖与部署

- 在 `AppSettings` 加 PRD 列出的设置，并增加可选
  `MEMORY_EMBEDDING_BASE_URL/API_KEY/VERSION` 与 rewrite timeout/output 设置。
- `pyproject.toml` 和 `infra/wsl/requirements.xiaomachi.txt` 同步 FastEmbed
  依赖；Dockerfile 继续构建期安装。
- Compose 将 cache 使用现有 `/workspace/data` 卷，无新 volume。
- `.env.example`、README、`infra/wsl/README.md` 记录 shadow -> backfill/eval
  -> enabled 的顺序和即时 V1 回退。

发布只运行 `docker compose ... build xiaomachi` 与
`up -d --no-deps --force-recreate xiaomachi`。发布前后记录 LLBot container ID /
StartedAt，必须不变。

## 12. 评测、性能与隐私

真实 JSONL 和逐题输出只放 `data/memory_eval/`（gitignored），不提交聊天。
脚本支持：

- 生成并人工逐题核验 64～100 题；最低分类配额：
  exact/paraphrase/vague_reference/temporal 各 10，
  multi_hop/update/abstention 各 8；
- 固定 dataset schema/version、题目顺序和文件 SHA-256；每题的 expected source
  IDs 必须人工确认，聚合报告只记录 hash/计数，不含聊天；
- 同一题运行 V1/V2；
- recall@10、MRR/NDCG、packed hit rate、token、latency、rewrite rate；
- 按 category 和 group 汇总；
- 输出不含完整消息文本的 aggregate JSON/Markdown 报告。

AC7 在实际 `xiaomachi-bot` 容器、真实回填库、embedding/model 已 warm 的条件下
测量；排除 rewrite/network，单并发，先 warm-up 20 次，再让全部题至少重复 5 轮
（不少于 250 次），统计 resolve + local channels + expansion + pack 的 p95。

启用 V2 前，以指定 `memory_backfill_runs.id`、该 run 的逐群 snapshot watermark、
segmentation/compaction/index generation 为验收边界：边界内 mandatory
episode/FTS/backfill job 必须 `pending=running=failed=0`，embedding coverage
ready 且 failed=0。水位外实时/shadow jobs 单独报告，不影响该冻结 run 的终态。
provider 明确 disabled 只允许停留在 shadow/V1，不满足 V2 enable 门槛。

## 13. 回滚

- 行为回滚：`MEMORY_ORCHESTRATION_V2_ENABLED=false`，保留 V1。
- 向量回滚：provider disabled，只用 FTS。
- 镜像回滚：重新标记发布前 `xiaomachi-bot` 镜像，仅重建 xiaomachi。
- Schema 为 additive；旧代码忽略新表。正常回滚不恢复数据库。
- 只有确有数据损坏且得到单独授权时才停止 writer 并从已验证 backup 恢复；
  不在本任务自动执行破坏性恢复。
