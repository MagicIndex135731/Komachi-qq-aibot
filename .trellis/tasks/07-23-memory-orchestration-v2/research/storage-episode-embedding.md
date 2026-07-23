# 存储、Episode、回填与 Embedding 索引研究

## 范围与结论

本文只覆盖 V2 的 SQLite schema、迁移并发、episode/窗口落库、后台回填、
FTS5/sqlite-vec、真实 embedding 以及备份/回滚边界，不讨论路由和提示词的业务实现。
研究期间未读取或输出 `infra/wsl/.env`。

结论：

1. 当前事实源只有 `messages`，V1 的 `summaries`、`memory_items` 与索引均不足以承载
   episode、重叠窗口和可版本化 embedding；应新增 canonical 派生表，但绝不修改或重建
   `messages`。
2. episode、检索文档、来源消息关系必须是普通 SQLite 表；FTS5 与 vec0 只能是可丢弃、
   可重建的加速器，不能参与 canonical 写入事务。
3. 消息主路径只做“原始消息落库 + 幂等排队”。episode 分配、窗口生成、FTS 同步和
   embedding 批处理均由带 lease/checkpoint 的 worker 完成。
4. sqlite-vec 应按 `group_id` partition key 先过滤再 KNN；维度/模型变化采用双槽
   generation 切换，禁止就地把 256 维哈希表当成 512 维语义索引继续使用。
5. 上线前必须用 SQLite Online Backup API 从 live connection 备份 WAL 一致视图，
   校验 `integrity_check`、`foreign_key_check` 和逐群消息计数；关闭 V2 即可立即回到
   V1，新表保留以便审计或重建。

## 已确认事实与路径证据

### 当前 schema 与迁移

- `messages` 保存内部自增 ID、全局唯一 `platform_msg_id`、可空 `group_id`、时间、
  原始 JSON、文本、回复 ID 与 mention 标志；这是现有唯一完整消息账本
  （`app/storage/models.py:36-48`）。
- `summaries` 只保存范围起止消息 ID，不保存完整消息集合
  （`app/storage/models.py:51-68`）；`memory_items` 已有 `source_msg_ids`、有效期和
  supersession 字段（`app/storage/models.py:71-97`）。
- `jobs` 只有 type/key/payload/status/run_at；attempt、lease owner、错误类别和完成时间
  均未结构化（`app/storage/models.py:100-108`）。
- `create_all()` 先执行 SQLAlchemy `create_all`，再跑兼容迁移，最后初始化 FTS/vector
  （`app/storage/db.py:43-47`）。兼容迁移按 `PRAGMA table_info` 加列，只把精确
  `duplicate column name` race 当成功（`app/storage/db.py:144-160`），但整个初始化
  没有 version ledger 或 `BEGIN IMMEDIATE` 串行化。
- 现有测试覆盖历史表升级与串行重复 `create_all`
  （`tests/storage/test_long_term_memory_storage.py:145-187,227-234`），没有并发
  `create_all` 测试。
- 连接只设置 `foreign_keys=ON`，并 best-effort 加载 sqlite-vec；没有 `busy_timeout`
  （`app/storage/db.py:18-40`）。

### 当前 FTS/vector 与检索

- FTS 仅索引 `memory_items`，使用 FTS5 trigram；启动时发现旧 tokenizer 会直接
  drop/recreate 派生表（`app/storage/db.py:163-205`）。
- vec0 仅索引 `memory_items`，固定默认 256 维，建表和补齐时直接调用
  `hashed_text_embedding`（`app/storage/db.py:208-235`）。
- `hashed_text_embedding` 是词、中文二/三元组的 Blake2 哈希桶，不是语义模型
  （`app/providers/embeddings.py:8-36`）。它还在向量查询和每次 memory 写入同步中被
  直接调用（`app/storage/repositories.py:1047-1100,1191-1204`）。
- 向量查询先把本群所有 memory ID 取回 Python，再拼成 `IN (...)`，然后对 vec0 做
  `vec_distance_cosine` 排序；这不是 vec0 的 partitioned KNN，群数据增大后会退化
  （`app/storage/repositories.py:1057-1085`）。
- FTS 查询和普通 memory 查询都显式带 group scope 与有效期
  （`app/storage/repositories.py:975-1045`）；现有测试覆盖 FTS 可选降级、群隔离和
  vector 先按群过滤（`tests/storage/test_long_term_memory_storage.py:190-224,533-551`）。
- `pyproject.toml:10-19` 只安装 `sqlite-vec==0.1.9`，没有 FastEmbed/ONNX 依赖。
- sqlite-vec 官方资料确认：vec0 支持 metadata/partition key，并可使用
  `embedding MATCH :query AND k = :k`；partition key 会在向量比较前过滤。
  `0.1.9` 仍是 pre-v1，应保持精确 pin 并加 SQL 合同测试：
  [sqlite-vec KNN](https://alexgarcia.xyz/sqlite-vec/features/knn.html)、
  [metadata/partition key](https://alexgarcia.xyz/blog/2024/sqlite-vec-metadata-release/index.html)。

### 当前 compaction、job 与回填

- 当前 compaction 仍按固定 50 条工作；启动只扫描每群最近若干完整批次，未满批次尾部
  被丢下，且不是全库回填（`app/core/memory_compaction_service.py:24-40,62-93`）。
- job key 是 `memory:{group}:{start_id}:{end_id}`，数据库唯一索引和
  `INSERT OR IGNORE` 提供幂等入队（`app/storage/db.py:127-141`；
  `app/storage/repositories.py:1222-1253`）。并发入队已有测试
  （`tests/core/test_memory_compaction_service.py:263-283`）。
- claim 使用单条 `UPDATE ... RETURNING`，并把 `run_at` 改作 15 分钟 lease deadline；
  调度时间和租约时间复用同一列（`app/storage/repositories.py:1270-1286`）。
- worker 有有限重试，但 attempts/last_error 塞在 JSON payload 中；异常字符串可能包含
  外部 provider 细节（`app/core/memory_compaction_service.py:306-318`）。
- compaction 的事实来源会严格校验到实际输入的 `source_msg_ids`
  （`app/core/memory_compaction.py:67-123,262-326`），但 semantic window 只保存范围
  起止，缺少完整 normalized provenance
  （`app/core/memory_compaction_service.py:217-228`）。
- 消息主路径目前在每 50 条时同事务入队并随后 wake worker
  （`app/core/router.py:1023-1115,1948-1953`）；这个“落库+排队”形状可以复用，但不能
  在 `_sync_vector` 中替换为真实模型同步调用。
- 现有 OneBot 历史补录只抓启用群、最多 8×50 条，并以最近已知消息作为停止重叠点；
  它是消息账本补录，不是 V2 全库 episode/index 回填
  （`app/core/group_history_backfill.py:15-49,65-118`）。
- 现有部署测试描述的是“停 writer 后文件复制并 integrity_check”
  （`tests/test_llbot_deployment.py:64-72`），不能替代 R11 要求的 live WAL Online
  Backup API。

## 当前行为与缺陷

1. 没有 `conversation_episodes`、`episode_messages`、`retrieval_documents` 或
   document→message 关系表，无法表达会话边界、重叠窗口、完整来源与版本。
2. FTS/vector 只覆盖原子 memory fact，不覆盖 episode/window/event/daily summary。
3. 256 维哈希向量的表 schema、写入和查询没有模型/版本/维度 signature；`CREATE
   VIRTUAL TABLE IF NOT EXISTS` 也不会发现已有维度不兼容。
4. 索引写入和 canonical memory 写入在同一 Session 内；可选索引失败虽被吞掉，但真实
   embedding 若照此接入会阻塞消息路径，并使 source transaction 的失败语义不清晰。
5. FTS/vector 初始化只返回 bool，无持久 health/generation/coverage，错误也缺少安全的
   分类指标，无法支持 shadow mode 和回填验收。
6. 固定、不重叠的 50 条批次不理解静默、跨天、回复链或 token 边界，也不覆盖全部历史
   和未满尾批次。
7. `jobs.run_at` 同时表示可运行时间和 lease deadline，且没有 `locked_by`；恢复虽可用，
   但难以形成可靠的逐群 backfill 报告。
8. 当前测试把哈希向量确定性和 256 维当成契约
   （`tests/core/test_memory_engine.py:117-123`），V2 必须明确删除/隔离这条错误契约，
   而不是让其继续进入语义通道。
9. 测试仅验证 job 并发幂等，没有 schema 并发初始化、模型维度变化、generation 切换、
   全库断点恢复或 backup API 验证。

## 推荐 schema 与约束

### `conversation_episodes`

建议字段：

- `id INTEGER PRIMARY KEY`
- `group_id INTEGER NOT NULL REFERENCES groups(group_id)`
- `segmentation_version TEXT NOT NULL`
- `status TEXT NOT NULL CHECK(status IN ('open','closed','processing','ready','failed','superseded'))`
- `is_current INTEGER NOT NULL DEFAULT 1 CHECK(is_current IN (0,1))`
- `start_message_id/end_message_id INTEGER NOT NULL REFERENCES messages(id)`
- `start_at/end_at DATETIME NOT NULL`
- `boundary_reason TEXT NOT NULL`
- `title TEXT NOT NULL DEFAULT ''`、`summary TEXT NOT NULL DEFAULT ''`
- `message_count INTEGER NOT NULL CHECK(message_count >= 0)`
- `estimated_tokens INTEGER NOT NULL CHECK(estimated_tokens >= 0)`
- `content_hash TEXT NOT NULL`、`compaction_version TEXT NOT NULL DEFAULT ''`
- `created_at/updated_at/closed_at`

约束/索引：

- `ux_conversation_episodes_identity(group_id, segmentation_version,
  start_message_id, end_message_id)`；
- `ux_conversation_episodes_open_group`：partial unique `(group_id)` where
  `status='open' AND is_current=1`；
- `ix_conversation_episodes_group_time(group_id,start_at,end_at,id)`；
- unique `(id,group_id)`，供下游 composite FK 强制群一致性。

### `episode_messages`

字段：`episode_id`、`group_id`、`message_id`、`ordinal`、`is_current`、
`added_at`。约束：

- PK `(episode_id,message_id)`，unique `(episode_id,ordinal)`；
- partial unique `(message_id) WHERE is_current=1`，保证一个消息只有一个当前主 episode；
- composite FK `(episode_id,group_id)` → episode，`(message_id,group_id)` →
  messages。为后者在 `messages(id,group_id)` 建 unique index；这样不能把别群消息挂到
  当前 episode，群隔离不只依赖 Python。

旧 segmentation 版本可将 episode/membership 同事务标记 `is_current=0` 后保留审计，
新版本再插入 current membership；若实现阶段决定不保留旧 episode，则也只能删除这些
派生关系，不能动 `messages`。

### `retrieval_documents`

建议字段：

- `id`、`group_id`、可空 `episode_id`；
- `document_type CHECK IN ('episode_window','episode_summary','fact','event',
  'daily_summary')`；
- `source_key`（如 `episode:{id}:window:{ordinal}`）、`source_version`、
  `chunk_ordinal`；
- `start_at/end_at`、`content`、`metadata_json`、`content_hash`；
- `status CHECK IN ('pending','active','superseded','failed')`；
- `embedding_provider/model/version/dimensions`、`embedding_generation`、
  `embedding_status CHECK IN ('pending','ready','failed','disabled','stale')`；
- `created_at/updated_at`、安全的 `last_error_code`（不存 provider 原始响应）。

唯一约束：

`ux_retrieval_documents_source(group_id,document_type,source_key,source_version,
chunk_ordinal)`。同一来源的新内容 hash/version 新建或 supersede，不能静默覆盖旧 provenance。

另建 `retrieval_document_messages(document_id,group_id,message_id,ordinal,role)`：

- PK `(document_id,message_id,role)`，unique `(document_id,ordinal,role)`；
- document/message 均用 composite FK 带 `group_id`；
- 完整保存窗口实际纳入的 `source_msg_ids`，JSON metadata 只放非关键扩展字段。

检索结果去重、周边展开、孤立消息统计都应从此关系表完成，不解析散落 JSON。

### 迁移、索引与回填状态

- `schema_migrations(version PRIMARY KEY, checksum, applied_at)`：记录不可变 migration，
  发现同 version 不同 checksum 必须失败。
- 在现有 `jobs` 上新增 `attempt_count`、`max_attempts`、`locked_by`、`locked_at`、
  `lease_until`、`completed_at`、`last_error_code`；保留旧列/V1 行为。
- 新增 `memory_backfill_runs` 和 `memory_backfill_group_progress`，保存 run、snapshot
  high-water mark、逐群 cursor、计数、状态与最后安全错误类别，支持 AC10 报告。
- 新增 `retrieval_index_state(channel,slot,signature,dimensions,status,total_documents,
  indexed_documents,activated_at,updated_at)`，FTS/vector 的 ready/coverage 不再靠猜。

## 幂等迁移与并发启动策略

1. 连接设置有限 `PRAGMA busy_timeout`；迁移使用同一 connection 的
   `BEGIN IMMEDIATE` 取得 RESERVED lock，其他入口等待后重读 migration ledger。
2. 把 canonical 新表、普通索引、partial unique index 和 migration ledger 放在串行化
   transaction 内；所有 `CREATE` 使用 `IF NOT EXISTS`，但不能把所有 “already exists”
   或 lock 错误宽泛吞掉。
3. 每个 migration 先检查 schema shape，再执行，再插入 version/checksum。精确
   duplicate-column race 可沿用现有处理；checksum、FK、unique 冲突和损坏必须终止。
4. FTS/vec0 初始化必须在 canonical migration commit 后进行，失败只把 channel 标记
   unavailable，不回滚 schema，也不阻止 bot 启动。
5. 所有入口并发测试至少包含：空库、V1 历史库、已升级库，两线程/两 Engine 同时
   `create_all`；最终 schema/migration 行唯一，消息计数不变，`integrity_check=ok`。

## Episode 分配、窗口与回填

### 在线路径

- 消息事务只插入/确认原始 `messages`，并 `INSERT OR IGNORE` 一个按群唯一的
  `episode_allocate:{group_id}:{segmentation_version}` job，然后 commit。
- allocator 对单群串行，按 `(timestamp,id)` 稳定排序处理未分配消息；静默 30 分钟、
  跨天、50 条、8000 token 是确定性边界。引用链/@小町连续问答可延迟 soft boundary，
  但 hard token/消息上限必须有测试。
- 关闭 episode 后再排 `window_materialize` job；12～24 条、最多 1800 token、约 5 条
  overlap，窗口内部按时间/id 排序，完整写入 `retrieval_document_messages`。
- reserved/QQ-blocked 行仍应有 episode 归属以满足“无孤立群消息”审计，但默认
  `retrieval_eligible=false`，不能进入 compaction/embedding 内容。实现前需把覆盖率
  denominator 固化在测试与报告中。

### 初始全库回填

1. 建立 run，捕获每群 `max(messages.id)` 作为 snapshot；原始消息始终只读。
2. 按群、按 keyset 分页，不用 OFFSET；每批完成 episode/membership upsert 并更新
   group cursor，同事务提交。
3. episode 全部分配后生成 windows/documents，再分别排 FTS 和 embedding job；
   每阶段可独立重跑，以 DB unique constraint 判定幂等。
4. worker crash 后从 lease + group cursor 恢复；最后再做一次未分配消息 reconciliation，
   捕获回填期间新增或晚到的历史消息。
5. 报告：raw group messages、current episode membership、documents、FTS coverage、
   embedding ready/stale/failed/disabled、pending/running/failed jobs、orphan messages，
   以及逐群同项计数。任何 cross-group FK/孤立 current membership 都是硬失败。

## FTS5、sqlite-vec 与真实 Embedding 集成

### FTS5

- 建 `retrieval_documents_fts(content,group_id UNINDEXED,document_id UNINDEXED,
  tokenize='trigram')`；document 是 canonical，FTS 是可重建副本。
- document commit 时只同事务排 `fts_sync` job；index worker 在单独事务 delete+insert，
  用 `content_hash` reconciliation 修复漏写/陈旧项。不要让可选 FTS SQL 失败污染
  document transaction。
- 查询必须带 group 条件，按 `bm25` rank 和 stable document ID 排序，再 join canonical
  表过滤 active/current 版本。FTS 不可用或中文短于 3 字时走有界 LIKE、精确实体/引用/
  时间通道，绝不能跨群。

### sqlite-vec

采用 `a/b` 双槽：

```sql
CREATE VIRTUAL TABLE retrieval_documents_vec_a USING vec0(
  document_id INTEGER PRIMARY KEY,
  embedding FLOAT[512] distance_metric=cosine,
  group_id INTEGER PARTITION KEY
);
```

`b` 同形。`retrieval_index_state` 只允许一个 active slot。模型/维度/version 变化时：

1. drop/recreate **inactive** slot；
2. 后台批量生成并校验 finite、非零、精确 dimensions 的 float32 vector；
3. 核对 signature、总量、逐群 coverage 和抽样 join；
4. 单事务切 active pointer；旧 active slot 保留到观察期结束，可 O(1) 回滚 pointer。

查询使用 vec0 KNN：

```sql
SELECT document_id, distance
FROM retrieval_documents_vec_a
WHERE embedding MATCH :query_embedding
  AND k = :candidate_limit
  AND group_id = :group_id;
```

随后 join `retrieval_documents` 再校验 group/status/version。不要继续构造全群 ID 的
`IN (...)`。vector 使用 compact float32 BLOB（并用 sqlite-vec 函数/测试验证长度），
不再写 JSON 哈希数组。

### `EmbeddingProvider`

接口至少暴露：

- immutable identity：`provider/model/revision/dimensions/encoding_version`；
- `embed_documents(texts)` 与 `embed_query(text)`，明确 passage/query 编码差异；
- availability/health 与安全错误类别；实现 `local`、`openai_compatible`、`disabled`。

local 使用 FastEmbed `TextEmbedding`，默认
`BAAI/bge-small-zh-v1.5`、512 维、cache `/workspace/data/models`；官方支持表确认该模型
为 512 维：[FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/)。
文档使用 `passage_embed`/`query_embed`，不混用。依赖在镜像构建期安装，模型只在 worker
后台首次缺失时下载；初始化失败、缓存只读、维度不符或扩展缺失时熔断 vector channel，
FTS 继续工作，消息路径不等待重试。

`openai_compatible` 还需要明确 base URL、API key、timeout 和返回维度设置；这些不在当前
R10 最小配置列表内，是实现前必须补齐的非阻塞配置合同。密钥只进 settings/provider，
不得进入 job payload、日志或 research artifact。

## 备份、发布与回滚

### 备份

1. 以只读定位后的真实 DB 路径连接 source，用 Python `sqlite3.Connection.backup`
   备份到新的时间戳文件；backup API 读取 live WAL 一致视图，不用 `cp bot.db`。
2. 关闭 destination，重新只读打开，要求 `PRAGMA integrity_check` 每行均为 `ok`、
   `PRAGMA foreign_key_check` 为空。
3. 记录备份位置、时间、文件大小、schema migration version、总消息数和逐群消息数；
   不记录聊天内容。备份文件本身不得进入 Git。
4. 迁移前后再次核对 `messages` 总数、逐群数、平台消息 ID distinct 数；V2 migration
   对 `messages` 的 UPDATE/DELETE 必须为 0。

### 回滚

- 行为回滚：关闭 V2/shadow worker，V1 继续读旧表；不 drop 新表。
- 索引回滚：FTS 标 unavailable 后重建；vector 切回旧 active slot。模型故障只关闭
  vector channel。
- 数据派生错误：按 segmentation/source version 标记 superseded 后重跑；只清理
  episode/doc/index 派生数据，绝不触碰 raw messages。
- schema 本身采用 forward-only additive migration。若 migration 未完成，恢复前述
  在线备份；不要尝试反向 ALTER 或重建 `bot.db`。

## 测试清单

### Schema/迁移/安全

- 空库、V1 历史库、重复初始化、两入口并发初始化；
- migration checksum、有限 busy retry、非 duplicate 错误必须传播；
- migration 前后原始消息总数/逐群 hash 一致，UPDATE/DELETE 计数为 0；
- composite FK、partial unique：跨群 membership/document source 失败，一消息只一个
  current 主 episode，一群只一个 current open episode；
- backup API 对 live WAL 写入快照一致，备份 `integrity_check=ok`、
  `foreign_key_check` 为空。

### Episode/window/backfill

- 30 分钟、跨天、50 条、8000 token 四类边界及并列时确定性；
- 回复链/@小町连续问答软边界、hard limit；
- 12/24 条窗口、1800 token、约 5 条 overlap、引用同窗、完整 source IDs；
- 群隔离、stable ordinal、重复回填、worker 中断/lease 恢复、未满尾批次；
- 回填期间新消息与晚到历史消息 reconciliation；
- reserved/blocked 的归属与 retrieval exclusion；
- coverage 报告逐群加总一致、orphan=0、pending/failed 达终态。

### FTS/vector/provider

- FTS5 缺失、trigram 不可用、2 字中文、重建/reconciliation；
- sqlite-vec 缺失、0.1.9 SQL 合同、group partition KNN、跨群相同向量；
- fake provider 的 query/passages、batch、timeout、坏响应、NaN/零向量、维度错误；
- 512→其他维度或 model/version 变化：inactive build、原 active 可查、原子切换与回滚；
- provider 下载/初始化失败不阻塞启动或群回复，FTS 仍返回；
- V2 测试禁止 import/call `hashed_text_embedding`；现有 V1 哈希测试仅保留在明确的
  legacy namespace，不能作为语义索引测试；
- embedding/LLM/OneBot 全部 fake/stub，单测离线。

### Job/事务/性能

- document 与 index job 同事务、唯一 job key、并发 claim 只一个 winner；
- lease expiry、有限重试、graceful stop、error code 无敏感文本；
- 同一文档重复处理不重复 FTS/vector 行；
- 大群下不生成全群 `IN` 列表；每路 20～40 candidate，验证本地无 LLM 查询
  p95 < 500ms。

## 风险与未知项

1. `messages.platform_msg_id` 当前全局唯一；V2 关系建议使用内部 `messages.id`，对外再映射
   platform ID，避免把“平台 ID 是否跨群全局唯一”的假设扩散到新 schema。
2. “所有群消息”的覆盖率 denominator 是否包含 reserved/QQ-blocked outbound 尚未在 PRD
   明说。本文推荐全部有 episode 归属、但敏感/未投递内容不进入 retrieval，需由主设计
   固化。
3. 晚到历史消息可能落在已关闭 episode 中间；需决定局部重分段还是按导入时间进入新
   episode。若要严格按真实时间重放，应从受影响前一个 boundary 重建该群后续派生版本。
4. sqlite-vec 是 pre-v1 且当前精确 pin 0.1.9；升级必须单独跑 DDL/KNN/delete/partition
   合同测试，不能放宽为无上限依赖。
5. FastEmbed 模型名不足以唯一标识 ONNX artifact。`embedding_version` 应包含显式
   revision/encoding contract（必要时缓存 artifact checksum），否则同名模型更新无法
   可靠触发重建。
6. `openai_compatible` 的 endpoint/key/timeout 配置和供应商返回格式尚未定义；provider
   接口可先落地，但启用前必须补齐配置与 fake 合同。
7. 双槽 vec0 会在 rebuild 期间暂时占用约两倍向量空间；发布前需用真实文档数估算持久卷
   空间并设置最低余量。
