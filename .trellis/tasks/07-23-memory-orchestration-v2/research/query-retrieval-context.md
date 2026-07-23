# V2 查询、检索与上下文编排集成研究

## 1. 范围与结论

本文只研究现有群聊 `router / context / memory retrieval / query rewrite /
QQ blocked` 链路如何接入 V2，不设计部署实现，也不修改业务代码。研究依据为：

- `.trellis/tasks/07-23-memory-orchestration-v2/prd.md`
- `.trellis/spec/backend/{index,directory-structure,database-guidelines,error-handling,logging-guidelines,quality-guidelines}.md`
- `app/core/router.py`
- `app/core/context_builder.py`
- `app/core/memory_engine.py`
- `app/core/memory_compaction.py`
- `app/core/memory_compaction_service.py`
- `app/main.py`
- `app/group_main.py`
- `app/storage/{models,repositories}.py`
- `app/adapters/{onebot_models,sender}.py`
- 对应的 router/context/memory/storage/sender 测试

核心结论：

1. **V2 不应继续把新逻辑塞进 `InboundRouter._prepare_group_reply`。**
   该函数当前同时承担回复策略、最近上下文、全历史、摘要、词面历史、
   memory FTS/“向量”、成员解析、联网、图片和最终 prompt 组装，已经是跨层
   编排热点。应把“查询解析 → 多路召回 → 融合 → 周边展开 → 装箱”封装成
   一个 `MemoryOrchestrator`，router 对最终记忆上下文只调用这一个入口。
2. **V1 必须先被原样抽成 adapter，而不是边写 V2 边重写 V1。**
   `MEMORY_ORCHESTRATION_V2_ENABLED=false` 时，输出与现有逻辑保持一致；
   shadow 时仍由 V1 构造实际 prompt；V2 active 任意异常回退 V1，再失败才
   回退为严格群隔离的近期连续上下文。
3. **当前 vector 不是语义向量，并且存在二次词面硬过滤。**
   `app/providers/embeddings.py:hashed_text_embedding` 生成 256 维哈希词特征；
   `MemoryRepository.search_group_memories_vector` 使用该特征；router 合并
   FTS/vector/current 后又调用 `retrieve_relevant_memories`，而后者只保留
   `overlap > 0` 的结果。V2 必须完全绕开这条 V1 vector 路径，且 RRF 后不得
   再做词面零重合过滤。
4. **QQ blocked 是安全协议，不是普通“召回噪声”。**
   被拦截的 bot 原回复连同 `QQ_BLOCKED_CONTEXT_NOTE` 会永久保留在原始
   `messages`，并有意继续出现在近期/全历史上下文；但它被排除在摘要、
   compaction window 和 compaction range 之外。V2 必须保留这一区分：
   原始账本与近期安全连续性保留，派生摘要/事实/embedding/检索文档禁止把
   blocked 原文作为来源。
5. **shadow 检索不应增加实际回复延迟。**
   shadow 的实际上下文必须同步走 V1；V2 检索应进入有界后台任务/队列，
   仅记录 ID、分数、token、耗时和错误类别。不要在 router 的同步 prepare
   线程里先跑完整 V2 再返回 V1。

## 2. 现有真实数据流与证据

### 2.1 入口、持久化和回复

真实调用链如下：

```text
OneBot payload
  -> app/main.py 或 app/group_main.py 的 handle_payload
  -> should_ingest_group_message（仅 enabled + speak 群）
  -> parse_group_message_event
  -> InboundRouter.handle_group_message
  -> ingest_live_group_message
  -> _persist_inbound_message（原消息 + V1 派生写入 + compaction job）
  -> archive / bbot listener cache
  -> memory_compaction_service.wake
  -> 图片/周报/bbot 等早返回
  -> 获取引用消息 payload
  -> asyncio.to_thread(_prepare_group_reply)
  -> asyncio.to_thread(_generate_group_reply_text)
  -> _send_prebuilt_reply
  -> 成功、发送失败或 QQ blocked 状态落库
```

证据：

- `app/main.py:256-317` 与 `app/group_main.py:76-113` 负责 group payload
  过滤、解析并调用 router；`main` 是合并进程入口，`group_main` 是群聊专用
  入口，两处都必须注入相同 orchestrator。
- `app/core/router.py:932-1117` 的 `_persist_inbound_message` 在同一事务中：
  1) upsert group/user；2) 保存原始 `Message`；3) 运行规则抽取并写
  `memory_items`；4) 每 25 条写 window/daily 摘要；5) 每达到 compaction
  batch 边界写 `jobs`。
- `app/core/router.py:1948-2037` 表明消息先持久化并唤醒 compaction，再经过
  图片、周报、bbot、引用解析和 prompt 生成。V2 episode/index 后台处理
  应沿用“先持久化、只唤醒/排队”的位置，不能把批量 embedding 或 episode
  LLM 放进这一主路径。
- `tests/core/test_router.py:2200-2244` 证明历史回填只持久化、不回复且不下载
  图片；`tests/core/test_router.py:2623-2644` 证明入站消息先于 LLM/回复策略
  持久化。这些顺序是 V2 不得改变的兼容契约。

一个重要边界是：`should_ingest_group_message` 当前等同
`should_speak_in_group`，即非 `enabled + speak` 群连原始消息都不会进入
router。V2 不能暗中扩大在线采集范围；“全部已有群消息回填”应由独立回填
入口处理，而不是改变在线 allowlist 行为。

### 2.2 `_prepare_group_reply` 中的 V1 上下文流

`app/core/router.py:1149-1759` 的实际顺序为：

1. 读取群策略、最近消息，并用最近消息计算回复策略。
2. `is_history_detail_query(event.plain_text)` 只通过正则判定宽历史模式。
3. 若群策略 `long_context_history=true`，读取完整群历史；否则后续走有限
   词面历史。
4. 检测最近/全历史中是否有 QQ blocked outbound，若有则追加禁止复述规则。
5. 完成回复决策；不回复或进入图片生成时，历史检索不会继续执行。
6. 读取摘要。若存在 `semantic_window/semantic_daily` 就只用语义摘要，否则
   用全部摘要；先按词面相关性排序，**若无命中则回退到最新摘要**。
7. 非 full-history 模式下：
   - `history_search_terms` 从中文连续串生成双字词，并提取英文数字 token；
   - `MessageRepository.list_group_messages_matching_terms` 用 SQL `ILIKE OR`
     获取候选；
   - `retrieve_relevant_history` 再按 phrase hit/token overlap 排序，并丢弃
     低于阈值的消息；
   - 只返回零散命中消息，不做 episode、引用链或前后文展开。
8. memory 召回并行概念上有三路，但实现为串行：
   - `list_current_group_memories`
   - `search_group_memories_fts`
   - `search_group_memories_vector`
   合并去重后调用 `retrieve_relevant_memories`。
9. 构建成员 focus、运行时事实、可选联网内容。
10. `ContextBuilder.build` 接收 recent/full history/member/summaries/
    relevant history/memories 等多组列表并最终装配 prompt。

现有 V1 的具体缺陷：

- `app/core/memory_engine.py:204-232` 的历史检索依赖中文双字词和 token
  重叠；语义改写无法稳定命中。
- `app/storage/repositories.py:1047-1102` 的 vector 搜索使用
  `hashed_text_embedding(query)`；`app/providers/embeddings.py` 明确说明其是
  dependency-free 的 lexical fuzzy 特征，不是真实语义模型。
- `app/core/memory_engine.py:181-201` 的 `retrieve_relevant_memories` 只保留
  `score(memory)[0] > 0`，因此即使 vector 路命中语义候选，只要和原查询没有
  词面重叠仍会被丢弃。这正是 PRD 禁止的硬过滤。
- `app/core/router.py:1344-1359` 在摘要无相关命中时选最新摘要。这会让无证据
  查询看起来像有历史依据；V2 必须改为无证据即不装摘要。
- 当前引用消息仅在 `app/core/router.py:1608-1617` 被拼到 target prompt；
  它没有参与 `history_detail`、查询解析或历史召回。
- 当前入站 target 已先落库，`list_recent_group_messages` 没有排除 target，
  因此 V1 recent 中会再次包含当前消息，随后 target section 又包含一次。
  V1 adapter 要保留这一行为以保证关开关回退一致；V2 可以消除重复，但要用
  独立测试锁定。

### 2.3 ContextBuilder 的真实装箱语义

`app/core/context_builder.py` 目前有两层预算：

- section 预算：recent 3600、summaries 1200、relevant history 2800、
  memories 1400，历史 detail 时后三类乘 2；
- 总预算：默认 `max_prompt_tokens=200000`。

关键行为：

- `_trim_lines_to_budget(..., keep_latest=True)` 让 recent 取较新的行，但遇到
  单条超大文本时会截断文本。
- `take_latest_history_within_budget` 明确保留连续的最新后缀，不从中间跳行。
- full history 模式完全替代 `Recent messages` section。
- `_trim_prompt_to_budget` 先删/截 web、历史、memory、summary 等引用数据，
  最后才动 persona/safety/group policy/target。
- 现有输出顺序由
  `tests/core/test_context_builder.py:116-149` 固定为：
  persona → safety → group policy → reply style → recent → member focus →
  summaries → earlier messages → memories → runtime/web → target。
- `tests/core/test_context_builder.py:274-390` 固定了 section 预算、detail 扩容、
  总 cap 和 full-history 最新后缀行为。

V2 不应把 episode 证据先拼成一个大字符串再交给当前通用截断器，因为这会
截断 source ID、引用配对或片段头。应先由 V2 packer 按结构化 segment 原子
装箱，再把已经预算完成的 section 交给 `ContextBuilder`；总 prompt 超限时，
ContextBuilder 只能整段移除 V2 evidence segment，不能任意从中间切断。

### 2.4 QQ blocked 协议

发送路径：

- `Sender` 对 group `retcode=1200` 且含 `waitForSelfEcho timeout` 的连续三次
  失败转换为 `QQMessageBlockedError`，并且不会把原文切块重发
  (`app/adapters/sender.py:45-185`；
  `tests/adapters/test_sender.py:196-225`)。
- router 捕获该专用异常，把预留 outbound 改为：
  `delivery_state=blocked`、`failure_kind=qq_sensitive_content`，原回复后附
  `QQ_BLOCKED_CONTEXT_NOTE`，再发送固定安全 notice
  (`app/core/router.py:421-460, 1798-1888`)。
- 下一轮若 recent/full/selected history 中含 blocked 消息，router 会追加
  “不得复述敏感细节”规则
  (`app/core/router.py:1186-1195, 1403-1409`)。

数据可见性契约：

- `MessageRepository.list_recent_group_messages` 和
  `list_group_messages_chronological` 保留 blocked 消息；
- `list_recent_group_messages_for_summarization`、
  `list_recent_group_message_windows`、
  `list_group_messages_by_id_range` 排除 blocked；
- `tests/storage/test_repositories.py:102-147` 直接验证“blocked 留在 context，
  不进入 memory compaction sources”；
- `tests/core/test_router.py:2270-2317` 还验证下一轮 prompt 中保留 blocked
  原文、系统投递 note 和禁止复述规则。

因此 V2 必须执行以下精确规则：

1. `messages` 原始行和 blocked metadata 不变。
2. blocked 内容不得生成 episode/window retrieval document、summary、fact、
   event 或 embedding，也不得成为独立召回候选。
3. 为保持即时对话连续性，blocked 行仍可出现在近期连续上下文和 V1 full
   history；若因周边展开被带入 V2 片段，必须标记 `blocked=true` 并同步加入
   `QQ_BLOCKED_CONTEXT_NOTE` 对应的安全 policy。
4. V2 日志、shadow diff 和错误中绝不能出现 blocked 原文。
5. V2 active 如选择更强的脱敏表示（例如只渲染固定 marker），必须先新增
   明确的产品/回归测试；不能在“兼容重构”中悄悄改变当前测试固定的行为。

### 2.5 后台 compaction 可复用与不可复用部分

可复用契约：

- `MemoryCompactionService` 的 `start/wake/stop`、单 worker、有界重试、
  stale job requeue、job key 幂等和 `CancelledError` 友好的关闭模型；
- `memory_compaction.py` 的严格 JSON、source ID 白名单、
  source→subject 校验、canonical dedupe 和 fact provenance；
- 测试已覆盖临时 LLM 失败重排、坏 JSON 不覆盖旧 daily、并发 job 幂等和
  未过 lease 不重排
  (`tests/core/test_memory_compaction_service.py:48-307`)。

不可直接继承的假设：

- 固定 50 条 message range 不是 episode；
- backfill window 只保留完整 fixed-size batch；
- 现有 compaction 排除 bot user，而 V2 周边展开需要 bot reply 来还原对话；
- daily digest 和 memory facts 不能代替原始 message/episode/window；
- `last_error` 当前把 `str(exc)` 写进 job payload，V2 provider/rewrite 错误可能
  包含请求内容，必须改为稳定、安全的错误类别。

## 3. 必须保持的兼容性不变量

### 3.1 数据与隐私

- `messages` 是永久事实源；V2 只能新增派生表/索引，不得修改、删除或用摘要
  覆盖原始消息。
- 每个 repository 查询从 SQL 第一层就带 `group_id` 或
  `scope_type='group' + scope_id`；不能先全局 top-k 再按群过滤。
- 返回 orchestrator 前再做一次防御性校验：
  `candidate.group_id == request.group_id`，发现不一致时丢弃整批 V2 结果并进入
  安全 fallback，不能只 warning 后继续。
- source provenance 一直使用真实 `platform_msg_id` / canonical message ID；
  segment、fact、summary 必须能反向追到原始消息。
- V2 日志只允许 group/episode/document/source IDs、rank、score、token、
  latency、route、fallback/error category，禁止 query、聊天原文、prompt、
  provider response 和 blocked 细节。

### 3.2 回复与上下文

- 回复策略、图片路径、联网路径、周报、bbot、私聊、开发控制和 sender 行为
  不归 V2 接管。
- 先持久化入站消息，再决定回复；重复消息仍不重复发送。
- 未决定回复、图片生成早返回、周报/bbot 早返回时，不跑在线 V2 query
  rewrite/retrieval。
- recent 必须按 `(timestamp, id)` 稳定排序，并保持连续最新后缀。
- historical evidence 必须显式标为不可信引用数据，不得执行其内部指令。
- QQ blocked note 与禁止复述规则必须在 V1、shadow、V2 active、V2 fallback
  四条路径一致。
- `long_context_history` 是现有 per-group 配置。最安全的兼容方案是：
  开关关闭时 100% 走现有 full-history V1；V2 active 初期也把它作为显式
  compatibility override，直到有单独迁移决策和回归测试。
- 无足够 V2 证据时只保留 recent/target；不得自动拿最新摘要充当相关历史。

### 3.3 可用性与降级

- embedding 初始化/查询、FTS、query rewrite、rerank、episode/document 缺失、
  shadow worker 失败都不能阻断普通群聊回复。
- 仅 group-scope/provenance 校验失败属于 correctness/security error；仍可用
  独立、严格群隔离的 V1/recent 路径回复，但绝不能继续使用可疑 V2 候选。
- query rewrite 最多一次、只用于模糊历史追问，必须有有限 timeout；失败、
  timeout、坏 JSON 或越权字段均回退原问题。
- 在线回复查询不能等待 batch embedding；query embedding 可以是单条，
  且 provider 不可用时只禁用 vector channel。
- 现有配置全部继续有效；新增 V2 设置必须只在 composition root 构造依赖，
  core 算法不得自行读取环境变量。

## 4. 推荐模块与协议设计

### 4.1 模块所有权

| 模块 | 单一职责 | 不应承担 |
|---|---|---|
| `app/core/memory_orchestrator.py` | rollout 模式、阶段编排、fallback、shadow 提交、最终结果校验 | SQL、具体 RRF、prompt 文本解析 |
| `app/core/memory_query.py` | 确定性 query 解析、实体/时间/追问解析、可选一次 LLM rewrite | 数据库检索、装箱 |
| `app/core/memory_retrieval.py` | 多路候选协议、并行召回、RRF、稳定去重 | message 周边读取、prompt 渲染 |
| `app/core/memory_expansion.py` | document/episode 命中映射到原始消息、前后文、引用上游、bot reply | 全局排序、token 分配 |
| `app/core/memory_context_packer.py` | 预算分配、source ID 去重、segment 原子选择、渲染为不可信证据 | 查询数据库、调用 provider |
| `app/core/legacy_memory_context.py` | 从 router 原样搬出的 V1 召回/格式化 adapter | 新 V2 行为 |
| `app/storage/repositories.py`（或拆分 memory retrieval repositories） | 所有带 group scope 的 SQL、FTS/vector/episode/reply 查询 | 排名策略、prompt |
| `app/providers/embeddings.py` | `EmbeddingProvider` 接口与 provider adapter | rollout/fallback；V2 禁止调用 hashed vector |
| `app/main.py` | 构造 provider/repository/orchestrator/shadow worker | 领域算法 |
| `app/group_main.py` | 复用 `app/main.py` 的 factory 并注入同样依赖 | 复制一份构造逻辑 |

为了降低合并风险，可以先让 `ContextBuilder.build` 增加一个
`memory_context: PackedMemoryContext | None` 参数，并暂时保留旧参数给现有
单测/legacy adapter；router 新路径只能传 `memory_context`，不得同时传
`summaries + relevant_history_messages + memories`。等回归稳定后再删除旧入口。

### 4.2 建议的核心数据协议

字段名可以调整，但职责不应混合：

```python
@dataclass(frozen=True, slots=True)
class MemoryQueryRequest:
    group_id: int
    current_msg_id: str
    user_id: int
    query_text: str
    occurred_at: datetime
    reply_to_msg_id: str | None
    quoted_message: QuotedMessageRef | None
    recent_messages: tuple[RecentMessageRef, ...]  # 有界 6~12 条，按时间正序

@dataclass(frozen=True, slots=True)
class ResolvedMemoryQuery:
    original_query: str
    resolved_query: str
    entity_ids: tuple[str, ...]       # 群内 canonical user/entity ID
    speaker_ids: tuple[str, ...]
    start_at: datetime | None
    end_at: datetime | None
    retrieval_mode: str              # none/exact/entity/temporal/vague/multi_hop
    needs_history: bool
    needs_detail: bool
    rewrite_used: bool
    confidence: float

@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    document_id: int
    group_id: int
    document_type: str
    episode_id: int | None
    source_msg_ids: tuple[str, ...]
    start_at: datetime
    end_at: datetime
    routes: tuple[str, ...]
    route_ranks: tuple[tuple[str, int], ...]
    fused_score: float

@dataclass(frozen=True, slots=True)
class EvidenceMessage:
    source_msg_id: str
    group_id: int
    user_id: int
    occurred_at: datetime
    text: str
    reply_to_msg_id: str | None
    is_bot: bool
    qq_blocked: bool

@dataclass(frozen=True, slots=True)
class EvidenceSegment:
    segment_id: str
    episode_id: int | None
    hit_source_msg_ids: tuple[str, ...]
    messages: tuple[EvidenceMessage, ...]
    score: float
    routes: tuple[str, ...]
    estimated_tokens: int

@dataclass(frozen=True, slots=True)
class PackedMemoryContext:
    recent_messages: tuple[EvidenceMessage, ...]
    facts: tuple[PackedFact, ...]
    evidence_segments: tuple[EvidenceSegment, ...]
    summaries: tuple[PackedSummary, ...]
    policy_notes: tuple[str, ...]
    selected_source_msg_ids: tuple[str, ...]
    estimated_tokens: int
    resolved_query: ResolvedMemoryQuery
    mode: str                        # v1/v2/recent_fallback
```

安全边界：

- request 可以含 query/近期文本，但不得直接进入日志；
- LLM rewrite 只接收必要、经过长度限制的引用与近期文本；
- candidate 不直接携带任意 metadata JSON 给 renderer；repository 必须把
  unknown JSON 解析为 typed projection；
- `PackedMemoryContext` 返回前统一验证 group、source ID、预算和 blocked
  policy，不让每个 renderer 私自解释 payload。

## 5. 查询解析与 rewrite

### 5.1 解析顺序

建议以确定性规则为主，顺序固定：

1. **规范化当前 query**：去 bot mention、空白和纯口头前缀，但保留原文。
2. **解析精确引用**：
   - `reply_to_msg_id` 先在当前 group 内查 canonical message；
   - 若是 bot reply，沿 `reply_to_msg_id` 找它回答的用户消息；
   - gateway 返回的 quoted payload 只作补充，不得绕过 group 校验。
3. **解析显式时间**：绝对日期、今天/昨天/上周/之前/后来等以
   `occurred_at` 和运行时本地时区解析，最终统一成 UTC 半开区间。
4. **解析显式人物/实体**：使用群内 user/card/nickname/alias 索引，输出
   canonical ID；显示名碰撞时保留多个候选而不是猜一个。
5. **解析代词和追问**：
   - 在最近 6~12 条中维护“最近明确实体/说话人/事件”；
   - “他/她/那个人”只在唯一高置信 antecedent 时绑定；
   - “详细讲讲/后来呢/最后呢/之前那个”继承最近明确主题；
   - 引用的 bot 回答可提供主题，但 QQ blocked 内容不得送入 rewrite。
6. **计算 flags**：
   - `needs_history`：显式历史词、引用旧消息、时间范围、模糊追问；
   - `needs_detail`：详细/经过/后来/最后/怎么处理等证据型请求；
   - `retrieval_mode` 决定启用的 channel 和预算。
7. **仅当仍是模糊历史追问且规则无法形成可检索 query 时调用一次 LLM**。

### 5.2 rewrite 协议

LLM 输出必须是单个严格 JSON 对象，例如：

```json
{
  "resolved_query": "群友 42 之前提到的上海出行计划后来是否取消",
  "entity_ids": ["42"],
  "speaker_ids": ["42"],
  "time_hint": null,
  "needs_history": true,
  "needs_detail": true
}
```

约束：

- schema 拒绝未知字段；LLM 不能返回/更改 `group_id`、source ID、limit 或
  SQL/FTS 表达式；
- 最多一次调用、低输出上限、有限 timeout；
- `resolved_query` 为空、坏 JSON、entity 不在群内、时间无法解析、provider
  异常时，整体回退 deterministic 结果/原问题；
- rewrite 只是 recall 辅助，不能决定安全策略、事实真假或最终回答；
- rewrite 结果和原 query 都不得出现在 INFO/WARNING 日志。

## 6. 多路召回与 RRF

### 6.1 统一候选通道

每一路默认取 20~40，且在 repository 内先做 group scope：

| route | 数据 | 作用 |
|---|---|---|
| `exact_quote` | 精确 source/reply ID | 最高优先级种子，不能被 RRF 稀释 |
| `bm25` | retrieval document FTS5 trigram | 词面/短语/中文精确召回 |
| `vector` | 真实 embedding + sqlite-vec | 改写、同义表达、语义召回 |
| `entity` | metadata/结构化 subject/speaker | 人物、昵称、说话人 |
| `temporal` | start/end 时间范围 | 显式日期、跨天、后来/最后 |
| `fact` | 当前/历史结构化事实事件 | 计划、决定、失效/替代、多跳 |
| `reply_graph` | reply_to/source links | 引用上游、bot 回答及被回答消息 |

FTS/vector 失败只关闭对应 route；exact/group/time/reply/fact 仍可工作。

### 6.2 RRF 算法

不要混合 BM25 distance、cosine distance、时间差等不可比 raw score。每路
先产生稳定 rank，再使用 weighted RRF：

```text
rrf(document) = Σ_route weight(route, query_mode) / (k + rank_route(document))
```

建议初值：

- `k = 60`
- `exact_quote = 6.0`
- `reply_graph = 4.0`
- `entity = 3.0`
- `fact = 2.5`
- `bm25 = 1.8`
- `vector = 1.8`
- `temporal = 1.2`

这些值必须通过真实 eval 校准，不应硬编码散落在 router。规则：

1. exact quote 作为 pin/强 boost；即使其他通道没命中也保留。
2. BM25 与 vector 权重相近，互补而不是 vector 被 lexical gate 否决。
3. 显式 entity/time 可以提高对应 route 权重；“最近”只加分，不作为通用
   硬过滤。
4. 只有以下条件是硬过滤：group 不匹配、document 非 active/版本不兼容、
   明确时间范围完全不相交、provenance 无法验证、blocked 派生文档。
5. document 去重键用 canonical `document_id`；不同 document 映射到相同
   `source_msg_ids` 时保留最高分并合并 routes，最终在 expansion/packing
   再按 source ID 去重。
6. tie-break 固定为：
   `fused_score desc, exact flag desc, end_at desc, document_id asc`，保证离线测试
   和评测可复现。
7. raw channel score可作为诊断字段，但不得直接跨 channel 相加，也不得记录
   文本。

## 7. 命中周边展开

RRF 的输出不是直接 prompt 文本。每个 top document 先定位 episode/window，
再展开为 `EvidenceSegment`：

1. 读取命中 document 的全部 `source_msg_ids`，映射到同 group 原始 message。
2. 定位主 episode；仅在同 episode 内以命中消息为中心向前/后各 5~10 条。
3. 对命中消息沿 `reply_to_msg_id` 向上追溯，建议深度上限 2；若命中用户问题，
   加入直接 bot reply；若命中 bot reply，加入其被回答消息。
4. 强引用/问答配对视为一个原子单元；不得因装箱只留下回答而丢问题。
5. episode 内统一按 `(timestamp, id)` 排序，不按 relevance 打乱对话。
6. 每个 segment 保存完整 `hit_source_msg_ids` 和最终
   `messages[].source_msg_id`；缺失/跨群 source 使该 segment 无效。
7. normal 最多选择约 2~4 个 segment，detail 最多 4~6 个；先限制 segment
   数，再限制 token，防止单次召回膨胀。
8. overlap window 和相邻 document 产生重复消息时，只保留一次；segment
   provenance 保留 route/document 来源。
9. blocked 消息不应成为命中种子；若作为 recent 或邻接消息被带入，标记并
   触发 blocked policy，且禁止派生/日志。

不要通过任意 `message.id ± N` 直接跨 episode 展开。即使全局 ID 连续，也可能
跨群或跨会话；必须由 episode membership + group predicate 定界。

## 8. 动态 token 装箱

### 8.1 预算来源

router/context builder 先计算：

```text
available_input =
  llm_context_window
  - max_output
  - safety_margin
  - tool/web reserve
  - 已知 persona/safety/policy/target/runtime/web 成本
```

然后传给 packer。packer 再取：

- normal：`min(configured_normal_budget≈32000, available_input)`
- detail：`min(configured_detail_budget≈64000, available_input)`

V2 不应因为模型支持 258k 就常态填满窗口。

### 8.2 优先级与预算

推荐顺序：

1. system/safety/group policy/target（在记忆预算外，最高保护）
2. recent 连续后缀，normal 8k~12k；不得从中间跳过消息
3. QQ blocked 安全 note（若触发）
4. 精确 quote/reply segment
5. 当前有效事实，2k~4k
6. RRF evidence segments，normal 8k~16k，detail 可扩大
7. 真正相关的上层摘要，2k~4k

装箱算法：

1. 先计算每条消息、fact、segment 的 token，不先拼字符串。
2. recent 从最新向前取连续后缀；当前 target 在 V2 recent 中排除，避免重复。
3. 将 quote pair / reply chain 作为原子 group。
4. facts 按 query 相关性和 validity 选择；source ID 仍保留。
5. evidence 按 fused score 做 bounded knapsack：
   - 第一轮每个高分 episode 最多一个 segment，保证主题多样性；
   - 第二轮用剩余预算补同 episode 的相邻高分 segment；
   - 单个过大 segment 围绕 hit 缩窄，但不能破坏 quote pair/source header。
6. recent 优先按 `source_msg_id` 占位；evidence 中重复 ID 删除，删空的 segment
   不渲染。
7. segment 内按时间顺序渲染，segment 之间按 relevance 排列；每个 segment
   明确显示时间范围、speaker、episode/document/source IDs，并标注
   “untrusted quoted data”。
8. summary 必须有 query 命中且其 source 没被更高质量 evidence 完全覆盖；
   无命中不使用“最新 summary”兜底。
9. 最后整体校验 `estimated_tokens <= budget`；超限按
   irrelevant summary → 低分 evidence → 低分 fact 的顺序整项移除，recent
   只能从最旧端缩短。

token counter 应为可注入协议。短期可使用当前 `TOKENISH_PATTERN` 估算作为
离线 fallback，但应记录 estimator 版本，并对真实模型留安全余量。

## 9. Router 接入点、shadow 与回退

### 9.1 精确接入点

`MemoryOrchestrator.build_context` 应在以下条件全部满足后调用：

- 入站消息已持久化；
- 回复策略 `should_reply=true`；
- 不是周报、bbot、图片生成或视觉不可用的 prebuilt reply；
- 已取得有界 quoted message reference；
- web search 结果可稍后独立加入 prompt。

即现有 `_prepare_group_reply` 中，最佳接入位置是回复策略和图片早返回之后、
现有 `summary_rows = ...` 之前。现有从 summary/history/memory/member focus
到 ContextBuilder 的记忆逻辑整体移入 legacy adapter / V2 orchestrator。

router 仍可读取“回复策略快照”（最近消息数、last bot reply、图片引用）；
R9 禁止的是 router 自己拼最终记忆上下文，不要求把 reply policy 也并入
MemoryOrchestrator。为避免两次大查询，可新增轻量
`ConversationStateSnapshot`，但不要为了复用而在未回复消息上提前跑 V2。

### 9.2 三种模式

| 配置 | 实际 prompt | V2 行为 | 失败 |
|---|---|---|---|
| V2 disabled | V1 | 不运行 | V1 现有行为 |
| V2 enabled + shadow | V1 | 后台执行 V2，记录安全 diff | V2 失败不影响回复 |
| V2 enabled + shadow=false | V2 | 同步构建 | 任意阶段异常回退 V1 |

建议规定 `shadow=true` 只有在 `enabled=true` 时生效，避免 disabled 的语义
含糊。

### 9.3 fallback ladder

```text
V2 active
  -> stage failure / empty-invalid result / scope-provenance violation
  -> discard all V2 output
  -> legacy V1 provider
  -> V1 failure
  -> minimal recent provider（同 group、连续、blocked-aware）
  -> 正常生成回复
```

每层返回同一个 `PackedMemoryContext` 协议，使 router/context builder 不根据
模式分叉。安全错误不能“部分使用”：例如 vector 返回一个跨群 candidate，
应丢弃整批 V2 context，再走独立 fallback。

### 9.4 shadow 实现

不建议在请求线程同步“双跑”：

- shadow 实际调用先得到 V1 context 并立即用于 prompt；
- 向 bounded shadow queue 提交仅含 group/current msg/evidence ID 的任务；
- worker 从数据库按 group 重新读取必要文本，运行 V2；
- diff 只记录：
  `v1_source_ids`、`v2_source_ids`、各 route count、tokens、latency、
  rewrite_used、fallback/error category；
- job key 以当前入站 message ID 幂等；
- queue 满时记录安全计数并保留 V1 回复，不阻塞；
- shutdown 复用后台 service 的 stop/cancel 等待模式。

若为了完整评测必须保证 shadow 不丢任务，应复用持久 `jobs`，不要使用无界
内存 task。任务 payload 只存 ID/配置版本，不复制聊天原文。

### 9.5 composition roots

- 在 `app/main.py` 增加统一 `build_memory_orchestrator(...)` factory；
- `app/main.py:run` 与 `app/group_main.py:run` 都调用同一 factory；
- 两处启动/停止 V2 shadow/background service；
- router 增加 `memory_orchestrator` 依赖；
- `private_main.py` 和 dev worker 不应因 import factory 而启动 group memory
  worker；
- `build_for_test` 默认构造 V1-only fake/adapter，避免既有 router 测试依赖
  embedding 或真实 LLM。

## 10. 测试建议与验收映射

### 10.1 现有必须保留的回归

- `tests/core/test_router.py`
  - 历史 ingest 不回复/不下载图片；
  - 入站先落库、重复消息幂等；
  - full history chronological 与 model cap 最新后缀；
  - bounded older history；
  - QQ blocked 保存、安全 notice、下一轮 policy；
  - 图片、联网、周报、成员 label、计划失效。
- `tests/core/test_context_builder.py`
  - section 顺序、预算、总 cap、recent/full-history 后缀。
- `tests/storage/test_repositories.py`
  - blocked 在 context 但不进 compaction；
  - group/bot/reserved 过滤。
- `tests/storage/test_long_term_memory_storage.py`
  - FTS/vector active validity、group-before-top-k、source merge/supersession。
- `tests/core/test_memory_compaction_service.py`
  - retry、坏 JSON、source provenance、job 幂等和 lease。
- `tests/adapters/test_sender.py`
  - blocked 不切原文、不递归发送。

### 10.2 新增离线单元测试

**Query resolver**

- 显式历史问题完全走 deterministic，不调用 LLM。
- “详细讲讲/后来呢/之前那个/那个人/他说了什么/最后怎么样”结合最近
  6~12 条、quote 和最近实体解析。
- 同名昵称、无 antecedent、多个 antecedent 时不乱绑定。
- 绝对/相对时间转换和跨天。
- rewrite timeout、provider error、坏 JSON、未知字段、伪造 group/source ID
  全部回退原问题。
- blocked quoted bot reply 不进入 rewrite prompt。

**Hybrid retrieval/RRF**

- BM25-only、vector-only、entity/time/fact/reply-only 候选都能进入最终结果。
- **vector candidate 与 query 词面重合为 0 仍被保留**，专门防止 V1 硬过滤
  回归。
- missing vector/FTS 单路降级。
- 每路 top-k 前已按群过滤；造 20 个其他群高分候选仍只能返回目标群。
- RRF tie-break 稳定、不同 raw score scale 不影响。
- expired/superseded/incompatible-version/blocked document 不返回。

**Expansion**

- episode 内 ±5/10 条，绝不跨 episode/群。
- reply ancestor + bot reply 成对加入，深度有界，循环引用不死循环。
- overlap windows/source IDs 去重，最终时间顺序稳定。
- blocked 邻接消息触发 policy，不生成派生候选。

**Packing**

- normal/detail 预算分别受约 32k/64k 配置限制。
- recent 是连续后缀；target 不重复；recent 与 evidence 按 source ID 去重。
- quote pair/segment 不被从中间截断。
- detail 最多 4~6 个 segment。
- 无证据时不塞最新摘要。
- system/safety/target 优先于所有 memory data。
- 渲染结果包含时间、speaker、episode/source ID 和 untrusted 标记。

**Orchestrator/rollout**

- disabled 输出与 V1 adapter golden 完全一致。
- shadow 实际 prompt 与 V1 一致，V2 只产生安全 metadata。
- active V2 每个阶段异常都回退 V1；V1 再异常回退 recent。
- group/provenance violation 丢弃整批 V2。
- shadow queue 满、worker crash、shutdown 不影响 reply。
- `main` 与 `group_main` factory 参数一致；private/dev 不启动 group worker。

**Router 集成**

- orchestrator 只在 `should_reply` 且非早返回路径调用一次。
- router 不再直接调用 history/memory retrieval helpers。
- QQ blocked 在 V2、V1 fallback、recent fallback 三路都保留 safety policy。
- web/image/private/weekly/bbot 既有行为不变。

### 10.3 评测与性能

- exact/paraphrase/vague_reference/temporal/multi_hop/update/abstention 分类别
  统计 recall@10 和最终 packed evidence 命中率。
- 额外统计：
  channel candidate count、RRF 后 unique count、expansion 后 source count、
  packing drop reason、rewrite rate、fallback rate。
- 不含 rewrite 的本地检索 p95 < 500ms；测试要把 query resolve、各 channel、
  expansion、packing 分段计时，不能只测总时间。
- shadow 对比必须以 source message ID 为准，不能用生成文本相似度替代证据
  recall。

本研究执行的现有离线契约验证：

```text
python -m pytest \
  tests/core/test_memory_engine.py \
  tests/core/test_context_builder.py \
  tests/storage/test_repositories.py::test_qq_blocked_reply_stays_in_context_but_not_memory_compaction_sources \
  tests/core/test_router.py::test_router_persists_qq_blocked_reply_and_sends_safe_notice \
  tests/core/test_router.py::test_router_uses_bounded_relevant_older_history_when_full_history_is_disabled -q
```

结果：`36 passed`。仅出现 pytest-asyncio 默认 fixture loop scope 的弃用警告，
不影响上述契约结论。

## 11. 风险与未知项

1. **blocked 原文兼容与最小披露冲突**：当前测试要求下一轮 prompt 仍含原
   blocked 原文；V2 派生索引又必须避免传播敏感内容。建议明确区分
   “近期原始上下文可见”与“任何派生文档/embedding 禁止”，不要无意改测试。
2. **`platform_msg_id` 当前全表 unique**：V2 协议应以 canonical message PK
   或 `(group_id, platform_msg_id)` 做内部引用校验，不能只因为 platform ID
   唯一就省略 group predicate。
3. **时间语义**：SQLite timestamp normalization 已有 UTC helper，但中文
   “今天/昨天/上周”应以事件发生时的本地时区解释；评测必须包含跨午夜。
4. **full-history 配置**：直接让 V2 忽略 `long_context_history` 会破坏现有
   用户配置；长期可迁移，但初次灰度应作为 V1 compatibility override。
5. **当前 ContextBuilder 会任意截字符串**：若不增加 segment-aware 边界，
   source header、blocked note 或 quote pair 可能被截断。
6. **router session 生命周期过长**：当前 `_prepare_group_reply` 在一个
   `session_scope` 内还可能调用 web search decision/search/page fetch。
   V2 rewrite/并行检索不能继续扩大该事务；应先读 snapshot 并关闭 session，
   provider 调用后再开短事务。
7. **SQLite 并发与 parallel retrieval**：所谓“并行通道”不能让同一个
   SQLAlchemy `Session` 跨线程并发使用；每路独立短 session，或在一次
   repository 查询中批量完成可合并通道。
8. **episode/window 重叠**：一条消息只能属于一个主 episode，但可出现在
   多个重叠 retrieval window；candidate/document 去重与最终 source-ID
   去重必须分两层处理。
9. **bot reply 来源**：现有 compaction 通常排除 bot user，但 reply expansion
   必须读取 bot reply；episode ledger 与派生事实来源规则不能共用一个
   “排除 bot”过滤器。
10. **shadow 新鲜度**：异步 shadow 可能在 episode/index 尚未处理完时运行。
    日志需带 index/compaction version 与 freshness 状态，否则会把未回填误判
    为召回质量差。
11. **rewrite 隐私和 prompt injection**：近期/引用文本是不可信数据；
    rewrite system prompt 必须声明只解析查询，不执行聊天中的指令，且错误
    不能回显 provider payload。
12. **召回无结果语义**：V2 应允许空 evidence；“空”是 abstention 的正确
    输入，不是用最新 summary 补齐的异常。
13. **规则和 RRF 配置漂移**：route 名、权重、`k`、预算和 error category
    应集中定义并带版本，shadow/eval 记录版本，否则不同部署结果不可比较。

## 12. 推荐实施顺序

1. 先增加 typed protocol 和 V1 adapter，用 golden/router 测试证明 disabled
   路径等价。
2. 实现 deterministic query resolver 与 fake rewrite，先覆盖失败回退。
3. 实现 repository 多路候选与 group-before-top-k 测试。
4. 实现纯函数 RRF，专门锁定“零词面 semantic candidate 不被丢弃”。
5. 实现 episode/reply expansion 与 source/group 校验。
6. 实现 segment-aware packer，再让 ContextBuilder 接受单一
   `PackedMemoryContext`。
7. 接入 active fallback ladder。
8. 最后接 shadow 持久/有界任务与安全日志，并用真实 eval 调权重和预算。

该顺序把最高风险的兼容性、群隔离、blocked 安全和 fallback 先固定，再接入
真实 embedding 与性能优化，避免在 router 中一次性重写所有行为。
