# 小町 QQ AI Bot 工程设计与运行原理

本文面向维护者、部署者和希望理解完整实现的开发者，说明小町如何连接 QQ、组织群聊上下文、调用商用模型与搜索/生图服务、持久化数据，以及 Memory V3 如何从原始消息中检索可核验的历史证据。

日常安装、启动和命令清单见 [WSL/Docker 运维手册](../infra/wsl/README.md)；本文重点解释系统为什么这样组成、数据如何流动，以及各模块的职责边界。

## 1. 系统定位

小町是一个运行在 WSL2 + Docker 上的 Python 3.12 QQ 群聊机器人。它不是直接实现 QQ 协议，而是通过 LLBot（默认）或 NapCat（回退选项）登录 QQ，再使用 OneBot 11 WebSocket 收发事件和调用 QQ 动作。

系统的主要能力包括：

- 群消息接收、归档、回复决策、主动插话和引用回复；
- 基于人格、安全规则、群配置和上下文的文本回复；
- OpenAI 兼容的 Responses 或 Chat Completions 模型调用；
- 模型内置联网搜索，或 Tavily/DDGS 外部搜索；
- 独立商用图片 API 的文生图、参考图生图和队列控制；
- SQLite 持久化、群历史归档、用量记录和 QQ 发送状态跟踪；
- Memory V3 的原始消息投影、本地向量化、混合检索、证据约束和受控发布；
- 私聊管理、开发任务和提醒功能对应的独立进程入口。

## 2. 总体架构

```mermaid
flowchart LR
    QQ["QQ 客户端与群聊"] <--> Server["QQ 服务"]
    Server <--> Gateway["LLBot / NapCat"]
    Gateway <--> OB["OneBot 11 WebSocket"]
    OB <--> Group["app.group_main"]
    Group --> Router["InboundRouter"]
    Router <--> DB[("SQLite bot.db")]
    Router --> Memory["Memory V3"]
    Memory <--> DB
    Memory --> Embed["本地 BGE / sqlite-vec"]
    Router --> LLM["商用 LLM API"]
    Router --> Search["内置搜索或 Tavily / DDGS"]
    Router --> Image["商用图片 API"]
    LLM --> Router
    Search --> Router
    Image --> Router
    Router --> OB
    Watchdog["systemd + watchdog"] --> Gateway
    Watchdog --> Group
```

核心边界如下：

| 边界 | 职责 |
|---|---|
| LLBot/NapCat | QQ 登录、QQ 协议适配、OneBot 事件与动作 |
| `app/adapters/` | WebSocket、OneBot payload、发送结果翻译 |
| `app/core/` | 回复策略、上下文、记忆、搜索/生图编排等业务规则 |
| `app/providers/` | 商用模型、搜索、embedding 等外部能力适配 |
| `app/storage/` | SQLite 表、迁移、仓储查询和事务边界 |
| `app/*_main.py` | 进程装配、生命周期和依赖构造 |
| `infra/wsl/` | Docker、systemd、健康探针、启动和发布操作 |

## 3. 一条群消息如何完成处理

```mermaid
sequenceDiagram
    participant U as QQ 群成员
    participant Q as LLBot / OneBot
    participant G as group_main
    participant R as InboundRouter
    participant D as SQLite
    participant M as Memory V3
    participant A as 商用 API

    U->>Q: 发送群消息
    Q->>G: OneBot message event
    G->>G: 校验群是否允许接收
    G->>R: typed group event
    R->>D: 幂等写入 canonical messages
    R->>R: ReplyPolicy 决定是否回复
    alt 不应回复
        R-->>G: 结束，消息仍已归档/索引
    else 应回复
        R->>M: 解析查询并构建受约束证据包
        M->>D: 同群、人物、时间、来源检索
        D-->>M: 原始证据与近期上下文
        M-->>R: bounded packed context
        R->>A: 人格 + 安全 + 上下文 + 当前问题
        A-->>R: 文本或图片结果
        R->>D: 预留 outbound 状态
        R->>Q: send_group_msg
        Q-->>R: 成功 / 拦截 / 状态不确定
        R->>D: 更新 sent / blocked / uncertain
        Q-->>U: QQ 群内可见回复
    end
```

### 3.1 接入与解析

`app/group_main.py` 创建 `NapCatGateway`。这个类名沿用早期实现，但当前同样连接 LLBot 提供的 OneBot WebSocket。连接断开时 gateway 会清理等待中的 RPC，并按配置持续重连。

`app/adapters/onebot_models.py` 将原始 payload 转成内部事件。只有配置允许接收的群会进入 `InboundRouter.handle_group_message`，因此未知群不会默认启用机器人能力。

### 3.2 先存消息，再考虑回复

路由器首先把入站消息幂等写入 `messages`。这张表是 canonical source of truth：Memory V3、历史归档和发送状态都以它为依据。即使机器人决定不回复，允许归档的消息仍可进入后续索引流程。

图片会被解析为附件并按策略缓存；撤回事件会更新对应消息的可用状态。reserved、blocked、uncertain、deleted 等 delivery state 会影响哪些内容能够进入长期记忆和模型提示词。

### 3.3 回复决策

`app/core/reply_policy.py` 负责是否回复，而不是让模型自行决定。主要信号包括：

- 是否明确 @ 小町、点名小町、回复小町或形成同一会话线程；
- 群是否允许发言；
- 是否处于静默时段或主动回复冷却期；
- 群内近期流量、直接提问和可插话机会；
- 群配置是否允许主动回复。

明确触发优先回复；主动插话需要达到阈值，并使用事件 ID 生成稳定的冷却间隔，避免每次重启改变同一事件的判断。

### 3.4 构造提示词

`InboundRouter._prepare_group_reply` 收集近期消息、引用消息、群成员标签、人格、安全策略、联网结果和记忆上下文。`ContextBuilder` 按显式 token 预算裁剪各区段，保留最新近期消息和已经通过来源校验的历史证据。

历史聊天内容始终被标记为不可信参考数据，不能覆盖 developer/instructions 层的人格、安全与引用约束。

### 3.5 发送与投递状态

`app/adapters/sender.py` 通过 OneBot `send_group_msg` 或 `send_private_msg` 发送结果。发送前会在数据库预留 outbound 记录：

- 正常 self-echo 后标记为 sent；
- QQ 接受请求但没有回显时标记为 blocked，并发送不包含原敏感内容的固定提示；
- WebSocket 断开或超时时标记为 uncertain，不盲目重发，避免群里出现重复消息。

blocked/uncertain 内容可以用于有限的会话连续性，但不会进入自动摘要、embedding 或长期派生记忆。

## 4. QQ、LLBot、NapCat 与 OneBot

### 4.1 各自做什么

- **QQ**：最终用户界面和消息网络。
- **LLBot**：默认 QQ 登录容器，保存会话、签名信息和 OneBot 配置；WebUI 默认在 `127.0.0.1:3080`，OneBot WebSocket 默认在 `127.0.0.1:3002`。
- **NapCat**：保留的回退平台，默认 OneBot WebSocket 端口为 `3001`。
- **OneBot 11**：机器人与 QQ 平台之间的统一事件/动作协议。入站消息是事件，发送消息、查询状态、读取引用消息等是带 echo 的动作调用。

LLBot 与 NapCat 的登录态互相独立，不能同时用同一个 QQ 号运行。切换平台时启动脚本会先停止另一平台。

### 4.2 容器拓扑

`infra/wsl/docker-compose.llbot.yml` 使用 host network：

- `xiaomachi-llbot` 运行固定版本的 LLBot 镜像；
- `xiaomachi-bot` 运行 `python -m app.group_main`；
- bot 数据卷以读写方式挂载到 `/workspace/data`；
- LLBot 只能只读访问 bot 数据卷，同时拥有独立的登录态数据卷；
- GPU 只分配给 `xiaomachi-bot`，LLBot 不需要 CUDA。

发布必须使用 bot-only 命令：只构建并重建 `xiaomachi`，不得重启或重建 LLBot。这样不会破坏 QQ 登录态。

### 4.3 健康检查

`infra/wsl/scripts/status.sh` 同时检查：

1. 当前 QQ 平台容器状态；
2. LLBot WebUI；
3. OneBot `get_status` 与 `get_login_info`；
4. `xiaomachi-bot` 容器；
5. `/workspace/data/logs/group.heartbeat.json` 的新鲜度。

watchdog 还会周期性调用 `get_status` 和 `get_group_list(no_cache=true)`。连续异常时只重启当前 QQ 平台一次；仍需登录则通知 Windows 用户处理，而不是删除登录数据。

## 5. 商用 API 如何接入

所有密钥只存在本地 `infra/wsl/.env`，由 `AppSettings` 读取，禁止提交 Git。仓库只保存无密钥示例 `infra/wsl/.env.example`。

### 5.1 文本与视觉模型

`app/providers/llm_client.py` 是 OpenAI 兼容客户端。主要配置为：

- `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`；
- `LLM_TEXT_ENDPOINT=responses|chat_completions`；
- `LLM_REASONING_EFFORT`、上下文窗口和输出预算；
- 可选视觉模型与模型 fallback。

Responses 模式使用原生 `instructions` 保持人格、安全和引用约束的高优先级，并可携带图片、内置 `web_search` 与 `image_generation` 工具。客户端解析 SSE 流，提取文本、工具事件和 token usage。

商用网关失败不会被解释为“记忆无素材”。HTTP 403、超时、SSE 拒绝和检索 no-hit 是不同错误类别。对已观察到会间歇性返回 403 的指定代理 host，Responses 请求执行有限次数、相同 payload 的退避重试；其他 host 的 403 仍立即失败。

每次模型调用的 model、endpoint、input/cached/output token 会写入 `usage_records`，用于成本核算；日志不记录 Authorization、完整 prompt 或聊天正文。

### 5.2 联网搜索

有两条互斥的主要路径：

- Responses 模型支持内置搜索时，由模型调用 `web_search`；工具事件写入 gitignored 日志供核验。
- 内置搜索不可用时，`WebSearchClient` 使用 Tavily 商用 API，或无需 API key 的 DDGS；它还可以抓取并清洗少量网页正文。

只有明确联网请求或策略允许时才提供搜索能力。搜索结果是不可信外部资料，必须作为引用上下文而不是系统指令。

### 5.3 图片生成

群聊生图使用独立的 OpenAI 兼容图片服务配置：`GROUP_IMAGE_BASE_URL`、`GROUP_IMAGE_API_KEY`、`GROUP_IMAGE_MODEL` 及 generations/edits endpoint。`GroupImageGenerationService` 负责队列容量、超时、参考图、输出文件和 QQ 发送。

独立 key 让图片成本、超时和供应商故障与主文本模型隔离。图片任务失败不会阻塞普通文本消息处理。

## 6. Memory V3 的构成与原理

Memory V3 的目标不是把整库聊天直接塞给模型，也不是让另一个生成模型自由总结后再猜答案。它从 canonical 原始消息建立可重建的检索投影，并在每次历史问题中生成一个有群、人物、时间和来源边界的证据包。

```mermaid
flowchart TD
    Msg[("canonical messages")]
    Msg --> Job["后台索引任务"]
    Job --> Raw["raw_message_v3 document"]
    Raw --> FTS["BM25 / FTS"]
    Raw --> Vec["BGE embedding / sqlite-vec"]

    Query["当前问题 + 近期/引用上下文"] --> Resolve["MemoryQueryResolver"]
    Resolve --> Plan["group + subject + speaker + time + mode"]
    Plan --> FTS
    Plan --> Vec
    Plan --> Entity["实体通道"]
    Plan --> Temporal["时间通道"]
    FTS --> Fuse["HybridMemoryRetriever"]
    Vec --> Fuse
    Entity --> Fuse
    Temporal --> Fuse
    Fuse --> Expand["原始来源/直接回复扩展"]
    Expand --> Eligible["逐消息资格复核"]
    Eligible --> Pack["MemoryContextPacker"]
    Pack --> Prompt["有引用边界的历史证据包"]
```

### 6.1 数据层

关键表包括：

| 表 | 作用 |
|---|---|
| `messages` | 不可替代的原始消息与 delivery state |
| `conversation_episodes` / `episode_messages` | V2 时代的会话分段和成员关系 |
| `retrieval_documents` | FTS、episode 和 `raw_message_v3` 检索文档 |
| `retrieval_document_messages` | 文档到 canonical 消息的来源映射 |
| `retrieval_index_state` | 向量 provider/model/dimensions/version/generation 状态 |
| `jobs` | 后台分段、压缩、投影和 embedding 任务 |
| `memory_backfill_runs` | 可恢复回填、snapshot watermark 和覆盖率 |
| `summaries` / `memory_items` | legacy/V2 摘要与结构化事实，保留用于兼容与回滚 |
| `usage_records` | 商用 API token 用量 |

V3 不删除 V1/V2 数据，也不修改原始消息正文。`raw_message_v3` 是派生投影，损坏时可以从 `messages` 重建。

### 6.2 写入与索引

每条合格群消息写入 `messages` 后，后台服务创建或更新 raw document。文本进入 SQLite FTS；本地 `BAAI/bge-small-zh-v1.5` 生成 512 维 embedding，向量保存到 generation 对应的 sqlite-vec 物理表。

embedding 默认配置为 `MEMORY_EMBEDDING_DEVICE=auto`，在 bot 容器内优先使用 CUDA、不可用时回退 CPU，模型缓存在 `/workspace/data/models`。向量通道不可用时在线查询可保留严格作用域的 FTS/entity/temporal 能力，但生产发布必须从 `memory_runtime` 日志确认实际 device 为 CUDA；门禁不会把降级状态当成合格的 V3 向量发布。

后台作业具有 owner/lease、有限重试和 generation 绑定。索引失败不能阻塞即时聊天路径。

### 6.3 查询解析

`MemoryQueryResolver` 先做确定性解析，再在确有需要且开启配置时使用短 LLM rewrite。它输出 `ResolvedMemoryQuery`，主要包含：

- 原问题与检索问题；
- 当前 `group_id`；
- `speaker_ids` / `subject_ids`；
- 上海时区语义转换后的 UTC 半开时间范围；
- 引用消息 ID、检索模式、回答模式和覆盖策略；
- 是否需要详细历史或按时间桶覆盖。

昵称、群名片、QQ 和 @ 只在目标群的成员快照内解析。唯一强指代可以绑定到一个 QQ；重复别名、真实多人指代、被排除用户或跨群别名会 fail closed。`subject_ids=None` 表示未绑定，空 tuple 表示歧义/禁止，二者不能混用。

### 6.4 混合检索

`HybridMemoryRetriever` 并行组合：

- BM25/FTS 词法命中；
- 本地 embedding 的向量语义命中；
- 已解析人物/实体通道；
- 明确日期、昨天、时间桶等 temporal 通道；
- 引用或回复关系产生的精确来源。

候选先在 SQL 中限制 `group_id`，人物和时间边界还会在 canonical 来源上再次验证。通道超时只标记该通道失败；所有通道失败时，历史问题返回明确 no-evidence，不回退到不受约束的全局 legacy 检索。

### 6.5 证据扩展与资格复核

检索命中的是来源单元，`MemoryEvidenceExpander` 可以补充有限的直接回复或 episode 邻域，帮助理解上下文。但扩展后的每条消息必须重新通过 `memory_eligibility.eligible`：

- 同一个群；
- 有稳定 source message ID；
- delivery state 合格；
- 位于半开时间窗内；
- 作者或 mention 满足已解析 subject；
- 不含 blocked/reserved/uncertain/deleted 来源。

任何作为命中依据的 source 不合格，整个对应证据段都会被丢弃。直接回复扫描有数据库硬上限，最终配额在资格检查之后消耗。

### 6.6 证据打包

`MemoryContextPacker` 按完整证据块打包，不在中间截断一条来源。旧配置保持固定的 60 条近期消息、150 条历史消息以及 24k 历史 token / 12,000 字符边界。启用 `MEMORY_ADAPTIVE_CONTEXT_ENABLED=true` 后，近期与历史不再各自占用固定配额，而是在模型实际可用输入窗口和 48,000 字符保护线内动态分配：

- 先为近期和历史各保留最低 token 预算，再把未使用额度双向让给另一侧；
- 强直接/词法/多通道证据使用最多 150 条候选的紧凑扩展，弱命中或通道降级才扩到最多 300 条；
- 120 条近期消息和 300 条历史消息只是防止异常短消息造成行数爆炸的应急上限，不是固定装满目标；
- 超预算时按确定顺序降级，但完整证据块、直接命中和已固定来源不会被截成半条；
- 每个来源 ID 去重并保留；
- 单一日期事件要求使用最小充分证据，后续更正优先。

模型得到的是“近期聊天”和“历史证据”两个不同区段。近期聊天可以包含当前群的其他成员，而历史证据必须满足当前查询的人物/时间约束。

### 6.7 无素材与故障语义

以下情况会明确返回证据不足，而不是猜测：

- 人物歧义或被排除；
- 当前群没有符合时间/人物边界的来源；
- 来源被撤回、blocked 或状态不确定；
- 所有检索通道失败；
- V3 内部异常且当前请求是历史查询。

LLM 403/超时发生在证据包之后，属于 upstream failure；它不会被记录为 V3 no-hit。

## 7. Memory V3 发布与回滚

V3 使用“不可变产物 + hash 绑定 + generation CAS”发布：

V3 的 raw generation 发现复用 V2 编排基础设施，因此正常运行必须保持 `MEMORY_ORCHESTRATION_V2_ENABLED=true`。V3 的启用与回滚只切换 `MEMORY_RAW_V3_ENABLED`，不能通过关闭 V2 总开关代替。

1. 用 SQLite backup API 创建一致性备份并验证 `integrity_check=ok`；
2. 生成 manifest 和每群 snapshot watermark；
3. prepare/catch-up 投影所有合格 raw 消息；
4. 构建独立向量 generation，要求全部文档 ready；
5. 生成并人工审核 64 题真实数据集；
6. 运行 retrieval、回答/引用质量、visibility 与离线 benchmark；
7. gate 重新计算 dataset、manifest、results、quality、benchmark 的 hash 与指标；
8. activation 脚本再次验证全部产物，再用 CAS 把目标 generation 切成 active；
9. 设置 `MEMORY_RAW_V3_ENABLED=true`，只重建 bot 容器；
10. 验证 `route=raw_v3`、CUDA、OneBot、heartbeat 和 LLBot 不变量。

评测要求零群泄漏、零人物/时间泄漏、零不合格来源和零引用越界，并对 recall、citation、answer、abstention、visibility、TTFT 和 retrieval p95 设置硬阈值。benchmark 禁止联网和 rerank，64 题至少需要 20 次预热和 320 次实测。

回滚不会删除 V3 数据。它用 CAS 恢复上一 active generation，把 `MEMORY_RAW_V3_ENABLED` 设回 false，再 bot-only 重建。CAS 后任何报告写出异常也会触发自动恢复，避免命令失败但 generation 已经暗中切换。

完整命令见 [Memory V3 运维清单](../infra/wsl/README.md#memory-v3-prepare-evaluate-activate-and-rollback)，严格契约见 [Memory orchestration spec](../.trellis/spec/backend/memory-orchestration-v2.md)。

## 8. 进程与项目结构

```text
app/
  adapters/          OneBot payload、WebSocket 和 QQ 发送
  admin/             私聊管理命令
  core/              路由、策略、上下文、记忆和业务工作流
  dev_control/       私聊开发任务与仓库控制
  jobs/              一次性/计划任务入口
  providers/         LLM、搜索、embedding 等外部适配
  storage/           SQLAlchemy models、schema 和 repositories
  group_main.py      当前生产群聊进程
  private_main.py    私聊管理与提醒进程入口
  dev_worker_main.py 开发任务 worker 入口
  main.py            共享 factory 和 legacy 组合入口
configs/             群白名单、人格、安全和提醒 YAML
infra/wsl/           Compose、Dockerfile、systemd、watchdog 和运维脚本
scripts/             备份、回填、评测、质量回放和激活工具
tests/               与源码层次对应的离线测试和部署合同测试
docs/                工程说明和历史设计资料
.trellis/spec/       可执行开发契约；不是生产数据
```

当前 Docker 生产栈只启动 `app.group_main`。`private_main` 和 `dev_worker_main` 是独立进程入口，只有单独部署时才运行；它们不会因群聊容器启动而自动出现。

## 9. 配置与数据边界

### 9.1 可提交配置

- `configs/groups.yaml`：群是否接收、归档、发言、主动回复和生图；
- `configs/persona.yaml`：小町人格、表达方式和特殊熟人规则；
- `configs/safety.yaml`：AI 身份披露、prompt 防泄漏和内容安全；
- `infra/wsl/.env.example`：无秘密的环境变量目录。

### 9.2 不可提交状态

- `infra/wsl/.env` 和任何 API key；
- LLBot/NapCat 登录态、WebUI 密码、token、二维码；
- `data/bot.db*`、备份、真实评测集和真实聊天归档；
- 图片缓存、生成图片、模型缓存、运行日志和 heartbeat；
- 本地虚拟环境、pytest 临时目录和 Docker runtime cache。

GitHub 保存的是可重建代码和无密钥配置，不是生产会话备份。恢复仓库不能恢复 QQ 登录态或聊天数据库。

## 10. 故障隔离原则

| 故障 | 预期行为 |
|---|---|
| QQ/OneBot 离线 | WebSocket 重连；status/watchdog 报告登录问题 |
| 商用 LLM 失败 | 记录 upstream failure；必要时发本地失败提示，不伪装成记忆无素材 |
| 图片 API 失败 | 当前图片任务失败，普通文本聊天继续 |
| 搜索 API 失败 | 不提供搜索资料，普通模型回复仍可继续 |
| vector 超时/不可用 | 当前通道失败；其他严格作用域通道继续，历史全失败则 no-evidence |
| 后台索引失败 | 有限重试并保留任务，不阻塞群消息处理 |
| QQ 发送状态不确定 | 标记 uncertain，不盲目重发 |
| V3 作用域或来源异常 | 历史请求 fail closed，不回退到不受约束记忆 |
| 发布 CAS 冲突 | 停止激活，保留原 active generation |

## 11. 开发、测试与发布

日常开发使用 Python 3.12 和 pytest。外部 LLM、搜索、图片、embedding 与 OneBot 在单元测试中使用 fake/mock，默认测试不应产生商用 API 费用。

主要验证层级：

1. 受影响模块的 focused tests；
2. `python -m pytest tests -q` 全量回归；
3. `python -m compileall -q app scripts`；
4. `git diff --check`；
5. Compose config 和部署合同测试；
6. 数据库 release 的 integrity、覆盖率与 activation gate；
7. bot-only 镜像构建、容器状态、OneBot、heartbeat、CUDA 和真实群消息 smoke test。

生产发布不以“测试通过”单独判定成功。测试、签名产物、运行时日志和外部服务健康必须共同闭环。

## 12. 继续阅读

- [README](../README.md)：首次配置、日常操作和常见故障；
- [WSL/Docker 运维手册](../infra/wsl/README.md)：安装、状态检查、Memory V3 命令与发布；
- [Memory orchestration spec](../.trellis/spec/backend/memory-orchestration-v2.md)：记忆系统的严格行为合同；
- [Backend directory structure](../.trellis/spec/backend/directory-structure.md)：源码边界；
- [Backend quality guidelines](../.trellis/spec/backend/quality-guidelines.md)：测试与发布质量要求。
