# 配置、权限与工具参考

用途：配置参考；读者：部署者与维护者；维护者：项目维护者。字段的最终验证约束以 `app/config.py` 和 `app/agent/social_policy.py` 为准。

## 文件与生效

| 文件 | 内容 | 公开规则 |
|---|---|---|
| `infra/wsl/.env.example` | 可复制的环境模板 | 密钥/账号留空 |
| `infra/wsl/.env` | 本机账号、模型、代理与开关 | 私有，不提交 |
| `configs/groups.yaml` | 合成群号和默认群策略 | 可公开 |
| `configs/groups.local.yaml` | 真实群号和策略 | 私有，群表整体覆盖模板 |
| `configs/social.yaml` | 社交、计算、服务许可模板 | 可公开 |
| `configs/social.local.yaml` | 本机授权对象与社交设置 | 私有，递归覆盖模板 |
| `configs/persona.yaml` | 启动默认“嘴臭小町”角色 | 仅公开通用角色描述 |
| `configs/persona_variants.yaml` | 默认角色的变体覆盖 | 可公开 |
| `configs/personas/`、`data/personas/` | 本地成员人格与动态覆盖 | 私有 |

环境和 YAML 不是同一套默认值：代码默认、公开模板、本地覆盖都需核对。已安装应用读取镜像中的配置与容器创建时的环境，修改源码侧文件后用 `rebuild-xiaomachi-wsl.bat`。人格的来源约束刷新由独立服务维护，不能靠修改说明快照触发。

## 必填环境

```dotenv
BOT_QQ=100000001
OWNER_QQ=100000002
LLM_BASE_URL=https://provider.example/v1
LLM_API_KEY=
LLM_MODEL=provider-supported-model
LLM_TEXT_ENDPOINT=responses
QQ_PLATFORM=snowluma
TZ=Asia/Shanghai
```

上面的数字与域名均为占位。密钥只在本机填写，模型名必须是服务端实际提供的。Responses 与 Chat Completions 均有适配；工具、视觉、reasoning 和绘图能力需实际验证，协议名称不保证兼容。

| 参数 | 用途 |
|---|---|
| `LLM_REASONING_EFFORT` | 主文本调用思考强度 |
| `LLM_SUPPORTS_VISION_INPUT` / `LLM_VISION_MODEL` | 是否发送真实像素、可选独立视觉模型 |
| `LLM_CONTEXT_WINDOW_TOKENS` / `LLM_MAX_OUTPUT_TOKENS` | 模型窗口与输出预算；须与实际模型匹配 |
| `GROUP_AGENT_ENABLED` / `AGENT_RUNTIME` | 默认持续群 Agent / `harness`；legacy 用于显式回退 |
| `AGENT_MODEL_FIRST_OUTPUT_TIMEOUT_SECONDS` | 首个有效输出期限，模板 90 秒 |
| `AGENT_MODEL_PROGRESS_TIMEOUT_SECONDS` | 后续有效增量停滞期限，模板 45 秒 |
| `AGENT_MODEL_REQUEST_TIMEOUT_SECONDS` | 每请求总预算，模板 120 秒 |
| `GROUP_AGENT_TURN_TIMEOUT_SECONDS` | 前台群回合上限，模板 180 秒 |
| `AGENT_BACKGROUND_TURN_TIMEOUT_SECONDS` | 后台目标回合上限，模板 900 秒 |
| `AGENT_MAX_EXTERNAL_WRITES` | 每轮外部写入上限，模板 2 |

生命周期/保活事件不是有效模型输出，不能延长停滞预算。模型失败可能有明确失败提示，成功必须看 QQ 回执。

## 群策略

复制模板后保留 `default_group_behavior`，在本地 `groups` 填真实对象。例如合成配置：

```yaml
groups:
  "100000003":
    enabled: true
    archive: true
    speak: true
    proactive_reply: true
    proactive_interval_seconds: "120-300"
    proactive_min_new_messages: 4
    image_generation: true
    memory_enabled: true
    recent_context_limit: null
```

`enabled` 控制接收；`speak` 控制发送；`archive` 控制归档；`memory_enabled` 控制长期生成和召回；`image_generation` 控制群绘图许可。开启记忆不会授予空间或跨群读取权限。

`proactive_interval_seconds` 为原有群发言间隔；Harness还执行下面的主动观察及发言限额，取更严格的门槛。直接 @、回复小町和同一人的有效近期接话使用明确互动路径，任务交付单独处理。

Harness在本地检查后才调用模型，每轮最近最多30条群消息；静默也推进该批观察位置，失败不消费来源。以下字段在`default_group_behavior`及各群均可设置：

| 字段 | 默认与含义 |
| --- | --- |
| `proactive_observation_minimum_messages` | 8条不同有效新消息；重复和纯笑声不计入 |
| `proactive_observation_interval_seconds` | 初始300秒；没有新内容不会仅因到时调用 |
| `proactive_observation_max_backoff_seconds` | 连续静默按600、1200秒退避，最多1800秒 |
| `proactive_observation_hourly_models` | 滚动一小时6次主动请求，工具后续、重试和内部压缩都计入 |
| `proactive_observation_hourly_speeches` | 滚动一小时3次自主插话；分段算一次，不要求用完 |
| `proactive_observation_speech_cooldown_seconds` | 自主插话至少间隔600秒 |

事件消息和定时群观察共用持久额度，重启不清零。发言冷却或额度不允许时不做插话推理；数量和时间达标仍可沉默。`proactive_min_new_messages`保留为既有参与策略，不能绕过上述限制。

`proactive_feedback_enabled` 默认开启。嫌烦/闭嘴请求结合真实引用和最近机器人发言识别，不要求写机器人名字；对工作或明确其他人的抱怨不计入。确认后该群主动静默 `proactive_feedback_quiet_seconds`（默认 1200 秒），随后在反馈发出后的 `proactive_feedback_duration_seconds`（默认 7200 秒）内使用 `proactive_feedback_interval_seconds`（默认 `600-1200` 秒）。新反馈重新计时；同一消息重放不延长，来源撤回后失效，过期恢复原间隔。普通 @ 提问保留，但嫌烦请求本身不会被接梗反击。各字段可按群覆盖，群间互不影响。

## 私聊与主人

`OWNER_QQ` 是主人专属行为权限依据，必须核验真实发送者。`PRIVATE_CHAT_QQS` 增加固定接收名单；`PRIVATE_CHAT_ALL_FRIENDS=true` 可启用动态好友准入，查询失败或删除好友不保留旧动态许可。

私聊与群聊进程分离。私聊用于对话、资料/图片相关能力以及主人对已启用工具的委派，不提供宿主 Shell、远程文件或项目控制。工具许可仍按使用者/作用域细分，私聊准入不等于取得主人权限。

主人可从消息请求切换已配置群人格、执行已启用空间/资料/计算等动作，行为结果回原会话。主人指令不占主动行为冷却/次数；不绕过许可名单、来源复验、单轮模型预算、未知回执规则和平台限制。

## 搜索、网页与绘图

| 参数 | 说明 |
|---|---|
| `SEARCH_PROVIDER` / `SEARCH_BASE_URL` / `SEARCH_API_KEY` | 配置 Tavily、DDGS、Wyzlab alpha 或 `gemini` 原生 Google 搜索；Gemini 使用独立搜索地址和密钥 |
| `AGENT_WEB_SEARCH_RESULTS` | Harness 搜索默认 8 条，单次最多 20 条 |
| `AGENT_WEB_TASK_MAX_CALLS` / `AGENT_WEB_TASK_MAX_CHARS` | 正式资料回合默认16次网络尝试、96000结果字符；可设8–24次/48000–128000字符。普通回合8次/48000字符，失败重试也计数 |
| `AGENT_TASK_TURN_TIMEOUT_SECONDS` | 直接群互动与私聊主回合默认600秒；不再以关键词决定路由或时限。独立资料子任务后台最多300秒，完成摘要回传主会话；模型单次仍受自己的超时控制 |
| `WEB_PAGE_PROXY_URL` | 留空直连；可设 `http://127.0.0.1:18797`，用于本地网页及网页图片/视频读取，聊天与搜索接口维持原出口 |
| `DOWNLOAD_MEDIA_RETENTION_HOURS` | 网页及X下载图片/视频保留24小时，可设1–168小时；群私启动及每小时清理，有效待回传任务保护文件 |
| `DOWNLOAD_MEDIA_CACHE_MAX_MIB` | 网页/X下载缓存共享上限512MiB，可设128–8192MiB；下载前预留空间，优先清理旧且未使用的媒体，满且全被保护时拒绝新下载 |
| `AGENT_CONTEXT_COMPACTION_ENABLED` / `AGENT_CONTEXT_COMPACTION_THRESHOLD_RATIO` | 原生压缩默认启用；阈值取历史预算加固定输入、窗口62%的较小值；摘要不发送QQ |
| `AGENT_CONTEXT_RETAIN_TOKENS` / `AGENT_CONTEXT_SUMMARY_TOKENS` | 近期保留上限16000且不超过历史预算三分之一；摘要4096输出token，每轮内部总结最多2次 |
| `AGENT_CONTEXT_HISTORY_TOKENS` / `AGENT_BACKGROUND_HISTORY_TOKENS` | 主聊天动态历史24000、后台8000 token；固定规则/schema另计，超预算可换段；临时观察约3000软预算，不截断当前明确请求 |
| `AGENT_MONITOR_ENABLED` | 默认false；启用私有最终请求/公开过程采集及本机只读8765网页，操作与保留限额见WSL指南和架构 |
| `WEB_SEARCH_REASONING_EFFORT` | 搜索独立思考强度；Sol 套餐 low，Gemini 套餐 medium，不随主聊天强度切换 |
| `LLM_WEB_SEARCH_MODEL` / `LLM_WEB_SEARCH_API_KEY` | 搜索模型留空复用主模型。主接口的 Wyzlab alpha 可覆盖搜索 key；独立外部搜索使用 `SEARCH_BASE_URL` / `SEARCH_API_KEY` 和指定搜索模型，凭据只放私有环境 |
| `LLM_BUILTIN_WEB_SEARCH` | 兼容消费者的模型内置搜索开关；不替代网页工具适配 |
| `GROUP_IMAGE_TRANSPORT` | `responses` 或 `images` |
| `GROUP_IMAGE_MODEL` | 绘图服务/工具模型 |
| `GROUP_IMAGE_EDIT_MODEL` | 可选参考图编辑工具模型；空值沿用生成模型，仅用于明确指定主模型的 Responses 传输 |
| `GROUP_IMAGE_RESPONSES_MODEL` | 标准 Responses 外层文本模型；与 image tool 模型分开 |
| `GROUP_IMAGE_*API_KEY` / `GROUP_IMAGE_*BASE_URL` | 独立图片服务；未填写部分字段时按代码规则复用文本服务 |

支持正文的搜索桥接提供 `search_web`、`open_web_page` 和 `find_web_page`，引用仅在当前回合有效；总调用和字符预算防止一次读入无限网页。`gemini` 独立调用原生 `streamGenerateContent`，仅声明 `googleSearch`；取得真实 `groundingMetadata` 后返回 `grounded_answer`，缺少来源或异常流即失败。生成摘要不代表网页原文；open/find 直接读取公开页面，逐跳检查重定向，网站请求不携带模型密钥，保留B站公开文字元数据回退。受网络或站点限制的正文可能不可读。传统搜索后端的结果能力不同。给一个 GitHub URL 不等于克隆完整仓库，应指出文件或目录再逐个读取。

Gemini 3.8 Flash 代理套餐以明确的 `-low/-medium/-high/-tiered` 型号接入 Responses。主调用按配置强度选择已验别名；辅助调用的 low 独立解析到 low 型号，不能只凭接口接受 reasoning 字段就认定预算生效。Gemini 原生群私会话绑定型号、协议和地址，切换时轮换并保留历史数据。Images 绘图使用独立 `GROUP_IMAGE_BASE_URL/API_KEY`；生成和编辑共用 `GROUP_IMAGE_MODEL`，`GROUP_IMAGE_EDIT_MODEL` 与 `GROUP_IMAGE_RESPONSES_MODEL` 留空，超时使用图片专属预算。multipart 参考图按成功上传顺序编号关联主体，用户最终指令置后，参考图全部失败时拒绝无图编辑。快捷套餐操作见 [WSL 指南](../infra/wsl/README.md#模型套餐快捷切换)；Gemini 套餐聊天、搜索和绘图均使用新接口，Sol 套餐恢复原搜索通道。

Grok 配置叠加示例在 `infra/wsl/model-profiles/grok-4.7.env.example`，用于候选通道验收；搜索与绘图各自使用独立模型/凭据。Grok Responses 传输将应用的 system/developer 文字统一到顶层 instructions，避免通道以该字段覆盖输入中的人格或发送规则；存储的历史角色不改写。出站工具 schema 省略超出 JSON 安全整数范围的整数上下界；执行端仍使用原 schema 校验，不改变权限或真实来源检查。直接群互动和私聊消息在取得确认写入前携带 `tool_choice=required`；该参数不能替代通道的实际行为验收，须覆盖真实20/31工具、大群上下文、图片及续会话，确认模型确实调用发送工具。内部最终文字不作为QQ回复，声称“已发送”不作为回执；自主回合仍可沉默，发送仍经过权限、来源、预算和真实回执边界。切入/切出 Grok 或更换 Grok 型号时，群私原生会话按模型版本轮换，保留来源与真实回执，避免跨模型重放不兼容的加密推理。

搜索遇到 429 或暂时性 5xx 最多重试一次，两次都计入调用预算；重试仍失败或超时后，同一回合更换关键词也不会继续请求失效服务，下一回合可重试。已有成功资料和明确 URL 的读取仍可使用。服务失败不代表没有搜索结果，应明确说明尚未查证。

APP 分享卡片解析无需新开关。群聊/私聊可发送卡片后引用提问，`read_message_links` 读取其标题、摘要和目标链接，再使用上述网页工具获取正文或补充搜索。只发送卡片不强制主动回复；联网仍依赖已配置的搜索服务。需要登录、反爬拦截或失效的页面可能无法读取，应说明实际取得的资料范围；不支持观看视频。

生成与编辑使用明确的图片来源/产物；参考图失效时不会静默退成无参考文生图。每轮绘图、图片大小、队列与服务超时另有限制。

## 主动空间与好友评论

环境 `SOCIAL_AGENT_ENABLED=true` 与本地 social 总开关/分项开关共同启用。公开模板保守关闭发布和评论。以下只示范字段，启用前须确认自己的账号、好友范围与预算：

```yaml
enabled: true
qzone:
  patrol:
    enabled: true
    interval_seconds: [5400, 10800]
    morning_delay_seconds: [1800, 5400]
    shared_cooldown_seconds: 3600
  publish:
    enabled: true
    autonomous_tools: true
    allow_images: true
    persona: default
    topics: ["日常小发现", "音乐与虚构故事"]
    visibility: 4
    cooldown_seconds: 64800
    max_daily: 1
    quiet_hours: ["23:00", "08:00"]
  comment:
    enabled: true
    autonomous_tools: true
    all_friends: true
    friend_ids: []
    max_daily: 3
    max_daily_per_friend: 1
    cooldown_seconds: 3600
    max_post_age_seconds: 86400
    quiet_hours: ["23:00", "08:00"]
```

`all_friends` 是明确授权当前全部好友；也可关闭并使用 `friend_ids` 固定名单。读取、看图和实际评论前都会再核验。上面空间可见性 4 表示好友可见，64 表示仅自己可见。

巡检是机会而非必发：90–180 分钟随机间隔、夜间静默、早晨延迟。午夜只重置自然日计数，不立即执行。发布每天最多 1 条且隔 18 小时；评论每好友每天 1 条、合计 3 条、每轮最多一条；主动空间写入共同间隔 60 分钟。读取/沉默不占写入次数；发布/评论模型预算分别默认 60 次/日，其他预算仍适用。

`groups` 中的空闲话题日程与 `profile` 中昵称/头像权限是独立分项；不启用就没有相应主动动作。群名片许可属于群，账户资料许可属于 profile。

## 记忆、人格服务与计算

`services.memory_write`、`persona_refresh`、`persona_task_execution` 控制服务工具；公开模板开启这些服务，但仍要求合法群、目标、真实来源及更新阈值。请求入队不等于人格已更新成功。

记忆相关开关与初始化流程见 [MEMORY](MEMORY.md)。`MEMORY_LLM_RERANK_ENABLED` 尚无完整 LLM 重排序实现，保持关闭；查询改写、话题判定/再分段属于可选付费辅助路径，不应凭开关名认为默认必需。

`MEMORY_MEMORY_TOOL_TIMEOUT_SECONDS` 默认 60 秒，覆盖自动记忆加载、显式记忆工具及最终来源复验；`MEMORY_RETRIEVAL_CHANNEL_TIMEOUT_SECONDS` 默认 45 秒，限制并行检索通道等待并为首次本地嵌入加载留余量。完整读取预算应给通道结束后的打包与校验留出时间；通道超时可能只返回降级证据。旧环境显式填写的 2 秒/4 秒会覆盖新默认值，升级时须同步修改私有环境并重建应用。预算是最大等待时间，成功后立即返回；显式较小预算仍会按时失败，取消后的结果不会进入会话。外层最高 120 秒、单通道最高 60 秒。

计算需要 `compute.enabled`、固定本机 endpoint、预建隔离镜像和一致的私有 `XIAOMACHI_SANDBOX_TOKEN`。MCP 由维护者在 social 中固定批准 URL、工具 schema、作用域和效果，模型不能修改这份许可。

## 代理与 GPU

无代理可留空 `XIAOMACHI_HTTP_PROXY` / `XIAOMACHI_HTTPS_PROXY`；QQ/OneBot 保持直连。Docker 构建代理使用独立 `DOCKER_*PROXY`。启用代理需按 [运维指南](../infra/wsl/README.md) 安装验证 WSL 内服务，不能把 Windows 代理端口当成 WSL 中一定可用的 localhost。

`ENABLE_GPU=0` 可用 CPU。启用 NVIDIA 透传后设置 `ENABLE_GPU=1`，向量设备 `auto` 优先 GPU、失败回退 CPU。首次模型下载保持 `MEMORY_EMBEDDING_LOCAL_FILES_ONLY=false`，缓存后才考虑只用本地。
