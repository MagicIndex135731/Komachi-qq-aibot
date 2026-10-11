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
| `configs/persona.yaml` | 启动默认“串子小町”角色 | 仅公开通用角色描述 |
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
    image_generation: true
    memory_enabled: true
    recent_context_limit: null
```

`enabled` 控制接收；`speak` 控制发送；`archive` 控制归档；`memory_enabled` 控制长期生成和召回；`image_generation` 控制群绘图许可。开启记忆不会授予空间或跨群读取权限。

`proactive_interval_seconds` 是主动发言后的间隔，模型在合格机会仍可沉默；直接 @ 请求使用直接互动路径。Harness 的主动群发送复用群间隔，没有再额外套一层 30 分钟社交冷却。每日/资源限制仍独立存在。

## 私聊与主人

`OWNER_QQ` 是主人专属行为权限依据，必须核验真实发送者。`PRIVATE_CHAT_QQS` 增加固定接收名单；`PRIVATE_CHAT_ALL_FRIENDS=true` 可启用动态好友准入，查询失败或删除好友不保留旧动态许可。

私聊与群聊进程分离。私聊用于对话、资料/图片相关能力以及主人对已启用工具的委派，不提供宿主 Shell、远程文件或项目控制。工具许可仍按使用者/作用域细分，私聊准入不等于取得主人权限。

主人可从消息请求切换已配置群人格、执行已启用空间/资料/计算等动作，行为结果回原会话。主人指令不占主动行为冷却/次数；不绕过许可名单、来源复验、单轮模型预算、未知回执规则和平台限制。

## 搜索、网页与绘图

| 参数 | 说明 |
|---|---|
| `SEARCH_PROVIDER` / `SEARCH_BASE_URL` / `SEARCH_API_KEY` | 配置 Tavily、DDGS 或接口兼容的 Wyzlab alpha 搜索服务 |
| `AGENT_WEB_SEARCH_RESULTS` | Harness 搜索默认 8 条，单次最多 20 条 |
| `WEB_SEARCH_REASONING_EFFORT` | 搜索独立思考强度；模板 low，不继承主聊天参数 |
| `LLM_BUILTIN_WEB_SEARCH` | 兼容消费者的模型内置搜索开关；不替代网页工具适配 |
| `GROUP_IMAGE_TRANSPORT` | `responses` 或 `images` |
| `GROUP_IMAGE_MODEL` | 绘图服务/工具模型 |
| `GROUP_IMAGE_RESPONSES_MODEL` | 标准 Responses 外层文本模型；与 image tool 模型分开 |
| `GROUP_IMAGE_*API_KEY` / `GROUP_IMAGE_*BASE_URL` | 独立图片服务；未填写部分字段时按代码规则复用文本服务 |

支持正文的搜索桥接提供 `search_web`、`open_web_page` 和 `find_web_page`，引用仅在当前回合有效；总调用和字符预算防止一次读入无限网页。传统搜索后端的结果能力不同。给一个 GitHub URL 不等于克隆完整仓库，应指出文件或目录再逐个读取。

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

计算需要 `compute.enabled`、固定本机 endpoint、预建隔离镜像和一致的私有 `XIAOMACHI_SANDBOX_TOKEN`。MCP 由维护者在 social 中固定批准 URL、工具 schema、作用域和效果，模型不能修改这份许可。

## 代理与 GPU

无代理可留空 `XIAOMACHI_HTTP_PROXY` / `XIAOMACHI_HTTPS_PROXY`；QQ/OneBot 保持直连。Docker 构建代理使用独立 `DOCKER_*PROXY`。启用代理需按 [运维指南](../infra/wsl/README.md) 安装验证 WSL 内服务，不能把 Windows 代理端口当成 WSL 中一定可用的 localhost。

`ENABLE_GPU=0` 可用 CPU。启用 NVIDIA 透传后设置 `ENABLE_GPU=1`，向量设备 `auto` 优先 GPU、失败回退 CPU。首次模型下载保持 `MEMORY_EMBEDDING_LOCAL_FILES_ONLY=false`，缓存后才考虑只用本地。
