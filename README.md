# QQ AI Bot

基于 NapCat 的本地 QQ AI Bot 公开版，支持群聊陪聊、私聊记忆、联网搜索、群内生图，以及仅限 `OWNER_QQ` 的项目管理员模式。

## 功能概览

- 群聊自然回复，尽量像群成员而不是客服机器人
- 私聊连续上下文
- 识图、跟图追问、引用图片继续聊
- 文生图和参考图重绘
- 可选实时联网搜索
- 白名单管理员指令
- 仅 `OWNER_QQ` 可进入的项目管理员模式
- Windows 一键启动脚本

## 环境要求

- Windows
- QQ 桌面端
- NapCat / NapCat Shell
- Python `>=3.12`
- 一个 OpenAI 兼容接口

## 快速开始

### 1. 安装依赖

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

### 2. 创建 `.env`

把 `.env.example` 复制成 `.env`，至少先填这些值：

```env
NAPCAT_WS_URL=ws://127.0.0.1:3001

LLM_BASE_URL=https://api.openai.com/v1
LLM_TEXT_ENDPOINT=/chat/completions
LLM_API_KEY=replace-me
LLM_MODEL=gpt-5.4

GROUP_IMAGE_BASE_URL=
GROUP_IMAGE_API_KEY=
GROUP_IMAGE_MODEL=gpt-image-2
GROUP_IMAGE_GENERATIONS_ENDPOINT=/images/generations
GROUP_IMAGE_EDITS_ENDPOINT=/images/edits

BOT_QQ=123456789
OWNER_QQ=987654321
ADMIN_QQS=
PRIVATE_CHAT_QQS=
```

推荐按下面理解这几个关键项：

- `LLM_BASE_URL`
  文本主聊天的 API 根地址。常见写法是 `https://api.openai.com/v1` 或你自己的代理地址 `https://your-host/v1`。
- `LLM_TEXT_ENDPOINT`
  主文本聊天链路。当前公开版统一走 OpenAI 兼容的 `/chat/completions`。
- `LLM_API_KEY`
  文本聊天用的 key。
- `LLM_MODEL`
  主文本聊天模型名，例如 `gpt-5.4`。
- `GROUP_IMAGE_BASE_URL`
  生图接口根地址。留空时复用 `LLM_BASE_URL`。
- `GROUP_IMAGE_API_KEY`
  生图接口 key。留空时复用 `LLM_API_KEY`。
- `GROUP_IMAGE_MODEL`
  生图模型名，默认 `gpt-image-2`。
- `GROUP_IMAGE_GENERATIONS_ENDPOINT`
  文生图接口路径，默认 `/images/generations`。
- `GROUP_IMAGE_EDITS_ENDPOINT`
  参考图重绘或编辑接口路径，默认 `/images/edits`。
- `OWNER_QQ`
  机器人拥有者 QQ。这个 QQ 永远拥有私聊权限，也只有它能进入项目管理员模式。
- `ADMIN_QQS`
  额外管理员 QQ 列表，多个 QQ 用英文逗号分隔。
- `PRIVATE_CHAT_QQS`
  允许私聊机器人的 QQ 列表，多个 QQ 用英文逗号分隔。`OWNER_QQ` 会自动加入，不用重复写。

常用可选项也都已经放进 `.env.example` 了，包括：

- `GROUP_IMAGE_SIZE`
- `GROUP_IMAGE_QUALITY`
- `GROUP_IMAGE_BACKGROUND`
- `GROUP_IMAGE_OUTPUT_FORMAT`
- `GROUP_IMAGE_OUTPUT_COMPRESSION`
- `GROUP_IMAGE_MODERATION`
- `GROUP_IMAGE_QUEUE_CAPACITY`
- `SEARCH_PROVIDER`
- `SEARCH_API_KEY`
- `CONTEXT_RECENT_LIMIT`
- `CONTEXT_SUMMARY_LIMIT`
- `CONTEXT_HISTORY_LIMIT`
- `QQ_EXE_PATH`
- `NAPCAT_SHELL_DIR`
- `NAPCAT_BOOT_PATH`
- `NAPCAT_INJECT_DLL_PATH`

### 3. 检查示例配置

你至少需要看这几个文件：

- `configs/groups.yaml`
- `configs/persona.yaml`
- `configs/private_reminders.yaml`
- `configs/safety.yaml`

使用前建议做这几件事：

- 把 `configs/groups.yaml` 里的示例群号换成你自己的
- 只给你真正想启用的群设置 `enabled: true`
- 只给你真正想让机器人开口的群设置 `speak: true`
- 如果不想使用默认人设，就改 `configs/persona.yaml`

only groups with both `enabled: true` and `speak: true` are ingested

### 4. 启动

前台直接跑：

```powershell
python -m app.main
```

PowerShell 启动脚本：

```powershell
powershell -ExecutionPolicy Bypass -File start_xiaomachi.ps1
```

如果你喜欢双击启动，仓库根目录也带了 `.bat` 启动器。
`启动小町.bat` starts QQ, NapCat, and the Python bot together，`关闭小町.bat` 用来关闭这一套本地启动栈。

## 主聊天与生图接口怎么配

### 主文本聊天

这个公开版的主文本聊天链路已经统一成：

```text
OpenAI-compatible /chat/completions
```

对应配置就是：

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_TEXT_ENDPOINT=/chat/completions
LLM_MODEL=gpt-5.4
```

如果你的供应商不是标准路径，只改 `LLM_TEXT_ENDPOINT` 即可，不需要再改代码。

### 群聊生图

生图链路单独走下面这些变量：

```env
GROUP_IMAGE_BASE_URL=
GROUP_IMAGE_API_KEY=
GROUP_IMAGE_MODEL=gpt-image-2
GROUP_IMAGE_GENERATIONS_ENDPOINT=/images/generations
GROUP_IMAGE_EDITS_ENDPOINT=/images/edits
```

说明：

- 想让文生图和主聊天共用同一个 host/key，就把 `GROUP_IMAGE_BASE_URL` 和 `GROUP_IMAGE_API_KEY` 留空
- `GROUP_IMAGE_GENERATIONS_ENDPOINT` 负责纯提示词生成
- `GROUP_IMAGE_EDITS_ENDPOINT` 负责拿已有图片做重绘或编辑
- 如果你的接口根地址已经写成 `.../v1`，这里通常仍然写 `/images/generations` 和 `/images/edits`

## QQ 权限配置说明

这三个变量的作用不一样：

### `OWNER_QQ`

- 永远拥有私聊权限
- 永远拥有管理员指令权限
- 只有它可以进入项目管理员模式
- 只有它能持续和机器人进行“查仓库 / 改代码 / 跑测试 / 重启交接”这类项目对话

### `ADMIN_QQS`

- 是额外管理员白名单
- 这些 QQ 可以使用白名单管理员指令
- 但它们不会自动获得私聊权限
- 它们也不会自动获得 `OWNER_QQ` 的项目管理员模式能力

如果你希望某个管理员既能用管理员指令，又能和机器人私聊，就要同时把它写进：

- `ADMIN_QQS`
- `PRIVATE_CHAT_QQS`

### `PRIVATE_CHAT_QQS`

- 这是允许和机器人私聊的 QQ 白名单
- `OWNER_QQ` 自动包含在内
- 其它 QQ 必须显式写进来才可以私聊

如果你后续打算让 `OWNER_QQ` 在管理员模式里要求机器人“给某个 QQ 发私聊”，目标 QQ 也需要在这个白名单里。

示例：

```env
OWNER_QQ=10001
ADMIN_QQS=10002,10003
PRIVATE_CHAT_QQS=10002,20001,20002
```

这表示：

- `10001` 是 owner
- `10002` 和 `10003` 能用管理员指令
- `10002`、`20001`、`20002` 能和机器人私聊
- `10002` 同时拥有“管理员指令 + 私聊权限”
- `10003` 只有管理员指令，没有私聊权限

## 管理员指令怎么用

白名单管理员指令不走 LLM 解析，只对白名单生效。

当前可用的管理员指令有：

- `/bot status`
- `/bot on`
- `/bot off`
- `/bot group allow <group_id>`
- `/bot group deny <group_id>`

其中：

- `/bot status` 用来查看当前状态
- `/bot on` 和 `/bot off` 用来开关机器人
- `/bot group allow <group_id>` 与 `/bot group deny <group_id>` 需要在私聊里用

## 项目管理员模式怎么用

这部分和上面的“管理员指令”不是一回事。

项目管理员模式只对 `OWNER_QQ` 开放，用法如下：

### 进入管理员模式

给机器人私聊：

```text
启动管理员模式
```

进入后，后续这条私聊会变成一个独立的“项目对话”上下文，和普通日常私聊分开记。

### 退出管理员模式

这几个口令都可以：

```text
结束管理员模式
退出管理员模式
关闭管理员模式
```

### 进入后能做什么

- 查当前仓库代码
- 看配置和运行脚本
- 修改代码
- 跑定向测试
- 在可用脚本范围内做重启交接

### 进入后怎么提需求

示例：

```text
启动管理员模式
把群聊里的生图触发补全一下，顺手跑相关测试
```

或者不切长期模式，直接单条前缀触发：

```text
管理员权限 帮我查一下 private chat 白名单里有没有 20002
```

更直观的完整例子：

```text
Owner: 启动管理员模式
Bot: 好，已经切到管理员模式了。接下来这条私聊会进入项目对话。
Owner: 把群聊修图触发补全一下，跑相关测试，改完后重启生效
Bot: 我先补触发分支和回归测试，跑完后给你结果。
```

## 运行模式

- `python -m app.main`
  全功能运行，包含群聊、私聊、管理员能力
- `python -m app.group_main`
  只跑群聊
- `python -m app.private_main`
  只跑私聊
- `python -m app.dev_worker_main`
  只跑本地开发 worker
- `powershell -ExecutionPolicy Bypass -File start_xiaomachi_runtime.ps1`
  分进程启动脚本
- `powershell -ExecutionPolicy Bypass -File scripts/install_service.ps1`
  安装为 Windows 服务

## 配置文件说明

### `configs/persona.yaml`

这里控制：

- 名字
- 人设
- 说话风格
- 句子长短
- 口头习惯
- 禁止词和回避表达

### `configs/groups.yaml`

按群配置：

- `enabled`
- `archive`
- `speak`
- `proactive_reply`
- `proactive_interval_seconds`

默认策略是白名单外一律不启用。

### `configs/private_reminders.yaml`

用于私聊提醒，例如：

- 起床提醒
- 一次性待办提醒
- 每日固定提醒

### `configs/safety.yaml`

用于约束：

- 敏感内容处理
- prompt 泄露防护
- 语气边界
- 回复限制

## 搜索与上下文相关配置

README 和 `.env.example` 中已经放了这些常用项：

- `SEARCH_PROVIDER`
- `SEARCH_BASE_URL`
- `SEARCH_API_KEY`
- `SEARCH_TIMEOUT_SECONDS`
- `SEARCH_REGION`
- `SEARCH_BACKEND`
- `CONTEXT_RECENT_LIMIT`
- `CONTEXT_SUMMARY_LIMIT`
- `CONTEXT_HISTORY_LIMIT`

说明：

- `SEARCH_PROVIDER=ddgs` 时通常可以不填 `SEARCH_API_KEY`
- `SEARCH_PROVIDER=tavily` 时需要填 `SEARCH_API_KEY`

## 一键启动路径相关配置

如果脚本自动探测不到 QQ 或 NapCat，可以在 `.env` 中手动设置：

- `QQ_EXE_PATH`
- `NAPCAT_SHELL_DIR`
- `NAPCAT_BOOT_PATH`
- `NAPCAT_INJECT_DLL_PATH`
- `NAPCAT_WAIT_TIMEOUT_SECONDS`

## 仓库结构

- `app/`
  运行时逻辑、路由、模型客户端、存储、搜索和管理员能力
- `configs/`
  示例配置文件
- `scripts/`
  Windows 启动、服务安装和辅助脚本
- `tests/`
  回归测试与 smoke tests
- `data/`
  公开版只保留占位目录，不带真实运行数据

## 公开版安全说明

这个 GitHub release 版本是清理过的公开版，默认不包含：

- `.env`
- 本地数据库
- 群聊历史归档
- 私聊历史
- runtime 日志
- 图片缓存
- 本地 dev-control 状态

保留在仓库里的只有示例文件：

- `.env.example`
- `configs/` 下的示例配置

## 限制与说明

- 这是本地自部署项目，不是 SaaS
- 你仍然需要先把 QQ 和 NapCat 跑通
- 生图和识图效果依赖模型能力、接口兼容度和你给的上下文
- 本项目不是腾讯官方项目，也不是 NapCat 官方项目
