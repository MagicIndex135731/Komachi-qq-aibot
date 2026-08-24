# 小町 QQ AI Bot

小町是一个面向 QQ 群聊的 AI 机器人。她能以稳定的人格参与群聊，理解文字与图片，按需联网检索，并通过分层记忆持续认识群成员和群内发生的事情。

项目目前以 **Windows 11 + WSL2 + Docker + LLBot** 为主要运行环境，NapCat 可作为备用 QQ 接入方式。文本模型支持 OpenAI 兼容的 Responses 或 Chat Completions 接口；NVIDIA GPU 为可选项。

> 这是一个仍在持续开发的个人项目。部署前请确认你了解 QQ 机器人、模型 API 和本地数据保存带来的风险。

## 核心能力

- **自然参与群聊**：支持被 @、被回复、被提及时答话，也能按群策略主动插话。
- **长期记忆**：保存群聊事件、事实、人物画像和关系，并结合近期对话与长期记忆回答问题。
- **图片理解**：读取消息中引用或最近出现的图片，同时保留图片发送者等上下文信息。
- **联网与多模态工具**：支持模型内置联网检索、图片生成与图片编辑。
- **按群独立配置**：分别控制监听、发言、主动回复、归档、记忆和图片能力。
- **可运维运行栈**：提供启动、停止、状态检查、会话看护、故障恢复和数据持久化能力。

想了解一条消息如何经过 QQ 接入、上下文组装、记忆检索、模型调用和消息投递，请阅读[工程架构与运行原理](docs/ARCHITECTURE.md)。

## 运行要求

- Windows 11
- 已启用 systemd 的 WSL2 Ubuntu
- WSL 内可用的 Docker Engine 与 Docker Compose
- 一个 QQ 机器人账号
- 一个 OpenAI 兼容模型接口（推荐使用 Responses API）
- 可选：NVIDIA GPU、Windows 驱动和 NVIDIA Container Toolkit

当前唯一受支持的生产运行栈位于 [`infra/wsl`](infra/wsl/README.md)。其他操作系统或部署方式可以自行适配，但不在现有脚本的保证范围内。

## 快速开始

### 1. 克隆项目

在 Windows PowerShell 中执行：

```powershell
git clone https://github.com/MagicIndex135731/Komachi-qq-aibot.git
cd Komachi-qq-aibot
```

### 2. 初始化 WSL 环境

在 WSL 中进入项目目录并执行：

```bash
bash infra/wsl/scripts/bootstrap_wsl.sh
```

脚本会创建本地运行目录、`infra/wsl/.env` 和看护程序所需的 Python 环境。完整的初始化和运行栈说明见 [WSL/Docker 部署文档](infra/wsl/README.md#初始化)。

### 3. 填写必要配置

编辑 `infra/wsl/.env`，至少确认以下内容：

```dotenv
BOT_QQ=机器人QQ号
OWNER_QQ=管理员QQ号
QQ_PLATFORM=llbot

LLM_BASE_URL=https://你的模型接口地址
LLM_API_KEY=你的API密钥
LLM_MODEL=你的模型名称
LLM_TEXT_ENDPOINT=responses
```

不要把 `.env`、API 密钥、QQ 密码、WebUI Token 或验证码链接提交到 Git。

然后从 `configs/groups.yaml` 复制一份本地群配置：

```powershell
Copy-Item configs/groups.yaml configs/groups.local.yaml
```

将 `configs/groups.local.yaml` 中的占位群号替换为真实群号，并按需启用 `enabled`、`speak`、`memory_enabled` 等选项。这个本地文件不会进入 Git。

### 4. 启动并登录 QQ

在资源管理器中双击：

```text
start-xiaomachi-wsl.bat
```

启动窗口会显示构建、QQ 平台、OneBot、小町容器、可选 CUDA 预热和网关就绪进度。窗口正常自动关闭时，小町已经可以接收消息；启动失败时窗口会保留错误信息。

LLBot 默认 WebUI 地址为 <http://127.0.0.1:3080/>。也可以双击 `open-llbot-webui.bat` 打开页面并完成首次登录。

### 5. 检查状态

双击 `status-xiaomachi-wsl.bat`。正常状态应包括：

- QQ 平台容器健康；
- OneBot 已连接；
- 小町心跳和消息网关已就绪；
- 至少存在一个允许发言的本地群策略。

如果状态检查未通过，请先查看[常见问题](#常见问题)和 [WSL/Docker 验收说明](infra/wsl/README.md#验收)。

## 日常操作

| 入口 | 用途 |
| --- | --- |
| `start-xiaomachi-wsl.bat` | 显示启动和预热进度，全部就绪后自动关闭窗口 |
| `stop-xiaomachi-wsl.bat` | 停止小町、QQ 接入和后台看护服务 |
| `status-xiaomachi-wsl.bat` | 检查 systemd、容器、OneBot、心跳和群策略 |
| `open-llbot-webui.bat` | 打开默认 LLBot 管理页面 |
| `open-napcat-webui.bat` | 使用 NapCat 备用接入时打开其管理页面 |

后台运行由 WSL systemd 和 Windows 登录级计划任务共同维持，不需要保留一个空白 CMD 窗口。启动入口、服务关系和日志位置见 [WSL/Docker 启动链路](infra/wsl/README.md#启动链路)。

## 配置指南

### 群与人格

| 文件 | 用途 | 是否应提交 |
| --- | --- | --- |
| `configs/groups.yaml` | 群策略模板和安全默认值 | 是 |
| `configs/groups.local.yaml` | 真实群号及本机群策略 | 否 |
| `configs/persona.yaml` | 小町的人格、口吻和行为风格 | 是 |
| `configs/safety.yaml` | 内容和操作安全限制 | 是 |
| `infra/wsl/.env` | 密钥、账号和运行参数 | 否 |

一个群只有同时设置 `enabled: true` 和 `speak: true` 才允许小町发言。`memory_enabled: true` 会为该群启用完整记忆；关闭时只使用近期上下文，不生成或检索长期记忆。

群策略的字段示例见 [`configs/groups.yaml`](configs/groups.yaml)，运行时的数据边界见[架构文档：配置与数据边界](docs/ARCHITECTURE.md#9-配置与数据边界)。

### 记忆系统

Memory V3 将原始消息、话题片段、结构化事实、人物画像和关系记忆分层保存，并使用关键词、时间、语义向量和来源证据进行混合检索。记忆按群隔离，写入和召回都保留来源约束。

- [Memory V3 构成与原理](docs/ARCHITECTURE.md#6-memory-v3-的构成与原理)
- [Memory V3 发布与回滚](docs/ARCHITECTURE.md#7-memory-v3-发布与回滚)
- [按群记忆策略与日常维护](infra/wsl/README.md#按群记忆策略与日常维护memory-v3)

### GPU 加速

GPU 只用于小町本地向量模型，不分配给 LLBot。没有 NVIDIA GPU 时保持：

```dotenv
ENABLE_GPU=0
MEMORY_EMBEDDING_DEVICE=auto
```

满足 NVIDIA Container Toolkit 和 CDI 设备条件后可以设置 `ENABLE_GPU=1`。首次启用、模型缓存和故障回退步骤见 [CUDA 向量加速](infra/wsl/README.md#cuda-向量加速)。

### 上游代理

小町可以通过 WSL 内独立的 Mihomo 实例访问模型接口，而 QQ、OneBot 和 Windows 系统代理保持互不影响。代理不是运行必需项；只有上游接口确实需要时才配置。

详见 [WSL 内置 Mihomo 上游代理](infra/wsl/README.md#wsl-内置-mihomo-上游代理)。

## 数据与安全

以下内容包含登录态、聊天数据或运行状态，请勿提交、公开或随意删除：

- `infra/wsl/.env`
- `configs/groups.local.yaml`
- `infra/wsl/runtime/llbot/data`
- `infra/wsl/runtime/napcat/ntqq`
- `infra/wsl/runtime/napcat/config`
- `infra/wsl/runtime/logs`
- 小町 SQLite 数据库与模型缓存卷

更新代码或重建镜像前应先备份数据库和登录态。更完整的数据边界、备份和发布要求见 [WSL/Docker 运行态保护](infra/wsl/README.md#运行态保护)和[工程架构](docs/ARCHITECTURE.md#9-配置与数据边界)。

## 常见问题

### 启动窗口一直不关闭

启动脚本会等待 QQ 平台、OneBot、小町心跳、消息网关和可选 CUDA 预热全部完成。窗口长期停留通常表示某项就绪检查失败，请保留窗口中的错误信息并运行 `status-xiaomachi-wsl.bat`。

### 小町在线但群里不回复

检查该群是否已经写入 `configs/groups.local.yaml`，并确认同时启用了 `enabled` 和 `speak`。修改本地群策略后需要按部署文档重新安装或发布运行版本，不能只修改工作区文件。

### LLBot 或 NapCat 要求重新登录

打开当前平台对应的 WebUI 完成登录，然后重新运行状态检查。不要删除平台数据目录，否则会丢失已保存的登录态。

### 模型请求失败或很慢

先区分上游接口耗时、本机网络代理、模型工具调用和本地记忆检索。相关日志、代理拓扑和排查入口见 [WSL/Docker 运行文档](infra/wsl/README.md)。

## 深入了解

- [工程架构与运行原理](docs/ARCHITECTURE.md)：消息处理、模型工具、Memory V3、进程拓扑和数据边界。
- [WSL/Docker 部署与运维](infra/wsl/README.md)：安装、启动、代理、GPU、健康检查、发布和回滚。
- [`configs/groups.yaml`](configs/groups.yaml)：群策略完整示例。
- [`infra/wsl/.env.example`](infra/wsl/.env.example)：运行参数模板。

## 开发

项目要求 Python 3.12。创建开发环境后安装依赖：

```bash
python -m pip install -e ".[dev]"
pytest
```

提交修改前请至少运行受影响测试，并检查 Docker Compose 配置和文档链接。项目内部设计约束与各层职责以 `.trellis/spec/` 和 [工程架构文档](docs/ARCHITECTURE.md)为准。

## License

本项目使用 [MIT License](LICENSE)。
