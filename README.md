# 小町 Komachi QQ AI Bot

小町是一个长期运行的 QQ 角色 Agent：在授权群里观察聊天、理解图片、检索历史，决定接话还是保持安静，也能通过消息接受绘图、联网查询和主人委派的行动。

当前工程使用 **Python 3.12 + DeepSeek Harness SDK 0.1.5rc1**，支持 **Windows + WSL2 + Docker**。群聊与私聊分别运行，QQ 通过 SnowLuma、LLBot 或 NapCat 的 OneBot 11 接入。文本、搜索和绘图服务由部署者自行配置。

## 能做什么

| 能力 | 当前实现 |
|---|---|
| 群聊与私聊 | 按群授权、直接 @ 和引用回复、好友准入、独立进程与持久发送记录 |
| 自主参与 | 每群观察与批处理、模型自主选择回复/等待/沉默、持久目标与到期唤醒 |
| 长期记忆 | 原始消息、话题片段、摘要、事实与人物资料；全文/向量/时间混合检索并展开来源 |
| 人格与模仿 | 默认角色、风格变体、来源约束的成员人格、动态事实和表达样本更新 |
| 图片与转发 | 群/私聊看图、引用图片、生成/编辑图片；群内合并转发按页展开并读取节点图片 |
| 联网资料 | 搜索、网页正文、分页和关键词查找；支持读取 GitHub 页面与指定源码文件 |
| QQ 行动 | 授权后发布文字/图片空间、读取好友动态并评论、修改允许的个人资料 |
| 隔离计算 | 可选 Docker Python 计算侧车，限制资源、网络、输入和产物 |

**功能存在不代表新安装全部启用。** 群、好友、社交、计算和写入权限由本地配置控制；公开模板默认关闭主动空间与计算。工具是否可用还取决于 QQ 桥接、模型接口和授权。

## 为什么这样设计

- **带来源的长期记忆**：回答可追溯到实际消息；撤回、失效、跨群或未知发送结果不能冒充可靠事实。
- **模型决定行为，代码约束范围**：Harness 负责模型与工具循环；权限、预算、冷却和提交前复验由业务层执行。
- **人格持续生效**：表达风格与事实资料分开，近期旧回复不会被当成必须照抄的台词。
- **能恢复的后台运行**：消息 inbox、目标、工具效果和回执持久化；systemd、watchdog 和 Windows 计划任务负责进程与连接恢复。
- **独立控制成本**：直接互动、主动群聊、空间发布/评论分预算；无任务或无合格机会时等待，模型也可以沉默。

## 开始使用

需要一个 QQ 机器人账号、可用的模型 API、启用 systemd 的 WSL2 Ubuntu，以及 WSL 内的 Docker Engine/Compose。NVIDIA GPU 可选，用于本地向量推理。

在 Windows PowerShell 中：

```powershell
git clone https://github.com/MagicIndex135731/Komachi-qq-aibot.git
cd Komachi-qq-aibot
Copy-Item infra/wsl/.env.example infra/wsl/.env
Copy-Item configs/groups.yaml configs/groups.local.yaml
```

在私有 `infra/wsl/.env` 中填写 `BOT_QQ`、`OWNER_QQ`、`LLM_BASE_URL`、`LLM_API_KEY`、实际可用的 `LLM_MODEL`；选择 `QQ_PLATFORM`。首次安装保持 `MEMORY_RAW_V3_ENABLED=false`，完成索引准备与验收后再激活 V3。

在 `configs/groups.local.yaml` 中替换示例群号，按需要设置 `enabled`、`speak`、`archive`、`memory_enabled` 和 `image_generation`。真实账号和策略不要写进公开模板。

在 WSL 项目目录执行初始化，然后从 Windows 启动：

```bash
bash infra/wsl/scripts/bootstrap_wsl.sh
```

```powershell
.\start-xiaomachi-wsl.bat
```

通过对应平台的管理页完成 QQ 登录和 OneBot 配置。SnowLuma 使用 `open-snowluma-webui.bat` 和 `open-snowluma-desktop.bat`；冷启动可能需要重新扫码。详细步骤见 [部署运维指南](infra/wsl/README.md)。

## 平时怎样使用

在允许发言的群里 @ 小町提问，或引用相关消息/图片：

```text
@小町 昨天群里讨论的活动最终怎么定的？给出对应来源。
@小町 看看这份合并转发，概括几个人分别说了什么。
@小町 根据这张图画一个冬日版本。
@小町 联网查找 requests 的文档，再打开指定源码解释 patch 函数。
```

主人可以通过真实群聊/私聊消息委派已启用的工具，例如“评论某位好友的最新空间”。昵称解析可能有歧义，必要时明确目标。普通聊天文字、转发署名和引用内容不能取得主人权限。

主动群聊受群间隔约束并由 Agent 自己判断是否开口。空间巡检采用随机间隔与夜间静默；推荐策略是每日主动发布最多 1 条、每个好友每日最多评论 1 条，具体值见 [配置参考](docs/CONFIGURATION.md)。主人指令不计入自主行动冷却/次数，仍受授权、单回合资源和平台限制。

## 启动、停止与更新

| Windows 入口 | 作用 |
|---|---|
| `start-xiaomachi-wsl.bat` | 首次安装，之后启动已部署版本 |
| `rebuild-xiaomachi-wsl.bat` | 把当前工作区代码/配置发布为新版本并重建群私应用 |
| `status-xiaomachi-wsl.bat` | 检查 QQ/OneBot、应用、记忆人格和已启用计算服务 |
| `stop-xiaomachi-wsl.bat` | 停止运行栈、计算侧车和看护服务 |
| `open-llbot-webui.bat` / `open-napcat-webui.bat` | 打开备选 QQ 平台管理页 |

修改人格 YAML、配置或代码后使用 **rebuild**；start 不会更新已有镜像。rebuild 会部署未提交的源码改动，并保留持久数据与 QQ 接入容器。日常状态检查不调用模型；`status.sh --deep` 会发起一次受限真实请求。

## 工程构成与文档

```text
app/agent/       Harness、群观察、工具、目标与自主社交
app/core/        记忆、人格、图片、上下文和消息投递
app/adapters/    OneBot、QQ 空间、转发及账户动作
app/providers/   文本模型、网页搜索、向量与图片服务
app/storage/     SQLite 模型、仓储、账本与迁移
app/runtime/     依赖装配和运行组件
configs/        可公开的配置模板与角色定义
infra/wsl/      Docker、systemd、部署与状态脚本
scripts/        回填、评估、审计与维护工具
tests/          使用合成 ID 的隔离测试
```

- [工程架构与运行原理](docs/ARCHITECTURE.md)
- [配置、权限与工具](docs/CONFIGURATION.md)
- [记忆、人格与索引发布](docs/MEMORY.md)
- [WSL/Docker 部署与运维](infra/wsl/README.md)
- [开发与测试](docs/DEVELOPMENT.md)
- [数据保护与安全发布](docs/SECURITY.md)
- [环境配置模板](infra/wsl/.env.example)

## 适用边界

这是一套自行部署的机器人工程。模型可能误解消息，QQ 桥接可能断线或拒绝动作，明确 @ 也不能保证任何情况下都成功。搜索不会自动读完整 GitHub 仓库，图片、动态和历史检索只覆盖已取得且获授权的资料。私聊不提供宿主文件、Shell 或远程项目控制。

聊天数据保存在本地持久卷，但选中的消息、记忆片段和图片会提交给你配置的模型服务。API、绘图和部分记忆/人格更新可能产生费用；启用主动功能前先配置许可与预算。

## 参考与致谢

本项目参考了 [fumingyang2004/Tulpa](https://github.com/fumingyang2004/Tulpa) 关于持续群聊、Agent 工具协作、人格与资料授权边界的设计思路。感谢 Tulpa 作者与贡献者公开分享实现和文档，为小町的重构提供了重要参考。

同时感谢 DeepSeek Harness、OneBot、SnowLuma、LLBot、NapCat、FastEmbed 与 sqlite-vec 等项目。小町自主代码采用 [MIT License](LICENSE)；第三方运行时、客户端、模型和服务遵循各自许可。
