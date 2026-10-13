# Windows / WSL2 / Docker 部署运维

用途：操作指南；读者：首次部署者与维护者；维护者：项目维护者。支持栈为 Windows + WSL2 Ubuntu/systemd + WSL 内 Docker Engine/Compose；不要混用另一套 Docker 数据根或把 Windows 运行缓存作为生产卷。

项目介绍见 [README](../../README.md)，配置见 [CONFIGURATION](../../docs/CONFIGURATION.md)，原理见 [ARCHITECTURE](../../docs/ARCHITECTURE.md)。

## 1. 运行要求

- Python 3.12、Git、Linux `python3-venv`、tar 和 systemd。
- WSL 内 Docker Engine/Compose，root 可调用 Docker；`systemctl show-environment` 应成功。
- 可用模型 API 和一个用于机器人运行的 QQ 账号。
- 足够磁盘容纳 CUDA 基础镜像、QQ 客户端和持久数据；GPU 可选。

WSL 的 `/etc/wsl.conf` 需启用 systemd：

```ini
[boot]
systemd=true
```

修改后在 Windows 执行 `wsl --shutdown` 再进入 WSL。此命令影响运行中的 WSL 服务，应在安装前或受控维护时使用。Windows BAT 入口使用默认发行版，先用 `wsl -l -v`、`wsl --set-default <发行版名称>` 确认目标。

在 WSL 检查基础环境：

```bash
python3 --version
docker version
docker compose version
systemctl show-environment
```

本栈 Compose 使用 host 网络；不依赖 bridge 端口映射。QQ 管理页、OneBot 和侧车端口仍需本机/防火墙保护。

## 2. 初始化配置

在 Windows PowerShell 的项目目录：

```powershell
Copy-Item infra/wsl/.env.example infra/wsl/.env
Copy-Item configs/groups.yaml configs/groups.local.yaml
```

只在本地 `.env` 填写机器人/主人账号、模型地址/密钥/实际模型名、QQ 平台和 VNC 密码。公开模板不包含 `BOT_QQ` / `OWNER_QQ` 的实际值，需要自行添加。`LLM_TEXT_ENDPOINT` 选择服务确实支持的 `responses` 或 `chat_completions`。

启用自主空间须同时在源目录 `infra/wsl/.env` 设置 `SOCIAL_AGENT_ENABLED=true`，并在私有 `configs/social.local.yaml` 开启总策略和相应 `qzone.publish` / `qzone.comment`。环境项缺失按关闭处理；仅策略开启、群聊能回复或 heartbeat 正常，不能证明空间巡检已运行。修改后 rebuild；检查实际空间任务和未来巡检时间，额度是上限，Agent仍可沉默。

新数据库先设置 `MEMORY_RAW_V3_ENABLED=false`，按 [记忆与索引流程](../../docs/MEMORY.md) 完成准备/评测/激活。可先开启按群记忆基础流程、分层工具和本地 embedding；这不等于 raw V3 已有合格活动代际。

完整基础编排使用 `MEMORY_ORCHESTRATION_V2_ENABLED=true`。Memory V3 发布与回滚的唯一命令来源为 [记忆指南](../../docs/MEMORY.md#raw-v3-准备评测激活与回滚)，不能跳过其质量门。

`configs/groups.local.yaml` 的群表会整体替换占位模板，删除不需要的示例群并加入自己的真实群。每群至少确认 `enabled`、`speak`、`archive`、`memory_enabled`、`image_generation` 和主动间隔。私聊按主人/固定名单/可选动态好友许可设置。

在 WSL 进入克隆目录：

```bash
bash infra/wsl/scripts/bootstrap_wsl.sh
```

bootstrap 创建初始本地运行目录、缺失的 `.env` 和持久 watchdog Python 环境；它不负责安装 Docker/systemd，也不替你填写账号密钥。

## 3. 首次安装与 QQ 登录

从 Windows 双击或在 PowerShell 执行：

```powershell
.\start-xiaomachi-wsl.bat
```

首次调用安装脚本，将运行源码/配置复制到 `/opt/xiaomachi/releases/<id>`，准备共享环境和数据卷，构建镜像并启用 systemd。之后正常 start 复用已安装镜像，缺失才构建。

| 平台 | QQ_PLATFORM | 管理入口 | 主要端口 |
|---|---|---|---|
| SnowLuma | `snowluma` | `open-snowluma-webui.bat`、`open-snowluma-desktop.bat` | WebUI 5099、noVNC 6081、VNC 5900、OneBot 3000/3001 |
| LLBot | `llbot` | `open-llbot-webui.bat` | WebUI 3080、WS 默认 3002 |
| NapCat | `napcat` | `open-napcat-webui.bat` | WebUI 6099、WS 3001 |

平台容器分别为 `xiaomachi-snowluma`、`xiaomachi-llbot` 和 `xiaomachi-napcat`。

一次只运行一个平台登录同一个 QQ。三个平台的登录态卷彼此独立；切换后需要对应平台登录。

### SnowLuma

Compose 固定使用上游维护的 `motricseven7/snowluma` 镜像及 digest。使用者需阅读发行包 EULA/隐私协议后确认相关变量。WebUI 初始密码由启动生成，入口脚本可帮助取得本机初始密码；登录后修改为自己的密码。

打开桌面入口完成手机 QQ 扫码，再在 WebUI 确认 OneBot 11 HTTP/正向 WebSocket 已启用。当前应用连接 `ws://127.0.0.1:3001`，配置默认不发送 access token；采用这个方案时清空 SnowLuma OneBot 连接的 token，并把端口限制在可信本机网络。不要把 WebUI 登录 token 与 OneBot token 混淆。

冷启动/重启 QQ 容器可能再次要求扫码；不要承诺持久卷能免除任何平台登录要求。watchdog 对“等扫码”先通知而不是重复重启，避免二维码不断变化。只有人工完成登录后才能通过在线检查。

### LLBot 与 NapCat

使用对应 WebUI 扫码或按平台支持的登录方式设置。LLBot 初始化由 `bootstrap_llbot_runtime.py` 写入自己的卷；NapCat OneBot 默认配置由初始化脚本生成。镜像/签名构建辅助脚本只在选用该平台时需要，见 `build_llbot_from_source.sh --help`；不要同时启动多个平台占用账号或端口。

## 4. 常用入口与返回值

| 入口 | 行为 |
|---|---|
| `start-xiaomachi-wsl.bat` | 首次安装，之后启动当前 release 的运行栈 |
| `rebuild-xiaomachi-wsl.bat` | 从当前工作区发布并重建群/私应用 |
| `status-xiaomachi-wsl.bat` | 读取实际运行实例状态 |
| `stop-xiaomachi-wsl.bat` | 先停看护和已启用计算，再停应用/QQ 栈 |

状态返回 0 表示通过；75 表示 QQ 暂时离线但仍在恢复；其他非零需查看错误。等待登录不是模型故障。日常检查不调用模型，深检查至多发起一次受限文本请求：

```bash
bash /opt/xiaomachi/current/infra/wsl/scripts/status.sh
bash /opt/xiaomachi/current/infra/wsl/scripts/status.sh --deep
```

深检查可能收费，只检查受限模型/人格文件契约，不替你执行空间/绘图真实写入。

安装后在 WSL 使用稳定入口：

```bash
/usr/local/bin/xiaomachi-wsl-entry start
/usr/local/bin/xiaomachi-wsl-entry status
/usr/local/bin/xiaomachi-wsl-entry stop
```

不要把工作区中尚未部署的 start/status 脚本当成现役发布。

重建在候选镜像中执行 `python -B -m scripts.check_public_persona`，通过后才激活。检查真实人格前16条例句完整进入公开渲染，并用合成样本发现字段遗漏与成员原话泄漏；超长配置导致例句被截断也会失败。失败保留现役应用并恢复原镜像标签。日常 status 同样检查实际加载的公开人格，只输出计数与错误类别，不调用模型。可在源码目录独立运行该命令；修改人格配置后必须重新通过，不能仅凭镜像构建成功认定生效。

## 5. 开机恢复与监督

Windows 登录后保持 WSL 活跃需要登录级任务。在 Windows PowerShell 安装：

```powershell
powershell -ExecutionPolicy Bypass -File infra/wsl/scripts/install_windows_runtime_task.ps1
```

任务通过隐藏锚点启动/维持 systemd，不需要一直保留空白命令窗口。显式 stop 会撤销运行标记，避免锚点立即重启。QQ 平台登录仍可能需要人工操作。

关键服务是 `xiaomachi-stack.service`、`xiaomachi-watchdog.service`、定期 `xiaomachi-memory-audit.timer`，以及按需启用的 `xiaomachi-sandbox.service` / `xiaomachi-mihomo.service`。

```bash
systemctl status xiaomachi-stack xiaomachi-watchdog --no-pager
journalctl -u xiaomachi-stack -n 80 --no-pager
journalctl -u xiaomachi-watchdog -n 80 --no-pager
docker logs --tail 80 xiaomachi-bot
docker logs --tail 80 xiaomachi-private
```

日志可能包含私有资料或请求错误细节，分享前需脱敏。查看服务 active 还不够，应同时确认 OneBot、当前实例心跳和实际消息回执。

## 6. 更新与回滚

修改人格 YAML、代码、`.env` 或本地策略后，从源项目 Windows 目录执行：

```powershell
.\rebuild-xiaomachi-wsl.bat
```

无需先停止。安装脚本复制运行白名单（app/configs/infra/scripts/依赖文件等），先构建新镜像，然后仅重建两个应用，保留现役 QQ 接入容器和数据卷。成功后检查 readiness。**未提交源码也会一起发布**，这不是从 GitHub 拉取最新版的命令。

Linux 等价入口为 `bash infra/wsl/scripts/install_linux_runtime.sh`；更新 Compose 服务 `xiaomachi` 与 `xiaomachi-private`，不重启当前 QQ 接入容器。

这里的 **rebuild 是应用镜像发布，不是驱动安装**：它把当前工作区的 `app/`、`configs/` 和运行脚本复制到新 release，再执行 `docker compose build xiaomachi`。Dockerfile 的 CUDA 基础镜像只在本机没有可复用层、或构建器需要重新解析标签时才会访问镜像仓库；宿主 NVIDIA 驱动、CUDA 运行时、SnowLuma/LLBot/NapCat 容器和数据卷不会因为 rebuild 被重新安装。`start` 只复用现有镜像，因此不会让已修改的代码自动生效。当前发布入口没有把应用文件热加载到运行容器内，修改人格 YAML 或 Python 代码仍应走受控 rebuild；若基础镜像仓库暂时不可用，应先保留现役容器并恢复网络/缓存，再发布应用层变更。

构建/激活失败会尝试恢复之前的 release 和镜像；保留最近一个发布目录及回滚镜像。脚本不是数据库回滚工具，不会撤销已经执行的 QQ 行为。恢复失败时检查 current、容器 image ID 和健康后再操作，不重复执行未知状态的外部动作。

只执行 start/stop 不会同步源码变化。修改生产 `shared/.env` 后容器须重新创建，单纯重启进程不会更新创建时的环境。正常维护优先修改源目录私有配置并 rebuild。

切换 QQ 平台需要在 `.env` 修改 `QQ_PLATFORM` 后受控停止/启动并更新部署配置，准备新平台登录；不要在持续业务中反复试多个平台。

## 7. 数据与备份

| 路径/卷 | 内容 |
|---|---|
| `/opt/xiaomachi/current` | 现役发布软链接 |
| `/opt/xiaomachi/releases/` | 运行源码和配置发布目录 |
| `/opt/xiaomachi/shared/.env`、`shared/configs/` | 私有共享环境、群配置 |
| `/opt/xiaomachi/shared/runtime/` | 看护日志、锁与运行标记 |
| `xiaomachi-bot-data` → `/workspace/data` | SQLite、聊天归档、记忆人格、图像、模型、会话与任务 |
| 平台专属卷 | QQ/WebUI/客户端登录状态 |
| `wsl_sandbox_data`（启用时） | 计算侧车持久运行数据 |

生产数据不用 Windows bind mount。曾经的 `infra/wsl/runtime/` 迁移目录不是现役数据库；只有确认卷迁移与备份后才清理旧文件。

创建一致 SQLite 备份（容器中执行）：

```bash
docker exec xiaomachi-bot python -m scripts.backup_memory_v2 \
  --database /workspace/data/bot.db \
  --backup-dir /workspace/data/backups --tag manual
```

验证输出 `integrity_check=ok`，保存生成的 manifest 与备份；真实数据不要上传 GitHub。其他账号卷另行备份。记忆索引的 prepare/activate/rollback 见 [MEMORY](../../docs/MEMORY.md)，代码升级不等于索引激活。

## 8. GPU 与模型缓存

基础镜像包含 CUDA/cuDNN 和 fastembed-gpu，但 GPU 设备是可选。无 NVIDIA 设置 `ENABLE_GPU=0`、`MEMORY_EMBEDDING_DEVICE=auto`，用 CPU；启用需 Windows NVIDIA 驱动、WSL GPU 支持和 NVIDIA Container Toolkit/CDI。

```bash
nvidia-smi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

确认透传后启用 `ENABLE_GPU=1` 并 rebuild；GPU overlay 只给群应用，不给 QQ 客户端。`auto` 遇 CUDA 推理错误可回退 CPU。首次模型下载需联网，缓存放持久 `/workspace/data/models`；缓存完才使用 `MEMORY_EMBEDDING_LOCAL_FILES_ONLY=true`。

GPU overlay 文件为 `docker-compose.gpu.yml`，挂载 CDI 设备 `nvidia.com/gpu=all`。向量通道故障可暂设 `MEMORY_EMBEDDING_PROVIDER=disabled` 保留 FTS，这属于降级，不代表完整向量检索正常。

## 9. 可选代理

上游可直连就留空 `XIAOMACHI_HTTP_PROXY`、`XIAOMACHI_HTTPS_PROXY`。QQ/OneBot 与应用模型代理分离；安装器通过 `runtime_egress.py` 判断是否启用 Mihomo，默认不要求代理服务。

需要代理时，使用提供的 `install_mihomo.sh`、`sync_mihomo_from_clash_verge.ps1` 和 `render_mihomo_config.py` 准备本机配置，再验证 `xiaomachi-mihomo.service` 与实际上游可达性。代理订阅/节点/Token 都是私有数据；不要复制到公开示例。

Docker 下载依赖使用 `DOCKER_HTTP_PROXY` / `DOCKER_HTTPS_PROXY`，应用运行用 `XIAOMACHI_*`。host 网络下 `127.0.0.1` 指向 WSL 主机；Windows 本地代理并不自动成为 WSL 的同一端口。改变出口后需重新创建应用并核对模型请求。

## 10. 可选隔离计算

默认关闭。先配置固定的 localhost endpoint 和私有 Token。应用 `.env` 与 root 可读的 `/opt/xiaomachi/shared/sandbox.env` 使用相同 `XIAOMACHI_SANDBOX_TOKEN`，不得输出或放进提示词。

本机 endpoint 为 `http://127.0.0.1:8217`，在私有 social 覆盖设置 `compute.endpoint`。

在现役发布目录构建：

```bash
cd /opt/xiaomachi/current
docker compose --env-file /opt/xiaomachi/shared/sandbox.env \
  -f infra/wsl/docker-compose.sandbox.yml --profile sandbox-build --profile sandbox \
  build sandbox-base xiaomachi-sandbox
docker image inspect xiaomachi-sandbox-base:py312 --format '{{.Id}}'
```

计算代码 `app/agent/python_sandbox.py` 固定执行基础镜像内容 ID。新机器自行构建得到不同 ID 时，维护者须核对基础镜像后同步该常量，再构建 gateway 和应用；不应绕过固定镜像校验。

```bash
install -m 0644 infra/wsl/systemd/xiaomachi-sandbox.service \
  /etc/systemd/system/xiaomachi-sandbox.service
systemctl daemon-reload
systemctl enable --now xiaomachi-sandbox
```

再在私有 social 配置开启 compute 并 rebuild。侧车 service 使用预建镜像，start/stop/status 会覆盖已启用侧车。认证失败、镜像缺失或执行不可用应返回 unavailable，不用宿主 Python 替代。

## 11. 常见问题

| 现象 | 核查顺序 |
|---|---|
| @ 没回复 | QQ 是否登录/OneBot连接；群enabled/speak；来源年龄/撤回；回合超时与发送回执 |
| 私聊不能执行空间动作 | 是否真实主人；social表面/好友许可；原帖读取；工具与QQ拒绝原因 |
| 图片没看见 | 视觉开关、真实下载像素、引用来源和图片数量/字节预算 |
| 搜索只有链接 | SEARCH_PROVIDER是否支持正文桥接；模型是否实际调用open/find；总预算是否已用完 |
| 记忆为空/旧日期 | 群memory_enabled；后台索引状态；V3活动代际；来源日期与上海当前日期 |
| 人格没变化 | 编辑的是实际YAML还是说明快照；是否rebuild；当前群选择及.live覆盖 |
| 启动一直等QQ | 先完成扫码，读取平台日志，不反复删除登录卷 |
| GPU不可用 | 驱动/CDI/模型缓存；CPU回退；不要删除数据库换取“修复” |

模型/API限制、网页反爬、桥接动作不支持或QQ拒绝均可能导致工具失败。错误结果不代表动作成功；未知回执不要再次提交同一个动作。

## 12. 空间清理

用 `docker system df`、`docker ps -a`、当前 `readlink -f /opt/xiaomachi/current` 和目录统计确认消费方，再清理过期 release、历史镜像、闲置构建缓存和已结束测试临时数据。不得一键删除现役数据卷或计算基础镜像。

测试临时产物在源码 `data/test-work/`，使用 `scripts/clean_test_tmp.ps1` 预览后按需清理。必要验收报告保留在私有工程 data 中，根目录不再创建测试缓存目录。

WSL 内删除文件后，Windows VHD 可能不立即缩小。压缩需要先核对发行版/VHD路径、停止服务、trim、退出WSL，再使用 Windows 支持的虚拟磁盘压缩操作并恢复启动/状态验证。不要在容器写数据库时对挂载中的 VHD 操作。
