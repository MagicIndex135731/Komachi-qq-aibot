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

近期公开评论记录上线后自动保存；迁移旧记录前先用SQLite在线backup备份并验证integrity_check，再在群容器执行 `python -B -m scripts.backfill_public_social_activity` 预览，加 `--apply` 导入。仅读取当前公开主体原生工具事件并匹配效果账本，不使用监控或群私历史，不发送QQ动作；重复导入幂等，过期或无法核对的评论不猜测补入。

记录保留14天、每账号最多512条，公开巡检与主人公开委派共享按需查询，不预装进每轮上下文。回滚旧应用代码可忽略新增记录表；不要用迁移前数据库覆盖上线后的新消息。行为记录与群长期记忆的分工见[记忆指南](../../docs/MEMORY.md#近期公开社交记录)。

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
| SnowLuma | `snowluma` | `infra/wsl/windows/open-snowluma-webui.bat`、`infra/wsl/windows/open-snowluma-desktop.bat` | WebUI 5099、noVNC 6081、VNC 5900、OneBot 3000/3001 |
| LLBot | `llbot` | `infra/wsl/windows/open-llbot-webui.bat` | WebUI 3080、WS 默认 3002 |
| NapCat | `napcat` | `infra/wsl/windows/open-napcat-webui.bat` | WebUI 6099、WS 3001 |

平台容器分别为 `xiaomachi-snowluma`、`xiaomachi-llbot` 和 `xiaomachi-napcat`。

一次只运行一个平台登录同一个 QQ。三个平台的登录态卷彼此独立；切换后需要对应平台登录。

### SnowLuma

运行镜像由固定 v1.14.15 上游构建本地兼容层（Dockerfile.snowluma）：空间凭据刷新只得到 p_skey 时，补齐当前登录账号的 uin/p_uin，账号冲突拒绝；不保存凭据或伪造认证键。原帖/动态网页读取限制为每次5秒、2MiB，旧接口明确限流才沿用上游备用路由。评论仍需好友/原帖复验，未知写入不重放。首次启动由 Compose 构建兼容镜像；修订兼容层后先 build snowluma，再只重建该服务。

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

若现役容器仍运行但旧镜像元数据已被清理，installer会要求显式的`XIAOMACHI_ROLLBACK_IMAGE_ID`，不以可能指向候选版本的local标签作为旧镜像。先从current旧release重建并核对应用/配置/脚本哈希，再传入该镜像sha256执行installer；数据卷及当前运行容器保持原状。

这里的 **rebuild 是应用镜像发布，不是驱动安装**：它把当前工作区的 `app/`、`configs/` 和运行脚本复制到新 release，再执行 `docker compose build xiaomachi`。Dockerfile 的 CUDA 基础镜像只在本机没有可复用层、或构建器需要重新解析标签时才会访问镜像仓库；宿主 NVIDIA 驱动、CUDA 运行时、SnowLuma/LLBot/NapCat 容器和数据卷不会因为 rebuild 被重新安装。`start` 只复用现有镜像，因此不会让已修改的代码自动生效。当前发布入口没有把应用文件热加载到运行容器内，修改人格 YAML 或 Python 代码仍应走受控 rebuild；若基础镜像仓库暂时不可用，应先保留现役容器并恢复网络/缓存，再发布应用层变更。

构建/激活失败会尝试恢复之前的 release 和镜像；保留最近一个发布目录及回滚镜像。脚本不是数据库回滚工具，不会撤销已经执行的 QQ 行为。恢复失败时检查 current、容器 image ID 和健康后再操作，不重复执行未知状态的外部动作。

仅改应用源码且需冻结依赖时，可从已核验的现役镜像SHA构建增量层；保存临时Dockerfile、源码哈希、依赖清单及回滚镜像，正式Dockerfile保留完整构建。候选基于最新release，只带本次已验收差异；已有未验工作区改动不得顺带上线。

只执行 start/stop 不会同步源码变化。修改生产 `shared/.env` 后容器须重新创建，单纯重启进程不会更新创建时的环境。正常维护优先修改源目录私有配置并 rebuild。

GitHub公开提交和本机运行镜像是两个版本身份。发布源码前应核对候选自身的测试与配置模板；同步仓库不会自动切换当前release，私有群策略、登录会话及数据库也不属于公开发布包。

### 模型套餐快捷切换

首次接入并验收两套接口后，双击根目录 `switch-xiaomachi-model.cmd`，选 `1` 使用原接口 Sol 6.1 low + image 2.5，选 `2` 使用 Gemini 套餐。也可在项目目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File infra/wsl/scripts/switch_model_profile.ps1 -Profile sol
powershell -ExecutionPolicy Bypass -File infra/wsl/scripts/switch_model_profile.ps1 -Profile gemini
powershell -ExecutionPolicy Bypass -File infra/wsl/scripts/switch_model_profile.ps1 -Profile gemini -Reasoning high
powershell -ExecutionPolicy Bypass -File infra/wsl/scripts/switch_model_profile.ps1 -Profile gemini -Check
```

私有套餐在 `infra/wsl/runtime/model-profiles/{sol,gemini}.env`，密钥只放此目录；占位模板在 `model-profiles/`。已保存的套餐不会因再次 capture 被覆盖。`gemini.reasoning.json` 声明经接口验证的 `low/medium/high/tiered` 别名；`minimal` 当前没有可切换的已验型号。成功选择强度会保存，之后菜单切回 Gemini 使用上次选择。辅助整理/判断保持 low；tiered 是代理动态别名，主强度字段为空，复杂任务可能显著变慢。套餐中的独立搜索地址、凭据、`LLM_WEB_SEARCH_MODEL` 和 `WEB_SEARCH_REASONING_EFFORT` 不随主强度改变。Gemini 套餐使用新接口的原生 Google Search，搜索型号和强度固定 medium；Sol 套餐使用原接口搜索。搜索返回带来源摘要，网页打开/查找直接读取实际页面；不保证绕过网站或网络限制。 Gemini 图片套餐的 medium 画质在已验通道返回2K；high可能返回4K并超过当前8MiB媒体预算，不能只根据请求size认定返回尺寸。

切换无需重建镜像：脚本保留实际运行应用的镜像、QQ 平台、GPU 和其他开关，原子更新源目录及共享环境，显式 `--no-deps` 仅重建群私两个应用，再验证健康、接口和 QQ/计算容器身份。失败自动恢复配置及应用；备份位于 `/opt/xiaomachi/shared/runtime/model-profile-backups/`。每次切换最多两次有限的合成文本探测，`-Check` 仅一次探测、不切换，不调用生图或发送 QQ。不要与 rebuild/start 并发；原生会话按 Gemini 型号/接口轮换，保留历史数据。此入口仅切换已接入套餐，应用代码升级仍使用受控 rebuild。

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

生产数据不用 Windows bind mount。曾经的 `infra/wsl/runtime/` 迁移目录不是现役数据库；只有确认卷迁移与备份后才清理旧文件。正式资料任务的检查点/证据在应用卷 `research_tasks/`，按群/私聊与提问者隔离，7天有效；不是事实库，原始聊天和记忆撤回规则仍适用。

创建一致 SQLite 备份（容器中执行）：

```bash
docker exec xiaomachi-bot python -m scripts.backup_memory_v2 \
  --database /workspace/data/bot.db \
  --backup-dir /workspace/data/backups --tag manual
```

验证输出 `integrity_check=ok`，保存生成的 manifest 与备份；真实数据不要上传 GitHub。其他账号卷另行备份。记忆索引的 prepare/activate/rollback 见 [MEMORY](../../docs/MEMORY.md)，代码升级不等于索引激活。

## 8. GPU 与模型缓存

网页独立出口：WSL回环18797，7897留给Windows；当前只用ikuuu日本x1节点。默认同步入口从Clash的profiles.yaml定位唯一名含ikuuu的订阅，不修改本机配置。私有YAML放 `/opt/xiaomachi/shared/mihomo/`（目录0700、文件0600），用 `render_mihomo_config.py --source <ikuuu私有文件> --output <候选文件> --page-only` 创建 `XIAOMACHI-WEB-JP-X1` url-test组，每300秒测延迟、容差0、关闭TUN/LAN；不继承其他国家节点、选择组、provider或监听端口。无日本节点或存在外部dialer依赖时拒绝生成。确认同一运营商未标倍率默认x1后才加 `--unmarked-rate-one`（同步脚本对应 `-UnmarkedRateOne`），明确其他倍率仍排除。先 `mihomo -t -d /opt/xiaomachi/shared/mihomo -f <候选文件>` 校验，再私密备份旧配置、原子替换并重启服务。只设置 `WEB_PAGE_PROXY_URL=http://127.0.0.1:18797`，主模型/搜索/find/QQ维持原出口，不设置全局代理；失败不静默换出口。重启后复验日本实际出口及页面代理链。环境与订阅不进入Git，模型套餐切换保留该非套餐字段。

基础镜像包含 CUDA/cuDNN 和 fastembed-gpu，但 GPU 设备是可选。无 NVIDIA 设置 `ENABLE_GPU=0`、`MEMORY_EMBEDDING_DEVICE=auto`，用 CPU；启用需 Windows NVIDIA 驱动、WSL GPU 支持和 NVIDIA Container Toolkit/CDI。

```bash
nvidia-smi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

确认透传后启用 `ENABLE_GPU=1` 并 rebuild；GPU overlay 只给群应用，不给 QQ 客户端。`auto` 遇 CUDA 推理错误可回退 CPU。首次模型下载需联网，缓存放持久 `/workspace/data/models`；缓存完才使用 `MEMORY_EMBEDDING_LOCAL_FILES_ONLY=true`。

GPU overlay 文件为 `docker-compose.gpu.yml`，挂载 CDI 设备 `nvidia.com/gpu=all`。向量通道故障可暂设 `MEMORY_EMBEDDING_PROVIDER=disabled` 保留 FTS，这属于降级，不代表完整向量检索正常。

## 9. 可选代理

Windows HY2故障守候：在仓库运行 `python -B -X utf8 scripts/watch_hy2.py start`，接受Windows UAC后通过按需计划任务 `Xiaomachi-HY2Watch` 隐藏运行，避免交互会话结束影响进程；无开机/登录触发器。用同脚本 `status` 查看，`stop` 请求保存并停止。启动先做独立抓包自检，再正式守候；仅新鲜心跳的 `state=armed` 且 `packet_capture=true` 表示已开始，超过90秒显示 `stale`，此时不能推断底层PktMon已停止，需核对会话路径再保全ETL。最长72小时，不跨重启恢复；当前不是HY2订阅时等待，不自动切换。PktMon按原HY2服务器IP过滤、普通包截断128字节（含头部）、64MiB循环文件；有既存抓包/过滤器则拒绝接管。仅新HY2组错误在30秒内累计2条触发，保存Windows/WSL/API快照和有限直连/当前代理/独立HY2核心对照，再保留60秒后自动停止。私有结果在 `data/hy2-watch/<启动时间>/`：ETL、pcapng、trigger及summary JSON；最近一小时状态/警告有界保存。`failed/cleanup_required`需按错误处理；无真实故障时自检不代表自然故障已捕获。不开全局代理、不改MTU/路由或重启QQ，所有证据禁止公开提交。

上游可直连就留空 `XIAOMACHI_HTTP_PROXY`、`XIAOMACHI_HTTPS_PROXY`。QQ/OneBot 与应用模型代理分离；安装器通过 `runtime_egress.py` 判断是否启用 Mihomo，默认不要求代理服务。

需要代理时，使用提供的 `install_mihomo.sh`、`sync_mihomo_from_clash_verge.ps1` 和 `render_mihomo_config.py` 准备本机配置，再验证 `xiaomachi-mihomo.service` 与实际上游可达性。代理订阅/节点/Token 都是私有数据；不要复制到公开示例。

X登录由账号本人在可信浏览器中人工完成。公开安装器不提供人工登录桌面，也不自动导出浏览器Cookie；部署者自行管理受保护的登录环境，再按下文配置私有会话。旧部署若保留`xiaomachi-x-login-manual.service`及`/opt/xiaomachi/browser-login/profile`，应将其作为独立私有设施维护，限定本机访问并保留profile，不视为公开安装的必需组件。

小红书采集：Go构建阶段将xiaohongshu-mcp固定提交的只读动作编译为`/usr/local/bin/xhs-readonly`，Chrome随小町镜像预置；不在宿主安装Go/Chrome/爬虫，不开独立采集服务。专用Docker卷`xiaomachi-xhs-session`仅挂载到群私应用的`/workspace/data/xhs-crawler`保存会话（目录0700、session.json 0600），常规rebuild保留。元数据和CDN直连，不继承X或模型代理。

运维人员可通过`printf '%s' '{"action":"login"}' | docker exec -i xiaomachi-bot xhs-readonly`取扫码入口；二维码写入上述目录的`login-qr.png`，本人用App扫码并确认，最多等240秒。仅复制临时QR图用于本机展示，不读取或复制session.json。`{"action":"status"}`同入口只返回登录布尔状态；模型工具不提供login/status。登录态成功后`search_xhs_posts`、原帖核验及图集/视频下载自动使用，无需新增.env字段。无会话的指定原帖保留公开解析；搜索明确报告需人工登录。挑战/权限/限流停止，不能换出口撞站。上限与未验边界见[小红书合同](../../docs/DEVELOPMENT.md#小红书容器采集合同)。

X采集在小町容器内运行，镜像固定安装gallery-dl==1.32.16和twscrape==0.20.1，前者主源、后者一次有界备选；不再部署自动浏览器工具或18810网关，安装器停用旧xiaomachi-x-browser服务，人工入口和profile保留。

结构化主题搜索由模型提供时间、媒体类型及latest/top；双类型分路共用现有预算，缺失类型如实返回。仅用户明确数字才加互动门槛；关键词不足可按主体补召回，再按原帖正文核验原主题。详情见[工具清单](../../docs/TOOLS.md)，认证/限流后本轮工具停止重查，不能用备库绕过。

WSL私密文件 `/opt/xiaomachi/shared/private-secrets/x-session.json` 只含用户授权的X `auth_token`、`ct0`，目录0700、文件0600。通过受保护的本地输入更新该JSON，禁止将值放到命令历史/报告/仓库；过期、挑战或限流时停止并让用户人工刷新会话。群私应用通过 `./private-secrets:/run/xiaomachi-secrets:ro` 只读挂载，QQ平台及监控不挂载。安装器将release目录链接到shared，构建上下文排除私密目录。

私有 `.env` 设置 `X_CRAWLER_SESSION_FILE=/run/xiaomachi-secrets/x-session.json`、`X_CRAWLER_PROXY_URL=http://127.0.0.1:18797` 后常规rebuild。host网络让容器直接使用WSL的独立日本代理；元数据及X媒体下载使用该出口，不改变聊天接口的代理配置。`list_x_posts/search_x_posts/inspect_web_resources/download_twitter_media`自动接入；无任意命令、浏览器或文件读取模型工具。清空会话路径再rebuild可关闭登录态采集，单帖/账号保留公开来源，主题搜索明确不可用。超时、身份与公开性校验及扩站见[开发合同](../../docs/DEVELOPMENT.md#x爬虫采集合同)。

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

## Agent本机监控

在私有 `infra/wsl/.env` 设置 `AGENT_MONITOR_ENABLED=true` 并 rebuild，两应用采集从新回合开始。安装/启动会加载可选 `docker-compose.monitor.yml`，独立查看容器根文件系统只读，仅可写挂载数据卷的 `agent_monitor` 子目录且不加载机器人 `.env`。启动脚本先确保该目录存在；Docker需支持卷 `subpath`。浏览器打开 [Agent现场](http://127.0.0.1:8765/)，也可双击根目录 `open-agent-monitor.cmd`；中文别名位于 `infra/wsl/windows/打开Agent监控网页.bat`（使用默认浏览器）。

一键入口先绕过系统代理检查Windows本机健康、页面和会话API；若WSL回环8765存在但Windows localhost转发失联，启动隐藏的Windows查看器，通过 `\\wsl.localhost\Ubuntu_Migrated\...` 处理同一Docker数据卷中的 `agent_monitor`，不复制聊天记录或加载机器人 `.env`。入口核对卷子挂载；兼容旧版整卷只读挂载。WSL仍监听回环8765；Windows优先复用8765，若该端口被系统保留或占用，则由系统分配可用的回环端口。实际端口保存在 `data/agent-monitor-launcher/viewer.json`，入口先验证健康、页面与会话API再复用，并自动打开实际地址；请双击入口，不要固定收藏8765。仅监听本机，不使用7897、不修改系统保留端口或防火墙，也不重启QQ/WSL。入口需本机Python；可用 `XIAOMACHI_MONITOR_PYTHON` 指定已安装的python.exe，默认检测PATH与本机Conda目录。日志在 `data/agent-monitor-launcher/viewer.log`；关闭浏览器不停止查看器。升级删除功能时需重启已运行的旧版Windows查看进程；重启Windows后再次双击入口即可恢复。

左侧选择会话，右侧连续查看触发消息、公开思考摘要/状态、模型请求、工具行动和发送回执。「为什么开始行动」显示@/回复、消息批次、到期目标或后台任务的实际原因；可展开查看来源、队列/任务号和到期时间。回合之间明确标记执行上下文延续或失败后重建；独立审查/委派任务单列。点击每步「查看本次输入」打开实际请求抽屉，按顺序展开历史并对比同一执行会话的上一请求；多次重试标注同一步的不同尝试。支持群/私筛选、加载更早回合、暂停更新和跟随最新过程。旧记录缺少逻辑关联时显示未知，不补采或推定；上游未提供公开摘要时仅显示进度，不显示隐藏思维链。

发送合法的`重置上下文`（含隔离/恢复和显式条数变体）成功后，该群或私聊仍在原主体入口中，网页自动清空之前流程和关联旧任务，底部请求/token统计从本轮开始；尚无请求时累计为0，命中率显示暂无请求。重置时间显示在流程上方，其他会话不受影响。旧观测仅退出当前展示并按原留存策略保留诊断；页面删除会话会同时删除其隐藏旧观测。此功能需更新群私应用及监控容器；Windows旧查看进程也需重启，已打开页面刷新一次加载新版脚本即可。

底部固定显示最近完整输入/配置窗口、累计消耗、新增输入、输出、缓存命中、创建、加权命中率及请求数。新增输入扣除缓存；累计消耗按同请求的含缓存输入＋输出，缓存不重复相加，创建未单独报告显示N/A。单位是token而非费用；配置窗口不等于日志KiB。窗口占用仍含缓存，实报优先，无实报才标估算；累计仅限当前保留记录，缺失不按0计算，覆盖不足标部分。上下文重建后不沿用旧执行段占用；旧记录未采集窗口时使用已核验的当前配置参考并标明；参考24小时过期，无有效参考才显示未知，新增请求以实际捕获值优先。切换会话同步更新，暂停更新也暂停统计；查看网页不调用模型。

媒体队列和委派研究任务在原请求提交工具的位置显示折叠卡，点击展开执行、回传、产物及交付回执；实时刷新保留展开状态。研究子模型的输入和用量可单独查看，用量不混入主会话。后台工具不创建模型子对话，绘图服务另计。回传完成不等于QQ送达；复用回执明确标为未再次发送。「查看父请求」可加载更早回合；父回合尚未加载或已过期时，任务置于顶部折叠区。缺失关联仅凭同作用域、来源、提问者的唯一任务回执补链；没有可靠证据仍保留独立记录。

列表项右侧的「删除」或选中会话右上角「删除记录」会显示确认框。确认后删除当前已采集会话及关联后台过程的输入/流/统计；聊天、记忆和正在执行的任务保留，后续新回合可重新出现。删除没有网页撤销入口；暂停更新时仍可删除并同步刷新。只整理显示列表不会重建机器人上下文。

手动仅启动查看器（从当前 release 的 `infra/wsl`，按实际平台替换基础Compose文件）：

```bash
docker compose -f docker-compose.snowluma.yml -f docker-compose.monitor.yml up -d --no-deps xiaomachi-monitor
curl --noproxy '*' http://127.0.0.1:8765/health
```

源码独立预览可用 `python -B -m app.monitor.server --root data/agent_monitor --port 8765`。必须通过本机回环访问；服务拒绝非本机Host和跨站Origin，不应增加LAN绑定/公网转发。页面显示私有聊天和脱敏请求，勿将记录目录复制到公开报告。内容/存储上限与故障隔离合同见 [架构观测说明](../../docs/ARCHITECTURE.md#本机-agent-观测)。保留清理在下一次采集时执行，页面关闭不会停止机器人或采集。

新记录按本地上下文分组，轮换另开段；群参与判断每次独立，纯工具归入父请求。页面区分本地上下文、上游响应续接链和缓存路由键；当前自定义上游在`AGENT_RESPONSES_CONTINUATION=auto`下使用本地完整重放，不声称拥有上游conversation。空巡检不追加历史，有实际研究/创作进展则保留；重置及恢复原因见各回合。

停用时设置 `AGENT_MONITOR_ENABLED=false` 并 rebuild，采集关闭且查看容器移除；已有私有记录仍在数据卷中。普通stop也移除查看容器，正常start根据开关恢复；查看器不可用不阻断群私聊天。回滚到旧release时只重建旧版应用并移除不支持的查看器，不切换QQ桥接。

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
在 PowerShell 中可用 `& ./scripts/clean_test_tmp.ps1 -Name 'qq-61','qq-65' -ExcludeName 'qq-64-git'` 仅预览指定名称，核对后加 `-Force`；名称只接受直接子项，不接受路径或通配符。`-ReportPath` 应位于仓库内、测试目录外；清理前保存必要证据并排除仍有消费方的目录。

WSL 内删除文件后，Windows VHD 可能不立即缩小。压缩需要先核对发行版/VHD路径、停止服务、trim、退出WSL，再使用 Windows 支持的虚拟磁盘压缩操作并恢复启动/状态验证。不要在容器写数据库时对挂载中的 VHD 操作。

群聊上下文升级会幂等新增`proactive_group_states`，保存每群观察进度、租约和滚动额度。上线前通过SQLite在线backup保留一致快照并执行`PRAGMA integrity_check`；不要删除生产数据库。群消息窗口为30条，主动观察默认8条新消息/5分钟、静默退避至30分钟，每小时最多6次主动模型请求和3次插话。状态跨重启保留，明确互动单独处理。`AGENT_CONTEXT_HISTORY_TOKENS=24000`、`AGENT_BACKGROUND_HISTORY_TOKENS=8000`控制动态历史；升级格式9会安全换原生会话，原始消息、任务及实际平台回执保留。监控同时显示实报未缓存输入和临时观察估算，两者不能互作计费依据。回滚代码可忽略新增表，不恢复旧数据库覆盖上线后的新消息。

辅助 Windows 入口集中在 `infra/wsl/windows/`：四种桥接管理/登录页面、隐藏启动 `start-xiaomachi-wsl.vbs` 和中文监控别名。根目录只保留启动、停止、状态、重建、监控、模型切换六个常用入口。隐藏启动仍调用根目录启动器；正式开机计划任务继续使用 `infra/wsl/scripts/run_windows_runtime_task.vbs`，路径不变。
