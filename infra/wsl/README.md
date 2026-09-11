# WSL/Docker 运行目录

这里是小町当前唯一受支持的运行栈。默认 `QQ_PLATFORM=llbot` 使用 LLBot；`QQ_PLATFORM=napcat` 是本地回退选项。两者共享同一套小町业务、数据库和模型配置，但登录态各自独立保存。

## 启动链路

```text
start-xiaomachi-wsl.bat
  -> D:\xiaomachi-wsl-entry.sh
  -> infra/wsl/scripts/start.sh
  -> 启动当前 QQ 平台容器
  -> 条件打开当前 QQ 平台 WebUI
  -> 无依赖启动小町（OneBot 未就绪时自动重连）
  -> OneBot 与小町心跳检查
```

停止和状态入口使用同一个固定脚本，分别调用 `stop.sh` 和 `status.sh`。

Windows 可能在最后一个交互式 `wsl.exe` 退出后回收 WSL VM。生产机应安装登录级
任务计划，让独立进程持有运行锚点并在登录后恢复 systemd 服务：

```powershell
powershell -ExecutionPolicy Bypass -File `
  infra/wsl/scripts/install_windows_runtime_task.ps1
```

任务名为 `Xiaomachi WSL Runtime`，不会修改 Windows 系统代理。需要卸载时传入
`-Action Remove`。BAT 仍是人工启动、停止和状态检查入口；start 会优先启动已安装
的任务实例，stop 会先停止任务再关闭 systemd 服务。任务启动器会在发布期间 anchor
短暂退出后立即重建它，避免 WSL 在重建间隙关机；但明确执行 stop 后不会把小町
重新拉起。

## 初始化

在 WSL 中执行：

```bash
cd "/mnt/d/qq群ai小人"
bash infra/wsl/scripts/bootstrap_wsl.sh
```

脚本会创建：

- `infra/wsl/.env`：从 `.env.example` 生成，需要手工填入本地密钥。
- `infra/wsl/runtime/napcat/config/onebot11.json`：本机 OneBot WebSocket 配置。
- `.venv-wsl`：watchdog 和探针环境。

## 操作命令

推荐从 Windows 使用仓库根目录的 BAT。WSL 内也可以直接运行：

```bash
cd "/mnt/d/qq群ai小人"
bash infra/wsl/scripts/start.sh
bash infra/wsl/scripts/status.sh
bash infra/wsl/scripts/stop.sh
```

`start.sh` 先启动 QQ 平台并尝试打开 WebUI，再启动小町，避免 Compose 的健康依赖阻塞登录页面。LLBot WebUI 为 `http://127.0.0.1:3080/`，OneBot 为 `ws://127.0.0.1:3002`；NapCat 回退平台仍使用 `6099` 与 `3001`。浏览器启动失败不会阻断容器。

文本模型使用 Responses 端点时，可在 `.env` 设置 `LLM_BUILTIN_WEB_SEARCH=true` 启用主模型内置联网检索。明确“联网/搜索/查资料”的群请求会强制检索；工具事件保存到 `runtime/logs/responses-tool-events.jsonl`，不进入 Git。

### WSL 内置 Mihomo 上游代理

生产使用 WSL 内独立的 Mihomo 规则实例，不再依赖 Windows Clash、WSL NAT
网关或反向 TCP 中继。配置由 Windows Clash Verge 当前合并配置生成，但私有
节点、订阅和规则数据库只写入 `/opt/xiaomachi/shared/mihomo/`，不会进入 Git。

在 Windows 仓库根目录同步当前 Clash Verge 配置并安装服务：

```powershell
powershell -ExecutionPolicy Bypass -File `
  infra/wsl/scripts/sync_mihomo_from_clash_verge.ps1
```

生成器会创建 `XIAOMACHI-NOVA-HK` 健康选择组，只包含当前配置里的香港节点；
`ai.novacode.top` 优先走该组，QQ、本地与私网规则优先 `DIRECT`，其余规则继承
当前机场配置。Mihomo 仅监听 WSL 回环地址，不启用 TUN，也不接管 Windows 系统
代理。运行配置使用：

`deepseek.com` 也在托管 `DIRECT` 规则内：小町的对话与联网检索都打官方
DeepSeek 接口，该域名在国内可直连，绕香港只会增加延迟并改变出口 IP，因此不
交给机场的兜底规则；只有生图供应商仍走 `XIAOMACHI-NOVA-HK`。这条规则在每次
重新渲染时会被先剥离再重新插入，不会重复堆叠。

```dotenv
XIAOMACHI_HTTP_PROXY=http://127.0.0.1:7897
XIAOMACHI_HTTPS_PROXY=http://127.0.0.1:7897
DOCKER_HTTP_PROXY=
DOCKER_HTTPS_PROXY=
```

LLBot/NapCat 不读取 `XIAOMACHI_*`，QQ 与 OneBot 始终直连。修改 `.env` 后只
重建 `xiaomachi` 容器，并在 `ENABLE_GPU=1` 时携带 GPU Compose 覆盖文件；不要
重建 LLBot。订阅更新后重新运行同步脚本即可，脚本会先渲染和校验新配置，再
重启 Mihomo。

Windows 快捷方式（`*.bat`）必须保持 CRLF 行尾：cmd.exe 无法解析 LF-only 脚本里的
`goto` 标签，错误分支会变成"闪退"（看不到任何提示）。`.gitattributes` 已用
`*.bat text eol=crlf` 固定这一点；改动这些脚本后如果 Windows 端仍是 LF，重新
检出一次即可。

LLBot 返回 `retcode=1200 / waitForSelfEcho timeout`、等待回执超时或发送过程中断线时，
系统会将本次投递标记为“结果未确认”。由于 QQ 可能已经收到消息，机器人不会自动重试、
切片补发或发送额外失败提示，以避免同一回复重复出现。该记录保留在近期上下文中以维持
对话连续性，但不参与自动摘要和长期记忆压缩；同一入站消息重放时也不会再次生成回复。

### QQ 1001 掉线与签名组件

主用 QQ 桥接现在是 **SnowLuma**（`QQ_PLATFORM=snowluma`）：它是同类的
"挂官方 QQ 客户端 → 转 OneBot v11"运行时，但 2026-09 的掉线潮里它的 issue 区
没有出现 LLBot/NapCat 那样的高频被踢反馈，因此作为 A/B 的候选先上线观察。

| 平台 | `QQ_PLATFORM` | 容器 | WebUI | OneBot v11 |
|---|---|---|---|---|
| SnowLuma | `snowluma` | `xiaomachi-snowluma` | 5099（noVNC 6081，VNC 5900） | 3001/3000 |
| LLBot | `llbot` | `xiaomachi-llbot` | 3080 | `LLBOT_WS_PORT`（默认 3002） |
| NapCat | `napcat` | `xiaomachi-napcat` | 6099 | 3001 |

切换流程（登录态各自独立，互不影响）：

```bash
# infra/wsl/.env 里改 QQ_PLATFORM=<napcat|llbot|snowluma>，然后
systemctl restart xiaomachi-stack.service   # stop.sh 会停掉另外两个 stack
# 首次登录：open-snowluma-webui.bat（密码自动复制到剪贴板，初始账号 admin）
```

SnowLuma 镜像来自上游维护的 `SnowLuma.Docker.Framework`（Docker Hub
`motricseven7/snowluma`，compose 里按 digest 固定 v1.14.15）。首次登录需要
在 WebUI 里改密码并接入 QQ；`SNOWLUMA_ACCEPT_EULA=1` 与
`SNOWLUMA_ACCEPT_PRIVACY=1` 表示运营者已阅读并接受发行包内的协议。
重启 stack 会让 SnowLuma 重新生成一次性初始密码，因此首次登录期间先不要让它
反复重启（watchdog 在这段时间保持停止）。

首次登录后 SnowLuma 会为该账号自动生成 OneBot v11 默认配置（HTTP 3000 /
WS 3001，绑定 127.0.0.1），并填入随机 `accessToken`。本栈的 bot 走
`NAPCAT_WS_URL=ws://127.0.0.1:3001` 且**不带令牌**（与 NapCat 路径一致，
`app/adapters/napcat_ws.py` 只在传入令牌时才发 `Authorization` 头），因此要把
两个 `accessToken` 清空：WebUI→连接配置里删掉，或

```bash
# WebUI 登录后会返回 Bearer token，用它调用配置接口（保存即热重载，无需重启）
curl -sS -X POST -H "Authorization: Bearer <webui-token>" \
  -H 'Content-Type: application/json' \
  --data-binary @onebot_config.json \
  http://127.0.0.1:5099/api/config/<uin>
```

清空后 `ws-default` 会显示 `1 个客户端`（即 xiaomachi bot 已连接）。

#### SnowLuma 的冷启动限制（实测 2026-09-11）

PC 开机（或 `wsl --shutdown`）后的自恢复链路里，WSL、Docker、两个容器、
SnowLuma 本体（含 hook 自动挂载）都会自动起来，**但 QQ 不会自动登录**：

- QQ 客户端默认停在"手机QQ扫码登录"页，需要人工扫码；
- 即使登录时勾选了"自动登录"，也没有效果：容器停止时 QQ 客户端是
  **崩溃退出**的（`/app/.config/QQ/crash_files/tomb_*.txt` 里 `signal: 5
  (SIGTRAP)`），登录态来不及落盘，下次启动回到扫码页；
- 上游 SnowLuma 没有免扫码的环境变量（只有 EULA/PRIVACY/DEV_MODE/
  UPDATE_CHECK/WEBUI_BOOTSTRAP_PASSWORD/TRUST_PROXY），`qq --help` 也没有
  自动登录开关；
- LLBot 自带会话文件（`qq-session-*.json`），容器/守护进程重启后能自动重登；
  NapCat 类似。若"开机零人工"是硬要求，这是选择平台时的关键差异。

因此开机后需要：双击 `open-snowluma-desktop.bat`（noVNC 直达 QQ 桌面并自动
填入 VNC 密码）→ 点"刷新" → 手机扫码。watchdog 在离线时会弹 Windows 通知提醒。

#### 开机流程（当前约定：每次开机人工扫一次码）

1. Windows 登录 → 计划任务 `Xiaomachi WSL Runtime` 自动拉起 WSL 与整套 stack
   （mihomo、SnowLuma、bot 容器都在其中）；
2. SnowLuma 会自己起来、自动挂 hook，但 QQ 客户端停在扫码页；
3. watchdog 探测到"WebUI 正常但 OneBot 未监听"时**不会重启容器**（重启只会换掉
   二维码），而是弹一次 Windows 通知：`SnowLuma is running but the QQ account is
   not logged in. Double-click open-snowluma-desktop.bat ...`；
4. 双击 `open-snowluma-desktop.bat` → 点"刷新" → 手机扫码；OneBot 起来后 bot
   自动连上（`ws-default` 显示 1 个客户端），无需其它操作；
5. 上线后**不会补答离线期间积压的消息**：`GROUP_MESSAGE_MAX_AGE_SECONDS=300`
   之外的旧消息只归档、不回复（日志 `group_message_stale_archived`）。

#### 回复长度与拆分的边界（2026-09-11 调整）

`configs/persona.yaml` 只描述**风格和人格**，不再出现任何"几条消息 / 多少字 /
用不用 Markdown"的格式约束；拆条与长度策略全部在系统层：

- 默认尽量短：`app/core/chat_style.py` 的 `Reply length` 两条规则要求"一条短消息
  为默认形态，只有问题确实需要长回答（教程 / 分析 / 详细说明）才展开"；
- 拆条由 `.env` 的 `GROUP_REPLY_SPLIT_*` 控制：`GROUP_REPLY_SPLIT_MAX_CHARS=64`
  之内的回答**原样单条发出**；超过时才在句号/逗号处切成最多
  `GROUP_REPLY_SPLIT_MAX_MESSAGES=3` 条（只拆不丢字，不切句中）；
- 回复里的换行会被合并成同一句，不再因为换行变成多条消息。

回滚：`QQ_PLATFORM=llbot` + `systemctl restart xiaomachi-stack.service`。

LLBot 8.1.10 是当前上游最新 release（GitHub release 与 Docker Hub `latest` 一致），其内置的
`@lucky-lillia/sign-proxy-loader`（20260813 构建）**没有导出 `setMachineGuid`**。QQ
1001 掉线后 LLBot 会重新生成 `machine_guid.bin`，但签名层无法切换设备指纹，日志会打印
`sign-proxy 未导出 setMachineGuid (老版 .node), GUID 切换不会生效`。该 sign-proxy 由上游
私有仓库构建、未发布到 npm，仓库内无法自行升级；等上游新版 release 后换回官方 digest。

由于 8.1.10 早于会话过期修复（PR #856），生产当前运行的是**上游 main 的源码构建**，
用仓库自带脚本重建（脚本会解析 commit、走代理下载、构建并校验产物）：

```bash
# 在 WSL 内（需要走代理，否则 GitHub 直连只有几十 KB/s）
LLBOT_IMAGE_TAG=xiaomachi-llbot:main-<commit> \
  bash infra/wsl/scripts/build_llbot_from_source.sh main
```

校验项包括：`/app/llbot/webui/index.html`、`llbot.js`（`node --check` 通过且含
`session-expired` 修复）、以及 `sign-proxy.*-musl.node`。任一项缺失即构建失败，
镜像不会带病上线。

**不要直接用上游的 `docker/Dockerfile.local`**：它把 `yarn build-webui` 放在
`yarn build` 之前，而主 bundle 的 vite 构建会先清空 `dist/`，把刚生成的
`dist/webui` 删掉，结果镜像里没有 `/app/llbot/webui/index.html`，WebUI 直接
HTTP 500（`open-llbot-webui.bat` 会因此报错打不开）。上游自己的 publish
workflow 是先 `yarn build` 再 `yarn build-webui`，顺序才对；
`infra/wsl/Dockerfile.llbot-source` 采用正确顺序并带断言。

v8.1.10 → main 的运行时代码差异只有会话鉴权/登录恢复这一组修复（`direct.ts`、
`direct-lib/client.ts`、`direct-lib/login.ts`、`base.ts`、`emailNotification.ts`、
`milky/adapter.ts`），没有其它功能改动。镜像只存在于本地 Docker daemon，不要清理；
上游 8.1.11+ 发布后把 `docker-compose.llbot.yml` 里的镜像换回官方 digest 即可回滚。

1001 掉线在历史日志中多次标注为“异地登录顶号”（同一 QQ 号在别处登录）。恢复顺序是：
先退出其它登录端，再打开 LLBot WebUI（`open-llbot-webui.bat`）扫码。

watchdog 侧的策略：检测到“等扫码”状态（最近 15 分钟内出现 `login-qrcode.png`）时**不再重启
容器**——重启无法完成扫码，只会重复一次失败登录；此时只通知一次。其余 recovery 重启按
120s → 240s → 480s 退避，上限 15 分钟，避免短时间内反复重登触发 QQ 风控。

## 运行态保护

不要删除：

- `.env`
- `runtime/llbot/data`
- `runtime/napcat/ntqq`
- `runtime/napcat/config`
- `runtime/logs`
- `runtime/onebot-watchdog.json`

`runtime/pip-cache` 可以重建，但保留它能显著缩短容器重建时间。

## Memory V3 发布与回滚

Memory V3 是生产启用的历史查询路径（生产 `.env` 中 `MEMORY_RAW_V3_ENABLED=true`，
运行时日志 `route=raw_v3`）；`.env.example` 已按生产模板全部开启，代码默认值保持
安全关闭。V3 运行仍要求 `MEMORY_ORCHESTRATION_V2_ENABLED=true`，但该开关只是 V2
兼容开关，不是 V3 回滚开关。发布前使用 SQLite backup API 创建并验证
`integrity_check=ok` 的备份，再按下方 V3 流程完成准备、评测、激活。

部署只构建和重建 `xiaomachi` service（容器名 `xiaomachi-bot`）：

```bash
docker compose -f docker-compose.llbot.yml build xiaomachi
# 无 NVIDIA 机器（ENABLE_GPU=0，默认）：
docker compose -f docker-compose.llbot.yml up -d --no-deps --force-recreate xiaomachi
# 有 NVIDIA 机器（ENABLE_GPU=1）：
docker compose -f docker-compose.llbot.yml -f docker-compose.gpu.yml up -d --no-deps --force-recreate xiaomachi
```

操作前后记录 `xiaomachi-llbot` 的 container ID 与 `StartedAt`；
**must not restart xiaomachi-llbot**。向量通道异常时回滚为
`MEMORY_EMBEDDING_PROVIDER=disabled` 保留 FTS；普通回滚不恢复数据库，也不得删除
LLBot 登录态。

### CUDA 向量加速

`xiaomachi` 镜像使用 CUDA 12.8、cuDNN 与 `fastembed-gpu`。GPU 设备是可选的：
基础 Compose 不再挂载 GPU，`ENABLE_GPU=1` 时才通过 `docker-compose.gpu.yml`
向 bot 服务挂载 `nvidia.com/gpu=all` CDI 设备；无 NVIDIA 机器无需任何改动即可运行
（嵌入 `MEMORY_EMBEDDING_DEVICE=auto` 自动回退 CPU）。
设置 `MEMORY_EMBEDDING_DEVICE=auto` 后优先使用 NVIDIA GPU，并在 CUDA 推理异常时
回退 CPU；LLBot 不申请 GPU。主机需安装 NVIDIA Container Toolkit，并确保
`nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` 已生成设备规范。
可用 `docker run --rm --device nvidia.com/gpu=all ...` 验证透传。
确认模型已经缓存后设置 `MEMORY_EMBEDDING_LOCAL_FILES_ONLY=true`，可保证离线重启不会
等待模型站点超时（首次部署请保持 `false` 以便联网下载模型）。嵌入模型缓存位于持久卷
`/workspace/data/models`，镜像重建不会丢失。

### 按群记忆策略与日常维护（Memory V3）

记忆系统按群开关，配置在 `configs/groups.yaml`：

- `memory_enabled: false`（默认）：该群**不使用任何记忆**，回复只取最近
  `recent_context_limit`（默认 100）条消息作为上下文；后台不生成、不检索、不入队。
- `memory_enabled: true`：启用完整分层记忆（episode 摘要、结构化事实、用户画像、
  记忆工具、语义排序）。真实群号在 `configs/groups.local.yaml`（gitignored）中配置。

日常维护命令（容器内，先备份）：

```bash
# 事实向量回填（幂等、可断点）
python -m scripts.backfill_memory_item_semantic_vectors --database /workspace/data/bot.db --batch-size 100
# 历史噪音清理（plan -> 备份 -> run；可恢复）
python -m scripts.cleanup_memory_noise plan --database /workspace/data/bot.db
python -m scripts.cleanup_memory_noise run --database /workspace/data/bot.db
# 全系统一致性审计（只读、只输出计数，不调用外部模型）
python -m scripts.maintain_memory_integrity audit --database /workspace/data/bot.db
# 确定性修复：必须显式 --apply，执行前自动创建 SQLite online backup
python -m scripts.maintain_memory_integrity repair --apply \
  --database /workspace/data/bot.db \
  --backup-dir /workspace/data/backups
# 关闭记忆的群：删除全部记忆派生数据（原始消息保留）
python -m scripts.purge_group_memory --database /workspace/data/bot.db --group-id <GROUP_ID> --dry-run
```

正式安装会启用 `xiaomachi-memory-audit.timer`，每天上海时间 04:15–04:20
运行一次只读审计，并在开机错过计划时间时补跑。定时任务绝不执行
`repair --apply`，也不调用外部模型。最新报告与按行历史分别保存在：

- `/opt/xiaomachi/shared/runtime/logs/memory-integrity-audit.latest.json`
- `/opt/xiaomachi/shared/runtime/logs/memory-integrity-audit.jsonl`

关键一致性指标非零时，`xiaomachi-memory-audit.service` 以退出码 2 失败并向
journal 写入 `memory_integrity_audit_alert`；缺少语义向量、缺少投影文档、重复
候选和旧式非结构化记录仍作为观察指标，不触发告警。查看最近结果：

```bash
systemctl status xiaomachi-memory-audit.timer --no-pager
journalctl -u xiaomachi-memory-audit.service -n 20 --no-pager
```

一致性修复仅处理可机械证明的状态：过期 active 记忆、倒置摘要时间、无来源旧
`window` 摘要、legacy compaction 积压、非 active 记忆残留向量/FTS/检索文档，
以及 active 记忆缺失的 FTS 行。它不会判断聊天内容真假，也不会自动改写人物画像。
修复后应再次运行 `audit`，要求上述可修复计数归零，并执行数据库
`PRAGMA integrity_check`。

发布只重建 `xiaomachi`，绝不重启 `xiaomachi-llbot`。

## 验收

```bash
docker compose config --quiet
docker compose ps
bash scripts/status.sh
```

正常在线时应看到当前 QQ 平台 healthy、OneBot `online=true`、主动群列表探针成功，以及新鲜的小町心跳。若 QQ 本身已离线，先完成对应 WebUI 登录，再重复状态检查。

### Memory V3 prepare, evaluate, activate, and rollback

V3 rollout is deliberately split into separate fail-closed phases. Preparing a
generation never changes the active vector generation:

Production may additionally enable the adaptive context profile:

```dotenv
MEMORY_ADAPTIVE_CONTEXT_ENABLED=true
MEMORY_ADAPTIVE_CONTEXT_BUDGET_CHARS=48000
MEMORY_ADAPTIVE_RECENT_PROTECTED_TOKENS=1200
MEMORY_ADAPTIVE_HISTORY_PROTECTED_TOKENS=2400
MEMORY_ADAPTIVE_RECENT_MIN_MESSAGES=1
MEMORY_ADAPTIVE_HISTORY_MIN_MESSAGES=1
MEMORY_ADAPTIVE_MAX_RECENT_MESSAGES=120
MEMORY_ADAPTIVE_MAX_HISTORY_MESSAGES=300
```

This profile dynamically shares the effective input-token budget between recent
and historical context. `120/300` are emergency row caps, not fixed quotas and
not targets to fill. Strong direct, lexical, or multi-channel evidence uses a
compact history expansion (up to 150 candidates); weak evidence or a failed
channel may expand up to 300. Disable only
`MEMORY_ADAPTIVE_CONTEXT_ENABLED` and recreate `xiaomachi` to restore the legacy
60/150 packer without changing the active V3 generation or restarting LLBot.

```bash
python -m scripts.backfill_memory_v3_raw \
  --phase prepare \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --output /workspace/data/backups/memory-v3-prepared.json
```

Build the real snapshot dataset and review every frozen evidence source. The
generated review sidecar starts with `approved=false`; do not activate until a
human reviewer has approved every case. The review bundle contains private chat
content: keep it under `/workspace/data/backups`, never commit or upload it.

```bash
python -m scripts.build_memory_eval_dataset \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --output /workspace/data/backups/memory-v3-cases.jsonl \
  --review-output /workspace/data/backups/memory-v3-review.json

python -m scripts.export_memory_eval_review_bundle \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --dataset /workspace/data/backups/memory-v3-cases.jsonl \
  --output /workspace/data/backups/memory-v3-review-bundle.json
```

After the review sidecar is approved, run the evaluator once without a quality
sidecar to freeze retrieval results and generate a retrieval-bound quality
template. This command is expected to exit with the missing-quality gate:

```bash
python -m scripts.run_memory_recall_eval \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --prepared-report /workspace/data/backups/memory-v3-prepared.json \
  --dataset /workspace/data/backups/memory-v3-cases.jsonl \
  --review /workspace/data/backups/memory-v3-review.json \
  --quality-template-output /workspace/data/backups/memory-v3-quality.json \
  --results-output /workspace/data/backups/memory-v3-results.jsonl \
  --report-output /workspace/data/backups/memory-v3-gate-draft.json \
  --benchmark-output /workspace/data/backups/memory-v3-benchmark-draft.json \
  --warmup 20 --benchmark-runs 250
```

Fill the template only from a controlled GPT answer replay and at least 20 real
index-visibility samples. Then rerun the V3 evaluator against the prepared,
non-active generation. Its passing report is bound to the manifest, dataset,
retrieval fingerprint, exact quality-sidecar digest, and `vector_generation`:

```bash
python -m scripts.run_memory_v3_quality_replay \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --prepared-report /workspace/data/backups/memory-v3-prepared.json \
  --dataset /workspace/data/backups/memory-v3-cases.jsonl \
  --review /workspace/data/backups/memory-v3-review.json \
  --quality-output /workspace/data/backups/memory-v3-quality.json \
  --private-replay-output /workspace/data/backups/memory-v3-quality-private.json \
  --visibility-output /workspace/data/backups/memory-v3-quality-visibility.json \
  --visibility-samples 20
```

After that controlled replay completes, run the final evaluator:

```bash
python -m scripts.run_memory_recall_eval \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --prepared-report /workspace/data/backups/memory-v3-prepared.json \
  --dataset /workspace/data/backups/memory-v3-cases.jsonl \
  --review /workspace/data/backups/memory-v3-review.json \
  --quality-sidecar /workspace/data/backups/memory-v3-quality.json \
  --quality-private-replay /workspace/data/backups/memory-v3-quality-private.json \
  --quality-visibility-artifact /workspace/data/backups/memory-v3-quality-visibility.json \
  --results-output /workspace/data/backups/memory-v3-results.jsonl \
  --report-output /workspace/data/backups/memory-v3-gate.json \
  --benchmark-output /workspace/data/backups/memory-v3-benchmark.json \
  --warmup 20 --benchmark-runs 320
```

Activation requires both the original prepared report and a passing gate
report. It performs final live catch-up and a locked manifest check before the
generation CAS. Immediately after this command succeeds, set
`MEMORY_RAW_V3_ENABLED=true` in `infra/wsl/.env` (and
`MEMORY_ADAPTIVE_CONTEXT_ENABLED=true` when releasing the adaptive profile), then recreate only
`xiaomachi`; never recreate LLBot. Production retrieval resolves the active
generation per query, so it does not keep reading the deactivated legacy table
between the CAS and this bounded restart:

```bash
python -m scripts.backfill_memory_v3_raw \
  --phase activate \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --prepared-report /workspace/data/backups/memory-v3-prepared.json \
  --gate-report /workspace/data/backups/memory-v3-gate.json \
  --dataset /workspace/data/backups/memory-v3-cases.jsonl \
  --quality-sidecar /workspace/data/backups/memory-v3-quality.json \
  --quality-private-replay /workspace/data/backups/memory-v3-quality-private.json \
  --quality-visibility-artifact /workspace/data/backups/memory-v3-quality-visibility.json \
  --results /workspace/data/backups/memory-v3-results.jsonl \
  --benchmark-report /workspace/data/backups/memory-v3-benchmark.json \
  --output /workspace/data/backups/memory-v3-activated.json
```

Emergency rollback preserves all raw messages and vector tables and switches
only the active vector generation back to the legacy generation recorded by
prepare. After rollback succeeds, set `MEMORY_RAW_V3_ENABLED=false` and
recreate only `xiaomachi`:

```bash
python -m scripts.backfill_memory_v3_raw \
  --phase rollback \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --prepared-report /workspace/data/backups/memory-v3-prepared.json \
  --output /workspace/data/backups/memory-v3-rollback.json
```

### Layered memory and memory tools

`MEMORY_LAYERED_MEMORY_ENABLED=true` adds episode summaries, structured
`memory_items`, and user profile facts to the V3 evidence packet while keeping
the vector channel raw-message-only. `MEMORY_MEMORY_TOOLS_ENABLED=true`
exposes `memory_search` / `memory_read` / `memory_write` to the model through
Responses function calling; writes are source-bound to the current group and
conversation. Both switches default to `false` and are enabled explicitly in
the deployed `.env`.

To fill summaries and facts for history that predates episode derivation, run
the bounded, resumable backfill (inside the `xiaomachi` container or against a
backup copy):

```bash
python -m scripts.backfill_structured_memory plan --database /workspace/data/bot.db
python -m scripts.backfill_structured_memory run \
  --database /workspace/data/bot.db --run-key layered-20260806 --finalize
python -m scripts.backfill_structured_memory status \
  --database /workspace/data/bot.db --run-key layered-20260806
```
