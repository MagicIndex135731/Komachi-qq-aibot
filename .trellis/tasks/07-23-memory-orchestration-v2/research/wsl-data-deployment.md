# V2 WSL 数据、依赖与仅重建 Bot 的部署研究

> 研究范围：静态仓库检查与只读宿主环境探测（2026-07-23）。未读取
> `infra/wsl/.env` 的内容，未读取数据库/模型缓存内容，未启动、停止、重建或迁移任何服务。

## 已确认的事实与证据

| 项目 | 结论 | 证据 |
| --- | --- | --- |
| 业务数据库 | 容器内默认路径是 `/workspace/data/bot.db`。`AppSettings.data_dir` 默认为 `data`，`sqlite_path` 为 `data/bot.db`。 | `app/config.py` |
| 业务持久卷 | LLBot Compose 将外部命名卷 `xiaomachi-bot-data` 挂载到 Bot 的 `/workspace/data`（可写），也以只读方式挂载给 LLBot。 | `infra/wsl/docker-compose.llbot.yml` |
| LLBot 登录态 | 独立的外部命名卷 `xiaomachi-llbot-data` 挂载到 `/app/llbot/data`。不得随 V2 发布重建、删除或覆盖。 | `infra/wsl/docker-compose.llbot.yml`、`tests/test_llbot_deployment.py` |
| 模型缓存目标 | R3 要求 V2 模型缓存为 `/workspace/data/models`；因此其真实宿主位置应位于 `xiaomachi-bot-data` 卷中的 `models/`。当前静态代码尚未实现该路径。 | `prd.md` R3；Compose 卷映射 |
| 原始数据与日志 | 同一业务卷还承载 `logs/`、`history/`、图片缓存等。`.gitignore` 与 `.dockerignore` 都排除了 `data`、数据库、`infra/wsl/runtime` 和 `.env`，构建上下文不会携带真实运行数据。 | `.gitignore`、`.dockerignore`、`app/config.py` |
| SQLite 运行方式 | 引擎设置 `journal_mode=WAL`；迁移为原地、幂等式 `create_all` + 兼容补列，明确禁止重建/删除 `bot.db`。 | `app/storage/db.py`；`.trellis/spec/backend/database-guidelines.md` |
| 当前派生索引 | 当前 FTS/向量索引仅针对 V1 `memory_items`；向量通过 `hashed_text_embedding` 写入 256 维 sqlite-vec。R3 明确禁止 V2 将其作为语义向量。 | `app/storage/db.py`；`prd.md` R3 |
| 当前镜像依赖 | `infra/wsl/requirements.xiaomachi.txt` 与 `pyproject.toml` 一致，包含 `sqlite-vec==0.1.9`，但不包含 FastEmbed、ONNX Runtime 或任何语义 embedding 包。Dockerfile 在构建期安装该 requirements 文件。 | `pyproject.toml`、`infra/wsl/requirements.xiaomachi.txt`、`infra/wsl/Dockerfile.xiaomachi`、部署测试 |
| 运行栈 | 生产安装目录是 `/opt/xiaomachi/current`；`current/infra/wsl/.env` 是共享 `.env` 的符号链接。systemd 的 stack 服务工作目录为 `/opt/xiaomachi/current/infra/wsl`。 | `install_linux_runtime.sh`、`xiaomachi-wsl-entry.sh`、`systemd/xiaomachi-stack.service` |
| 只重建业务 Bot 的基础能力 | LLBot Compose 的 `xiaomachi` 服务名为 `xiaomachi-bot` 容器，LLBot 服务名为 `llbot`/容器名 `xiaomachi-llbot`。现有 `start.sh` 已采用 `up -d --no-deps xiaomachi`，可避免把 Bot 发布扩展为 QQ 平台重启。 | `docker-compose.llbot.yml`、`scripts/start.sh`、`tests/test_llbot_deployment.py` |
| 验收工具 | `status.sh` 依次检查 Compose 状态、LLBot WebUI、OneBot `get_status`、登录信息、Bot 心跳；`onebot_probe.py` 只在显式给定群 ID 时读取最近群消息。 | `scripts/status.sh`、`scripts/onebot_probe.py`、WSL 部署测试 |

当前工作区可见 `data/bot.db` 和 `data/logs/`，但这只说明 Windows 工作区仍有历史本地数据；**不能据此推断生产卷的内容或新鲜度**。

## 当前环境探测结论

- 宿主机能找到 `wsl.exe`，但 `wsl --status` 表示没有已安装的 Linux 发行版。
- 当前 PowerShell PATH 中不存在 `docker` CLI。因此未执行 `docker ps`、`inspect`、卷检查或 Compose 配置展开，实际容器、卷和登录态均为“未验证”，不是“未运行”。
- 因没有可用目标 WSL/Docker，以下命令是交给目标 Linux 运行环境操作员执行的流程，不应在本宿主机重试后假定结果相同。

在目标 WSL 上的**只读预检**（不打印环境变量、不导出容器环境、不读取聊天/数据库内容）：

```bash
set -euo pipefail
cd /opt/xiaomachi/current/infra/wsl
docker compose -f docker-compose.llbot.yml config --quiet
docker compose -f docker-compose.llbot.yml ps
docker inspect --format '{{.Name}} state={{.State.Status}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}n/a{{end}}' \
  xiaomachi-bot xiaomachi-llbot
docker inspect --format '{{range .Mounts}}{{.Name}} -> {{.Destination}} rw={{.RW}}{{println}}{{end}}' \
  xiaomachi-bot xiaomachi-llbot
docker volume inspect --format '{{.Name}} driver={{.Driver}}' \
  xiaomachi-bot-data xiaomachi-llbot-data
```

避免使用 `docker inspect` 的默认 JSON（可能输出容器环境变量）；避免 `docker compose config` 的非 quiet 输出（可能展开环境变量）。

## V2 依赖与数据契约

1. 在 `pyproject.toml` 和 `infra/wsl/requirements.xiaomachi.txt` 同步加入经 Python 3.12 验证的 FastEmbed/ONNX 依赖（版本由实施阶段锁定），并保留 `sqlite-vec==0.1.9`。部署测试已要求这两处依赖集合一致。
2. Dockerfile 已满足“构建期安装、运行期不安装”的位置；不要在 entrypoint、worker 或每条消息路径执行 pip。模型文件首次缺失时才下载到 `/workspace/data/models`，因其位于 `xiaomachi-bot-data`，容器重建不会丢失。
3. `/workspace/data/models`、`bot.db`、备份文件和日志均为持久运行数据：不提交 Git、不复制进镜像、不以 `docker compose down -v` 或卷删除处理。
4. 若 FastEmbed 初始化、模型下载或维度/版本检查失败，V2 必须禁用向量通道并退回 FTS/V1，不阻塞 `app.group_main`、持久化或 QQ 回复。日志只能含模型/版本、维度、错误类别、耗时和证据 ID，不能含聊天、prompt、密钥或 provider 原始异常体。

## 线上 SQLite 备份方案（WAL 安全）

不要把正在写入的 `bot.db`、`-wal`、`-shm` 当普通文件 `cp`。现有首次卷迁移脚本会先停所有 writer 再复制并校验，适合一次性迁移；它不是本次“在线 V2 迁移前”的首选。

在实际变更前，授权操作员以 **SQLite backup API** 从运行中的 `xiaomachi-bot` 创建一致性快照。下例只在业务数据卷写入一个备份文件，脚本不回显数据库内容：

```bash
set -euo pipefail
backup_tag="pre-v2-$(date -u +%Y%m%dT%H%M%SZ)"
export BACKUP_TAG="$backup_tag"
docker exec -i -e BACKUP_TAG xiaomachi-bot python - <<'PY'
import os
import sqlite3
from pathlib import Path

data_dir = Path('/workspace/data')
source_path = data_dir / 'bot.db'
backup_dir = data_dir / 'backups'
backup_dir.mkdir(mode=0o700, exist_ok=True)
target_path = backup_dir / f"bot-{os.environ['BACKUP_TAG']}.db"
tmp_path = target_path.with_suffix('.tmp')
if not source_path.is_file():
    raise SystemExit('source database is missing')
if target_path.exists() or tmp_path.exists():
    raise SystemExit('backup target already exists')
source = sqlite3.connect(source_path)
target = sqlite3.connect(tmp_path)
try:
    with target:
        source.backup(target, pages=256, sleep=0.05)
    result = target.execute('PRAGMA integrity_check').fetchone()
    if result != ('ok',):
        raise SystemExit(f'backup integrity_check failed: {result!r}')
finally:
    target.close()
    source.close()
tmp_path.replace(target_path)
print(f'backup_ok={target_path.name}')
PY
```

记录命令输出中的文件名、UTC 时间和 `backup_ok` 到受控发布记录；不要把备份带入 Git。发布后应再次对该备份使用同样的 `PRAGMA integrity_check`，并对在线库执行只读完整性检查。若实施脚本需保存备份到卷外位置，必须使用权限为 `0700` 的受控目录，并另行明确保留期和访问者。

## LLBot 模式的精确发布、回滚与验收

以下步骤假设目标已完成上述只读预检与在线备份，且确认当前 QQ 平台确为 LLBot。每条命令均显式指向 `docker-compose.llbot.yml`；不要运行 `start.sh`，因为它包含平台切换、迁移和登录辅助逻辑，超出本次“仅重建 Bot”范围。

### 发布（只构建和重建 `xiaomachi-bot`）

```bash
set -euo pipefail
cd /opt/xiaomachi/current/infra/wsl
release_tag="pre-v2-$(date -u +%Y%m%dT%H%M%SZ)"
docker image inspect xiaomachi-bot:local >/dev/null
docker image tag xiaomachi-bot:local "xiaomachi-bot:${release_tag}"
docker compose -f docker-compose.llbot.yml build xiaomachi
docker compose -f docker-compose.llbot.yml up -d --no-deps --force-recreate xiaomachi
```

- `--no-deps` 是限制边界：不会停止、重建或拉起 `llbot` 服务。
- 不使用 `docker compose down`、`down -v`、`restart llbot`、`up --force-recreate llbot` 或任何卷删除命令。
- `docker image tag` 只保留本地旧镜像标签；它不触碰数据卷或登录态。发布记录必须保存实际 `release_tag`。

### 回滚（只恢复 Bot 镜像并重建 Bot）

仅在发布后 Bot 健康检查、OneBot smoke 或消息链路验收失败时执行。将占位符替换为发布时记录的真实标签：

```bash
set -euo pipefail
cd /opt/xiaomachi/current/infra/wsl
rollback_tag='pre-v2-YYYYMMDDTHHMMSSZ'
docker image inspect "xiaomachi-bot:${rollback_tag}" >/dev/null
docker image tag "xiaomachi-bot:${rollback_tag}" xiaomachi-bot:local
docker compose -f docker-compose.llbot.yml up -d --no-deps --force-recreate xiaomachi
```

这只能回滚应用镜像，不能回滚已执行的 schema/数据回填。因此数据库迁移必须向后兼容且幂等；若数据层需要恢复，先停止 V2 写入路径、从上节的 backup API 备份制定独立恢复计划，并取得明确的恢复授权，不能把恢复混入镜像回滚。

### 验收顺序

1. 确认 LLBot 没有被重建：记录发布前后 `xiaomachi-llbot` 容器 ID、`StartedAt` 和登录/健康状态。发布后 ID 与 `StartedAt` 应保持不变。
2. 运行仓库已有的运行态验收脚本（它会检查 LLBot WebUI、OneBot `get_status`/登录状态、Bot heartbeat）：

   ```bash
   cd /opt/xiaomachi/current/infra/wsl
   bash scripts/status.sh
   ```

3. 对 Bot 执行最小只读数据库验收，不输出消息文本：`PRAGMA integrity_check = ok`、`messages` 行数与备份前发布记录一致或仅增加正常新消息、V2 回填统计无孤立消息/不可接受的 failed/pending jobs，且逐群统计不跨群。
4. 在已授权测试群发送一条正常群消息，确认入库、Bot 回复和 OneBot `get_status` 均成功；随后按 PRD 手工验收“详细讲讲”“后来呢”和跨天历史提问。任何日志审计只检查 episode/证据 ID、分数、token、耗时和错误类别，不读取完整聊天/prompt。
5. 关闭 shadow mode 前，比较评测 JSONL 的 V1/V2 evidence recall、token、延迟与错误统计；不达阈值即保留 V1/影子模式或按上节回滚 Bot。

## 风险、权限与未验证项

| 风险/未知项 | 影响 | 需要的控制或权限 |
| --- | --- | --- |
| 当前机器没有 WSL 发行版和 Docker CLI | 无法证明实际卷、容器、健康或 QQ 登录状态 | 在真实 WSL/Docker 主机上由具备只读 Docker 权限的操作员执行预检；把结果附到发布记录 |
| FastEmbed/ONNX 依赖尚未加入 | V2 不能提供真实语义向量 | 实施者在隔离构建环境验证 Python 3.12、镜像大小、模型下载、维度和 FTS 降级；需网络/镜像构建权限 |
| 首次模型下载 | 可能慢、失败或占用持久卷 | 允许 Bot 容器访问模型源；设置可观测的超时、磁盘余量和失败降级，不在消息主路径同步下载 |
| 发布期间 schema/回填 | 迁移不能靠镜像回滚逆转 | 先执行 backup API；需要业务数据卷写入、Docker build/up 权限及明确的数据恢复授权 |
| Docker inspect/config 输出 | 可能泄露 `.env` 展开值或容器环境 | 仅使用文中的 format/`config --quiet`；禁止默认 inspect、完整 config、`docker exec env` 和日志全文导出 |
| LLBot 登录态 | 错误的 Compose 操作可能导致需重新登录 | 发布命令必须包含 `--no-deps` 且只指定 `xiaomachi`；禁止 llbot restart/recreate、卷删除与平台切换 |
| 现有迁移脚本 | 该脚本会停止 writer，且早期 Windows 路径会备份 `.env` 到受限目录 | 本次在线 V2 备份不调用它；若未来做首次 Linux 卷迁移，需由有 root/Docker 权限的运维人员按其专门变更流程执行 |

## 实施前的最小验收门

- 依赖清单、Dockerfile、Compose 和部署测试已同步；镜像构建通过。
- 数据库 migration/backfill 有离线并发/幂等测试，真实 embedding 不可用时的 FTS/V1 降级测试通过。
- 目标 WSL 预检、在线 backup API、备份与在线库 `integrity_check` 均通过。
- 发布命令仅涉及 `xiaomachi`，LLBot 容器 ID/`StartedAt` 不变，`status.sh` 和受权群消息链路均通过。
