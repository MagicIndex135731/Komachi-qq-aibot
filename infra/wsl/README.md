# WSL/Docker 运行目录

这里是小町当前唯一受支持的运行栈。`QQ_PLATFORM=llbot` 使用 LLBot，`QQ_PLATFORM=napcat` 回退到原 NapCat；两者共享同一套小町业务、数据库和模型配置，但登录态各自独立保存。

## 启动链路

```text
start-xiaomachi-wsl.bat
  -> D:\xiaomachi-wsl-entry.sh
  -> infra/wsl/scripts/start.sh
  -> docker compose up -d
  -> 条件打开当前 QQ 平台 WebUI
  -> OneBot 与小町心跳检查
```

停止和状态入口使用同一个固定脚本，分别调用 `stop.sh` 和 `status.sh`。

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

`start.sh` 在 WebUI 就绪后检查 QQ 登录状态。明确已登录时不打开浏览器；未登录、需要验证或无法确认状态时打开登录页面。浏览器启动失败不会阻断容器。

## 运行态保护

不要删除：

- `.env`
- `runtime/napcat/ntqq`
- `runtime/napcat/config`
- `runtime/logs`
- `runtime/onebot-watchdog.json`

`runtime/pip-cache` 可以重建，但保留它能显著缩短容器重建时间。

## 验收

```bash
docker compose config --quiet
docker compose ps
bash scripts/status.sh
```

正常在线时应看到 NapCat healthy、OneBot `online=true`、主动群列表探针成功，以及新鲜的小町心跳。若 QQ 本身已离线，先完成 WebUI 登录，再重复状态检查。
