# 小町 WSL 隔离部署

目标：NapCat、QQ 登录态、小町 Python 进程都运行在 WSL2/Docker 内。Windows 只保留启动、停止、状态查看入口。

Windows 入口使用英文文件名，避免中文路径或文件名经过 `cmd.exe` / WSL 参数传递时发生乱码：

- `start-xiaomachi-wsl.bat`
- `stop-xiaomachi-wsl.bat`
- `status-xiaomachi-wsl.bat`

这三个 BAT 会调用固定 ASCII 入口 `/mnt/d/xiaomachi-wsl-entry.sh`。该入口脚本会在 WSL 内部自动查找仓库目录，再执行 `infra/wsl/scripts/start.sh`、`stop.sh` 或 `status.sh`。

不要删除：

- `D:\xiaomachi-wsl-entry.sh`
- `D:\qq群ai小人\infra\wsl\runtime\napcat\ntqq`

前者是 Windows BAT 的固定 WSL 入口；后者保存 NapCat/QQ 登录态。删除登录态目录后可能需要重新扫码。

验收标准：

- `status-xiaomachi-wsl.bat` 显示 NapCat 容器 `healthy`。
- OneBot `get_status` 返回 `online=true`。
- `get_login_info` 返回小町账号。
- 目标群能收到并回复。
- Windows 进程列表里没有由小町启动的 Windows `QQ.exe`。

原有 `启动小町.bat` / `关闭小町.bat` 保留为 Windows 版回滚入口，不要在 WSL 版验收前覆盖。
