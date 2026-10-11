# 开发与测试

用途：操作指南；读者：开发者；维护者：项目维护者。

## 环境

工程要求 Python 3.12 或更新版本，生产镜像基于 Ubuntu 24.04/Python 3.12。Agent SDK 与原生运行时均固定为 `0.1.5rc1`，不要只升级其中一个。开发机需要 Git；Windows 的生产入口还需要 WSL2/systemd、Docker/Compose。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q app scripts
python -m pip check
```

Linux 对应使用 `python3.12 -m venv .venv` 和 `source .venv/bin/activate`。依赖安装需要联网；离线测试使用 fake provider，不应访问生产账号或调用付费模型。个别原生 Harness 测试依赖已安装的运行时。

## 测试目录

测试源码放 `tests/`，按模块镜像分类。默认 pytest 夹具和缓存统一 `data/test-work/`，每次运行生成独立临时目录，防止并行测试相互删除。手动指定仓库内 `--basetemp` 时也必须位于这里；仓库外独立临时目录可用于隔离回放。

```powershell
python -m pytest -q tests/agent tests/private_chat
python -m pytest -q --junitxml=data/test-work/full.xml
powershell -ExecutionPolicy Bypass -File scripts/clean_test_tmp.ps1
```

清理脚本默认只预览；确认没有仍在运行的测试后加 `-Force` 清理，先将必要报告移到私有 `data/` 工程目录。不要清理生产数据卷，不要在根目录创建 `.test-tmp-*`、复制依赖或留下私有测试聊天。

## 修改边界

- `app/adapters/` 解析平台数据并返回真实回执，不负责推理和长期记忆策略。
- `app/core/` 的业务规则尽量独立于 `.env` 和网络，使来源、时间、预算与检索可隔离验证。
- `app/storage/` 负责表/仓储/迁移，避免在路由器中创建 schema 或拼重复 SQL。
- `app/runtime/` / `*_main.py` 是依赖装配与进程入口；群记忆初始化不迁入私聊进程。
- `app/agent/` 向 Harness 暴露允许的工具，来源、权限和副作用仍由 Python 业务层掌握。

共享配置、工具 schema、数据库字段或跨层数据结构变更前，检索所有消费方并一起校验。保留群隔离、真实作者、canonical 来源关联、有限重试和 unknown receipt 状态。

## 验收层级

1. 先用合成账号/群号、临时数据库和 fake gateway 验证本次规则。
2. 运行相关测试，再运行完整 pytest、compileall 与 pip check。
3. 改部署时验证 Bash/Compose、构建快照和实际健康检查。
4. 真实模型回放与实际 QQ 动作分别记录；没有 QQ 回执不能声称平台提交成功。

模型评估可能收费；账号登录、空间评论/发布等会产生外部影响。不要把离线测试通过或旧测试数量当作当前真实链路结果。

## 公开文档与发布

公开入口为 [README](../README.md)、[配置](CONFIGURATION.md)、[架构](ARCHITECTURE.md)、[记忆](MEMORY.md)、[运维](../infra/wsl/README.md) 和 [安全](SECURITY.md)。读者不需要任何本机 Backlog、内部验收报告或历史快照才能安装使用。

```powershell
python -B -X utf8 scripts/check_public_docs.py
python -B -X utf8 scripts/check_github_sensitive.py --include-untracked
git diff --cached --stat
python -B -X utf8 scripts/check_github_sensitive.py
```

敏感检查读取本机 `data/project-tools/.secrets-blocklist`（一行一个禁止公开值），扫描文件和真正暂存 blob，不回显敏感值。独立发布目录可通过 `--root`、`--blocklist` 指向候选仓库与本机标识表。未知账号或新密钥需先加入本地表；模式扫描不能代替人工核对。

完整发布包含运行源码、配置模板、脚本和测试。`data/`、真实人物档案、`.env`、登录态、日志、备份和本地管理材料不属于源码分发。详情见 [安全发布](SECURITY.md)。
