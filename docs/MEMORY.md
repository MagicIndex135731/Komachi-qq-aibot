# 长期记忆、人格与索引维护

用途：解释与操作参考；读者：维护者；维护者：项目维护者。本文的命令针对容器内 `/workspace/data`，全部产物是私有数据。

## 记忆从哪里来

允许接收的实际群消息进入 canonical `messages` 账本。后台按事件、时间和话题构建 episode、摘要、结构化事实、人物资料、FTS 与向量；主动发言使用同一记忆链路，也受实际来源和成功投递状态约束。模型内部草稿、合并转发署名、失效来源与未知发送状态不能伪装成可靠长期事实。

群 `memory_enabled` 控制长期生成和召回。Harness 的普通回合接入既有 `MemoryOrchestrator`，按需工具使用完整 V3 通道，而不是另外建立一套独立事实库。

| 层 | 用途 |
|---|---|
| 原始消息/来源 | 实际作者、时间、引用与可回查正文 |
| episode/摘要 | 将连续讨论组织为可检索事件 |
| `memory_items` | 有来源和有效期的结构化事实 |
| 人物资料/关系 | 同群人物背景、称呼和近期状态 |
| 检索投影与向量 | 全文、语义、人物、时间组合定位证据 |

搜索不是直接相信摘要：系统组合全文/向量/时间/人物候选，排除撤回与不合格来源，展开原始上下文，再按预算提供证据。当前 LLM 重排序开关没有完整实现，不应启用并宣称已经得到额外重排能力。

## 上下文与工具

Agent默认观察最多30条近期群消息，`recent_context_limit`可为单群缩小窗口；`memory_enabled=false`关闭该群长期生成和召回，近期观察仍可使用。查询可在获准的V3范围扩大历史候选。候选上限与实际输入不同，最终按窗口、字符/token保护配额、证据强度和来源截取，不能把候选数量理解为每次固定放入。

自动召回只提供历史原文、事实和摘要；近期群聊由独立观察窗口负责。Agent适配层用保守token估算重新打包完整证据单元，每次正文默认8000 token，再登记最终证据的真实来源和语义修订。自动证据放本轮模型输入尾部，不写入持久原生日志，也不再被约3000 token的观察预算二次删块。显式`memory_search`/`memory_read`的有界结果进入本群工具历史。检索成功、候选命中与最终给模型的内容是三个不同指标。

本轮读取仍有200条来源的保护上限；`agent_session_dependencies`单独保存当前保留历史的来源及事实/摘要修订，累计超过200不因此清空主会话。原生压缩后的摘要保守继承依赖；每次模型和外部写入前分批复验。来源撤回或事实更正后，无法精确拆分的历史整体失效，保留原始消息和真实副作用账本。该账本另有16384项资源保护，不能用删除来源ID保留正文的方式绕过。

后台任务、产物和工作笔记用不可变依赖快照延续创建时的历史来源。快照按账号与作用域隔离，7天有效；主会话重建不会擦掉快照，恢复/交付仍检查来源及修订。过期快照分批回收，每账号最多1024份；达到资源上限会拒绝新增，而不是丢失来源。旧检查点在首次保存时幂等迁移，旧任务继续沿用其原有来源合同。

主对话统一按完整输入预算维护：默认96000软阈值，压缩后目标64000，近期完整交互最多24000（后台8000），摘要最多8192（后台4096）；固定规则、工具和本轮资料均计入。硬保护取160000与模型实际余量的较小值。旧历史字段不再单独触发换段；故障、格式不兼容或失效来源仍可安全重建。监控分别显示观察、记忆和压缩参数，实际token、缓存与压缩请求费用仍以渠道usage/账单为准。

分层工具默认每层最多 5 条，综合结果也有独立上限；必要时追加检索和出处。记忆写入还要求 `services.memory_write`、真实授权来源和提交前复验。Agent 的临时工作笔记/目标用来继续行为，有时效，不是新的长期事实来源。

主聊天显式委派的资料子任务使用独立原生会话，建立任务检查点不会自动切换主会话；任务证据只在 `data/research_tasks/` 按提问者保存。摘要携带V3依赖的语义版本与实际消息来源，恢复会重新验证并纳入当前执行守卫。任务7天过期或来源撤回、事实更正/失效后不能直接沿用旧结论。网页原文不作为新增V3事实/人格样本导入；来源明确的用户可见回复仍属于实际聊天账本。全请求预算覆盖规则/schema/回放/图片；V3召回先扣历史估算、输出、安全及工具余量，相同快照只保留一次。缓存usage、请求估算和内部压缩成本分开记录，缺失缓存字段保持未知。

直接群聊互动的追加记忆检索受 `MEMORY_MEMORY_TOOL_MAX_ROUNDS` 控制，Harness 总读取上限为该值的两倍，默认 4 次（含 `search_history`、`memory_search`、`memory_read`）。达到上限、进入最后一步或剩余时间不足时，模型用已有证据交付，并明确未知与推测；不继续检索到整轮超时。工具 schema 与结果会显示实际 `MEMORY_MEMORY_TOOL_MAX_RESULTS` 上限。

常用开关：`MEMORY_ORCHESTRATION_V2_ENABLED`（完整编排兼容开关）、`MEMORY_RAW_V3_ENABLED`（活动raw索引）、`MEMORY_LAYERED_MEMORY_ENABLED`、`MEMORY_MEMORY_TOOLS_ENABLED`、`MEMORY_EMBEDDING_PROVIDER`。即使名字含V2，编排开关也不是V3回滚按钮。

embedding 默认本地 BGE 与 sqlite-vec；可配置兼容服务。关闭向量仅保留FTS属于降级。改变模型/维度/版本后不能继续把旧向量当作同一语义空间，须重新准备检索代际。

## 近期公开社交记录

`public_social_records`保存同一机器人账号的公开动态观察、评论草稿/结果，以及工具明确提交的简短想法、目标和下一步。保留14天、每账号最多512条，优先保留评论；不读取隐藏推理，也不导入群私原文或主人私聊任务便笺。它是行为记录，不属于群V3事实或人格语料。

公开巡检和主人委派的公开任务通过`read_social_activity`按需查询，默认5条、最多10条，可按上海日期、类型、好友或记录ID定位；记录不会预装到每次模型请求。评论结果按相同参数摘要关联真实效果账本。草稿改变后不能继承旧参数的成功状态；未知、发送中或未核实状态不会自动重放。补发仍需当前授权和原帖复验。

旧版本迁移使用`python -m scripts.backfill_public_social_activity`，默认只预览；备份后才加`--apply`。只从固定公开原生目录有界读取工具事件，不导入任意目录或模型思考。完整步骤见[WSL指南](../infra/wsl/README.md)。

## 人格怎样生效

加载顺序为默认 `configs/persona.yaml`，默认基础上的 `persona_variants.yaml`，独立 `configs/personas/*.yaml`，本地 `data/personas/*.yaml`，以及对应 `.live.yaml` 动态覆盖。最终选择当前群人格，渲染成模型规则/提示词；角色文件不是模型权重。

默认小町与通用变体可公开。成员人格、来源人物原话、事实与表达样本仅本地保存。模仿中的表达样本不是事实证据，跨群选择口吻不允许搬运来源群人物关系和原话。导出的 `persona_descriptions.json` 只是查看快照，修改它不会改变运行人格。

人格事实刷新、表达样本索引和主动任务执行分别有开关、来源群/用户、阈值与任务状态。请求刷新只表示请求/排队，更新完成必须查看实际任务回执与文件/索引状态。修改静态YAML后用rebuild；动态服务遵守自己的更新门限，不在每轮聊天无条件蒸馏全部历史。

启动默认人格是“嘴臭小町”，主人可以明确请求切换已配置人格，例如 `切换人格为:粘人小町`；群状态按当前管理器合同选择，重启后应核对实际活跃角色，不假设说明快照就是当前选择。

## 新数据库与历史回填

首次部署保持 `MEMORY_RAW_V3_ENABLED=false`，先确认群消息落库、基础记忆/人格服务、embedding和备份可用。raw V3 是带质量门的检索代际发布，不靠把环境变量改成true完成初始化。

一致备份：

```bash
python -m scripts.backup_memory_v2 \
  --database /workspace/data/bot.db \
  --backup-dir /workspace/data/backups --tag bot-memory-v3
```

使用输出的实际 manifest 路径，确认 `integrity_check=ok`。下文示例假设 manifest 为 `/workspace/data/backups/bot-memory-v3.manifest.json`；如工具生成其他名称，所有命令一致替换。

历史结构化回填独立运行，不用每个新聊天回合重扫：

```bash
python -m scripts.backfill_structured_memory plan --database /workspace/data/bot.db
python -m scripts.backfill_structured_memory run \
  --database /workspace/data/bot.db --run-key history-structured --finalize
```

运行前阅读脚本 `--help`、确认范围和模型预算；回填与评测可能使用付费上游。真实数据集、review bundle与人格蒸馏产物不能公开。

## Raw V3 准备、评测、激活与回滚

以下保留发布质量合同。prepare创建非活动代际；人工review和真实回答/可见性评估通过后才activate。质量报告、数据、manifest和向量代际必须匹配，不允许手填通过标记代替证据。



V3 rollout is deliberately split into separate fail-closed phases. Preparing a
generation never changes the active vector generation:

Production may additionally enable the adaptive context profile:

```dotenv
MEMORY_ADAPTIVE_CONTEXT_ENABLED=true
MEMORY_ADAPTIVE_CONTEXT_BUDGET_CHARS=48000
MEMORY_RECENT_PROTECTED_MIN_TOKENS=1200
MEMORY_HISTORY_PROTECTED_MIN_TOKENS=2400
MEMORY_RECENT_PROTECTED_MIN_MESSAGES=1
MEMORY_HISTORY_PROTECTED_MIN_MESSAGES=1
MEMORY_ADAPTIVE_MAX_RECENT_MESSAGES=60
MEMORY_ADAPTIVE_MAX_HISTORY_MESSAGES=300
```

This profile dynamically shares the effective input-token budget between recent
and historical context. `60/300` are emergency row caps in this example, not fixed quotas and
not targets to fill. Strong direct, lexical, or multi-channel evidence uses a
compact history expansion (up to 150 candidates); weak evidence or a failed
channel may expand up to 300. Disable only
`MEMORY_ADAPTIVE_CONTEXT_ENABLED` and recreate both application containers to restore the legacy
60/150 packer without changing the active V3 generation or restarting the QQ bridge.

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
`MEMORY_ADAPTIVE_CONTEXT_ENABLED=true` when releasing the adaptive profile), then recreate
only the two application containers; never recreate the current QQ bridge. Production retrieval resolves the active
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
recreate only the two application containers:

```bash
python -m scripts.backfill_memory_v3_raw \
  --phase rollback \
  --database /workspace/data/bot.db \
  --manifest /workspace/data/backups/bot-memory-v3.manifest.json \
  --prepared-report /workspace/data/backups/memory-v3-prepared.json \
  --output /workspace/data/backups/memory-v3-rollback.json
```


## 日常状态与审计

```bash
bash /opt/xiaomachi/current/infra/wsl/scripts/status.sh
systemctl status xiaomachi-memory-audit.timer --no-pager
journalctl -u xiaomachi-memory-audit.service -n 20 --no-pager
```

检查数据库完整性、原始消息/向量覆盖、后台队列、动态人格文件与样本索引。状态正常只能证明当时组件可用；是否正确回答历史问题需核对来源与实际模型回答。

一致性修复只处理可机械证明的状态，不能据此判定人物事实真假。代码回滚不恢复数据库；索引回滚不删除聊天原文或QQ登录态。涉及生产修复先创建验证过的一致备份，保留失败证据，避免重复未知写入。
