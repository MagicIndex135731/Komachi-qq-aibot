# Production Validation

- Online SQLite backup and immutable ledger manifest verified before migration.
- Frozen backfill completed with no pending/running/failed mandatory jobs,
  orphan rows, or failed embeddings inside the pinned run.
- Approved real dataset: 64 cases. Exact recall@10 = 1.0, vague = 1.0,
  paraphrase = 0.3; vague+paraphrase improvement over V1 = 65 percentage
  points.
- Warm local benchmark: 20 warmups, 320 measured runs, p95 = 197.508 ms.
- Final local suite: 773 passed; compileall and git diff check passed.
- Production image uses CUDA 12.8 runtime and detects RTX 5060 with
  `MEMORY_EMBEDDING_DEVICE=auto`; CPU fallback remains enabled.
- Only `xiaomachi-bot` was recreated. LLBot ID and StartedAt remained unchanged.
- WebUI probe passed, OneBot reported online/good, bot heartbeat was alive.
- Post-deploy SQLite: integrity ok, foreign keys 0, superseded memberships 0,
  duplicate ordinals 0, groups with multiple current open episodes 0.
