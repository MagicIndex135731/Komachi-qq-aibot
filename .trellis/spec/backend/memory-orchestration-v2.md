# Memory Orchestration V2

## Scenario: Versioned group-memory orchestration and live migration

### 1. Scope / Trigger

Use this contract when changing group-memory episode allocation, derived
documents, semantic indexes, query orchestration, shadow execution, historical
backfill, or WSL rollout. Private/dev flows must not start this runtime.

### 2. Signatures

Runtime composition:

```python
build_memory_runtime(
    *,
    settings: AppSettings,
    engine,
    llm_client,
    bot_display_name: str,
) -> MemoryRuntimeComposition
```

Persistent job entry points:

```python
MemoryBackgroundService.enqueue_message(
    *,
    group_id: int,
    message_id: int,
    backfill_run_id: int | None = None,
    watermark_message_id: int | None = None,
) -> BackgroundJob

MemoryBackgroundService.enqueue_shadow(
    request: ShadowJobRequest,  # IDs and generations only
) -> BackgroundJob
```

Migration commands:

```text
backup_memory_v2.py --database PATH --backup-dir PATH --tag ID
backfill_memory_v2.py --database PATH --manifest PATH --run-key ID
build_memory_eval_dataset.py --database PATH --output PATH --review-output PATH
run_memory_recall_eval.py --database PATH --dataset PATH
  --review PATH --backfill-run-key ID
  --results-output PATH --report-output PATH --benchmark-output PATH
  --warmup 20 --benchmark-runs 250 --enforce-real-dataset
```

Additive tables include `conversation_episodes`, `episode_messages`,
`retrieval_documents`, `retrieval_document_messages`,
`retrieval_index_state`, and `memory_backfill_runs`. Jobs carry
`backfill_run_id`, `target_generation`, owner/lease, requested generation, and
claimed generation.

### 3. Contracts

- Every group query, document, episode, provenance row, fact, and summary is
  scoped by `group_id`; scope is checked in SQL and again before packing.
- Raw `messages` are immutable source-of-truth rows. Derived rows retain real
  platform source message IDs.
- Reserved outbound placeholders are not episode-eligible. QQ-blocked rows may
  remain assigned to an episode but cannot enter windows, prompts, embeddings,
  documents, facts, summaries, events, provenance, or shadow payloads.
- Shadow payload is exactly group/message ID plus configuration/index
  generation. The worker reloads the request and executes V2; persisted results
  contain IDs, finite candidate scores, route counts, tokens, latency,
  rewrite/fallback flags, and a bounded error category only. Shadow enqueue
  persistence runs outside the reply path and is drained during shutdown.
- Vector identity is provider + model + dimensions + version + integer
  generation. Query and background share one lazy provider. A new physical
  sqlite-vec table is activated only after complete eligible-document coverage
  and CAS. Provider initialization and local inference are concurrency-safe.
- A backfill run is pinned to the verified backup manifest's per-group
  `max(messages.id)` watermarks and to segmentation/compaction/index
  generations. Rows above the watermark are reported separately.
- Backfill completion rechecks the raw ledger within every manifest watermark.
  A real evaluation requires a dataset-hash-bound, per-case approved review
  sidecar; machine gates fail closed on recall, benchmark mode, run count, and
  local p95 thresholds.
- Destructive late-arrival resegmentation is idempotent per
  `(group_id, message_id, segmentation_generation)` under concurrent delivery.
- Episode membership append is a database CAS operation. The `INSERT ...
  SELECT` must require the target episode to be both `status='open'` and
  `is_current=1`, and must derive `ordinal` from that row's `message_count` in
  the same statement. A CAS miss raises `EpisodeAppendConflict`; the allocator
  discards its cached batch, reloads all unassigned messages, sorts them by
  `(timestamp, id)`, and retries at most three times.
- Base and late generations are compatible only when their base prefix is the
  same: `segment-v2`, `segment-v2:late:1`, and `segment-v2:late:2` may share the
  current suffix. A different base generation must remain isolated.
- Real recall evaluation uses typed top-10 raw `episode` units. Summaries,
  facts, and memory-item source IDs must not be flattened into an episode hit.
  A vector benchmark is successful only when the vector channel returns at
  least one candidate; SQL/vector errors propagate into the failed channel.
- Environment keys are the `MEMORY_ORCHESTRATION_*`,
  `MEMORY_EMBEDDING_*`, `MEMORY_EPISODE_*`, `MEMORY_CHUNK_*`,
  `MEMORY_QUERY_*`, and `MEMORY_*_BUDGET/CANDIDATE/LIMIT` fields declared in
  `AppSettings`. Defaults remain V1-compatible; production rollout begins with
  V2 enabled in shadow mode.
- Long-running WSL production uses `MEMORY_EMBEDDING_DEVICE=auto` with the
  CUDA-enabled image. CUDA is preferred when `CUDAExecutionProvider` is
  available and inference falls back to CPU on initialization/runtime failure.
  Compose reads the effective deployment environment from `infra/wsl/.env`;
  editing a similarly named shared file does not change this deployment.

### 4. Validation & Error Matrix

| Condition | Required result |
|---|---|
| Any source or candidate belongs to another group | Raise scope/provenance error; active request falls back to V1 |
| Blocked/reserved content reaches a derivation input | Reject job or filter before provider call; never persist a derived row |
| Embedding provider/model/extension unavailable | Disable vector channel only; FTS and normal replies continue |
| Production backfill/evaluation has no ready local vector generation | Exit nonzero and remain shadow/V1; FTS degradation is not rollout eligibility |
| Building vector generation has missing/failed documents | Do not activate; mark migration failure and retain old active generation |
| Claimed generation/owner no longer matches | Reject completion by CAS; newer work remains queued |
| Cached open episode is superseded before append | CAS rejects the write; requery the full unassigned batch and retry in chronological order |
| Vector SQL/runtime channel raises or returns zero candidates | Mark the vector channel failed/unsuccessful; do not count an attempted call as vector success |
| Worker lease expires | Requeue idempotently with finite attempts |
| Backup ledger differs within a snapshot watermark | Abort before backfill, or fail the run if the final recheck differs |
| Backfill has wrong-generation or nonterminal jobs, orphan rows, blocked provenance, or incomplete eligible embeddings | Do not report success |
| Real dataset lacks bound per-case approval, or an AC threshold fails | Exit nonzero with safe error codes; do not write success artifacts |
| V2/query rewrite/all retrieval channels fail | V2 → V1 → recent → safe empty fallback |
| Shadow worker fails | Record a bounded error category; reply path still returns V1 |

### 5. Good / Base / Bad Cases

- Good: online backup verifies `integrity_check=ok`; backfill drains all
  generation-pinned jobs; coverage and ledger match; 64-case real evaluation
  and 20+250 warm benchmark pass; only `xiaomachi` is recreated.
- Base: sqlite-vec or the model is unavailable. The bot stays in shadow/V1 and
  uses scoped FTS; vector activation is not claimed.
- Bad: copy a live WAL database, enqueue raw query/recent messages in shadow,
  reuse an incompatible vector table, treat reserved placeholders as missing
  episode coverage, or rebuild `xiaomachi-llbot`.

### 6. Tests Required

- Repeated and concurrent `create_all`; raw message count/hash unchanged.
- Composite FK and high-score cross-group retrieval leakage tests.
- Coalescing rearm, two-worker claim, generation CAS, stale lease recovery,
  finite retry, graceful stop, and late-arrival resegmentation tests. Include a
  real SQLite barrier test that pauses after reading an open episode, supersedes
  it in another transaction, then proves the resumed worker leaves zero
  memberships on the superseded episode and preserves chronological ordinals.
- Mixed safe/blocked and reserved-only episodes proving zero blocked
  provenance and retained raw rows.
- sqlite-vec missing-extension fallback, zero lexical-overlap semantic hit,
  coverage failure, ready activation, and active-generation CAS tests.
- Resolver timeout/malformed JSON, V2/V1/recent fallback, token budgets,
  pinned evidence, reply ancestors, and source deduplication tests.
- Manifest ledger, interrupted backfill resume, reserved eligibility,
  real-dataset schema/quotas, metrics, and benchmark-count tests.
- Full pytest/compile checks plus production integrity, image build, health,
  OneBot status, LLBot ID/StartedAt invariance, and real group smoke.

### 7. Wrong vs Correct

#### Wrong

```python
# Cross-group/global top-k and content-bearing shadow payload.
hits = vector.search(query, limit=30)
jobs.add({"group_id": group_id, "query": query, "recent": recent})
```

#### Correct

```python
# Scope inside the retrieval query, validate provenance, and persist IDs only.
hits = vector.search(group_id=group_id, embedding=embedding, limit=30)
background.enqueue_shadow(
    ShadowJobRequest(
        group_id=group_id,
        message_id=canonical_message_id,
        config_generation=config_generation,
        index_generation=index_generation,
    )
)
```

sqlite-vec KNN compatibility note: use the supported `MATCH` query with
`group_id` partition and `k` inside the virtual-table query. Do not add an
outer `ORDER BY distance` unless the deployed sqlite-vec version is explicitly
tested for it.

Episode append anti-pattern:

```python
# Wrong: current-ness was checked by an earlier read and may now be stale.
repository.add_message(episode_id=cached_episode.id, message_id=message.id)

# Correct: the write itself proves that the episode is still current/open.
if not repository.add_message_if_current(
    episode_id=cached_episode.id,
    group_id=message.group_id,
    message_id=message.id,
    estimated_tokens=tokens,
):
    raise EpisodeAppendConflict
```
