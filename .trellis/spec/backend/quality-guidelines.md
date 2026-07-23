# Quality Guidelines

## Baseline

The project targets Python 3.12 and uses pytest with `pytest-asyncio`
(`asyncio_mode = "auto"` in `pyproject.toml`). New behavior is test-first:
add a focused failing test, implement the smallest complete change, then run
the related suite and the full suite before completion.

## Required design patterns

- Keep core parsing/ranking/policy functions deterministic and testable without
  network access.
- Inject repositories/providers into services instead of constructing live
  dependencies inside algorithms.
- Preserve compatibility for existing environment settings and persistent
  SQLite data.
- Make background jobs idempotent, retryable, and independently stoppable.
- Keep source message IDs through every memory derivation and de-duplicate by
  those IDs before context packing.
- Treat group isolation, blocked-output safety, and raw-message retention as
  invariants, not optional checks.
- Use type hints on public functions and dataclasses for structured results.

## Test placement and fakes

- Mirror source areas under `tests/`.
- Use `tmp_path` and `tests/conftest.py:sqlite_engine` for database tests.
- Use `pytest.mark.asyncio` for service-level async behavior; small coroutine
  tests may use `asyncio.run` where that is already the local style.
- Stub/fake LLM, embedding, web, image, and OneBot calls. Unit tests must run
  offline and deterministically.
- Add deployment manifest tests when changing Docker/WSL wiring, following
  `tests/test_llbot_deployment.py` and
  `tests/test_wsl_deployment_artifacts.py`.
- Test fallback paths explicitly: missing extension/model, timeout, malformed
  JSON, worker failure, repeated migration, and concurrent initialization.

## Verification

Run checks proportional to the changed surface:

```powershell
python -m pytest tests/<affected-area> -q
python -m pytest -q
python -m compileall -q app scripts
docker compose -f infra/wsl/docker-compose.llbot.yml config
```

Database/runtime releases also require `PRAGMA integrity_check`, migration
coverage/row counts, image build, service health, OneBot `get_status`, and a
real group-message smoke test. A passing unit suite does not replace runtime
acceptance.

## Review checklist

- Schema changes are idempotent under concurrent startup.
- No query or cache can cross group scope.
- Original messages and provenance remain intact.
- Optional provider/index failures degrade without blocking chat.
- Token budgets and candidate limits are explicit and tested.
- Logs contain metrics/IDs rather than sensitive content.
- Configuration defaults are backward compatible and documented in
  `.env.example`/README.
- No `.env`, database, model cache, credentials, or generated runtime data is
  staged.

## Forbidden patterns

- Destructive database recreation or unverified file-copy backups of a live
  WAL database.
- Live external calls in unit tests.
- Hard keyword-overlap filters that discard semantic candidates.
- Blocking batch embedding/LLM calls in the per-message reply path.
- Broad exception swallowing, infinite retries, or unbounded context assembly.
- Changing a shared config/schema value without searching every consumer and
  test first.
