# Directory Structure

## Runtime shape

This repository is a Python 3.12 single-package backend. Keep code in the
existing `app/` package and tests in the matching `tests/` subtree.

- `app/main.py` wires the legacy combined runtime and shared factories.
- `app/group_main.py`, `app/private_main.py`, and `app/dev_worker_main.py`
  are process-specific composition roots. They construct dependencies; they
  do not own domain algorithms.
- `app/adapters/` translates OneBot/NapCat payloads and performs outbound QQ
  calls. See `onebot_models.py`, `napcat_ws.py`, and `sender.py`.
- `app/core/` owns conversation policy and domain workflows. Representative
  modules are `router.py`, `context_builder.py`, `memory_engine.py`, and
  `memory_compaction_service.py`.
- `app/providers/` wraps external services such as LLM, search, image, and
  embedding providers.
- `app/storage/models.py` owns SQLAlchemy mappings, `db.py` owns engine/schema
  setup, and `repositories.py` owns persisted queries.
- `app/jobs/` contains scheduled job entry points. Long-running service-style
  workers live beside the domain they orchestrate in `app/core/`.
- `app/dev_control/` is isolated private administrative repository control.
- `configs/` contains non-secret YAML policy/persona configuration.
- `infra/wsl/` contains Docker/WSL runtime manifests and operational scripts.
- `scripts/` contains one-shot import, backfill, probe, and release utilities.

## Module boundaries

- Parse transport payloads at `app/adapters/onebot_models.py`; pass typed
  dataclasses into core services.
- Keep orchestration in a service or router and persistence in repositories.
  `app/core/router.py` depends on repository interfaces/objects instead of
  embedding schema creation.
- Put pure parsing, ranking, formatting, and policy decisions in small
  functions so they can be tested without a gateway or database. Existing
  examples include `app/core/search_policy.py` and
  `app/core/memory_compaction.py`.
- Construct concrete providers and repositories only at composition roots.
  Follow the `build_*` factories in `app/main.py`.
- When synchronous SQLite work is called from an async entry point, offload
  coarse startup work with `asyncio.to_thread`, as in `app/main.py:run`.
- Preserve group/private/dev process separation; a new group-chat capability
  must not accidentally start in the private or development worker.

## Naming and placement

- Use snake_case module/function names and PascalCase classes.
- Use dataclasses (often `slots=True`) for internal typed results and Pydantic
  settings for environment configuration.
- Name repository classes `<Entity>Repository` and runtime service classes
  after their responsibility.
- Add tests under the mirrored area: `app/core/x.py` ->
  `tests/core/test_x.py`, `app/storage/x.py` -> `tests/storage/test_x.py`.
- Put deployment-contract tests at `tests/test_*deployment*.py` rather than
  hiding operational assumptions in shell scripts.

## Avoid

- Do not put SQL/ranking/token-packing details directly in `router.py`.
- Do not make core functions read `.env` or global process state directly.
- Do not place secrets, databases, model caches, or runtime logs in source
  directories or commits.
- Do not couple normal group reply handling to optional background workers;
  failures in maintenance/indexing work must not stop the chat path.
