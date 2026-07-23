# Backend Development Guidelines

## Scope

These rules cover the Python 3.12 QQ bot, SQLite persistence, asynchronous
workers, external providers, tests, and WSL/Docker deployment contracts.

## Pre-Development Checklist

Always read:

- [Directory Structure](./directory-structure.md)
- [Quality Guidelines](./quality-guidelines.md)

Also read:

- [Database Guidelines](./database-guidelines.md) for schema, repository,
  migration, retrieval-index, or persistent-job changes.
- [Error Handling](./error-handling.md) for adapters, providers, routers, and
  background services.
- [Logging Guidelines](./logging-guidelines.md) for runtime, provider, worker,
  shadow-mode, or deployment changes.
- `../guides/cross-layer-thinking-guide.md` for work crossing three or more
  layers.
- `../guides/code-reuse-thinking-guide.md` before changing configuration or
  adding helpers.

## Guidelines Index

| Guide | Project-specific focus |
|---|---|
| [Directory Structure](./directory-structure.md) | Runtime boundaries and module ownership |
| [Database Guidelines](./database-guidelines.md) | SQLAlchemy/SQLite, idempotent migrations, derived indexes |
| [Error Handling](./error-handling.md) | Safe degradation, retries, cancellation, QQ blocked output |
| [Logging Guidelines](./logging-guidelines.md) | Event-style logs and sensitive-data restrictions |
| [Quality Guidelines](./quality-guidelines.md) | Offline tests, compatibility, deployment verification |
| [Memory Orchestration V2](./memory-orchestration-v2.md) | Episode jobs, scoped retrieval, backfill, evaluation, and rollout contracts |

## Quality Check

Before accepting backend work:

1. Run focused tests, then `python -m pytest -q`.
2. Run `python -m compileall -q app scripts`.
3. Verify cross-layer configuration/schema fields at every consumer.
4. Check group isolation, source-message provenance, retries, and fallbacks.
5. Inspect logs and staged files for chat content, credentials, `.env`,
   databases, and model caches.
6. For deployment changes, validate Compose, build the image, and perform
   runtime/OneBot smoke checks.
