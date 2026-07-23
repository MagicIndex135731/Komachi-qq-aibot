# Database Guidelines

## Stack and ownership

SQLite is the primary persistent store and SQLAlchemy 2.x is the access layer.

- Declare mappings with typed `Mapped[...]` columns in
  `app/storage/models.py`.
- Build the engine, enable optional SQLite extensions, create tables, and run
  compatibility migrations in `app/storage/db.py`.
- Put application queries and writes in repository classes in
  `app/storage/repositories.py`.
- Use the `session_scope(engine)` context manager. It commits on success,
  rolls back on exceptions, and always closes the session.
- Tests use the temporary `sqlite_engine` fixture from `tests/conftest.py`.

## Schema and migration rules

The deployed database is upgraded in place during `create_all(engine)`.
There is no destructive migration framework.

- Make every `CREATE TABLE`, `CREATE INDEX`, and data repair idempotent.
- Before adding a column, inspect `PRAGMA table_info`; use
  `_add_missing_columns` rather than assuming a pristine schema.
- Multiple bot processes may initialize concurrently. Treat only the exact
  duplicate-column race as success; re-raise unrelated SQLAlchemy failures.
- Use `IF NOT EXISTS` for indexes and uniqueness constraints where SQLite
  supports it.
- Never delete or recreate `data/bot.db` to apply a schema change.
- Never mutate original messages as a substitute for a derived index or
  summary migration.
- Before a production migration, use SQLite's online backup mechanism against
  the live connection/WAL state and verify the backup with
  `PRAGMA integrity_check`.

`app/storage/db.py:_run_schema_migrations` and
`tests/storage/test_long_term_memory_storage.py` are the primary examples.

## Query and transaction patterns

- Scope group data by `group_id` (or `scope_type` + `scope_id`) in every
  relevant query. Cross-group retrieval is a correctness and privacy bug.
- Keep deterministic ordering explicit, normally timestamp plus stable ID.
- Use database uniqueness for idempotent jobs and derived documents, then make
  repository upserts safe to retry.
- Store provenance (`source_msg_id` or `source_msg_ids`) for summaries and
  memory facts. Derived text is not a replacement for raw messages.
- Normalize timezone-aware datetimes before SQLite comparisons; follow
  `_normalize_utc_sqlite_timestamp` in `repositories.py`.
- Batch related writes in one `session_scope`; do not commit each row inside a
  loop unless resumability explicitly requires separate checkpoints.

## Rebuildable accelerators

FTS and vector tables are accelerators, not sources of truth.

- Their initialization may fail without preventing the bot from starting.
- Rebuild an incompatible derived index from canonical tables instead of
  rewriting source rows.
- Chinese FTS uses FTS5 trigram tokenization; tests must cover environments
  where FTS5 is unavailable.
- `sqlite-vec` loading is best effort. A missing extension disables only the
  vector path.
- Persist model/version/dimension metadata for vector indexes so incompatible
  embeddings can be rebuilt safely.

Existing examples are `_initialize_optional_memory_fts` and
`_initialize_optional_memory_vectors` in `app/storage/db.py`.

## Naming

- Tables and columns: lower snake_case.
- Normal indexes: `ix_<table>_<columns-or-purpose>`.
- Unique indexes: `ux_<table>_<purpose>`.
- Status values are short lowercase strings and must have explicit defaults.

## Avoid

- Do not issue unguarded `ALTER TABLE` statements from multiple startup
  processes.
- Do not silently swallow migration errors that are not an explicitly handled
  optional capability or concurrency race.
- Do not log SQL parameters containing full chat content or credentials.
- Do not make hashed lexical vectors masquerade as semantic embeddings.
