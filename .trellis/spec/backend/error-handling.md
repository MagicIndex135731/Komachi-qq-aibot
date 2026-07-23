# Error Handling

## Boundary rules

Use exceptions for failures and catch them at the boundary that can either
recover, degrade, retry, or present a safe user-visible result.

- Validate malformed configuration/file shapes with `ValueError`, as in
  `app/config.py:_read_yaml` and `app/private_reminders.py`.
- External gateway/provider adapters should preserve actionable failure
  details while avoiding secrets. `app/adapters/sender.py` uses the dedicated
  `QQMessageBlockedError` when QQ rejects sensitive output.
- Transaction failures must propagate through
  `app/storage/db.py:session_scope`, which rolls back automatically.
- Background worker loops catch per-job failures, record/retry them, and
  continue. They must still propagate `asyncio.CancelledError` during
  shutdown.
- Optional accelerators/providers may degrade to a documented fallback; core
  persistence and normal replies may not be silently skipped.

## Recovery and propagation

- Catch the narrowest expected exception set. When a broad `Exception` is
  required at a long-running service boundary, log with `logger.exception`
  and keep the protected scope small.
- Preserve exception chaining when converting a provider/library failure into
  a domain-specific error.
- Use finite timeouts for network calls, optional query rewriting, and worker
  shutdown. A timeout must have a deterministic fallback.
- When cancelling tasks, await them with `asyncio.gather(...,
  return_exceptions=True)` only during cleanup, as demonstrated by the runtime
  entry points.
- Return safe, useful QQ-facing messages from routing/sender boundaries; never
  expose stack traces, provider payloads, tokens, or blocked sensitive text.

## QQ blocked-output invariant

The blocked reply path is intentional behavior, not a generic provider error.
Preserve the contextual note and the rule that later replies must not
reproduce the blocked sensitive details. Tests in
`tests/core/test_router.py`, `tests/core/test_context_builder.py`, and
`tests/adapters/test_sender.py` are the contract.

## Testing failures

- Use `pytest.raises` for synchronous failures and async tests for cancellation
  or timeout paths.
- Fake external services; never require live LLM, embedding, search, OneBot, or
  image services in unit tests.
- Assert both failure outcome and side effects: rollback, retry state, retained
  raw data, or continued worker operation.

## Avoid

- Do not use `except Exception: pass`.
- Do not downgrade schema corruption, group leakage, or lost source-message
  provenance to a best-effort warning.
- Do not let optional memory/index failures block a normal group reply.
- Do not retry forever or retry non-idempotent work without an idempotency key.
