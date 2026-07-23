# Logging Guidelines

## Local convention

The project uses the standard `logging` module with message-only process
formatting (`logging.basicConfig(level=logging.INFO, format="%(message)s")`).
Modules that emit repeatedly should use `logger = logging.getLogger(__name__)`;
composition roots may use `logging.info`.

Log messages are compact event names followed by stable `key=value` fields.
Examples are the startup banners in `app/main.py` and operational events in
`app/private_reminders.py` and `app/providers/llm_client.py`.

## Levels

- `INFO`: startup/shutdown, enabled runtime mode, completed jobs, compact
  latency/count/token metrics, and deployment health transitions.
- `WARNING`: recoverable provider degradation, retry scheduling, disabled
  optional FTS/vector capability, and invalid non-fatal external responses.
- `ERROR`/`logger.exception`: an operation failed and was abandoned, or a
  long-running worker caught an unexpected exception.
- `DEBUG`: diagnostic IDs and branch decisions only when they are safe and
  useful; normal production operation does not enable it.

## Required context

Include identifiers and metrics needed to diagnose behavior without logging
the content itself:

- process/component and event name;
- group/scope ID when applicable;
- job/document/episode/source message IDs;
- provider/model/version names;
- candidate counts, selected evidence IDs, token counts, elapsed milliseconds,
  status, retry number, and fallback reason.

Use stable event names so tests can assert them with `caplog`, following
`tests/providers/test_llm_client.py`.

## Sensitive-data rules

Never log:

- API keys, tokens, passwords, cookies, authorization headers, or `.env`
  contents;
- complete private/group chat text, full prompts, full provider payloads, or
  full LLM responses;
- the sensitive details of a QQ-blocked reply;
- raw database/model-cache contents.

For memory shadow mode, log only selected evidence/source IDs, scores, token
counts, timings, and bounded error categories. If a provider error embeds a
request URL or body, sanitize it before logging.

## Avoid

- Do not use f-strings with user text or secret-bearing exception payloads.
- Do not emit one INFO record per historical message during backfill.
- Do not claim a background job succeeded before its transaction commits.
- Do not suppress an unexpected exception without an event and traceback at
  the owning service boundary.
