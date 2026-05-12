# QQ AI Bot

A NapCat-based QQ bot with group chat, private chat, image turns, optional web search, owner-only dev control, and Windows launch scripts for running QQ, NapCat, and the bot together.

## Public Release Notes

- This package is a sanitized public release.
- Runtime data, logs, local databases, image cache, and private chat history are not included.
- `.env.example` and the files under `configs/` are sample values only. Replace them before production use.
- `configs/persona.yaml` ships with an unofficial fan-persona example inspired by Hikigaya Komachi. Swap it out if you want a different persona or tone.

## Setup

1. `python -m venv .venv`
2. `. .\.venv\Scripts\Activate.ps1`
3. `python -m pip install -e ".[dev]"`
4. Copy `.env.example` to `.env`.
5. Fill in the required runtime values:
   - `NAPCAT_WS_URL`
   - `LLM_BASE_URL`
   - `LLM_API_KEY`
   - `LLM_MODEL`
   - `BOT_QQ`
   - `OWNER_QQ`
6. Fill any optional values you need:
   - `LLM_FALLBACK_MODEL`
   - `ADMIN_QQS`
   - `PRIVATE_CHAT_QQS`
   - `SEARCH_PROVIDER`
   - `SEARCH_BASE_URL`
   - `SEARCH_API_KEY`
   - `SEARCH_REGION`
   - `SEARCH_BACKEND`
   - `SEARCH_TIMEOUT_SECONDS`
   - `CONTEXT_RECENT_LIMIT`
   - `CONTEXT_SUMMARY_LIMIT`
   - `CONTEXT_HISTORY_LIMIT`
   - `QQ_EXE_PATH`
   - `NAPCAT_SHELL_DIR`
   - `NAPCAT_BOOT_PATH`
   - `NAPCAT_INJECT_DLL_PATH`
   - `NAPCAT_WAIT_TIMEOUT_SECONDS`
7. Review the example configs under `configs/`:
   - `configs/groups.yaml`
   - `configs/persona.yaml`
   - `configs/private_reminders.yaml`
   - `configs/safety.yaml`
8. Update `configs/groups.yaml` so approved groups have both `enabled: true` and `speak: true`.

## Search Providers

- `SEARCH_PROVIDER=tavily`
  - Requires `SEARCH_API_KEY`
  - Uses the Tavily HTTP API
- `SEARCH_PROVIDER=ddgs`
  - Does not require `SEARCH_API_KEY`
  - Uses the `ddgs` package for live web search
  - Optional tuning:
    - `SEARCH_REGION`, for example `wt-wt` or `cn-zh`
    - `SEARCH_BACKEND`, default `auto`

## Group Scope

Group routing is default-deny for ingestion and speaking outside enabled groups. The current policy is that only groups with both `enabled: true` and `speak: true` are ingested and archived locally. Non-speaking groups stay out of the local history path. The sample config includes one example group `10001` that is allowed to archive, speak, and use proactive reply.

## Run

- Foreground: `python -m app.main`
- Split runtime only: `powershell -ExecutionPolicy Bypass -File start_xiaomachi_runtime.ps1`
- Full launcher: `powershell -ExecutionPolicy Bypass -File start_xiaomachi.ps1`
- Double-click in the repo root: `启动小町.bat` / `关闭小町.bat`
  - `启动小町.bat` starts QQ, NapCat, and the Python bot together
  - If QQ is not logged in yet, it waits for the local websocket and tells you to finish the QQ login first
  - `关闭小町.bat` stops the bot and also closes launcher-managed QQ/NapCat processes
- Service: `powershell -ExecutionPolicy Bypass -File scripts/install_service.ps1`

## Windows Launch Dependencies

- `QQ_EXE_PATH` points to your local `QQ.exe`
- `NAPCAT_SHELL_DIR` points to the local `NapCat.Shell` directory
- `NAPCAT_BOOT_PATH` and `NAPCAT_INJECT_DLL_PATH` are optional overrides if your NapCat layout differs from the default

## Repository Layout

- `app/`: bot runtime, routing, providers, storage, and dev-control flow
- `configs/`: sample runtime configuration
- `scripts/`: service helpers and maintenance scripts
- `tests/`: regression and smoke tests
- `data/`: placeholder runtime directories only; actual runtime artifacts stay ignored

## Safety

- Group speaking is default-deny.
- Admin commands are whitelist-only and never parsed by the LLM.
- Prompt leaks and explicit content are blocked by configuration and policy code.
- This public package intentionally excludes private runtime data and owner-specific history.
