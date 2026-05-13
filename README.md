# QQ AI Bot

A NapCat-based QQ bot for people who want a bot that feels like a real QQ group member instead of a formal assistant.

This project combines:

- natural group chat replies with short, human-sounding interjections
- persistent private chat
- persona configuration
- image-aware turns
- optional live web search
- owner-only local dev control
- Windows launch scripts for QQ + NapCat + bot runtime

It is designed for a personal or small-circle setup where tone, presence, and control matter more than "enterprise bot" features.

## What This Bot Is Good At

Use this if you want a QQ bot that can:

- stay in-character with a configurable persona instead of sounding like customer support
- join group chat occasionally with short, casual replies instead of long lecture-style messages
- keep private chat context across turns
- look at image turns and continue the same topic in follow-up messages
- use live search for time-sensitive or real-world questions
- let the owner switch into a private project-control mode for local repo and runtime work
- run on a Windows machine that already has QQ and NapCat

## Typical Use Cases

### 1. A "present" group companion

Instead of only replying when hard-triggered, the bot can be configured to occasionally join the conversation in enabled groups.

Example vibe:

```text
User A: this milk tea is almost 30 now
User B: that is ridiculous
Bot: yeah that's kind of a scam
```

The goal is not to dump an explanation. The goal is to sound like someone in the chat.

### 2. A private chat bot that remembers what you were talking about

Private chat is useful when you want a single long-running conversation instead of reopening a new thread every time.

Example:

```text
You: I want to keep working on the bot tonight.
Bot: okay, what do you want to tune first
You: the private image recognition is still weak
Bot: then I would start from image follow-up and candidate narrowing first
```

When `Responses` mode is available, the text conversation can continue with `previous_response_id` instead of rebuilding every turn as a brand-new conversation.

### 3. Image follow-up without forcing the user to restate everything

The bot supports QQ image turns and follow-up messages around them.

Example:

```text
You: [send image]
You: who is this again
Bot: I can narrow from the image, but give me the work or game title too if you want a better guess.
You: Blue Archive
Bot: then I should narrow inside Blue Archive first instead of jumping across unrelated series
```

This is useful for:

- character guessing
- meme/image reaction threads
- quoted image follow-ups
- "look at the picture I just sent" style conversations

### 4. Real-world lookup instead of bluffing

For questions that depend on current facts, the bot can use web search instead of pretending it already knows.

Example:

```text
User: will it rain in Hangzhou tomorrow
Bot: let me check. There is a decent chance of rain tomorrow, so bring an umbrella.
```

Search can be powered by `ddgs` or `tavily`, depending on how you configure the runtime.

### 5. Owner-only project and runtime control

This repo also includes a private owner workflow for local project operation and inspection.

Example:

```text
Owner: check why the websocket dropped again
Bot: I will inspect the runtime state, logs, and current repo changes first.
```

This mode is for the owner only. Normal users do not get repo control, restart ability, or admin actions.

## Feature Summary

- Group chat with allowlist-based enable/speak/archive/proactive settings
- Private chat with persistent session context
- Persona and speaking-style configuration in `configs/persona.yaml`
- Image-aware turns, including quoted-image and follow-up-image handling
- Optional live web search for current information
- Owner/admin commands with whitelist enforcement
- Private reminders from `configs/private_reminders.yaml`
- Windows launch scripts for QQ, NapCat, and the bot runtime
- Public release sync workflow for maintaining a sanitized GitHub version

## Quick Start

### Requirements

- Windows
- QQ desktop client
- NapCat / NapCat Shell
- Python `>=3.12`
- An OpenAI-compatible API endpoint

### 1. Install

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

### 2. Create `.env`

Copy `.env.example` to `.env`, then fill at least these values:

```env
NAPCAT_WS_URL=ws://127.0.0.1:3001
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=replace-me
LLM_MODEL=gpt-5.4
BOT_QQ=123456789
OWNER_QQ=987654321
```

Useful optional values:

- `LLM_FALLBACK_MODEL`
- `ADMIN_QQS`
- `PRIVATE_CHAT_QQS`
- `SEARCH_PROVIDER`
- `SEARCH_API_KEY`
- `CONTEXT_RECENT_LIMIT`
- `CONTEXT_SUMMARY_LIMIT`
- `CONTEXT_HISTORY_LIMIT`
- `QQ_EXE_PATH`
- `NAPCAT_SHELL_DIR`

### 3. Review the sample configs

- `configs/groups.yaml`
- `configs/persona.yaml`
- `configs/private_reminders.yaml`
- `configs/safety.yaml`

Before production use:

- replace the example group id in `configs/groups.yaml`
- set `enabled: true` and `speak: true` only for groups you actually want the bot to join
- rewrite the persona if you do not want the included example tone
- only groups with both `enabled: true` and `speak: true` are ingested

### 4. Run the bot

Foreground:

```powershell
python -m app.main
```

Windows launcher:

```powershell
powershell -ExecutionPolicy Bypass -File start_xiaomachi.ps1
```

If you prefer a double-click workflow, root-level `.bat` launchers are also included for local Windows start/stop.
`启动小町.bat` starts QQ, NapCat, and the Python bot together, and `关闭小町.bat` stops the launcher-managed local stack.

## Run Modes

- `python -m app.main`
  - full runtime
- `python -m app.group_main`
  - group runtime only
- `python -m app.private_main`
  - private runtime only
- `python -m app.dev_worker_main`
  - owner/dev worker process
- `powershell -ExecutionPolicy Bypass -File start_xiaomachi_runtime.ps1`
  - split runtime start script
- `powershell -ExecutionPolicy Bypass -File scripts/install_service.ps1`
  - install as a Windows service

## Configuration Map

### `configs/persona.yaml`

Use this to change:

- name and identity
- core traits
- speaking tone
- sentence length
- speech habits
- phrases to avoid

This file is the fastest way to make the bot feel less like a generic assistant and more like "someone specific in the chat".

### `configs/groups.yaml`

Per-group control includes:

- `enabled`
- `archive`
- `speak`
- `proactive_reply`
- `proactive_interval_seconds`

The default stance is deny-by-default outside approved groups.

### `configs/private_reminders.yaml`

Use this for scheduled private messages such as:

- wake-up reminders
- one-off follow-ups
- daily routine nudges

### `configs/safety.yaml`

Use this to tighten:

- sensitive content handling
- prompt leak defenses
- tone boundaries
- reply constraints

## Model and Transport Notes

The public example defaults to:

```env
LLM_MODEL=gpt-5.4
```

This repo supports a mixed transport approach:

- use `Responses` for text conversations when available
- keep a compat path for models or proxies that still behave better on chat-completions-style payloads
- fall back per runtime configuration

Example proxy-style setup:

```env
LLM_MODEL=cc-gpt-5.4
LLM_FALLBACK_MODEL=gpt-5.4
```

That lets you keep a compat-facing model name for proxy handling while still using `gpt-5.4` as the `Responses` text model when supported.

## Repository Layout

- `app/`
  - runtime, routing, model client, storage, image flow, search flow, and dev-control logic
- `configs/`
  - sample runtime config
- `scripts/`
  - launchers, service helpers, sync scripts, maintenance scripts
- `tests/`
  - regression and smoke tests
- `data/`
  - placeholder directories only in the public release

## Public Release Notes

This GitHub version is a sanitized public release.

Not included:

- local message database
- group archive history
- private chat history
- runtime logs
- image cache
- owner-only local control state
- `.env`

Sample files only:

- `.env.example`
- everything under `configs/`

Replace those values before real use.

## Safety and Scope

- Group speaking is default-deny outside approved groups.
- Admin commands are whitelist-only and not parsed by the LLM.
- The public release intentionally excludes private runtime data.
- This is not an official Tencent or NapCat project.

## Limitations

- This is a local self-hosted bot, not a one-click cloud SaaS.
- You still need QQ + NapCat correctly installed and connected.
- Image understanding is useful, but not magic; better source images and clearer follow-up context improve results.
- Persona quality depends heavily on your config, group policy, and model choice.

If you want a QQ bot with personality, controlled group participation, private chat continuity, and a practical Windows local runtime, this repo is built for that use case.
