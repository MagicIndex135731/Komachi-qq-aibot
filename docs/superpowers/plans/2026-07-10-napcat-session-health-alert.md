# NapCat Session Health Alert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect an invalid or unresponsive QQ session without relying on group activity, recover it once, and show a Windows login alert when manual action is required.

**Architecture:** Extend the WSL Python watchdog with two independent read-only probes. The authenticated NapCat WebUI probe reports explicit login errors, while OneBot `get_group_list` with `no_cache=true` actively requests NTQQ state; three consecutive failures trigger one NapCat restart. After the recovery grace period, an explicit WebUI login failure or continued active-probe failure launches the existing Windows notifier.

**Tech Stack:** Python 3, asyncio/websockets, urllib, Windows PowerShell, Windows Batch, pytest

---

### Task 1: Specify session-health behavior

**Files:**
- Modify: `tests/test_wsl_onebot_watchdog.py`
- Modify: `tests/test_wsl_deployment_artifacts.py`

- [ ] Add failing state-machine tests for three active-probe failures, incident reset after successful active probing, immediate explicit login-error notification, and notification after a failed post-restart recovery.
- [ ] Add failing probe tests with local fake WebSocket and HTTP servers so no QQ account, group message, or model call is used.
- [ ] Add a failing artifact test for the ASCII-only `open-napcat-webui.bat` entry.
- [ ] Run the focused tests and confirm failure because the new fields, probes, and BAT do not exist.

### Task 2: Implement active session probes

**Files:**
- Modify: `infra/wsl/scripts/onebot_watchdog.py`

- [ ] Add a WebUI login probe that authenticates using the local `webui.json` token, never logs the token or credential, and returns only `isLogin`, `isOffline`, and `loginError`.
- [ ] Extend the OneBot probe to request both `get_status` and `get_group_list` with `no_cache=true`, with bounded timeouts.
- [ ] Extend persisted state with consecutive active-probe failures while preserving compatibility with existing state JSON.
- [ ] Restart NapCat once after three active-probe failures or three explicit offline checks.
- [ ] Notify immediately for an explicit WebUI login error, and notify after recovery grace when the session remains unavailable.
- [ ] Run focused watchdog tests until all pass.

### Task 3: Add the Windows login entry and alert text

**Files:**
- Create: `open-napcat-webui.bat`
- Modify: `infra/wsl/scripts/notify_windows.ps1`

- [ ] Add an ASCII BAT that opens `/webui/qq_login` only when port 6099 is reachable and otherwise points to `start-xiaomachi-wsl.bat`.
- [ ] Add notifier reasons for explicit QQ login failure and active-session recovery failure, with no secrets in the displayed text.
- [ ] Run the watchdog and deployment artifact tests.

### Task 4: Verify runtime behavior

**Files:**
- No production file changes expected.

- [ ] Run all WSL watchdog and deployment artifact tests.
- [ ] Run the watchdog once against the currently logged-out NapCat and confirm it records an explicit login failure without printing credentials.
- [ ] After the user completes QQ login, verify `CheckLoginStatus.isLogin=true`, OneBot `online=true`, `get_login_info` is account `1807533371`, and `get_group_list(no_cache=true)` succeeds.
- [ ] Do not send a group message or invoke a model during automated verification.
