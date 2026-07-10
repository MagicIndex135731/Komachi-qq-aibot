# NapCat WebUI Shortcut Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ASCII Windows batch shortcut that opens the local NapCat QQ login page only when the WebUI is reachable.

**Architecture:** Extend the existing WSL deployment artifact test with static safety assertions, then add one root-level batch file. The batch file uses the Windows-provided `curl.exe` to check the local WebUI and never invokes WSL, Docker, or a startup script.

**Tech Stack:** Windows Batch, curl.exe, pytest

---

### Task 1: Add the tested WebUI shortcut

**Files:**
- Modify: `tests/test_wsl_deployment_artifacts.py`
- Create: `open-napcat-webui.bat`

- [ ] **Step 1: Write the failing artifact test**

```python
def test_napcat_webui_shortcut_only_opens_an_existing_local_service() -> None:
    shortcut = REPO_ROOT / "open-napcat-webui.bat"
    assert shortcut.exists()
    content = shortcut.read_text(encoding="ascii")
    assert "curl.exe" in content
    assert "http://127.0.0.1:6099/webui/qq_login" in content
    assert "start-xiaomachi-wsl.bat" in content
    assert "wsl.exe" not in content.lower()
    assert "docker" not in content.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv-wsl/bin/python -m pytest tests/test_wsl_deployment_artifacts.py::test_napcat_webui_shortcut_only_opens_an_existing_local_service -v`

Expected: FAIL because `open-napcat-webui.bat` does not exist.

- [ ] **Step 3: Add the minimal batch implementation**

```bat
@echo off
setlocal
set "NAPCAT_WEBUI=http://127.0.0.1:6099/webui/qq_login"

curl.exe --silent --fail --max-time 2 http://127.0.0.1:6099/ >nul 2>&1
if errorlevel 1 (
    echo NapCat WebUI is not running.
    echo Run start-xiaomachi-wsl.bat first, then try again.
    pause
    exit /b 1
)

start "" "%NAPCAT_WEBUI%"
exit /b 0
```

- [ ] **Step 4: Run the focused and deployment artifact tests**

Run: `.venv-wsl/bin/python -m pytest tests/test_wsl_deployment_artifacts.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Verify the current offline behavior**

Run: `cmd.exe /c open-napcat-webui.bat < nul`

Expected while login is unavailable: prints `NapCat WebUI is not running.` and does not start WSL or Docker.

- [ ] **Step 6: Commit only the shortcut files when Git identity is available**

```powershell
git add tests/test_wsl_deployment_artifacts.py open-napcat-webui.bat
git commit -m "feat: add NapCat WebUI shortcut"
```

If Git identity remains unavailable, leave the verified files uncommitted and report that limitation.
