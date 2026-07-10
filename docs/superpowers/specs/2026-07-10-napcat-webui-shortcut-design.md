# NapCat WebUI Shortcut Design

## Goal

Provide an ASCII-named Windows batch entry in the repository root that opens the local NapCat QQ login page without starting or changing any runtime service.

## Behavior

- The entry file is `open-napcat-webui.bat`.
- It checks whether TCP port `127.0.0.1:6099` is reachable.
- If reachable, it opens `http://127.0.0.1:6099/webui/qq_login` in the default browser.
- If unreachable, it tells the user to run `start-xiaomachi-wsl.bat` first and exits without starting WSL or Docker.
- The batch file name and contents remain ASCII to avoid Windows command-line encoding problems.

## Security

The shortcut does not read, print, or persist the NapCat WebUI token, QQ credentials, `.env` values, login state, QR codes, logs, caches, or API keys.

## Verification

An artifact test will verify the batch file is ASCII-only, points to the expected local URL, checks port 6099, references the existing start entry in its failure message, and does not invoke WSL or Docker.
