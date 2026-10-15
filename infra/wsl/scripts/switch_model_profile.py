"""Switch private provider overlays without rebuilding images or QQ services.

Run with Python 3 in WSL as root; the Windows launcher selects that environment.
Secrets travel through files/stdin, never command arguments or console output.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import urlparse


DEFAULTS = {
    "LLM_BASE_URL": "", "LLM_API_KEY": "", "LLM_MODEL": "",
    "LLM_FALLBACK_MODEL": "", "LLM_TEXT_ENDPOINT": "responses",
    "LLM_REASONING_EFFORT": "low", "LLM_VISION_MODEL": "",
    "LLM_SUPPORTS_VISION_INPUT": "true", "LLM_TIMEOUT_SECONDS": "180",
    "LLM_BUILTIN_WEB_SEARCH": "false", "LLM_BUILTIN_WEB_SEARCH_CONTEXT_SIZE": "high",
    "LLM_WEB_SEARCH_MODEL": "", "LLM_WEB_SEARCH_API_KEY": "",
    "LLM_CONTEXT_WINDOW_TOKENS": "258000", "LLM_MAX_OUTPUT_TOKENS": "8192",
    "LLM_CONTEXT_SAFETY_MARGIN_TOKENS": "32768", "LLM_TOOL_CONTEXT_RESERVE_TOKENS": "32768",
    "PROACTIVE_JUDGE_MODEL": "", "PROACTIVE_JUDGE_REASONING_EFFORT": "low",
    "GROUP_AGENT_DECISION_REASONING_EFFORT": "low",
    "MEMORY_COMPACTION_REASONING_EFFORT": "low",
    "MEMORY_EPISODE_TOPIC_JUDGE_REASONING_EFFORT": "low",
    "MEMORY_EPISODE_POST_SEGMENT_REASONING_EFFORT": "low",
    "WEB_SEARCH_REASONING_EFFORT": "low",
    "SEARCH_PROVIDER": "ddgs", "SEARCH_BASE_URL": "", "SEARCH_API_KEY": "",
    "SEARCH_BACKEND": "auto", "SEARCH_REGION": "wt-wt", "SEARCH_TIMEOUT_SECONDS": "8",
    "GROUP_IMAGE_BASE_URL": "", "GROUP_IMAGE_API_KEY": "",
    "GROUP_IMAGE_CHAT_BASE_URL": "", "GROUP_IMAGE_CHAT_API_KEY": "",
    "GROUP_IMAGE_TRANSPORT": "responses", "GROUP_IMAGE_MODEL": "gpt-image-2",
    "GROUP_IMAGE_EDIT_MODEL": "", "GROUP_IMAGE_RESPONSES_MODEL": "",
    "GROUP_IMAGE_GENERATIONS_ENDPOINT": "/images/generations",
    "GROUP_IMAGE_EDITS_ENDPOINT": "/images/edits", "GROUP_IMAGE_SIZE": "auto",
    "GROUP_IMAGE_QUALITY": "high", "GROUP_IMAGE_OUTPUT_FORMAT": "png",
    "GROUP_IMAGE_TIMEOUT_SECONDS": "900",
}
TEXT_MODEL_KEYS = ("LLM_MODEL", "LLM_FALLBACK_MODEL", "LLM_VISION_MODEL")
REASONING_KEYS = tuple(key for key in DEFAULTS if key.endswith("REASONING_EFFORT") and key != "WEB_SEARCH_REASONING_EFFORT")
EFFORTS = {"", "minimal", "low", "medium", "high", "xhigh", "max"}
ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*(.*)$")


class SwitchError(RuntimeError):
    """Only safe, authored diagnostics may be displayed by the launcher."""


def parse_env(text: str, *, strict: bool = False) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = ASSIGNMENT.match(line)
        if not match:
            if strict:
                raise SwitchError("invalid profile assignment")
            continue
        key, value = match.groups()
        if strict and (key not in DEFAULTS or key in result):
            raise SwitchError("profile contains an unapproved or duplicate key")
        value = value.strip()
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1].replace("\\'", "'").replace("\\\\", "\\")
        elif value.startswith('"') and value.endswith('"'):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise SwitchError("invalid quoted profile value") from None
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        result[key] = value
    return result


def env_value(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:@+%,=\-]*", value):
        return value
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def overlay_env(original: str, updates: dict[str, str]) -> str:
    newline = "\r\n" if "\r\n" in original else "\n"
    lines, seen = [], set()
    for line in original.splitlines(keepends=True):
        match = ASSIGNMENT.match(line.rstrip("\r\n"))
        if match and match[1] in updates:
            key = match[1]
            if key not in seen:
                lines.append(f"{key}={env_value(updates[key])}{newline}")
                seen.add(key)
        else:
            lines.append(line)
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += newline
    lines.extend(f"{key}={env_value(value)}{newline}" for key, value in updates.items() if key not in seen)
    return "".join(lines)


def validate_profile(values: dict[str, str]) -> dict[str, str]:
    if set(values) - DEFAULTS.keys():
        raise SwitchError("profile contains settings outside the model/provider overlay")
    settings = {**DEFAULTS, **values}
    if any(not isinstance(value, str) or any(c in value for c in "\n\r\x00") for value in settings.values()):
        raise SwitchError("profile values must be single-line strings")
    if not all(settings[key].strip() for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "GROUP_IMAGE_MODEL")):
        raise SwitchError("profile is missing a required provider setting")
    for key in ("LLM_BASE_URL", "GROUP_IMAGE_BASE_URL", "GROUP_IMAGE_CHAT_BASE_URL", "SEARCH_BASE_URL"):
        if settings[key]:
            url = urlparse(settings[key])
            if url.scheme not in {"https", "http"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise SwitchError("profile contains an invalid provider URL")
    if settings["LLM_TEXT_ENDPOINT"] not in {"responses", "chat_completions"}:
        raise SwitchError("unsupported text endpoint")
    if settings["GROUP_IMAGE_TRANSPORT"] not in {"images", "responses"}:
        raise SwitchError("unsupported image transport")
    for key, value in settings.items():
        if key.endswith("REASONING_EFFORT") and value not in EFFORTS | ({"none"} if key == "WEB_SEARCH_REASONING_EFFORT" else set()):
            raise SwitchError("unsupported reasoning effort")
    return settings


def with_reasoning(settings: dict[str, str], effort: str, sidecar: Path) -> dict[str, str]:
    if effort not in {"low", "medium", "high", "tiered"}:
        raise SwitchError("unsupported reasoning choice")
    try:
        aliases = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise SwitchError("missing verified reasoning model map") from None
    if not isinstance(aliases, dict) or effort not in aliases or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in aliases.items()):
        raise SwitchError("this reasoning choice has no verified model alias")
    if settings["LLM_MODEL"] not in aliases.values():
        raise SwitchError("profile main model does not match its reasoning map")
    updated = dict(settings)
    for key in TEXT_MODEL_KEYS:
        if settings[key] in aliases.values():
            updated[key] = aliases[effort]
    # Tiered is a provider alias, not an AppSettings reasoning token. Empty
    # primary effort preserves that alias; independent auxiliary calls stay low.
    updated["LLM_REASONING_EFFORT"] = "" if effort == "tiered" else effort
    for key in ("PROACTIVE_JUDGE_MODEL",):
        if settings[key] in aliases.values():
            if "low" not in aliases:
                raise SwitchError("reasoning map is missing the verified auxiliary low alias")
            updated[key] = aliases["low"]
    for key in REASONING_KEYS:
        if key != "LLM_REASONING_EFFORT":
            updated[key] = "low"
    return validate_profile(updated)


def atomic_write(path: Path, content: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".profile-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# Input is confidential JSON on stdin. Suppress upstream logs and return only
# an exit code; neither error bodies nor the synthetic model answer are shown.
PROBE_CODE = r'''
import json, logging, os, sys
logging.disable(logging.CRITICAL)
try:
    payload = json.load(sys.stdin)
    settings = payload['settings']
    if payload['verify'] and any(os.environ.get(k) != v for k, v in settings.items()):
        raise ValueError('runtime mismatch')
    os.environ.update(settings)
    from app.config import AppSettings
    from app.providers.llm_client import LlmClient
    config = AppSettings()
    if payload['billed']:
        client = LlmClient(base_url=config.llm_base_url, api_key=config.llm_api_key,
            model=config.llm_model, responses_model=config.llm_model,
            responses_only=config.llm_text_endpoint == 'responses',
            reasoning_effort=config.llm_reasoning_effort, max_output_tokens=1024,
            timeout_seconds=75, request_max_attempts=1)
        try:
            answer = client.generate_text(['This is a synthetic model switch check. Reply only XIAOMACHI_PROFILE_OK.'], allow_web_search=False)
            if 'XIAOMACHI_PROFILE_OK' not in answer:
                raise ValueError('unexpected model check answer')
        finally:
            client.http_client.close()
except Exception:
    sys.exit(1)
'''


class Runtime:
    def __init__(self, install_root: Path, local_env: Path, profiles: Path):
        self.root = install_root
        self.local_env = local_env
        self.shared_env = install_root / "shared/.env"
        self.profiles = profiles

    @contextmanager
    def lock(self):
        import fcntl
        path = self.root / "shared/runtime/start.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SwitchError("another startup or model switch is in progress") from None
            yield

    def execute(self, command: list[str], *, input_text: str | None = None, timeout: int = 120) -> str:
        try:
            result = subprocess.run(command, input=input_text, capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise SwitchError(f"runtime command unavailable or timed out: {command[0]}") from None
        if result.returncode:
            raise SwitchError(f"runtime check failed: {command[0]} (exit {result.returncode})")
        return result.stdout

    def compose_command(self, original: dict[str, str]) -> list[str]:
        compose = {"llbot": "docker-compose.llbot.yml", "snowluma": "docker-compose.snowluma.yml", "napcat": "docker-compose.yml"}.get(original.get("QQ_PLATFORM", "").lower())
        if not compose:
            raise SwitchError("runtime QQ platform is missing or unsupported")
        # Resolve once before mutation, so an unrelated release activation
        # cannot change the manifests used by this transaction or its rollback.
        directory = (self.root / "current/infra/wsl").resolve()
        command = ["docker", "compose", "--env-file", str(self.shared_env), "-f", str(directory / compose)]
        if original.get("ENABLE_GPU") == "1":
            command += ["-f", str(directory / "docker-compose.gpu.yml")]
        return command + ["up", "-d", "--no-deps", "--force-recreate", "--no-build", "xiaomachi", "xiaomachi-private"]

    def probe(self, settings: dict[str, str], *, verify: bool, billed: bool = True, container: str = "xiaomachi-bot") -> None:
        self.execute(["docker", "exec", "-i", container, "python", "-c", PROBE_CODE],
                     input_text=json.dumps({"settings": settings, "verify": verify, "billed": billed}), timeout=100)

    def protected_identities(self, current: dict[str, str]) -> dict[str, str]:
        gateway = {"llbot": "xiaomachi-llbot", "snowluma": "xiaomachi-snowluma", "napcat": "xiaomachi-napcat"}[current["QQ_PLATFORM"].lower()]
        identities = {gateway: self.execute(["docker", "inspect", "--format", "{{.Id}}", gateway]).strip()}
        try:
            identities["xiaomachi-sandbox"] = self.execute(["docker", "inspect", "--format", "{{.Id}}", "xiaomachi-sandbox"]).strip()
        except SwitchError:
            pass  # The computation sidecar is optional.
        return identities

    def application_images(self) -> dict[str, str]:
        images = {}
        for service, container in (("xiaomachi", "xiaomachi-bot"), ("xiaomachi-private", "xiaomachi-private")):
            image = self.execute(["docker", "inspect", "--format", "{{.Image}}", container]).strip()
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
                raise SwitchError("cannot determine the running application image")
            images[service] = image
        return images

    def capture(self, name: str) -> None:
        target = self.profiles / f"{name}.env"
        if target.exists():
            raise SwitchError("profile already exists; refusing to overwrite its saved credentials")
        current = parse_env(self.shared_env.read_bytes().decode("utf-8-sig"))
        settings = validate_profile({key: current.get(key, value) for key, value in DEFAULTS.items()})
        atomic_write(target, overlay_env("# Private model/provider overlay. Never publish this file.\n", settings).encode("utf-8"))
        print(f"Saved private profile: {name}", flush=True)

    def activate(self, name: str, settings: dict[str, str]) -> None:
        settings = validate_profile(settings)
        local_original, shared_original = self.local_env.read_bytes(), self.shared_env.read_bytes()
        current = parse_env(shared_original.decode("utf-8-sig"))
        command = self.compose_command(current)
        status_script = Path(command[command.index("-f") + 1]).parent / "scripts/status.sh"
        images = self.application_images()
        protected = self.protected_identities(current)
        profile_path, state_path = self.profiles / f"{name}.env", self.profiles / "current.json"
        profile_original = profile_path.read_bytes() if profile_path.exists() else None
        state_original = state_path.read_bytes() if state_path.exists() else None
        print("Checking candidate configuration and provider (one bounded synthetic request)...", flush=True)
        self.probe(settings, verify=False)
        print("Saving private backups and recreating the group/private applications...", flush=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
        backup = self.root / "shared/runtime/model-profile-backups" / stamp
        atomic_write(backup / "source.env", local_original)
        atomic_write(backup / "shared.env", shared_original)
        # A model switch must preserve the exact code of both running apps.
        # The mutable :local tag may point at an unactivated candidate build.
        # JSON is also valid Compose YAML; never pull or build these image IDs.
        image_override = backup / "apps.compose.yml"
        atomic_write(image_override, json.dumps({"services": {
            service: {"image": image, "pull_policy": "never"}
            for service, image in images.items()
        }}, indent=2).encode("utf-8"))
        command[-7:-7] = ["-f", str(image_override)]
        watchdog_active = False
        try:
            self.execute(["systemctl", "is-active", "--quiet", "xiaomachi-watchdog.service"])
            watchdog_active = True
        except SwitchError:
            pass
        if watchdog_active:
            self.execute(["systemctl", "stop", "xiaomachi-watchdog.service"])
        watchdog_stopped = watchdog_active
        changed = False
        try:
            changed = True
            atomic_write(self.local_env, overlay_env(local_original.decode("utf-8-sig"), settings).encode("utf-8"))
            atomic_write(self.shared_env, overlay_env(shared_original.decode("utf-8-sig"), settings).encode("utf-8"))
            self.execute(command, timeout=120)
            print("Waiting for deployed readiness and verifying the selected provider...", flush=True)
            self.execute(["bash", str(status_script)], timeout=600)
            self.probe(settings, verify=True)
            self.probe(settings, verify=True, billed=False, container="xiaomachi-private")
            if self.protected_identities(current) != protected:
                raise SwitchError("QQ or computation container identity changed during model switch")
            if self.application_images() != images:
                raise SwitchError("application image identity changed during model switch")
            if watchdog_active:
                self.execute(["systemctl", "start", "xiaomachi-watchdog.service"])
                watchdog_stopped = False
            # Persist a successful strength choice, so toggling away and back
            # uses the user's last selected Gemini setting.
            atomic_write(profile_path, overlay_env("# Private model/provider overlay. Never publish this file.\n", settings).encode("utf-8"))
            atomic_write(state_path, json.dumps({"profile": name, "model": settings["LLM_MODEL"], "reasoning": settings["LLM_REASONING_EFFORT"], "image_model": settings["GROUP_IMAGE_MODEL"], "updated_at": stamp}, indent=2).encode("utf-8"))
        except (Exception, KeyboardInterrupt) as failure:
            if changed:
                print("Activation failed; restoring both previous environment files and applications...", flush=True)
                try:
                    atomic_write(self.local_env, local_original)
                    atomic_write(self.shared_env, shared_original)
                    self.execute(command, timeout=120)
                    self.execute(["bash", str(status_script)], timeout=600)
                    if self.application_images() != images:
                        raise SwitchError("rollback application image identity mismatch")
                    for path, original in ((profile_path, profile_original), (state_path, state_original)):
                        if original is not None:
                            atomic_write(path, original)
                        else:
                            path.unlink(missing_ok=True)
                except (Exception, KeyboardInterrupt):
                    raise SwitchError("activation failed; rollback needs manual runtime verification; private backups retained") from None
            raise SwitchError(f"activation failed ({type(failure).__name__}); previous configuration rolled back") from None
        finally:
            if watchdog_stopped:
                self.execute(["systemctl", "start", "xiaomachi-watchdog.service"])
        print(f"Model profile active: {name}; model={settings['LLM_MODEL']}; reasoning={settings['LLM_REASONING_EFFORT']}; image={settings['GROUP_IMAGE_MODEL']}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--profile", choices=("sol", "gemini"))
    action.add_argument("--capture", choices=("sol", "gemini"))
    action.add_argument("--list", action="store_true")
    parser.add_argument("--reasoning", choices=("minimal", "low", "medium", "high", "tiered"))
    parser.add_argument("--check", action="store_true", help="check candidate using one model request; do not activate")
    parser.add_argument("--install-root", type=Path, default=Path("/opt/xiaomachi"))
    wsl_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--local-env", type=Path, default=wsl_dir / ".env")
    parser.add_argument("--profiles-dir", type=Path, default=wsl_dir / "runtime/model-profiles")
    args = parser.parse_args()
    runtime = Runtime(args.install_root, args.local_env, args.profiles_dir)
    try:
        if args.list:
            print("Available private profiles: " + ", ".join(name for name in ("sol", "gemini") if (args.profiles_dir / f"{name}.env").is_file()))
            return 0
        if os.name != "posix" or os.geteuid() != 0:
            raise SwitchError("run this utility as root in WSL using the PowerShell launcher")
        with runtime.lock():
            if args.capture:
                runtime.capture(args.capture)
            else:
                settings = validate_profile(parse_env((args.profiles_dir / f"{args.profile}.env").read_bytes().decode("utf-8-sig"), strict=True))
                if args.reasoning:
                    if args.profile == "sol":
                        if args.reasoning != "low":
                            raise SwitchError("the saved Sol profile uses low reasoning")
                    else:
                        settings = with_reasoning(settings, args.reasoning, args.profiles_dir / f"{args.profile}.reasoning.json")
                if args.check:
                    runtime.probe(settings, verify=False)
                    print("Candidate configuration/provider check passed; no activation performed.")
                else:
                    runtime.activate(args.profile, settings)
    except SwitchError as error:
        print(f"Model switch failed: {error}")
        return 1
    except (OSError, ValueError, KeyboardInterrupt) as error:
        print(f"Model switch failed: {type(error).__name__}; inspect private profile files and runtime.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
