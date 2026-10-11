"""Optional Docker-only computation tool with fixed resource and source limits.

This tool never evaluates model code in the bot process or a host interpreter.
Outputs live in a size-limited container tmpfs; only bounded, verified regular
files are copied into the artifact directory after model code has finished.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import tarfile
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol
from uuid import uuid4

from PIL import Image


SANDBOX_IMAGE = "sha256:f3a81c821fa7a0566e19601000f863d3ba332c64ee363f54073dcb0a87ef3ce0"
_PYTHON = "/usr/local/bin/python"
_EXECUTION_SECONDS = 20
_CONTROL_SECONDS = 5
_CODE_BYTES = 8192
_STDOUT_BYTES = 16384
_ARCHIVE_BYTES = 40 * 1024 * 1024
_FILE_BYTES = 8 * 1024 * 1024
_TOTAL_FILE_BYTES = 16 * 1024 * 1024
_MAX_FILES = 8
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_ARCHIVE_EXPORT = (
    "import sys,tarfile;"
    "archive=tarfile.open(fileobj=sys.stdout.buffer,mode='w|');"
    "archive.add('/outputs',arcname='.',recursive=True);archive.close()"
)
_INPUT_IMPORT = """import base64,json,os,re,sys
items=json.loads(sys.stdin.read(24*1024*1024))
if not isinstance(items,list) or len(items)>8: raise ValueError('invalid input list')
total=0
for item in items:
    name=item['name']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\\.[a-z0-9]{1,10}',name): raise ValueError('invalid input name')
    contents=base64.b64decode(item['base64'],validate=True)
    total+=len(contents)
    if len(contents)>8388608 or total>16777216: raise ValueError('input byte limit')
    with open('/inputs/'+name,'xb') as stream: stream.write(contents)
    os.chmod('/inputs/'+name,0o444)
"""


@dataclass(frozen=True, slots=True)
class SandboxResult:
    stdout: str
    artifacts: tuple[Path, ...]
    status: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class DockerProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes = b""


class SandboxOutputLimitError(RuntimeError):
    def __init__(self, stdout: bytes):
        super().__init__("Docker output exceeded its bounded capture budget")
        self.stdout = stdout


class DockerExecutor(Protocol):
    async def execute(
        self, argv: Sequence[str], *, stdin: bytes = b"", maximum_output: int = _STDOUT_BYTES
    ) -> DockerProcessResult: ...


class DockerCommandExecutor:
    """Invoke an administrator-configured Docker CLI without any shell."""

    async def execute(
        self, argv: Sequence[str], *, stdin: bytes = b"", maximum_output: int = _STDOUT_BYTES
    ) -> DockerProcessResult:
        process = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout = bytearray()
        stderr = bytearray()
        readers = []

        async def read(stream, captured):
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                captured.extend(chunk)
                if len(stdout) + len(stderr) > maximum_output:
                    raise SandboxOutputLimitError(bytes(stdout + stderr)[:maximum_output])

        try:
            if stdin:
                process.stdin.write(stdin)
                await process.stdin.drain()
            process.stdin.close()
            readers = [asyncio.create_task(read(process.stdout, stdout)),
                       asyncio.create_task(read(process.stderr, stderr))]
            await asyncio.gather(*readers)
            return DockerProcessResult(await process.wait(), bytes(stdout), bytes(stderr))
        finally:
            for reader in readers:
                if not reader.done():
                    reader.cancel()
            if readers:
                await asyncio.gather(*readers, return_exceptions=True)
            if process.returncode is None:
                try:
                    process.kill()
                    await asyncio.wait_for(process.wait(), 1)
                except (asyncio.TimeoutError, ProcessLookupError):
                    pass


class PythonSandbox:
    """Run bounded computation with files supplied by the trusted tool executor.

    ``enabled``, ``docker_command``, path mapping, and input roots are runtime
    configuration, not model tool arguments. No Docker availability means an
    explicit unavailable result. The bot deployment does not acquire a Docker
    socket or a host subprocess fallback from enabling this class.
    """

    def __init__(
        self,
        *,
        output_root: Path | str | None = None,
        allowed_input_roots: Sequence[Path | str] = (),
        enabled: bool = False,
        executor: DockerExecutor | None = None,
        docker_command: Sequence[str] = ("docker",),
        docker_path_mapper: Callable[[Path], str] | None = None,
    ) -> None:
        root = Path(output_root) if output_root is not None else Path("data/autonomous_artifacts/sandbox").resolve()
        if not root.is_absolute() or PureWindowsPath(str(root)).drive.startswith("\\\\"):
            raise ValueError("Sandbox artifact root must be an absolute local directory")
        if type(enabled) is not bool or not docker_command or any(not isinstance(v, str) or not v for v in docker_command):
            raise ValueError("Sandbox runtime configuration is invalid")
        roots = []
        for value in allowed_input_roots:
            path = Path(value)
            if not path.is_absolute() or PureWindowsPath(str(path)).drive.startswith("\\\\"):
                raise ValueError("Sandbox input root must be an absolute local directory")
            roots.append(path.resolve())
        self.output_root = root.resolve()
        self.allowed_input_roots = tuple(roots)
        self.enabled = enabled
        self.executor = executor or DockerCommandExecutor()
        self.docker_command = tuple(docker_command)
        self.docker_path_mapper = docker_path_mapper or str
        self._cleanup_failed = False

    async def available(self) -> bool:
        return await self._image_environment() is not None

    async def run(self, code: str, input_files: Mapping[str, Path] | None = None) -> SandboxResult:
        if not self.enabled or self._cleanup_failed:
            return SandboxResult("", (), "unavailable", "sandbox_disabled" if not self.enabled else "sandbox_cleanup_failed")
        if not isinstance(code, str) or not code.strip() or "\x00" in code:
            return SandboxResult("", (), "error", "invalid_code")
        try:
            encoded_code = code.encode("utf-8")
        except UnicodeError:
            return SandboxResult("", (), "error", "invalid_code")
        if len(encoded_code) > _CODE_BYTES:
            return SandboxResult("", (), "error", "invalid_code")
        try:
            inputs = self._capture_inputs({} if input_files is None else input_files)
        except (ValueError, OSError):
            return SandboxResult("", (), "error", "invalid_input_files")
        environment = await self._image_environment()
        if environment is None:
            return SandboxResult("", (), "unavailable", "docker_unavailable")

        run_id = uuid4().hex
        container_name = "xiaomachi-sandbox-" + run_id
        directory = self.output_root / run_id
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {token: "/inputs/" + name for token, name, contents in inputs}
        argv = self._create_command(container_name, environment)
        result = SandboxResult("", (), "error", "sandbox_execution_failed")
        try:
            creation = await self._control(argv)
            if creation.returncode != 0:
                return SandboxResult("", (), "unavailable", "container_creation_failed")
            started = await self._control(self.docker_command + ("start", container_name))
            if started.returncode != 0:
                return SandboxResult("", (), "unavailable", "container_start_failed")
            if inputs:
                payload = json.dumps([
                    {"name": name, "base64": base64.b64encode(contents).decode("ascii")}
                    for token, name, contents in inputs
                ], separators=(",", ":")).encode("ascii")
                imported = await self._control(
                    self.docker_command + (
                        "exec", "--interactive", "--user", "0:0", container_name,
                        _PYTHON, "-I", "-B", "-c", _INPUT_IMPORT,
                    ), stdin=payload,
                )
                if imported.returncode != 0:
                    return SandboxResult("", (), "error", "input_transfer_failed")
            try:
                execution = await asyncio.wait_for(
                    self.executor.execute(
                        self.docker_command + (
                            "exec", "--interactive", container_name,
                            _PYTHON, "-I", "-B", "-u", "-c", _bootstrap(manifest),
                        ),
                        stdin=encoded_code, maximum_output=_STDOUT_BYTES,
                    ),
                    _EXECUTION_SECONDS,
                )
            except asyncio.TimeoutError:
                result = SandboxResult("", (), "timeout", "execution_timeout")
            except SandboxOutputLimitError as exc:
                result = SandboxResult(_safe_text(exc.stdout), (), "error", "stdout_limit")
            else:
                text_output = execution.stdout + execution.stderr
                if len(text_output) > _STDOUT_BYTES:
                    result = SandboxResult(_safe_text(text_output), (), "error", "stdout_limit")
                elif execution.returncode != 0:
                    result = SandboxResult(_safe_text(text_output), (), "error", "python_execution_failed")
                else:
                    copied = await self._control(
                        # Docker's archive API can omit live tmpfs contents;
                        # export from the container mount namespace explicitly.
                        self.docker_command + (
                            "exec", container_name, _PYTHON, "-I", "-B", "-c", _ARCHIVE_EXPORT
                        ),
                        maximum_output=_ARCHIVE_BYTES,
                    )
                    if copied.returncode != 0:
                        result = SandboxResult(_safe_text(text_output), (), "error", "artifact_copy_failed")
                    else:
                        try:
                            artifacts = self._store_artifacts(copied.stdout, directory)
                        except (ValueError, OSError, tarfile.TarError):
                            result = SandboxResult(_safe_text(text_output), (), "error", "invalid_artifact_archive")
                        else:
                            result = SandboxResult(_safe_text(execution.stdout), artifacts, "ok")
        except asyncio.CancelledError:
            raise
        except (OSError, asyncio.TimeoutError, SandboxOutputLimitError):
            result = SandboxResult("", (), "error", "sandbox_execution_failed")
        finally:
            # The container is created separately from start: a cancelled
            # create can leave only a stopped container, never executing code.
            cleaned = await self._remove_container(container_name)
            if not cleaned:
                self._cleanup_failed = True
        if self._cleanup_failed:
            return SandboxResult(result.stdout, (), "error", "container_cleanup_failed")
        return result

    async def _image_environment(self) -> tuple[str, ...] | None:
        if not self.enabled or self._cleanup_failed:
            return None
        try:
            checked = await self._control(
                self.docker_command + ("image", "inspect", "--format", "{{json .Config}}", SANDBOX_IMAGE)
            )
            config = json.loads(checked.stdout)
            if checked.returncode != 0 or not isinstance(config, dict) or config.get("Volumes"):
                return None
            env = config.get("Env", [])
            if not isinstance(env, list) or len(env) > 128:
                return None
            names = []
            for item in env:
                if not isinstance(item, str) or "=" not in item:
                    return None
                key = item.split("=", 1)[0]
                if not _ENV_NAME.fullmatch(key):
                    return None
                names.append(key)
            return tuple(dict.fromkeys(names))
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError, asyncio.TimeoutError, SandboxOutputLimitError):
            return None

    async def _control(self, argv: Sequence[str], *, maximum_output: int = _STDOUT_BYTES,
                       stdin: bytes = b"") -> DockerProcessResult:
        return await asyncio.wait_for(
            self.executor.execute(argv, maximum_output=maximum_output, stdin=stdin), _CONTROL_SECONDS
        )

    async def _remove_container(self, name: str) -> bool:
        try:
            result = await self._control(self.docker_command + ("rm", "--force", name))
            # Docker returns nonzero if a failed create never produced a
            # container. Confirm absence instead of treating that as a leak.
            if result.returncode == 0:
                return True
            absent = await self._control(self.docker_command + ("container", "inspect", name))
            message = (absent.stdout + absent.stderr).lower()
            return absent.returncode != 0 and (b"no such container" in message or b"no such object" in message)
        except (OSError, asyncio.TimeoutError, SandboxOutputLimitError):
            return False

    def _capture_inputs(self, values: Mapping[str, Path]) -> tuple[tuple[str, str, bytes], ...]:
        if not isinstance(values, Mapping) or len(values) > _MAX_FILES:
            raise ValueError("Sandbox inputs must be bounded token references")
        result = []
        total = 0
        for token, path in values.items():
            if not isinstance(token, str) or not _TOKEN.fullmatch(token) or not isinstance(path, Path):
                raise ValueError("Sandbox input token or source is invalid")
            actual = path.resolve()
            if (not path.is_absolute() or not actual.is_file()
                    or not any(actual.is_relative_to(root) for root in self.allowed_input_roots)):
                raise ValueError("Sandbox input is outside trusted artifact roots")
            with actual.open("rb") as stream:
                contents = stream.read(_FILE_BYTES + 1)
            total += len(contents)
            if len(contents) > _FILE_BYTES or total > _TOTAL_FILE_BYTES:
                raise ValueError("Sandbox input byte budget exceeded")
            suffix = actual.suffix.lower()
            if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
                suffix = ".bin"
            result.append((token, token + suffix, contents))
        return tuple(result)

    def _create_command(self, name: str, environment: tuple[str, ...]) -> tuple[str, ...]:
        args = list(self.docker_command) + [
            "create", "--name", name, "--pull", "never", "--network", "none", "--read-only",
            "--memory", "256m", "--memory-swap", "256m", "--cpus", "1", "--pids-limit", "64",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534",
            "--no-healthcheck", "--log-driver", "none", "--workdir", "/outputs",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=32m,mode=1777",
            "--tmpfs", "/outputs:rw,noexec,nosuid,nodev,size=32m,mode=1777",
            "--tmpfs", "/inputs:rw,noexec,nosuid,nodev,size=20m,mode=0755",
            "--tmpfs", "/workspace:ro,noexec,nosuid,nodev,size=1m",
            "--tmpfs", "/app:ro,noexec,nosuid,nodev,size=1m",
            "--ulimit", "fsize=8388608:8388608", "--ulimit", "nofile=64:64",
        ]
        for key in environment:
            args.extend(("--env", key + "="))
        args.extend(("--env", "PATH=/usr/local/bin:/usr/bin:/bin", "--env", "HOME=/tmp"))
        # tmpfs vanishes when a container stops. Keep one bounded supervisor
        # alive while exec runs and validated output is copied; never restart it.
        supervisor = "import os,time;os.environ.clear();time.sleep(60)"
        args.extend(("--entrypoint", _PYTHON, SANDBOX_IMAGE, "-I", "-B", "-u", "-c", supervisor))
        return tuple(args)

    def _store_artifacts(self, raw: bytes, directory: Path) -> tuple[Path, ...]:
        if len(raw) > _ARCHIVE_BYTES:
            raise ValueError("Sandbox archive exceeds its capture budget")
        files = []
        total = 0
        names = set()
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            for index, member in enumerate(archive):
                if index >= 64:
                    raise ValueError("Sandbox archive has too many entries")
                name = member.name.removeprefix("./")
                parts = PurePosixPath(name).parts
                if (not name or "\\" in name or ":" in name or name.startswith("/")
                        or ".." in parts or len(parts) > 4
                        or any(len(part) > 120 or part.startswith(".") for part in parts)):
                    if member.isdir() and name in {".", ""}:
                        continue
                    raise ValueError("Sandbox artifact path is invalid")
                if member.isdir():
                    continue
                if not member.isfile() or name in names or member.size > _FILE_BYTES:
                    raise ValueError("Sandbox archive contains unsafe file entries")
                names.add(name)
                total += member.size
                if len(files) >= _MAX_FILES or total > _TOTAL_FILE_BYTES:
                    raise ValueError("Sandbox artifact budget exceeded")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError("Sandbox artifact has no file payload")
                contents = handle.read(_FILE_BYTES + 1)
                if len(contents) != member.size:
                    raise ValueError("Sandbox artifact content length is invalid")
                _verify_image(name, contents)
                files.append((name, contents))
        # Validate the entire archive before creating any result artifact.
        results = []
        for name, contents in files:
            path = directory.joinpath(*PurePosixPath(name).parts)
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Sandbox artifact escaped its output directory")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(contents)
            results.append(path)
        return tuple(results)


def _verify_image(name: str, contents: bytes) -> None:
    if Path(name).suffix.lower() not in _IMAGE_SUFFIXES:
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(contents)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP", "GIF"} or image.width * image.height > 16_000_000:
                    raise ValueError("Sandbox artifact image exceeds its decode budget")
                image.verify()
            with Image.open(io.BytesIO(contents)) as image:
                image.load()
    except (OSError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise ValueError("Sandbox artifact is not a bounded decodable image") from None


def _safe_text(raw: bytes) -> str:
    text = raw[:_STDOUT_BYTES].decode("utf-8", errors="replace")
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")
    return text.encode("utf-8")[:_STDOUT_BYTES].decode("utf-8", errors="ignore")


def _bootstrap(manifest: dict[str, str]) -> str:
    return (
        "import os,sys,resource;"
        "os.environ.clear();"
        "os.environ.update({'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':'/tmp','TMPDIR':'/tmp',"
        "'MPLCONFIGDIR':'/tmp/matplotlib','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'});"
        "resource.setrlimit(resource.RLIMIT_CPU,(18,20));"
        "resource.setrlimit(resource.RLIMIT_FSIZE,(8388608,8388608));"
        "code=sys.stdin.read(8193);"
        "exec(compile(code,'<autonomous-computation>','exec'),"
        "{'__name__':'__main__','INPUT_FILES':" + repr(manifest) + ", 'OUTPUT_DIR':'/outputs'})"
    )
