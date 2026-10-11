from __future__ import annotations

import asyncio
import base64
import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from app.agent.python_sandbox import (
    DockerProcessResult,
    PythonSandbox,
    SANDBOX_IMAGE,
    SandboxOutputLimitError,
)


def archive(files=(), *, special=None):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for name, contents in files:
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            tar.addfile(info, io.BytesIO(contents))
        if special:
            tar.addfile(special)
    return stream.getvalue()


class Executor:
    def __init__(self, *, output=b"42\n", tar=None, failure=None):
        self.output = output
        self.tar = archive() if tar is None else tar
        self.failure = failure
        self.calls = []
        self.started = asyncio.Event()
        self.release = None
        self.cleanups = 0

    async def execute(self, argv, *, stdin=b"", maximum_output=16384):
        self.calls.append((tuple(argv), stdin, maximum_output))
        if "inspect" in argv:
            return DockerProcessResult(0, json.dumps({"Env": ["TOKEN=do-not-pass", "PATH=/secret"], "Volumes": None}).encode())
        if "create" in argv:
            return DockerProcessResult(0, b"container-id\n")
        if "start" in argv:
            return DockerProcessResult(0, b"container-id\n")
        if "exec" in argv:
            if "--user" in argv:
                return DockerProcessResult(0, b"")
            if "--interactive" not in argv:
                return DockerProcessResult(0, self.tar)
            self.started.set()
            if self.release is not None:
                await self.release.wait()
            if self.failure:
                raise self.failure
            return DockerProcessResult(0, self.output)
        if "rm" in argv:
            self.cleanups += 1
            return DockerProcessResult(0, b"")
        raise AssertionError(argv)


def sandbox(tmp_path, executor, **kwargs):
    return PythonSandbox(output_root=tmp_path / "sandbox", executor=executor, enabled=True, **kwargs)


async def test_disabled_sandbox_never_executes_or_uses_host_python(tmp_path):
    executor = Executor()
    result = await PythonSandbox(output_root=tmp_path / "sandbox", executor=executor).run("print(42)")
    assert result.status == "unavailable"
    assert result.reason == "sandbox_disabled"
    assert executor.calls == []
    assert result.artifacts == ()


async def test_success_has_fixed_container_isolation_and_bounded_artifacts(tmp_path):
    executor = Executor(tar=archive([("answer.json", b'{"answer":42}')]))
    result = await sandbox(tmp_path, executor).run("print(42)")
    assert result.status == "ok"
    assert result.stdout == "42\n"
    assert result.artifacts[0].read_bytes() == b'{"answer":42}'
    assert result.artifacts[0].is_relative_to(tmp_path / "sandbox")
    create = next(args for args, _, _ in executor.calls if "create" in args)
    for flag, value in [
        ("--network", "none"), ("--memory", "256m"), ("--memory-swap", "256m"),
        ("--cpus", "1"), ("--pids-limit", "64"), ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"), ("--user", "65534:65534"),
    ]:
        assert create[create.index(flag) + 1] == value
    assert "--read-only" in create
    assert "--no-healthcheck" in create
    assert SANDBOX_IMAGE in create
    assert "/outputs:rw,noexec,nosuid,nodev,size=32m,mode=1777" in create
    assert not any("docker.sock" in arg for arg in create)
    assert not any("/workspace:" in arg and arg != "/workspace:ro,noexec,nosuid,nodev,size=1m" for arg in create)
    assert not any("do-not-pass" in arg for arg in create)
    assert "TOKEN=" in create
    assert not any(arg == "--env-file" for arg in create)
    assert "-I" in create
    assert executor.cleanups >= 1
    assert next(data for argv, data, _ in executor.calls if "exec" in argv) == b"print(42)"


async def test_only_caller_trusted_inputs_are_snapshotted_readonly(tmp_path):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    source = trusted / "source.csv"
    source.write_text("a,b\n1,2\n")
    executor = Executor()
    result = await sandbox(tmp_path, executor, allowed_input_roots=(trusted,)).run(
        "print(INPUT_FILES['table'])", {"table": source}
    )
    assert result.status == "ok"
    create = next(args for args, _, _ in executor.calls if "create" in args)
    mounts = [create[i+1] for i, arg in enumerate(create) if arg == "--mount"]
    assert mounts == []
    imported = next(call for call in executor.calls if "exec" in call[0] and "--user" in call[0])
    assert imported[0][imported[0].index("--user")+1] == "0:0"
    assert json.loads(imported[1]) == [{"name":"table.csv","base64": base64.b64encode(source.read_bytes()).decode()}]
    executed = next(args for args, _, _ in executor.calls if "exec" in args and "--user" not in args)
    assert "table.csv" in executed[-1]
    assert not (result.artifacts)


@pytest.mark.parametrize("code", ["", " ", "x" * 8193, "😀" * 2049, "print(1)\x00", "\ud800", None])
async def test_code_byte_budget_before_executor(tmp_path, code):
    executor = Executor()
    result = await sandbox(tmp_path, executor).run(code)
    assert result.status == "error"
    assert result.reason == "invalid_code"
    assert executor.calls == []


@pytest.mark.parametrize("token", ["../escape", "/absolute", "a,b", "x" * 65, ""])
async def test_input_token_cannot_affect_host_paths_or_mounts(tmp_path, token):
    source = tmp_path / "source.txt"
    source.write_text("trusted")
    executor = Executor()
    result = await sandbox(tmp_path, executor, allowed_input_roots=(tmp_path,)).run("print(1)", {token: source})
    assert result.status == "error"
    assert executor.calls == []


async def test_input_without_trusted_root_is_rejected(tmp_path):
    source = tmp_path / "secret.txt"
    source.write_text("secret")
    executor = Executor()
    result = await sandbox(tmp_path, executor).run("print(1)", {"source": source})
    assert result.status == "error"
    assert result.reason == "invalid_input_files"
    assert executor.calls == []


@pytest.mark.parametrize("inputs", [[], "", (), ["token"]])
async def test_input_container_must_be_a_mapping(tmp_path, inputs):
    executor = Executor()
    result = await sandbox(tmp_path, executor).run("print(1)", inputs)
    assert result.reason == "invalid_input_files"
    assert executor.calls == []


async def test_timeout_stops_known_named_container_and_returns_no_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agent.python_sandbox._EXECUTION_SECONDS", 0.01)
    executor = Executor()
    executor.release = asyncio.Event()
    result = await sandbox(tmp_path, executor).run("while True: pass")
    assert result.status == "timeout"
    assert executor.cleanups >= 1
    assert result.artifacts == ()
    create = next(args for args, _, _ in executor.calls if "create" in args)
    name = create[create.index("--name") + 1]
    assert any(argv[-1] == name and "rm" in argv for argv, _, _ in executor.calls)


async def test_cancellation_waits_for_bounded_container_cleanup(tmp_path):
    executor = Executor()
    executor.release = asyncio.Event()
    task = asyncio.create_task(sandbox(tmp_path, executor).run("import time; time.sleep(1000)"))
    await executor.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert executor.cleanups >= 1


async def test_stdout_limit_kills_container_and_cannot_return_artifacts(tmp_path):
    executor = Executor(failure=SandboxOutputLimitError(b"a" * 16384))
    result = await sandbox(tmp_path, executor).run("print('a'*1000000)")
    assert result.status == "error"
    assert result.reason == "stdout_limit"
    assert len(result.stdout.encode()) <= 16384
    assert executor.cleanups >= 1
    assert not result.artifacts


@pytest.mark.parametrize("name", ["../escape.txt", "/escape.txt", "a/../../escape.txt", "C:\\escape.txt", "sub\\escape.txt"])
async def test_tar_path_escape_is_rejected_without_extracting_anything(tmp_path, name):
    executor = Executor(tar=archive([(name, b"bad")]))
    result = await sandbox(tmp_path, executor).run("print(42)")
    assert result.status == "error"
    assert result.reason == "invalid_artifact_archive"
    assert result.artifacts == ()
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE])
async def test_tar_links_and_special_files_are_never_extracted(tmp_path, kind):
    info = tarfile.TarInfo("link.txt")
    info.type = kind
    info.linkname = "/etc/passwd"
    executor = Executor(tar=archive(special=info))
    result = await sandbox(tmp_path, executor).run("print(42)")
    assert result.status == "error"
    assert not result.artifacts


async def test_more_than_eight_output_files_is_rejected_before_extraction(tmp_path):
    executor = Executor(tar=archive([(f"file-{i}.txt", b"x") for i in range(9)]))
    result = await sandbox(tmp_path, executor).run("print(42)")
    assert result.status == "error"
    assert result.artifacts == ()


async def test_artifact_binary_budget_is_enforced(tmp_path):
    executor = Executor(tar=archive([("big.bin", b"x" * (8 * 1024 * 1024 + 1))]))
    result = await sandbox(tmp_path, executor).run("print(42)")
    assert result.status == "error"
    assert not result.artifacts


async def test_invalid_image_is_not_registered_as_a_media_artifact(tmp_path):
    executor = Executor(tar=archive([("image.png", b"not pixels")]))
    result = await sandbox(tmp_path, executor).run("print(42)")
    assert result.status == "error"
    assert not result.artifacts


async def test_valid_image_and_bounded_other_binary_can_be_retained(tmp_path):
    image = io.BytesIO()
    Image.new("RGB", (16, 16), "pink").save(image, format="PNG")
    executor = Executor(tar=archive([("image.png", image.getvalue()), ("report.pdf", b"%PDF-test")]))
    result = await sandbox(tmp_path, executor).run("print(42)")
    assert result.status == "ok"
    assert len(result.artifacts) == 2


async def test_missing_docker_returns_unavailable_without_host_fallback(tmp_path):
    class Missing(Executor):
        async def execute(self, argv, **kwargs):
            self.calls.append((argv, b"", 0))
            raise FileNotFoundError("secret-path")

    executor = Missing()
    result = await sandbox(tmp_path, executor).run("print(42)")
    assert result.status == "unavailable"
    assert result.reason == "docker_unavailable"
    assert "secret" not in result.stdout
    assert len(executor.calls) == 1


async def test_image_declared_volumes_are_not_accepted(tmp_path):
    class Volumes(Executor):
        async def execute(self, argv, **kwargs):
            if "inspect" in argv:
                return DockerProcessResult(0, b'{"Env":[],"Volumes":{"/secret":{}}}')
            return await super().execute(argv, **kwargs)

    executor = Volumes()
    result = await sandbox(tmp_path, executor).run("print(42)")
    assert result.status == "unavailable"
    assert not any("create" in argv for argv, _, _ in executor.calls)


async def test_cleanup_failure_disables_further_code_execution(tmp_path):
    class FailedCleanup(Executor):
        async def execute(self, argv, **kwargs):
            if "rm" in argv:
                self.cleanups += 1
                return DockerProcessResult(1, b"", b"Cannot connect to the Docker daemon")
            if "container" in argv and "inspect" in argv:
                return DockerProcessResult(1, b"", b"Cannot connect to the Docker daemon")
            return await super().execute(argv, **kwargs)

    executor = FailedCleanup()
    tool = sandbox(tmp_path, executor)
    first = await tool.run("print(42)")
    assert first.status == "error"
    assert first.reason == "container_cleanup_failed"
    assert first.artifacts == ()
    called = len(executor.calls)
    second = await tool.run("print(99)")
    assert second.status == "unavailable"
    assert second.reason == "sandbox_cleanup_failed"
    assert len(executor.calls) == called


async def test_execution_error_includes_bounded_python_stderr(tmp_path):
    class FailedCode(Executor):
        async def execute(self, argv, **kwargs):
            if "exec" in argv and "--interactive" in argv:
                return DockerProcessResult(1, b"", b"ValueError: synthetic-input\n")
            return await super().execute(argv, **kwargs)

    executor = FailedCode()
    result = await sandbox(tmp_path, executor).run("raise ValueError('synthetic-input')")
    assert result.status == "error"
    assert result.reason == "python_execution_failed"
    assert "ValueError" in result.stdout
    assert result.artifacts == ()
    assert executor.cleanups >= 1


async def test_archive_capture_limit_rejects_an_oversized_export(tmp_path):
    class BigArchive(Executor):
        async def execute(self, argv, **kwargs):
            if "exec" in argv and "--interactive" not in argv:
                assert kwargs["maximum_output"] == 40 * 1024 * 1024
                raise SandboxOutputLimitError(b"synthetic-oversized-archive")
            return await super().execute(argv, **kwargs)

    executor = BigArchive()
    result = await sandbox(tmp_path, executor).run("print(42)")
    assert result.status == "error"
    assert result.artifacts == ()
    assert executor.cleanups >= 1


async def test_concurrent_runs_have_separate_container_names_and_artifact_dirs(tmp_path):
    executor = Executor(tar=archive([("answer.txt", b"42")]))
    tool = sandbox(tmp_path, executor)
    first, second = await asyncio.gather(tool.run("print(1)"), tool.run("print(2)"))
    names = [argv[argv.index("--name")+1] for argv, _, _ in executor.calls if "create" in argv]
    assert len(set(names)) == 2
    assert first.artifacts[0].parent != second.artifacts[0].parent
