from __future__ import annotations

import asyncio
import base64
import hashlib
from pathlib import Path

import httpx
import pytest

from app.agent.python_sandbox import SandboxResult
from app.agent.sandbox_gateway import HttpPythonSandbox, SandboxServer


TOKEN = "synthetic-sandbox-token"


class FakeSandbox:
    def __init__(self, root):
        self.output_root = root
        self.calls = []
        self.started = asyncio.Event()
        self.release = None
        self.cancelled = False

    async def available(self):
        return True

    async def run(self, code, input_files=None):
        self.calls.append((code, {token: path.read_bytes() for token, path in (input_files or {}).items()}))
        self.started.set()
        try:
            if self.release is not None:
                await self.release.wait()
            self.output_root.mkdir(parents=True, exist_ok=True)
            artifact = self.output_root / "answer.json"
            artifact.write_bytes(b'{"answer":42}')
            return SandboxResult("42\n", (artifact,), "ok")
        except asyncio.CancelledError:
            self.cancelled = True
            raise


async def fixture_gateway(tmp_path):
    fake = FakeSandbox(tmp_path / "server-artifacts")
    server = SandboxServer(fake, token=TOKEN, host="127.0.0.1", port=0)
    await server.start()
    url = f"http://127.0.0.1:{server.port}"
    return fake, server, url


async def test_health_and_computation_round_trip_use_local_artifact_copy(tmp_path):
    fake, server, url = await fixture_gateway(tmp_path)
    tool = HttpPythonSandbox(url, token=TOKEN, output_root=tmp_path / "client-artifacts")
    try:
        assert await tool.available()
        result = await tool.run("print(42)")
        assert result.status == "ok"
        assert result.stdout == "42\n"
        assert result.artifacts[0].read_bytes() == b'{"answer":42}'
        assert result.artifacts[0].is_relative_to(tmp_path / "client-artifacts")
        assert not result.artifacts[0].is_relative_to(fake.output_root)
        assert fake.calls == [("print(42)", {})]
    finally:
        await tool.aclose()
        await server.close()


async def test_trusted_client_inputs_are_bytes_not_server_host_paths(tmp_path):
    source_root = tmp_path / "trusted"
    source_root.mkdir()
    source = source_root / "values.csv"
    source.write_bytes(b"a,b\n1,2\n")
    fake, server, url = await fixture_gateway(tmp_path)
    tool = HttpPythonSandbox(url, token=TOKEN, output_root=tmp_path / "client", allowed_input_roots=(source_root,))
    try:
        result = await tool.run("print(INPUT_FILES)", {"values": source})
        assert result.status == "ok"
        assert fake.calls[0][1] == {"values": source.read_bytes()}
        assert not list(server.input_root.iterdir())
    finally:
        await tool.aclose()
        await server.close()


@pytest.mark.parametrize("url", ["http://example.com", "http://169.254.169.254", "http://localhost/?token=x", "http://user:pass@localhost", "file:///tmp/a", "http://localhost/path"])
def test_plain_http_and_url_surface_is_limited(tmp_path, url):
    with pytest.raises(ValueError):
        HttpPythonSandbox(url, token=TOKEN, output_root=tmp_path)


@pytest.mark.parametrize("token", ["", "bad\r\nheader", "x" * 257])
def test_client_and_server_require_safe_nonempty_token(tmp_path, token):
    with pytest.raises(ValueError):
        HttpPythonSandbox("http://localhost", token=token, output_root=tmp_path)
    with pytest.raises(ValueError):
        SandboxServer(FakeSandbox(tmp_path), token=token)


async def test_wrong_token_never_executes_code(tmp_path):
    fake, server, url = await fixture_gateway(tmp_path)
    tool = HttpPythonSandbox(url, token="wrong-synthetic-token", output_root=tmp_path / "client")
    try:
        assert not await tool.available()
        result = await tool.run("print(42)")
        assert result.status == "unavailable"
        assert fake.calls == []
    finally:
        await tool.aclose()
        await server.close()


async def test_concurrent_request_is_unavailable_and_does_not_queue_code(tmp_path):
    fake, server, url = await fixture_gateway(tmp_path)
    fake.release = asyncio.Event()
    tool = HttpPythonSandbox(url, token=TOKEN, output_root=tmp_path / "client")
    first = asyncio.create_task(tool.run("print(1)"))
    try:
        await fake.started.wait()
        second = await tool.run("print(2)")
        assert second.status == "unavailable"
        assert second.reason == "sandbox_busy"
        assert len(fake.calls) == 1
        fake.release.set()
        assert (await first).status == "ok"
    finally:
        if not first.done():
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
        await tool.aclose()
        await server.close()


async def test_request_cancellation_closes_connection_and_cancels_server_code(tmp_path):
    fake, server, url = await fixture_gateway(tmp_path)
    fake.release = asyncio.Event()
    tool = HttpPythonSandbox(url, token=TOKEN, output_root=tmp_path / "client")
    request = asyncio.create_task(tool.run("print(42)"))
    try:
        await fake.started.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        # The code task observes cancellation before the connection handler's
        # finally block finishes cleaning its input directory. Wait for both
        # observable outcomes, bounded by the same short cancellation window.
        for _ in range(200):
            if fake.cancelled and not list(server.input_root.iterdir()):
                break
            await asyncio.sleep(.01)
        assert fake.cancelled
        assert not list(server.input_root.iterdir())
        assert not list((tmp_path / "client").glob("**/answer.json"))
    finally:
        await tool.aclose()
        await server.close()


async def test_server_rejects_host_paths_extra_fields_and_bad_code(tmp_path):
    fake, server, url = await fixture_gateway(tmp_path)
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            for body in [
                {"code": "print(42)", "host_path": "/etc/passwd"},
                {"code": "x" * 8193},
                {"code": "print(42)", "inputs": [{"token": "../../escape", "name": "secret.txt", "base64": ""}]},
                {"code": "print(42)", "inputs": [{"token": "file", "name": "file.txt", "path": "/etc/passwd"}]},
            ]:
                response = await client.post(url + "/run", headers={"Authorization": "Bearer " + TOKEN}, json=body)
                assert response.status_code == 400
        assert fake.calls == []
    finally:
        await server.close()


async def test_server_closure_cancels_active_code_and_closes_connections(tmp_path):
    fake, server, url = await fixture_gateway(tmp_path)
    fake.release = asyncio.Event()
    tool = HttpPythonSandbox(url, token=TOKEN, output_root=tmp_path / "client")
    request = asyncio.create_task(tool.run("print(42)"))
    await fake.started.wait()
    await server.close()
    result = await request
    assert result.status == "unavailable"
    assert fake.cancelled
    await tool.aclose()


async def test_redirect_does_not_forward_token_to_other_service(tmp_path):
    seen = []
    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(307, headers={"Location": "http://example.com/run"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = HttpPythonSandbox("http://localhost", token=TOKEN, output_root=tmp_path, client=client)
    try:
        result = await tool.run("print(42)")
        assert result.status == "unavailable"
        assert seen == ["http://localhost/run"]
    finally:
        await client.aclose()


@pytest.mark.parametrize("artifact", [
    {"name": "../escape.png", "base64": base64.b64encode(b"x").decode(), "sha256": hashlib.sha256(b"x").hexdigest()},
    {"name": "image.png", "base64": base64.b64encode(b"not image").decode(), "sha256": hashlib.sha256(b"not image").hexdigest()},
    {"name": "result.txt", "base64": base64.b64encode(b"x").decode(), "sha256": "0" * 64},
])
async def test_client_validates_artifact_path_hash_and_actual_pixels(tmp_path, artifact):
    def handler(request):
        return httpx.Response(200, json={"status": "ok", "reason": "", "stdout": "42", "artifacts": [artifact]})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = HttpPythonSandbox("http://localhost", token=TOKEN, output_root=tmp_path, client=client)
    try:
        result = await tool.run("print(42)")
        assert result.status == "error"
        assert result.reason == "invalid_sandbox_response"
        assert result.artifacts == ()
    finally:
        await client.aclose()
