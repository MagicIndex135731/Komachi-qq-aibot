"""Authenticated, bounded HTTP access to the optional Docker computation tool.

The bot gets only this protocol. Docker CLI/socket access stays in a separately
configured service; requests contain code and approved artifact bytes, never
host paths, shell commands, Docker options, or model-controlled resource limits.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import io
import json
import os
import re
import tarfile
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from app.agent.python_sandbox import (
    PythonSandbox,
    SandboxResult,
    _CODE_BYTES,
    _FILE_BYTES,
    _MAX_FILES,
    _STDOUT_BYTES,
    _TOKEN,
    _TOTAL_FILE_BYTES,
    _verify_image,
)


_HEADER_BYTES = 8192
_REQUEST_BYTES = 24 * 1024 * 1024
_RESPONSE_BYTES = 32 * 1024 * 1024
_REQUEST_SECONDS = 30
_SERVER_SECONDS = 25
_STATUSES = frozenset({"ok", "error", "timeout", "unavailable"})
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "xiaomachi-sandbox"})


class HttpPythonSandbox:
    def __init__(self, url: str, *, token: str, output_root: Path | str,
                 allowed_input_roots=(), client: httpx.AsyncClient | None = None) -> None:
        _require_token(token)
        parsed = urlsplit(url)
        if (parsed.scheme not in {"https", "http"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"}
                or parsed.scheme == "http" and parsed.hostname not in _LOCAL_HOSTS):
            raise ValueError("Sandbox gateway URL must be HTTPS or an approved internal HTTP host")
        self.url = url.rstrip("/")
        self._token = token
        self.output_root = Path(output_root).resolve()
        self._artifacts = PythonSandbox(output_root=self.output_root, allowed_input_roots=allowed_input_roots)
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(_REQUEST_SECONDS), trust_env=False, follow_redirects=False,
        )

    async def available(self) -> bool:
        try:
            response = await self._request("GET", "/health")
            return response is not None and response.get("status") == "ok" and response.get("available") is True
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, ValueError, asyncio.TimeoutError):
            return False

    async def run(self, code: str, input_files: Mapping[str, Path] | None = None) -> SandboxResult:
        if not _valid_code(code):
            return SandboxResult("", (), "error", "invalid_code")
        try:
            files = self._artifacts._capture_inputs({} if input_files is None else input_files)
            body = {"code": code, "inputs": [
                {"token": token, "name": name, "base64": base64.b64encode(contents).decode("ascii")}
                for token, name, contents in files
            ]}
        except (ValueError, OSError):
            return SandboxResult("", (), "error", "invalid_input_files")
        try:
            response = await self._request("POST", "/run", body)
            if response is None:
                return SandboxResult("", (), "unavailable", "sandbox_http_unavailable")
            return self._decode_result(response)
        except asyncio.CancelledError:
            # Closing the streamed request signals the server's disconnect
            # watcher, which cancels PythonSandbox and awaits Docker cleanup.
            raise
        except (httpx.HTTPError, asyncio.TimeoutError):
            return SandboxResult("", (), "unavailable", "sandbox_http_unavailable")
        except (ValueError, OSError, tarfile.TarError):
            return SandboxResult("", (), "error", "invalid_sandbox_response")

    async def _request(self, method: str, path: str, body: dict | None = None) -> dict | None:
        async with asyncio.timeout(_REQUEST_SECONDS):
            async with self.client.stream(
                method, self.url + path,
                headers={"Authorization": "Bearer " + self._token}, json=body,
                follow_redirects=False,
            ) as response:
                if response.status_code not in {200, 503}:
                    return None
                captured = bytearray()
                async for chunk in response.aiter_bytes():
                    captured.extend(chunk)
                    if len(captured) > _RESPONSE_BYTES:
                        raise ValueError("Sandbox HTTP response exceeded its byte budget")
                parsed = json.loads(captured)
                if not isinstance(parsed, dict):
                    raise ValueError("Sandbox HTTP response is not an object")
                return parsed

    def _decode_result(self, response: dict) -> SandboxResult:
        if set(response) != {"status", "reason", "stdout", "artifacts"}:
            raise ValueError("Sandbox result fields are invalid")
        status, reason, stdout = response["status"], response["reason"], response["stdout"]
        if (not isinstance(status, str) or status not in _STATUSES
                or not isinstance(reason, str) or not re.fullmatch(r"[a-z0-9_]{0,128}", reason)
                or not isinstance(stdout, str) or len(stdout.encode("utf-8")) > _STDOUT_BYTES):
            raise ValueError("Sandbox result metadata is invalid")
        items = response["artifacts"]
        if not isinstance(items, list) or len(items) > _MAX_FILES or status != "ok" and items:
            raise ValueError("Sandbox result artifacts are invalid")
        total = 0
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for item in items:
                if not isinstance(item, dict) or set(item) != {"name", "base64", "sha256"}:
                    raise ValueError("Sandbox artifact shape is invalid")
                name = item["name"]
                if not isinstance(name, str) or not 1 <= len(name) <= 512:
                    raise ValueError("Sandbox artifact name is invalid")
                contents = _base64_bytes(item["base64"])
                total += len(contents)
                if total > _TOTAL_FILE_BYTES or hashlib.sha256(contents).hexdigest() != item["sha256"]:
                    raise ValueError("Sandbox artifact hash or byte budget is invalid")
                header = tarfile.TarInfo(name)
                header.size = len(contents)
                archive.addfile(header, io.BytesIO(contents))
        if not items:
            return SandboxResult(stdout, (), status, reason)
        directory = self.output_root / uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        artifacts = self._artifacts._store_artifacts(buffer.getvalue(), directory)
        return SandboxResult(stdout, artifacts, status, reason)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()


class SandboxServer:
    """Small HTTP/1.1 service with one executing request and no queued code."""

    def __init__(self, sandbox, *, token: str, host: str = "127.0.0.1", port: int = 8217) -> None:
        _require_token(token)
        if host not in {"127.0.0.1", "::1", "localhost", "0.0.0.0"} or type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("Sandbox service bind configuration is invalid")
        self.sandbox = sandbox
        self._token = token
        self.host = host
        self.port = port
        self.input_root = Path(sandbox.output_root).resolve().parent / "sandbox-rpc-inputs"
        self._server = None
        self._connections = set()
        self._writers = set()
        self._busy = False

    async def start(self) -> None:
        self.input_root.mkdir(parents=True, exist_ok=True)
        self._server = await asyncio.start_server(self._handle, self.host, self.port, limit=_HEADER_BYTES)
        self.port = self._server.sockets[0].getsockname()[1]

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
        tasks = tuple(self._connections)
        for task in tasks:
            task.cancel()
        for writer in tuple(self._writers):
            writer.close()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._server is not None:
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if len(self._connections) >= 4:
            writer.close()
            return
        self._connections.add(task)
        self._writers.add(writer)
        code_task = None
        disconnect_task = None
        input_directory = None
        claimed = False
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            method, path, headers = _headers(head)
            if not hmac.compare_digest(headers.get("authorization", ""), "Bearer " + self._token):
                await _respond(writer, 401, _message("unavailable", "authentication_failed"))
                return
            if method == "GET" and path == "/health":
                if headers.get("content-length", "0") != "0":
                    raise ValueError("Health request body is invalid")
                available = await asyncio.wait_for(self.sandbox.available(), 5)
                await _respond(writer, 200, {"status": "ok", "available": available})
                return
            if method != "POST" or path != "/run":
                await _respond(writer, 404, _message("unavailable", "unknown_route"))
                return
            if self._busy:
                await _respond(writer, 503, _message("unavailable", "sandbox_busy"))
                return
            length = headers.get("content-length", "")
            if not length.isascii() or not length.isdecimal() or not 1 <= int(length) <= _REQUEST_BYTES:
                raise ValueError("Request size is invalid")
            self._busy = claimed = True
            body = await asyncio.wait_for(reader.readexactly(int(length)), 10)
            parsed = json.loads(body)
            code, inputs = _request_body(parsed)
            input_directory = self.input_root / uuid4().hex
            input_directory.mkdir(mode=0o700)
            sources = {}
            for token, name, contents in inputs:
                file = input_directory / name
                file.write_bytes(contents)
                sources[token] = file
            code_task = asyncio.create_task(self.sandbox.run(code, sources))
            disconnect_task = asyncio.create_task(reader.read(1))
            done, _ = await asyncio.wait(
                {code_task, disconnect_task}, timeout=_SERVER_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                code_task.cancel()
                await asyncio.gather(code_task, return_exceptions=True)
                return
            if code_task not in done:
                code_task.cancel()
                await asyncio.gather(code_task, return_exceptions=True)
                await _respond(writer, 200, _message("timeout", "gateway_execution_timeout"))
                return
            result = code_task.result()
            await _respond(writer, 200, _encode_result(result, Path(self.sandbox.output_root).resolve()))
        except asyncio.CancelledError:
            raise
        except (ValueError, UnicodeError, asyncio.IncompleteReadError, asyncio.LimitOverrunError,
                asyncio.TimeoutError, OSError):
            try:
                await _respond(writer, 400, _message("error", "invalid_sandbox_request"))
            except (OSError, asyncio.TimeoutError):
                pass
        finally:
            for pending in (code_task, disconnect_task):
                if pending is not None and not pending.done():
                    pending.cancel()
            pending_tasks = [pending for pending in (code_task, disconnect_task) if pending is not None]
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            if input_directory is not None:
                for child in input_directory.iterdir():
                    child.unlink()
                input_directory.rmdir()
            if claimed:
                self._busy = False
            self._connections.discard(task)
            self._writers.discard(writer)
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), 1)
            except (OSError, asyncio.TimeoutError):
                pass


def _headers(raw: bytes) -> tuple[str, str, dict[str, str]]:
    if len(raw) > _HEADER_BYTES:
        raise ValueError("Sandbox headers exceed their byte budget")
    lines = raw.decode("ascii").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or parts[2] != "HTTP/1.1":
        raise ValueError("Sandbox request line is invalid")
    headers = {}
    for line in lines[1:]:
        if not line:
            continue
        if ":" not in line:
            raise ValueError("Sandbox header is invalid")
        name, value = line.split(":", 1)
        name = name.lower()
        if not re.fullmatch(r"[a-z0-9-]+", name) or name in headers:
            raise ValueError("Sandbox header shape is invalid")
        headers[name] = value.strip()
    if "transfer-encoding" in headers:
        raise ValueError("Chunked sandbox requests are unsupported")
    return parts[0], parts[1], headers


def _valid_code(code) -> bool:
    try:
        return isinstance(code, str) and bool(code.strip()) and "\x00" not in code and len(code.encode("utf-8")) <= _CODE_BYTES
    except UnicodeError:
        return False


def _request_body(body) -> tuple[str, tuple[tuple[str, str, bytes], ...]]:
    if not isinstance(body, dict) or set(body) - {"code", "inputs"} or not _valid_code(body.get("code")):
        raise ValueError("Sandbox request body is invalid")
    inputs = body.get("inputs", [])
    if not isinstance(inputs, list) or len(inputs) > _MAX_FILES:
        raise ValueError("Sandbox input list is invalid")
    result = []
    tokens = set()
    total = 0
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {"token", "name", "base64"}:
            raise ValueError("Sandbox input fields are invalid")
        token, name = item["token"], item["name"]
        if (not isinstance(token, str) or not _TOKEN.fullmatch(token) or token in tokens
                or not isinstance(name, str) or not re.fullmatch(re.escape(token) + r"\.[a-z0-9]{1,10}", name)):
            raise ValueError("Sandbox input provenance token is invalid")
        contents = _base64_bytes(item["base64"])
        total += len(contents)
        if total > _TOTAL_FILE_BYTES:
            raise ValueError("Sandbox input total byte budget exceeded")
        tokens.add(token)
        result.append((token, name, contents))
    return body["code"], tuple(result)


def _base64_bytes(value) -> bytes:
    if not isinstance(value, str) or len(value) > 4 * ((_FILE_BYTES + 2) // 3):
        raise ValueError("Sandbox artifact encoded size is invalid")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("Sandbox artifact encoding is invalid") from None
    if len(raw) > _FILE_BYTES:
        raise ValueError("Sandbox artifact byte budget exceeded")
    return raw


def _encode_result(result: SandboxResult, root: Path) -> dict:
    if (not isinstance(result, SandboxResult) or result.status not in _STATUSES
            or not re.fullmatch(r"[a-z0-9_]{0,128}", result.reason)
            or len(result.stdout.encode("utf-8")) > _STDOUT_BYTES
            or len(result.artifacts) > _MAX_FILES or result.status != "ok" and result.artifacts):
        raise ValueError("Sandbox produced invalid result metadata")
    artifacts = []
    total = 0
    names = set()
    for path in result.artifacts:
        actual = path.resolve()
        if not actual.is_relative_to(root) or not actual.is_file() or path.is_symlink():
            raise ValueError("Sandbox output source is invalid")
        name = actual.relative_to(root).as_posix()
        if name in names:
            raise ValueError("Sandbox output names conflict")
        with actual.open("rb") as stream:
            contents = stream.read(_FILE_BYTES + 1)
        total += len(contents)
        if len(contents) > _FILE_BYTES or total > _TOTAL_FILE_BYTES:
            raise ValueError("Sandbox output exceeded its byte budget")
        _verify_image(name, contents)
        names.add(name)
        artifacts.append({"name": name, "base64": base64.b64encode(contents).decode("ascii"),
                          "sha256": hashlib.sha256(contents).hexdigest()})
    return {"status": result.status, "reason": result.reason, "stdout": result.stdout, "artifacts": artifacts}


def _message(status: str, reason: str) -> dict:
    return {"status": status, "reason": reason, "stdout": "", "artifacts": []}


async def _respond(writer, status: int, body: dict) -> None:
    data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(data) > _RESPONSE_BYTES:
        raise ValueError("Sandbox response exceeded its byte budget")
    line = "OK" if status == 200 else "Unavailable" if status == 503 else "Rejected"
    writer.write(f"HTTP/1.1 {status} {line}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode("ascii"))
    writer.write(data)
    await asyncio.wait_for(writer.drain(), 5)


def _require_token(value) -> None:
    if (not isinstance(value, str) or not 1 <= len(value) <= 256
            or not value.isascii() or any(ord(char) <= 32 or ord(char) >= 127 for char in value)):
        raise ValueError("Sandbox gateway requires a bounded nonempty ASCII authentication token")


async def _main() -> None:
    token = os.environ.get("XIAOMACHI_SANDBOX_TOKEN", "")
    root = Path(os.environ.get("XIAOMACHI_SANDBOX_DATA_DIR", "/data/sandbox")).resolve()
    sandbox = PythonSandbox(output_root=root, allowed_input_roots=(root.parent / "sandbox-rpc-inputs",), enabled=True)
    server = SandboxServer(sandbox, token=token,
                           host=os.environ.get("XIAOMACHI_SANDBOX_BIND", "127.0.0.1"),
                           port=int(os.environ.get("XIAOMACHI_SANDBOX_PORT", "8217")))
    try:
        await server.serve_forever()
    finally:
        await server.close()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
