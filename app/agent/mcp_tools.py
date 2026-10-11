"""Operator-configured MCP capabilities, exposed through the same agent loop.

Endpoints, schemas, scopes and effect classes come from local configuration,
never a chat/model request. No dynamic shell/module loading is involved.
"""
from __future__ import annotations

import asyncio
import json
import re
from copy import deepcopy
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from app.agent.tool_agent import AgentTool, ToolOutcome


class McpToolProvider:
    def __init__(self, config: dict, *, client=None, permission_check=None):
        self.config = deepcopy(config)
        self.permission_check = permission_check or (lambda: True)
        endpoint = urlsplit(config.get("url", ""))
        if (endpoint.username or endpoint.password or endpoint.fragment or endpoint.query
                or not endpoint.hostname or endpoint.scheme not in {"https", "http"}
                or endpoint.scheme == "http" and endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}):
            raise ValueError("MCP requires a fixed HTTPS or local HTTP endpoint without embedded credentials")
        if config.get("enabled") is not True or not isinstance(config.get("tools"), list) or len(config["tools"]) > 8:
            raise ValueError("invalid MCP tool manifest")
        self.headers = config.get("headers", {})
        if not isinstance(self.headers, dict) or any(not isinstance(k, str) or not isinstance(v, str) or "\n" in v or "\r" in v for k, v in self.headers.items()):
            raise ValueError("invalid MCP authentication headers")
        self._session_id = None
        self._ready = False
        self._lock = asyncio.Lock()
        for tool in config["tools"]:
            if (not isinstance(tool, dict) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", tool.get("name", ""))
                or tool.get("effect") not in {"read", "write"} or tool.get("parameters", {}).get("type") != "object"
                or not isinstance(tool.get("scopes"), list) or set(tool["scopes"]) - {"group", "qzone_publish", "qzone_comment", "profile"}):
                raise ValueError("invalid MCP capability")
        self.client = client or httpx.AsyncClient(timeout=20, follow_redirects=False, trust_env=False)
        self.owned_client = client is None

    async def _rpc(self, method, params, *, before_dispatch=None):
        return await asyncio.wait_for(self._rpc_inner(method, params, before_dispatch=before_dispatch), timeout=20)

    async def _rpc_inner(self, method, params, *, before_dispatch=None):
        headers = {**self.headers, "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        notification = method.startswith("notifications/")
        request = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notification:
            request["id"] = uuid4().hex
        if before_dispatch:
            before_dispatch()
        async with self.client.stream("POST", self.config["url"], headers=headers, json=request) as response:
            response.raise_for_status()
            parts, size = [], 0
            async for part in response.aiter_bytes():
                size += len(part)
                if size > 128 * 1024:
                    raise ValueError("MCP response exceeds budget")
                parts.append(part)
            if response.headers.get("Mcp-Session-Id"):
                self._session_id = response.headers["Mcp-Session-Id"]
            if notification:
                return {}
            raw = b"".join(parts).decode("utf-8")
            if "text/event-stream" in response.headers.get("Content-Type", ""):
                replies = [json.loads(line[5:].strip()) for line in raw.splitlines() if line.startswith("data:")]
                result = next((r for r in replies if isinstance(r, dict) and r.get("id") == request["id"]), None)
            else:
                result = json.loads(raw)
            if not isinstance(result, dict) or result.get("id") != request["id"] or "error" in result or not isinstance(result.get("result"), dict):
                raise ValueError("invalid MCP result")
            return result["result"]

    async def _initialize(self):
        async with self._lock:
            if self._ready:
                return
            result = await self._rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                           "clientInfo": {"name": "xiaomachi-agent", "version": "1"}})
            if result.get("protocolVersion") != "2025-06-18":
                raise ValueError("unsupported MCP protocol version")
            await self._rpc("notifications/initialized", {})
            self._ready = True

    def tools_for(self, observation, guard):
        tools = []
        for spec in self.config["tools"]:
            if observation.job.kind not in spec["scopes"]:
                continue
            async def call(args, spec=spec):
                guard.require_valid()
                if not self.permission_check():
                    return ToolOutcome({"error": "mcp_permission_revoked"}, "blocked")
                try:
                    await self._initialize()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    return ToolOutcome({"error": "mcp_initialize:" + type(exc).__name__}, "failed")
                guard.require_valid()
                if not self.permission_check():
                    return ToolOutcome({"error": "mcp_permission_revoked"}, "blocked")
                dispatched = False
                def dispatch():
                    nonlocal dispatched
                    guard.require_valid()
                    if not self.permission_check():
                        raise RuntimeError("MCP permission revoked")
                    guard.on_dispatch()
                    dispatched = True
                try:
                    result = await self._rpc("tools/call", {"name": spec.get("remote_name", spec["name"]), "arguments": args},
                                             before_dispatch=dispatch if spec["effect"] == "write" else None)
                    # A general MCP result doesn't prove a platform receipt;
                    # writes require an explicitly configured acknowledgement key.
                    if spec["effect"] == "write":
                        key = spec.get("acknowledgement_key")
                        receipt = result.get(key) if isinstance(key, str) else None
                        acknowledged = isinstance(receipt, str) and bool(receipt) and not result.get("isError")
                        return ToolOutcome({"acknowledged": acknowledged}, "sent" if acknowledged else "uncertain", receipt if acknowledged else None)
                    guard.require_valid()
                    return ToolOutcome({"mcp_result": result, "trust": "external_data"}, "failed" if result.get("isError") else "ok")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    return ToolOutcome({"error": type(exc).__name__}, "uncertain" if dispatched else "failed")
            tools.append(AgentTool(spec["name"], str(spec.get("description", "Operator-enabled MCP capability"))[:500], spec["parameters"], call, spec["effect"]))
        return tools

    async def close(self):
        if self.owned_client:
            await self.client.aclose()
