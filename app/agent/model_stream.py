"""Finite model-stream authority; keepalives cannot renew a request budget."""
from __future__ import annotations

import socket
import threading
import time


class ModelStreamBudget:
    def __init__(self, bridge, handler):
        self.bridge, self.handler = bridge, handler
        self.started = self.last_progress = time.monotonic()
        self.deadline = min(bridge.deadline, self.started + bridge.owner.model_request_timeout_seconds)
        self.progress_seconds = bridge.owner.model_progress_timeout_seconds
        self.first_seconds = bridge.owner.model_first_output_timeout_seconds
        self.first_progress_ms = None
        self.progress_events = self.sse_events = 0
        self.response = None
        self.reason = ""
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._watch, name="model-stream-budget", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.closed.set()
        self.thread.join(timeout=0.5)

    def check(self, *, authority=True):
        if self.reason:
            raise RuntimeError(self.reason)
        if self.bridge.cancelled.is_set():
            raise RuntimeError("Harness action authority ended")
        if authority:
            self.bridge.require_valid()
        now = time.monotonic()
        reason = ("model_request_timeout" if now >= self.deadline else
                  "model_first_output_timeout" if not self.progress_events and now - self.started >= self.first_seconds else
                  "model_progress_timeout" if self.progress_events and now - self.last_progress >= self.progress_seconds else "")
        if reason:
            self._expire(reason)
            raise RuntimeError(reason)

    def observe(self, chunk, protocol):
        self.sse_events += 1
        if protocol == "openai-responses":
            progress = (chunk.get("type") in {
                "response.output_text.delta", "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta", "response.function_call_arguments.delta",
            } and isinstance(chunk.get("delta"), str) and bool(chunk["delta"]))
        else:
            progress = False
            for choice in chunk.get("choices", []) if isinstance(chunk.get("choices"), list) else []:
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                if not isinstance(delta, dict):
                    continue
                progress |= any(isinstance(delta.get(key), str) and bool(delta[key])
                                for key in ("content", "reasoning_content", "reasoning"))
                for call in delta.get("tool_calls", []) if isinstance(delta.get("tool_calls"), list) else []:
                    function = call.get("function", {}) if isinstance(call, dict) else {}
                    if isinstance(function, dict):
                        progress |= any(isinstance(function.get(key), str) and bool(function[key])
                                        for key in ("name", "arguments"))
        if progress:
            self.last_progress = time.monotonic()
            if not self.progress_events:
                self.first_progress_ms = round((self.last_progress - self.started) * 1000, 3)
            self.progress_events += 1

    @staticmethod
    def _shutdown(connection):
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except (OSError, ValueError):
                pass

    def interrupt(self):
        # shutdown wakes a synchronous socket recv; close alone may leave it
        # waiting for the HTTP read timeout. Do not close a shared client/pool.
        response = self.response
        stream = response.extensions.get("network_stream") if response is not None else None
        if stream is not None and callable(getattr(stream, "get_extra_info", None)):
            self._shutdown(stream.get_extra_info("socket"))
        self._shutdown(getattr(self.handler, "connection", None))

    def _expire(self, reason):
        if not self.reason:
            self.reason = reason
            self.bridge.stop_reason = self.bridge.stop_reason or reason
            self.bridge.revoke()
        self.interrupt()

    def _watch(self):
        while not self.closed.wait(0.02):
            if self.bridge.cancelled.is_set():
                self.interrupt()
                return
            try:
                self.check(authority=False)
            except RuntimeError:
                self.interrupt()
                return
