"""Typed, bounded OneBot get_forward_msg reads; no writes or automatic retry."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import re

from app.agent.social_images import allowed_media_url
from app.core.forward_content import cq_unescape, extract_forward_ids, valid_forward_id


MAX_NODES = 1000
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_TEXT = 320


class ForwardReadError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ForwardNode:
    text: str
    text_truncated: bool
    reported_author: str
    reported_time: str
    image_urls: tuple[str, ...]
    image_count: int
    forward_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ForwardResource:
    nodes: tuple[ForwardNode, ...]
    truncated: bool


def _label(value, maximum=80):
    return str(value)[:maximum] if type(value) in {str, int, float} else ""


def _node(row) -> ForwardNode:
    if not isinstance(row, dict):
        raise ForwardReadError("forward_response_malformed")
    if row.get("type") == "node":
        row = row.get("data")
        if not isinstance(row, dict):
            raise ForwardReadError("forward_response_malformed")
    message = row.get("message", row.get("content"))
    if not isinstance(message, (list, str)):
        raise ForwardReadError("forward_response_malformed")
    sender = row.get("sender")
    sender = sender if isinstance(sender, dict) else {}
    author = sender.get("nickname") or row.get("nickname") or row.get("name") or sender.get("user_id") or row.get("user_id") or row.get("uin")
    parts, images, image_count = [], [], 0
    truncated = False
    if isinstance(message, str):
        # Old CQ is quoted data. Never turn its @ into a current-group mention.
        parts.append(cq_unescape(re.sub(r"\[CQ:[^\]]{1,4096}\]", "[引用消息附件]", message[:65536])))
        truncated = len(message) > 65536
        for match in re.finditer(r"\[CQ:image,([^\]]{1,4096})\]", message[:65536]):
            fields = dict(part.split("=", 1) for part in match[1].split(",") if "=" in part)
            image_count += 1
            url = cq_unescape(fields.get("url", ""))
            if allowed_media_url(url) and len(images) < 2:
                images.append(url)
    else:
        truncated = len(message) > 256
        for item in message[:256]:
            if not isinstance(item, dict) or not isinstance(item.get("data"), dict):
                continue
            kind, data = item.get("type"), item["data"]
            if kind == "text" and isinstance(data.get("text"), str):
                parts.append(data["text"][:65536])
                truncated |= len(data["text"]) > 65536
            elif kind == "at":
                parts.append("[引用的@" + _label(data.get("qq"), 24) + "]")
            elif kind == "image":
                image_count += 1
                url = data.get("url")
                if allowed_media_url(url) and len(images) < 2:
                    images.append(url)
            elif kind in {"forward", "json", "xml"} and extract_forward_ids([item]):
                parts.append("[嵌套合并转发]")
    text = "".join(parts).strip()
    return ForwardNode(text[:MAX_TEXT], truncated or len(text) > MAX_TEXT, _label(author),
        _label(row.get("time"), 32), tuple(dict.fromkeys(images)), image_count, extract_forward_ids(message))


class ForwardMessageAdapter:
    def __init__(self, gateway, *, timeout_seconds=20):
        self.gateway, self.timeout_seconds = gateway, timeout_seconds

    async def read(self, resource_id: str) -> ForwardResource:
        if not valid_forward_id(resource_id):
            raise ValueError("forward_handle_invalid")
        if hasattr(self.gateway, "websocket") and self.gateway.websocket is None:
            raise ForwardReadError("forward_gateway_unavailable")
        try:
            response = await asyncio.wait_for(self.gateway.call_api("get_forward_msg", {"id": resource_id}), self.timeout_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            raise ForwardReadError("forward_gateway_unavailable") from None
        if (not isinstance(response, dict) or response.get("status") != "ok"
                or type(response.get("retcode")) is not int or response["retcode"] != 0):
            raise ForwardReadError("forward_gateway_rejected")
        data = response.get("data")
        rows = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise ForwardReadError("forward_response_malformed")
        try:
            if len(json.dumps(data, ensure_ascii=False).encode("utf8")) > MAX_RESPONSE_BYTES:
                raise ForwardReadError("forward_response_budget_exhausted")
        except (ValueError, TypeError, RecursionError):
            raise ForwardReadError("forward_response_malformed") from None
        return ForwardResource(tuple(_node(row) for row in rows[:MAX_NODES]), len(rows) > MAX_NODES)
