"""Bounded decoder for named records carried inside one QQ long-message bundle.

Only SsoRecvLongMsg is constructed here. No caller can select an SSO command.
Wire tags follow SnowLuma's longmsg/message/element protocol definitions.
"""
from __future__ import annotations

import copy
from urllib.parse import parse_qs, quote, urlsplit
import zlib

from app.agent.social_images import allowed_media_url
from app.core.forward_content import valid_forward_id

READ_COMMAND = "trpc.group.long_msg_interface.MsgService.SsoRecvLongMsg"
MAX_BYTES = 2 * 1024 * 1024
MAX_ACTIONS = 32
MAX_FIELDS = 10000
MAX_BUNDLE_NODES = 4096


def _varint(value: int) -> bytes:
    output = bytearray()
    while value > 127:
        output.append((value & 127) | 128)
        value >>= 7
    return bytes(output) + bytes([value])


def _blob(field: int, value: bytes) -> bytes:
    return _varint(field * 8 + 2) + _varint(len(value)) + value


def read_request(resource_id: str) -> dict:
    if not valid_forward_id(resource_id):
        raise ValueError("forward_handle_invalid")
    info = _blob(2, resource_id.encode()) + b"\x18\x01"
    settings = b"\x08\x02\x10\x00\x18\x00\x20\x00"
    return {"cmd": READ_COMMAND, "data": (_blob(1, info) + _blob(15, settings)).hex(), "rsp": True}


def _fields(body: bytes) -> dict[int, list[bytes | int]]:
    if not isinstance(body, bytes) or len(body) > MAX_BYTES:
        raise ValueError("forward_bundle_budget_exhausted")
    def integer(position):
        value = 0
        for shift in range(0, 70, 7):
            if position >= len(body):
                raise ValueError("forward_bundle_malformed")
            byte = body[position]
            position += 1
            value |= (byte & 127) << shift
            if not byte & 128:
                return value, position
        raise ValueError("forward_bundle_malformed")
    result, position, count = {}, 0, 0
    while position < len(body):
        key, position = integer(position)
        field, wire = key >> 3, key & 7
        count += 1
        if not field or count > MAX_FIELDS:
            raise ValueError("forward_bundle_malformed")
        if wire == 0:
            value, position = integer(position)
        elif wire in (1, 2, 5):
            if wire == 2:
                length, position = integer(position)
            else:
                length = 8 if wire == 1 else 4
            end = position + length
            if end > len(body):
                raise ValueError("forward_bundle_malformed")
            value, position = body[position:end], end
        else:
            raise ValueError("forward_bundle_malformed")
        result.setdefault(field, []).append(value)
    return result


def _first(fields, field, default=b""):
    return fields.get(field, [default])[0]


def _message(fields, field):
    return _fields(_first(fields, field))


def _text(value):
    if not isinstance(value, bytes):
        raise ValueError("forward_bundle_malformed")
    return value.decode("utf8", errors="replace")


def _inflate(value: bytes, maximum: int, *, gzip=False) -> bytes:
    if not isinstance(value, bytes) or len(value) > MAX_BYTES:
        raise ValueError("forward_bundle_malformed")
    try:
        decoder = zlib.decompressobj(31 if gzip else 15)
        result = decoder.decompress(value, maximum + 1)
        if len(result) > maximum or decoder.unconsumed_tail:
            raise ValueError("forward_bundle_budget_exhausted")
        if not decoder.eof or decoder.unused_data:
            raise ValueError("forward_bundle_malformed")
        return result
    except zlib.error:
        raise ValueError("forward_bundle_malformed") from None


def _card(value: bytes, budget: list[int]) -> str:
    if not isinstance(value, bytes) or not value:
        raise ValueError("forward_bundle_malformed")
    maximum = min(65536, budget[0])
    if value[0] == 1:
        content = _inflate(value[1:], maximum)
    elif value[0] == 0:
        content = value[1:]
    else:
        raise ValueError("forward_bundle_malformed")
    if len(content) > maximum:
        raise ValueError("forward_bundle_budget_exhausted")
    budget[0] -= len(content)
    return _text(content)


def _image(element):
    common = _message(element, 53)
    if _first(common, 1, 0) != 48 or _first(common, 3, 0) not in (10, 20):
        return None
    info = _message(common, 2)
    body = _message(info, 1)
    picture = _message(body, 2)
    domain = _text(_first(picture, 3)) or "multimedia.nt.qq.com.cn"
    path = _text(_first(picture, 1))
    parameter = _text(_first(_message(picture, 2), 1))
    url = f"https://{domain}{path}{parameter}" if path.startswith("/") else ""
    if not allowed_media_url(url):
        url = ""
    return {"type": "image", "data": {"url": url}}


def _row(payload: bytes, budget: list[int]) -> dict:
    row = _fields(payload)
    head, content = _message(row, 1), _message(row, 2)
    group, forwarded = _message(head, 8), _message(head, 7)
    author = _text(_first(group, 4)) or _text(_first(group, 2)) or _text(_first(forwarded, 6))
    rich = _message(_message(row, 3), 1)
    segments = []
    for raw in rich.get(2, [])[:256]:
        element = _fields(raw)
        if 1 in element:
            segments.append({"type": "text", "data": {"text": _text(_first(_message(element, 1), 1))}})
        elif 12 in element or 51 in element:
            kind, tag = ("xml", 12) if 12 in element else ("json", 51)
            segments.append({"type": kind, "data": {"data": _card(_first(_message(element, tag), 1), budget)}})
        elif 53 in element and (image := _image(element)) is not None:
            segments.append(image)
        elif 4 in element or 8 in element:
            tag, url_tag = (4, 15) if 4 in element else (8, 16)
            url = _text(_first(_message(element, tag), url_tag))
            segments.append({"type": "image", "data": {"url": url if allowed_media_url(url) else ""}})
        elif 2 in element:
            segments.append({"type": "text", "data": {"text": "[引用表情]"}})
        elif any(tag in element for tag in (3, 5, 6, 13, 19, 45, 53)):
            segments.append({"type": "text", "data": {"text": "[未展开的引用附件]"}})
    if len(rich.get(2, [])) > 256 or 3 in rich or 4 in rich:
        segments.append({"type": "text", "data": {"text": "[未展开的引用附件]"}})
    return {"sender": {"nickname": author, "user_id": _first(head, 1, 0)},
            "time": _first(content, 6, 0), "message": segments}


def decode_packet(packet: bytes) -> dict[str, list[dict]]:
    response = _fields(packet)
    compressed = _first(_message(response, 1), 4)
    if not compressed:
        raise ValueError("forward_bundle_malformed")
    actions = _fields(_inflate(compressed, MAX_BYTES, gzip=True)).get(2, [])
    if len(actions) > MAX_ACTIONS:
        raise ValueError("forward_bundle_budget_exhausted")
    result, budget, node_count = {}, [MAX_BYTES], 0
    for raw in actions:
        action = _fields(raw)
        name = _text(_first(action, 1))
        if not valid_forward_id(name) or name in result:
            raise ValueError("forward_bundle_malformed")
        # One extra row preserves the existing adapter's truncation flag.
        rows = _message(action, 2).get(1, [])
        node_count += len(rows)
        if node_count > MAX_BUNDLE_NODES:
            raise ValueError("forward_bundle_budget_exhausted")
        result[name] = [_row(row, budget) for row in rows[:1001]]
    return result


def sign_image_urls(records: dict[str, list[dict]], rkeys: dict[int, str]) -> dict[str, list[dict]]:
    result = copy.deepcopy(records)
    for rows in result.values():
        for row in rows:
            for segment in row["message"]:
                url = segment["data"].get("url", "") if segment["type"] == "image" else ""
                if not allowed_media_url(url) or urlsplit(url).hostname != "multimedia.nt.qq.com.cn":
                    continue
                query = parse_qs(urlsplit(url).query)
                scene = {"1406": 10, "1407": 20}.get(query.get("appid", [""])[0])
                key = rkeys.get(scene)
                if key and "rkey" not in query:
                    key = key.removeprefix("&rkey=").removeprefix("rkey=")
                    segment["data"]["url"] = url + ("&" if "?" in url else "?") + "rkey=" + quote(key, safe="")
    return result
