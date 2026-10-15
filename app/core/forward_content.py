"""Recognize QQ forward handles at the outer message level only."""
from __future__ import annotations

import json
import re
from xml.etree import ElementTree


MAX_FORWARD_REFS = 8
MAX_CONTENT = 65536
_HANDLE = re.compile(r"[A-Za-z0-9_+/=.-]{1,512}\Z")
_CQ_FORWARD = re.compile(r"\[CQ:forward,([^\]\r\n]{1,2048})\]")


def valid_forward_id(value) -> bool:
    return isinstance(value, str) and _HANDLE.fullmatch(value) is not None


def cq_unescape(value: str) -> str:
    return value.replace("&#91;", "[").replace("&#93;", "]").replace("&#44;", ",").replace("&amp;", "&")


def _card_id(kind, value):
    if isinstance(value, str):
        if len(value) > MAX_CONTENT:
            return None
        if kind == "json":
            try:
                value = json.loads(value)
            except (ValueError, RecursionError):
                return None
        elif kind == "xml":
            if "<!DOCTYPE" in value.upper() or "<!ENTITY" in value.upper():
                return None
            try:
                root = ElementTree.fromstring(value)
            except ElementTree.ParseError:
                return None
            if root.tag == "msg" and root.get("serviceID") == "35":
                return root.get("m_resid") or root.get("resid")
            return None
    if kind == "json" and isinstance(value, dict) and value.get("app") == "com.tencent.multimsg":
        meta = value.get("meta")
        detail = meta.get("detail") if isinstance(meta, dict) else None
        if isinstance(detail, dict):
            return detail.get("resid") or detail.get("resId")
    return None


def extract_forward_ids(message) -> tuple[str, ...]:
    """Ignore quoted text, inline node bodies and unrelated JSON/XML cards."""
    candidates = []
    if isinstance(message, str):
        for match in _CQ_FORWARD.finditer(message[:MAX_CONTENT]):
            fields = dict(part.split("=", 1) for part in match[1].split(",") if "=" in part)
            candidates.append(cq_unescape(fields.get("id", "")))
    elif isinstance(message, list):
        for item in message[:256]:
            if not isinstance(item, dict) or not isinstance(item.get("data"), dict):
                continue
            data, kind = item["data"], item.get("type")
            if kind == "forward":
                candidates.append(data.get("id") or data.get("res_id") or data.get("forward_id"))
            elif kind in {"json", "xml"}:
                candidates.append(_card_id(kind, data.get("data")))
    return tuple(dict.fromkeys(value for value in candidates if valid_forward_id(value)))[:MAX_FORWARD_REFS]


def extract_forward_file_names(message) -> tuple[str, ...]:
    """Bundle-local XML references, usable only beneath an observed outer resource.

    A filename is a lookup key within that resource, never a filesystem path or
    a downloadable resource ID. The XML segment's numeric resid is a subtype.
    """
    candidates = []
    if not isinstance(message, list):
        return ()
    for item in message[:256]:
        if not isinstance(item, dict) or item.get("type") != "xml":
            continue
        data = item.get("data")
        value = data.get("data") if isinstance(data, dict) else None
        if (not isinstance(value, str) or len(value) > MAX_CONTENT
                or "<!DOCTYPE" in value.upper() or "<!ENTITY" in value.upper()):
            continue
        try:
            root = ElementTree.fromstring(value)
        except ElementTree.ParseError:
            continue
        if (root.tag == "msg" and root.get("serviceID") == "35"
                and root.get("action") == "viewMultiMsg"
                and not (root.get("m_resid") or root.get("resid"))):
            name = root.get("m_fileName")
            if valid_forward_id(name):
                candidates.append(name)
    return tuple(dict.fromkeys(candidates))[:MAX_FORWARD_REFS]
