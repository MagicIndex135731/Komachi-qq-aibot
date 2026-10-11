from __future__ import annotations

import re
import ipaddress
from urllib.parse import urlsplit, urlunsplit


def public_web_url(value: str) -> str:
    """Allow only public HTTP URLs for the remote web bridge."""
    parts = urlsplit(value.strip())
    host = (parts.hostname or "").lower().rstrip(".")
    if (parts.scheme not in {"http", "https"} or not host or parts.username is not None
            or parts.password is not None or parts.port not in {None, 80, 443}
            or any(ord(c) < 33 for c in value) or "\\" in value):
        raise ValueError("invalid_public_url")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if ("." not in host or host.endswith((".localhost", ".local", ".internal", ".test", ".invalid"))
                or host == "localhost" or host.replace(".", "").isdigit()
                or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host)):
            raise ValueError("invalid_public_url")
    else:
        if not address.is_global:
            raise ValueError("invalid_public_url")
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, parts.query, ""))


_EXPLICIT_URL_REQUEST = re.compile(
    r"(?:"
    r"(?:给|发|贴|附|提供|带|留下).{0,10}(?:网址|链接|url|官网)"
    r"|(?:网址|链接|url).{0,10}(?:给我|发我|贴出|提供|附上|是什么|在哪)"
    r"|(?:官网|原文|来源|出处).{0,4}(?:网址|链接)"
    r"|(?:send|give|share|provide|include).{0,20}(?:link|url|website)"
    r"|(?:source|official|original).{0,10}(?:link|url)"
    r")",
    re.IGNORECASE,
)
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)", re.IGNORECASE)
_PARENTHESIZED_LINK_CITATION = re.compile(
    r"[（(]\s*\[[^\]]*\]\((?:https?://|www\.)[^)]+\)\s*[)）]",
    re.IGNORECASE,
)
_AUTOLINK = re.compile(r"<(?:https?://|www\.)[^>]+>", re.IGNORECASE)
_SCHEME_URL = re.compile(r"(?i)(?:https?://|www\.)[^\s<>\[\]{}，。；：！？、（）]+")
_BARE_DOMAIN = re.compile(
    r"(?i)\b(?:[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?\.)+"
    r"(?:com|cn|net|org|io|ai|app|dev|gov|edu|me|tv|cc|co|xyz)"
    r"(?:/[^\s<>\[\]{}，。；：！？、（）]*)?"
)
_LINK_LINE_LABEL = re.compile(r"^\s*([^:：\n。！？!?，,;；|]{1,32}[:：])")
_EMPTY_LINK_LABEL = re.compile(r"^[^:：\n。！？!?，,;；|]{1,32}[:：]\s*$")


def explicitly_requests_urls(text: str) -> bool:
    return bool(_EXPLICIT_URL_REQUEST.search(str(text or "")))


def url_reply_policy_instruction(text: str) -> str:
    if explicitly_requests_urls(text):
        return "The user explicitly requested links. Include only the URLs needed to answer that request."
    return (
        "Do not include URLs, website addresses, Markdown links, citations, or parenthetical source markers in the reply. "
        "Summarize source information directly without links or citation brackets."
    )


def filter_reply_urls(text: str, *, allow_urls: bool) -> str:
    original = str(text or "").strip()
    if allow_urls or not original:
        return original

    filtered = _PARENTHESIZED_LINK_CITATION.sub("", original)
    filtered = _MARKDOWN_LINK.sub(r"\1", filtered)
    filtered = _AUTOLINK.sub("", filtered)
    filtered = _SCHEME_URL.sub("", filtered)
    filtered = _BARE_DOMAIN.sub("", filtered)
    filtered = re.sub(r"(?:\(\s*\)|（\s*）|\[\s*\]|【\s*】)", "", filtered)
    # Remove only standalone labels whose link was removed. Keep ordinary
    # headings and source lines that still contain descriptive answer text.
    link_labels = set()
    for line in original.splitlines():
        label = _LINK_LINE_LABEL.match(line)
        if label and (_SCHEME_URL.search(line) or _BARE_DOMAIN.search(line)):
            link_labels.add(label.group(1).strip())
    filtered = "\n".join(line.rstrip() for line in filtered.splitlines()
        if not (_EMPTY_LINK_LABEL.fullmatch(line.strip()) and line.strip() in link_labels))
    filtered = re.sub(r"[ \t]+([,.;:!?，。；：！？])", r"\1", filtered)
    filtered = re.sub(r"[ \t]{2,}", " ", filtered)
    filtered = re.sub(r"\n[ \t]+", "\n", filtered)
    filtered = re.sub(r"\n{3,}", "\n\n", filtered).strip(" \t\r\n-—|，,；;")
    return filtered or "相关信息已查到。"
