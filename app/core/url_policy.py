from __future__ import annotations

import re


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
    filtered = re.sub(r"[ \t]+([,.;:!?，。；：！？])", r"\1", filtered)
    filtered = re.sub(r"[ \t]{2,}", " ", filtered)
    filtered = re.sub(r"\n[ \t]+", "\n", filtered)
    filtered = re.sub(r"\n{3,}", "\n\n", filtered).strip(" \t\r\n-—|，,；;")
    return filtered or "相关信息已查到。"
