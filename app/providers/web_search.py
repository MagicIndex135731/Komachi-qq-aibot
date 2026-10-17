from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from math import ceil
from pathlib import Path
import re
import time
import uuid
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import httpx

from app.core.message_content import ImageAttachment
from app.core.url_policy import media_resource_url
from app.providers.shared_page_text import read_bilibili_text


@dataclass(slots=True)
class WebSearchResult:
    title: str
    snippet: str
    source: str
    date: str
    ref_id: str = ""


@dataclass(slots=True)
class WebSearchReport:
    results: list[WebSearchResult]
    content: str
    content_kind: str = 'web_page'


class WebSearchProviderError(RuntimeError):
    """Safe error categories; never retain upstream credentials or payloads."""


@dataclass(slots=True)
class WebPageContent:
    title: str
    url: str
    content: str


HTML_TITLE_PATTERN = re.compile(r"(?is)<title[^>]*>(.*?)</title>")
HTML_SCRIPTLIKE_PATTERN = re.compile(r"(?is)<(script|style|noscript|svg|iframe).*?>.*?</\1>")
HTML_COMMENT_PATTERN = re.compile(r"(?is)<!--.*?-->")
HTML_NOISE_TAG_PATTERN = re.compile(r"(?is)<(nav|header|footer|aside|form|button|select|option)\b.*?>.*?</\1>")
HTML_NOISE_ATTR_PATTERN = re.compile(
    r'(?is)<([a-z0-9]+)\b[^>]*(?:id|class|role|aria-label)=["\'][^"\']*'
    r"(?:nav|menu|header|footer|sidebar|aside|related|recommend|login|sign|register|comment|share|app|cookie|"
    r'breadcrumb|pagination|toolbar|search|subscribe|banner|promo|social)[^"\']*["\'][^>]*>.*?</\1>'
)
HTML_BLOCK_BREAK_PATTERN = re.compile(r"(?is)</?(?:p|div|section|article|main|li|ul|ol|h[1-6]|br|tr|table)\b[^>]*>")
HTML_TAG_PATTERN = re.compile(r"(?is)<[^>]+>")
HTML_META_IMAGE_PATTERN = re.compile(
    r'(?is)<meta\b[^>]*(?:property|name)=["\'](?:og:image|twitter:image)["\'][^>]*content=["\']([^"\']+)["\'][^>]*>'
)
HTML_IMG_SRC_PATTERN = re.compile(r'(?is)<img\b[^>]*src=["\']([^"\']+)["\'][^>]*>')
WHITESPACE_PATTERN = re.compile(r"\s+")
TEXT_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")

PREFERRED_HTML_REGION_PATTERNS = (
    re.compile(r"(?is)<article\b[^>]*>(.*?)</article>"),
    re.compile(r"(?is)<main\b[^>]*>(.*?)</main>"),
    re.compile(
        r'(?is)<(?:div|section)\b[^>]*(?:id|class|role)=["\'][^"\']*'
        r"(?:main|content|article|post|entry|story|正文)[^\"\']*[\"'][^>]*>(.*?)</(?:div|section)>"
    ),
)

ENGLISH_SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "why",
    "with",
}
NOISE_TEXT_KEYWORDS = (
    "home",
    "trending",
    "login",
    "log in",
    "register",
    "sign in",
    "sign up",
    "privacy policy",
    "cookie",
    "open app",
    "download app",
    "related topics",
    "related posts",
    "share",
    "comment",
    "comments",
    "subscribe",
    "search",
    "menu",
    "breadcrumb",
    "pagination",
    "footer",
    "header",
    "copyright",
    "首页",
    "登录",
    "注册",
    "打开app",
    "下载app",
    "相关推荐",
    "相关内容",
    "评论",
    "隐私",
    "版权",
    "菜单",
)


@dataclass(slots=True)
class WebPageCandidate:
    result: WebSearchResult
    page: WebPageContent
    score: float
    index: int


class WebSearchClient:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        http_client: httpx.Client | None = None,
        ddgs_factory: Callable[..., Any] | None = None,
        region: str = "wt-wt",
        backend: str = "auto",
        model: str = "",
        reasoning_effort: str = "medium",
    ) -> None:
        self.provider = provider.strip().lower()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client or httpx.Client(timeout=timeout_seconds)
        self.page_reader = None
        self.ddgs_factory = ddgs_factory
        self.region = region
        self.backend = backend
        self.model = (model or "").strip()
        if reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("invalid web search reasoning effort")
        self.reasoning_effort = reasoning_effort

    @property
    def supports_page_tools(self) -> bool:
        return self.page_reader is not None or self.provider in {"wyzlab", "wyzlab_alpha"}

    def search_report(self, query: str, max_results: int = 8) -> WebSearchReport:
        if not 1 <= max_results <= 20:
            raise ValueError("search result limit must be between 1 and 20")
        if self.provider not in {'wyzlab', 'wyzlab_alpha'}:
            return WebSearchReport(self.search(query, max_results=max_results), "")
        return self._alpha_report({"search_query": [{"q": query}]},
            instruction=f"搜索公开资料，最多收集{max_results}条相关来源：{query}", max_results=max_results)

    def open_page(self, url: str, *, lineno: int = 0) -> WebSearchReport:
        if self.page_reader is not None:
            return self.page_reader.open_page(url, lineno=lineno)
        if not self.supports_page_tools:
            raise WebSearchProviderError("page_tools_unavailable")
        deadline = time.monotonic() + self.timeout_seconds
        try:
            return self._alpha_report({"open": [{"ref_id": url, "lineno": lineno}]}, instruction="读取公开网页正文", max_results=20)
        except (httpx.HTTPError, WebSearchProviderError):
            page = read_bilibili_text(url, client=self.http_client, deadline=deadline)
            if page is None:
                raise
            body = '\n'.join(f'L{i}: {line}' for i, line in enumerate(page.lines) if i >= lineno)
            content = (f'{page.title} ({page.url})\n'
                f'citeturn0view0 Total lines: {len(page.lines)}\n{body}')
            return WebSearchReport([WebSearchResult(page.title, page.lines[3], page.url, '', 'turn0view0')],
                content, 'bilibili_public_metadata')

    def search_resources(self, query: str, max_results: int = 8, *, goal: str = '') -> WebSearchReport:
        """Use the configured search once; page discovery never calls an LLM."""
        deadline = time.monotonic() + self.timeout_seconds
        intent = f'\n当前检索目的：{goal[:1000]}' if goal else ''
        report = self.search_report(query + intent, max_results=max_results)
        discover = getattr(self.page_reader, 'discover_resources', None)
        if callable(discover):
            report = discover(report, query=query, goal=goal, deadline=deadline)
        return report

    def find_page(self, url: str, *, pattern: str) -> WebSearchReport:
        if self.page_reader is not None and self.provider not in {'wyzlab', 'wyzlab_alpha'}:
            return self.page_reader.find_page(url, pattern=pattern)
        if self.provider not in {'wyzlab', 'wyzlab_alpha'}:
            raise WebSearchProviderError("page_tools_unavailable")
        return self._alpha_report({"find": [{"ref_id": url, "pattern": pattern}]}, instruction="查找公开网页中的内容", max_results=20)

    def search(self, query: str, max_results: int = 3) -> list[WebSearchResult]:
        if self.provider == "ddgs":
            return self._search_ddgs(query=query, max_results=max_results)
        if self.provider in {"wyzlab", "wyzlab_alpha"}:
            return self._search_wyzlab_alpha(query=query, max_results=max_results)
        return self._search_tavily(query=query, max_results=max_results)

    def image_search(self, query: str, max_results: int = 3) -> list[ImageAttachment]:
        if self.provider == "ddgs":
            return self._image_search_ddgs(query=query, max_results=max_results)
        return self._image_search_from_web_pages(query=query, max_results=max_results)

    def read_pages(
        self,
        results: list[WebSearchResult],
        *,
        query: str | None = None,
        max_pages: int = 3,
        skim_limit: int = 6,
    ) -> list[WebPageContent]:
        candidates: list[WebPageCandidate] = []
        seen_urls: set[str] = set()
        candidate_limit = max(max_pages, skim_limit)
        for result in results:
            url = result.source.strip()
            if not url or url in seen_urls:
                continue
            if urlparse(url).scheme not in {"http", "https"}:
                continue
            seen_urls.add(url)
            page = self._read_single_page(url=url, fallback_title=result.title)
            if page is not None:
                candidates.append(
                    WebPageCandidate(
                        result=result,
                        page=page,
                        score=self._score_page_candidate(query=query or "", result=result, page=page),
                        index=len(candidates),
                    )
                )
            if len(candidates) >= candidate_limit:
                break
        if not query:
            return [candidate.page for candidate in candidates[:max_pages]]

        ranked = sorted(candidates, key=lambda candidate: (-candidate.score, candidate.index))
        return [candidate.page for candidate in ranked[:max_pages]]

    def _search_tavily(self, *, query: str, max_results: int) -> list[WebSearchResult]:
        if not self.api_key.strip():
            return []

        response = self.http_client.post(
            self.base_url,
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
        )
        response.raise_for_status()

        payload: dict[str, Any] = response.json()
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            return []

        results: list[WebSearchResult] = []
        for item in raw_results[:max_results]:
            if not isinstance(item, dict):
                continue
            results.append(
                WebSearchResult(
                    title=str(item.get("title", "")).strip(),
                    snippet=str(item.get("content", "")).strip(),
                    source=str(item.get("url", "")).strip(),
                    date=str(item.get("published_date", "")).strip(),
                )
            )
        return results

    def _search_wyzlab_alpha(self, *, query: str, max_results: int) -> list[WebSearchResult]:
        return self.search_report(query, max_results=max_results).results

    def _alpha_report(self, commands: dict[str, Any], *, instruction: str, max_results: int) -> WebSearchReport:
        """The bridge's plaintext output is independent of encrypted_output."""
        base = self.base_url.rstrip("/")
        if not base.endswith("/alpha/search"):
            base = f"{base}/alpha/search" if base.endswith("/v1") else f"{base}/v1/alpha/search"
        payload = {
            "id": f"xiaomachi-search-{uuid.uuid4().hex}",
            "model": self.model or "gpt-6.1-sol",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": instruction}],
            }],
            "commands": {**commands, "response_length": "long"},
            "settings": {"allowed_callers": ["direct"], "external_web_access": True},
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": 10000,
        }
        # Stream with a byte cap before JSON parsing. No retries; the turn
        # runner retains its slot if a cancelled HTTP worker is still alive.
        with self.http_client.stream(
            "POST", base,
            headers={
                "Accept": "application/json",
                "Originator": "Codex Desktop",
                "User-Agent": "axonhub/1.0",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json=payload,
            timeout=self.timeout_seconds,
        ) as response:
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > 2 * 1024 * 1024:
                    raise WebSearchProviderError("web_response_too_large")
        import json
        try:
            data = json.loads(body)
        except (ValueError, UnicodeError) as exc:
            raise WebSearchProviderError("web_response_invalid_json") from exc
        if not isinstance(data, dict) or data.get("error") or not isinstance(data.get("output", ""), str):
            raise WebSearchProviderError("web_response_invalid")
        content = data.get("output", "").strip()
        if content.startswith("Internal Error"):
            raise WebSearchProviderError("web_page_unavailable")
        if "search_query" not in commands and not content:
            raise WebSearchProviderError("web_page_empty")
        raw_results = data.get("results") or []
        if not isinstance(raw_results, list):
            raise WebSearchProviderError("web_results_invalid")
        results: list[WebSearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict) or item.get("type") not in {None, "text_result"}:
                continue
            results.append(
                WebSearchResult(
                    title=str(item.get("title", "")).strip(),
                    snippet=str(item.get("snippet", "")).strip(),
                    source=str(item.get("url", "")).strip(),
                    date=str(item.get("date", "")).strip(),
                    ref_id=str(item.get("ref_id", "")).strip(),
                )
            )
            if len(results) == max_results:
                break
        # Transport is capped above; tools apply tighter context budgets.
        return WebSearchReport(results, content)

    def _search_ddgs(self, *, query: str, max_results: int) -> list[WebSearchResult]:
        factory = self.ddgs_factory
        if factory is None:
            try:
                from ddgs import DDGS
            except ImportError as exc:
                raise RuntimeError("ddgs provider requires the 'ddgs' package to be installed") from exc
            factory = DDGS

        ddgs_client = factory(timeout=max(1, int(ceil(self.timeout_seconds))))
        last_error: Exception | None = None
        for region, backend in self._ddgs_search_variants():
            try:
                raw_results = ddgs_client.text(
                    query,
                    region=region,
                    safesearch="moderate",
                    max_results=max_results,
                    backend=backend,
                )
            except Exception as exc:
                last_error = exc
                continue

            results = self._map_ddgs_results(raw_results=raw_results, max_results=max_results)
            if results:
                return results

        if last_error is not None:
            raise last_error
        return []

    def _ddgs_search_variants(self) -> list[tuple[str, str]]:
        variants = [
            (self.region, self.backend),
            ("wt-wt", self.backend),
            (self.region, "lite"),
            ("wt-wt", "lite"),
        ]
        deduped: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for variant in variants:
            if variant in seen:
                continue
            seen.add(variant)
            deduped.append(variant)
        return deduped

    def _map_ddgs_results(self, *, raw_results: Any, max_results: int) -> list[WebSearchResult]:
        if raw_results is None:
            return []

        results: list[WebSearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            results.append(
                WebSearchResult(
                    title=str(item.get("title", "")).strip(),
                    snippet=str(item.get("body", "")).strip(),
                    source=str(item.get("href", "")).strip(),
                    date=str(item.get("date", "")).strip(),
                )
            )
            if len(results) >= max_results:
                break
        return results

    def _image_search_ddgs(self, *, query: str, max_results: int) -> list[ImageAttachment]:
        factory = self.ddgs_factory
        if factory is None:
            try:
                from ddgs import DDGS
            except ImportError as exc:
                raise RuntimeError("ddgs provider requires the 'ddgs' package to be installed") from exc
            factory = DDGS

        ddgs_client = factory(timeout=max(1, int(ceil(self.timeout_seconds))))
        last_error: Exception | None = None
        for region, backend in self._ddgs_search_variants():
            try:
                raw_results = ddgs_client.images(
                    query,
                    region=region,
                    safesearch="moderate",
                    max_results=max_results,
                )
            except Exception as exc:
                last_error = exc
                continue
            images = self._map_ddgs_image_results(raw_results=raw_results, max_results=max_results)
            if images:
                return images

        if last_error is not None:
            raise last_error
        return []

    def _map_ddgs_image_results(self, *, raw_results: Any, max_results: int) -> list[ImageAttachment]:
        if raw_results is None:
            return []
        images: list[ImageAttachment] = []
        for index, item in enumerate(raw_results, start=1):
            if not isinstance(item, dict):
                continue
            image_url = str(item.get("image", "") or item.get("thumbnail", "")).strip()
            if not image_url:
                continue
            thumbnail_url = str(item.get("thumbnail", "")).strip()
            file_id = str(item.get("title", "")).strip() or f"search-image-{index}"
            images.append(
                ImageAttachment(
                    url=image_url,
                    file_id=file_id,
                    fallback_url=thumbnail_url if thumbnail_url and thumbnail_url != image_url else None,
                )
            )
            if len(images) >= max_results:
                break
        return images

    def _image_search_from_web_pages(self, *, query: str, max_results: int) -> list[ImageAttachment]:
        text_results = self.search(query=query, max_results=max_results)
        images: list[ImageAttachment] = []
        seen_urls: set[str] = set()
        for result in text_results:
            for candidate_url in self._extract_image_urls_from_page(result.source):
                normalized_url = candidate_url.strip()
                if not normalized_url or normalized_url in seen_urls:
                    continue
                seen_urls.add(normalized_url)
                images.append(ImageAttachment(url=normalized_url, file_id=Path(urlparse(normalized_url).path).name or None))
                if len(images) >= max_results:
                    return images
        return images

    def _extract_image_urls_from_page(self, url: str) -> list[str]:
        try:
            response = self.http_client.get(
                url,
                follow_redirects=True,
                headers={"User-Agent": "qq-ai-bot/0.1"},
            )
            response.raise_for_status()
        except Exception:
            return []
        html_text = response.text or ""
        if not html_text.strip():
            return []

        candidates: list[str] = []
        for match in HTML_META_IMAGE_PATTERN.finditer(html_text):
            candidates.append(urljoin(str(response.url), unescape(match.group(1)).strip()))
        for match in HTML_IMG_SRC_PATTERN.finditer(html_text):
            candidates.append(urljoin(str(response.url), unescape(match.group(1)).strip()))

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            parsed = urlparse(candidate)
            if parsed.scheme not in {"http", "https"}:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        return deduped

    def _read_single_page(self, *, url: str, fallback_title: str) -> WebPageContent | None:
        try:
            response = self.http_client.get(
                url,
                follow_redirects=True,
                headers={"User-Agent": "qq-ai-bot/0.1"},
            )
            response.raise_for_status()
        except Exception:
            return None

        content_type = response.headers.get("content-type", "").lower()
        raw_text = response.text
        if not raw_text.strip():
            return None

        if "html" in content_type or "<html" in raw_text.lower():
            title, content = self._extract_html_page_text(raw_text, fallback_title=fallback_title)
        elif content_type.startswith("text/") or not content_type:
            title = fallback_title.strip() or url
            content = self._normalize_text(raw_text)
        else:
            return None

        if not content:
            return None

        return WebPageContent(
            title=title or fallback_title.strip() or url,
            url=str(response.url),
            content=content[:4000],
        )

    def _extract_html_page_text(self, html_text: str, *, fallback_title: str) -> tuple[str, str]:
        title_match = HTML_TITLE_PATTERN.search(html_text)
        title = self._normalize_text(unescape(title_match.group(1))) if title_match else fallback_title.strip()

        without_scripts = HTML_SCRIPTLIKE_PATTERN.sub(" ", html_text)
        without_comments = HTML_COMMENT_PATTERN.sub(" ", without_scripts)
        cleaned_html = self._strip_noise_blocks(without_comments)

        candidate_fragments = self._candidate_html_fragments(cleaned_html)
        best_lines = self._select_best_text_lines(candidate_fragments)
        if not best_lines:
            best_lines = self._extract_meaningful_lines(cleaned_html)
        content = self._normalize_text(" ".join(best_lines))

        if title and content.startswith(title):
            content = content[len(title) :].strip()
        return title, content

    def _strip_noise_blocks(self, html_text: str) -> str:
        cleaned = html_text
        for _ in range(3):
            next_cleaned = HTML_NOISE_TAG_PATTERN.sub(" ", cleaned)
            next_cleaned = HTML_NOISE_ATTR_PATTERN.sub(" ", next_cleaned)
            if next_cleaned == cleaned:
                break
            cleaned = next_cleaned
        return cleaned

    def _extract_html_page_links(self, html_text: str, *, base_url: str) -> list[str]:
        """Expose bounded article hrefs as data without fetching any target."""
        from app.core.url_policy import public_web_url
        fragments = self._candidate_html_fragments(self._strip_noise_blocks(
            HTML_COMMENT_PATTERN.sub(' ', HTML_SCRIPTLIKE_PATTERN.sub(' ', html_text))))
        # Embeds and next-page links can live in a shorter article region.
        # Text extraction selects one region; link discovery must cover all
        # eligible regions, retaining the same noise and public-URL checks.
        fragment = '\n'.join(fragments)
        links, seen = [], set()
        class Links(HTMLParser):
            href, label = '', ''
            def handle_starttag(inner, tag, attrs):
                if tag == 'a':
                    inner.href, inner.label = dict(attrs).get('href') or '', ''
            def handle_data(inner, data):
                if inner.href: inner.label = (inner.label + ' ' + data)[:200]
            def handle_endtag(inner, tag):
                if tag != 'a': return
                href, label = inner.href, inner.label
                inner.href, inner.label = '', ''
                if not href or href.startswith('#') or len(href) > 2000 or len(links) >= 64: return
                try: target = public_web_url(urljoin(base_url, href))
                except (ValueError, TypeError): return
                if target not in seen:
                    seen.add(target)
                    links.append((target, self._normalize_text(label)[:90]))
        Links().feed(fragment)
        links.sort(key=lambda item: (not media_resource_url(item[0]),
            urlparse(item[0]).hostname == urlparse(base_url).hostname))
        lines, size = [], 0
        for target, label in links[:24]:
            line = f'[网页链接，尚未读取] {label} {target}'
            if size + len(line) > 6000: break
            lines.append(line); size += len(line)
        return lines

    def _candidate_html_fragments(self, html_text: str) -> list[str]:
        fragments: list[str] = []
        seen: set[str] = set()
        for pattern in PREFERRED_HTML_REGION_PATTERNS:
            for fragment in pattern.findall(html_text):
                normalized = self._normalize_text(HTML_TAG_PATTERN.sub(" ", fragment))
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                fragments.append(fragment)
        if not fragments:
            fragments.append(html_text)
        return fragments

    def _select_best_text_lines(self, fragments: list[str]) -> list[str]:
        best_lines: list[str] = []
        best_score = float("-inf")
        for fragment in fragments:
            lines = self._extract_meaningful_lines(fragment)
            score = self._score_text_lines(lines)
            if score > best_score:
                best_score = score
                best_lines = lines
        return best_lines

    def _extract_meaningful_lines(self, fragment: str) -> list[str]:
        text_with_breaks = HTML_BLOCK_BREAK_PATTERN.sub("\n", fragment)
        text_only = HTML_TAG_PATTERN.sub(" ", text_with_breaks)
        lines: list[str] = []
        seen: set[str] = set()
        for raw_line in text_only.splitlines():
            normalized = self._normalize_text(unescape(raw_line))
            if not normalized or normalized in seen or self._is_noise_line(normalized):
                continue
            seen.add(normalized)
            lines.append(normalized)
        return lines

    def _score_text_lines(self, lines: list[str]) -> float:
        score = 0.0
        for line in lines:
            line_score = min(len(line), 220) / 2
            if any(punct in line for punct in ".!?;:。！？；："):
                line_score += 24
            if len(line) >= 40:
                line_score += 12
            score += line_score
        return score

    def _is_noise_line(self, line: str) -> bool:
        lowered = line.lower()
        if len(line) <= 40 and any(keyword in lowered for keyword in NOISE_TEXT_KEYWORDS):
            return True
        if len(line) <= 24 and line.count(" ") >= 3 and all(token.isalpha() for token in lowered.split()):
            return True
        return False

    def _score_page_candidate(self, *, query: str, result: WebSearchResult, page: WebPageContent) -> float:
        if not query.strip():
            return 0.0

        query_tokens = self._tokenize_relevance_text(query)
        if not query_tokens:
            return 0.0

        title_text = f"{result.title} {page.title}"
        snippet_text = result.snippet
        content_preview = page.content[:1500]
        url_text = page.url

        score = 0.0
        score += self._count_token_hits(query_tokens, title_text) * 10
        score += self._count_token_hits(query_tokens, snippet_text) * 5
        score += self._count_token_hits(query_tokens, content_preview) * 3
        score += self._count_token_hits(query_tokens, url_text) * 2

        if len(page.content) < 120:
            score -= 6
        else:
            score += min(len(page.content), 1200) / 200
        return score

    def _count_token_hits(self, tokens: list[str], text: str) -> float:
        lowered = text.lower()
        score = 0.0
        for token in tokens:
            if token in lowered:
                score += 1 + min(len(token), 8) / 8
        return score

    def _tokenize_relevance_text(self, text: str) -> list[str]:
        tokens: list[str] = []
        seen: set[str] = set()
        for raw_token in TEXT_TOKEN_PATTERN.findall(text.lower()):
            if raw_token in ENGLISH_SEARCH_STOPWORDS:
                continue
            if raw_token.isascii():
                if len(raw_token) <= 1:
                    continue
                if raw_token not in seen:
                    seen.add(raw_token)
                    tokens.append(raw_token)
                continue

            if len(raw_token) <= 1:
                continue
            if raw_token not in seen:
                seen.add(raw_token)
                tokens.append(raw_token)
            for window in (2, 3):
                if len(raw_token) < window:
                    continue
                for index in range(len(raw_token) - window + 1):
                    token = raw_token[index : index + window]
                    if token in seen:
                        continue
                    seen.add(token)
                    tokens.append(token)
        return tokens

    def _normalize_text(self, text: str) -> str:
        return WHITESPACE_PATTERN.sub(" ", text).strip()
