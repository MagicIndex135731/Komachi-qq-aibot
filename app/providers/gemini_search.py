"""Independent native Gemini Google Search with verifiable grounding sources."""
from __future__ import annotations

from datetime import datetime
import ipaddress
import json
import logging
import re
from html import unescape
from pathlib import Path
import socket
import time
import threading
from collections import OrderedDict
from typing import Callable
from urllib.parse import quote, urljoin, urlparse
from zoneinfo import ZoneInfo

import httpx

from app.core.url_policy import media_resource_url, public_web_url
from app.core.message_content import ImageAttachment
from app.providers.llm_client import resolve_reasoning_model
from app.providers.shared_page_text import read_bilibili_text
from app.providers.web_search import (
    HTML_IMG_SRC_PATTERN, HTML_META_IMAGE_PATTERN, WebPageContent,
    WebSearchClient, WebSearchProviderError, WebSearchReport, WebSearchResult,
)

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
logger = logging.getLogger(__name__)


def _resolve_addresses(host: str, port: int) -> list[str]:
    return [item[4][0] for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


class GeminiSearchClient(WebSearchClient):
    def __init__(self, *, address_resolver: Callable = _resolve_addresses,
                 page_http_client: httpx.Client | None = None, page_proxy_url: str = '', **kwargs) -> None:
        super().__init__(**kwargs)
        self.address_resolver = address_resolver
        if page_proxy_url:
            proxy = urlparse(page_proxy_url)
            if (proxy.scheme != 'http' or proxy.hostname not in {'127.0.0.1', 'localhost', '::1'}
                    or proxy.port != 18797 or proxy.username or proxy.password):
                raise ValueError('page proxy must be the local Mihomo listener')
        self.page_proxy_url = page_proxy_url
        self._proxy_dns_cache = OrderedDict()
        self._dns_http_client = (page_http_client or httpx.Client(proxy=page_proxy_url, trust_env=False,
            timeout=self.timeout_seconds)) if page_proxy_url else None
        if page_proxy_url and page_http_client is None:
            from app.providers.page_proxy_transport import ValidatedPageProxyTransport
            self.page_http_client = httpx.Client(transport=ValidatedPageProxyTransport(page_proxy_url,
                self._proxy_addresses), trust_env=False, timeout=self.timeout_seconds)
        else:
            self.page_http_client = page_http_client or self.http_client
        self._page_cache = OrderedDict()
        self._page_cache_lock = threading.Lock()

    @property
    def supports_page_tools(self) -> bool:
        return True

    @property
    def search_content_kind(self) -> str:
        return "grounded_answer"

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise httpx.TimeoutException("web deadline exceeded")
        return remaining

    def search(self, query: str, max_results: int = 8) -> list[WebSearchResult]:
        return self.search_report(query, max_results=max_results).results

    def search_report(self, query: str, max_results: int = 8) -> WebSearchReport:
        return self._search_report(query, max_results, time.monotonic() + self.timeout_seconds)

    def _search_report(self, query: str, max_results: int, deadline: float) -> WebSearchReport:
        if not 1 <= max_results <= 20:
            raise ValueError("search result limit must be between 1 and 20")
        model = resolve_reasoning_model(self.model, self.reasoning_effort)
        if not model.startswith("gemini-"):
            raise ValueError("native Gemini search requires a Gemini model")
        base = self.base_url.removesuffix("/v1").removesuffix("/v1beta")
        endpoint = f"{base}/v1beta/models/{quote(model, safe='-._')}:streamGenerateContent"
        today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
        payload = {
            "contents": [{"role": "user", "parts": [{"text":
                f"今天是{today}（上海）。请使用Google搜索查证以下问题，优先直接来源，"
                f"给出简明结果、相关日期及最多{max_results}条来源。下面是检索式，不要解释检索式或账号路径的含义。"
                "找帖子、图片或视频时优先定位具体原帖或含原帖链接的报道，不能只介绍账号。"
                "只保留用户明确的日期要求；无日期时不要自加近期窗口或本人发布、真人自拍等条件，历史素材也可作为候选并注明日期。来源可少于上限。"
                "只需简短候选列表。不要拼接、猜测链接或未经来源确认的事实；报道日期不是原帖日期。\n" + query}]}],
            # Mixing server search with ordinary functions disables grounding
            # on this relay. This request deliberately contains one native tool.
            "tools": [{"googleSearch": {}}],
            # Native Gemini includes thinking in this cap. 4096 repeatedly
            # ended real grounded searches with MAX_TOKENS before metadata.
            "generationConfig": {"maxOutputTokens": 8192},
        }
        from app.core.request_cost import current_request_budget
        budget=current_request_budget.get()
        if budget:
            budget.spend('model_calls');budget.spend('estimated_tokens',9000);budget.reserve_network(MAX_RESPONSE_BYTES)
        texts, metadata, terminal = [], {}, False
        with self.http_client.stream("POST", endpoint, params={"alt": "sse"},
            headers={"x-goog-api-key": self.api_key, "Accept": "text/event-stream"},
            json=payload, timeout=self._remaining(deadline)) as response:
            response.raise_for_status()
            pending, size = bytearray(), 0
            for chunk in response.iter_bytes():
                self._remaining(deadline)
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise WebSearchProviderError("web_response_too_large")
                pending.extend(chunk)
                while b"\n" in pending:
                    line, _, rest = pending.partition(b"\n")
                    pending = bytearray(rest)
                    if not line.startswith(b"data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == b"[DONE]":
                        continue
                    try:
                        invalid_reason = 'invalid_frame'
                        frame = json.loads(raw)
                        if not isinstance(frame, dict) or frame.get("error"):
                            raise ValueError("invalid frame")
                        candidates = frame.get("candidates") or []
                        candidate = candidates[0] if candidates else {}
                        for part in (candidate.get("content") or {}).get("parts") or []:
                            if part.get("functionCall"):
                                invalid_reason = 'unexpected_client_function'
                                raise ValueError("client function is not server search")
                            if isinstance(part.get("text"), str) and not part.get("thought"):
                                texts.append(part["text"])
                        grounding = candidate.get("groundingMetadata")
                        if isinstance(grounding, dict):
                            metadata.update(grounding)
                        finish = candidate.get("finishReason")
                        if finish and finish != "STOP":
                            invalid_reason = ('incomplete_' + finish if finish in {
                                'MAX_TOKENS', 'SAFETY', 'RECITATION', 'MALFORMED_FUNCTION_CALL',
                                'OTHER', 'BLOCKLIST', 'PROHIBITED_CONTENT'} else 'incomplete_unknown')
                            raise ValueError("incomplete native search")
                        terminal = finish == "STOP"
                    except (ValueError, TypeError, AttributeError, IndexError) as exc:
                        logger.warning('gemini_search_response_invalid reason=%s', invalid_reason)
                        if invalid_reason in {'incomplete_SAFETY','incomplete_BLOCKLIST','incomplete_PROHIBITED_CONTENT'}:
                            raise WebSearchProviderError('web_search_content_rejected') from exc
                        raise WebSearchProviderError("web_response_invalid") from exc
                    if terminal:
                        break
                if terminal:
                    break  # Do not wait for EOF after the completed native frame.
        if not terminal:
            raise WebSearchProviderError("web_response_invalid")
        results = self._grounded_results(metadata, max_results)
        answer = "".join(texts).strip()
        if not results or not answer:
            raise WebSearchProviderError("web_grounding_unavailable")
        # Model-authored markdown targets are not grounding metadata. A
        # plausible-looking invented article URL must not become a locator.
        allowed = {hit.source for hit in results}
        answer = self._grounded_links_only(answer, allowed)
        for hit in results:
            hit.snippet = self._grounded_links_only(hit.snippet, allowed)
        sources = "\n".join(f"{hit.title} ({hit.source}) cite{hit.ref_id}" for hit in results)
        return WebSearchReport(results, f"Google搜索生成的有来源摘要（不是网页原文）：\n{answer}\n\n来源：\n{sources}",
            "grounded_answer")

    @staticmethod
    def _grounded_links_only(text: str, allowed: set[str]) -> str:
        return re.sub(r'https?://[^\s<>\[\]()`，。；]+',
            lambda match: match.group() if match.group() in allowed else '链接须从来源页核验', text)

    def search_resources(self, query: str, max_results: int = 8, *, goal: str = '') -> WebSearchReport:
        """Discover actual page links, never links invented in search prose.

        One search and at most three source reads share the normal deadline.
        Reads use the page/Mihomo transport; targets remain untrusted data and
        are not downloaded, followed recursively, or treated as verified posts.
        """
        deadline = time.monotonic() + self.timeout_seconds
        intent = f'\n当前用户的检索目的（保留对象和时间范围）：{goal[:1000]}' if goal else ''
        report = self._search_report(query + intent, max_results, deadline)
        return self.discover_resources(report, query=query, goal=goal, deadline=deadline)

    def discover_resources(self, report, *, query='', goal='', deadline=None):
        """Shared HTTP-only discovery; callers have already performed one search."""
        deadline = deadline or time.monotonic() + 15
        page_deadline = min(deadline, time.monotonic() + 15)
        results, sections, seen = list(report.results), [], {hit.source for hit in report.results}
        candidates = list(enumerate(report.results))
        if re.search(r'最新|最近|近期|\b(?:latest|recent)\b', goal + ' ' + query, re.I):
            today = datetime.now(ZoneInfo('Asia/Shanghai'))
            def priority(item):
                hit = item[1]
                # These are unverified ranking hints only, never returned as
                # post dates or evidence of freshness. A recent source near
                # the end must not lose all read slots to old top results.
                hints = [(int(y), int(m)) for y, m in re.findall(
                    r'\b(20\d{2})[年/-](\d{1,2})(?:月|[/\s-]|\b)', hit.snippet + ' ' + hit.title)
                    if 1 <= int(m) <= 12 and (int(y), int(m)) <= (today.year, today.month)]
                return max(hints, default=(0, 0)), media_resource_url(hit.source)
            candidates.sort(key=priority, reverse=True)
        for index, hit in candidates[:3]:
            if time.monotonic() >= page_deadline:
                break
            try:
                title, target, lines, _ = self._page_lines(hit.source,
                    min(page_deadline, time.monotonic() + 6), client=self.page_http_client)
            except (httpx.HTTPError, WebSearchProviderError, ValueError, OSError):
                # Failure of one result says nothing about other sources.
                sections.append(f'来源{index + 1}本次未能读取；不代表没有原帖或媒体。')
                continue
            results[index] = WebSearchResult(title, hit.snippet, target, hit.date, hit.ref_id)
            links = [line for line in lines if line.startswith('[网页链接，尚未读取]')]
            sections.append(f'实际读取来源：{title} ({target})\n' + '\n'.join(links[:12]))
            # Fair room for later sources: the first index page must not fill
            # the entire 20-source envelope with navigation links.
            for line in links[:4]:
                url = line.rsplit(' ', 1)[-1]
                if url in seen or len(results) >= 20:
                    continue
                seen.add(url)
                results.append(WebSearchResult('来源页内链接（目标尚未核验）',
                    f'由实际页面 {target} 提取；' + line[:180], url, '', f'turn0search{len(results)}'))
        if sections:
            report = WebSearchReport(results,
                '真实来源读取与链接发现（页面链接不证明作者、日期或媒体存在）：\n' +
                '\n\n'.join(sections)[:6500] + '\n\n' + report.content, report.content_kind)
        return report

    @staticmethod
    def _grounded_results(metadata: dict, limit: int) -> list[WebSearchResult]:
        chunks = metadata.get("groundingChunks") or []
        supports = metadata.get("groundingSupports") or []
        if not isinstance(chunks, list) or not isinstance(supports, list):
            raise WebSearchProviderError("web_results_invalid")
        results, seen = [], set()
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict) or not isinstance(chunk.get("web"), dict):
                continue
            web = chunk["web"]
            try:
                url = public_web_url(web.get("uri", ""))
            except (ValueError, TypeError):
                continue
            if url in seen:
                continue
            snippets = []
            for support in supports:
                if not isinstance(support, dict):
                    continue
                indices = support.get("groundingChunkIndices")
                segment = support.get("segment")
                if isinstance(indices, list) and index in indices and isinstance(segment, dict):
                    text = segment.get("text")
                    if isinstance(text, str) and text not in snippets:
                        snippets.append(text)
            seen.add(url)
            results.append(WebSearchResult(str(web.get("title") or url)[:200], " ".join(snippets)[:2000],
                url, "", f"turn0search{len(results)}"))
            if len(results) >= limit:
                break
        return results

    def _proxy_addresses(self, url: str, deadline: float) -> list[str]:
        url = public_web_url(url)
        host = urlparse(url).hostname
        try:
            literal = ipaddress.ip_address(host)
            if not literal.is_global:
                raise ValueError('nonpublic_address')
            return [str(literal)]
        except ValueError:
            pass
        with self._page_cache_lock:
            cached = self._proxy_dns_cache.get(host)
            if cached and time.monotonic() < cached[0]:
                return cached[1]
        # Fixed, certificate-verified public resolver through the same proxy.
        # No system DNS change, fake-IP allowlist, or model credentials.
        with self._dns_http_client.stream('GET', 'https://1.1.1.1/dns-query',
                params={'name': host, 'type': 'A'}, headers={'Accept': 'application/dns-json'},
                follow_redirects=False, timeout=min(10, self._remaining(deadline))) as response:
            response.raise_for_status()
            payload = bytearray()
            for chunk in response.iter_bytes():
                self._remaining(deadline);payload.extend(chunk)
                if len(payload) > 65536:
                    raise WebSearchProviderError('web_page_unavailable')
        value = json.loads(payload)
        answers = value.get('Answer', []) if isinstance(value, dict) and value.get('Status') == 0 else []
        addresses = [str(item.get('data', '')) for item in answers if isinstance(item, dict) and item.get('type') == 1]
        if not addresses or len(addresses) > 16 or any(not ipaddress.ip_address(address).is_global for address in addresses):
            raise ValueError('nonpublic_address')
        ttl = min([60, *(item.get('TTL', 30) for item in answers if isinstance(item, dict)
            and type(item.get('TTL', 30)) is int)])
        with self._page_cache_lock:
            self._proxy_dns_cache[host] = (time.monotonic() + max(0, ttl), addresses)
            while len(self._proxy_dns_cache) > 32:
                self._proxy_dns_cache.popitem(last=False)
        return addresses

    def _public_target(self, url: str, *, client=None, deadline=None) -> str:
        try:
            url = public_web_url(url)
            parsed = urlparse(url)
            addresses = (self._proxy_addresses(url, deadline or time.monotonic() + self.timeout_seconds)
                if self.page_proxy_url and client is self.page_http_client else
                self.address_resolver(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)))
            if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
                raise ValueError("nonpublic address")
        except (ValueError, TypeError, OSError) as exc:
            raise WebSearchProviderError("web_page_unavailable") from exc
        return url

    def _fetch_page(self, url: str, deadline: float, *, client=None) -> tuple[str, str, bool]:
        # Every redirect is validated independently. Provider credentials are
        # supplied only on the native search POST, never on these website GETs.
        for _ in range(6):
            url = self._public_target(url, client=client, deadline=deadline)
            accept = 'application/vnd.github+json' if urlparse(url).hostname == 'api.github.com' else 'text/html,text/plain,application/json;q=0.9'
            with (client or self.http_client).stream("GET", url, follow_redirects=False,
                timeout=self._remaining(deadline), extensions={'page_deadline': deadline},
                headers={"Accept": accept, 'User-Agent': 'Xiaomachi-WebReader/1.0'}) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise WebSearchProviderError("web_page_unavailable")
                    url = urljoin(url, location)
                    continue
                response.raise_for_status()
                kind = response.headers.get("content-type", "").split(";", 1)[0].lower()
                json_type = kind == 'application/json' or kind.endswith('+json')
                if kind and kind not in {"text/html", "application/xhtml+xml", "text/plain", "text/markdown"} and not json_type:
                    raise WebSearchProviderError("web_page_unavailable")
                body = bytearray()
                for chunk in response.iter_bytes():
                    self._remaining(deadline)
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise WebSearchProviderError("web_response_too_large")
                text = body.decode(response.encoding or 'utf-8', errors='replace')
                if json_type:
                    try:
                        value = json.loads(text)
                    except ValueError as exc:
                        raise WebSearchProviderError('web_response_invalid_json') from exc
                    if (urlparse(url).hostname == 'api.github.com' and isinstance(value, dict)
                            and isinstance(value.get('tree'), list)):
                        # Directory metadata, never a claim that any code was
                        # read. One entry per line avoids replaying node URLs/SHA.
                        lines = ['GitHub tree index (metadata only): ' + json.dumps({
                            'sha': value.get('sha'), 'truncated': value.get('truncated', False)}, ensure_ascii=False)]
                        lines.extend(json.dumps({k: item[k] for k in ('path', 'type', 'size') if k in item},
                            ensure_ascii=False, separators=(',', ':')) for item in value['tree'] if isinstance(item, dict))
                        text = '\n'.join(lines)
                    else:
                        text = json.dumps(value, ensure_ascii=False, indent=2)
                return url, text, kind in {'text/html', 'application/xhtml+xml'}
        raise WebSearchProviderError("web_page_unavailable")

    def _page_lines(self, url: str, deadline: float, *, client=None) -> tuple[str, str, list[str], str]:
        self._public_target(url, client=client, deadline=deadline)
        key = (url, client is self.page_http_client and client is not self.http_client)
        with self._page_cache_lock:
            cached = self._page_cache.get(key)
            if cached and time.monotonic() - cached[0] < 300:
                self._page_cache.move_to_end(key)
            else:
                cached = None
        if cached:
            self._public_target(cached[1][1], client=client, deadline=deadline)
            return cached[1]
        try:
            url, body, html = self._fetch_page(url, deadline, client=client)
        except (httpx.HTTPError, WebSearchProviderError):
            page = read_bilibili_text(url, client=client or self.http_client, deadline=deadline)
            if page is None:
                raise
            return page.title, page.url, page.lines, "bilibili_public_metadata"
        title, content = self._extract_html_page_text(body, fallback_title=url) if html else (url, body)
        raw_source = not html and urlparse(url).hostname in {'raw.githubusercontent.com', 'gist.githubusercontent.com'}
        if raw_source:
            # L is a pagination index. Source positions survive blank lines
            # and wrapping independently, so an audit can cite the real file.
            lines = [f'[source_line={number};part={offset // 240 + 1}] {line[offset:offset + 240]}'
                for number, line in enumerate(content.splitlines(), 1)
                for offset in range(0, max(1, len(line)), 240)]
        else:
            lines = [part for line in content.splitlines() if line.strip() for part in
                (line[i:i + 240] for i in range(0, len(line), 240))]
        if html:
            lines.extend(self._extract_html_page_links(body, base_url=url))
        if not lines:
            raise WebSearchProviderError("web_page_empty")
        result = (title, url, lines, 'raw_source' if raw_source else 'web_page')
        with self._page_cache_lock:
            self._page_cache[key] = (time.monotonic(), result)
            self._page_cache.move_to_end(key)
            while len(self._page_cache) > 8 or sum(sum(len(line.encode('utf-8')) for line in item[1][2]) for item in self._page_cache.values()) > 8 * 1024 * 1024:
                self._page_cache.popitem(last=False)
        return result

    @staticmethod
    def _page_report(title: str, url: str, lines: list[str], start: int, kind: str = "web_page") -> WebSearchReport:
        content = f"{title} ({url})\nciteturn0view0 Total lines: {len(lines)}\n"
        content += "\n".join(f"L{i}: {line}" for i, line in enumerate(lines) if i >= start)
        return WebSearchReport([WebSearchResult(title, lines[min(start, len(lines) - 1)][:500], url, "", "turn0view0")], content, kind)

    def open_page(self, url: str, *, lineno: int = 0) -> WebSearchReport:
        title, target, lines, kind = self._page_lines(url, time.monotonic() + self.timeout_seconds, client=self.page_http_client)
        return self._page_report(title, target, lines, max(0, lineno), kind)

    def find_page(self, url: str, *, pattern: str) -> WebSearchReport:
        title, target, lines, kind = self._page_lines(url, time.monotonic() + self.timeout_seconds, client=self.page_http_client)
        searchable = [line.split('] ', 1)[1] for line in lines] if kind == 'raw_source' else lines
        index = next((i for i, line in enumerate(searchable) if pattern.casefold() in line.casefold()), None)
        report = self._page_report(title, target, lines, max(0, (index or 0) - 3), kind)
        if index is None:
            report.content = f"{title} ({target})\nTotal lines: {len(lines)}\nPattern not found in retrieved page."
        return report

    def read_pages(self, results: list[WebSearchResult], *, query: str | None = None, max_pages: int = 3,
        skim_limit: int = 6) -> list[WebPageContent]:
        deadline, pages = time.monotonic() + self.timeout_seconds, []
        seen = set()
        for index, hit in enumerate(results[:max(max_pages, skim_limit)]):
            if hit.source in seen:
                continue
            seen.add(hit.source)
            try:
                title, url, lines, _ = self._page_lines(hit.source, deadline)
                page = WebPageContent(title, url, "\n".join(lines)[:4000])
                score = self._score_page_candidate(query=query or "", result=hit, page=page)
                pages.append((score, index, page))
            except (httpx.HTTPError, WebSearchProviderError):
                continue
        if query:
            pages.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in pages[:max_pages]]

    def image_search(self, query: str, max_results: int = 3) -> list[ImageAttachment]:
        if not 1 <= max_results <= 20:
            raise ValueError("image result limit must be between 1 and 20")
        deadline, images, seen = time.monotonic() + self.timeout_seconds, [], set()
        report = self._search_report(query + " 图片 参考图", min(6, max_results), deadline)
        for hit in report.results[:6]:
            try:
                url, body, html = self._fetch_page(hit.source, deadline)
            except (httpx.HTTPError, WebSearchProviderError):
                continue
            if not html:
                continue
            for raw in HTML_META_IMAGE_PATTERN.findall(body) + HTML_IMG_SRC_PATTERN.findall(body):
                try:
                    image = public_web_url(urljoin(url, unescape(raw).strip()))
                except ValueError:
                    continue
                if image not in seen:
                    seen.add(image)
                    images.append(ImageAttachment(url=image, file_id=Path(urlparse(image).path).name or None))
                if len(images) >= max_results:
                    return images
        return images
