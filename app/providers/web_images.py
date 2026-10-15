"""Public HTML image discovery and bounded, credential-free pixel downloads."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from html.parser import HTMLParser
import http.client
import ipaddress
import json
import re
import socket
import ssl
import time
from urllib.parse import unquote, urljoin, urlsplit

import httpx

from app.core.url_policy import public_web_url
from app.core.video_file import MAX_VIDEO_BYTES
from app.providers.page_proxy_transport import ValidatedPageProxyTransport, _PageStream, remaining

MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_CANDIDATES = 160
_IMAGE_PATH = re.compile(r"\.(?:png|jpe?g|webp|gif)(?:$|\?)", re.I)


class WebImageError(ValueError):
    """A safe error code; never includes a URL, response body or credentials."""


class PublicImageTransport(httpx.BaseTransport):
    """Pin public DNS results for the socket while retaining TLS Host/SNI."""
    def handle_request(self, request):
        connection = None
        try:
            target = httpx.URL(public_web_url(str(request.url)))
            deadline = request.extensions['page_deadline']
            port = target.port or (443 if target.scheme == 'https' else 80)
            addresses = socket.getaddrinfo(target.host, port, type=socket.SOCK_STREAM)
            if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
                raise WebImageError('web_image_nonpublic_address')
            connection = http.client.HTTPConnection(addresses[0][4][0], port, timeout=remaining(deadline))
            connection.connect()
            connection.sock.settimeout(remaining(deadline))
            if target.scheme == 'https':
                connection.sock = ssl.create_default_context().wrap_socket(connection.sock, server_hostname=target.host)
            headers = {'Host': target.netloc.decode('ascii'), 'Accept-Encoding': 'identity',
                'User-Agent': request.headers.get('user-agent', 'Xiaomachi-WebReader/1.0'), 'Accept': request.headers['accept']}
            if 'referer' in request.headers:
                headers['Referer'] = request.headers['referer']
            connection.request('GET', target.raw_path.decode('ascii'), headers=headers)
            connection.sock.settimeout(remaining(deadline))
            response = connection.getresponse()
            return httpx.Response(response.status, headers=response.getheaders(),
                stream=_PageStream(connection, response, deadline), request=request)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            if connection is not None:
                connection.close()
            if isinstance(exc, WebImageError):
                raise
            raise WebImageError('web_image_network_failed') from exc


@dataclass(slots=True)
class PageImage:
    url: str
    alt: str = ''
    context: str = ''
    link_url: str = ''
    width: str = ''
    height: str = ''


def _url(base, value):
    if not isinstance(value, str) or not value.strip():
        return ''
    try:
        value = public_web_url(urljoin(base, value.strip()))
        return value if len(value) <= 2000 else ''
    except (ValueError, TypeError):
        return ''


def _original(url):
    # MediaWiki's thumb/hash/hash/name/300px-name path names the original
    # at hash/hash/name. No guessing of unrelated filenames or sites.
    parts = urlsplit(url)
    path = re.sub(r'/thumb/([^/]+/[^/]+/[^/]+)/[^/]+$', r'/\1', parts.path)
    return parts._replace(path=path).geturl()


class _Images(HTMLParser):
    def __init__(self, url):
        super().__init__(convert_charrefs=True)
        self.url, self.images, self.links = url, [], []
        self.seen, self.text = {}, ''
        self.anchor, self.anchor_text, self.suppressed = '', '', 0
        self.current_image = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {'script', 'style', 'noscript'}:
            self.suppressed += 1
        if self.suppressed:
            return
        if tag == 'a':
            self.anchor, self.anchor_text = _url(self.url, attrs.get('href', '')), ''
        if tag == 'img':
            values = [attrs.get(key, '') for key in ('data-original', 'data-src', 'data-lazy-src', 'src')]
            srcset = attrs.get('data-srcset') or attrs.get('srcset', '')
            variants = [item.strip().split()[0] for item in srcset.split(',') if item.strip()]
            values = variants[-1:] + values
            target = next((candidate for value in values if value and (candidate := _url(self.url, value))), '')
            file_page = unquote(urlsplit(self.anchor).path).rsplit('/', 1)[-1].lower().startswith(('file:', '文件:', '檔案:', 'image:'))
            if self.anchor and _IMAGE_PATH.search(self.anchor) and not file_page:
                target = self.anchor
            if not target:
                return
            target = _original(target)
            alt = (attrs.get('alt') or attrs.get('title') or unquote(urlsplit(target).path.rsplit('/', 1)[-1]))[:240]
            if target in self.seen:
                self.current_image = self.seen[target]
                if alt not in self.current_image.context:
                    self.current_image.context = (self.current_image.context + ' ' + alt)[-600:]
                return
            if len(self.images) >= MAX_CANDIDATES:
                return
            image = PageImage(target, alt, self.text[-350:], self.anchor, (attrs.get('width') or '')[:10], (attrs.get('height') or '')[:10])
            self.images.append(image)
            self.seen[target] = self.current_image = image
        if tag == 'meta' and (attrs.get('property') or attrs.get('name')) in {'og:image', 'twitter:image'}:
            target = _url(self.url, attrs.get('content', ''))
            if target and target not in self.seen and len(self.images) < MAX_CANDIDATES:
                image = PageImage(target, context='页面分享封面')
                self.images.append(image)
                self.seen[target] = image

    def handle_endtag(self, tag):
        if tag in {'script', 'style', 'noscript'}:
            self.suppressed = max(0, self.suppressed - 1)
        if tag == 'a':
            if self.anchor and len(self.links) < 60 and self.anchor_text.strip():
                self.links.append({'url': self.anchor, 'text': self.anchor_text.strip()[:100]})
            self.anchor = ''

    def handle_data(self, data):
        if self.suppressed:
            return
        text = ' '.join(data.split())
        if not text:
            return
        self.text = (self.text + ' ' + text)[-600:]
        if self.anchor:
            self.anchor_text = (self.anchor_text + ' ' + text)[-100:]
        if self.current_image is not None:
            self.current_image.context = (self.current_image.context + ' ' + text)[:600]


def parse_page_images(html: str, url: str) -> dict:
    parser = _Images(url)
    parser.feed(html)
    return {'page_url': url, 'images': [asdict(item) for item in parser.images],
        'page_links': parser.links, 'candidate_limit_reached': len(parser.images) >= MAX_CANDIDATES}


class WebImageClient:
    def __init__(self, web_client=None, *, http_client=None):
        self.reader = getattr(web_client, 'page_reader', None) or web_client
        self.http_client = http_client  # Explicit injection for isolated tests.

    def fetch(self, url, *, image=False, video=False, output=None, referer='', check=lambda: None, deadline=None, byte_limit=MAX_IMAGE_BYTES):
        limit = 90 if video else 30
        deadline = min(deadline or time.monotonic() + limit, time.monotonic() + limit)
        cap = min(MAX_VIDEO_BYTES if video else MAX_IMAGE_BYTES, byte_limit) if image or video else MAX_PAGE_BYTES
        if self.http_client is None:
            proxy = getattr(self.reader, 'page_proxy_url', '')
            transport = (ValidatedPageProxyTransport(proxy, self.reader._proxy_addresses)
                if proxy else PublicImageTransport())
            client = httpx.Client(transport=transport, trust_env=False)
        else:
            client = self.http_client
        try:
            for _ in range(5):
                check()
                url = public_web_url(url)
                headers = {'Accept': 'video/mp4' if video else 'image/*' if image else 'text/html,application/xhtml+xml',
                    'User-Agent': 'Xiaomachi-WebReader/1.0'}
                if _xiaohongshu(url):
                    headers['User-Agent'] = ('Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) '
                        'AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
                        if '/discovery/item/' in urlsplit(url).path else
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                        '(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36')
                if referer:
                    # Only the page origin; query strings can contain user tokens.
                    origin = urlsplit(public_web_url(referer))
                    headers['Referer'] = f'{origin.scheme}://{origin.netloc}/'
                with client.stream('GET', url, headers=headers, follow_redirects=False,
                        timeout=remaining(deadline), extensions={'page_deadline': deadline}) as response:
                    if response.is_redirect:
                        location = response.headers.get('location')
                        if not location:
                            raise WebImageError('web_image_redirect_invalid')
                        url = urljoin(url, location)
                        continue
                    response.raise_for_status()
                    kind = response.headers.get('content-type', '').split(';')[0].lower()
                    if not image and not video and kind not in {'text/html', 'application/xhtml+xml', ''}:
                        raise WebImageError('web_image_page_not_html')
                    length = response.headers.get('content-length')
                    if length and (not length.isdigit() or int(length) > cap):
                        raise WebImageError('web_image_size_budget')
                    body = bytearray()
                    size = 0
                    for chunk in response.iter_bytes(chunk_size=65536):
                        check()
                        remaining(deadline)
                        size += len(chunk)
                        if size > cap:
                            raise WebImageError('web_video_size_budget' if video else 'web_image_size_budget')
                        if output is not None:
                            output.write(chunk)
                        else:
                            body.extend(chunk)
                    check()
                    return url, bytes(body), response.encoding or 'utf-8'
            raise WebImageError('web_image_redirect_limit')
        finally:
            if self.http_client is None:
                client.close()

    def list_images(self, url, *, check=lambda: None):
        target, content, encoding = self.fetch(url, check=check)
        html = content.decode(encoding, errors='replace')
        if _xiaohongshu(target) or _xiaohongshu(url):
            try:
                result = parse_xiaohongshu_images(html, target, requested_url=url)
            except WebImageError as exc:
                original = urlsplit(url)
                # The public mobile share route may include server-rendered
                # note data when the desktop shell only offers a login page.
                # Same note/token, one bounded retry, no cookies or API signing.
                if str(exc) != 'web_image_note_unavailable' or not _xiaohongshu(url) or not original.path.startswith('/explore/'):
                    raise
                mobile = original._replace(path=original.path.replace('/explore/', '/discovery/item/', 1)).geturl()
                target, content, encoding = self.fetch(mobile, check=check)
                result = parse_xiaohongshu_images(content.decode(encoding, errors='replace'), target, requested_url=url)
        else:
            result = parse_page_images(html, target)
        if not result['images']:
            raise WebImageError('web_image_page_no_images')
        return result

    def video(self, url, *, check=lambda: None):
        from app.providers.web_videos import parse_note_video
        target, content, encoding = self.fetch(url, check=check)
        if not _xiaohongshu(target):
            raise WebImageError('web_video_site_unsupported')
        try:
            note = parse_xiaohongshu_note(content.decode(encoding, errors='replace'), target, requested_url=url)
        except WebImageError as exc:
            original = urlsplit(url)
            if str(exc) != 'web_image_note_unavailable' or not _xiaohongshu(url) or not original.path.startswith('/explore/'):
                raise
            mobile = original._replace(path=original.path.replace('/explore/', '/discovery/item/', 1)).geturl()
            target, content, encoding = self.fetch(mobile, check=check)
            note = parse_xiaohongshu_note(content.decode(encoding, errors='replace'), target, requested_url=url)
        return parse_note_video(note, target)


def _xiaohongshu(url):
    host = (urlsplit(url).hostname or '').lower()
    return host == 'xiaohongshu.com' or host.endswith('.xiaohongshu.com')


def parse_xiaohongshu_note(html: str, url: str, *, requested_url='') -> dict:
    """Parse only this note's embedded public state, never execute JavaScript."""
    match = re.search(r'window\.__INITIAL_STATE__\s*=\s*(.*?)</script\s*>', html, re.S | re.I)
    identifier = re.search(r'/(?:explore|discovery/item)/([a-f0-9]{24})(?:[/?#]|$)', url)
    identifier = identifier or re.search(r'/(?:explore|discovery/item)/([a-f0-9]{24})(?:[/?#]|$)', requested_url)
    if not match or not identifier:
        raise WebImageError('web_image_note_unavailable')
    # undefined is the only non-JSON token accepted. Quoted strings remain
    # byte-for-byte intact; eval/JS engines and network API signing are absent.
    state = re.sub(r'"(?:\\.|[^"\\])*"|\bundefined\b',
        lambda item: 'null' if item.group() == 'undefined' else item.group(), match[1])
    try:
        state, _ = json.JSONDecoder().raw_decode(state)
        notes = state.get('note', {}).get('noteDetailMap', {})
        note = notes.get(identifier[1], {}).get('note')
        if not isinstance(note, dict):
            note = state.get('noteData', {}).get('data', {}).get('noteData', {})
        if not isinstance(note, dict) or str(note.get('noteId') or note.get('id') or '') != identifier[1]:
            raise WebImageError('web_image_note_unavailable')
        return note
    except (ValueError, TypeError, AttributeError, RecursionError) as exc:
        if isinstance(exc, WebImageError):
            raise
        raise WebImageError('web_image_note_state_invalid') from exc


def parse_xiaohongshu_images(html: str, url: str, *, requested_url='') -> dict:
    note = parse_xiaohongshu_note(html, url, requested_url=requested_url)
    try:
        if note.get('type') == 'video':
            raise WebImageError('web_image_video_has_no_album')
        items = note.get('imageList')
        if not isinstance(items, list) or not items:
            raise WebImageError('web_image_note_no_images')
        images, seen = [], set()
        for index, item in enumerate(items[:MAX_CANDIDATES]):
            if not isinstance(item, dict):
                continue
            infos = item.get('infoList') if isinstance(item.get('infoList'), list) else []
            values = [item.get('urlDefault'), item.get('url'), *(info.get('url') for info in infos
                if isinstance(info, dict) and info.get('imageScene') in {'WB_DFT', 'H5_DTL'}), item.get('urlPre')]
            target = next((target for value in values if value and (target := _url(url, value))), '')
            if not target or target in seen:
                continue
            seen.add(target)
            images.append(asdict(PageImage(target, f'笔记配图{index + 1}', str(note.get('title') or '')[:240],
                '', str(item.get('width') or '')[:10], str(item.get('height') or '')[:10])))
        if not images:
            raise WebImageError('web_image_note_no_images')
        return {'page_url': url, 'images': images, 'page_links': [],
            'candidate_limit_reached': len(items) > MAX_CANDIDATES, 'album': True,
            'declared_image_count': len(items), 'title': str(note.get('title') or '')[:240]}
    except (ValueError, TypeError, AttributeError, RecursionError) as exc:
        if isinstance(exc, WebImageError):
            raise
        raise WebImageError('web_image_note_state_invalid') from exc
