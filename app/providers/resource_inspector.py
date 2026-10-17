"""Bounded public-post metadata; mature parsers never own downloads or sessions.

Public extractor requests use the existing public DNS-pinned GET transport.
Configured XHS notes can use the container-local metadata worker; CDN bytes
remain credential-free. No playlists, arbitrary user config, or LLM is used.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, UTC
from html.parser import HTMLParser
import io
import json
import re
import time
from urllib.parse import urlencode, urlsplit, parse_qs

import httpx

from app.core.url_policy import public_web_url
from app.core.video_file import MAX_VIDEO_BYTES, MAX_VIDEO_SECONDS
from app.providers.web_images import WebImageClient, WebImageError, parse_page_images

_NOISE = re.compile(r'(?:favicon|apple-touch-icon|/ico/|/avatars?/|/avatar[._-]|/logo[._-])', re.I)
_ERROR = re.compile(r'页面不见了|当前笔记暂时无法浏览|访问验证|Just a moment|Access Denied|Page Not Found', re.I)


class _Meta(HTMLParser):
    def __init__(self):
        super().__init__(); self.values = {}; self.in_title = False; self.title = ''

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'title':self.in_title = True
        if tag == 'meta' and a.get('content'):
            self.values[a.get('property', a.get('name', ''))] = a['content']

    def handle_endtag(self, tag):
        if tag == 'title':self.in_title = False

    def handle_data(self, data):
        if self.in_title:self.title += data


def _date(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:return datetime.fromtimestamp(value, UTC).isoformat()
        except (ValueError, OverflowError, OSError):return ''
    return str(value or '')[:80]


class PublicResourceClient:
    """One instance per inspection; at most 12 bounded reads / 25 seconds."""
    def __init__(self, web_client=None, *, reader=None):
        self.reader = reader or WebImageClient(web_client)

    def inspect(self, url: str, *, check=lambda: None, deadline=None) -> dict:
        check(); url = public_web_url(url)
        self.deadline = min(deadline or time.monotonic() + 25, time.monotonic() + 25)
        self.calls = 0
        host, path = urlsplit(url).hostname, urlsplit(url).path
        if host in {'b23.tv', 'bili2233.cn'}:
            target, _, _ = self._read(url, check)
            if urlsplit(target).hostname not in {'bilibili.com', 'www.bilibili.com'}:
                raise WebImageError('web_resource_redirect_invalid')
            return self.inspect(target, check=check, deadline=self.deadline)
        result = {'source_url': url, 'site': host, 'post_id': '', 'author': '', 'published_at': '',
            'title': '', 'text': '', 'media': [], 'identity_status': 'unverified'}
        if host in {'x.com', 'twitter.com', 'www.x.com', 'www.twitter.com'}:
            from app.providers.twitter_media import TwitterMediaDownloader
            async def inspect_x():
                async with httpx.AsyncClient(proxy=getattr(getattr(self.reader, 'reader', None), 'page_proxy_url', '') or None,
                        trust_env=False, follow_redirects=False) as client:
                    return await TwitterMediaDownloader(client, timeout_seconds=25, check=check).inspect(url)
            result.update(asyncio.run(inspect_x()))
        elif host in {'bgm.tv', 'bangumi.tv', 'chii.in'} and (m := re.fullmatch(r'/subject/(\d+)/?', path)):
            d = self._json('https://api.bgm.tv/v0/subjects/' + m[1], check)
            if str(d.get('id')) != m[1]:raise WebImageError('web_resource_id_mismatch')
            result.update(post_id=m[1], title=d.get('name_cn') or d.get('name') or '',
                published_at=_date(d.get('date')), text=str(d.get('summary') or '')[:2000], identity_status='matched')
            if (d.get('images') or {}).get('large'):
                result['media'] = [{'kind': 'image', 'role': 'cover', 'url': d['images']['large']}]
        elif host in {'fantia.jp', 'www.fantia.jp'} and (m := re.fullmatch(r'/posts/(\d+)/?', path)):
            files = self._gallery(url, check, self.deadline)
            for item in files:
                if str(item.get('post_id')) != m[1]:raise WebImageError('web_resource_id_mismatch')
                result.update(post_id=m[1], identity_status='matched', title=str(item.get('post_title') or result['title'])[:240],
                    author=str(item.get('fanclub_user_name') or result['author'])[:160],
                    published_at=_date(item.get('posted_at') or result['published_at']))
                role = 'cover' if item.get('content_category') == 'thumb' else 'post_image'
                if re.search(r'\.(?:jpg|jpeg|png|webp|gif)(?:$|\?)', item['url'], re.I):
                    result['media'].append({'kind': 'image', 'role': role, 'url': item['url']})
            result['extractor'] = 'gallery-dl'
        elif host in {'www.bilibili.com', 'bilibili.com'} and (m := re.fullmatch(r'/video/(BV\w+|av\d+)/?', path)):
            result.update(self._bilibili(url, m[1], check))
        elif host in {'www.douyin.com', 'douyin.com'} and (m := re.fullmatch(r'/video/(\d+)/?', path)):
            d = self._yt(url, check, self.deadline)
            if str(d.get('id')) != m[1]:raise WebImageError('web_resource_id_mismatch')
            result.update(post_id=m[1], identity_status='matched', title=str(d.get('title') or '')[:240],
                author=str(d.get('uploader') or '')[:160], published_at=_date(d.get('timestamp')),
                media=self._progressive(d), extractor='yt-dlp')
        else:
            note = None
            if host in {'xiaohongshu.com', 'www.xiaohongshu.com', 'xhslink.com', 'www.xhslink.com'} and hasattr(self.reader, 'read_note'):
                note, target, text = self.reader.read_note(url, check=check, deadline=self.deadline)
                host = urlsplit(target).hostname
                result['site'] = host
            else:
                target, b, encoding = self._read(url, check)
                text = b.decode(encoding, errors='replace')
            meta = _Meta(); meta.feed(text)
            if _ERROR.search(meta.title) or '/404' in urlsplit(target).path:
                raise WebImageError('web_resource_unavailable')
            if not meta.title.strip() and note is None:raise WebImageError('web_resource_dynamic_or_empty')
            result.update(source_url=target, title=meta.values.get('og:title', meta.title)[:240],
                author=meta.values.get('author', '')[:160],
                published_at=_date(meta.values.get('article:published_time')),
                text=meta.values.get('og:description', meta.values.get('description', ''))[:2000])
            if host and host.endswith('xiaohongshu.com'):
                from app.providers.web_images import parse_xiaohongshu_note, parse_xiaohongshu_images
                note = note or parse_xiaohongshu_note(text, target, requested_url=url)
                result.update(post_id=str(note.get('noteId') or note.get('id')), identity_status='matched',
                    title=str(note.get('title') or '')[:240], text=str(note.get('desc') or '')[:2000],
                    author=str((note.get('user') or {}).get('nickname') or '')[:160],
                    published_at=_date(note.get('time') / 1000 if isinstance(note.get('time'), int) else None))
                if note.get('type') == 'video':
                    from app.providers.web_videos import parse_note_video
                    result['media'] = [{'kind': 'video', 'role': 'post_video', **parse_note_video(note, target)}]
                else:
                    result['media'] = [{'kind': 'image', 'role': 'post_image', 'url': i['url']}
                        for i in parse_xiaohongshu_images(text, target, requested_url=url)['images']]
            else:
                result['media'] = [{'kind': 'image', 'role': 'page_image', 'url': i['url']}
                    for i in parse_page_images(text, target)['images'] if not _NOISE.search(i['url'])]
        check()
        result['media'] = [{**i, 'url': public_web_url(i['url'])} for i in result['media'][:20]
                           if isinstance(i.get('url'), str) and len(i['url']) <= 4096]
        result['media_count'] = len(result['media'])
        result['date_status'] = 'source_reported' if result['published_at'] else 'unknown'
        result['author_status'] = 'source_reported' if result['author'] else 'unknown'
        result['coverage'] = 'returned_public_metadata_only; completeness_and_latest_not_proven'
        return result

    def _read(self, url, check, headers=None):
        check(); url = public_web_url(url); self.calls += 1
        if self.calls > 12 or time.monotonic() >= self.deadline:
            raise WebImageError('web_resource_request_budget')
        return self.reader.fetch(url, check=check, deadline=self.deadline, metadata=True,
            extra_headers=headers or {}, referer=url)

    def _json(self, url, check):
        _, b, _ = self._read(url, check)
        try:
            data = json.loads(b)
            if not isinstance(data, dict):raise ValueError()
            return data
        except (ValueError, UnicodeError) as exc:raise WebImageError('web_resource_invalid_metadata') from exc

    def _yt(self, url, check, deadline):
        try:
            from yt_dlp import YoutubeDL
            from yt_dlp.networking import Request
            from yt_dlp.networking.common import Response
        except ImportError as exc:raise WebImageError('web_resource_extractor_unavailable') from exc
        owner = self
        class SafeDL(YoutubeDL):
            def urlopen(self, req):
                if isinstance(req, str):req = Request(req)
                if req.method != 'GET' or req.data is not None:
                    raise WebImageError('web_resource_get_only')
                target, data, encoding = owner._read(req.url, check, dict(req.headers))
                return Response(io.BytesIO(data), target, {'Content-Type': 'text/html; charset=' + encoding})
        class Quiet:
            def debug(self, msg):pass
            def warning(self, msg):pass
            def error(self, msg):pass
        try:
            with SafeDL({'quiet': True, 'logger': Quiet(), 'cachedir': False, 'skip_download': True,
                         'noplaylist': True, 'playlist_items': '1', 'retries': 0, 'extractor_retries': 0,
                         'socket_timeout': 8}) as y:
                d = y.extract_info(url, download=False, process=False)
                if d and d.get('entries'):d = next(iter(d['entries']), None)
                if not isinstance(d, dict):raise WebImageError('web_resource_unavailable')
                return d
        except (httpx.HTTPError, ValueError):raise
        except Exception as exc:
            # Never return a library exception containing URLs/cookies/payloads.
            code = ('web_resource_session_required' if re.search(r'fresh cookies|login|sign.in|403|captcha', str(exc), re.I)
                    else 'web_resource_extraction_failed')
            raise WebImageError(code) from exc

    def _gallery(self, url, check, deadline):
        try:
            from gallery_dl import extractor
            import requests
        except ImportError as exc:raise WebImageError('web_resource_extractor_unavailable') from exc
        e = extractor.find(url)
        if e is None or e.category != 'fantia':raise WebImageError('web_resource_site_unsupported')
        def request(target, **kwargs):
            if kwargs.get('method', 'GET') != 'GET' or kwargs.get('data') or kwargs.get('json'):
                raise WebImageError('web_resource_get_only')
            if kwargs.get('params'):target += ('&' if '?' in target else '?') + urlencode(kwargs['params'])
            target, b, enc = self._read(target, check, kwargs.get('headers'))
            r = requests.Response();r.status_code = 200;r.url = target;r._content = b;r.encoding = enc
            return r
        # No global config mutations: concurrent group/private jobs stay isolated.
        e.config = lambda key, default=None: default
        e.request = request
        files = []
        try:
            for message in e:
                check()
                if message[0] == 3:
                    files.append({'url': message[1], **{k: message[2].get(k) for k in
                        ('post_id', 'post_title', 'fanclub_user_name', 'posted_at', 'content_category')}})
                    if len(files) >= 20:break
            return files
        finally:
            if e.session is not None:e.session.close()

    @staticmethod
    def _progressive(d):
        candidates = []
        for f in d.get('formats', [])[:100]:
            if (f.get('ext') != 'mp4' or f.get('acodec') in {None, 'none'}
                    or not str(f.get('vcodec', '')).startswith(('avc', 'h264')) or f.get('has_drm')
                    or f.get('protocol') not in {None, 'https', 'http'}):continue
            size, duration = f.get('filesize') or 0, d.get('duration') or 0
            if size > MAX_VIDEO_BYTES or duration > MAX_VIDEO_SECONDS:continue
            candidates.append({'kind': 'video', 'role': 'post_video', 'url': f['url'],
                'declared_bytes': size, 'duration_seconds': duration, 'width': f.get('width') or 0, 'height': f.get('height') or 0})
        return sorted(candidates, key=lambda x: x['width'] * x['height'], reverse=True)[:1]

    def _bilibili(self, url, identifier, check):
        query = 'aid=' + identifier[2:] if identifier.startswith('av') else 'bvid=' + identifier
        d = self._json('https://api.bilibili.com/x/web-interface/view?' + query, check)
        d = d.get('data') if d.get('code') == 0 else None
        if not isinstance(d, dict):raise WebImageError('web_resource_unavailable')
        matches = str(d.get('aid')) == identifier[2:] if identifier.startswith('av') else d.get('bvid') == identifier
        if not matches:
            raise WebImageError('web_resource_id_mismatch')
        try:
            page = int(parse_qs(urlsplit(url).query).get('p', ['1'])[0])
            if not 1 <= page <= 1000:raise ValueError()
            part = (d.get('pages') or [d])[page - 1]
        except (ValueError, TypeError, IndexError) as exc:raise WebImageError('web_resource_invalid_part') from exc
        try:
            info = self._yt(url, check, self.deadline)
        except (WebImageError, httpx.HTTPError):
            # A valid primary ID can still expose a progressive stream when
            # the extractor encounters a changed DASH response or old fixture.
            check();info = {'id': d['bvid'], 'formats': []}
        if str(info.get('id', '')).split('_p')[0] != d.get('bvid'):
            raise WebImageError('web_resource_id_mismatch')
        returned_part = re.search(r'_p(\d+)$', str(info.get('id', '')))
        if returned_part and int(returned_part[1]) != page:
            raise WebImageError('web_resource_id_mismatch')
        media = self._progressive(info)
        if not media:
            # A returned progressive stream preserves audio without installing a
            # browser/ffmpeg download loop or treating DASH video-only as complete.
            q = f"avid={d['aid']}&cid={part['cid']}&qn=16&fnval=1&platform=html5"
            play = self._json('https://api.bilibili.com/x/player/playurl?' + q, check)
            streams = (play.get('data') or {}).get('durl') if play.get('code') == 0 else []
            duration = part.get('duration') or info.get('duration') or d.get('duration') or 0
            if isinstance(streams, list) and len(streams) == 1 and duration <= MAX_VIDEO_SECONDS:
                item = streams[0]
                if (item.get('size') or 0) <= MAX_VIDEO_BYTES:
                    media = [{'kind': 'video', 'role': 'post_video', 'url': item['url'],
                        'declared_bytes': item.get('size') or 0, 'duration_seconds': duration}]
        return {'post_id': str(d['bvid']), 'title': str(d.get('title') or '')[:240],
            'author': str((d.get('owner') or {}).get('name') or '')[:160], 'published_at': _date(d.get('pubdate')),
            'text': str(d.get('desc') or '')[:2000], 'identity_status': 'matched', 'media': media, 'extractor': 'yt-dlp'}
