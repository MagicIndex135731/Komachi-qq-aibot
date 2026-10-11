"""Bounded, credential-free public text fallback for Bilibili share links."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import re
import time
from urllib.parse import urljoin, urlsplit

import httpx

from app.core.url_policy import public_web_url

_SHORT_HOSTS = {'b23.tv', 'www.b23.tv'}
_PAGE_HOSTS = {'bilibili.com', 'www.bilibili.com', 'm.bilibili.com'}
_VIDEO_PATH = re.compile(r'/video/(BV[0-9A-Za-z]{10})/?')
_SHORT_PATH = re.compile(r'/[0-9A-Za-z]{1,32}/?')
_HEADERS = {'User-Agent': 'Mozilla/5.0', 'Authorization': '', 'Cookie': ''}


@dataclass(slots=True)
class SharedPageText:
    title: str
    url: str
    lines: list[str]


def _allowed_url(value: str) -> str:
    value = public_web_url(value)
    p = urlsplit(value)
    if p.hostname not in _SHORT_HOSTS | _PAGE_HOSTS or p.port not in {None, 443}:
        raise ValueError('shared_text_target_unsupported')
    if p.hostname in _SHORT_HOSTS:
        if not _SHORT_PATH.fullmatch(p.path):
            raise ValueError('shared_text_target_unsupported')
        return f'https://{p.hostname}{p.path}'
    if not _VIDEO_PATH.fullmatch(p.path):
        raise ValueError('shared_text_target_unsupported')
    return f'https://www.bilibili.com{p.path.rstrip("/")}'


def read_bilibili_text(url: str, *, client: httpx.Client, deadline: float) -> SharedPageText | None:
    """Only fixed first-party hosts and a fixed JSON metadata API are fetched.

    Unsupported/failed metadata returns None so the original page error stays
    authoritative. No cookies, media, subtitles, login or arbitrary redirects.
    """
    def remaining() -> float:
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise ValueError('shared_text_budget_exhausted')
        return min(10.0, budget)

    try:
        target = _allowed_url(url)
        for _ in range(2):
            if urlsplit(target).hostname not in _SHORT_HOSTS:
                break
            with client.stream('GET', target, headers=_HEADERS, auth=None,
                    follow_redirects=False, timeout=remaining()) as response:
                if response.status_code not in {301, 302, 303, 307, 308}:
                    return None
                location = response.headers.get('location', '')
            if not location:
                return None
            target = _allowed_url(urljoin(target, location))
        match = _VIDEO_PATH.fullmatch(urlsplit(target).path)
        if urlsplit(target).hostname not in _PAGE_HOSTS or match is None:
            return None
        bvid = match.group(1)
        with client.stream('GET', 'https://api.bilibili.com/x/web-interface/view',
                params={'bvid': bvid}, headers=_HEADERS, auth=None,
                follow_redirects=False, timeout=remaining()) as response:
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes():
                remaining()
                body.extend(chunk)
                if len(body) > 512 * 1024:
                    return None
        payload = json.loads(body)
        if not isinstance(payload, dict) or payload.get('code') != 0:
            return None
        data = payload.get('data')
        if not isinstance(data, dict) or data.get('bvid') != bvid:
            return None
        def text(value, size):
            return re.sub(r'\s+', ' ', value).strip()[:size] if isinstance(value, str) else ''
        title = text(data.get('title'), 500)
        if not title:
            return None
        desc = text(data.get('desc'), 6000)
        owner = data.get('owner')
        author = text(owner.get('name'), 120) if isinstance(owner, dict) else ''
        lines = ['读取范围：B站公开文字元数据；未观看视频，未读取音频、画面或字幕。',
            f'标题：{title}', f'UP主：{author or "未提供"}',
            f'简介：{desc or "简介为空，不能据此判断视频里的具体内容。"}']
        stamp = data.get('pubdate')
        if type(stamp) is int and 0 <= stamp <= 253402300799:
            lines.append('发布时间（UTC）：' + datetime.fromtimestamp(stamp, UTC).isoformat())
        duration = data.get('duration')
        if type(duration) is int and 0 <= duration <= 10**7:
            lines.append(f'时长：{duration}秒（仅元数据）')
        lines.append('信息不足时按标题和UP主搜索补充，需区分同一条分享与相关背景资料。')
        return SharedPageText(title, f'https://www.bilibili.com/video/{bvid}', lines)
    except (httpx.HTTPError, ValueError, UnicodeError, OverflowError, OSError):
        return None
