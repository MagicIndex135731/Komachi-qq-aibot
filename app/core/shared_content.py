"""Bounded display metadata from untrusted OneBot share/Ark segments."""
from __future__ import annotations

from dataclasses import dataclass
import html
import json
import re
from xml.etree import ElementTree as ET

from app.core.url_policy import public_web_url

MAX_SEGMENTS, MAX_RESOURCES, MAX_PAYLOAD = 100, 8, 65536
_URL = re.compile(r'https?://[^\s<>"\[\]]{1,4090}')
_URL_KEYS = ('qqdocurl', 'jumpUrl', 'jump_url', 'url', 'targetUrl')


@dataclass(frozen=True, slots=True)
class SharedLink:
    url: str
    title: str = ''
    description: str = ''
    kind: str = 'web'


def _clean(value, limit=240):
    return ''.join(char for char in str(value or '') if ord(char) >= 32)[:limit]


def _url(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 4096:
        return None
    try:
        return public_web_url(html.unescape(value))
    except (ValueError, TypeError):
        return None


def _card_fields(card):
    if not isinstance(card, dict) or card.get('app') == 'com.tencent.multimsg':
        return []
    # Explicit schema keys exclude thumbnail URLs, opaque IDs and arbitrary
    # embedded JSON. Cards are display quotations, never trusted instructions.
    result, pending, visited = [], [(card.get('meta', {}), 0)], 0
    while pending and visited < 80:
        value, depth = pending.pop(0)
        visited += 1
        if not isinstance(value, dict) or depth > 4:
            continue
        urls = [_url(value.get(key)) for key in _URL_KEYS]
        for url in urls:
            if url:
                result.append(SharedLink(url, _clean(value.get('title') or value.get('prompt')),
                                         _clean(value.get('desc') or value.get('summary'))))
        pending.extend((child, depth + 1) for child in list(value.values())[:20] if isinstance(child, dict))
    return result[:MAX_RESOURCES]


def extract_shared_links(message) -> tuple[SharedLink, ...]:
    if isinstance(message, str):
        message = [{'type': 'text', 'data': {'text': message[:MAX_PAYLOAD]}}]
    if not isinstance(message, list):
        return ()
    links = {}
    for item in message[:MAX_SEGMENTS]:
        if not isinstance(item, dict) or not isinstance(item.get('data'), dict):
            continue
        kind, data = item.get('type'), item['data']
        found = []
        if kind == 'text':
            text = data.get('text', '')
            if isinstance(text, str):
                found = [SharedLink(url) for value in _URL.findall(text[:MAX_PAYLOAD])[:MAX_RESOURCES]
                         if (url := _url(value.rstrip('。，、；！?？)）')))]
        elif kind == 'share':
            url = _url(data.get('url'))
            if url:
                found = [SharedLink(url, _clean(data.get('title')), _clean(data.get('content')))]
        elif kind in {'json', 'xml'}:
            value = data.get('data')
            if isinstance(value, str) and len(value) <= MAX_PAYLOAD:
                try:
                    if kind == 'json':
                        found = _card_fields(json.loads(value))
                    elif '<!DOCTYPE' not in value.upper() and '<!ENTITY' not in value.upper():
                        tree = ET.fromstring(value)
                        for element in list(tree.iter())[:80]:
                            if element.tag not in {'msg', 'item', 'url', 'news', 'link'}:
                                continue
                            url = _url(element.attrib.get('url')) or (_url(element.text) if element.tag == 'url' else None)
                            if url:
                                found.append(SharedLink(url, _clean(tree.findtext('.//title')),
                                                        _clean(tree.findtext('.//summary'))))
                except (ValueError, ET.ParseError, RecursionError):
                    pass
            elif kind == 'json' and isinstance(value, dict):
                try:
                    if len(json.dumps(value)) <= MAX_PAYLOAD:
                        found = _card_fields(value)
                except (ValueError, RecursionError):
                    pass
        for link in found[:MAX_RESOURCES]:
            if len(links) < MAX_RESOURCES:
                previous = links.get(link.url)
                if previous is None or not previous.title and link.title:
                    links[link.url] = link
    return tuple(links.values())


def segment_display(item):
    if not isinstance(item, dict):
        return ''
    if item.get('type') not in {'share', 'json', 'xml'}:
        return ''
    links = extract_shared_links([item])
    return '\n'.join('[分享卡片：' + (link.title or '外链') + '] '
                     + (link.description + ' ' if link.description else '') + link.url for link in links)
