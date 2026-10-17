"""Stable source identity excludes expiring XHS locator tokens."""
import re
import hashlib
from urllib.parse import urlsplit


def canonical_media_source(value):
    if not isinstance(value,str):return value
    parts=urlsplit(value);host=parts.hostname or ''
    if host in {'x.com','www.x.com','twitter.com','www.twitter.com','mobile.twitter.com'}:
        match=re.search(r'/status/([1-9][0-9]*)',parts.path)
        if match:return 'https://x.com/i/status/'+match[1]
    if host=='xiaohongshu.com' or host.endswith('.xiaohongshu.com'):
        match=re.search(r'/(?:explore|discovery/item)/([a-f0-9]{24})(?:/|$)',parts.path)
        if match:return 'https://www.xiaohongshu.com/explore/'+match[1]
    return value


def media_source_key(value):
    if not isinstance(value,str) or value.startswith(('x:','xhs:','source:')):return value
    canonical=canonical_media_source(value)
    parts=urlsplit(canonical)
    if parts.hostname=='x.com' and '/status/' in parts.path:return 'x:'+parts.path.rsplit('/',1)[-1]
    if parts.hostname=='www.xiaohongshu.com' and '/explore/' in parts.path:return 'xhs:'+parts.path.rsplit('/',1)[-1]
    return 'source:'+(parts.hostname or 'unknown')+':'+hashlib.sha256(canonical.encode()).hexdigest()
