"""Container-local, read-only Xiaohongshu metadata; never download account cookies.

Only this fixed worker can read its private session. CDN downloads keep using
the existing credential-free, DNS-pinned WebImageClient transport.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from urllib.parse import parse_qs, urlencode, urlsplit


WORKER = Path('/usr/local/bin/xhs-readonly')
SESSION = Path('/workspace/data/xhs-crawler/session.json')
FAILURES = frozenset({'invalid_request', 'invalid_action', 'invalid_target', 'browser_unavailable',
    'session_invalid', 'session_storage_failed', 'crawler_busy', 'response_size', 'request_budget', 'api_request_budget',
    'login_required', 'login_timeout', 'login_unavailable', 'login_qr_invalid',
    'challenge_required', 'rate_limited', 'permission_denied', 'note_unavailable',
    'identity_mismatch', 'observation_failed', 'acquisition_failed', 'search_failed'})


class XHSCrawlerError(ValueError):
    """A stable code; never include upstream messages, cookies or browser URLs."""


def note_target(url: str) -> tuple[str, str]:
    try:
        p = urlsplit(url)
        match = re.fullmatch(r'/(?:explore|discovery/item)/([a-f0-9]{24})/?', p.path)
        if (p.scheme != 'https' or p.hostname not in {'www.xiaohongshu.com', 'xiaohongshu.com'}
                or p.username or p.password or p.port not in {None, 443} or not match):
            raise ValueError()
        tokens = parse_qs(p.query, keep_blank_values=True).get('xsec_token', [''])
        if len(tokens) != 1:raise ValueError()
        token = tokens[0]
        if len(token) > 1024 or not re.fullmatch(r'[A-Za-z0-9_+/=-]*', token):raise ValueError()
        return match[1], token
    except (ValueError, TypeError, AttributeError) as exc:
        raise XHSCrawlerError('xhs_invalid_target') from exc


def canonical_url(identifier: str, token: str = '') -> str:
    return 'https://www.xiaohongshu.com/explore/' + identifier + (
        '?' + urlencode({'xsec_token': token, 'xsec_source': 'pc_search'}) if token else '')


class XHSCrawlerClient:
    @staticmethod
    def installed() -> bool:
        return os.name == 'posix' and WORKER.is_file()

    @classmethod
    def configured(cls) -> bool:
        return cls.installed() and SESSION.is_file()

    def acquire(self, request, *, check=lambda: None, deadline=None):
        if request.get('action') not in {'search', 'inspect'}:
            raise XHSCrawlerError('xhs_invalid_action')
        body = json.dumps(request, ensure_ascii=False).encode()
        if len(body) > 8192:raise XHSCrawlerError('xhs_invalid_request')
        from app.core.request_cost import current_request_budget
        budget=current_request_budget.get()
        if budget:budget.reserve_network(32*1024*1024)
        check()
        if not self.configured():raise XHSCrawlerError('xhs_login_required')
        timeout = min(85, max(0, (deadline - time.monotonic()) if deadline else 85))
        if timeout <= 0:raise XHSCrawlerError('xhs_deadline_exceeded')
        env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ') if k in os.environ}
        env.update(HOME='/nonexistent', XDG_CACHE_HOME='/tmp/xhs-worker-cache')
        # No model credentials or ambient proxies reach the browser process.
        # The process group is reaped on timeout, cancellation, or revoked source.
        with subprocess.Popen([str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=env, start_new_session=True) as process:
            try:
                result, _ = process.communicate(body, timeout=timeout)
                check()
            except BaseException as exc:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                if isinstance(exc, subprocess.TimeoutExpired):
                    raise XHSCrawlerError('xhs_deadline_exceeded') from exc
                raise
        if process.returncode or len(result) > 512 * 1024:
            raise XHSCrawlerError('xhs_invalid_response')
        try:
            envelope = json.loads(result)
            if not isinstance(envelope, dict):raise ValueError()
            if envelope.get('ok') is not True:
                code = envelope.get('error')
                raise XHSCrawlerError('xhs_' + (code if code in FAILURES else 'acquisition_failed'))
            if envelope.get('authenticated') is not True:raise ValueError()
            return envelope['data']
        except (ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, XHSCrawlerError):raise
            raise XHSCrawlerError('xhs_invalid_response') from exc

    def read_note(self, url, *, check=lambda: None, deadline=None):
        identifier, token = note_target(url)
        note = self.acquire({'action': 'inspect', 'id': identifier, 'token': token}, check=check, deadline=deadline)
        if (not isinstance(note, dict) or note.get('noteId') != identifier
                or note.get('type') not in {'normal', 'video'}
                or not isinstance(note.get('user'), dict) or not note['user'].get('userId')):
            raise XHSCrawlerError('xhs_identity_mismatch')
        target = canonical_url(identifier, token)
        # An INTERNAL JSON envelope for the existing pure parser. This is never
        # reported as fetched HTML, nor used as a substitute for source evidence.
        state = {'note': {'noteDetailMap': {identifier: {'note': note}}}}
        snapshot = '<script>window.__INITIAL_STATE__=' + json.dumps(state, ensure_ascii=False).replace('<', '\\u003c') + '</script>'
        return note, target, snapshot

    def search_posts(self, query, *, limit=6, sort='relevance', media_type='', check=lambda: None, deadline=None):
        if (not isinstance(query, str) or not 1 <= len(query.strip()) <= 120 or type(limit) is not int
                or not 1 <= limit <= 12 or sort not in {'latest', 'relevance'} or media_type not in {'', 'image', 'video'}):
            raise XHSCrawlerError('xhs_invalid_request')
        rows = self.acquire({'action': 'search', 'query': query.strip(), 'limit': limit,
            'sort': sort, 'media_type': media_type}, check=check, deadline=deadline)
        if not isinstance(rows, list) or len(rows) > limit:raise XHSCrawlerError('xhs_invalid_response')
        posts, seen = [], set()
        for row in rows:
            if not isinstance(row, dict):raise XHSCrawlerError('xhs_invalid_response')
            identifier, token = row.get('id', ''), row.get('xsecToken', '')
            if not isinstance(identifier, str) or not isinstance(token, str):
                raise XHSCrawlerError('xhs_invalid_response')
            card = row.get('noteCard') or {}
            if (row.get('modelType') != 'note' or not isinstance(card, dict)
                    or card.get('type') not in {'normal', 'video'}):continue
            if media_type and card['type'] != ('normal' if media_type == 'image' else 'video'):continue
            url = canonical_url(identifier, token)
            if note_target(url)[0] != identifier:raise XHSCrawlerError('xhs_identity_mismatch')
            if identifier in seen:continue
            seen.add(identifier)
            posts.append({'source_url': url, 'post_id': identifier,
                'title': str(card.get('displayTitle') or '')[:240],
                'author': str((card.get('user') or {}).get('nickname') or '')[:160],
                'note_type': card['type'], 'published_at': '', 'date_status': 'unknown',
                'identity_status': 'candidate', 'media_roles': [], 'status': 'ok'})
        return {'posts': posts, 'visible_posts': len(posts), 'provider': 'xiaohongshu-mcp',
            'coverage': 'visible_search_candidates_only; dates_and_attachments_require_inspection',
            'search_visibility': 'candidates_observed' if posts else 'not_verified'}
