"""Bounded container-local worker transport; credentials never cross its pipes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from urllib.parse import urlsplit

from app.providers.x_posts import XSearchOptions, parse_target, search_target
from app.providers.x_crawler import FAILURE_CODES
from app.providers.twitter_media import TwitterMediaError, parse_post_url


class XCrawlerClient:
    _slots = threading.BoundedSemaphore(1)

    def __init__(self, session_file: str, proxy: str, *, runner=None):
        parts = urlsplit(proxy)
        if (parts.scheme != 'http' or parts.hostname != '127.0.0.1' or not parts.port
                or parts.path not in {'', '/'} or parts.query or parts.fragment
                or parts.username or parts.password or not Path(session_file).is_absolute()):
            raise ValueError('invalid_x_crawler_configuration')
        self.session_file, self.proxy, self.runner = session_file, proxy, runner or subprocess.run

    @classmethod
    def from_settings(cls, settings):
        session_file = getattr(settings, 'x_crawler_session_file', '')
        proxy = getattr(settings, 'x_crawler_proxy_url', 'http://127.0.0.1:18797')
        return cls(session_file, proxy) if session_file else None

    def acquire(self, action, target, *, limit=6, lookback_days=None, check=lambda: None,
            media_types=None,sort=None,min_likes=None,min_replies=None,min_reposts=None,
            subject_terms=None,topic_terms=None,topic_match='keywords'):
        if type(limit) is not int or not 1<=limit<=24 or (lookback_days is not None and (type(lookback_days) is not int or not 1<=lookback_days<=365)):
            raise TwitterMediaError('invalid_x_target')
        parsed = parse_target(target)
        options=XSearchOptions.parse(media_types=media_types,sort=parsed.sort if sort is None else sort,min_likes=min_likes,
            min_replies=min_replies,min_reposts=min_reposts,subject_terms=subject_terms,topic_terms=topic_terms,topic_match=topic_match)
        if (len(options.media_types)==2 and limit<2) or (action!='search' and options!=XSearchOptions()):
            raise TwitterMediaError('invalid_x_target')
        if {'list': 'account', 'search': 'search', 'inspect': 'post'}.get(action) != parsed.kind:
            raise TwitterMediaError('invalid_x_target')
        from app.core.request_cost import current_request_budget
        budget=current_request_budget.get()
        if budget:budget.reserve_network(32*1024*1024)
        check()
        if not self._slots.acquire(timeout=8):
            raise TwitterMediaError('x_crawler_crawler_busy')
        try:
            check()
            root = Path(__file__).resolve().parents[2]
            env = {k: os.environ[k] for k in ('PATH', 'LANG', 'SYSTEMROOT') if k in os.environ}
            env.update(PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1', TWS_TELEMETRY='0',
                HOME='/nonexistent', X_CRAWLER_SESSION_FILE=self.session_file, X_CRAWLER_PROXY_URL=self.proxy)
            result = self.runner([sys.executable, '-B', str(root / 'scripts/x_crawler_worker.py')],
                input=json.dumps({'action': action, 'target': parsed.url, 'limit': limit,
                    'lookback_days': lookback_days,**options.as_dict()}).encode(), capture_output=True, timeout=45,
                cwd=root, env=env)
            if len(result.stdout) > 256 * 1024:
                raise TwitterMediaError('x_crawler_response_size')
            if result.returncode:
                raise TwitterMediaError('x_crawler_unavailable')
            envelope = json.loads(result.stdout)
            if not isinstance(envelope, dict): raise TwitterMediaError('x_crawler_invalid_response')
            if envelope.get('ok') is not True:
                code = envelope.get('error')
                raise TwitterMediaError('x_crawler_' + (code if code in FAILURE_CODES else 'unavailable'))
            data = envelope.get('data')
            check()
            if (not isinstance(data, dict) or data.get('provider') != 'x_crawler'
                    or data.get('authenticated') is not True):
                raise TwitterMediaError('x_crawler_invalid_response')
            rows = data.get('posts')
            if not isinstance(rows, list) or len(rows) > limit:
                raise TwitterMediaError('x_crawler_invalid_response')
            for row in rows:
                if (not isinstance(row, dict) or parse_post_url(row.get('source_url')).post_id != row.get('post_id')
                        or row.get('identity_status') != 'matched' or row.get('protected') is not False):
                    raise TwitterMediaError('x_crawler_identity_mismatch')
                if parsed.kind == 'post' and row['post_id'] != parsed.identifier:
                    raise TwitterMediaError('x_crawler_identity_mismatch')
                if parsed.kind == 'account' and str(row.get('author', '')).lower() != parsed.identifier.lower():
                    raise TwitterMediaError('x_crawler_identity_mismatch')
            return data
        except subprocess.TimeoutExpired as exc:
            raise TwitterMediaError('x_crawler_deadline_exceeded') from exc
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            if isinstance(exc, TwitterMediaError): raise
            raise TwitterMediaError('x_crawler_unavailable') from exc
        finally:
            self._slots.release()

    def inspect(self, url, *, check=lambda: None):
        data = self.acquire('inspect', url, check=check)
        if len(data['posts']) != 1: raise TwitterMediaError('post_not_observed')
        return {**data['posts'][0], 'extractor': data['provider']}

    def list_posts(self, account, *, limit=6, lookback_days=None, check=lambda: None, deadline=None):
        data = self.acquire('list', account, limit=limit, lookback_days=lookback_days, check=check)
        return self._summaries(data, {'account': account})

    def search_posts(self, query, *, limit=12, lookback_days=None, check=lambda: None, deadline=None,
            media_types=None,sort='latest',min_likes=None,min_replies=None,min_reposts=None,
            subject_terms=None,topic_terms=None,topic_match='keywords'):
        target = search_target(query,sort)
        data = self.acquire('search', target.url, limit=limit, lookback_days=lookback_days, check=check,
            media_types=media_types,sort=sort,min_likes=min_likes,min_replies=min_replies,min_reposts=min_reposts,
            subject_terms=subject_terms,topic_terms=topic_terms,topic_match=topic_match)
        return self._summaries(data, {'query': query})

    @staticmethod
    def _summaries(data, selection):
        posts = [{k: v for k, v in row.items() if k != 'media'} | {
            'media_roles': list(dict.fromkeys('post_image' if m['kind'] == 'photo' else 'post_video'
                for m in row.get('media', [])))} for row in data['posts']]
        return {k: v for k, v in data.items() if k not in {'posts', 'acquisition'}} | {
            **selection, 'posts': posts, 'visible_posts': len(posts),
            'newest_visible_at': max((r['published_at'] for r in posts), default=''),
            **({'gap': 'no_verified_posts_in_requested_window'} if not posts else {})}
