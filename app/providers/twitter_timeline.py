"""Public X account discovery, bounded and explicit about incomplete timelines."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
import json
import re
import time
from urllib.parse import urlsplit

from app.core.post_freshness import publication_time
from app.providers.web_images import WebImageClient, WebImageError

_ACCOUNT = re.compile(r'[A-Za-z0-9_]{1,15}\Z')
_POST_ID = re.compile(r'[0-9]{6,20}\Z')
_RESERVED = frozenset({'home','search','i','intent','share','settings','messages',
    'notifications','explore','login','logout','signup'})


def parse_account(value: str) -> str:
    """Accept an account name or exact public profile, never an arbitrary URL."""
    if not isinstance(value, str) or len(value) > 128:
        raise ValueError('invalid_twitter_account')
    account = value.strip()
    if account.startswith(('https://','http://')):
        parts = urlsplit(account)
        if (parts.scheme != 'https' or parts.hostname not in {'x.com','www.x.com','twitter.com','www.twitter.com'}
                or parts.username or parts.password or parts.port or parts.query or parts.fragment):
            raise ValueError('invalid_twitter_account')
        account = parts.path.strip('/')
    account = account.removeprefix('@')
    if not _ACCOUNT.fullmatch(account) or account.lower() in _RESERVED:
        raise ValueError('invalid_twitter_account')
    return account


class _TimelineData(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.active = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('id') == '__NEXT_DATA__':self.active = True

    def handle_endtag(self, tag):
        if tag == 'script':self.active = False

    def handle_data(self, data):
        if self.active:self.parts.append(data)


def parse_timeline(html: str, account: str) -> list[dict]:
    """Read only root account entries; quoted posts and navigation are excluded."""
    account = parse_account(account)
    if not isinstance(html,str) or len(html)>4*1024*1024:
        raise WebImageError('twitter_timeline_invalid_data')
    parser = _TimelineData();parser.feed(html)
    try:
        data = json.loads(''.join(parser.parts))['props']['pageProps']
        reported = data['headerProps']['screenName']
        entries = data['timeline']['entries']
        if not isinstance(reported,str) or reported.lower()!=account.lower() or not isinstance(entries,list):
            raise ValueError()
    except (ValueError, KeyError, TypeError, RecursionError) as exc:
        raise WebImageError('twitter_timeline_invalid_data') from exc
    posts = {}
    for entry in entries[:300]:
        if not isinstance(entry,dict) or entry.get('type')!='tweet':continue
        content = entry.get('content')
        tweet = content.get('tweet') if isinstance(content,dict) else None
        if not isinstance(tweet,dict):continue
        author = tweet.get('user') or {}
        identifier = tweet.get('id_str')
        if (not isinstance(author,dict) or str(author.get('screen_name','')).lower()!=account.lower()
                or not isinstance(identifier,str) or not _POST_ID.fullmatch(identifier)
                or entry.get('entry_id')!='tweet-'+identifier
                or tweet.get('retweeted_status') or tweet.get('in_reply_to_status_id_str')):continue
        entities=tweet.get('extended_entities')
        media = entities.get('media',[]) if isinstance(entities,dict) else []
        if not isinstance(media,list):media=[]
        roles = [('post_image' if item.get('type')=='photo' else 'post_video')
            for item in media[:16] if isinstance(item,dict) and item.get('type') in {'photo','video','animated_gif'}]
        stamp = publication_time(tweet.get('created_at'))
        posts[identifier] = {'source_url':f'https://x.com/{account}/status/{identifier}',
            'post_id':identifier,'author':account,'published_at':stamp.isoformat() if stamp else '',
            'identity_status':'matched','author_status':'source_reported',
            'date_status':'source_reported' if stamp else 'unknown',
            'text':str(tweet.get('full_text') or tweet.get('text') or '')[:1000],
            'media_count':len(roles),'media_roles':list(dict.fromkeys(roles))}
    return sorted(posts.values(),key=lambda p:p['published_at'],reverse=True)


class TwitterTimelineClient:
    def __init__(self, web_client=None, *, reader=None, clock=None):
        self.reader = reader or WebImageClient(web_client)
        self.clock = clock or (lambda:datetime.now(UTC))

    def list_posts(self, account: str, *, limit=6, lookback_days=None, check=lambda:None, deadline=None) -> dict:
        check();account=parse_account(account)
        if type(limit) is not int or not 1<=limit<=12 or (lookback_days is not None and (type(lookback_days) is not int or not 1<=lookback_days<=365)):
            raise ValueError('invalid_twitter_timeline_window')
        deadline = min(deadline or time.monotonic()+20,time.monotonic()+20)
        url = 'https://syndication.twitter.com/srv/timeline-profile/screen-name/'+account
        _, body, encoding = self.reader.fetch(url,check=check,deadline=deadline,metadata=True)
        check();posts = parse_timeline(body.decode(encoding,errors='replace'),account)
        now = self.clock().astimezone(UTC)
        cutoff = now-timedelta(days=lookback_days) if lookback_days is not None else None
        eligible, old, invalid = [], 0, 0
        visible=[]
        for post in posts:
            check();stamp=publication_time(post['published_at'])
            if stamp is None or stamp>now+timedelta(minutes=5):invalid+=1;continue
            visible.append(post)
            if cutoff is not None and stamp<cutoff:old+=1;continue
            if post['media_count']:eligible.append(post)
        check()
        return {'account':account,'source_url':url,'provider':'x_public_embed',
            'posts':eligible[:limit],'status':'ok' if eligible else 'partial',
            'observed_at':now.isoformat(),'published_after':cutoff.isoformat() if cutoff else None,
            'lookback_days':lookback_days,'visible_posts':len(posts),'eligible_posts':len(eligible),
            'excluded_old':old,'excluded_invalid_date':invalid,
            'newest_visible_at':max((p['published_at'] for p in visible),default=''),
            'complete':False,'latest_proven':False,
            'coverage':'public_embedded_account_entries_only; pinned_and_popular_possible; latest_not_proven',
            **({'gap':'no_verified_posts_in_requested_window'} if not eligible else {})}
