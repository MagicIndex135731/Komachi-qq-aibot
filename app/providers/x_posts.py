"""Public X post targets and evidence shared by HTTP crawler adapters."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
import hashlib
import json
import re
from urllib.parse import parse_qsl, urlencode, urlsplit

from app.providers.twitter_media import TwitterMediaError, _parse_media, parse_post_url
from app.providers.twitter_timeline import parse_account

MAX_X_SUBJECT_TERMS = 3
MAX_X_TOPIC_TERMS = 4
MAX_X_SEMANTIC_QUERY_CHARS = 400
MAX_X_CRAWLER_REQUEST_BYTES = 8192


@dataclass(frozen=True)
class XTarget:
    url: str
    kind: str
    identifier: str
    sort: str = 'latest'


def search_target(query: str, sort: str = 'latest') -> XTarget:
    if (sort not in {'latest','top'} or not isinstance(query,str) or not query.strip() or len(query)>600
            or any(ord(c)<32 or ord(c)==127 for c in query)):
        raise ValueError('invalid_x_target')
    query=query.strip()
    return XTarget('https://x.com/search?'+urlencode({'q':query,'f':'live' if sort=='latest' else 'top','src':'typed_query'}),'search',query,sort)


@dataclass(frozen=True)
class XSearchOptions:
    media_types: tuple[str, ...] = ()
    sort: str = 'latest'
    min_likes: int | None = None
    min_replies: int | None = None
    min_reposts: int | None = None
    subject_terms: tuple[str, ...] = ()
    topic_terms: tuple[str, ...] = ()
    topic_match: str = 'keywords'

    @classmethod
    def parse(cls, *, media_types=None, sort='latest', min_likes=None, min_replies=None, min_reposts=None,
            subject_terms=None,topic_terms=None,topic_match='keywords'):
        if (sort not in {'latest','top'} or (media_types is not None and
                (not isinstance(media_types,list) or not 1<=len(media_types)<=2
                 or any(t not in ('image','video') for t in media_types) or len(set(media_types))!=len(media_types)))):
            raise ValueError('invalid_x_search_conditions')
        for value in (min_likes,min_replies,min_reposts):
            if value is not None and (type(value) is not int or not 0<=value<=1000000000):
                raise ValueError('invalid_x_search_conditions')
        for terms,maximum in ((subject_terms,MAX_X_SUBJECT_TERMS),(topic_terms,MAX_X_TOPIC_TERMS)):
            if terms is not None and (not isinstance(terms,list) or not 1<=len(terms)<=maximum
                    or any(not isinstance(t,str) or not t.strip() or len(t)>40 for t in terms)):
                raise ValueError('invalid_x_search_conditions')
        if topic_match not in {'keywords','review'}:raise ValueError('invalid_x_search_conditions')
        return cls(tuple(t for t in ('image','video') if t in (media_types or ())),sort,min_likes,min_replies,min_reposts,
            tuple(subject_terms or ()),tuple(topic_terms or ()),topic_match)

    def as_dict(self):
        return {'media_types':list(self.media_types) or None,'sort':self.sort,
            'min_likes':self.min_likes,'min_replies':self.min_replies,'min_reposts':self.min_reposts,
            'subject_terms':list(self.subject_terms) or None,'topic_terms':list(self.topic_terms) or None,
            'topic_match':self.topic_match}


def matches_terms(text: str, terms: tuple[str, ...]) -> bool:
    """Conservative lexical candidate check, never proof of who's in the pixels."""
    text=' '.join(text.casefold().split())
    for term in terms:
        term=' '.join(term.replace('"',' ').replace('(',' ').replace(')',' ').casefold().split())
        if not term:continue
        if re.fullmatch(r'[@#]?[a-z0-9_ ]+',term):
            if re.search(r'(?<![a-z0-9_])'+re.escape(term)+r'(?![a-z0-9_])',text):return True
        elif term in text:return True
    return not terms


def compile_search_targets(query: str, options: XSearchOptions, now: datetime, days: int | None):
    """Calendar bounds overfetch slightly; the exact rolling window is checked locally."""
    if days is not None and (type(days) is not int or not 1<=days<=365):raise ValueError('invalid_x_search_window')
    suffix=(f' since:{(now-timedelta(days=days)).date()}' if days is not None else '')+f' until:{(now+timedelta(days=1)).date()}'
    for name,operator in (('min_likes','min_faves'),('min_replies','min_replies'),('min_reposts','min_reposts')):
        value=getattr(options,name)
        if value is not None:suffix+=f' {operator}:{value}'
    filters={'image':'filter:images','video':'filter:native_video','any':'filter:media'}
    return [(kind,search_target('('+query+')'+suffix+' '+filters[kind],options.sort))
        for kind in (options.media_types or ('any',))]


def engagement_counts(data: dict) -> dict:
    """Unknown counters stay unknown; a missing value never proves a threshold."""
    return {name:value if type(value) is int and 0<=value<=10**12 else None
        for name,key in (('likes','favorite_count'),('replies','reply_count'),('reposts','retweet_count'))
        for value in (data.get(key),)}


def parse_target(value: str) -> XTarget:
    if not isinstance(value,str) or len(value)>4096:raise ValueError('invalid_x_target')
    if value.startswith('https://') and '/status/' in value:
        post=parse_post_url(value)
        return XTarget(post.source_url,'post',post.post_id)
    if value.startswith('https://') and urlsplit(value).path=='/search':
        parts=urlsplit(value);pairs=parse_qsl(parts.query,keep_blank_values=True);params=dict(pairs)
        if (parts.hostname not in {'x.com','www.x.com','twitter.com','www.twitter.com'}
                or parts.username or parts.password or parts.port or parts.fragment
                or len(params)!=len(pairs) or set(params)-{'q','f','src'}
                or params.get('f') not in {'live','top'} or params.get('src','typed_query')!='typed_query'):
            raise ValueError('invalid_x_target')
        return search_target(params.get('q',''),'top' if params['f']=='top' else 'latest')
    account=parse_account(value)
    return XTarget('https://x.com/'+account,'account',account)


def make_snapshot(target: XTarget, rows: list[dict], **observation) -> dict:
    body={'source_url':target.url,'posts':rows,'provider':'x_crawler','authenticated':True,**observation}
    sid='x-posts-'+hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False,
        separators=(',',':')).encode()).hexdigest()[:24]
    return {**body,'schema_version':1,'snapshot_id':sid,'observed_at':datetime.now(UTC).isoformat(),
        'posts':[{**row,'evidence_ref':f'{sid}:post:{i}'} for i,row in enumerate(rows)],
        'trust':'untrusted_external_source','complete':False,'latest_proven':False}


def parse_api_posts(payload) -> list[dict]:
    """Project twscrape raw responses; never descend into quotes or retweets."""
    found,visited={},0
    def visit(node,depth=0):
        nonlocal visited
        visited+=1
        if visited>25000 or depth>40:return
        if isinstance(node,list):
            for child in node[:400]:visit(child,depth+1)
        if not isinstance(node,dict):return
        legacy=node.get('legacy')
        if isinstance(legacy,dict) and 'rest_id' in node and 'full_text' in legacy:
            pid=node['rest_id']
            if (not isinstance(pid,str) or not re.fullmatch(r'[1-9][0-9]{0,24}',pid)
                    or legacy.get('id_str',pid)!=pid or legacy.get('retweeted_status_result')
                    or node.get('retweeted_status_result') or legacy.get('retweeted_status')):return
            user=((node.get('core') or {}).get('user_results') or {}).get('result') or {}
            flags=[user.get('protected'),(user.get('legacy') or {}).get('protected'),(user.get('privacy') or {}).get('protected')]
            protected=True if any(f is True for f in flags) else False if any(f is False for f in flags) else None
            if protected is not False:return
            author=(user.get('core') or {}).get('screen_name') or (user.get('legacy') or {}).get('screen_name')
            if not isinstance(author,str) or not re.fullmatch(r'[A-Za-z0-9_]{1,15}',author):return
            try:
                parsed=parsedate_to_datetime(legacy.get('created_at'))
                stamp=parsed.astimezone(UTC).isoformat() if parsed.tzinfo else ''
            except (ValueError,TypeError,AttributeError,OverflowError):stamp=''
            items=(legacy.get('extended_entities') or {}).get('media') or []
            clean={'id_str':pid,'mediaDetails':items}
            media,error=[],''
            try:media=[{'kind':m.kind,'url':m.url,**({'variants':list(m.variants)} if m.variants else {})} for m in _parse_media(clean,parse_post_url(f'https://x.com/{author}/status/{pid}'),'syndication')]
            except TwitterMediaError as exc:error=exc.code
            found[pid]={'post_id':pid,'source_url':f'https://x.com/{author}/status/{pid}','author':author,
                'published_at':stamp,'text':str(legacy.get('full_text') or '')[:2000],
                'protected':False,'identity_status':'matched','media_count':len(items) if isinstance(items,list) else 0,
                'media':media,'media_error':error,'is_reply':bool(legacy.get('in_reply_to_status_id_str')),
                'engagement':engagement_counts(legacy)}
            return
        for key,child in list(node.items())[:200]:
            if key not in {'quoted_status_result','retweeted_status_result','card','core'}:visit(child,depth+1)
    visit(payload)
    return list(found.values())
