"""Container-local X metadata acquisition using gallery-dl, with twscrape fallback.

No browser, model, arbitrary library actions, media download or QQ send here.
Each production call runs in an isolated, hard-deadline worker process.
"""
from __future__ import annotations

import asyncio
from contextlib import aclosing, contextmanager
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from app.providers.twitter_media import _allowed_url, _original_photo, parse_post_url
from app.providers.x_posts import (XSearchOptions, compile_search_targets, engagement_counts,
    make_snapshot, matches_terms, parse_api_posts, parse_target)


FAILURE_CODES=frozenset({'login_required','challenge','rate_limited','access_denied','identity_mismatch',
    'invalid_metadata','parse_failed','network_error','unavailable','deadline_exceeded','request_budget',
    'response_too_large','crawler_busy','credentials_invalid','read_only_boundary'})
FALLBACK_CODES=frozenset({'invalid_metadata','parse_failed','network_error','unavailable'})


class CrawlerError(ValueError):
    def __init__(self,code):
        self.code=code if code in FAILURE_CODES else 'parse_failed'
        super().__init__(self.code)


def load_session(path: Path) -> dict:
    if (path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077
            or path.stat().st_size>8192):raise CrawlerError('credentials_invalid')
    try:
        data=json.loads(path.read_text())
        cookies={k:data[k] for k in ('auth_token','ct0')}
        if any(not isinstance(v,str) or not 20<=len(v)<=2048 for v in cookies.values()):raise ValueError()
        return cookies
    except (ValueError,KeyError,TypeError,OSError) as exc:raise CrawlerError('credentials_invalid') from exc


def project_gallery(data: dict, media: list[dict]) -> dict:
    author=data.get('author') or {}
    if author.get('protected') is not False:raise CrawlerError('invalid_metadata')
    pid=str(data.get('tweet_id',''));stamp=data.get('date')
    if isinstance(stamp,datetime):
        stamp=(stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)).astimezone(UTC).isoformat()
    else:stamp=''
    if type(data.get('count')) is not int or data['count']!=len(media):raise CrawlerError('invalid_metadata')
    return {'post_id':pid,'source_url':f'https://x.com/{author.get("name", "")}/status/{pid}',
        'author':author.get('name'),'published_at':stamp,'text':str(data.get('content',''))[:2000],
        'protected':False,'identity_status':'matched','is_reply':bool(data.get('reply_id')),
        'media_count':len(media),'media_error':'','media':media,'engagement':engagement_counts(data)}


def _validate_row(row):
    if not isinstance(row,dict):raise CrawlerError('invalid_metadata')
    if row.get('protected') is True:return None
    if row.get('protected') is not False:raise CrawlerError('invalid_metadata')
    try:
        author=row['author'];post=parse_post_url(row['source_url'])
        if (post.post_id!=row.get('post_id') or row.get('identity_status')!='matched'
                or not isinstance(author,str) or not re.fullmatch(r'[A-Za-z0-9_]{1,15}',author)
                or urlsplit(row['source_url']).path.split('/')[1].lower() not in {'i',author.lower()}):
            raise CrawlerError('identity_mismatch')
        media=row['media']
        if not isinstance(media,list) or len(media)>16:raise CrawlerError('invalid_metadata')
        for item in media:
            if item['kind']=='photo':_original_photo(item['url'])
            elif item['kind'] in {'video','gif'}:
                url=_allowed_url(item['url'],{'video.twimg.com'},'invalid_media_url')
                if not urlsplit(url).path.endswith('.mp4'):raise CrawlerError('invalid_metadata')
            else:raise CrawlerError('invalid_metadata')
        if row.get('media_count')!=len(media) and not row.get('media_error'):raise CrawlerError('invalid_metadata')
    except (KeyError,ValueError,TypeError,AttributeError) as exc:
        if isinstance(exc,CrawlerError):raise
        raise CrawlerError('invalid_metadata') from exc
    return row


class XCrawler:
    def __init__(self,cookies,*,primary=None,fallback=None,clock=None):
        self.cookies=cookies
        self.primary,self.fallback=primary or gallery_posts,fallback or twscrape_posts
        self.clock=clock or (lambda:datetime.now(UTC))

    def acquire(self,action,target,*,limit=6,lookback_days=None,deadline=None,
            media_types=None,sort=None,min_likes=None,min_replies=None,min_reposts=None,
            subject_terms=None,topic_terms=None,topic_match='keywords'):
        if (type(limit) is not int or not 1<=limit<=24 or (lookback_days is not None and (type(lookback_days) is not int or not 1<=lookback_days<=365))):
            raise ValueError('invalid_request')
        parsed=parse_target(target)
        if {'list':'account','search':'search','inspect':'post'}.get(action)!=parsed.kind:raise ValueError('invalid_request')
        options=XSearchOptions.parse(media_types=media_types,sort=parsed.sort if sort is None else sort,min_likes=min_likes,
            min_replies=min_replies,min_reposts=min_reposts,subject_terms=subject_terms,topic_terms=topic_terms,topic_match=topic_match)
        if len(options.media_types)==2 and limit<2:raise ValueError('invalid_request')
        if action!='search' and options!=XSearchOptions():raise ValueError('invalid_request')
        now=self.clock().astimezone(UTC)
        targets=compile_search_targets(parsed.identifier,options,now,lookback_days) if action=='search' else [('any',parsed)]
        deadline=deadline or time.monotonic()+35
        budget={'count':0,'status':0,'bytes':0}
        def fetch(backend):
            groups=[]
            for kind,target in targets:
                if time.monotonic()>deadline:raise CrawlerError('deadline_exceeded')
                values=backend(target,self.cookies,limit=max(limit,12) if action=='search' else limit,
                    deadline=deadline,budget=budget)
                if not isinstance(values,list) or len(values)>120:raise CrawlerError('invalid_metadata')
                rows=[r for r in (_validate_row(v) for v in values) if r is not None]
                if parsed.kind=='post':
                    if any(r['post_id']!=parsed.identifier for r in rows):raise CrawlerError('identity_mismatch')
                    if not rows:raise CrawlerError('unavailable')
                if parsed.kind=='account' and any(r['author'].lower()!=parsed.identifier.lower() for r in rows):
                    raise CrawlerError('identity_mismatch')
                groups.append((kind,rows))
            return groups
        fallback_reason=None;backend='gallery-dl'
        try:groups=fetch(self.primary)
        except CrawlerError as exc:
            if exc.code not in FALLBACK_CODES:raise
            fallback_reason=exc.code;backend='twscrape';groups=fetch(self.fallback)
        window_excluded=condition_excluded=0;accepted=[];counts={}
        def kinds(row):
            return {m['kind']=='photo' and 'image' or 'video' for m in row['media'] if m['kind'] in {'photo','video'}}
        for kind,rows in groups:
            kept=[]
            for row in rows:
                if parsed.kind!='post':
                    try:stamp=datetime.fromisoformat(row['published_at'])
                    except (ValueError,TypeError):stamp=None
                    if stamp is None or not stamp.tzinfo or (stamp>now+timedelta(minutes=5) or lookback_days is not None and stamp<now-timedelta(days=lookback_days)):
                        window_excluded+=1;continue
                metrics=row.get('engagement') or {}
                if ((kind!='any' and kind not in kinds(row)) or any(value is not None and
                        (type(metrics.get(name)) is not int or metrics[name]<value)
                        for name,value in (('likes',options.min_likes),('replies',options.min_replies),('reposts',options.min_reposts)))):
                    condition_excluded+=1;continue
                if options.subject_terms or options.topic_terms:
                    row={**row,'lexical_match':{
                        'subject':matches_terms(row.get('text','')+' @'+row['author'],options.subject_terms) if options.subject_terms else None,
                        'topic':matches_terms(row.get('text',''),options.topic_terms) if options.topic_terms else None}}
                kept.append(row)
            counts[kind]=len(kept);accepted.append(kept)
        # Interleave platform-ranked branches. Latest is global, with a place reserved per requested type.
        rows=list({r['post_id']:r for i in range(max(map(len,accepted),default=0))
            for branch in accepted if i<len(branch) for r in [branch[i]]}.values())
        if options.sort=='latest':rows.sort(key=lambda r:r['published_at'],reverse=True)
        # Rank likely matches first; do not discard semantic candidates solely for word overlap.
        rows.sort(key=lambda r:r.get('lexical_match',{}).get('subject') is False)
        selected=[]
        if len(options.media_types)==2:
            for kind in options.media_types:
                if not any(kind in kinds(r) for r in selected):
                    match=next((r for r in rows if kind in kinds(r)),None)
                    if match is not None:selected.append(match)
        selected.extend(r for r in rows if r['post_id'] not in {s['post_id'] for s in selected})
        selected=selected[:limit]
        order={r['post_id']:i for i,r in enumerate(rows)}
        rows=sorted(selected,key=lambda r:order[r['post_id']])
        type_counts={kind:sum(kind in kinds(r) for r in rows) for kind in options.media_types}
        missing=[kind for kind,count in type_counts.items() if not count]
        return make_snapshot(parsed,rows,backend=backend,**({'fallback_reason':fallback_reason} if fallback_reason else {}),
            status='ok' if rows and not missing else 'partial',limit=limit,lookback_days=lookback_days,
            excluded_by_window=window_excluded,excluded_by_conditions=condition_excluded,
            **({'search_conditions':options.as_dict(),'executed_queries':[t.url for _,t in targets],
                'branch_counts':counts,'media_type_counts':type_counts,'missing_media_types':missing,
                'subject_unverified_count':sum(r.get('lexical_match',{}).get('subject') is False for r in rows),
                'ranking':'bounded_platform_top_with_media_diversity' if options.sort=='top' else 'latest_with_media_diversity'} if action=='search' else {}),
            coverage='resolved_root_post' if parsed.kind=='post' else 'bounded_search_posts' if parsed.kind=='search' else 'bounded_account_posts')


def _classify(exc,status=0):
    if isinstance(exc,CrawlerError):return exc
    name=type(exc).__name__.lower()
    if any(word in name for word in ('challenge','captcha','locked')):return CrawlerError('challenge')
    if status==429:return CrawlerError('rate_limited')
    if status in {401,403} or 'auth' in name:return CrawlerError('login_required')
    if 'timeout' in name or 'connect' in name or status>=500:return CrawlerError('network_error')
    if status==404 or 'notfound' in name or 'unavailable' in name:return CrawlerError('unavailable')
    return CrawlerError('parse_failed')


@contextmanager
def network_budget(deadline,proxy,state=None):
    """Restrict stock library requests, including redirects; never log headers."""
    import requests
    import httpx
    if state is None:state={'count':0,'status':0}
    def admit(method,url):
        parts=urlsplit(str(url));host=parts.hostname or ''
        if (method.upper() not in {'GET','HEAD','OPTIONS'} or parts.scheme!='https'
                or not any(host==h or host.endswith('.'+h) for h in ('x.com','twitter.com','twimg.com'))
                or parts.username or parts.password or parts.port not in {None,443}):raise CrawlerError('read_only_boundary')
        if '/graphql/' in parts.path and parts.path.rsplit('/',1)[-1] not in {
                'UserByScreenName','UserTweets','UserOriginalsTimeline','UserMedia','SearchTimeline','TweetDetail','TweetResultByRestId'}:
            raise CrawlerError('read_only_boundary')
        if '/i/api/' in parts.path and '/graphql/' not in parts.path:raise CrawlerError('read_only_boundary')
        if time.monotonic()>=deadline:raise CrawlerError('deadline_exceeded')
        if state['count']>=18:raise CrawlerError('request_budget')
        state['count']+=1
    send=requests.Session.send;async_send=httpx.AsyncClient._send_single_request
    def bounded_send(client,request,**kwargs):
        admit(request.method,request.url);kwargs['timeout']=max(.1,min(8,deadline-time.monotonic()));kwargs['stream']=True
        kwargs['proxies']={'http':proxy,'https':proxy}
        response=send(client,request,**kwargs);state['status']=response.status_code
        body=bytearray()
        try:
            for chunk in response.iter_content(65536):
                if time.monotonic()>=deadline:raise CrawlerError('deadline_exceeded')
                state['bytes']=state.get('bytes',0)+len(chunk)
                if state['bytes']>32*1024*1024:raise CrawlerError('response_too_large')
                body.extend(chunk)
                if len(body)>8*1024*1024:raise CrawlerError('response_too_large')
            response._content=bytes(body);response._content_consumed=True
            return response
        finally:response.close()
    async def bounded_async(client,request):
        admit(request.method,request.url)
        async with asyncio.timeout(max(.1,min(8,deadline-time.monotonic()))):
            response=await async_send(client,request);state['status']=response.status_code;body=bytearray()
            try:
                async for chunk in response.aiter_bytes(65536):
                    if time.monotonic()>=deadline:raise CrawlerError('deadline_exceeded')
                    state['bytes']=state.get('bytes',0)+len(chunk)
                    if state['bytes']>32*1024*1024:raise CrawlerError('response_too_large')
                    body.extend(chunk)
                    if len(body)>8*1024*1024:raise CrawlerError('response_too_large')
                response._content=bytes(body)
                return response
            finally:await response.aclose()
    requests.Session.send=bounded_send;httpx.AsyncClient._send_single_request=bounded_async
    try:yield state
    finally:requests.Session.send=send;httpx.AsyncClient._send_single_request=async_send


def gallery_posts(target,cookies,*,limit,deadline,budget=None):
    from gallery_dl import config,extractor
    config.clear();config.set(('cache',),'file',':memory:')
    proxy=os.environ.get('X_CRAWLER_PROXY_URL','http://127.0.0.1:18797')
    for key,value in {'proxy':proxy,'timeout':8,'retries':0,'sleep-request':0}.items():config.set(('extractor',),key,value)
    for key,value in {'cookies':dict(cookies),'search-results':target.sort,'search-limit':limit,'limit':limit,
            'search-stop':1,'ratelimit':'abort','locked':'abort','logout':False,'retries-api':0,
            'articles':False,'quoted':False,'retweets':False,'replies':True,'cards':False,'text-tweets':True}.items():
        config.set(('extractor','twitter'),key,value)
    rows=[];current=None;media=[]
    def append():
        if current is not None and (current.get('author') or {}).get('protected') is not True:
            rows.append(project_gallery(current,media))
    state={'status':0}
    try:
        with network_budget(deadline,proxy,budget) as state:
            from app.providers.x_posts import search_target
            url=search_target('from:'+target.identifier+' -filter:replies -filter:retweets').url if target.kind=='account' else target.url
            worker=extractor.find(url)
            original_extract=worker._extract_media
            def retain_variants(tweet,entities,files):
                start=len(files)
                original_extract(tweet,entities,files)
                for file in files[start:]:
                    for entity in entities:
                        info=entity.get('video_info') or {}
                        raw=info.get('variants') or []
                        if any(v.get('url')==file.get('url') for v in raw if isinstance(v,dict)):
                            from app.providers.twitter_media import _media_item
                            parsed=_media_item(entity,syndication=True)
                            file['native_variants']=list(parsed.variants)
                            break
            worker._extract_media=retain_variants
            for message in worker:
                if message[0]==2:
                    append()
                    if len(rows)>=limit:current=None;break
                    current=dict(message[2]);media=[]
                elif message[0]==3:
                    if current is None or str(message[2].get('tweet_id'))!=str(current.get('tweet_id')):
                        raise CrawlerError('identity_mismatch')
                    url=message[1];kind=('gif' if message[2].get('type')=='animated_gif' else 'video') if urlsplit(url).path.endswith('.mp4') else 'photo'
                    media.append({'kind':kind,'url':url,
                        **({'variants':message[2]['native_variants']} if message[2].get('native_variants') else {})})
                elif message[0]!=1:raise CrawlerError('invalid_metadata')
            append()
        return rows
    except Exception as exc:raise _classify(exc,state['status']) from exc
    finally:config.clear()


def twscrape_posts(target,cookies,*,limit,deadline,budget=None):
    import os
    os.environ['TWS_TELEMETRY']='0'
    os.environ['TWS_HTTP_BACKEND']='httpx'
    from twscrape import API
    from twscrape.account import Account
    from twscrape.accounts_pool import AccountsPool
    from twscrape.logger import logger
    logger.remove()
    class MemoryPool(AccountsPool):
        def __init__(self):
            self.account=Account(username='container-session',password='_',email='_',email_password='_',
                user_agent='@chrome',active=True,cookies=dict(cookies),proxy=proxy)
            self.locked=False
        async def get_for_queue_or_wait(self,queue):return self.account if self.account.active and not self.locked else None
        async def unlock(self,*args,**kwargs):pass
        async def lock_until(self,*args,**kwargs):self.locked=True
        async def mark_inactive(self,*args,**kwargs):self.account.active=False
    proxy=os.environ.get('X_CRAWLER_PROXY_URL','http://127.0.0.1:18797')
    async def run():
        pool=MemoryPool();api=API(pool,debug=False,proxy=proxy);rows={}
        try:
            async with asyncio.timeout(max(.1,deadline-time.monotonic())):
                if target.kind=='post':
                    responses=[await api.tweet_details_raw(int(target.identifier))]
                    for response in responses:
                        if response:
                            for row in parse_api_posts(response.json()):
                                if row['post_id']==target.identifier:rows[row['post_id']]=row
                else:
                    query=target.identifier if target.kind=='search' else 'from:'+target.identifier+' -filter:replies -filter:retweets'
                    async with aclosing(api.search_raw(query,limit=limit,kv={'product':'Top' if target.sort=='top' else 'Latest'})) as results:
                        async for response in results:
                            for row in parse_api_posts(response.json()):rows[row['post_id']]=row
                            if len(rows)>=limit:break
            if not pool.account.active:raise CrawlerError('login_required')
            if pool.locked:raise CrawlerError('rate_limited')
            return list(rows.values())[:limit]
        finally:pool.account.cookies.clear()
    state={'status':0}
    try:
        with network_budget(deadline,proxy,budget) as state:return asyncio.run(run())
    except Exception as exc:raise _classify(exc,state['status']) from exc
