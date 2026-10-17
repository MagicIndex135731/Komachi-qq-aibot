"""Account discovery uses public HTTP and scoped evidence, without a model."""
from __future__ import annotations

import asyncio
from datetime import UTC,datetime,timedelta
import json
import re
from threading import Event
import time

import httpx

from app.agent.tool_agent import AgentTool, ToolOutcome
from app.providers.twitter_timeline import TwitterTimelineClient, parse_account
from app.providers.web_images import WebImageError
from app.providers.twitter_media import TwitterMediaError
from app.providers.x_posts import MAX_X_SUBJECT_TERMS, MAX_X_TOPIC_TERMS, MAX_X_SEMANTIC_QUERY_CHARS


def normalized_terms(terms:list[str],maximum:int)->list[str]:
    """Canonicalize equivalent OR terms; never introduce a new semantic facet."""
    if (not isinstance(terms,list) or not 1<=len(terms)<=maximum
            or any(not isinstance(term,str) or not 1<=len(term)<=40 for term in terms)):
        raise ValueError('invalid_media_query_terms')
    values={}
    for term in terms:
        term=' '.join(term.replace('"',' ').replace('(',' ').replace(')',' ').split())
        if not term:raise ValueError('invalid_media_query_terms')
        values.setdefault(term.casefold(),term)
    return list(values.values())


def media_query(subject_terms,topic_terms,topic_match='keywords'):
    """Model supplies semantic facets; each facet is OR, facets are AND."""
    def group(terms,maximum):
        values=[]
        for term in normalized_terms(terms,maximum):
            values.append(term if re.fullmatch(r'[\w@#]+',term) and term.upper() not in {'OR','AND'} else '"'+term+'"')
        return '('+' OR '.join(dict.fromkeys(values))+')'
    subject=group(subject_terms,MAX_X_SUBJECT_TERMS)
    topic=group(topic_terms,MAX_X_TOPIC_TERMS) if topic_terms is not None else ''
    if topic_match not in {'keywords','review'}:raise ValueError('invalid_media_query_terms')
    return subject+(' '+topic if topic and topic_match=='keywords' else '')+' filter:media'


def build_twitter_account_tool(client, *, runner, guard, workspace=None, settings=None, search=False) -> AgentTool:
    from app.providers.x_crawler_client import XCrawlerClient
    crawler = XCrawlerClient.from_settings(settings)
    timeline = crawler or TwitterTimelineClient(client)
    completed={}
    lock=asyncio.Lock()
    terminal_error=None

    async def discover(args):
        nonlocal terminal_error
        async with lock:
            guard.require_valid()
            if terminal_error:
                return ToolOutcome({'error':terminal_error,'stopped_in_turn':True,
                    'guidance':'本轮X已报告限流或认证障碍，停止站内请求；改关键词、窗口或排序不会重试。说明缺口，等待恢复或人工更新会话。'},'failed')
            if search and crawler is None:
                return ToolOutcome({'error':'x_crawler_search_unavailable',
                    'guidance':'登录态X搜索未配置；可用search_web发现公开原帖，不代表没有相关媒体。'},'blocked')
            try:
                args=dict(args)
                if search:
                    from app.providers.x_posts import XSearchOptions, search_target
                    if args.get('mode','semantic') not in {'semantic','raw'}:raise ValueError()
                    if args.get('mode','semantic')=='semantic':
                        args['subject_terms']=normalized_terms(args.get('subject_terms'),MAX_X_SUBJECT_TERMS)
                        if args.get('topic_terms') is not None:
                            args['topic_terms']=normalized_terms(args['topic_terms'],MAX_X_TOPIC_TERMS)
                        query=media_query(args.get('subject_terms'),args.get('topic_terms'),args.get('topic_match','keywords'))
                    else:
                        query=args['query']
                        request=getattr(workspace,'request','')
                        if not isinstance(query,str) or not query.strip() or query.strip() not in request:
                            return ToolOutcome({'error':'raw_x_query_requires_user_source',
                                'guidance':'raw只能复用用户消息中实际给出的原样查询。普通自然语言用semantic和subject_terms/topic_terms；主题词不足用topic_match=review，不得擅自加from限制发布者。'},'blocked')
                    maximum=MAX_X_SEMANTIC_QUERY_CHARS if args.get('mode','semantic')=='semantic' else 300
                    if not isinstance(query,str) or len(query)>maximum:raise ValueError()
                    account=search_target(query).identifier
                    conditions=XSearchOptions.parse(**{k:args[k] for k in (
                        'media_types','sort','min_likes','min_replies','min_reposts') if k in args},
                        **({k:args[k] for k in ('subject_terms','topic_terms','topic_match') if k in args}
                            if args.get('mode','semantic')=='semantic' else {}))
                else:account=parse_account(args['account'])
                from app.core.post_freshness import explicit_recent_days
                declared=getattr(workspace,'request_control',{}).get('media_selection',{}).get('lookback_days')
                explicit=explicit_recent_days(getattr(workspace,'request',''))
                limit,days=args.get('limit',12 if search else 6),args.get('lookback_days',declared)
                if type(limit) is not int or not 1<=limit<=12 or (days is not None and (type(days) is not int or not 1<=days<=365)):raise ValueError()
                if search and len(conditions.media_types)==2 and limit<2:raise ValueError()
                if explicit:days=min(days,explicit) if days is not None else explicit
                # Case/order changes are the same search, including lexical metadata.
                # Different themes in review mode retain separate evidence and judgments.
                canonical_query=(media_query(sorted(args['subject_terms'],key=str.casefold),
                    sorted(args['topic_terms'],key=str.casefold) if args.get('topic_terms') else None,
                    args.get('topic_match','keywords')).casefold()
                    if search and args.get('mode','semantic')=='semantic' else account.lower() if not search else account)
                key=(canonical_query,
                    limit,days,(conditions.media_types,conditions.sort,conditions.min_likes,
                        conditions.min_replies,conditions.min_reposts,
                        tuple(sorted(t.casefold() for t in conditions.subject_terms)),
                        tuple(sorted(t.casefold() for t in conditions.topic_terms)),conditions.topic_match) if search else None)
            except (ValueError,KeyError,TypeError):
                return ToolOutcome({'error':'structured_media_query_required' if search else 'invalid_twitter_account',
                    'guidance':'subject_terms需列出人名/本地名/已知完整账号别名；仅按人物和图片/视频类型检索可省略topic_terms。有额外内容主题时用topic_terms提供主题词，两组AND。用户给出原样X查询才用mode=raw和query。'},'blocked')
            if key in completed:
                old=completed[key]
                return ToolOutcome({k:v for k,v in old.data.items() if k in {
                    'account','query','status','error','gap','task_id','evidence_id','coverage','newest_visible_at','guidance','returned_count',
                    'search_conditions','media_type_counts','missing_media_types','ranking','topic_match','requested_topic_terms',
                    'empty_reason','search_visibility'}}
                    | {'cached_in_turn':True,'context_retention':'compact_v1'},old.status)
            if len(completed)>=(4 if search else 2):
                return ToolOutcome({'error':'twitter_timeline_turn_budget'},'blocked')
            cancelled=Event()
            def check():
                guard.require_valid()
                if cancelled.is_set():raise WebImageError('twitter_timeline_cancelled')
            try:
                async with asyncio.timeout(57 if crawler else 22):
                    full=await runner.run(timeline.search_posts if search else timeline.list_posts,account,
                        limit=limit,lookback_days=days,check=check,deadline=time.monotonic()+20,
                        **(conditions.as_dict() if search else {}))
                check()
                resources=[{**post,'text':post.get('text','')[:180 if search else 100]} for post in full['posts']]
                data={k:v for k,v in full.items() if k not in {'posts','source_url'}}
                data.update(resources=resources,content_kind='search_posts' if search else 'account_posts',
                    trust='untrusted_external_sources',context_retention='compact_v1',
                    guidance='仅为当前来源可见的近期原帖，可能缺最新或完整列表；无日期默认不限下界。没有合格项只说明抓取缺口，不证明没发帖，不得用旧帖冒充最新。已有候选可直接download_twitter_media；按正文挑选cos内容可inspect_web_resources，不必反复搜索。配置登录态采集时由程序自动调用，无需单独操作采集工具。')
                if search:
                    data.update(topic_match=args.get('topic_match','keywords'),requested_topic_terms=args.get('topic_terms',[]))
                    data['guidance']='站内结构化媒体搜索，发布者可不同于内容主体；命中不证明本人出镜或cos相关。图片和视频同时需要时media_types填两种，会分路召回并保留各类位置；missing_media_types表示未找到该类，不能用封面冒充视频。latest按时间兼顾类型，top保留各分路平台相关排序，均非全站最高或完整列表。结合正文及实际预览筛选不同媒体，裁切/封面不重复计数；保留作者和原帖日期，近期发布不证明素材新拍。零结果可调整别名；用户未指定天数且近期不足时可说明日期后扩大窗口，用户明确窗口不可放宽。'
                    data['guidance']+='先筛已有候选，足够时queue_media_batch批量review下载；失败换剩余原帖，不换词序重复查。同名噪声多时将已知完整handle或高精度别名独立查询，避免总与泛名OR；每组通常1至2个准确同义词，无需凑满4个；不同类别不充当同义词。严格主题不足可review补搜，仍按正文核对，不预览视频/动图。'
                    data['guidance']+='lexical_match只标记正文/作者词面命中，不证明人物身份或图片内容；subject=false可能是平台夹带的不相关推荐，未核验前不得当成目标人物素材。词面不命中不自动排除语义候选。'
                    try:
                        newest=datetime.fromisoformat(full.get('newest_visible_at',''))
                        now=datetime.fromtimestamp(workspace.clock(),UTC) if workspace is not None else datetime.now(UTC)
                        stale=(conditions.sort=='latest' and args.get('mode','semantic')=='semantic'
                            and args.get('topic_match','keywords')=='keywords' and days is not None and days>7
                            and newest.tzinfo and newest<now-timedelta(days=7))
                    except (ValueError,TypeError,AttributeError):stale=False
                    if stale:
                        data['freshness_gap']='keyword_results_older_than_7_days'
                        data['guidance']='严格主题词的可见最新帖已超过7天，可能漏掉未写主题标签的新素材。下载一批旧候选前，优先在同一时间窗口做一次topic_match=review、sort=latest的主体检索交叉补充，再核验实际内容；不扩大用户时间范围。'+data['guidance']
                    if full.get('missing_media_types') and args.get('topic_match','keywords')=='keywords':
                        data['guidance']='当前缺少'+','.join(full['missing_media_types'])+'候选。下一次优先保持时间/主体/类型和明确热度门槛，改topic_match=review补召回；不要为了缺视频改成top或反复扩大日期。'+data['guidance']
                    if not resources:
                        data['empty_reason']='outside_selected_window' if full.get('excluded_by_window',0) else 'no_visible_candidates'
                        if data['empty_reason']=='no_visible_candidates':
                            data['search_visibility']='not_verified'
                            data['guidance']='当前账号本次查询没有取得候选，尚未启动下载；不证明名字错误、账号不存在或没有视频。敏感媒体展示与搜索过滤是不同设置，本工具未读取账号设置；搜索过滤、索引覆盖或昵称匹配都可能影响召回，不能仅靠空结果归因。说明实际窗口、筛选条件与可见性未核验，不责怪用户、不反复改天数或猜别名撞站；明确内容拒绝不改词重试。检查missing_media_types和excluded_by_conditions，不得擅自放宽用户明确门槛。'+data['guidance']
                        else:
                            data['guidance']='已观察到匹配原帖，但被日期窗口排除。用户未指定天数时可声明近30天等合理lookback_days再查同一有效关键词，并说明实际日期；明确近N天不可放宽。'+data['guidance']
                else:
                    data['guidance']+='此列表仅覆盖该发布者，不代表主题全站结果；未限定作者且这里无合格内容时，改用search_x_posts，把已知账号全名作跨发布者关键词。'
                while len(json.dumps(data,ensure_ascii=False))>(8000 if search else 5500) and data['resources']:
                    remove=len(data['resources'])-1
                    if search and len(conditions.media_types)==2:
                        # The result budget must not erase the only video after fair selection.
                        roles={'image':'post_image','video':'post_video'}
                        for index in range(len(data['resources'])-1,-1,-1):
                            if all(roles[kind] not in data['resources'][index].get('media_roles',[]) or
                                    any(roles[kind] in r.get('media_roles',[]) for i,r in enumerate(data['resources']) if i!=index)
                                    for kind in conditions.media_types):
                                remove=index;break
                    data['resources']=data['resources'][:remove]+data['resources'][remove+1:]
                    data['truncated']=True
                data['returned_count']=len(data['resources'])
                if search:
                    data['subject_unverified_count']=sum(r.get('lexical_match',{}).get('subject') is False for r in data['resources'])
                if search and conditions.media_types:
                    data['media_type_counts']={kind:sum(('post_image' if kind=='image' else 'post_video') in r.get('media_roles',[])
                        for r in data['resources']) for kind in conditions.media_types}
                    data['missing_media_types']=[kind for kind,count in data['media_type_counts'].items() if not count]
                if workspace is not None:
                    data['task_id']=workspace.task_id
                    data['evidence_id']=workspace.save_evidence({'content':json.dumps(full,ensure_ascii=False),
                        'sources':[{'source':full['source_url'],'title':'X 最新搜索' if search else 'X @'+account+' 可见帖子'}],
                        'resources':resources,'content_kind':data['content_kind'],'trust':'untrusted_external_sources'})
                outcome=ToolOutcome(data)
            except (httpx.HTTPError,ValueError,OSError,TimeoutError,KeyError,TypeError,AttributeError,RecursionError) as exc:
                check()
                code=(str(exc) if isinstance(exc,(WebImageError,TwitterMediaError)) else
                    f'twitter_timeline_http_{exc.response.status_code}' if isinstance(exc,httpx.HTTPStatusError)
                    else 'twitter_timeline_unavailable')
                if code in {'x_crawler_'+value for value in ('rate_limited','login_required','challenge','access_denied','credentials_invalid')}:
                    terminal_error=code
                outcome=ToolOutcome({'account':account,'error':code[:100],
                    'guidance':'X来源本次未取得，不是没有帖子；说明日期与覆盖缺口，不重复撞站。可用已有可靠链接或search_web发现公开来源，搜索索引不证明最新。'},'failed')
            finally:cancelled.set()
            guard.require_valid();completed[key]=outcome
            return outcome

    if search:
        return AgentTool('search_x_posts',
            '主题媒体优先用此工具跨发布者检索。semantic模式用subject_terms给人名/本地名/完整handle别名，有额外内容主题才用topic_terms给主题词，各组OR、两组AND。仅按人物找图片/视频可省略topic_terms，类型直接填media_types，不必编造主题。默认topic_match=keywords，严格主题词不足时可改review按主体召回，仍以原帖正文核对主题，不把搜索命中直接当作相关，不预览视频/动图。media_types按需求填image/video，两种都要时填两种。无日期默认不限下界/12条/latest，时间进入站内搜索再精确核验；热门用sort=top，不代表回复数最高。min_likes/min_replies/min_reposts仅用户明确数字门槛时填，禁止编造100赞等条件。未限定天数且不足可说明日期后合理扩大，明确窗口不可放宽。最多4次查询。raw仅用于用户原样查询。',
            {'type':'object','properties':{
                'subject_terms':{'type':'array','items':{'type':'string','minLength':1,'maxLength':40},'minItems':1,'maxItems':MAX_X_SUBJECT_TERMS,
                    'description':'同一内容主体的别名，包含已知完整账号名/提及，不是from发布者限制'},
                'topic_terms':{'type':'array','items':{'type':'string','minLength':1,'maxLength':40},'minItems':1,'maxItems':MAX_X_TOPIC_TERMS,
                    'description':'可省略；仅用户要求额外内容主题时提供同义/多语言关键词，不要填视频或用户未要求的类别'},
                'mode':{'type':'string','enum':['semantic','raw']},
                'topic_match':{'type':'string','enum':['keywords','review'],
                    'description':'keywords要求正文主题词；候选不足可review只检索主体，仍按原帖正文判断topic_terms，不改变用户主题目标，不强制视觉预览'},
                'query':{'type':'string','minLength':1,'maxLength':300,'description':'仅mode=raw时使用，普通主题请求提供subject_terms/topic_terms'},
                'limit':{'type':'integer','minimum':1,'maximum':12},
                'media_types':{'type':'array','items':{'type':'string','enum':['image','video']},
                    'minItems':1,'maxItems':2,'uniqueItems':True,'description':'image=原生图片，video=真实原生视频；两种都需要时同时填写，limit至少2。不包含GIF或外链视频。'},
                'sort':{'type':'string','enum':['latest','top'],'description':'latest近期时间优先；top平台热门/相关排序，不保证互动数最高'},
                **{key:{'type':'integer','minimum':0,'maximum':1000000000,
                    'description':'仅用户明确数字门槛时填写；缺失互动数不视为达标'} for key in ('min_likes','min_replies','min_reposts')},
                'lookback_days':{'anyOf':[{'type':'integer','minimum':1,'maximum':365},{'type':'null'}],'description':'可选检索窗口；无日期要求默认null不限下界，0无效。用户明确窗口不可放宽'}},
                'required':[],'additionalProperties':False},discover)
    return AgentTool('list_x_posts',
        '读取指定发布者的近期公开媒体原帖，默认7天/6条；返回URL、作者、日期、正文和媒体数。用户明确限定该账号发布时用此工具；关于人物/主题而未限定作者时优先search_x_posts跨发布者搜索，不能因已知本人账号就只查本人。配置X会话时自动使用爬虫，否则公开嵌入；覆盖不完整/最新未证明。',
        {'type':'object','properties':{'account':{'type':'string','minLength':1,'maxLength':128},
            'limit':{'type':'integer','minimum':1,'maximum':12},
            'lookback_days':{'anyOf':[{'type':'integer','minimum':1,'maximum':365},{'type':'null'}],'description':'可选检索窗口；无日期要求默认null不限下界，0无效。用户明确窗口不可放宽'}},
            'required':['account'],'additionalProperties':False},discover)
