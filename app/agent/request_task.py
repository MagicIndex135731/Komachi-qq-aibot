"""Source-bound request goals and bounded feedback, also outside research mode.

Lexical seeds are conservative hints. The main model can describe any other
goal through the same stable tool; neither mechanism grants tool authority.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
import re

from app.core.post_freshness import explicit_recent_days,publication_time, recent_cutoff, publication_filter
from app.agent.tool_agent import AgentTool, ToolOutcome
from app.agent.tool_evidence import evidence_descriptor, resource_ready

_MEDIA=re.compile(r'图片|原图|照片|图集|cos图|视频|相片|\b(?:photos?|images?|videos?)\b',re.I)
_TRANSFER=re.compile(r'下载|搬运|搬过来|扒|提取|保存|发(?:给|我|来|到)|传(?:给|我|来)|\b(?:download|send)\b',re.I)
_WANT=re.compile(r'(?:我要看|我想看|想看|给我看|想要|我要|来点|来张).{0,100}(?:图|照片|视频)|(?:帮我|给我).{0,80}(?:找|搜集|搜罗).{0,80}(?:图片|照片|视频|cos图)|\bshow me\b',re.I)
_ANALYZE=re.compile(r'图(?:片)?里|图中|视频(?:里|中|说|讲)|写了什么|说的是什么|描述这|分析这|解释这|小诗|故事|作文')
_EXTERNAL=re.compile(r'联网|上网|搜索|检索|查询|查证|核实|原文|https?://|\bsearch\b',re.I)
_TASK=re.compile(r'研究|调研|审阅|审查|漏洞|分析|总结|整理|比较|对比|查询|查证|下载|搬运|提取|保存|帮我写|给我写|\b(?:research|review|download)\b',re.I)
_PRIMARY=re.compile(r'原文|核实|查证|最新|最近|近期|漏洞|审阅|审查|\b(?:latest|recent|verify)\b',re.I)
# A conservative floor, not a substitute for the main model's semantic goal.
_PLURAL=re.compile(r'一些|若干|几(?:张|个|段|份)|多(?:张|个|段|份)|来点|\b(?:some|several|multiple)\b',re.I)
_BACKGROUND_CLAIM=re.compile(r'(?:还|正|仍)(?:在)?后台.{0,12}(?:下载|排队|处理|给你下|下着)|'
    r'后台(?:还|正|仍)(?:在)?.{0,8}(?:下载|排队|处理)|'
    r'\b(?:is|are) (?:still )?(?:downloading|queued|running) in (?:the )?background\b',re.I)
_READS={'search_web','open_web_page','find_web_page','inspect_web_resources','list_x_posts','memory_search','memory_read','search_history',
        'read_research_evidence','read_research_resources','read_forward_messages','list_web_images',
        'download_web_image','download_web_images','download_web_video','download_twitter_media','read_self_activity','search_x_posts','search_xhs_posts'}
_ARTIFACT_TOOLS={'download_web_image','download_web_images','download_web_video','download_twitter_media',
                 'read_artifact','media_result','generate_image','generate_private_image',
                 'search_image_references','execute_python','list_artifacts',
                 'observe_group_images','observe_forward_images','observe_post_images'}
_DOWNLOAD_TOOLS={'download_twitter_media','download_web_images','download_web_video'}
_REQUIREMENTS={'source_evidence','primary_source','media_delivery'}
TASK_POLICY=(
    '当前请求目标始终有轻量记录，不必先建正式研究任务。按用户目标选择工具，工具ok不等于完成目标。'
    '需要时用set_task_goal描述语义目标/交付条件；不扩权限或另开执行。'
    '媒体目标分别理解内容主体、发布者限制、日期、数量和类型；关于某人不等于仅限本人发帖，只有用户限定发布者时才限制来源。'
    '开放媒体搜集先用set_task_goal记录media_selection：一些/若干最低2份，期望可设3至5份，数量按实际不同媒体而非工具次数；'
    '仅排除用户已收到的相同素材，不能自行排除整个角色/服装、人物或来源；同一造型的新镜头仍可合格。用户举例不变成排他条件。'
    'X主题检索用search_x_posts的subject_terms（名字/本地名/完整账号别名）和topic_terms（主题多语言同义词），交给程序组合；小红书用search_xhs_posts，再核验原帖日期和正文；'
    '每组通常1至2个准确别名/同义词即可，4主题词是上限，不必凑满；不同类别不是同义词，不擅自添加。'
    '默认一次取12条后先筛已有候选，足够时queue_media_batch批量提交不同原帖；单条失败换剩余候选，不以换序或同义重复查询代替下载。'
    '图片和视频同时需要时media_types填两种。主题词召回不足可topic_match=review按主体补搜，再依据原帖正文核对主题；不反复空搜，不擅自限制发布者或编造热度门槛。'
    '用户要最新而严格主题词只找到较旧或已收过的素材时，先在同窗用review/Latest补查未写主题标签的较新候选，不根据旧关键词命中推断没有新内容。'
    '主题收集下载用review，在当前主回合查询状态并按原帖正文、日期、媒体类型筛选。默认不调用read_artifact逐份看图；仅文字有歧义且确有必要时少量看静态图片。视频和动图不预览、不抽帧，直接依据帖子文本判断。'
    '描述注明依据原帖，不声称看过视频、听过音频或鉴定过真实性；一帖提及多位主体而无法从文字分辨时，跳过或如实说明不确定。'
    '单帖筛完不足时继续取其他来源补足，不先发进度结束筛选；明确给定来源全部媒体原样下载才用auto。'
    '同一照片的封面、裁切、缩略图、不同分辨率不算多份；达不到份数时如实partial，不能凑数。'
    '未指定日期时publication_filter无下界；最近但未定天数只是软排序偏好，可先试7天但不足须扩大或取消窗口。lookback_days是检索策略，不能升格硬约束。'
    '用户明确近N天/周不能放宽。原帖近期发布不代表素材新拍，也不能据少量可见帖子说账号全是日常或没有新cos。'
    '图片或视频可任选，图片和视频尽量兼顾。内容相关性以帖子正文为主，不能只靠账号名或搜索命中；新发帖也可能是旧素材。'
    '缺原文、参数、日期或产物时优先从获准工具发现；摘要不能证明最新、没有发布或不支持。失败/空候选不证明全平台没有；受阻说明须写实际尝试的窗口与缺口。'
    '历史回复里“不能下载/没有工具”等口头断言不代表当前能力；以本轮工具目录、权限和实际执行结果为准，不能直接沿用旧拒绝。'
    '跨轮证据使用原结果task_id加evidence_id读取，不切换当前目标；本地证据定位失败后按当前请求恢复读取或搜索来源，不能当成网站下载失败。'
    '新闻报道日期、搜索收录日期和原帖发布日期是不同事实；下载最新或指定日期的帖子须核对原帖，缺证据时说明日期未核实。'
    '媒体请求可用search_web发现实际来源页内的原帖链接，站点条件限制最终媒体来源而非发现方法；站内无成果时按预算切换跨发布者、独立高精度handle或WebSearch，不机械遍历全部工具。优先source/ref_id，不使用生成摘要猜出的链接。'
    '一个报道404或一次页面未找到关键词只说明该次定位失败，不证明用户必须亲自提供链接。'
    '工具反馈的missing/no_progress用于调整方法；反复没有新证据时换策略，必要时review_task_progress最多一次短复核。'
    'send_group/send_private可标task_status=progress/partial/blocked/completed：进展提示与诚实受阻可以发送，'
    '后台承诺须有真实pending_jobs；本地校验拦截不代表已排队。已有合格媒体可先partial交付，仍有预算和候选继续找；首次发送不等于请求完成。'
    '按原帖正文/日期/类型确认合格后，用candidate_reviews标selected并记录依据，随后下载；这使程序在模型预算耗尽时能交付已有合格文件。未评估候选和未知回执不进入自动收尾。图片视频完成须本请求实际产物及真实回执；无依据不要宣称已完成或做不到。'
    '未知写入不重试，硬权限不绕过；有界尝试仍缺条件时说明具体尝试、缺口和下一步。')


def delivery_properties():
    return {'task_status':{'type':'string','enum':['progress','partial','blocked','completed'],
        'description':'本条交付的作用；排队提示用progress，实际受阻用blocked，尚有缺口用partial。'}}


class RequestTaskControl:
    def __init__(self,workspace,*,capability_version,initial_result=None):
        self.workspace=workspace
        self.version=capability_version
        text=workspace.request
        self.requires_media=bool((_MEDIA.search(text) and _TRANSFER.search(text) or _WANT.search(text))
            and not _ANALYZE.search(text))
        self.delivery='media' if self.requires_media else 'text'
        self.active=bool(self.requires_media or _TASK.search(text) or _EXTERNAL.search(text))
        self.goal=text[:1000]
        self.requirements=set()
        if self.active and (_EXTERNAL.search(text) or self.requires_media and _PRIMARY.search(text)):
            self.requirements.add('source_evidence')
            if _PRIMARY.search(text):self.requirements.add('primary_source')
        self._seed_requirements=set(self.requirements)
        self.kinds=set()
        self.artifacts=set()
        self.artifact_sources={}
        self.delivery_approved=set()
        self.asset_details={}
        self.asset_order=[]
        self.logical_media={}
        self.content_hashes={}
        self.video_artifacts=set()
        self.expected_artifacts=0
        self.minimum_media_count=2 if self.requires_media and _PLURAL.search(text) else 1
        self.media_selection={}
        self.constraint_evidence={}
        from app.core.post_freshness import constraint_text
        original=constraint_text(text)
        explicit_count=re.search(r'(?<!\d)(\d{1,6}|[零〇一二三四五六七八九十百千两]{1,6})\s*(?:张|个|段|份)',original)
        if explicit_count:
            value=explicit_count[1]
            if value.isdigit():number=int(value)
            else:
                number=0;digit=0
                digits={'零':0,'〇':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
                for char in value:
                    if char in digits:digit=digits[char]
                    else:number+=(digit or 1)*{'十':10,'百':100,'千':1000}[char];digit=0
                number+=digit
            self.minimum_media_count=max(self.minimum_media_count,number)
            self.constraint_evidence['count']={'proof':explicit_count[0],'provenance':'explicit_user'}
        image=bool(re.search(r'图片|原图|照片|图集|相片|图(?:和|与|及|、|片|$)|photos?|images?',original,re.I))
        video=bool(re.search(r'视频|videos?',original,re.I))
        alternatives=bool(re.search(r'任选|任意一种|(?:图(?:片)?|照片|images?|photos?).{0,8}(?:或|\bor\b|[/／]).{0,8}(?:视频|videos?)|(?:视频|videos?).{0,8}(?:或|\bor\b|[/／]).{0,8}(?:图(?:片)?|照片|images?|photos?)',original,re.I))
        self.requested_media_types=({'image'} if image else set())|({'video'} if video else set())
        if not self.requires_media or alternatives and len(self.requested_media_types)==2:self.requested_media_types=set()
        self.delivered_media_types={}
        original=constraint_text(text)
        self.publisher_proof=next((m[0] for m in re.finditer(
            r'(?:本人|自己)(?:的)?(?:账号|账户)?[^，。；;\n]{0,12}?(?:发布|上传|发帖|发的)|'
            r'@[A-Za-z0-9_]{1,15}\s*(?:的账号)?\s*(?:发布|上传|发帖|发的)|'
            r'\b(?:own account|posted by|from:[A-Za-z0-9_]+)',original,re.I)),None)
        account=re.search(r'@([A-Za-z0-9_]{1,15})(?![A-Za-z0-9_])',original) if self.publisher_proof else None
        self.explicit_publisher_account=account[1] if account else None
        self.requested_sites=set()
        for site,pattern in {'x':r'\b(?:x|twitter)\b|推特','xhs':r'小红书|xiaohongshu|xhslink','bilibili':r'哔哩哔哩|B站|bilibili','douyin':r'抖音|douyin'}.items():
            match=re.search(pattern,original,re.I)
            if match:self.requested_sites.add(site);self.constraint_evidence['site:'+site]={'proof':match[0],'provenance':'explicit_user'}
        if self.publisher_proof:self.media_selection['publisher_scope']='subject_only'
        self.delivered_artifacts=set()
        self.content_groups={}
        self._observed_photos={}
        self.image_artifacts=set()
        self.reviewed_images=set()
        self.candidate_reviews={}
        self.resource_urls=set()
        self.post_metadata={}
        self.source_refs={}
        self.attempted_resources=set()
        self.discovery_attempts=0
        self.publication_filter=publication_filter(text,datetime.fromtimestamp(workspace.clock(),UTC))
        self.published_after=recent_cutoff(text,datetime.fromtimestamp(workspace.clock(),UTC))
        value=self.publication_filter['published_before']
        self.published_before=datetime.fromisoformat(value) if value else None
        self.method_progress={}
        self.explicit_recent_days=explicit_recent_days(text)
        self.jobs={}
        self.job_status_tools={}
        self.calls=[]
        self.observations=[]
        self.failures=[]
        self.useful=set()
        self.reads=0
        self.stalled=0
        self.corrections=0
        self.review_used=False
        self.review_feedback={}
        self.last_delivery=None
        self.unknown=False
        self.stage='active'
        self._experience_saved=False
        self._last_preflight={}
        self._hints=None
        if self.requires_media:self._ensure_budget()
        if initial_result:
            results=initial_result if isinstance(initial_result,(tuple,list)) else (initial_result,)
            if any(isinstance(value,dict) and value.get('kind')=='media_result' for value in results):
                saved=workspace.media_selection_for_source()
                # Re-derive hard quantities from the original request, not legacy model preferences.
                self.constraint_evidence.update(saved.get('constraint_evidence',{}))
                if (saved.get('last_delivery') or {}).get('receipt_id'):
                    self.delivered_artifacts.update(saved.get('delivered_artifacts',[]))
                    self.content_groups.update(saved.get('delivered_content_groups',{}))
                    self.delivered_media_types.update(saved.get('delivered_media_types',{}))
                self.media_selection=dict(saved.get('media_selection',{}))
                if self.media_selection.get('lookback_days'):
                    self._use_media_window(self.media_selection['lookback_days'])
                evidence=saved.get('candidate_evidence',{})
                for resource in evidence.get('resources',[]):
                    self.resource_urls.add(resource['url'])
                    if resource.get('kind')=='inspected_post' and resource.get('post_id'):
                        self.post_metadata[resource['post_id']]=resource
                self.source_refs.update(evidence.get('source_refs',{}))
                self.attempted_resources.update(evidence.get('attempted_resources',[]))
                self.candidate_reviews.update(evidence.get('candidate_reviews',{}))
                self.discovery_attempts=evidence.get('discovery_attempts',0)
                if self.post_metadata:self.kinds.add('primary_metadata')
            for index,value in enumerate(results):
                if isinstance(value,dict):
                    media=value.get('kind')=='media_result'
                    delegated=value.get('source_bound') is True and bool(value.get('task_id'))
                    self.observe('media_result' if media else 'research_result' if delegated else 'initial_evidence',{},
                        ToolOutcome(value.get('data',value) if media else value,
                            value.get('status','ok') if media else
                            'blocked' if delegated and value.get('status') in {'blocked','failed'} else 'ok'),f'initial-{index}')

    def _ensure_budget(self):
        if getattr(self.workspace,'media_budget',None) is None:
            from app.agent.request_budget import RequestBudget
            self.workspace.media_budget=RequestBudget(self.workspace)
        return self.workspace.media_budget

    @property
    def needs_review(self):
        return self.active and (self.stalled>=2 or self.corrections>=2)

    def _content_count(self,identifiers):
        if self.media_selection.get('variants_requested') is True:return len(set(identifiers))
        return len({self.content_groups.get(i,i) for i in identifiers})

    @property
    def topic_selection(self):
        return self.requires_media and self.media_selection.get('selection_scope')!='all_from_explicit_source' and bool(
            self.media_selection.get('subject') or self.media_selection.get('selection_scope')=='topic')

    def _resource_key(self,reference):
        target=self.source_refs.get(reference,reference)
        from app.core.media_source import media_source_key
        return media_source_key(target)

    def _use_media_window(self,days):
        # A model-selected exploration window never changes user admission bounds.
        self.media_selection['lookback_days']=self.explicit_recent_days or days

    def _remaining_candidates(self):
        stale=set()
        if self.published_after or self.published_before:
            now=datetime.fromtimestamp(self.workspace.clock(),UTC)
            stale={self._resource_key(r['url']) for r in self.post_metadata.values()
                if (stamp:=publication_time(r.get('published_at'))) is not None
                and (stamp>now+timedelta(minutes=5)
                    or self.published_after is not None and stamp<self.published_after
                    or self.published_before is not None and stamp>=self.published_before)}
        return sorted(url for url in self.resource_urls if resource_ready(url)
            and self._resource_key(url) not in self.attempted_resources
            and self._resource_key(url) not in stale
            and self.candidate_reviews.get(self._resource_key(url),{}).get('decision')!='excluded')

    def _missing(self,*,before_delivery=False):
        gaps=[]
        if self.unknown:gaps.append('unknown_external_effect')
        if any(v in {'queued','pending','running','result_ready','delivering'} for v in self.jobs.values()):
            gaps.append('background_pending')
        if 'source_evidence' in self.requirements and not self.kinds.intersection(
                {'search_summary','delegated_summary','primary_text','primary_metadata','canonical_messages','memory_evidence','action_receipts','media_artifacts'}):
            gaps.append('source_evidence_missing')
        if 'primary_source' in self.requirements and not self.kinds.intersection(
                {'primary_text','primary_metadata','canonical_messages','action_receipts','media_artifacts'}):
            gaps.append('primary_source_unverified')
        if self.expected_artifacts>len(self.artifacts):gaps.append('resource_batch_incomplete')
        if self.requires_media and self.minimum_media_count>self._content_count(self.artifacts|self.delivered_artifacts):gaps.append('media_quantity_incomplete')
        if (self.requires_media or 'media_delivery' in self.requirements) and (
                before_delivery or not self.last_delivery or not self.last_delivery.get('media')):
            gaps.append('media_delivery_unconfirmed')
        available_types=({'image'} if self.image_artifacts else set())|({'video'} if self.video_artifacts else set())|set(self.delivered_media_types.values())
        if self.requested_media_types-available_types:gaps.append('media_types_incomplete')
        if self.last_delivery and self.requested_media_types-set(self.delivered_media_types.values()):gaps.append('media_delivery_types_incomplete')
        if self.last_delivery and self.requires_media and self._content_count(self.delivered_artifacts)<self.minimum_media_count:
            gaps.append('media_delivery_quantity_incomplete')
        return gaps

    def state(self):
        if not self.active:return {}
        status=self.stage
        if self.unknown:status='uncertain'
        elif 'background_pending' in self._missing():status='waiting'
        return {'source_id':self.workspace.source_id,'goal':self.goal,'delivery':self.delivery,
            'requirements':sorted(self.requirements),'status':status,
            'completion_gaps':self._missing(),'evidence_kinds':sorted(self.kinds),
            'artifact_count':len(self.artifacts),'attempts':len(self.calls),
            'expected_artifact_count':self.expected_artifacts,
            'minimum_media_count':self.minimum_media_count,'media_selection':self.media_selection,
            'constraint_evidence':self.constraint_evidence,
            'user_constraints':{'request':self.workspace.request,'source_id':self.workspace.source_id,
                'sites':sorted(self.requested_sites),'publisher_proof':self.publisher_proof,
                'publication_filter':self.publication_filter,'minimum_count':self.minimum_media_count,'media_types':sorted(self.requested_media_types)},
            'published_after':self.published_after.isoformat() if self.published_after else None,
            'publication_filter':self.publication_filter,'method_progress':self.method_progress,
            'delivered_media_count':self._content_count(self.delivered_artifacts),
            'distinct_media_count':self._content_count(self.artifacts),
            'remaining_media_candidates':self._remaining_candidates()[:12],
            'visual_review_required':False,'selection_basis':'post_text_primary',
            'reviewed_image_count':len(self.reviewed_images),
            'pending_jobs':[{'tool':self.job_status_tools.get(job_id,'media_task_status'),
                'arguments':{'task_id' if self.job_status_tools.get(job_id)=='private_image_status' else 'job_id':job_id},
                'observed_status':status} for job_id,status in self.jobs.items()
                if status in {'queued','pending','running','result_ready','delivering'}][:16],
            'duplicate_media':[{'artifact_id':i,'same_content_as':g} for i,g in self.content_groups.items() if i!=g],
            'no_progress':self.stalled>=2,'corrections':self.corrections,
            'semantic_coverage':'not_independently_proven'}

    def describe(self,args):
        self.workspace._check()
        reviews={}
        for item in args.get('media_selection',{}).get('candidate_reviews',[]):
            key=self._resource_key(item['ref_id'])
            if not any(self._resource_key(url)==key for url in self.resource_urls):
                raise ValueError('task_candidate_not_observed')
            reviews[key]={'decision':item['decision'],'reason':item['reason'][:300]}
        self.candidate_reviews.update(reviews)
        self.active=True
        self.goal=args.get('goal',self.goal)[:1000]
        self.delivery='media' if self.requires_media else args.get('delivery','text')
        self.requires_media=self.delivery=='media'
        if self.requires_media:self._ensure_budget()
        selection=dict(args.get('media_selection',{}))
        from app.core.post_freshness import constraint_text
        original=constraint_text(self.workspace.request)
        supplied=selection.pop('constraint_evidence',{})
        for field,proof in supplied.items():
            if not proof or proof not in original:raise ValueError('user_constraint_evidence_unverified')
            self.constraint_evidence[field]={'proof':proof,'provenance':'explicit_user'}
        if self.publisher_proof:selection['publisher_scope']='subject_only'
        if self.explicit_publisher_account:selection['publisher_account']=self.explicit_publisher_account
        if selection.get('publisher_scope')=='subject_only' and not self.publisher_proof:
            selection['publisher_scope']='any'
        if 'minimum_count' in selection:
            # A suggested target is a preference until backed by the user wording.
            selection['target_count']=max(selection.get('target_count',1),selection.pop('minimum_count'))
        self.media_selection.update({k:v for k,v in selection.items() if k!='minimum_count'})
        # Hard quantity is re-derived from the canonical user request.
        if 'lookback_days' in selection:
            self._use_media_window(selection['lookback_days'])
        self.requirements=self._seed_requirements|set(args.get('requirements',[])).intersection(_REQUIREMENTS)
        self._persist()
        return {**self.state(),'authority':'unchanged; current canonical request only'}

    def prepare_tool_arguments(self,name,args):
        """Resolve observed source aliases and narrow automatic topic delivery."""
        def reviewed(operation,arguments):
            prepared=arguments
            if operation in _DOWNLOAD_TOOLS:
                from app.core.post_freshness import constraint_text
                strict=bool(re.search(r'必须.{0,4}原图|只要原图|不接受.{0,6}(?:降档|压缩)|original only',constraint_text(self.workspace.request),re.I))
                if strict:prepared={**prepared,'quality':'original_only'}
                elif arguments.get('quality')=='original_only':prepared={**prepared,'quality':'best_available'}
                ref=arguments.get('ref_id')
                if ref in self.source_refs:prepared={**prepared,'ref_id':self.source_refs[ref]}
                if self.topic_selection and arguments.get('delivery_mode')=='auto':
                    prepared={**prepared,'delivery_mode':'review'}
            return prepared
        if name in {'send_group','send_private'}:
            ids=set(args.get('artifact_ids',[]))|set(args.get('media_artifact_ids',[]))
            if ids-self.delivered_artifacts and ids&self.delivered_artifacts:
                args={**args,**{key:[i for i in args[key] if i not in self.delivered_artifacts]
                    for key in ('artifact_ids','media_artifact_ids') if key in args}}
        if name!='queue_media_batch':
            return reviewed(name,args)
        entries=[];changed=False
        for entry in args.get('operations',[]):
            arguments=reviewed(entry['operation'],entry['arguments'])
            changed=changed or arguments is not entry['arguments']
            entries.append({**entry,'arguments':arguments})
        return {**args,'operations':entries} if changed else args

    def _check_publication(self,post_id):
        metadata=self.post_metadata.get(post_id,{})
        stamp=publication_time(metadata.get('published_at'))
        now=datetime.fromtimestamp(self.workspace.clock(),UTC)
        if (metadata.get('identity_status')!='matched' or not metadata.get('author')
                or stamp is None or stamp>now+timedelta(minutes=5)
                or self.published_after is not None and stamp<self.published_after
                or self.published_before is not None and stamp>=self.published_before):
            return ToolOutcome({'error':'resource_freshness_unverified',
                'handler_executed':False,'background_queued':False,
                'published_after':self.published_after.isoformat() if self.published_after else None,
                'publication_filter':self.publication_filter,
                'published_at':metadata.get('published_at',''),
                'guidance':'本次未执行下载、未排队。按published_after核对原帖日期，不得用超窗或未知日期冒充近期。优先read_research_evidence恢复原查询证据或inspect_web_resources核验该帖；不符则换剩余候选，再按需补搜。用户未限定天数可声明合理lookback_days再查并说明日期，明确近N天不可放宽。'},'blocked')
        return None

    def _source_constraints(self,source):
        from urllib.parse import urlsplit
        key=self._resource_key(source)
        host=urlsplit(source).hostname or ''
        if key.startswith('source:'):host=key.split(':',2)[1]
        def hosted(domain):return host==domain or host.endswith('.'+domain)
        site=('x' if key.startswith('x:') else 'xhs' if key.startswith('xhs:') or (hosted('xiaohongshu.com') or hosted('xhslink.com')) else
            'bilibili' if (hosted('bilibili.com') or hosted('b23.tv')) else 'douyin' if hosted('douyin.com') else host)
        if self.requested_sites and site not in self.requested_sites:return 'user_source_site_mismatch'
        if self.publisher_proof:
            publisher=(self.explicit_publisher_account or self.media_selection.get('publisher_account','')).lstrip('@').casefold()
            row=next((r for r in self.post_metadata.values() if self._resource_key(r['url'])==key),{})
            if not publisher or row.get('identity_status')!='matched' or str(row.get('author','') if site=='x' else row.get('author_id','')).lstrip('@').casefold()!=publisher:
                return 'user_publisher_unverified'
        return None

    def preflight(self,name,args):
        downloads=(args.get('operations',[]) if name=='queue_media_batch' else
            [{'operation':name,'arguments':args}])
        if self.topic_selection and any(e.get('operation') in _DOWNLOAD_TOOLS
                and e.get('arguments',{}).get('delivery_mode')=='auto' for e in downloads):
            return ToolOutcome({'error':'media_topic_review_required',
                'handler_executed':False,'suggested_delivery_mode':'review',
                'guidance':'主题搜集用review下载，media_task_status取结果，按原帖正文/日期/类型筛选后交付；不要求视觉预览，不查看视频或动图。单帖不足继续其他来源，不重复下载同帖，不用auto提前交接。'},'blocked')
        if (self.requires_media and self.minimum_media_count>1 and downloads
                and all(e.get('operation') in {'download_twitter_media','download_web_images','download_web_video'}
                    and e.get('arguments',{}).get('delivery_mode')=='auto' for e in downloads)):
            selected={}
            for entry in downloads:
                operation,arguments=entry['operation'],entry['arguments']
                if operation=='download_web_images':
                    for identifier in arguments.get('image_ids',[]):selected[('image',identifier)]=1
                elif operation=='download_web_video':
                    selected[('video',arguments.get('ref_id',''))]=1
                else:
                    from app.providers.twitter_media import parse_post_url
                    target=self.source_refs.get(arguments.get('ref_id'),arguments.get('ref_id',''))
                    try:pid=parse_post_url(target).post_id
                    except ValueError:continue
                    metadata=self.post_metadata.get(pid,{})
                    count=metadata.get('media_count',0)
                    selected[('post',pid)]=count if (metadata.get('identity_status')=='matched'
                        and type(count) is int and 0<count<=16) else 0
            if sum(selected.values())<self.minimum_media_count:
                return ToolOutcome({'error':'media_selection_quantity_unverified',
                    'minimum_media_count':self.minimum_media_count,'verified_selected_count':sum(selected.values()),
                    'guidance':'单份不能完成多份请求。先发现足够不同且相关的候选，再queue_media_batch一次提交；'
                        '或用review下载并核对实际产物后汇总交付。仅找到少量时如实partial说明，不把排队当完成。'},'blocked')
        for entry in downloads:
            arguments=entry.get('arguments',{})
            if entry.get('operation') in _DOWNLOAD_TOOLS|{'download_web_image'}:
                refs=arguments.get('image_ids',[]) or [arguments.get('ref_id') or arguments.get('image_id')]
                for ref in refs:
                    target=self.source_refs.get(ref,ref)
                    if target and (problem:=self._source_constraints(target)):
                        return ToolOutcome({'error':problem,'handler_executed':False,'guidance':'保留用户的最终来源站点和明确发布者要求；可用WebSearch发现该站原帖，但不能交付其他站点媒体。'},'blocked')
        if self.published_after is not None or self.published_before is not None:
            for entry in downloads:
                operation=entry.get('operation')
                arguments=entry.get('arguments',{})
                targets=([arguments.get('ref_id','')] if operation in {'download_twitter_media','download_web_video'} else
                    arguments.get('image_ids',[]) if operation=='download_web_images' else
                    [arguments.get('image_id','')] if operation=='download_web_image' else [])
                if not targets:continue
                for raw_target in targets:
                    target=self.source_refs.get(raw_target,raw_target)
                    if operation=='download_twitter_media':
                        from app.providers.twitter_media import parse_post_url
                        try:post_id=parse_post_url(target).post_id
                        except ValueError:post_id=None
                    else:
                        from app.providers.xhs_crawler_client import note_target
                        try:post_id=note_target(target)[0]
                        except ValueError:
                            from urllib.parse import urlsplit
                            if urlsplit(target).hostname not in {'xhslink.com','www.xhslink.com'}:continue
                            post_id=None
                    failure=self._check_publication(post_id)
                    if failure is not None:return failure
        if name not in {'send_group','send_private'} or not self.active:return None
        checkpoint=self.workspace.execution_status()
        declared=checkpoint.get('status') if checkpoint.get('status') in {'partial','blocked'} else 'completed'
        stage=args.get('task_status',declared)
        ids=set(args.get('artifact_ids') or [])|set(args.get('media_artifact_ids') or [])
        if args.get('video_artifact_id'):ids.add(args['video_artifact_id'])
        missing=[]
        if self.requested_sites or self.publisher_proof or self.publication_filter['provenance']=='explicit_user':
            for identifier in ids:
                source=self.artifact_sources.get(identifier)
                if not source or self._source_constraints(source):
                    missing.append('user_media_source_unverified');continue
                if self.publication_filter['provenance']=='explicit_user':
                    post=next((pid for pid,row in self.post_metadata.items() if self._resource_key(row['url'])==source),None)
                    if self._check_publication(post) is not None:missing.append('user_media_date_unverified')
        pending='background_pending' in self._missing()
        if not pending and _BACKGROUND_CLAIM.search(args.get('text','')):
            missing.append('background_claim_unverified')
        if stage in {'blocked','partial'}:
            if pending and not ids:missing.append('background_pending')
            if (not ids and self.requires_media and self.minimum_media_count>1 and self.corrections<2
                    and self._content_count(self.artifacts|self.delivered_artifacts)>=self.minimum_media_count
                    and self._content_count(ids|self.delivered_artifacts)<self.minimum_media_count):
                missing.append('available_media_not_delivered')
            if (not ids and self.requires_media and self.minimum_media_count>1
                    and self._content_count(ids|self.delivered_artifacts)<self.minimum_media_count
                    and self.corrections<2 and self._remaining_candidates()
                    and not any(f['category'] in {'authority','capacity','access','unknown_effect'} for f in self.failures)):
                missing.append('media_candidates_remaining')
            elif (not ids and self.requires_media and self.minimum_media_count>1 and self.topic_selection
                    and self._content_count(ids|self.delivered_artifacts)<self.minimum_media_count
                    and self.corrections<2 and self.resource_urls
                    and self.discovery_attempts<2
                    and not any(f['category'] in {'authority','capacity','access','unknown_effect'} for f in self.failures)):
                missing.append('media_discovery_incomplete')
            if stage=='blocked' and ids:missing.append('failure_notice_cannot_claim_media_completion')
            attempts=[call for call in self.calls if call['tool'] not in {
                'read_context','read_private_context','begin_research_task','update_research_task',
                'read_research_task','remember_intention','save_work_note'}]
            if not attempts and not self.failures:missing.append('no_actual_attempt_or_blocker')
            external_attempts = [call for call in attempts if call['tool'] not in {
                'read_research_evidence', 'read_research_resources', 'resume_research_task',
                'send_group', 'send_private','media_result'} and call.get('handler_executed') is not False]
            if (self.requires_media and self.failures and not external_attempts
                    and all(failure['category'] == 'local_lookup' for failure in self.failures)):
                missing.append('recoverable_lookup_not_task_blocker')
            if (stage=='blocked' and not self.failures and self.reads<2 and set(self._missing()).intersection(
                    {'source_evidence_missing','primary_source_unverified'})):
                missing.append('insufficient_attempts_before_blocked')
            if (self.requires_media and not ids and self.corrections < 2
                    and any(resource_ready(url) and self.candidate_reviews.get(self._resource_key(url),{}).get('decision')!='excluded'
                        for url in self.resource_urls)
                    and not any(call['tool'] in {'download_twitter_media','download_web_image',
                        'download_web_images','download_web_video','queue_media_batch'} for call in self.calls)
                    and not any(failure['category'] in {'authority','capacity','access'} for failure in self.failures)):
                missing.append('discovered_media_not_attempted')
            if ids-self.artifacts:missing.append('artifact_not_observed_for_request')
        elif stage=='progress':
            pass
        else:
            missing+=self._missing(before_delivery=True)
            if ids and ids.issubset(self.artifacts):
                missing=[x for x in missing if x!='media_delivery_unconfirmed']
                if self.requires_media and self._content_count(ids|self.delivered_artifacts)<self.minimum_media_count:
                    missing.append('media_delivery_quantity_incomplete')
                else:missing=[x for x in missing if x!='media_delivery_quantity_incomplete']
            if ids-self.artifacts:missing.append('artifact_not_observed_for_request')
            selected_types=set(self.delivered_media_types.values())|({'image'} if ids&self.image_artifacts else set())|({'video'} if ids&self.video_artifacts else set())
            if self.requested_media_types-selected_types:missing.append('media_delivery_types_incomplete')
            else:missing=[gap for gap in missing if gap not in {'media_types_incomplete','media_delivery_types_incomplete'}]
        if not missing:
            self.delivery_approved.update(ids & self.artifacts)
            self._last_preflight={'stage':stage,'media':bool(ids),'artifact_ids':sorted(ids)}
            return None
        self.corrections=min(3,self.corrections+1)
        self._persist()
        return ToolOutcome({'error':'task_completion_unverified','missing':list(dict.fromkeys(missing)),
            'task':self.state(),'review_available':not self.review_used,
            'guidance':'尚未达到当前目标，未发送。按缺口获取原文/资源/产物后交付；实际尝试仍受阻时'
                '可用task_status=blocked或partial如实说明范围与原因，不能说未发布或无工具。'
                +('background_pending可能是已完成任务的旧观察。先按task.pending_jobs中列出的tool和arguments刷新状态；不要重新下载或生成，也不要更换目标。'
                  if 'background_pending' in missing else '')
                +('旧证据定位失败不是下载失败；用原task_id重读，或按当前目标搜索/打开来源，不能直接要求用户代找链接。'
                  if 'recoverable_lookup_not_task_blocker' in missing else '')
                +('尚未满足最低份数；先用task.remaining_media_candidates中的其他原帖补足，可queue_media_batch批量review下载。日期缺证据先恢复/核验；不符候选记录排除原因，实际失败/耗尽才partial。'
                  if 'media_candidates_remaining' in missing else '')
                +('已经取得足够不同产物，先从本请求已观察句柄补足交付；不必重复下载。若实际不相关则如实说明排除依据与缺口。'
                  if 'available_media_not_delivered' in missing else '')
                +('当前候选已用尽但最低份数未满足，按同一主题补查一次；保留用户日期/类型/作者条件，严格主题不足可review。仍不足再如实partial，不反复空搜。'
                  if 'media_discovery_incomplete' in missing else '')
                +('没有观察到排队或运行任务，不得宣称后台还在下载。改为说明未启动或实际失败；只有真实pending_jobs可发progress等待。'
                  if 'background_claim_unverified' in missing else '')
                +('已有本轮可下载的原帖或媒体候选，先用对应下载工具核验；不能略过后要求用户代找链接。'
                  '若候选实际不符，用set_task_goal.media_selection.candidate_reviews逐项记录ref_id和排除原因后如实partial；不得下载无关内容凑数。'
                  if 'discovered_media_not_attempted' in missing else '')
                +('纠偏预算已到建议上限，避免重复搜索，按已有实际尝试报告缺口。' if self.corrections>=2 else '')},'blocked')

    def observe(self,name,args,outcome,call_id):
        # Other goals are inferred by the main model. Once it actually starts
        # work, the lightweight record applies even without a lexical hint or
        # an explicit set_task_goal. Plain chat/context reads stay cheap.
        if name not in {'send_group','send_private','read_context','read_private_context',
                'read_persona','set_task_goal','review_task_progress','initial_evidence'}:
            self.active=True
        descriptor=evidence_descriptor(name,args,outcome)
        data=outcome.data
        if name in {'search_x_posts','list_x_posts','search_web','search_xhs_posts'} and not data.get('cached_in_turn'):
            self.discovery_attempts=min(4,self.discovery_attempts+1)
        if (outcome.status!='uncertain' and data.get('handler_executed') is not False
                and not (descriptor['failure'] and descriptor['failure']['category']=='local_lookup')):
            operations=args.get('operations',[]) if name=='queue_media_batch' else [{'operation':name,'arguments':args}]
            for index,entry in enumerate(operations):
                arguments=entry.get('arguments',{})
                # A batch can accept only some entries. Unaccepted entries remain candidates.
                if name=='queue_media_batch' and not any(j.get('index')==index for j in data.get('jobs',[])):continue
                if entry.get('operation') in _DOWNLOAD_TOOLS and arguments.get('ref_id'):
                    self.attempted_resources.add(self._resource_key(arguments['ref_id']))
        if name not in {'set_task_goal','review_task_progress'}:
            excerpt=data.get('content') or data.get('evidence') or ''
            self.observations.append({'tool':name,'status':outcome.status,'kind':descriptor['kind'],
                'excerpt':str(excerpt)[:600], 'coverage':descriptor['coverage'],
                'absence_proven':False})
            self.observations=self.observations[-6:]
        old=len(self.useful)
        if name not in {'set_task_goal','review_task_progress','initial_evidence'}:
            self.calls.append({'id':call_id,'tool':name,'status':outcome.status,
                **({'handler_executed':False} if data.get('handler_executed') is False else {})})
            self.calls=self.calls[-32:]
        if descriptor['failure']:
            self.failures.append(descriptor['failure'])
            self.failures=self.failures[-4:]
        if outcome.status=='uncertain':self.unknown=True
        if outcome.status in {'ok','sent'}:
            if self.requires_media and name=='search_x_posts' and args.get('subject_terms') and args.get('topic_terms'):
                self.media_selection['selection_scope']='topic'
            if (self.requires_media and name in {'list_x_posts','search_x_posts'}
                    and type(args.get('lookback_days')) is int and 1<=args['lookback_days']<=90):
                self._use_media_window(args['lookback_days'])
            kind=descriptor['kind']
            if kind!='tool_result':self.kinds.add(kind)
            if kind in {'primary_text','primary_metadata','canonical_messages','quoted_messages','action_receipts','media_artifacts','media_candidates'} and not data.get('cached_in_turn'):
                identity=(name,str(args.get('ref_id') or args.get('query') or data.get('artifact_id') or
                    data.get('evidence_id') or data.get('source_message_ids') or data.get('source_ids') or ''),
                    str(args.get('lineno','')))
                self.useful.add(identity)
            for resource in descriptor['resources']:
                self.resource_urls.add(resource['url'])
                if resource.get('kind')=='inspected_post' and resource.get('post_id'):
                    self.post_metadata[resource['post_id']]=resource
                    self.source_refs[resource['post_id']]=resource['url']
                if not self.requires_media or resource_ready(resource['url']):
                    self.useful.add(('resource',resource['url']))
            for source in data.get('sources',[])[:20]:
                if isinstance(source,dict) and isinstance(source.get('ref_id'),str) and isinstance(source.get('source'),str):
                    self.source_refs[source['ref_id']]=source['source']
            if name=='list_web_images' and data.get('identity_status')=='matched':
                if isinstance(args.get('ref_id'),str):self.source_refs[args['ref_id']]=data.get('page_url','')
                for image in data.get('images',[])[:20]:
                    if isinstance(image,dict) and isinstance(image.get('image_id'),str):
                        self.source_refs[image['image_id']]=data.get('page_url','')
            if name=='inspect_web_resources':
                for item in data.get('resources',[])[:6]:
                    if not isinstance(item,dict) or item.get('identity_status')!='matched':continue
                    index=item.get('input_index')
                    if type(index) is not int:continue
                    index-=1
                    refs=args.get('ref_ids',[])
                    if 0<=index<len(refs) and isinstance(refs[index],str) and item.get('source_url'):
                        self.source_refs[refs[index]]=item['source_url']
            artifact_data=data
            if name=='media_task_status':
                result=data.get('result')
                artifact_data=result.get('data',{}) if isinstance(result,dict) else {}
            if name in _ARTIFACT_TOOLS or name=='media_task_status':
                requested=artifact_data.get('requested_count')
                if type(requested) is int and requested>0:
                    self.expected_artifacts=max(self.expected_artifacts,min(requested,20))
                identifiers=[]
                for key in ('media_artifact_ids','artifact_ids','video_artifact_ids'):
                    identifiers.extend(v for v in artifact_data.get(key,[]) if isinstance(v,str))
                for key in ('artifact_id','video_artifact_id'):
                    if isinstance(artifact_data.get(key),str):identifiers.append(artifact_data[key])
                for artifact in artifact_data.get('artifacts',[])[:20]:
                    if isinstance(artifact,dict):
                        identifier=artifact.get('artifact_id') or artifact.get('id')
                        if isinstance(identifier,str):identifiers.append(identifier)
                for key in ('images','references'):
                    for item in artifact_data.get(key,[])[:20]:
                        if isinstance(item,dict) and isinstance(item.get('artifact_id'),str):
                            identifiers.append(item['artifact_id'])
                if name=='read_artifact' and outcome.images and args.get('artifact_id'):
                    identifiers.append(args['artifact_id'])
                for identifier in identifiers[:20]:
                    if identifier not in self.asset_order:self.asset_order.append(identifier)
                self.artifacts.update(identifiers[:20])
                self.video_artifacts.update(artifact_data.get('video_artifact_ids',[]))
                if artifact_data.get('video_artifact_id'):self.video_artifacts.add(artifact_data['video_artifact_id'])
                source=artifact_data.get('page_url') or artifact_data.get('source_url')
                if source:
                    for identifier in identifiers:self.artifact_sources[identifier]=self._resource_key(source)
                for item in artifact_data.get('media',[]):
                    if isinstance(item,dict) and item.get('artifact_id'):
                        identifier=item['artifact_id']
                        self.asset_details[identifier]={k:item[k] for k in ('kind','bytes','quality','image_size') if k in item}
                        if item.get('source_ref'):self.artifact_sources[identifier]=self._resource_key(item['source_ref'])
                        self._observe_media_identity(identifier,item.get('logical_media_id'),item.get('sha256'))
                if len(set(identifiers))==1:
                    self.asset_details.setdefault(identifiers[0],{}).update({k:artifact_data[k] for k in ('bytes','quality','width','height') if k in artifact_data})
                if artifact_data.get('logical_media_id') and len(set(identifiers))==1:
                    identifier=identifiers[0]
                    self._observe_media_identity(identifier,artifact_data['logical_media_id'],artifact_data.get('sha256'))
                if name in {'download_twitter_media','download_web_images','media_result','media_task_status',
                        'generate_image','generate_private_image'}:
                    self.image_artifacts.update(v for v in artifact_data.get('artifact_ids',[]) if isinstance(v,str))
                if name in {'download_web_image','observe_group_images','observe_forward_images','observe_post_images'}:
                    self.image_artifacts.update(identifiers)
                if name in {'download_web_image','read_artifact'} and identifiers and outcome.images:
                    from app.agent.media_identity import observed_thumbnail,same_observed_photo
                    identifier=identifiers[0]
                    thumbnail=observed_thumbnail(outcome.images[0])
                    if thumbnail is not None and len(self._observed_photos)<20:
                        self.reviewed_images.add(identifier)
                        self.image_artifacts.add(identifier)
                        group=next((old for old,pixels in self._observed_photos.items()
                            if same_observed_photo(pixels,thumbnail)),identifier)
                        self.content_groups[identifier]=self.content_groups.get(group,group)
                        self._observed_photos[identifier]=thumbnail
                if identifiers:self.kinds.add('media_artifacts')
            entries=data.get('jobs',[]) if name=='queue_media_batch' else [data]
            for entry in entries:
                if isinstance(entry,dict) and type(entry.get('job_id')) is int:
                    self.jobs[entry['job_id']]=entry.get('execution_status',entry.get('status'))
                    self.job_status_tools[entry['job_id']]='media_task_status'
            if name in {'generate_private_image','private_image_status'} and type(data.get('task_id')) is int:
                self.jobs[data['task_id']]=data.get('status')
                self.job_status_tools[data['task_id']]='private_image_status'
            if name in {'send_group','send_private'} and outcome.status=='sent':
                self.last_delivery={**(self._last_preflight or {'stage':args.get('task_status','completed'),
                    'media':bool(args.get('artifact_ids') or args.get('media_artifact_ids') or args.get('video_artifact_id')),
                    'artifact_ids':list(set(args.get('artifact_ids') or [])|set(args.get('media_artifact_ids') or [])|
                        ({args['video_artifact_id']} if args.get('video_artifact_id') else set()))}),
                    'receipt_id':outcome.receipt_id}
                if outcome.receipt_id:self.delivered_artifacts.update(
                    outcome.data.get('confirmed_artifact_ids',self.last_delivery.get('artifact_ids',[])))
                if outcome.receipt_id:
                    for identifier in self.delivered_artifacts:
                        if identifier in self.image_artifacts:self.delivered_media_types[identifier]='image'
                        elif identifier in self.video_artifacts:self.delivered_media_types[identifier]='video'
                stage=self.last_delivery['stage']
                self.stage=('waiting' if self.jobs and stage=='progress' else
                    'active' if stage=='progress' else stage)
                if stage=='completed' and self._missing():self.stage='partial'
        if name in _READS:
            previous=self.method_progress.get(name,{'calls':0,'no_gain':0})
            self.method_progress[name]={'calls':previous['calls']+1,
                'no_gain':previous['no_gain']+1 if len(self.useful)==old else 0,
                'last_gain':len(self.useful)-old}
            self.reads+=1
            self.stalled=self.stalled+1 if len(self.useful)==old else 0
        if self.active:self._persist()
        if self.active and self.stage=='completed' and not self._missing():self._remember()
        return {'new_evidence':len(self.useful)-old,'no_progress':self.stalled>=2,
            'missing':self._missing(before_delivery=not bool(self.last_delivery)),
            'evidence_kind':descriptor['kind'],'failure':descriptor['failure'],
            'guidance':'根据实际缺口调整方法；搜索摘要不证明最新/原文或没有结果。'
                if self.stalled>=2 else '按实际资源、证据和回执判断下一步。'}

    @property
    def can_stop(self):
        if self.unknown or not self.last_delivery:return False
        if self.stage=='partial' and (self._remaining_candidates() or self.artifacts-self.delivered_artifacts
                or 'background_pending' in self._missing()):return False
        return (self.stage in {'completed','blocked','partial','waiting'} and
                (self.stage!='completed' or not self._missing()))

    def _observe_media_identity(self,identifier,logical,digest):
        if logical:self.logical_media[identifier]=logical;self.content_groups[identifier]=logical
        if digest:
            previous=self.content_hashes.get(digest)
            if previous:
                before=self.content_groups.get(identifier,identifier)
                after=self.content_groups.get(previous,previous)
                for previous_identifier in [i for i,g in self.content_groups.items() if g==before]:
                    self.content_groups[previous_identifier]=after
                self.content_groups[identifier]=after
            self.content_hashes[digest]=identifier

    def approved_artifacts(self):
        if self.unknown:return []
        from urllib.parse import urlsplit
        explicit={self._resource_key(url.rstrip('，。；）)')) for url in re.findall(r'https?://[^\s<>]+',self.workspace.request)}
        approved=[]
        for identifier in self.asset_order:
            if identifier in self.delivered_artifacts:continue
            source=self.artifact_sources.get(identifier)
            if identifier in self.delivery_approved or source and not self._source_constraints(source) and (source in explicit or self.candidate_reviews.get(source,{}).get('decision')=='selected'):
                if self.publication_filter['provenance']=='explicit_user':
                    post=next((pid for pid,row in self.post_metadata.items() if self._resource_key(row['url'])==source),None)
                    if self._check_publication(post) is not None:continue
                approved.append(identifier)
        return approved

    def _persist(self):
        resources=list(self.post_metadata.values())[-32:]
        known={r['url'] for r in resources}
        resources+=[{'kind':'web_url','url':url} for url in sorted(self.resource_urls-known)][:(32-len(resources))]
        candidates={'resources':resources,'source_refs':dict(list(self.source_refs.items())[-32:]),
            'attempted_resources':sorted(self.attempted_resources)[-32:],
            'candidate_reviews':dict(list(self.candidate_reviews.items())[-32:]),
            'discovery_attempts':self.discovery_attempts}
        self.workspace.save_request_control({'goal':self.goal,'active':self.active,**self.state(),'failures':self.failures,
            'calls':self.calls,'last_delivery':self.last_delivery,'candidate_evidence':candidates,
            'delivered_artifacts':sorted(self.delivered_artifacts),'delivered_media_types':self.delivered_media_types,
            'delivered_content_groups':{i:self.content_groups.get(i,i) for i in self.delivered_artifacts}})

    def _remember(self):
        if self._experience_saved or not self.calls or not self.last_delivery.get('receipt_id'):return
        last={call['tool']:call['status'] for call in self.calls}
        if any(status in {'failed','blocked','uncertain'} for status in last.values()):return
        self._experience_saved=True
        self.workspace.save_execution_experience(self.version,{'delivery':self.delivery,
            'requirements':sorted(self.requirements),'tools':list(dict.fromkeys(x['tool'] for x in self.calls))[-8:],
            'failures':[x['code'] for x in self.failures],
            'verified_basis':'observed_tools_and_confirmed_request_delivery'})

    def experiences(self):
        try:return self.workspace.execution_experiences(self.version,self.delivery)
        except (ValueError,OSError,TypeError):return []

    def method_hints(self):
        if not self.active:return []
        if self._hints is None:
            self._hints=[{key:item[key] for key in ('tools','requirements','failures') if key in item}
                for item in self.experiences()[-2:]]
        return self._hints

    def tools(self,guard,review=None):
        def obj(properties,required=()):
            return {'type':'object','properties':properties,'required':list(required),'additionalProperties':False}
        async def describe(args):
            guard.require_valid()
            return ToolOutcome(self.describe(args))
        async def reflect(args):
            guard.require_valid()
            advice={}
            if args.get('model_review',True) and self.needs_review and not self.review_used and review:
                self.review_used=True
                advice=await review(self,args.get('proposed_answer',''))
                guard.require_valid()
                self.review_feedback=advice
            return ToolOutcome({'task':self.state(),'actual_failures':self.failures,
                'review':advice,'authority':'diagnostic only; executor restrictions unchanged'})
        return (
            AgentTool('set_task_goal','明确当前真实请求的目标、交付和证据要求；轻量记录，不增加权限或研究预算',obj({
                'goal':{'type':'string','minLength':1,'maxLength':1000},
                'delivery':{'type':'string','enum':['text','media']},
                'requirements':{'type':'array','items':{'type':'string','enum':sorted(_REQUIREMENTS)},'maxItems':5},
                'media_selection':obj({
                    'constraint_evidence':obj({key:{'type':'string','minLength':1,'maxLength':200} for key in ('subject','publisher','site','date','count','quality')}),
                    'subject':{'type':'string','maxLength':200,'description':'媒体内容关于谁/什么，与发帖作者分开'},
                    'publisher_account':{'type':'string','maxLength':160,'description':'仅用户明确限定发布者时记录已从来源核实的账号，与内容主体别名分开'},
                    'publisher_scope':{'type':'string','enum':['any','subject_only'],
                        'description':'仅在用户明确要求本人发帖时选subject_only；关于某人的媒体选any'},
                    'minimum_count':{'type':'integer','minimum':1,'maximum':16,
                        'description':'明确数量附constraint_evidence.count原话；无依据时仅作target_count偏好，一些最低2已由程序记录'},
                    'target_count':{'type':'integer','minimum':1,'maximum':16,
                        'description':'可选的期望份数，例如一些通常争取3至5份；不替代用户最低要求，不强迫凑数'},
                    'variants_requested':{'type':'boolean',
                        'description':'仅用户明确要原图/裁切/尺寸等多个变体时为true，按文件计数。普通搜集一些媒体保持false，不能靠变体凑份数'},
                    'selection_scope':{'type':'string','enum':['topic','all_from_explicit_source'],
                        'description':'主题搜集选topic，以原帖正文为主筛选，不强制图片/视频预览；仅用户明确要求给定来源全部媒体原样发送时选all_from_explicit_source'},
                    'lookback_days':{'anyOf':[{'type':'integer','minimum':1,'maximum':365},{'type':'null'}],
                        'description':'可选软检索窗口；null无下界，0无效。不会改变用户明确日期硬门，交付说明实际日期/覆盖'},
                    'candidate_reviews':{'type':'array','maxItems':16,'items':obj({
                        'ref_id':{'type':'string','minLength':1,'maxLength':2000},
                        'decision':{'type':'string','enum':['selected','excluded','uncertain']},
                        'reason':{'type':'string','minLength':1,'maxLength':300}},('ref_id','decision','reason')),
                        'description':'只评估本轮已观察候选。排除日常/同名/非本人/重复等不符项，必须给具体依据；排除后可如实partial，不必下载无关素材凑数'},
                    'criteria':{'type':'string','maxLength':600,'description':'日期含义、内容相关性及图片/视频偏好；候选以原帖正文判断，视频和动图不预览'}})} ,
                ('goal','delivery','requirements')),describe,side_effects=('local_state',)),
            AgentTool('review_task_progress','卡住或证据矛盾时核对实际缺口；必要时至多一次短模型复核，不授予权限',obj({
                'model_review':{'type':'boolean'},'proposed_answer':{'type':'string','maxLength':1000}}),reflect),
        )
