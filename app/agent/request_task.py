"""Source-bound request goals and bounded feedback, also outside research mode.

Lexical seeds are conservative hints. The main model can describe any other
goal through the same stable tool; neither mechanism grants tool authority.
"""
from __future__ import annotations

import re

from app.agent.tool_agent import AgentTool, ToolOutcome
from app.agent.tool_evidence import evidence_descriptor, resource_ready

_MEDIA=re.compile(r'图片|原图|照片|图集|cos图|视频|相片|\b(?:photos?|images?|videos?)\b',re.I)
_TRANSFER=re.compile(r'下载|搬运|搬过来|扒|提取|保存|发(?:给|我|来|到)|传(?:给|我|来)|\b(?:download|send)\b',re.I)
_WANT=re.compile(r'(?:我要看|我想看|想看|给我看|想要|我要|来点|来张).{0,100}(?:图|照片|视频)|\bshow me\b',re.I)
_ANALYZE=re.compile(r'图(?:片)?里|图中|视频(?:里|中|说|讲)|写了什么|说的是什么|描述这|分析这|解释这|小诗|故事|作文')
_EXTERNAL=re.compile(r'联网|上网|搜索|检索|查询|查证|核实|原文|https?://|\bsearch\b',re.I)
_TASK=re.compile(r'研究|调研|审阅|审查|漏洞|分析|总结|整理|比较|对比|查询|查证|下载|搬运|提取|保存|帮我写|给我写|\b(?:research|review|download)\b',re.I)
_PRIMARY=re.compile(r'原文|核实|查证|最新|最近|近期|漏洞|审阅|审查|\b(?:latest|recent|verify)\b',re.I)
_READS={'search_web','open_web_page','find_web_page','memory_search','memory_read','search_history',
        'read_research_evidence','read_research_resources','read_forward_messages','list_web_images',
        'download_web_image','download_web_images','download_web_video','download_twitter_media','read_self_activity'}
_ARTIFACT_TOOLS={'download_web_image','download_web_images','download_web_video','download_twitter_media',
                 'read_artifact','media_result','generate_image','generate_private_image',
                 'search_image_references','execute_python','list_artifacts',
                 'observe_group_images','observe_forward_images','observe_post_images'}
_REQUIREMENTS={'source_evidence','primary_source','media_delivery'}
TASK_POLICY=(
    '当前请求目标始终有轻量记录，不必先建正式研究任务。按用户目标选择工具，工具ok不等于完成目标。'
    '需要时用set_task_goal描述语义目标/交付条件；不扩权限或另开执行。'
    '缺原文、参数、日期或产物时优先从获准工具发现；摘要不能证明最新、没有发布或不支持。'
    '历史回复里“不能下载/没有工具”等口头断言不代表当前能力；以本轮工具目录、权限和实际执行结果为准，不能直接沿用旧拒绝。'
    '跨轮证据使用原结果task_id加evidence_id读取，不切换当前目标；本地证据定位失败后按当前请求恢复读取或搜索来源，不能当成网站下载失败。'
    '新闻报道日期、搜索收录日期和原帖发布日期是不同事实；下载最新或指定日期的帖子须核对原帖，缺证据时说明日期未核实。'
    '工具反馈的missing/no_progress用于调整方法；反复没有新证据时换策略，必要时review_task_progress最多一次短复核。'
    'send_group/send_private可标task_status=progress/partial/blocked/completed：进展提示与诚实受阻可以发送，'
    '图片视频完成须本请求实际产物及真实回执；无依据不要宣称已完成或做不到。'
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
        self.expected_artifacts=0
        self.resource_urls=set()
        self.jobs={}
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
        if initial_result:
            results=initial_result if isinstance(initial_result,(tuple,list)) else (initial_result,)
            for index,value in enumerate(results):
                if isinstance(value,dict):
                    media=value.get('kind')=='media_result'
                    delegated=value.get('source_bound') is True and bool(value.get('task_id'))
                    self.observe('media_result' if media else 'research_result' if delegated else 'initial_evidence',{},
                        ToolOutcome(value.get('data',value) if media else value,
                            value.get('status','ok') if media else
                            'blocked' if delegated and value.get('status') in {'blocked','failed'} else 'ok'),f'initial-{index}')

    @property
    def needs_review(self):
        return self.active and (self.stalled>=2 or self.corrections>=2)

    def _missing(self,*,before_delivery=False):
        gaps=[]
        if self.unknown:gaps.append('unknown_external_effect')
        if any(v in {'queued','pending','running','result_ready','delivering'} for v in self.jobs.values()):
            gaps.append('background_pending')
        if 'source_evidence' in self.requirements and not self.kinds.intersection(
                {'search_summary','delegated_summary','primary_text','canonical_messages','memory_evidence','action_receipts','media_artifacts'}):
            gaps.append('source_evidence_missing')
        if 'primary_source' in self.requirements and not self.kinds.intersection(
                {'primary_text','canonical_messages','action_receipts','media_artifacts'}):
            gaps.append('primary_source_unverified')
        if self.expected_artifacts>len(self.artifacts):gaps.append('resource_batch_incomplete')
        if (self.requires_media or 'media_delivery' in self.requirements) and (
                before_delivery or not self.last_delivery or not self.last_delivery.get('media')):
            gaps.append('media_delivery_unconfirmed')
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
            'no_progress':self.stalled>=2,'corrections':self.corrections,
            'semantic_coverage':'not_independently_proven'}

    def describe(self,args):
        self.workspace._check()
        self.active=True
        self.goal=args.get('goal',self.goal)[:1000]
        self.delivery='media' if self.requires_media else args.get('delivery','text')
        self.requires_media=self.delivery=='media'
        self.requirements=self._seed_requirements|set(args.get('requirements',[])).intersection(_REQUIREMENTS)
        self._persist()
        return {**self.state(),'authority':'unchanged; current canonical request only'}

    def preflight(self,name,args):
        if name not in {'send_group','send_private'} or not self.active:return None
        checkpoint=self.workspace.execution_status()
        declared=checkpoint.get('status') if checkpoint.get('status') in {'partial','blocked'} else 'completed'
        stage=args.get('task_status',declared)
        ids=set(args.get('artifact_ids') or [])|set(args.get('media_artifact_ids') or [])
        if args.get('video_artifact_id'):ids.add(args['video_artifact_id'])
        missing=[]
        if stage in {'blocked','partial'}:
            if stage=='blocked' and ids:missing.append('failure_notice_cannot_claim_media_completion')
            attempts=[call for call in self.calls if call['tool'] not in {
                'read_context','read_private_context','begin_research_task','update_research_task',
                'read_research_task','remember_intention','save_work_note'}]
            if not attempts and not self.failures:missing.append('no_actual_attempt_or_blocker')
            external_attempts = [call for call in attempts if call['tool'] not in {
                'read_research_evidence', 'read_research_resources', 'resume_research_task',
                'send_group', 'send_private'}]
            if (self.requires_media and self.failures and not external_attempts
                    and all(failure['category'] == 'local_lookup' for failure in self.failures)):
                missing.append('recoverable_lookup_not_task_blocker')
            if (stage=='blocked' and not self.failures and self.reads<2 and set(self._missing()).intersection(
                    {'source_evidence_missing','primary_source_unverified'})):
                missing.append('insufficient_attempts_before_blocked')
            if ids-self.artifacts:missing.append('artifact_not_observed_for_request')
        elif stage=='progress':
            pass
        else:
            missing=self._missing(before_delivery=True)
            if ids and ids.issubset(self.artifacts):
                missing=[x for x in missing if x!='media_delivery_unconfirmed']
            if ids-self.artifacts:missing.append('artifact_not_observed_for_request')
        if not missing:
            self._last_preflight={'stage':stage,'media':bool(ids),'artifact_ids':sorted(ids)}
            return None
        self.corrections=min(3,self.corrections+1)
        self._persist()
        return ToolOutcome({'error':'task_completion_unverified','missing':list(dict.fromkeys(missing)),
            'task':self.state(),'review_available':not self.review_used,
            'guidance':'尚未达到当前目标，未发送。按缺口获取原文/资源/产物后交付；实际尝试仍受阻时'
                '可用task_status=blocked或partial如实说明范围与原因，不能说未发布或无工具。'
                +('旧证据定位失败不是下载失败；用原task_id重读，或按当前目标搜索/打开来源，不能直接要求用户代找链接。'
                  if 'recoverable_lookup_not_task_blocker' in missing else '')
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
        if name not in {'set_task_goal','review_task_progress'}:
            excerpt=data.get('content') or data.get('evidence') or ''
            self.observations.append({'tool':name,'status':outcome.status,'kind':descriptor['kind'],
                'excerpt':str(excerpt)[:600], 'coverage':descriptor['coverage'],
                'absence_proven':False})
            self.observations=self.observations[-6:]
        old=len(self.useful)
        if name not in {'set_task_goal','review_task_progress','initial_evidence'}:
            self.calls.append({'id':call_id,'tool':name,'status':outcome.status})
            self.calls=self.calls[-32:]
        if descriptor['failure']:
            self.failures.append(descriptor['failure'])
            self.failures=self.failures[-4:]
        if outcome.status=='uncertain':self.unknown=True
        if outcome.status in {'ok','sent'}:
            kind=descriptor['kind']
            if kind!='tool_result':self.kinds.add(kind)
            if kind in {'primary_text','canonical_messages','quoted_messages','action_receipts','media_artifacts','media_candidates'} and not data.get('cached_in_turn'):
                identity=(name,str(args.get('ref_id') or args.get('query') or data.get('artifact_id') or
                    data.get('evidence_id') or data.get('source_message_ids') or data.get('source_ids') or ''),
                    str(args.get('lineno','')))
                self.useful.add(identity)
            for resource in descriptor['resources']:
                self.resource_urls.add(resource['url'])
                if not self.requires_media or resource_ready(resource['url']):
                    self.useful.add(('resource',resource['url']))
            artifact_data=data
            if name=='media_task_status':
                result=data.get('result')
                artifact_data=result.get('data',{}) if isinstance(result,dict) else {}
            if name in _ARTIFACT_TOOLS or name=='media_task_status':
                requested=artifact_data.get('requested_count')
                if type(requested) is int and requested>0:
                    self.expected_artifacts=max(self.expected_artifacts,min(requested,20))
                identifiers=[]
                for key in ('artifact_ids','video_artifact_ids','media_artifact_ids'):
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
                self.artifacts.update(identifiers[:20])
                if identifiers:self.kinds.add('media_artifacts')
            entries=data.get('jobs',[]) if name=='queue_media_batch' else [data]
            for entry in entries:
                if isinstance(entry,dict) and type(entry.get('job_id')) is int:
                    self.jobs[entry['job_id']]=entry.get('execution_status',entry.get('status'))
            if name in {'generate_private_image','private_image_status'} and type(data.get('task_id')) is int:
                self.jobs[data['task_id']]=data.get('status')
            if name in {'send_group','send_private'} and outcome.status=='sent':
                self.last_delivery={**(self._last_preflight or {'stage':args.get('task_status','completed'),
                    'media':bool(args.get('artifact_ids') or args.get('media_artifact_ids') or args.get('video_artifact_id'))}),
                    'receipt_id':outcome.receipt_id}
                stage=self.last_delivery['stage']
                self.stage=('waiting' if self.jobs and stage=='progress' else
                    'active' if stage=='progress' else stage)
                if stage=='completed' and self._missing():self.stage='partial'
        if name in _READS:
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
        return (self.stage in {'completed','blocked','partial','waiting'} and
                (self.stage!='completed' or not self._missing()))

    def _persist(self):
        self.workspace.save_request_control({'goal':self.goal,'active':self.active,**self.state(),'failures':self.failures,
            'calls':self.calls,'last_delivery':self.last_delivery})

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
                'requirements':{'type':'array','items':{'type':'string','enum':sorted(_REQUIREMENTS)},'maxItems':5}},
                ('goal','delivery','requirements')),describe,side_effects=('local_state',)),
            AgentTool('review_task_progress','卡住或证据矛盾时核对实际缺口；必要时至多一次短模型复核，不授予权限',obj({
                'model_review':{'type':'boolean'},'proposed_answer':{'type':'string','maxLength':1000}}),reflect),
        )
