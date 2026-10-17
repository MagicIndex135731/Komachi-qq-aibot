"""Small, typed descriptions of actual tool observations, never permissions."""
from __future__ import annotations

import hashlib
import json
from app.core.url_policy import media_resource_url as resource_ready


def evidence_descriptor(name, arguments, outcome):
    data=outcome.data if isinstance(outcome.data,dict) else {}
    kind='tool_result'
    resources=[]
    if data.get('kind')=='source_bound_memory_evidence':
        kind=('canonical_messages' if data.get('layer_counts',{}).get('raw',0)>0
            or name=='search_history' and data.get('evidence')
            else 'memory_evidence' if data.get('evidence') else 'empty_result')
    elif name=='read_self_activity':
        kind='action_receipts'
    elif name=='research_result' and data.get('source_bound') is True:
        kind='delegated_summary'
    elif name in {'search_web','open_web_page','find_web_page','read_research_evidence','inspect_web_resources','list_x_posts','search_x_posts','search_xhs_posts'}:
        content_kind=str(data.get('content_kind',''))
        kind=('search_summary' if name in {'search_web','search_xhs_posts'} or 'summary' in content_kind or content_kind=='grounded_answer'
              else 'primary_text' if data.get('content') else 'empty_result')
        for item in data.get('sources',[])[:20]:
            if not isinstance(item,dict):continue
            value=item.get('source')
            if isinstance(value,str) and value.startswith(('https://','http://')):
                resources.append({'kind':'web_url','url':value[:2000],
                    'date_status':'unverified','author_status':'unverified'})
        for item in data.get('resources', [])[:12]:
            if not isinstance(item, dict) or item.get('status') not in {None, 'ok'}:continue
            url = item.get('source_url')
            if not url and type(item.get('input_index')) is int:
                refs = arguments.get('ref_ids', [])
                index = item['input_index'] - 1
                if isinstance(refs, list) and 0 <= index < len(refs):url = refs[index]
            if isinstance(url, str) and url.startswith(('https://', 'http://')):
                resources = [r for r in resources if r['url'] != url]
                if name=='search_xhs_posts' and item.get('identity_status')!='matched':
                    resources.append({'kind':'web_url','url':url[:2000],
                        'date_status':'unverified','author_status':'unverified'})
                    continue
                resources.append({'kind': 'inspected_post', 'url': url[:2000], **{k: item[k] for k in
                    ('post_id', 'author', 'published_at', 'identity_status', 'date_status', 'author_status',
                     'media_count', 'media_roles') if k in item}})
                kind = 'primary_metadata'
    elif name in {'download_web_image','download_web_images','download_web_video','download_twitter_media',
                  'read_artifact','media_result','generate_image','generate_private_image','search_image_references',
                  'execute_python'}:
        kind='media_artifacts' if any(data.get(key) for key in
            ('artifact_id','artifact_ids','video_artifact_id','video_artifact_ids','media_artifact_ids','artifacts')) else 'tool_result'
    elif name=='list_web_images':
        kind='media_candidates'
        if data.get('post_id') and data.get('identity_status')=='matched' and data.get('page_url'):
            resources.append({'kind':'inspected_post','url':data['page_url'],**{k:data[k] for k in
                ('post_id','author','published_at','identity_status','date_status','author_status','media_count') if k in data}})
    elif name in {'read_research_resources','read_message_links'}:
        kind='resource_index'
        for item in data.get('resources',data.get('links',[]))[:20]:
            url=item.get('url') if isinstance(item,dict) else None
            if isinstance(url,str) and url.startswith(('https://','http://')):
                resources.append({'kind':'web_url','url':url[:2000],
                    'date_status':'unverified','author_status':'unverified'})
    elif name=='read_forward_messages':
        kind='quoted_messages' if data.get('nodes') else 'empty_result'
    elif name in {'send_group','send_private'}:
        kind='delivery_receipt' if outcome.status in {'sent','uncertain'} else 'delivery_failure'
    error=data.get('error')
    failure=None
    if outcome.status in {'failed','blocked','uncertain'}:
        code=error if isinstance(error,str) and len(error)<120 else outcome.status
        failure={'code':code,'category':('unknown_effect' if outcome.status=='uncertain' else
            'local_lookup' if code in {'task_evidence_unavailable','task_checkpoint_unavailable',
                                      'invalid_task_id','invalid_task_evidence','invalid_resource_page',
                                      'resource_freshness_unverified'} else
            'authority' if code in {'owner_required','web_requires_explicit_request','share_request_revoked',
                                   'share_requires_quoted_request','memory_out_of_scope'} else
            'access' if code in {'web_resource_session_required', 'web_resource_extractor_unavailable',
                'web_resource_http_401', 'web_resource_http_403', 'web_image_note_unavailable',
                'web_resource_dynamic_or_empty'} else
            'capacity' if any(v in code for v in ('budget','capacity','limit','cache_busy')) else
            'arguments' if any(v in code for v in ('argument','invalid_','not_allowed')) else 'provider'),
            'retryable':outcome.status!='uncertain' and any(v in code for v in
                ('timeout','http_429','http_500','http_502','http_503','http_504'))}
    return {'kind':kind,'resources':resources,'failure':failure,
        'coverage':'observed_results_only','absence_proven':False}


def capability_version(client,tools):
    return hashlib.sha256(json.dumps(['request-task-v1',getattr(client,'base_url',''),
        getattr(client,'responses_model',None) or getattr(client,'model',''),
        [(t.name,t.parameters) for t in sorted(tools,key=lambda t:t.name)]],
        ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:24]
