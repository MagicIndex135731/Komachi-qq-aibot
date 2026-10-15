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
    elif name in {'search_web','open_web_page','find_web_page','read_research_evidence'}:
        content_kind=str(data.get('content_kind',''))
        kind=('search_summary' if name=='search_web' or 'summary' in content_kind or content_kind=='grounded_answer'
              else 'primary_text' if data.get('content') else 'empty_result')
        for item in data.get('sources',[])[:20]:
            if not isinstance(item,dict):continue
            value=item.get('source')
            if isinstance(value,str) and value.startswith(('https://','http://')):
                resources.append({'kind':'web_url','url':value[:2000],
                    'date_status':'unverified','author_status':'unverified'})
    elif name in {'download_web_image','download_web_images','download_web_video','download_twitter_media',
                  'read_artifact','media_result','generate_image','generate_private_image','search_image_references',
                  'execute_python'}:
        kind='media_artifacts' if any(data.get(key) for key in
            ('artifact_id','artifact_ids','video_artifact_id','video_artifact_ids','media_artifact_ids','artifacts')) else 'tool_result'
    elif name=='list_web_images':
        kind='media_candidates'
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
                                      'invalid_task_id','invalid_task_evidence','invalid_resource_page'} else
            'authority' if code in {'owner_required','web_requires_explicit_request','share_request_revoked',
                                   'share_requires_quoted_request','memory_out_of_scope'} else
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
