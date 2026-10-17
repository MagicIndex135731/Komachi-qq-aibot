"""Bounded choices between versions of one source asset, never different content."""
from __future__ import annotations
import math
import time
from app.core.video_file import MAX_VIDEO_SECONDS


class TransferBudget:
    def __init__(self, max_bytes, *, deadline=None, check=lambda:None, max_requests=48, parent=None):
        self.max_bytes,self.received,self.requests=max_bytes,0,0
        self.deadline=deadline or time.monotonic()+180
        self.parent=parent
        if parent:self.deadline=min(self.deadline,parent.deadline)
        self.authority=check;self.max_requests=max_requests

    @property
    def remaining(self):return max(0,min(self.max_bytes-self.received,
        self.parent.remaining if self.parent else self.max_bytes))

    def check(self):
        self.authority()
        if self.parent:self.parent.check()
        if time.monotonic()>=self.deadline:raise ValueError('media_transfer_deadline')
        if self.received>=self.max_bytes:raise ValueError('media_transfer_bytes_exhausted')

    def request(self):
        self.check()
        if self.requests>=self.max_requests:raise ValueError('media_transfer_request_budget')
        self.requests+=1
        if self.parent:self.parent.request()

    def charge(self,size):
        self.received+=size
        if self.parent:self.parent.charge(size)
        if self.received>self.max_bytes:raise ValueError('media_transfer_bytes_exhausted')


def native_variants(variants, *, max_bytes, original_only=False, limit=3, diagnostics=None):
    candidates=[];seen=set()
    rejected={}
    def reject(reason):rejected[reason]=rejected.get(reason,0)+1
    for item in variants[:32]:
        if not isinstance(item,dict) or not isinstance(item.get('url'),str):
            reject('invalid_metadata');continue
        if item['url'] in seen:
            reject('duplicate');continue
        seen.add(item['url'])
        if item.get('codec') not in {None,'','h264','avc','avc1'}:
            reject('unsupported_codec');continue
        numbers=[item.get(k,0) or 0 for k in ('declared_bytes','bitrate','width','height','duration_seconds')]
        if any(type(v) not in {int,float} or not math.isfinite(v) or v<0 for v in numbers):
            reject('invalid_metadata');continue
        declared,rate,width,height,duration=numbers
        if duration>MAX_VIDEO_SECONDS:
            reject('duration_limit');continue
        estimated=declared or rate*duration/8
        candidates.append({**item,'estimated_bytes':int(estimated),'codec':item.get('codec') or 'unknown_requires_validation'})
    candidates.sort(key=lambda i:(i.get('width',0)*i.get('height',0),i.get('bitrate',0),i.get('declared_bytes',0)),reverse=True)
    if original_only:candidates=candidates[:1]
    best=candidates[0]['url'] if candidates else None
    for item in candidates:
        if item['estimated_bytes']>max_bytes:reject('estimated_size_limit')
    if diagnostics is not None:
        diagnostics.update(variant_count=min(len(variants),32),byte_limit=max_bytes,
            duration_limit_seconds=MAX_VIDEO_SECONDS,rejected_by_reason=rejected)
    return [{**i,'quality':'highest_native' if i['url']==best else 'lower_native',
        'fallback_reason':'' if i['url']==best else 'higher_version_over_budget_or_failed'}
        for i in candidates if not i['estimated_bytes'] or i['estimated_bytes']<=max_bytes][:limit]
