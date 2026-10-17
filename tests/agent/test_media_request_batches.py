"""One original media request, independently downloaded and jointly returned."""
import asyncio
from datetime import timedelta

import pytest

from app.agent.media_tasks import MediaTaskQueue, media_result
from app.agent.tool_agent import ToolOutcome
from app.storage.db import session_scope
from app.storage.models import Job, Message
from test_media_tasks import setup_queue


async def drain(queue, limit=100):
    for _ in range(limit):
        if not await queue.tick():return
    raise AssertionError('media request did not settle')


@pytest.mark.parametrize('count,failed', [(3,False), (3,True), (20,False)])
async def test_review_results_return_together_without_capacity_deadlock(sqlite_engine, count, failed):
    returned=[]
    async def execute(payload, guard):
        index=int(payload['arguments']['ref_id'].rsplit('/',1)[1])
        if failed and index==1:return media_result(ToolOutcome({'error':'synthetic_404'},'failed'),payload)
        identifier='art-'+str(index)
        return media_result(ToolOutcome({'media_artifact_ids':[identifier],'video_artifact_ids':[identifier],
            'requested_count':1,'downloaded_count':1,'media':[{'artifact_id':identifier,'kind':'video',
            'source_ref':payload['arguments']['ref_id'],'bytes':100+index}]}),payload)
    async def deliver(payload, guard):
        guard.require_valid();returned.append(payload['result']);return 'completed'
    queue,payload,_=setup_queue(sqlite_engine,execute=execute,deliver=deliver)
    jobs=[queue.submit({**payload,'arguments':{'ref_id':f'https://example.com/video/{i}'}},
        parent_turn=42,before_commit=lambda:None).data['job_id'] for i in range(count)]
    await drain(queue)
    assert len(returned)==1
    result=returned[0]
    assert result['status']=='ok'
    assert len(result['data']['media_artifact_ids'])==count-int(failed)
    assert len(result['data']['failures'])==int(failed)
    with session_scope(sqlite_engine) as session:
        assert all(session.get(Job,job).status=='completed' for job in jobs)


async def test_review_batch_restart_and_single_coordinator(sqlite_engine):
    entered,release=asyncio.Event(),asyncio.Event();returned=[]
    async def deliver(payload,guard):
        guard.require_valid();returned.append(payload['result']);entered.set();await release.wait();return 'completed'
    queue,payload,_=setup_queue(sqlite_engine,deliver=deliver)
    jobs=[queue.submit({**payload,'arguments':{'ref_id':f'https://example.com/video/{i}'}},
        parent_turn=42,before_commit=lambda:None).data['job_id'] for i in range(3)]
    # Persist ready results from a different worker owner, as after a restart.
    with session_scope(sqlite_engine) as session:
        for i,identifier in enumerate(jobs):
            row=session.get(Job,identifier);row.status='result_ready';row.locked_by='old-worker'
            row.payload_json={**row.payload_json,'result':media_result(ToolOutcome(
                {'media_artifact_ids':[f'art-{i}'],'video_artifact_ids':[f'art-{i}']}),row.payload_json)}
    active=asyncio.create_task(queue.tick())
    try:
        async with asyncio.timeout(2):await entered.wait()
        assert not await asyncio.wait_for(queue.tick(),timeout=0.5)
        assert len(returned)==1 and len(returned[0]['data']['media_artifact_ids'])==3
    finally:release.set();await active
    assert not await queue.tick()


@pytest.mark.parametrize('state',['uncertain','cancelled'])
async def test_unknown_or_cancelled_review_sibling_stops_the_batch(sqlite_engine,state):
    calls=[]
    async def deliver(payload,guard):calls.append(payload);return 'completed'
    queue,payload,_=setup_queue(sqlite_engine,deliver=deliver)
    jobs=[queue.submit({**payload,'arguments':{'ref_id':f'https://example.com/video/{i}'}},
        parent_turn=42,before_commit=lambda:None).data['job_id'] for i in range(2)]
    with session_scope(sqlite_engine) as session:
        session.get(Job,jobs[0]).status=state;session.get(Job,jobs[1]).status='result_ready'
    assert not await queue.tick() and not calls


def test_forward_planner_keeps_three_ordinary_videos_in_one_packet(tmp_path):
    from app.agent.media_delivery import pack_media_files
    files=[]
    for i,size in enumerate([2,41,3]):
        path=tmp_path/f'{i}.mp4'
        with path.open('wb') as stream:stream.truncate(size*1024**2)
        files.append(('video',str(path)))
    assert pack_media_files(['a','b','c'],files)==[['a','b','c']]


def test_large_aggregate_compaction_keeps_request_and_settled_job_identity():
    from app.agent.media_tasks import merge_media_results
    payload={'operation':'download_web_video','task_id':'synthetic-batch'}
    results=[(i,{'status':'ok','data':{'media_artifact_ids':['art-'+str(i)],
        'media':[{'artifact_id':'art-'+str(i),'kind':'video','source_ref':'https://example.com/'+str(i),
            'bytes':100,'description':'ordinary '*1000}]}}) for i in range(20)]
    merged=merge_media_results(results,payload)['data']
    assert merged['result_compacted'] and merged['request_aggregated']
    assert len(merged['jobs'])==20 and len(merged['media_artifact_ids'])==20
