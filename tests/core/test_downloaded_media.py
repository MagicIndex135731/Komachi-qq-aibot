import json
import multiprocessing
import os
from pathlib import Path
import time

import pytest
from app.agent.artifacts import AutonomousArtifactStore
from app.core.downloaded_media import DownloadedMediaCache, DownloadedMediaJanitor, LEASE_SECONDS
from tests.agent.test_autonomous_social import BOT, GROUP, image_bytes


def library(engine, root):
    return AutonomousArtifactStore(root / 'autonomous_artifacts', account_id=BOT,
        scope=f'group:{GROUP}', engine=engine, source_message_ids=lambda: ())


def picture(store, origin='web_page_image', cache=None):
    store.root.mkdir(parents=True, exist_ok=True)
    path = store.root / '.input.png'; path.write_bytes(image_bytes())
    try:
        return cache.register(store,path,origin) if cache else store.add(path,origin)
    finally:
        path.unlink()


def test_ttl_removes_web_images_but_preserves_generated_art_and_user_files(sqlite_engine,tmp_path):
    store = library(sqlite_engine,tmp_path)
    now=[time.time()]
    cache=DownloadedMediaCache.for_store(store); cache.clock=lambda:now[0]
    downloaded=picture(store,cache=cache)
    reference=picture(store,origin='web_reference_pixels',cache=cache)
    generated=picture(store,origin='autonomous_generated')
    unrelated=cache.root/'keep.txt'; unrelated.write_text('user file')
    now[0]+=24*3600+2
    result=cache.cleanup()
    assert result['removed']==2 and store.get(downloaded) is None and store.get(reference) is None
    assert store.get(generated) is not None and unrelated.read_text()=='user file'


def test_quota_evicts_old_unpinned_and_refuses_when_all_in_use(sqlite_engine,tmp_path):
    store=library(sqlite_engine,tmp_path); now=[time.time()]
    cache=DownloadedMediaCache(tmp_path,max_bytes=65536,clock=lambda:now[0])
    image=picture(store,cache=cache)
    with pytest.raises(ValueError,match='full_in_use'):
        with cache.reserve(65536): pass
    assert store.get(image)
    now[0]+=LEASE_SECONDS+2
    with cache.reserve(65536):
        assert store.get(image) is None
    assert not list(cache.root.glob('.reserve-*'))


def test_active_lease_protects_expired_file_until_it_finishes(sqlite_engine,tmp_path):
    store=library(sqlite_engine,tmp_path); now=[time.time()]
    cache=DownloadedMediaCache(tmp_path,retention_seconds=1,clock=lambda:now[0])
    image=picture(store,cache=cache)
    now[0]+=10
    cache.cleanup(); assert store.get(image)
    now[0]+=LEASE_SECONDS
    cache.cleanup(); assert store.get(image) is None


def test_stale_parts_reservations_and_unreferenced_video_are_removed(sqlite_engine,tmp_path):
    store=library(sqlite_engine,tmp_path)
    cache=DownloadedMediaCache.for_store(store)
    cache.root.mkdir(parents=True)
    paths=[cache.root/'video-old.part',cache.root/('webvid-'+'a'*32+'.mp4')]
    for path in paths:
        path.write_bytes(b'old');os.utime(path,(time.time()-LEASE_SECONDS-5,)*2)
    (cache.root/'.reserve-dead.json').write_text(json.dumps({'bytes':65536,'until':0}))
    fresh=cache.root/'video-current.part';fresh.write_bytes(b'active')
    unrelated=cache.root/'user.part';unrelated.write_bytes(b'keep')
    os.utime(unrelated,(time.time()-LEASE_SECONDS-5,)*2)
    cache.cleanup()
    assert not any(path.exists() for path in paths) and fresh.exists()
    assert not (cache.root/'.reserve-dead.json').exists()
    assert unrelated.read_bytes()==b'keep'


def test_video_pointer_retains_scope_and_cleanup_removes_both_files(sqlite_engine,tmp_path):
    store=library(sqlite_engine,tmp_path);now=[time.time()]
    cache=DownloadedMediaCache.for_store(store);cache.clock=lambda:now[0]
    with cache.reserve(1024):
        path=cache.root/'video.part';path.write_bytes(b'synthetic-video')
        artifact=cache.save_video(store,path,{'bytes':15,'duration_seconds':1})
    target,_=cache.video(store,artifact);assert target.read_bytes()==b'synthetic-video'
    foreign=AutonomousArtifactStore(store.root,account_id=BOT,scope='group:900000002',engine=sqlite_engine,source_message_ids=lambda:())
    with pytest.raises(ValueError,match='unavailable'):
        cache.video(foreign,artifact)
    now[0]+=24*3600+2
    cache.cleanup();assert not target.exists() and store.get(artifact) is None


def test_malicious_manifest_cannot_delete_outside_cache(sqlite_engine,tmp_path):
    store=library(sqlite_engine,tmp_path);artifact=picture(store)
    outside=tmp_path/'outside.png';outside.write_bytes(b'keep')
    manifest=store.root/(artifact+'.meta.json')
    data=json.loads(manifest.read_text());data['file']='../outside.png';data['created_ns']=0
    manifest.write_text(json.dumps(data))
    DownloadedMediaCache.for_store(store).cleanup()
    assert outside.read_bytes()==b'keep'


def child_admission(directory,queue):
    try:
        cache=DownloadedMediaCache(Path(directory),max_bytes=65536)
        with cache.reserve(65536): queue.put('admitted')
    except ValueError as exc:
        queue.put(str(exc))


def test_processes_share_reservations_and_disk_quota(tmp_path):
    cache=DownloadedMediaCache(tmp_path,max_bytes=65536)
    context=multiprocessing.get_context('spawn');queue=context.Queue()
    with cache.reserve(65536):
        worker=context.Process(target=child_admission,args=(str(tmp_path),queue));worker.start()
        try:
            assert queue.get(timeout=15)=='media_cache_full_in_use'
        finally:
            worker.join(timeout=5)
            if worker.is_alive(): worker.terminate();worker.join()
    assert worker.exitcode==0


async def test_periodic_service_runs_without_chat_and_stops(tmp_path):
    import asyncio
    from types import SimpleNamespace
    worker=DownloadedMediaJanitor(SimpleNamespace(data_dir=tmp_path),interval_seconds=.01)
    calls=[]
    worker.cache.cleanup=lambda: calls.append(True) or {'removed':0,'charged_bytes':0,'protected_count':0}
    await worker.start()
    try:
        for _ in range(100):
            if len(calls)>=2: break
            await asyncio.sleep(.01)
        assert len(calls)>=2
    finally: await worker.stop()
    assert worker.task is None
