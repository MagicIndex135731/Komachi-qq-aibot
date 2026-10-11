"""Explicit owner-message pixels; no surrounding chat or foreign cache inputs."""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
import time
from uuid import uuid4

from app.agent.group_vision import canonical_group_image_inputs, _local_pixels
from app.agent.social_images import MAX_IMAGES, MAX_TOTAL_BYTES, FETCH_SECONDS
from app.core.message_content import ImageAttachment
from app.storage.db import session_scope
from app.storage.models import Message


@contextmanager
def owner_image_inputs(service, original, guard, store):
    guard.require_valid()
    if original is None or service.source(original['source_id']) != original:
        raise RuntimeError('owner_input_source_revoked')
    settings=service.runtime.settings
    group=original['source_group_id']
    if group is not None:
        media=canonical_group_image_inputs(service.engine,group,(original['source_id'],),
            cache_dir=settings.data_dir/'image_cache')
    else:
        pictures=[]
        root=(settings.data_dir/'image_cache'/'private'/str(settings.bot_qq)/str(settings.owner_qq)).resolve()
        with session_scope(service.engine) as db:
            row=db.get(Message,original['source_id'])
            message=row.raw_json.get('message',[])
            candidates=[part.get('data',{}) for part in message if isinstance(part,dict) and part.get('type')=='image'] if isinstance(message,list) else []
            platform_id=row.platform_msg_id
        budget,deadline=[MAX_TOTAL_BYTES],time.monotonic()+FETCH_SECONDS
        temporary=[]
        try:
            store.root.mkdir(parents=True,exist_ok=True)
            for candidate in candidates[:MAX_IMAGES]:
                if not isinstance(candidate,dict): continue
                pixels=_local_pixels(candidate.get('local_path'),group_cache=root,budget=budget,deadline=deadline)
                if pixels is None: continue
                content,_,suffix=pixels
                path=store.root/('.owner-input-'+uuid4().hex+suffix)
                path.write_bytes(content); temporary.append(path)
                pictures.append(ImageAttachment(url='',local_path=str(path),source_user_id=settings.owner_qq,
                    source_message_id=platform_id,reference_subject='主人当前私聊消息明确提供的图片'))
            with _owned_pixels(service,original,guard,store,nullcontext(pictures)) as value:
                yield value
        finally:
            for path in temporary: path.unlink(missing_ok=True)
        return
    with _owned_pixels(service,original,guard,store,media) as value:
        yield value


@contextmanager
def _owned_pixels(service,original,guard,store,media):
    result=[]
    with media as pictures:
        for picture in pictures[:MAX_IMAGES]:
            guard.require_valid()
            if service.source(original['source_id'])!=original:
                raise RuntimeError('owner_input_source_revoked')
            store.root.mkdir(parents=True,exist_ok=True)
            path=store.root/('.owner-copy-'+uuid4().hex+Path(picture.local_path).suffix)
            try:
                path.write_bytes(Path(picture.local_path).read_bytes())
                artifact=store.add(path,'explicit_owner_message_pixels')
            finally:
                path.unlink(missing_ok=True)
            result.append(replace(picture,local_path=str(store.get(artifact)),
                reference_subject=f'主人明确提供的真实图片，当前任务产物ID={artifact}'))
        guard.require_valid()
        yield result
