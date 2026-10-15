from datetime import UTC, datetime
import io
from pathlib import Path
import threading

import httpx
from PIL import Image
import pytest

from app.agent import group_vision
from app.agent.group_vision import canonical_group_image_inputs
from app.storage.db import session_scope
from app.storage.repositories import GroupRepository, MessageRepository, UserRepository


GROUP = 900000001
AUTHOR = 900000101


def pixels(color="red", *, size=(24, 24)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def add_source(engine, *, platform_id, group_id=GROUP, author=AUTHOR, images=(), state=""):
    with session_scope(engine) as session:
        GroupRepository(session).upsert_group(group_id=group_id, group_name="synthetic", enabled=True, speak_enabled=True)
        UserRepository(session).upsert_user(user_id=author, nickname="synthetic author", group_card="")
        source = MessageRepository(session).add_group_message(
            platform_msg_id=platform_id, group_id=group_id, user_id=author,
            timestamp=datetime.now(UTC), plain_text="", msg_type="image", reply_to_msg_id=None,
            mentioned_bot=False, raw_json={"message": [{"type": "image", "data": image} for image in images],
                                            "delivery_state": state,
                                            "user_id": 900000999, "message_id": "spoofed-raw-id"},
        )
        session.flush()
        return source.id


def cached(cache, filename, data, *, group=GROUP):
    path = cache / str(group) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"local_path": str(path)}


def test_canonical_pixels_keep_actual_group_author_and_source_and_cleanup(sqlite_engine, tmp_path):
    cache = tmp_path / "image_cache"
    body = pixels()
    source_id = add_source(sqlite_engine, platform_id="real-photo", images=(cached(cache, "photo.png", body),))
    with canonical_group_image_inputs(sqlite_engine, GROUP, (source_id,), cache_dir=cache) as images:
        assert len(images) == 1
        image = images[0]
        snapshot = Path(image.local_path)
        assert snapshot.read_bytes() == body
        assert image.url == "" and not image.fallback_url
        assert image.source_user_id == AUTHOR
        assert image.source_message_id == "real-photo"
        assert f"group_id={GROUP}" in image.reference_subject
        assert f"author_id={AUTHOR}" in image.reference_subject
        assert "message_id=real-photo" in image.reference_subject
    assert not snapshot.exists()
    assert (cache / str(GROUP) / "photo.png").exists()


@pytest.mark.parametrize("state", ["deleted", "uncertain", "blocked", "reserved"])
def test_canonical_media_rejects_ineligible_sources(sqlite_engine, tmp_path, state):
    cache = tmp_path / "image_cache"
    source_id = add_source(sqlite_engine, platform_id=state, state=state, images=(cached(cache, "photo.png", pixels()),))
    with canonical_group_image_inputs(sqlite_engine, GROUP, (source_id,), cache_dir=cache) as images:
        assert images == []


def test_canonical_media_rejects_cross_group_sources_paths_and_non_images(sqlite_engine, tmp_path):
    cache = tmp_path / "image_cache"
    cross_group = GROUP + 1
    other_id = add_source(sqlite_engine, platform_id="other-group", group_id=cross_group,
                          images=(cached(cache, "other.png", pixels(), group=cross_group),))
    malicious_id = add_source(sqlite_engine, platform_id="bad-path", images=(
        cached(cache, "other.png", pixels(), group=cross_group),
        cached(cache, "secret.png", b"secret text"),
    ))
    with canonical_group_image_inputs(sqlite_engine, GROUP, (other_id, malicious_id), cache_dir=cache) as images:
        assert images == []


def test_image_selection_is_bounded_and_preserves_multiple_authors(sqlite_engine, tmp_path):
    cache = tmp_path / "image_cache"
    first_id = add_source(sqlite_engine, platform_id="first", images=tuple(cached(cache, f"first-{i}.png", pixels()) for i in range(4)))
    second_id = add_source(sqlite_engine, platform_id="second", author=AUTHOR + 1,
                           images=tuple(cached(cache, f"second-{i}.png", pixels("blue")) for i in range(4)))
    with canonical_group_image_inputs(sqlite_engine, GROUP, (first_id, second_id), cache_dir=cache) as images:
        assert len(images) == 2
        assert {image.source_user_id for image in images} == {AUTHOR, AUTHOR + 1}


def test_remote_images_use_allowed_host_then_local_pixels_not_provider_urls(sqlite_engine, tmp_path):
    cache = tmp_path / "image_cache"
    calls = []
    body = pixels()

    def response(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=body, headers={"content-type": "image/png"})

    source_id = add_source(sqlite_engine, platform_id="qq-photo", images=(
        {"url": "https://evil.example.test/secret.png"}, {"url": "https://m.qpic.cn/valid.png"},
    ))
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        with canonical_group_image_inputs(sqlite_engine, GROUP, (source_id,), cache_dir=cache, http_client=client) as images:
            assert len(images) == 1
            assert Path(images[0].local_path).read_bytes() == body
            assert images[0].url == ""
    assert calls == ["https://m.qpic.cn/valid.png"]


def test_remote_redirect_to_unapproved_host_is_never_requested(sqlite_engine, tmp_path):
    calls = []

    def response(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    source_id = add_source(sqlite_engine, platform_id="redirect", images=({"url": "https://m.qpic.cn/photo.png"},))
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        with canonical_group_image_inputs(sqlite_engine, GROUP, (source_id,), cache_dir=tmp_path / "image_cache", http_client=client) as images:
            assert images == []
    assert calls == ["https://m.qpic.cn/photo.png"]


def test_cancelled_observation_downloads_no_images(sqlite_engine, tmp_path):
    cancelled = threading.Event()
    cancelled.set()
    source_id = add_source(sqlite_engine, platform_id="cancelled", images=({"url": "https://m.qpic.cn/photo.png"},))
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("download after cancellation"))) as client:
        with canonical_group_image_inputs(sqlite_engine, GROUP, (source_id,), cache_dir=tmp_path / "image_cache",
                                          cancelled=cancelled, http_client=client) as images:
            assert images == []


def test_oversized_media_bytes_and_pixels_are_rejected(sqlite_engine, tmp_path, monkeypatch):
    cache = tmp_path / "image_cache"
    oversized = add_source(sqlite_engine, platform_id="oversized-pixels", images=(cached(cache, "too-big.png", pixels(size=(30, 30))),))
    monkeypatch.setattr(group_vision, "MAX_PIXELS", 200)
    with canonical_group_image_inputs(sqlite_engine, GROUP, (oversized,), cache_dir=cache) as images:
        assert images == []
    remote = add_source(sqlite_engine, platform_id="oversized-bytes", images=({"url": "https://m.qpic.cn/large.png"},))
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, headers={"content-length": "99999999"}))) as client:
        with canonical_group_image_inputs(sqlite_engine, GROUP, (remote,), cache_dir=cache, http_client=client) as images:
            assert images == []


def test_truncated_jpeg_is_not_sent_as_pixels(sqlite_engine, tmp_path):
    cache = tmp_path / "image_cache"
    output = io.BytesIO()
    Image.new("RGB", (100, 100), "red").save(output, format="JPEG")
    source_id = add_source(sqlite_engine, platform_id="truncated", images=(cached(cache, "truncated.jpg", output.getvalue()[:-20]),))
    with canonical_group_image_inputs(sqlite_engine, GROUP, (source_id,), cache_dir=cache) as images:
        assert images == []


def test_symlink_to_another_groups_cache_cannot_supply_pixels(sqlite_engine, tmp_path):
    cache = tmp_path / "image_cache"
    other_group = cache / str(GROUP + 1)
    other_group.mkdir(parents=True)
    photo = other_group / "photo.png"
    photo.write_bytes(pixels())
    try:
        (cache / str(GROUP)).symlink_to(other_group, target_is_directory=True)
    except OSError:
        pytest.skip("host cannot create directory symlinks")
    source_id = add_source(sqlite_engine, platform_id="symlink", images=({"local_path": str(cache / str(GROUP) / "photo.png")},))
    with canonical_group_image_inputs(sqlite_engine, GROUP, (source_id,), cache_dir=cache) as images:
        assert images == []


def test_recent_image_is_not_dropped_when_old_memory_ids_are_appended(sqlite_engine, tmp_path):
    cache = tmp_path / "image_cache"
    old_ids = tuple(add_source(sqlite_engine, platform_id=f"old-memory-{index}") for index in range(26))
    latest = add_source(sqlite_engine, platform_id="recent-photo", images=(cached(cache, "recent.png", pixels()),))
    with canonical_group_image_inputs(sqlite_engine, GROUP, (latest, *old_ids), cache_dir=cache) as images:
        assert [image.source_message_id for image in images] == ["recent-photo"]


def add_question(engine, text, *, author=AUTHOR, reply=None, age=0):
    from datetime import timedelta
    with session_scope(engine) as session:
        row=MessageRepository(session).add_group_message(platform_msg_id='question-'+str(author)+'-'+text,
            group_id=GROUP,user_id=author,timestamp=datetime.now(UTC)-timedelta(seconds=age),
            plain_text=text,msg_type='text',reply_to_msg_id=reply,mentioned_bot=True,raw_json={'message':[]})
        session.flush()
        return row.id


def test_plain_chat_does_not_select_recent_group_images(sqlite_engine):
    photo=add_source(sqlite_engine,platform_id='recent',images=({'url':'https://m.qpic.cn/photo.png'},))
    target=add_question(sqlite_engine,'今晚吃什么')
    selection=group_vision.select_group_image_sources(sqlite_engine,GROUP,(photo,target),target_source_id=target)
    assert selection.source_ids==() and selection.reason=='no_image_request'


def test_send_picture_then_mention_this_picture_selects_same_sender(sqlite_engine):
    photo=add_source(sqlite_engine,platform_id='my-photo',images=({'url':'https://m.qpic.cn/photo.png'},))
    foreign=add_source(sqlite_engine,platform_id='someone-else',author=AUTHOR+1,images=({'url':'https://m.qpic.cn/other.png'},))
    target=add_question(sqlite_engine,'小町这图怎么样')
    selection=group_vision.select_group_image_sources(sqlite_engine,GROUP,(photo,foreign,target),target_source_id=target)
    assert selection.source_ids==(photo,) and selection.reason=='recent_same_sender'


def test_quoted_image_is_selected_without_other_recent_images(sqlite_engine):
    quote=add_source(sqlite_engine,platform_id='quoted',images=({'url':'https://m.qpic.cn/photo.png'},))
    other=add_source(sqlite_engine,platform_id='other',images=({'url':'https://m.qpic.cn/other.png'},))
    target=add_question(sqlite_engine,'怎么看',reply='quoted')
    selection=group_vision.select_group_image_sources(sqlite_engine,GROUP,(other,target),target_source_id=target)
    assert selection.source_ids==(quote,) and selection.reason=='quoted'


@pytest.mark.parametrize('state', ['deleted','blocked','uncertain'])
def test_retracted_or_ineligible_image_never_supplies_followup(sqlite_engine,state):
    photo=add_source(sqlite_engine,platform_id='invalid',state=state,images=({'url':'https://m.qpic.cn/photo.png'},))
    target=add_question(sqlite_engine,'这图怎么样',reply='invalid')
    selection=group_vision.select_group_image_sources(sqlite_engine,GROUP,(photo,target),target_source_id=target)
    assert selection.source_ids==() and selection.reason=='image_reference_unavailable'
