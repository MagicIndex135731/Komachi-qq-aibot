from datetime import UTC, datetime
from types import SimpleNamespace
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
import pytest
from PIL import Image

from app.adapters.onebot_models import parse_private_message_event
from app.core.image_cache import cache_images_in_raw_payload
import app.private_chat.dispatcher as dispatcher_module
from app.private_chat.dispatcher import PrivateMessageDispatcher
from app.storage.db import session_scope
from app.storage.repositories import MessageRepository


class FakeService:
    def __init__(self):
        self.events = []

    async def handle_private_message(self, event):
        self.events.append(event)
        return True


def event(user_id=900000101):
    return SimpleNamespace(
        platform_msg_id="private-1",
        user_id=user_id,
        nickname="test-user",
        timestamp=datetime(2026, 10, 9, tzinfo=UTC),
        plain_text="hello",
        raw_payload={"message_id": "private-1"},
        images=[],
        msg_type="text",
        reply_to_msg_id=None,
    )


@pytest.mark.asyncio
async def test_dispatcher_persists_and_deduplicates_without_group_services(sqlite_engine, tmp_path):
    service = FakeService()
    dispatcher = PrivateMessageDispatcher(
        engine=sqlite_engine,
        data_dir=tmp_path,
        private_chat_service=service,
    )
    incoming = event()
    await dispatcher.handle_private_message(incoming)
    await dispatcher.handle_private_message(incoming)
    assert service.events == [incoming]
    with session_scope(sqlite_engine) as session:
        row = MessageRepository(session).get_by_platform_msg_id(
            "private-inbound-900000101-private-1"
        )
        assert row is not None
        assert row.group_id is None
        assert row.user_id == 900000101


@pytest.mark.asyncio
async def test_dispatcher_keeps_same_platform_id_separate_by_sender(sqlite_engine, tmp_path):
    service = FakeService()
    dispatcher = PrivateMessageDispatcher(
        engine=sqlite_engine,
        data_dir=tmp_path,
        private_chat_service=service,
    )
    await dispatcher.handle_private_message(event())
    await dispatcher.handle_private_message(event(900000102))
    assert len(service.events) == 2


@pytest.mark.asyncio
async def test_dispatcher_caches_quoted_private_image_and_preserves_message_contract(
    sqlite_engine, tmp_path, monkeypatch
):
    image_buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color="red").save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    requests = []

    def download(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url)))
        return httpx.Response(
            200,
            content=image_bytes,
            headers={"content-type": "image/png"},
        )

    incoming = parse_private_message_event(
        {
            "post_type": "message",
            "message_type": "private",
            "sub_type": "friend",
            "message_id": "private-image-1",
            "user_id": 900000101,
            "sender": {"user_id": 900000101, "nickname": "test-user"},
            "time": 1791504000,
            "message": [
                {"type": "reply", "data": {"id": "private-quoted-1"}},
                {"type": "text", "data": {"text": "看这张图"}},
                {
                    "type": "image",
                    "data": {
                        "file": "test-image.png",
                        "url": "https://img.example.test/private-image.png",
                    },
                },
            ],
        }
    )
    service = FakeService()
    dispatcher = PrivateMessageDispatcher(
        engine=sqlite_engine,
        data_dir=tmp_path,
        private_chat_service=service,
    )
    with httpx.Client(transport=httpx.MockTransport(download)) as client:
        def cache_with_offline_download(raw_payload, *, cache_dir):
            cache_images_in_raw_payload(
                raw_payload, cache_dir=cache_dir, http_client=client
            )

        monkeypatch.setattr(
            dispatcher_module, "cache_images_in_raw_payload", cache_with_offline_download
        )
        await dispatcher.handle_private_message(incoming)
        await dispatcher.handle_private_message(incoming)

    assert requests == [("GET", "https://img.example.test/private-image.png")]
    assert len(service.events) == 1
    forwarded = service.events[0]
    assert forwarded.msg_type == "mixed"
    assert forwarded.reply_to_msg_id == "private-quoted-1"
    assert forwarded.plain_text == "看这张图"
    assert len(forwarded.images) == 1
    attachment = forwarded.images[0]
    cached_path = Path(attachment.local_path)
    assert cached_path.is_relative_to(tmp_path / "image_cache")
    assert cached_path.read_bytes() == image_bytes
    assert attachment.source_message_id == "private-image-1"
    assert attachment.source_user_id == 900000101
    assert attachment.source_nickname == "test-user"

    with session_scope(sqlite_engine) as session:
        row = MessageRepository(session).get_by_platform_msg_id(
            "private-inbound-900000101-private-image-1"
        )
        assert row is not None
        assert row.group_id is None
        assert row.msg_type == "mixed"
        assert row.reply_to_msg_id == "private-quoted-1"
        assert row.raw_json["message"][2]["data"]["local_path"] == str(cached_path)


def test_private_entry_point_does_not_load_group_agent_or_memory_composition():
    # The suite may be started with an injected dependency path (for example
    # the bundled Python 3.12 runtime). The fresh import must use that same
    # environment; otherwise a missing dependency hides the module boundary.
    dependency_path = os.pathsep.join(str(Path(path).resolve()) for path in sys.path if path)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-X",
            "utf8",
            "-c",
            (
                "import json,sys; import app.private_main; "
                "print(json.dumps([name for name in sys.modules if "
                "name in ('app.main','app.core.router','app.runtime.memory_composition') "
                "or name.startswith('app.agent')]))"
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONPATH": dependency_path},
        check=True,
    )
    assert json.loads(result.stdout) == []
