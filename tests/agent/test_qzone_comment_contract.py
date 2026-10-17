"""Reading post pixels must not imply permission to attach them to a comment."""
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from app.agent.public_tools import build_public_tools
from app.agent.social_contracts import SocialResult
from app.agent.social_policy import configured_social_jobs
from app.agent.tool_agent import validate_arguments
from tests.adapters.test_qzone import OWNER, post
from tests.agent.test_current_friend_permissions import setup
from tests.agent.test_social_executor import Guard


@pytest.mark.parametrize('allow_images', [False, True])
def test_comment_schema_matches_image_permission(sqlite_engine, allow_images):
    executor, observation = setup(sqlite_engine, object())
    executor.router.runtime.social_policy['qzone']['comment']['allow_images'] = allow_images
    observation = replace(observation, job=configured_social_jobs(executor.router.runtime)[0])
    tool = next(t for t in build_public_tools(executor, observation, Guard(), object())
                if t.name == 'comment_qzone')
    args = {'owner_id': OWNER, 'post_id': post().post_id, 'text': 'synthetic comment', 'artifact_ids': []}
    assert validate_arguments(args, tool.parameters)
    assert validate_arguments({**args, 'artifact_ids': ['art-' + 'a' * 32]}, tool.parameters) == allow_images


async def test_disabled_comment_attachments_fail_before_materializing_or_dispatching(sqlite_engine):
    class Friends:
        async def get_friend_ids(self):
            return (OWNER,)
    executor, observation = setup(sqlite_engine, Friends())
    observation = replace(observation, posts=(post(),))
    guard = Guard()
    tool = next(t for t in build_public_tools(executor, observation, guard, object())
                if t.name == 'comment_qzone')
    result = await tool.handler({'owner_id': OWNER, 'post_id': post().post_id,
                                'text': 'synthetic comment', 'artifact_ids': ['not-materialized']})
    assert result.status == 'blocked'
    assert result.data['error'] == 'comment_images_disabled'
    assert result.data['write_dispatched'] is False
    assert 'artifact_ids=[]' in result.data['guidance']
    assert not guard.dispatched


@pytest.mark.parametrize('status,reason', [
    ('blocked', 'post_changed_or_unavailable'),
    ('failed', 'qzone_definite_rejection'),
    ('uncertain', 'delivery_result_unknown'),
])
async def test_comment_preserves_executor_reason(sqlite_engine, status, reason):
    class Friends:
        async def get_friend_ids(self):
            return (OWNER,)
    executor, observation = setup(sqlite_engine, Friends())
    observation = replace(observation, posts=(post(),))
    async def execute(*args):
        return SocialResult(status, reason=reason)
    executor.legacy = SimpleNamespace(execute=execute)
    tool = next(t for t in build_public_tools(executor, observation, Guard(), object())
                if t.name == 'comment_qzone')
    result = await tool.handler({'owner_id': OWNER, 'post_id': post().post_id,
                                'text': 'synthetic comment', 'artifact_ids': []})
    assert result.status == status and result.data['reason'] == reason


async def test_native_read_pixels_reject_attachment_then_correct_and_send_once(sqlite_engine, tmp_path, monkeypatch):
    from app.agent import public_tools
    from app.agent.social_images import post_image_inputs
    from tests.agent.test_autonomous_social import composed, wake, writes, FRIEND, image_bytes
    from tests.agent.test_mature_composition import model_steps, latest

    args = {'owner_id': FRIEND, 'post_id': 'observed-photo', 'text': '合成图片中的红色很鲜明', 'artifact_ids': []}
    def attach_read_pixels(body):
        viewed = latest(body, 'observe_post_images')['data']
        assert viewed['pixels_attached']
        assert 'artifact_ids=[]' in viewed['guidance']
        return 'comment_qzone', {**args, 'artifact_ids': viewed['artifact_ids']}
    def correct_attachment(body):
        rejected = latest(body, 'comment_qzone')
        assert rejected['status'] == 'blocked'
        assert rejected['data']['handler_executed'] is False
        assert rejected['data']['invalid_arguments'] == [
            {'parameter': 'artifact_ids', 'expected': {'type': 'array', 'maxItems': 0}}]
        return 'comment_qzone', args

    client, requests, _ = model_steps(
        ('read_friend_posts', {'owner_id': FRIEND}),
        ('observe_post_images', {'owner_id': FRIEND, 'post_id': args['post_id']}),
        attach_read_pixels, correct_attachment, None,
        ('read_friend_posts', {'owner_id': FRIEND}), ('comment_qzone', args), None)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, content=image_bytes(), headers={'Content-Type': 'image/png'}))) as transport:
        monkeypatch.setattr(public_tools, 'post_image_inputs', lambda post, *, cache_dir:
            post_image_inputs(post, cache_dir=cache_dir, http_client=transport))
        async with composed(sqlite_engine, tmp_path, client, runtime='harness', kind='qzone_comment',
                            options={'allow_images': False, 'max_steps': 8}) as fixture:
            fixture.gateway.posts[0].update(content='', images=['https://synthetic.qpic.cn/offline.png'])
            await wake(fixture)
            await wake(fixture)
            assert [name for name, _ in writes(fixture)] == ['comment_qzone']
            assert latest(requests[-1], 'comment_qzone')['data']['error'] == 'effect_already_recorded'
            assert not writes(fixture)[0][1].get('images')
