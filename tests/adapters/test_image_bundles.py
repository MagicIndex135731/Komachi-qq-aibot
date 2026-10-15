import asyncio
import base64

import pytest

from app.adapters.sender import Sender, QQMessageDeliveryUncertainError
from app.adapters.outbound_guard import GroupDispatchScope, GroupDispatchRevokedError, group_dispatch_scope
from tests.agent.test_autonomous_social import image_bytes


class Gateway:
    def __init__(self, outcome='ok'):
        self.calls, self.outcome = [], outcome
    async def call_api(self, action, params):
        self.calls.append((action, params))
        if self.outcome == 'timeout':
            raise asyncio.TimeoutError()
        return {'status': 'ok', 'retcode': 0, 'data': {'message_id': 'bundle-receipt'} if self.outcome == 'ok' else {}}


@pytest.mark.parametrize('surface', ['group', 'private'])
async def test_bundle_nodes_use_bot_identity_order_and_single_receipt(tmp_path, surface):
    paths = []
    for n in range(3):
        path = tmp_path / f'{n}.png'
        path.write_bytes(image_bytes())
        paths.append(str(path))
    gateway = Gateway()
    target = {'group_id': 900000001} if surface == 'group' else {'user_id': 900000002}
    actual = await Sender(gateway).send_image_bundle(self_id=900000003, name='小町', text='笔记配图', image_files=tuple(paths), **target)
    assert actual == ('bundle-receipt',) and len(gateway.calls) == 1
    action, params = gateway.calls[0]
    assert action == f'send_{surface}_forward_msg' and len(params['messages']) == 3
    assert params.get('group_id', params.get('user_id')) == next(iter(target.values()))
    for index, node in enumerate(params['messages']):
        assert node['type'] == 'node' and node['data']['uin'] == '900000003' and node['data']['name'] == '小町'
        assert f'{index+1}/3' in node['data']['content'][0]['data']['text']
        assert base64.b64decode(node['data']['content'][1]['data']['file'][9:]) == image_bytes()


@pytest.mark.parametrize('surface', ['group', 'private'])
@pytest.mark.parametrize('outcome', ['timeout', 'missing_receipt'])
async def test_bundle_ambiguous_receipt_never_retries(tmp_path, surface, outcome):
    path = tmp_path / 'one.png'; path.write_bytes(image_bytes())
    gateway = Gateway(outcome)
    target = {'group_id': 900000001} if surface == 'group' else {'user_id': 900000002}
    with pytest.raises(QQMessageDeliveryUncertainError):
        await Sender(gateway).send_image_bundle(self_id=900000003, name='小町', text='图片', image_files=(str(path),), **target)
    assert len(gateway.calls) == 1


async def test_group_forward_preserves_dispatch_revocation(tmp_path):
    path = tmp_path / 'one.png'; path.write_bytes(image_bytes())
    gateway = Gateway()
    scope = GroupDispatchScope(900000001, lambda: False, lambda: None)
    with group_dispatch_scope(scope), pytest.raises(GroupDispatchRevokedError):
        await Sender(gateway).send_image_bundle(self_id=900000003, name='小町', text='图片', image_files=(str(path),), group_id=900000001)
    assert gateway.calls == []
