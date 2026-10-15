import asyncio
import json
import pytest
from app.adapters.sender import Sender,QQMessageDeliveryUncertainError
from app.adapters.napcat_ws import NapCatGateway
from app.adapters.outbound_guard import GroupDispatchScope,GroupDispatchRevokedError,group_dispatch_scope


@pytest.mark.parametrize('target',[{'group_id':900000001},{'user_id':900000002}])
async def test_missing_video_receipt_is_unknown_and_never_retried(tmp_path,target):
    path=tmp_path/'video.mp4';path.write_bytes(b'synthetic-video')
    class Gateway:
        calls=[]
        async def call_api(self,action,params):
            self.calls.append((action,params));return {'status':'ok','data':{}}
    gateway=Gateway()
    with pytest.raises(QQMessageDeliveryUncertainError):
        await Sender(gateway).send_video(video_file=str(path),**target)
    assert len(gateway.calls)==1


async def test_video_send_preserves_group_dispatch_revocation(tmp_path):
    path=tmp_path/'video.mp4';path.write_bytes(b'synthetic-video')
    class Gateway:
        async def call_api(self,*args):raise AssertionError('revoked dispatch reached QQ')
    with group_dispatch_scope(GroupDispatchScope(900000001,lambda:False,lambda:None)):
        with pytest.raises(GroupDispatchRevokedError):
            await Sender(Gateway()).send_video(group_id=900000001,video_file=str(path))


@pytest.mark.parametrize('video',[True,False])
async def test_gateway_only_extends_video_upload_timeout(monkeypatch,video):
    gateway=NapCatGateway(ws_url='ws://synthetic.invalid')
    class Socket:
        async def send(self,value):
            assert json.loads(value)['action']=='send_private_msg'
    gateway.websocket=Socket()
    timeouts=[]
    async def wait(future,*,timeout):
        timeouts.append(timeout);future.set_result({'status':'ok'});return await future
    monkeypatch.setattr('app.adapters.napcat_ws.asyncio.wait_for',wait)
    message=[{'type':'video','data':{'file':'base64://synthetic'}}] if video else '普通消息'
    await gateway.call_api('send_private_msg',{'user_id':900000001,'message':message})
    assert timeouts==[90.0 if video else 20.0]
