import asyncio
import json
import pytest
from app.adapters.sender import Sender,QQMessageDeliveryUncertainError
from app.adapters.napcat_ws import NapCatGateway
from app.adapters.outbound_guard import GroupDispatchScope,GroupDispatchRevokedError,group_dispatch_scope


@pytest.mark.parametrize('target',[{'group_id':900000001},{'user_id':900000002}])
async def test_caption_and_mixed_media_follow_real_video_element_contract(tmp_path,target):
    path=tmp_path/'video.mp4';path.write_bytes(b'synthetic-video')
    class Gateway:
        def __init__(self):self.calls=[]
        async def call_api(self,action,params):
            self.calls.append((action,params))
            for node in params.get('messages',()):
                content=node['data']['content']
                if any(item['type']=='video' for item in content):assert len(content)==1
            if 'message' in params and any(item['type']=='video' for item in params['message']):
                assert len(params['message'])==1
            return {'status':'ok','data':{'message_id':'synthetic-receipt'}}
    gateway=Gateway();sender=Sender(gateway)
    await sender.send_video(video_file=str(path),text='视频说明',self_id=900000003,**target)
    assert len(gateway.calls)==1 and gateway.calls[0][0].endswith('forward_msg')
    nodes=gateway.calls[0][1]['messages']
    assert [node['data']['content'][0]['type'] for node in nodes]==['text','video']
    await sender.send_media_bundle(self_id=900000003,name='小町',text='两个视频',
        media_files=(('video',str(path)),('video',str(path))),**target)
    assert [node['data']['content'][0]['type'] for node in gateway.calls[1][1]['messages']]==['text','video','video']


@pytest.mark.parametrize('response,uncertain', [
    ({'status':'failed','retcode':1400,'message':'message element "video" must be the only segment in a message'},False),
    ({},True),({'status':'ok','retcode':'invalid'},True),
    ({'status':'failed','retcode':1200,'message':'waitForSelfEcho timeout'},True)])
async def test_explicit_bridge_rejection_and_unknown_response_are_distinct(tmp_path,response,uncertain):
    from app.adapters.sender import QQMessageRejectedError
    path=tmp_path/'video.mp4';path.write_bytes(b'synthetic-video')
    class Gateway:
        async def call_api(self,*_):return response
    with pytest.raises(QQMessageDeliveryUncertainError if uncertain else QQMessageRejectedError) as error:
        await Sender(Gateway()).send_video(video_file=str(path),group_id=900000001)
    if not uncertain:assert error.value.code=='qq_video_requires_standalone_message'


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
@pytest.mark.parametrize('forward',[True,False])
async def test_gateway_only_extends_video_upload_timeout(monkeypatch,video,forward):
    gateway=NapCatGateway(ws_url='ws://synthetic.invalid')
    class Socket:
        async def send(self,value):
            assert json.loads(value)['action']==('send_private_forward_msg' if forward else 'send_private_msg')
    gateway.websocket=Socket()
    timeouts=[]
    async def wait(future,*,timeout):
        timeouts.append(timeout);future.set_result({'status':'ok'});return await future
    monkeypatch.setattr('app.adapters.napcat_ws.asyncio.wait_for',wait)
    message=[{'type':'video','data':{'file':'base64://synthetic'}}] if video else '普通消息'
    if forward:
        content=message if isinstance(message,list) else [{'type':'text','data':{'text':message}}]
        params={'user_id':900000001,'messages':[{'type':'node','data':{'content':content}}]}
    else:params={'user_id':900000001,'message':message}
    await gateway.call_api('send_private_forward_msg' if forward else 'send_private_msg',params)
    assert timeouts==[90.0 if video else 20.0]
