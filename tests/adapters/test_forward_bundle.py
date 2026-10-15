"""Minimal synthetic protobuf; never contains live QQ payloads."""
import zlib

import pytest

from app.adapters import forward_bundle as bundle


def integer(value):
    out=bytearray()
    while value>127: out.append((value&127)|128); value>>=7
    return bytes(out)+bytes([value])


def b(field,value):
    if isinstance(value,str):value=value.encode()
    return integer(field*8+2)+integer(len(value))+value


def n(field,value):return integer(field*8)+integer(value)


def row(elements):
    author=b(8,b(4,"未核实署名"))+n(1,900000333)
    return b(1,author)+b(2,n(6,100))+b(3,b(1,b''.join(b(2,e) for e in elements)))


def action(name,rows):return b(2,b(1,name)+b(2,b''.join(b(1,r) for r in rows)))


def packet(actions):return b(1,b(4,zlib.compress(actions,wbits=31)))


def image(appid=1407,domain="multimedia.nt.qq.com.cn"):
    picture=b(1,f"/download?appid={appid}&fileid=synthetic")+b(3,domain)
    info=b(1,b(2,picture))
    return b(53,n(1,48)+b(2,info)+n(3,20))


def test_named_rows_text_nested_and_images():
    xml='<msg serviceID="35" action="viewMultiMsg" m_fileName="deeper"/>'
    elements=[b(1,b(1,"合成正文")), b(12,b(1,b'\x01'+zlib.compress(xml.encode()))),image()]
    result=bundle.decode_packet(packet(action("MultiMsg",[])+action("child",[row(elements)])))
    node=result["child"][0]
    assert node['sender']['nickname']=="未核实署名" and node['time']==100
    assert node['message'][0]['data']['text']=="合成正文"
    assert node['message'][1]['type']=='xml'
    assert node['message'][2]['data']['url'].startswith('https://multimedia.nt.qq.com.cn/download?')
    signed=bundle.sign_image_urls(result,{20:'group-token',10:'private-token'})
    assert signed['child'][0]['message'][2]['data']['url'].endswith('rkey=group-token')


def test_image_key_follows_upload_scene_and_rejects_other_hosts():
    result=bundle.decode_packet(packet(action('child',[row([image(1406),image(domain='127.0.0.1')])])))
    signed=bundle.sign_image_urls(result,{10:'private-token',20:'group-token'})
    assert signed['child'][0]['message'][0]['data']['url'].endswith('rkey=private-token')
    assert signed['child'][0]['message'][1]['data']['url']==''


@pytest.mark.parametrize('payload',[b'\x0a\xff',b'\x00',b'\x0b',b'\x0a\x10tiny'])
def test_malformed_wire_is_rejected(payload):
    with pytest.raises(ValueError):bundle.decode_packet(payload)


def test_decompression_and_action_budgets(monkeypatch):
    monkeypatch.setattr(bundle,'MAX_BYTES',128)
    with pytest.raises(ValueError):bundle.decode_packet(packet(b'x'*129))
    monkeypatch.setattr(bundle,'MAX_BYTES',2*1024*1024)
    with pytest.raises(ValueError):bundle.decode_packet(packet(action('duplicate',[])*2))
    monkeypatch.setattr(bundle, 'MAX_BUNDLE_NODES', 1)
    with pytest.raises(ValueError):bundle.decode_packet(packet(action('child',[row([]), row([])])))


@pytest.mark.asyncio
async def test_adapter_embedded_reads_use_only_fixed_read_command_and_cache():
    from app.adapters.forward_messages import ForwardMessageAdapter,ForwardReadError
    class Gateway:
        websocket=object()
        def __init__(self):self.calls=[]
        async def call_api(self,action_name,params):
            self.calls.append((action_name,params))
            if action_name=='send_packet':
                assert params['cmd']==bundle.READ_COMMAND and params['rsp'] is True
                return {'status':'ok','retcode':0,'data':packet(action('child',[row([b(1,b(1,'second'))])])).hex()}
            raise AssertionError('unexpected API')
    gateway=Gateway();adapter=ForwardMessageAdapter(gateway)
    assert (await adapter.read_embedded('synthetic-root','child')).nodes[0].text=='second'
    assert (await adapter.read_embedded('synthetic-root','child')).nodes[0].text=='second'
    assert len(gateway.calls)==1
    with pytest.raises(ForwardReadError,match='forward_embedded_not_found'):
        await adapter.read_embedded('synthetic-root','unknown')
