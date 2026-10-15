import httpx
import pytest
import time
from types import SimpleNamespace
from app.providers.gemini_search import GeminiSearchClient


def test_page_proxy_validates_remote_dns_without_accepting_fake_or_private_addresses():
    calls=[]
    def response(request):
        calls.append(request)
        if request.url.host=='1.1.1.1':
            return httpx.Response(200,json={'Status':0,'Answer':[{'type':1,'TTL':30,'data':'93.184.216.34'}]})
        return httpx.Response(200,headers={'content-type':'text/plain'},text='actual public source')
    client=GeminiSearchClient(provider='gemini',base_url='',api_key='SYNTHETIC',model='',timeout_seconds=10,
        page_proxy_url='http://127.0.0.1:18797',page_http_client=httpx.Client(transport=httpx.MockTransport(response)),
        address_resolver=lambda *_:['198.18.0.2'])
    page=client.open_page('https://raw.githubusercontent.com/example/repo/main/main.py')
    assert 'actual public source' in page.content
    assert calls[0].url.host=='1.1.1.1'
    assert all('authorization' not in r.headers and 'x-goog-api-key' not in r.headers for r in calls)
    client.open_page('https://raw.githubusercontent.com/example/repo/main/main.py',lineno=1)
    assert len(calls)==2


def test_proxy_dns_private_answer_blocks_before_website_dispatch():
    calls=[]
    def response(request):
        calls.append(request)
        return httpx.Response(200,json={'Status':0,'Answer':[{'type':1,'data':'127.0.0.1'}]})
    client=GeminiSearchClient(provider='gemini',base_url='',api_key='SYNTHETIC',model='',timeout_seconds=10,
        page_proxy_url='http://127.0.0.1:18797',page_http_client=httpx.Client(transport=httpx.MockTransport(response)))
    with pytest.raises(Exception):client.open_page('https://example.com/private')
    assert len(calls)==1 and calls[0].url.host=='1.1.1.1'


def test_transport_connects_validated_ip_and_authenticates_original_host():
    from app.providers.page_proxy_transport import ValidatedPageProxyTransport
    events=[]
    class Socket:
        def settimeout(self,timeout):assert timeout>0
    class Response:
        status=200
        def getheaders(self):return [('content-type','text/plain')]
        def read1(self,size):
            if events[-1]=='read':return b''
            events.append('read');return b'actual source'
        def close(self):pass
    class Connection:
        def __init__(self,host,port,timeout):self.sock=Socket();events.append((host,port))
        def set_tunnel(self,host,port):events.append(('connect_ip',host,port))
        def connect(self):pass
        def request(self,method,path,headers):events.append(('request',method,path,headers['Host']))
        def getresponse(self):return Response()
        def close(self):pass
    transport=ValidatedPageProxyTransport('http://127.0.0.1:18797',lambda *_:['93.184.216.34'],connection_factory=Connection)
    def tls(sock,server_hostname):events.append(('tls',server_hostname));return sock
    transport.ssl_context=SimpleNamespace(wrap_socket=tls)
    with httpx.Client(transport=transport) as client:
        response=client.get('https://example.com/source?q=one')
    assert response.text=='actual source'
    assert ('connect_ip','93.184.216.34',443) in events and ('tls','example.com') in events
    assert ('request','GET','/source?q=one','example.com') in events


@pytest.mark.parametrize('proxy_url',['http://127.0.0.1:7897','http://localhost:7897','http://127.0.0.1:7900'])
def test_page_proxy_rejects_windows_proxy_ports(proxy_url):
    from app.providers.page_proxy_transport import ValidatedPageProxyTransport
    with pytest.raises(ValueError):
        ValidatedPageProxyTransport(proxy_url,lambda *_:['93.184.216.34'])
    with pytest.raises(ValueError):
        GeminiSearchClient(provider='gemini',base_url='',api_key='SYNTHETIC',model='',timeout_seconds=10,page_proxy_url=proxy_url)
