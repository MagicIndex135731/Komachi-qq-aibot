"""Public-address CONNECT with original Host/SNI, without changing system DNS."""
from __future__ import annotations

import http.client
import ipaddress
import ssl
import time
from urllib.parse import urlparse

import certifi
import httpx


def remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise httpx.ReadTimeout('page_proxy_deadline')
    return value


class _PageStream(httpx.SyncByteStream):
    def __init__(self, connection, response, deadline):
        self.connection, self.response, self.deadline = connection, response, deadline

    def __iter__(self):
        try:
            while True:
                timeout = remaining(self.deadline)
                if self.connection.sock is not None:
                    self.connection.sock.settimeout(timeout)
                block = self.response.read1(65536)
                if not block:
                    break
                yield block
        except (OSError, http.client.HTTPException) as exc:
            raise httpx.ReadError('page_proxy_stream_failed') from exc

    def close(self):
        self.response.close()
        self.connection.close()


class ValidatedPageProxyTransport(httpx.BaseTransport):
    """The local proxy receives a validated IP; TLS still authenticates the domain."""
    def __init__(self, proxy_url: str, resolve, *, connection_factory=http.client.HTTPConnection):
        proxy = urlparse(proxy_url)
        if proxy.scheme != 'http' or proxy.hostname not in {'127.0.0.1', 'localhost', '::1'} or proxy.port != 18797:
            raise ValueError('invalid_page_proxy')
        self.proxy, self.resolve, self.connection_factory = proxy, resolve, connection_factory
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != 'GET':
            raise httpx.RequestError('page_proxy_get_only', request=request)
        timeout = request.extensions.get('timeout', {}).get('read') or 45
        deadline = request.extensions.get('page_deadline', time.monotonic() + timeout)
        connection = None
        try:
            address = self.resolve(str(request.url), deadline)[0]
            if not ipaddress.ip_address(address).is_global:
                raise ValueError('nonpublic_address')
            target = request.url
            port = target.port or (443 if target.scheme == 'https' else 80)
            connection = self.connection_factory(self.proxy.hostname, self.proxy.port, timeout=remaining(deadline))
            connection.set_tunnel(address, port=port)
            connection.connect()
            connection.sock.settimeout(remaining(deadline))
            if target.scheme == 'https':
                connection.sock = self.ssl_context.wrap_socket(connection.sock, server_hostname=target.host)
            # Explicit Host is the original origin, not the pinned IP/proxy.
            headers = {key: value for key, value in request.headers.items()
                if key.lower() not in {'host', 'accept-encoding', 'proxy-authorization'}}
            headers['Host'] = target.netloc.decode('ascii')
            headers['Accept-Encoding'] = 'identity'
            connection.request('GET', target.raw_path.decode('ascii'), headers=headers)
            connection.sock.settimeout(remaining(deadline))
            response = connection.getresponse()
            return httpx.Response(response.status, headers=response.getheaders(),
                stream=_PageStream(connection, response, deadline), request=request)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            if connection is not None:
                connection.close()
            raise httpx.RequestError('page_proxy_request_failed', request=request) from exc
