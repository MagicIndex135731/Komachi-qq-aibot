"""Loopback viewer with monitor-record deletion; no Agent control endpoints."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import secrets
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from app.agent.runtime_trace import TraceReader
from app.monitor.metrics import ConversationMetrics
from app.monitor.records import MonitorRecords


def create_server(root: Path, *, port=8765):
    reader = TraceReader(root)
    metrics = ConversationMetrics(reader)
    records = MonitorRecords(reader)
    records.purge_pending()
    deletion_token = secrets.token_urlsafe(24)
    static = Path(__file__).with_name("static")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, status, data, mime="application/json; charset=utf-8"):
            body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode("utf8")
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _local(self):
            host = self.headers.get("Host", "")
            valid = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return (self.client_address[0] == "127.0.0.1" and host in valid
                    and (origin is None or origin in {"http://" + h for h in valid})
                    and self.headers.get("Sec-Fetch-Site", "same-origin") in {"same-origin", "none"})

        def do_GET(self):
            if not self._local():
                return self._reply(403, {"error": "local_access_required"})
            parsed = urlsplit(self.path)
            path, query = unquote(parsed.path), parse_qs(parsed.query)
            if len(self.path) > 4096:
                return self._reply(400, {"error": "request_invalid"})
            try:
                if path == "/health":
                    return self._reply(200, {"status": "ok", "read_only": False,
                        "record_deletion": True, "bot_control": False})
                if path == '/api/controls':
                    return self._reply(200, {'record_deletion': True, 'deletion_token': deletion_token})
                if path == "/api/traces":
                    return self._reply(200, {"traces": reader.list_traces(), "read_only": True})
                if path == '/api/conversations':
                    return self._reply(200, {'conversations': reader.list_conversations(), 'read_only': True})
                if path.startswith('/api/conversations/'):
                    parts = path.split('/')
                    if len(parts) == 5 and parts[4] == 'metrics':
                        return self._reply(200, metrics.read(parts[3]))
                    if len(parts) != 4:
                        return self._reply(400, {'error': 'request_invalid'})
                    return self._reply(200, reader.read_conversation(parts[3]))
                if path.startswith("/api/traces/"):
                    parts = path.split("/")
                    if len(parts) == 4:
                        after = int(query.get("after", ["0"])[0])
                        return self._reply(200, reader.read_trace(parts[3], after=max(0, after)))
                    if len(parts) == 6 and parts[4] == "requests":
                        return self._reply(200, reader.read_request(parts[3], parts[5]))
                    return self._reply(400, {"error": "request_invalid"})
                assets = {"/": ("index.html", "text/html; charset=utf-8"),
                          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                          "/style.css": ("style.css", "text/css; charset=utf-8")}
                if path in assets:
                    name, mime = assets[path]
                    return self._reply(200, (static / name).read_bytes(), mime)
                return self._reply(404, {"error": "not_found"})
            except (ValueError, TypeError):
                return self._reply(400, {"error": "request_invalid"})
            except FileNotFoundError:
                return self._reply(404, {"error": "record_not_found"})
            except OSError:
                return self._reply(503, {"error": "records_unavailable"})

        def do_POST(self):
            self._reply(405, {"error": "read_only"})
        do_PUT = do_PATCH = do_POST

        def do_DELETE(self):
            if not self._local():
                return self._reply(403, {'error': 'local_access_required'})
            if not secrets.compare_digest(self.headers.get('X-Monitor-Token', ''), deletion_token):
                return self._reply(403, {'error': 'delete_token_required'})
            parsed = urlsplit(self.path)
            parts = unquote(parsed.path).split('/')
            if (len(self.path) > 4096 or parsed.query or len(parts) != 4 or parts[:3] != ['', 'api', 'conversations']
                    or self.headers.get('Content-Length', '0') != '0' or self.headers.get('Transfer-Encoding')):
                return self._reply(400, {'error': 'request_invalid'})
            try:
                result = records.delete_conversation(parts[3])
                with metrics.lock:
                    for identifier in result['trace_ids']:
                        metrics.cache.pop(identifier, None)
                return self._reply(200, result)
            except (ValueError, TypeError):
                return self._reply(400, {'error': 'request_invalid'})
            except FileNotFoundError:
                return self._reply(404, {'error': 'record_not_found'})
            except OSError:
                return self._reply(503, {'error': 'records_delete_unavailable'})
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description="小町 Agent 本机监控（可删除监控记录）")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("DATA_DIR", "data")) / "agent_monitor")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = create_server(args.root, port=args.port)
    print(f"agent_monitor_ready url=http://127.0.0.1:{server.server_port} record_deletion=true bot_control=false", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
