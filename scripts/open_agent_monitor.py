"""Open the local monitor viewer; recover missing Windows/WSL forwarding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'data/agent-monitor-launcher'
URL = 'http://127.0.0.1:8765/'
DISTRO = 'Ubuntu_Migrated'
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)


def viewer_ready() -> bool:
    """Identify the viewer through direct loopback HTTP, including readable data."""
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(URL + 'health', timeout=2) as response:
            health = json.loads(response.read(4096))
        if health.get('status') != 'ok' or not (health.get('read_only') is True
                or health.get('record_deletion') is True and health.get('bot_control') is False):
            return False
        with opener.open(URL, timeout=2) as response:
            if '小町 · Agent 现场'.encode() not in response.read(65536):
                return False
        with opener.open(URL + 'api/conversations', timeout=3) as response:
            listing = json.loads(response.read(4 * 1024 * 1024))
        return listing.get('read_only') is True and isinstance(listing.get('conversations'), list)
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def resolve_data_root(*, run=subprocess.run) -> Path:
    """Only resolve the monitor subdirectory, never the bot environment."""
    result = run(['wsl.exe', '-d', DISTRO, '-u', 'root', '--exec', 'docker', 'inspect',
                  'xiaomachi-monitor', '--format', '{"mounts":{{json .Mounts}},"options":{{json .HostConfig.Mounts}}}'],
                 capture_output=True, text=True, encoding='utf8', timeout=30,
                 creationflags=NO_WINDOW)
    if result.returncode:
        raise RuntimeError('无法读取 WSL 监控服务。请确认小町运行栈已启动。')
    try:
        data = json.loads(result.stdout)
        mounts = data if isinstance(data, list) else data['mounts']
        options = [] if isinstance(data, list) else data.get('options', [])
        values = [m for m in mounts if m.get('Destination') == '/workspace/data' and m.get('RW') is False
            or m.get('Destination') == '/workspace/data/agent_monitor' and any(
                option.get('Target') == '/workspace/data/agent_monitor' and option.get('Type') == 'volume'
                and option.get('Source') == m.get('Name') and option.get('VolumeOptions', {}).get('Subpath') == 'agent_monitor'
                for option in options)]
        if len(values) != 1:
            raise ValueError('monitor_scoped_mount_required')
        source = PurePosixPath(values[0]['Source'])
        if (not source.is_relative_to('/var/lib/docker/volumes') or source.name != '_data'
                or '..' in source.parts):
            raise ValueError('monitor_volume_path_invalid')
        root = Path('\\\\wsl.localhost\\' + DISTRO + '\\' + '\\'.join(source.parts[1:])) / 'agent_monitor'
        if not root.is_dir():
            raise ValueError('monitor_records_unavailable')
        return root
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        raise RuntimeError('监控记录目录不可读取，尚未启动本机查看器。') from exc


def ensure_viewer(*, run=subprocess.run, spawn=subprocess.Popen) -> str:
    if viewer_ready():
        return 'existing'
    root = resolve_data_root(run=run)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / 'viewer.log').open('ab') as log:
        spawn([sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--serve', '--root', str(root)],
              cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
              creationflags=NO_WINDOW)
    for _ in range(30):
        if viewer_ready():
            return 'windows'
        time.sleep(0.2)
    raise RuntimeError('8765 未能启动监控查看器。请查看 data/agent-monitor-launcher/viewer.log；不会占用其他端口。')


def serve(root: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from app.monitor.server import create_server
    server = create_server(root, port=8765)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description='打开或恢复小町本机 Agent 监控')
    parser.add_argument('--no-open', action='store_true', help='检查/恢复，但不打开浏览器')
    parser.add_argument('--serve', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--root', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.serve:
            if args.root is None or not args.root.is_dir():
                raise RuntimeError('监控记录目录不可读取。')
            serve(args.root)
        else:
            ensure_viewer()
            print('Agent 监控已就绪：' + URL)
            if not args.no_open:
                webbrowser.open(URL)
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
