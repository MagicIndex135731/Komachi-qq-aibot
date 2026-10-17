"""Bounded Windows HY2 incident recorder. No proxy or network settings are changed."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data' / 'hy2-watch'
VERGE = Path(os.environ.get('APPDATA', '')) / 'io.github.clash-verge-rev.clash-verge-rev'
HIDDEN = 0x08000000 if os.name == 'nt' else 0
PYTHON = Path(sys.executable)
if PYTHON.name.lower() == 'pythonw.exe':
    PYTHON = PYTHON.with_name('python.exe')
TIMEZONE = timezone(timedelta(hours=8))


def now() -> str:
    return datetime.now(TIMEZONE).isoformat()


def visible_status(state: dict, current: datetime | None = None) -> dict:
    result = dict(state)
    age = ((current or datetime.now(TIMEZONE)) - datetime.fromisoformat(state['updated'])).total_seconds()
    result['heartbeat_age_seconds'] = age
    if state['state'] in ('starting', 'armed', 'triggered', 'stopping') and age > 90:
        result.update(state='stale', saved_state=state['state'], packet_capture=None,
                      warning='Recorder heartbeat expired; query PktMon before assuming capture stopped.')
    return result


def write_json(path: Path, value: object) -> None:
    candidate = path.with_suffix(path.suffix + '.next')
    candidate.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    candidate.replace(path)


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def command(args: list[str], timeout: float = 12) -> dict:
    try:
        p = subprocess.run(args, capture_output=True, timeout=timeout, creationflags=HIDDEN)
        encoding = 'utf-16le' if b'\x00' in p.stdout[:80] else 'utf-8'
        return {'exit': p.returncode, 'stdout': p.stdout.decode(encoding, errors='replace')[:150000],
                'stderr': p.stderr.decode('utf-8', errors='replace')[:2000]}
    except subprocess.TimeoutExpired:
        return {'exit': -1, 'error': 'timeout'}
    except OSError as exc:
        return {'exit': -1, 'error': type(exc).__name__}


def powershell(source: str, timeout: float = 15) -> dict:
    return command(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', source], timeout)


def clean_line(line: str) -> str:
    line = re.sub(r'https?://\S+', '<url>', line)
    line = re.sub(r'(?i)(password|secret|token|authorization|cookie)\s*[:=]\s*\S+', r'\1=<redacted>', line)
    return line[:1000]


def source_node() -> tuple[dict, dict]:
    profiles = read_yaml(VERGE / 'profiles.yaml')
    matches = [p for p in profiles['items'] if 'Japan-Public-HY2' in str(p.get('name'))]
    if len(matches) != 1:
        raise ValueError('expected_one_hy2_profile')
    profile = matches[0]
    config = read_yaml(VERGE / 'profiles' / profile['file'])
    nodes = list(config.get('proxies') or [])
    for provider in (config.get('proxy-providers') or {}).values():
        path = (VERGE / provider['path']).resolve()
        if not path.is_relative_to(VERGE.resolve()):
            raise ValueError('provider_path_outside_verge')
        nodes.extend(read_yaml(path).get('proxies') or [])
    nodes = [p for p in nodes if p.get('type') == 'hysteria2']
    if len(nodes) != 1:
        raise ValueError('expected_one_hy2_node')
    ipaddress.ip_address(nodes[0]['server'])
    return profile, nodes[0]


def identity() -> dict:
    p = read_yaml(VERGE / 'profiles.yaml')
    c = read_yaml(VERGE / 'clash-verge.yaml')
    return {'profile_uid': p.get('current'), 'mixed_port': c.get('mixed-port'),
            'tun_enabled': bool((c.get('tun') or {}).get('enable')),
            'config_sha256': hashlib.sha256((VERGE / 'clash-verge.yaml').read_bytes()).hexdigest()}


def is_hy2_error(line: str, earliest: datetime) -> bool:
    if not re.search(r'level=(warning|error)', line):
        return False
    if not re.search(r'dial (PROXY|OpenAI-Fast)\b', line):
        return False
    if not re.search(r'no recent network activity|deadline exceeded|i/o timeout|connection reset|network is unreachable', line, re.I):
        return False
    stamp = re.search(r'time="([^"]+)"', line)
    if not stamp:
        return False
    try:
        return datetime.fromisoformat(stamp[1]) >= earliest
    except ValueError:
        return False


class LogTail:
    """Track file identity across renames and ignore everything predating arming."""
    def __init__(self, directory: Path):
        self.directory = directory
        self.positions: dict[tuple[int, int], int] = {}
        self.pending: dict[tuple[int, int], bytes] = {}
        for f in directory.glob('*.log'):
            s = f.stat()
            self.positions[(s.st_dev, s.st_ino)] = s.st_size

    def poll(self) -> list[str]:
        lines = []
        present = set()
        for path in sorted(self.directory.glob('*.log')):
            try:
                stat = path.stat()
                key = (stat.st_dev, stat.st_ino)
                present.add(key)
                previous = self.positions.get(key, 0)
                offset = previous if previous <= stat.st_size else 0
                with path.open('rb') as stream:
                    stream.seek(max(offset, stat.st_size - 256 * 1024))
                    raw = stream.read(256 * 1024)
                    self.positions[key] = stream.tell()
                pieces = (self.pending.pop(key, b'') + raw).split(b'\n')
                self.pending[key] = pieces.pop()[-4096:]
                lines.extend(part.decode('utf-8', errors='replace') for part in pieces)
            except OSError:
                continue
        self.positions = {k: v for k, v in self.positions.items() if k in present}
        self.pending = {k: v for k, v in self.pending.items() if k in present}
        return lines


class Trigger:
    def __init__(self, threshold: int = 2, window: float = 30):
        self.threshold = threshold
        self.window = window
        self.errors: deque[float] = deque()

    def feed(self, count: int, stamp: float, active: bool) -> bool:
        if not active:
            self.errors.clear()
            return False
        while self.errors and stamp - self.errors[0] > self.window:
            self.errors.popleft()
        self.errors.extend([stamp] * min(count, self.threshold))
        return len(self.errors) >= self.threshold


class Capture:
    def __init__(self, directory: Path, server: str, execute=command):
        self.directory, self.server, self.execute = directory, server, execute
        self.etl = directory / 'hy2-packets.etl'
        self.filter_name = 'HY2Watch-' + directory.name
        self.started = False
        self.filtered = False
        self.owned_filters = None

    def pkt(self, *args: str) -> dict:
        return self.execute(['pktmon.exe', *args])

    def start(self) -> None:
        status = self.pkt('status')
        write_json(self.directory / 'pktmon-before.json', status)
        if status['exit'] != 0:
            raise RuntimeError('pktmon_admin_required')
        text = status.get('stdout', '')
        if not re.search(r'Stopped|Not running|已停止|未运行|没有运行|未启动|停止', text, re.I):
            raise RuntimeError('pktmon_session_active_or_status_unknown')
        filters = self.pkt('filter', 'list')
        write_json(self.directory / 'pktmon-filters-before.json', filters)
        if filters['exit'] != 0 or not re.search(r'No (?:packet )?filters|没有.*筛选器|无.*筛选器|未.*筛选器|^\s*(?:无|None)\s*$', filters.get('stdout', ''), re.I | re.M):
            raise RuntimeError('existing_pktmon_filters_preserved')
        added = self.pkt('filter', 'add', self.filter_name, '-i', self.server)
        if added['exit'] != 0:
            raise RuntimeError('pktmon_filter_failed')
        self.filtered = True
        owned = self.pkt('filter', 'list')
        if owned['exit'] == 0:
            self.owned_filters = owned.get('stdout')
        write_json(self.directory / 'pktmon-filters-owned.json', owned)
        started = self.pkt('start', '--capture', '--pkt-size', '128', '--file-name', str(self.etl),
                           '--file-size', '64', '--log-mode', 'circular')
        write_json(self.directory / 'pktmon-start.json', started)
        if started['exit'] != 0:
            raise RuntimeError('pktmon_start_failed')
        self.started = True

    def finish(self) -> dict:
        report = {}
        if self.started:
            status = self.pkt('status')
            if str(self.etl).casefold() not in status.get('stdout', '').casefold():
                report['error'] = 'capture_ownership_changed_stop_refused'
                return report
            report['counters'] = self.pkt('counters')
            report['stop'] = self.pkt('stop')
            if report['stop']['exit'] != 0:
                report['error'] = 'pktmon_stop_failed'
                return report
            self.started = False
            report['conversion'] = self.execute(['pktmon.exe', 'etl2pcap', str(self.etl), '--out', str(self.directory / 'hy2-packets.pcapng')], 60)
        if self.filtered:
            filters = self.pkt('filter', 'list')
            if filters['exit'] == 0 and self.owned_filters and filters.get('stdout') == self.owned_filters:
                report['filter_cleanup'] = self.pkt('filter', 'remove')
            else:
                report['filter_cleanup'] = {'error': 'filter_set_changed_preserved'}
        pcap = self.directory / 'hy2-packets.pcapng'
        if pcap.is_file() and pcap.stat().st_size > 160 * 1024 * 1024:
            pcap.unlink()
            report['conversion'] = {'error': 'converted_capture_size_limit_exceeded_etl_retained'}
        return report


def host_snapshot() -> dict:
    return powershell("""
$ErrorActionPreference='Stop'
$value=[ordered]@{
time=(Get-Date).ToString('o')
proxy=(Get-ItemProperty 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings' | Select-Object ProxyEnable,ProxyServer)
routes=@(Get-NetRoute -AddressFamily IPv4 | Select-Object DestinationPrefix,NextHop,InterfaceAlias,RouteMetric)
interfaces=@(Get-NetIPInterface | Select-Object InterfaceAlias,AddressFamily,NlMtu,ConnectionState)
statistics=@(Get-NetAdapterStatistics | Select-Object Name,ReceivedBytes,SentBytes,ReceivedDiscardedPackets,OutboundDiscardedPackets,ReceivedPacketErrors,OutboundPacketErrors)
cores=@(Get-CimInstance Win32_Process | Where-Object Name -match 'verge-mihomo|clash-verge|wslrelay|tailscaled|vmware-vmx' | Select-Object Name,ProcessId,@{n='Created';e={$_.CreationDate.ToString('o')}})
listeners=@(Get-NetTCPConnection -State Listen | Where-Object LocalPort -in 7897,18797,19090 | Select-Object LocalAddress,LocalPort,OwningProcess)
udp=@(Get-NetUDPEndpoint | Select-Object LocalAddress,LocalPort,OwningProcess)
events=@(Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-WLAN-AutoConfig/Operational';StartTime=(Get-Date).AddMinutes(-10)} -MaxEvents 25 -ErrorAction SilentlyContinue | Select-Object Id,@{n='Time';e={$_.TimeCreated.ToString('o')}})
}
$value | ConvertTo-Json -Depth 5 -Compress
""", 25)


def wsl_snapshot() -> dict:
    running = command(['wsl.exe', '--list', '--running', '--quiet'])
    if 'Ubuntu_Migrated' not in running.get('stdout', ''):
        return {'skipped': 'distro_not_running'}
    source = """
import subprocess,json,pathlib
commands=[['ip','-j','-s','link'],['ip','route'],['ip','rule'],['ss','-s'],['systemctl','show','xiaomachi-mihomo','-p','MainPID','-p','NRestarts','-p','ExecMainStartTimestamp','-p','ActiveState'],['docker','ps','--format','{{.Names}} {{.Status}}']]
records=[]
for c in commands:
 try:
  p=subprocess.run(c,capture_output=True,text=True,timeout=3);records.append({'command':c,'exit':p.returncode,'stdout':p.stdout[:16000]})
 except Exception as e:records.append({'command':c,'error':type(e).__name__})
records.append({'snmp':pathlib.Path('/proc/net/snmp').read_text()})
print(json.dumps(records))
"""
    return command(['wsl.exe', '-d', 'Ubuntu_Migrated', '--exec', 'python3', '-B', '-c', source], 25)


class PipeSocket:
    def __init__(self, stream):
        self.stream = stream

    def makefile(self, *args, **kwargs):
        return self.stream


def pipe_snapshot() -> dict:
    proc = powershell("Get-CimInstance Win32_Process | Where-Object Name -eq 'verge-mihomo.exe' | Select-Object -First 1 -ExpandProperty CommandLine", 5)
    found = re.search(r'-ext-ctl-pipe\s+(\S+)', proc.get('stdout', ''))
    if not found:
        return {'error': 'core_pipe_unavailable'}
    c = read_yaml(VERGE / 'clash-verge.yaml')
    result = {}
    for endpoint in ('version', 'proxies', 'connections'):
        with open(found[1].strip('"'), 'r+b', buffering=0) as stream:
            stream.write((f'GET /{endpoint} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {c.get("secret", "")}\r\nConnection: close\r\n\r\n').encode())
            response = http.client.HTTPResponse(PipeSocket(stream))
            response.begin()
            data = json.loads(response.read())
        if endpoint == 'version':
            result['version'] = data
        elif endpoint == 'proxies':
            result['groups'] = {n: {k: p.get(k) for k in ('type', 'now', 'all', 'alive')} for n, p in data.get('proxies', {}).items() if n in ('PROXY', 'OpenAI-Fast')}
        else:
            chains = [p.get('chains') for p in data.get('connections') or []]
            result.update(connection_count=len(chains), chains=chains[:100], upload=data.get('uploadTotal'), download=data.get('downloadTotal'))
    return result


def pipe_child() -> dict:
    result = command([str(PYTHON), '-B', str(Path(__file__).resolve()), 'pipe'], 8)
    if result['exit'] == 0:
        try:
            return json.loads(result['stdout'])
        except ValueError:
            pass
    return {'error': 'pipe_snapshot_failed', 'exit': result['exit']}


def curl(url: str, proxy: str = '') -> dict:
    p = command(['curl.exe', '-sS', '--connect-timeout', '4', '--max-time', '6', '--proxy', proxy,
                 '-o', 'NUL', '-w', '%{http_code} %{time_connect} %{time_appconnect} %{time_total}', url], 8)
    return {'time': now(), 'target': url, 'proxy': proxy, **p}


def incident_probes(node: dict) -> dict:
    result = {'time': now(), 'current_proxy': curl('https://www.gstatic.com/generate_204', 'http://127.0.0.1:7897'),
              'direct': curl('https://www.baidu.com/'),
              'wsl_proxy': curl('https://www.gstatic.com/generate_204', 'http://127.0.0.1:18797')}
    binary = Path('C:/Program Files/Clash Verge/verge-mihomo.exe')
    (ROOT / 'data' / 'test-work').mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hy2-', dir=ROOT / 'data' / 'test-work') as name:
        temp = Path(name)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        c = {'mixed-port': port, 'bind-address': '127.0.0.1', 'allow-lan': False, 'mode': 'rule', 'log-level': 'silent',
             'tun': {'enable': False}, 'dns': {'enable': False}, 'proxies': [node], 'rules': [f'MATCH,{node["name"]}']}
        cfg = temp / 'config.yaml'
        cfg.write_text(yaml.safe_dump(c, allow_unicode=True), encoding='utf-8')
        core = subprocess.Popen([str(binary), '-d', name, '-f', str(cfg)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=HIDDEN)
        try:
            time.sleep(0.7)
            result['independent_core_pid'] = core.pid
            result['independent_core'] = curl('https://www.gstatic.com/generate_204', f'http://127.0.0.1:{port}')
        finally:
            core.terminate()
            try:
                core.wait(timeout=4)
            except subprocess.TimeoutExpired:
                core.kill()
                core.wait(timeout=4)
    return result


@contextmanager
def singleton(root: Path):
    import msvcrt
    with (root / 'watch.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def watch(hours: float = 72, smoke: bool = False) -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with singleton(OUTPUT):
        stopfile = OUTPUT / 'stop.request'
        stopfile.unlink(missing_ok=True)
        started = datetime.now(TIMEZONE)
        run = OUTPUT / started.strftime('%Y%m%d-%H%M%S')
        run.mkdir()
        state = {'pid': os.getpid(), 'started': now(), 'deadline': (started + timedelta(hours=hours)).isoformat(),
                 'directory': str(run), 'state': 'starting', 'packet_capture': False}
        def status(**values):
            state.update(values, updated=now())
            write_json(OUTPUT / 'status.json', state)
        status()
        capture = None
        history: deque[dict] = deque(maxlen=240)
        recent: deque[str] = deque(maxlen=600)
        try:
            if shutil.disk_usage(OUTPUT).free < 512 * 1024 * 1024:
                raise RuntimeError('insufficient_free_space_512MiB_required')
            profile, node = source_node()
            state['target_node'] = node['name']
            state['endpoint_hash'] = hashlib.sha256(f'{node["server"]}:{node["port"]}'.encode()).hexdigest()
            tail = LogTail(VERGE / 'logs' / 'sidecar')
            trigger = Trigger()
            capture = Capture(run, str(node['server']))
            capture.start()
            write_json(run / 'baseline-host.json', host_snapshot())
            write_json(run / 'baseline-wsl.json', wsl_snapshot())
            status(state='armed', packet_capture=True)
            deadline = time.monotonic() + hours * 3600
            post_until = None
            next_sample = 0.0
            while time.monotonic() < deadline:
                if stopfile.exists():
                    status(state='stopping', reason='user_stop')
                    break
                stamp = time.monotonic()
                try:
                    current = identity()
                except (OSError, ValueError, yaml.YAMLError):
                    status(snapshot_error='profile_reload_in_progress')
                    time.sleep(2)
                    continue
                active = current['profile_uid'] == profile['uid']
                lines = tail.poll()
                recent.extend(clean_line(line) for line in lines if 'level=warning' in line or 'level=error' in line)
                hits = sum(is_hy2_error(line, started) for line in lines)
                if stamp >= next_sample:
                    sample = {'time': now(), 'identity': current, 'api': pipe_child()}
                    history.append(sample)
                    write_json(run / 'recent-samples.json', list(history))
                    write_json(run / 'recent-warnings.json', list(recent))
                    next_sample = time.monotonic() + 15
                matched = trigger.feed(hits, stamp, active)
                if post_until is None and (matched or smoke):
                    status(state='triggered', trigger='self_test' if smoke else 'new_hy2_errors', triggered_at=now())
                    write_json(run / 'trigger.json', {**state, 'identity': current, 'warnings': list(recent)})
                    write_json(run / 'incident-host.json', host_snapshot())
                    write_json(run / 'incident-wsl.json', wsl_snapshot())
                    write_json(run / 'incident-api.json', pipe_child())
                    if not smoke:
                        write_json(run / 'incident-probes.json', incident_probes(node))
                    post_until = time.monotonic() + (2 if smoke else 60)
                if post_until is not None and time.monotonic() >= post_until:
                    status(reason='incident_captured')
                    break
                status(active_hy2_profile=active)
                time.sleep(2)
            else:
                status(reason='time_limit')
        except Exception as exc:
            status(state='failed', error=str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__)
        finally:
            if capture:
                cleanup = capture.finish()
                write_json(run / 'capture-finish.json', cleanup)
                if cleanup.get('error'):
                    status(state='cleanup_required', error=cleanup['error'])
            write_json(run / 'recent-warnings.json', list(recent))
            if state['state'] not in ('failed', 'cleanup_required'):
                status(state='complete')
            status(packet_capture=bool(capture and capture.started), finished=now())
            write_json(run / 'summary.json', state)
        return state


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'watch', 'status', 'stop', 'pipe', 'smoke', 'arm', 'install'])
    parser.add_argument('--hours', type=float, default=72)
    args = parser.parse_args()
    if not 0 < args.hours <= 72:
        parser.error('--hours must be in (0, 72]')
    if args.action == 'pipe':
        print(json.dumps(pipe_snapshot(), ensure_ascii=False))
    elif args.action == 'install':
        # Task Scheduler owns the worker so ending a Codex turn cannot own its lifetime.
        pythonw = PYTHON.with_name('pythonw.exe')
        arguments = subprocess.list2cmdline(['-B', str(Path(__file__).resolve()), 'arm', '--hours', str(args.hours)])
        quote = lambda value: "'" + str(value).replace("'", "''") + "'"
        script = f"""
$ErrorActionPreference='Stop'
$action=New-ScheduledTaskAction -Execute {quote(pythonw)} -Argument {quote(arguments)} -WorkingDirectory {quote(ROOT)}
$principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Highest
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 74) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName 'Xiaomachi-HY2Watch' -Action $action -Principal $principal -Settings $settings -Description 'Bounded HY2 incident capture; demand start only, no boot or logon trigger.' -Force | Out-Null
Start-ScheduledTask -TaskName 'Xiaomachi-HY2Watch'
"""
        result = powershell(script, 20)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_json(OUTPUT / 'scheduler-install.json', result)
    elif args.action == 'arm':
        production = OUTPUT
        OUTPUT = ROOT / 'data' / 'test-work' / 'qq114-capture-smoke'
        result = watch(1 / 30, smoke=True)
        OUTPUT = production
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_json(OUTPUT / 'self-test.json', result)
        if result['state'] == 'complete' and result.get('reason') == 'incident_captured':
            watch(args.hours)
        else:
            write_json(OUTPUT / 'status.json', {**result, 'state': 'failed', 'error': 'capture_self_test_failed'})
    elif args.action in ('watch', 'smoke'):
        watch(args.hours, smoke=args.action == 'smoke')
    elif args.action == 'status':
        try:
            state = json.loads((OUTPUT / 'status.json').read_text(encoding='utf-8'))
            print(json.dumps(visible_status(state), ensure_ascii=False, indent=2))
        except FileNotFoundError:
            print('Not started')
    elif args.action == 'stop':
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / 'stop.request').touch()
        print('Stop requested; the recorder will finalize its capture and report.')
    else:
        try:
            state = json.loads((OUTPUT / 'status.json').read_text(encoding='utf-8'))
            age = (datetime.now(TIMEZONE) - datetime.fromisoformat(state['updated'])).total_seconds()
            if state['state'] in ('starting', 'armed', 'triggered') and age < 90:
                print('Recorder already running. Use status for details.')
                return
        except (OSError, ValueError, KeyError):
            pass
        pythonw = PYTHON.with_name('pythonw.exe')
        if not pythonw.is_file():
            raise RuntimeError('pythonw_missing')
        params = subprocess.list2cmdline(['-B', str(Path(__file__).resolve()), 'install', '--hours', str(args.hours)])
        rc = ctypes.windll.shell32.ShellExecuteW(None, 'runas', str(pythonw), params, str(ROOT), 0)
        if rc <= 32:
            raise RuntimeError(f'UAC_start_failed_{rc}')
        print('UAC launch requested. Use status to confirm armed packet capture.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_json(OUTPUT / 'fatal-error.json', {'time': now(), 'type': type(exc).__name__})
        raise
