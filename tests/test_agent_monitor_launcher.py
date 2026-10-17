from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import importlib.util
import json
import errno

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('monitor_launcher', ROOT / 'scripts/open_agent_monitor.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_existing_viewer_does_not_spawn_or_read_docker(monkeypatch):
    monkeypatch.setattr(launcher, 'viewer_ready', lambda *args: True)
    run = Mock(); spawn = Mock()
    assert launcher.ensure_viewer(run=run, spawn=spawn) == launcher.URL
    run.assert_not_called(); spawn.assert_not_called()


def test_fallback_uses_read_only_volume_and_hidden_native_process(tmp_path, monkeypatch):
    data = tmp_path / 'records'; data.mkdir()
    monkeypatch.setattr(launcher, 'viewer_ready', Mock(side_effect=[False, True]))
    monkeypatch.setattr(launcher, 'resolve_data_root', lambda **_: data)
    monkeypatch.setattr(launcher, 'RUNTIME', tmp_path / 'runtime')
    spawn = Mock()
    def start(*args, **kwargs):
        (launcher.RUNTIME / 'viewer.json').write_text('{"port":54321}', encoding='utf8')
        return Mock()
    spawn.side_effect = start
    assert launcher.ensure_viewer(spawn=spawn) == 'http://127.0.0.1:54321/'
    args, kwargs = spawn.call_args
    command = args[0]
    assert '--serve' in command and str(data) in command
    assert kwargs['creationflags'] == launcher.NO_WINDOW
    assert kwargs.get('shell', False) is False
    assert '7897' not in ' '.join(command)


@pytest.mark.parametrize('mounts', [[], [{'Destination':'/workspace/data','Source':'/var/lib/docker/volumes/test/_data','RW':True}], [{'Destination':'/workspace/data','Source':'/etc','RW':False}]])
def test_invalid_or_writable_mount_is_rejected(mounts):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps(mounts)))
    with pytest.raises(RuntimeError): launcher.resolve_data_root(run=run)


def test_mount_lookup_never_reads_bot_environment(monkeypatch):
    mounts = [{'Destination':'/workspace/data','Source':'/var/lib/docker/volumes/synthetic/_data','RW':False}]
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps(mounts)))
    monkeypatch.setattr(Path, 'is_dir', lambda _: True)
    root = launcher.resolve_data_root(run=run)
    assert 'wsl.localhost' in str(root) and 'agent_monitor' in str(root)
    args = run.call_args.args[0]
    assert '{{json .Mounts}}' in ' '.join(args) and '{{json .HostConfig.Mounts}}' in ' '.join(args)
    assert 'xiaomachi-monitor' in args
    assert '.Config.Env' not in ' '.join(args)


def test_serve_keeps_existing_loopback_server_contract(tmp_path, monkeypatch):
    from app.monitor import server
    fake = Mock()
    create = Mock(return_value=fake)
    monkeypatch.setattr(server, 'create_server', create)
    monkeypatch.setattr(launcher, 'RUNTIME', tmp_path / 'runtime')
    fake.server_port = 8765
    launcher.serve(tmp_path)
    create.assert_called_once_with(tmp_path, port=8765)
    fake.serve_forever.assert_called_once()
    fake.server_close.assert_called_once()


@pytest.mark.parametrize('error', [PermissionError(errno.EACCES, 'reserved'), OSError(errno.EADDRINUSE, 'occupied')])
def test_reserved_or_busy_port_uses_system_assigned_port(tmp_path, monkeypatch, error):
    from app.monitor import server
    fake = Mock(server_port=54321)
    create = Mock(side_effect=[error, fake])
    monkeypatch.setattr(server, 'create_server', create)
    monkeypatch.setattr(launcher, 'RUNTIME', tmp_path / 'runtime')
    launcher.serve(tmp_path)
    assert [call.kwargs['port'] for call in create.call_args_list] == [8765, 0]
    assert json.loads((launcher.RUNTIME / 'viewer.json').read_text()) == {'port': 54321}
    fake.serve_forever.assert_called_once()
    fake.server_close.assert_called_once()


def test_saved_viewer_is_health_checked_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'RUNTIME', tmp_path)
    (tmp_path / 'viewer.json').write_text('{"port":54321}')
    probe = Mock(side_effect=[False, True])
    monkeypatch.setattr(launcher, 'viewer_ready', probe)
    run = Mock(); spawn = Mock()
    assert launcher.ensure_viewer(run=run, spawn=spawn) == 'http://127.0.0.1:54321/'
    assert probe.call_args.args == ('http://127.0.0.1:54321/',)
    run.assert_not_called(); spawn.assert_not_called()


@pytest.mark.parametrize('state', ['bad json', '[]', '{"port":true}', '{"port":7897}', '{"port":80}', '{"port":65536}', '{"port":"54321"}'])
def test_invalid_saved_port_is_ignored(tmp_path, monkeypatch, state):
    monkeypatch.setattr(launcher, 'RUNTIME', tmp_path)
    (tmp_path / 'viewer.json').write_text(state)
    assert launcher.saved_viewer_url() is None


def test_other_startup_errors_do_not_retry_or_publish_state(tmp_path, monkeypatch):
    from app.monitor import server
    create = Mock(side_effect=OSError('synthetic records failure'))
    monkeypatch.setattr(server, 'create_server', create)
    monkeypatch.setattr(launcher, 'RUNTIME', tmp_path)
    with pytest.raises(OSError):
        launcher.serve(tmp_path)
    assert create.call_count == 1
    assert not (tmp_path / 'viewer.json').exists()


def test_dead_child_reports_failure_without_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'RUNTIME', tmp_path)
    monkeypatch.setattr(launcher, 'viewer_ready', lambda *args: False)
    monkeypatch.setattr(launcher, 'resolve_data_root', lambda **kwargs: tmp_path)
    sleep = Mock(); monkeypatch.setattr(launcher.time, 'sleep', sleep)
    with pytest.raises(RuntimeError, match='viewer.log'):
        launcher.ensure_viewer(spawn=Mock(return_value=Mock(poll=Mock(return_value=1))))
    sleep.assert_not_called()


def test_stale_state_does_not_prevent_recovery_and_wait_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'RUNTIME', tmp_path)
    (tmp_path / 'viewer.json').write_text('{"port":54321}')
    monkeypatch.setattr(launcher, 'viewer_ready', lambda *args: False)
    monkeypatch.setattr(launcher, 'resolve_data_root', lambda **kwargs: tmp_path)
    sleep = Mock(); monkeypatch.setattr(launcher.time, 'sleep', sleep)
    spawn = Mock(return_value=Mock(poll=Mock(return_value=None)))
    with pytest.raises(RuntimeError, match='viewer.log'):
        launcher.ensure_viewer(spawn=spawn)
    assert spawn.call_count == 1 and sleep.call_count == 30


def test_browser_opens_selected_port(monkeypatch, capsys):
    monkeypatch.setattr(launcher.sys, 'argv', ['open_agent_monitor.py'])
    monkeypatch.setattr(launcher, 'ensure_viewer', lambda: 'http://127.0.0.1:54321/')
    open_browser = Mock(); monkeypatch.setattr(launcher.webbrowser, 'open', open_browser)
    assert launcher.main() == 0
    open_browser.assert_called_once_with('http://127.0.0.1:54321/')
    assert 'http://127.0.0.1:54321/' in capsys.readouterr().out


def test_health_probe_bypasses_proxy_and_checks_live_records(monkeypatch):
    from io import BytesIO
    replies = [BytesIO(b'{"status":"ok","read_only":true}'), BytesIO('小町 · Agent 现场'.encode()), BytesIO(b'{"read_only":true,"conversations":[]}')]
    opener = Mock(); opener.open.side_effect = replies
    builder = Mock(return_value=opener)
    monkeypatch.setattr(launcher, 'build_opener', builder)
    assert launcher.viewer_ready()
    assert builder.call_args.args[0].proxies == {}
    assert [v.args[0] for v in opener.open.call_args_list] == [launcher.URL+'health',launcher.URL,launcher.URL+'api/conversations']


def test_unrelated_server_is_not_considered_monitor(monkeypatch):
    from io import BytesIO
    opener = Mock(); opener.open.return_value = BytesIO(b'{"status":"ok"}')
    monkeypatch.setattr(launcher, 'build_opener', lambda *_: opener)
    assert not launcher.viewer_ready()
    assert opener.open.call_count == 1


def test_unreadable_records_fail_health_check(monkeypatch):
    from io import BytesIO
    opener = Mock(); opener.open.side_effect = [BytesIO(b'{"status":"ok","read_only":true}'),BytesIO('小町 · Agent 现场'.encode()),OSError('synthetic offline records')]
    monkeypatch.setattr(launcher, 'build_opener', lambda *_: opener)
    assert not launcher.viewer_ready()


@pytest.mark.parametrize('subpath,accepted',[('agent_monitor',True),('../bot.db',False),('',False)])
def test_fallback_accepts_only_monitor_subpath_mount(monkeypatch,subpath,accepted):
    mount={'Destination':'/workspace/data/agent_monitor','Source':'/var/lib/docker/volumes/synthetic/_data',
           'Name':'synthetic','RW':True}
    option={'Target':mount['Destination'],'Source':mount['Name'],'Type':'volume','VolumeOptions':{'Subpath':subpath}}
    run=Mock(return_value=SimpleNamespace(returncode=0,stdout=json.dumps({'mounts':[mount],'options':[option]})))
    monkeypatch.setattr(Path,'is_dir',lambda _:True)
    if accepted:
        assert str(launcher.resolve_data_root(run=run)).endswith('agent_monitor')
    else:
        with pytest.raises(RuntimeError):launcher.resolve_data_root(run=run)


def test_deletion_capable_viewer_passes_health_probe(monkeypatch):
    from io import BytesIO
    opener=Mock();opener.open.side_effect=[BytesIO(b'{"status":"ok","read_only":false,"record_deletion":true,"bot_control":false}'),
        BytesIO('小町 · Agent 现场'.encode()),BytesIO(b'{"read_only":true,"conversations":[]}')]
    monkeypatch.setattr(launcher,'build_opener',lambda *_:opener)
    assert launcher.viewer_ready()
