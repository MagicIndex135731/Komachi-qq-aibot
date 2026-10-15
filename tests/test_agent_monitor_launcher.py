from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import importlib.util
import json

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('monitor_launcher', ROOT / 'scripts/open_agent_monitor.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_existing_viewer_does_not_spawn_or_read_docker(monkeypatch):
    monkeypatch.setattr(launcher, 'viewer_ready', lambda: True)
    run = Mock(); spawn = Mock()
    assert launcher.ensure_viewer(run=run, spawn=spawn) == 'existing'
    run.assert_not_called(); spawn.assert_not_called()


def test_fallback_uses_read_only_volume_and_hidden_native_process(tmp_path, monkeypatch):
    data = tmp_path / 'records'; data.mkdir()
    monkeypatch.setattr(launcher, 'viewer_ready', Mock(side_effect=[False, True]))
    monkeypatch.setattr(launcher, 'resolve_data_root', lambda **_: data)
    monkeypatch.setattr(launcher, 'RUNTIME', tmp_path / 'runtime')
    spawn = Mock()
    assert launcher.ensure_viewer(spawn=spawn) == 'windows'
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
    launcher.serve(tmp_path)
    create.assert_called_once_with(tmp_path, port=8765)
    fake.serve_forever.assert_called_once()
    fake.server_close.assert_called_once()


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
