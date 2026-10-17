from datetime import datetime, timezone
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('hy2_watch', Path(__file__).resolve().parents[1] / 'scripts/watch_hy2.py')
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


def error(stamp='2026-10-16T18:00:00+08:00', group='PROXY', suffix='context deadline exceeded'):
    return f'time="{stamp}" level=warning msg="[TCP] dial {group} (match Match/) error: {suffix}"'


def test_old_and_unrelated_errors_cannot_trigger():
    earliest = datetime.fromisoformat('2026-10-16T18:00:00+08:00')
    assert watch.is_hy2_error(error(), earliest)
    assert not watch.is_hy2_error(error('2026-10-16T17:59:59+08:00'), earliest)
    assert not watch.is_hy2_error(error(group='DIRECT'), earliest)
    assert not watch.is_hy2_error(error(suffix='HTTP 403'), earliest)
    assert not watch.is_hy2_error('level=warning context deadline exceeded', earliest)


def test_trigger_window_and_profile_switch():
    trigger = watch.Trigger()
    assert not trigger.feed(1, 1, True)
    assert not trigger.feed(1, 32, True)
    assert trigger.feed(1, 33, True)
    assert not trigger.feed(10, 34, False)
    assert not trigger.feed(1, 35, True)


def test_tail_ignores_history_and_handles_partial_rotation_and_truncation(tmp_path):
    latest = tmp_path / 'latest.log'
    latest.write_bytes(b'old failure\n')
    tail = watch.LogTail(tmp_path)
    assert tail.poll() == []
    with latest.open('ab') as f:
        f.write(b'first new\npart')
    assert tail.poll() == ['first new']
    latest.rename(tmp_path / 'rotated.log')
    latest.write_bytes(b'new file\n')
    with (tmp_path / 'rotated.log').open('ab') as f:
        f.write(b'ial\n')
    assert set(tail.poll()) == {'new file', 'partial'}
    latest.write_bytes(b'x\n')
    assert tail.poll() == ['x']


def test_capture_does_not_touch_existing_session(tmp_path):
    calls = []
    def execute(args, *rest):
        calls.append(args)
        return {'exit': 0, 'stdout': 'Running file other.etl'}
    capture = watch.Capture(tmp_path, '192.0.2.1', execute)
    with pytest.raises(RuntimeError, match='session_active'):
        capture.start()
    capture.finish()
    assert len(calls) == 1


def test_capture_does_not_clear_existing_filters(tmp_path):
    calls = []
    def execute(args, *rest):
        calls.append(args)
        return {'exit': 0, 'stdout': 'Stopped' if args[1] == 'status' else '  1 OtherCapture UDP'}
    capture = watch.Capture(tmp_path, '192.0.2.1', execute)
    with pytest.raises(RuntimeError, match='existing_pktmon'):
        capture.start()
    capture.finish()
    assert not any('remove' in c or 'start' in c for c in calls)


def test_capture_bounds_and_ownership_cleanup(tmp_path):
    calls = []
    capture = None
    def execute(args, *rest):
        calls.append(args)
        if args[1] == 'status':
            return {'exit': 0, 'stdout': str(capture.etl) if capture.started else 'Stopped'}
        if args[1:3] == ['filter', 'list']:
            return {'exit': 0, 'stdout': f' 1 {capture.filter_name}' if capture.filtered else 'No filters'}
        return {'exit': 0, 'stdout': 'ok'}
    capture = watch.Capture(tmp_path, '192.0.2.1', execute)
    capture.start()
    start = next(c for c in calls if c[1] == 'start')
    assert start[start.index('--pkt-size') + 1] == '128'
    assert start[start.index('--file-size') + 1] == '64'
    assert 'circular' in start
    report = capture.finish()
    assert report['stop']['exit'] == 0
    assert report['filter_cleanup']['exit'] == 0
    assert not capture.started


def test_ownership_change_refuses_to_stop(tmp_path):
    calls = []
    def execute(args, *rest):
        calls.append(args)
        return {'exit': 0, 'stdout': 'Running another.etl'}
    capture = watch.Capture(tmp_path, '192.0.2.1', execute)
    capture.started = True
    assert capture.finish()['error'] == 'capture_ownership_changed_stop_refused'
    assert not any(c[1] == 'stop' for c in calls)


def test_sanitization_strips_tokens_and_url_query():
    result = watch.clean_line('password=example token=private https://example.test/a?key=hidden')
    assert 'example' not in result and 'private' not in result and 'hidden' not in result


def test_chinese_pktmon_inactive_status(tmp_path):
    calls = []
    def execute(args, *rest):
        calls.append(args)
        if args[1] == 'status':
            return {'exit': 0, 'stdout': '\u6570\u636e\u5305\u76d1\u89c6\u5668\u6ca1\u6709\u8fd0\u884c\u3002'}
        if args[1:3] == ['filter', 'list']:
            return {'exit': 0, 'stdout': '\u6570\u636e\u5305\u7b5b\u9009\u5668:\n    \u65e0\n'}
        return {'exit': 0, 'stdout': 'ok'}
    capture = watch.Capture(tmp_path, '192.0.2.1', execute)
    capture.start()
    assert capture.started


def test_unknown_filter_listing_is_preserved(tmp_path):
    def execute(args, *rest):
        return {'exit': 0, 'stdout': 'Stopped' if args[1] == 'status' else 'unrecognized listing'}
    capture = watch.Capture(tmp_path, '192.0.2.1', execute)
    with pytest.raises(RuntimeError, match='existing_pktmon'):
        capture.start()


def test_stale_heartbeat_cannot_claim_armed_capture():
    state = {'state': 'armed', 'packet_capture': True, 'updated': '2026-10-16T18:19:35+08:00'}
    result = watch.visible_status(state, datetime.fromisoformat('2026-10-16T20:55:00+08:00'))
    assert result['state'] == 'stale'
    assert result['packet_capture'] is None
    assert state['state'] == 'armed'


def test_completed_capture_is_not_marked_stale():
    state = {'state': 'complete', 'packet_capture': False, 'updated': '2026-10-16T18:19:35+08:00'}
    assert watch.visible_status(state, datetime.fromisoformat('2026-10-16T20:55:00+08:00'))['state'] == 'complete'
