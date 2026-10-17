from hashlib import sha256

from scripts.backfill_public_social_activity import public_native_paths


def test_history_discovery_excludes_private_group_and_owner_task_namespaces(tmp_path):
    for scope in ['public', 'qzone_comment', 'background:public:qzone_comment',
                  'group:900000001', 'private:900000101', 'owner:public:42']:
        root = tmp_path/'agent_sessions'/'900000501'/sha256(scope.encode()).hexdigest()
        path = root/'harness/sessions/workspace/session-id/session.v3.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text(scope)
    found = public_native_paths(tmp_path, 900000501)
    assert {p.read_text() for p in found} == {'public', 'qzone_comment', 'background:public:qzone_comment'}
    assert not public_native_paths(tmp_path, 900000502)


def test_history_discovery_obeys_file_count_and_total_byte_budgets(tmp_path):
    root = tmp_path/'agent_sessions'/'900000501'/sha256(b'public').hexdigest()
    for n in range(6):
        path = root/f'harness/sessions/workspace/{n}/session.v3.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text('x' * 100)
    assert len(public_native_paths(tmp_path, 900000501, max_files=2)) == 2
    assert len(public_native_paths(tmp_path, 900000501, max_bytes=250)) == 2


def test_public_namespace_alias_cannot_read_a_private_directory(tmp_path):
    private = tmp_path/'private'
    path = private/'harness/sessions/workspace/private-session/session.v3.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text('PRIVATE')
    public = tmp_path/'agent_sessions'/'900000501'/sha256(b'public').hexdigest()
    public.parent.mkdir(parents=True)
    try:
        public.symlink_to(private, target_is_directory=True)
    except OSError:
        import pytest
        pytest.skip('directory symlinks unavailable')
    assert not public_native_paths(tmp_path, 900000501)
