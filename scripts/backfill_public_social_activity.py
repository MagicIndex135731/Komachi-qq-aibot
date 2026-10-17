"""Bounded one-time recovery from the own-public native tool events, never monitors."""
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import argparse
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.harness_session_files import native_file_path
from app.agent.social_memory import import_public_native
from app.config import AppSettings
from app.storage.agent_effects import AgentToolEffect
from app.storage.db import build_engine
from app.storage.public_social import PublicSocialRepository


PUBLIC_HISTORY_SCOPES = ('public', 'background:public:qzone_comment', 'qzone_comment')


def public_native_paths(data_dir, account_id, *, max_files=16, max_bytes=8_000_000):
    """Fixed public namespaces only, with per-file and aggregate read bounds."""
    candidates = []
    for scope in PUBLIC_HISTORY_SCOPES:
        raw_root = data_dir/'agent_sessions'/str(account_id)/sha256(scope.encode()).hexdigest()
        if raw_root.is_symlink():
            continue
        root = native_file_path(raw_root).resolve()
        for path in root.glob('harness/sessions/*/*/session.v3.jsonl'):
            if path.resolve().is_relative_to(root) and path.is_file() and path.stat().st_size <= 2_000_000:
                candidates.append(path)
    selected, size = [], 0
    for path in sorted(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True):
        if len(selected) >= max_files:
            break
        if size + path.stat().st_size <= max_bytes:
            selected.append(path)
            size += path.stat().st_size
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Import bounded recent own-public native tool events')
    args = parser.parse_args()
    settings = AppSettings()
    engine = build_engine(settings.sqlite_path)
    now = datetime.now(UTC)
    try:
        with Session(engine) as session:
            paths = public_native_paths(settings.data_dir, settings.bot_qq)
            if not paths:
                print(json.dumps({'imported': 0, 'reason': 'no_public_native_history'}))
                return
            effects = list(session.scalars(select(AgentToolEffect).where(
                AgentToolEffect.account_id == settings.bot_qq, AgentToolEffect.operation == 'comment_qzone',
                AgentToolEffect.scope.startswith('public:'),
                AgentToolEffect.created_at >= (now-timedelta(days=14)).replace(tzinfo=None))))
            if len(effects) > 1024:
                raise ValueError('public effect import bound exceeded')
            if not args.apply:
                print(json.dumps({'apply': False, 'native_files': len(paths),
                                  'native_bytes': sum(p.stat().st_size for p in paths),
                                  'eligible_effects': len(effects)}))
                return
            repository = PublicSocialRepository(engine, account_id=settings.bot_qq)
            imported = []
            for path in reversed(paths):
                imported.extend(import_public_native(repository, path, effects, now=now))
            print(json.dumps({'apply': True, 'imported_records': len(set(imported)),
                              'native_files': len(paths),
                              'source': 'own_public_native_tool_events', 'platform_writes': 0}))
    finally:
        engine.dispose()


if __name__ == '__main__':
    main()
