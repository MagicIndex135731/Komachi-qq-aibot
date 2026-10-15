"""Shared disk quotas, expiring use leases, and cleanup for web downloads only."""
from __future__ import annotations
import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime
import hashlib
import json
import logging
import math
import os
import shutil
import sqlite3
from pathlib import Path
import re
import threading
import time
from uuid import uuid4

logger = logging.getLogger(__name__)
_ID = re.compile(r'art-[0-9a-f]{32}\Z')
_VIDEO = re.compile(r'webvid-[0-9a-f]{32}\.mp4\Z')
_LOCK = threading.RLock()
LEASE_SECONDS = 1200
DEFAULT_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_RETENTION_SECONDS = 24 * 3600


def _read_json(path, limit=32768):
    if path.is_symlink() or not path.is_file():
        return {}
    try:
        with path.open('rb') as stream:
            raw = stream.read(limit + 1)
        value = json.loads(raw) if len(raw) <= limit else {}
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, RecursionError):
        return {}


class DownloadedMediaCache:
    def __init__(self, data_dir: Path, *, artifact_root=None, max_bytes=DEFAULT_MAX_BYTES,
                 retention_seconds=DEFAULT_RETENTION_SECONDS, clock=time.time, sqlite_path=None):
        self.root = Path(data_dir).resolve() / 'downloaded_media'
        self.artifacts = Path(artifact_root or (Path(data_dir) / 'autonomous_artifacts')).resolve()
        self.max_bytes, self.retention, self.clock = int(max_bytes), float(retention_seconds), clock
        self.sqlite_path = sqlite_path
        if self.max_bytes <= 0 or self.retention <= 0:
            raise ValueError('invalid_media_cache_policy')

    @classmethod
    def for_store(cls, store, settings=None):
        return cls(store.root.parent, artifact_root=store.root,
            max_bytes=int(getattr(settings, 'download_media_cache_max_mib', 512)) * 1024 * 1024,
            retention_seconds=float(getattr(settings, 'download_media_retention_hours', 24)) * 3600,
            sqlite_path=getattr(settings, 'sqlite_path', None))

    def _pending_artifacts(self):
        """Pending durable results survive lease expiry and capacity pressure."""
        if self.sqlite_path is None or not Path(self.sqlite_path).is_file():
            return set()
        try:
            with sqlite3.connect('file:' + Path(self.sqlite_path).resolve().as_posix() + '?mode=ro', uri=True, timeout=2) as db:
                if not db.execute("SELECT 1 FROM sqlite_master WHERE name='jobs'").fetchone():
                    return set()
                rows = db.execute("SELECT job_type,payload_json,run_at FROM jobs WHERE "
                    "(job_type='agent_media_operation' AND status IN ('running','result_ready','delivering')) OR "
                    "(job_type='private_agent_turn' AND status IN ('queued','running') "
                    "AND json_extract(payload_json,'$.delegated_result.kind')='media_result') LIMIT 129").fetchall()
            if len(rows) > 128:
                raise ValueError('media_cache_pending_scan_budget')
            protected = set()
            for kind,raw,run_at in rows:
                if len(raw) > 128000:
                    raise ValueError('media_cache_pending_payload_budget')
                payload = json.loads(raw)
                created = (datetime.fromisoformat(run_at).replace(tzinfo=UTC).timestamp()
                    if kind == 'private_agent_turn' else float(payload.get('created_at', 0)))
                if self.clock() - created > 7 * 86400:
                    continue
                packet = payload.get('delegated_result', {}) if kind == 'private_agent_turn' else payload.get('result', {})
                data = packet.get('data', {})
                values = [*packet.get('preview_artifact_ids', ()), *data.get('artifact_ids', ()),
                    *data.get('media_artifact_ids', ()), *data.get('video_artifact_ids', ())]
                values += [data.get('artifact_id'), data.get('video_artifact_id')]
                protected.update(value for value in values if isinstance(value, str) and _ID.fullmatch(value))
            return protected
        except (sqlite3.Error, TypeError, ValueError, KeyError) as exc:
            raise ValueError('media_cache_pending_state_unavailable') from exc

    @contextmanager
    def _locked(self):
        with _LOCK:
            if self.root.is_symlink() or self.artifacts.is_symlink():
                raise ValueError('media_cache_linked_root')
            self.root.mkdir(parents=True, exist_ok=True)
            lock_path = self.root / '.cleanup.lock'
            if lock_path.is_symlink():
                raise ValueError('media_cache_linked_lock')
            with lock_path.open('a+b') as lock:
                if os.name == 'nt':
                    import msvcrt
                    if lock.tell() == 0:
                        lock.write(b'0'); lock.flush()
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if os.name == 'nt':
                        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(lock, fcntl.LOCK_UN)

    def _write(self, path, data):
        if path.is_symlink():
            raise ValueError('media_cache_linked_file')
        staging = path.with_name('.write-' + uuid4().hex + '.part')
        try:
            staging.write_text(json.dumps(data), encoding='utf8')
            staging.replace(path)
        finally:
            staging.unlink(missing_ok=True)

    def _target(self, manifest):
        item = _read_json(manifest)
        artifact_id = item.get('id', '')
        if (not isinstance(artifact_id, str) or not _ID.fullmatch(artifact_id)
                or manifest.name != artifact_id + '.meta.json'
                or item.get('origin') not in {'web_page_image', 'web_reference_pixels', 'web_page_video'}):
            return None
        suffix = '.data.json' if item['origin'] == 'web_page_video' else Path(str(item.get('file', ''))).suffix
        if suffix not in {'.png', '.jpg', '.webp', '.gif', '.data.json'} or item.get('file') != artifact_id + suffix:
            return None
        path = self.artifacts / item['file']
        if path.is_symlink() or not path.is_file():
            return None
        video = None
        if item['origin'] == 'web_page_video':
            pointer = _read_json(path)
            filename = pointer.get('file', '')
            if not isinstance(filename, str) or not _VIDEO.fullmatch(filename):
                return None
            video = self.root / filename
            if video.is_symlink():
                return None
        try:
            created = float(item['created_ns']) / 1e9
            if not math.isfinite(created):
                return None
            size = manifest.stat().st_size + path.stat().st_size + (video.stat().st_size if video and video.is_file() else 0)
            return {'id': artifact_id, 'path': path, 'meta': manifest, 'video': video,
                'created': created, 'size': max(size, 65536)}
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def _remove(self, item):
        for path in (item['path'], item['meta'], item['video'], self.root / ('.pin-' + item['id'] + '.json')):
            if path is not None and not path.is_symlink():
                path.unlink(missing_ok=True)

    def _cleanup(self, required=0):
        now = self.clock()
        targets, protected, reservations = [], self._pending_artifacts(), 0
        for path in self.root.iterdir():
            if path.is_symlink() or not path.is_file():
                continue
            if path.name.startswith(('.reserve-', '.pin-')) and path.suffix == '.json':
                data = _read_json(path, 2048)
                until = data.get('until', 0)
                valid = type(until) in {float, int} and math.isfinite(until) and now < until <= now + LEASE_SECONDS + 1
                if not valid:
                    path.unlink(missing_ok=True)
                elif path.name.startswith('.reserve-'):
                    size = data.get('bytes', 0)
                    if type(size) is int and 0 < size <= self.max_bytes:
                        reservations += size
                else:
                    protected.add(path.name.removeprefix('.pin-').removesuffix('.json'))
            elif path.name.startswith(('video-', '.write-')) and path.name.endswith('.part') and now - path.stat().st_mtime > LEASE_SECONDS:
                path.unlink(missing_ok=True)
        if self.artifacts.is_dir():
            with os.scandir(self.artifacts) as entries:
                for index, entry in enumerate(entries):
                    if index >= 50000:
                        raise ValueError('media_cache_scan_budget')
                    path = Path(entry.path)
                    if entry.name.endswith('.meta.json') and not entry.is_symlink():
                        item = self._target(path)
                        if item:
                            targets.append(item)
                    elif entry.name.startswith(('.web-download-', '.web-video-')) and not entry.is_symlink() and entry.is_file():
                        if now - entry.stat().st_mtime > LEASE_SECONDS:
                            path.unlink(missing_ok=True)
                    elif entry.name.startswith('.twitter-download-') and not entry.is_symlink() and entry.is_dir():
                        if now - entry.stat().st_mtime > LEASE_SECONDS:
                            if path.resolve().parent != self.artifacts:
                                raise ValueError('media_cache_staging_path_invalid')
                            shutil.rmtree(path)
        targets.sort(key=lambda item: item['created'])
        kept, removed = [], 0
        for item in targets:
            if item['id'] not in protected and now - item['created'] >= self.retention:
                self._remove(item); removed += 1
            else:
                kept.append(item)
        total = sum(item['size'] for item in kept) + reservations
        for item in kept:
            if total + required <= self.max_bytes:
                break
            if item['id'] not in protected:
                self._remove(item); total -= item['size']; removed += 1
        # Interrupted video writes never become permanent orphan files.
        referenced = {item['video'].name for item in kept if item['video'] is not None and item['path'].is_file()}
        for path in self.root.glob('webvid-*.mp4'):
            if _VIDEO.fullmatch(path.name) and not path.is_symlink() and path.name not in referenced and now-path.stat().st_mtime > LEASE_SECONDS:
                path.unlink(missing_ok=True)
        if required and total + required > self.max_bytes:
            raise ValueError('media_cache_full_in_use')
        return {'removed': removed, 'charged_bytes': total, 'protected_count': len(protected)}

    def cleanup(self):
        with self._locked():
            return self._cleanup()

    @contextmanager
    def reserve(self, size):
        if not 0 < size <= self.max_bytes:
            raise ValueError('media_cache_item_exceeds_quota')
        reservation = self.root / ('.reserve-' + uuid4().hex + '.json')
        with self._locked():
            self._cleanup(required=size)
            self._write(reservation, {'bytes': size, 'until': self.clock() + LEASE_SECONDS})
        try:
            yield
        finally:
            with self._locked():
                reservation.unlink(missing_ok=True)

    def pin(self, artifact_ids):
        if not artifact_ids:
            return
        with self._locked():
            for artifact_id in artifact_ids:
                if not isinstance(artifact_id, str) or not _ID.fullmatch(artifact_id):
                    continue
                target = self._target(self.artifacts / (artifact_id + '.meta.json'))
                if target:
                    self._write(self.root / ('.pin-' + artifact_id + '.json'), {'until': self.clock() + LEASE_SECONDS})

    def register(self, store, path, origin):
        # Creation and its first lease are atomic relative to both janitors.
        with self._locked():
            artifact_id = store.add(path, origin)
            try:
                self._write(self.root / ('.pin-' + artifact_id + '.json'), {'until': self.clock() + LEASE_SECONDS})
            except BaseException:
                target = self._target(self.artifacts / (artifact_id + '.meta.json'))
                if target:
                    self._remove(target)
                raise
            return artifact_id

    def discard(self, artifact_ids):
        """Remove this failed operation's registered download artifacts only."""
        with self._locked():
            for identifier in artifact_ids:
                if isinstance(identifier, str) and _ID.fullmatch(identifier):
                    item = self._target(self.artifacts / (identifier + '.meta.json'))
                    if item:
                        self._remove(item)

    def save_video(self, store, path, info):
        from app.core.video_file import MAX_VIDEO_BYTES
        if path.parent != self.root or path.is_symlink() or not 0 < path.stat().st_size <= MAX_VIDEO_BYTES:
            raise ValueError('web_video_cache_path_invalid')
        target = self.root / ('webvid-' + uuid4().hex + '.mp4')
        pointer = self.artifacts / ('.web-video-' + uuid4().hex + '.json')
        self.artifacts.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        try:
            pointer.write_text(json.dumps({'kind': 'downloaded_video', 'file': target.name,
                'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), **info}), encoding='utf8')
            artifact_id = self.register(store, pointer, 'web_page_video')
            return artifact_id
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        finally:
            pointer.unlink(missing_ok=True)

    def video(self, store, artifact_id):
        self.pin((artifact_id,))
        path = store.get(artifact_id)
        description = store.describe(artifact_id)
        if path is None or not description or description.get('origin') != 'web_page_video':
            raise ValueError('web_video_artifact_unavailable')
        pointer = _read_json(path)
        filename = pointer.get('file', '')
        if not isinstance(filename, str) or not _VIDEO.fullmatch(filename):
            raise ValueError('web_video_cache_path_invalid')
        target = self.root / filename
        from app.core.video_file import MAX_VIDEO_BYTES
        if target.is_symlink() or not target.is_file() or not 0 < target.stat().st_size <= MAX_VIDEO_BYTES:
            raise ValueError('web_video_cache_expired')
        if target.stat().st_size != pointer.get('bytes') or hashlib.sha256(target.read_bytes()).hexdigest() != pointer.get('sha256'):
            raise ValueError('web_video_cache_corrupt')
        return target, pointer


class DownloadedMediaJanitor:
    def __init__(self, settings, *, interval_seconds=3600):
        self.cache = DownloadedMediaCache(settings.data_dir,
            max_bytes=int(getattr(settings, 'download_media_cache_max_mib', 512))*1024*1024,
            retention_seconds=float(getattr(settings, 'download_media_retention_hours', 24))*3600,
            sqlite_path=getattr(settings, 'sqlite_path', None))
        self.interval = interval_seconds
        self.task = None

    async def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self._run(), name='downloaded-media-cleanup')

    async def stop(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None

    async def _run(self):
        while True:
            try:
                result = await asyncio.to_thread(self.cache.cleanup)
                logger.info('downloaded_media_cleanup removed=%d charged_bytes=%d protected=%d',
                    result['removed'], result['charged_bytes'], result['protected_count'])
            except (OSError, ValueError) as exc:
                logger.warning('downloaded_media_cleanup_failed error_type=%s', type(exc).__name__)
            await asyncio.sleep(self.interval)
