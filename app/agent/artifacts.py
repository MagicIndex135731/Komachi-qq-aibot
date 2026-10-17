"""Scoped durable outputs that retain their canonical evidence across wakes."""

from __future__ import annotations

import hashlib
import io
from itertools import islice
import json
import os
from pathlib import Path
import re
import threading
import time
from uuid import uuid4
import warnings

from PIL import Image
from sqlalchemy import select

from app.core.memory_source_guard import is_usable_memory_source
from app.storage.db import session_scope
from app.storage.models import Message


MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_SOURCES = 200
MAX_MANIFEST_BYTES = 32768
MAX_DIRECTORY_ENTRIES = 512
MAX_LIST_CHECKS = 64
MAX_CATALOG_IDS = 128
MAX_CATALOG_BYTES = 8192
_CATALOG_LOCKS: dict[Path, threading.RLock] = {}
_CATALOG_LOCKS_GUARD = threading.Lock()
_ID = re.compile(r"art-[0-9a-f]{32}\Z")
_IMAGE_TYPES = {"PNG": ("image/png", ".png"), "JPEG": ("image/jpeg", ".jpg"),
                "WEBP": ("image/webp", ".webp"), "GIF": ("image/gif", ".gif")}
_TEXT_TYPES = {".txt": "text/plain", ".md": "text/markdown", ".csv": "text/csv",
               ".tsv": "text/tab-separated-values", ".json": "application/json"}


def _local_file(path: Path, roots: tuple[Path, ...]) -> Path | None:
    try:
        raw = path.absolute()
        actual = raw.resolve(strict=True)
        if not actual.is_file() or not any(actual.is_relative_to(root) for root in roots):
            return None
        # Reject links even when their destination happens to share a root.
        # Walking ancestors also catches linked input subdirectories.
        for item in (raw, *raw.parents):
            if item.is_symlink() or (item.exists() and item.resolve() != item):
                return None
            if item in roots:
                break
        return actual
    except (OSError, ValueError):
        return None


def _validated_type(raw: bytes, suffix: str) -> tuple[str, str]:
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("artifact byte budget exceeded")
    if suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format not in _IMAGE_TYPES or image.width * image.height > MAX_PIXELS:
                        raise ValueError("artifact image format or pixel budget invalid")
                    media_type, extension = _IMAGE_TYPES[image.format]
                    image.verify()
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
            return media_type, extension
        except (OSError, Image.DecompressionBombWarning, Image.DecompressionBombError):
            raise ValueError("artifact pixels cannot be decoded") from None
    media_type = _TEXT_TYPES.get(suffix.lower())
    if media_type is None:
        raise ValueError("artifact file type is unsupported")
    try:
        text = raw.decode("utf-8-sig")
        if any(ord(char) < 32 and char not in "\t\r\n" or ord(char) == 127 for char in text):
            raise ValueError("artifact text has unsupported control bytes")
        if suffix.lower() == ".json":
            json.loads(text)
    except (UnicodeDecodeError, RecursionError):
        raise ValueError("artifact text cannot be decoded") from None
    return media_type, suffix.lower()


def _filename(artifact_id: str, suffix: str) -> str:
    # JSON output and its JSON manifest must occupy distinct paths.
    return artifact_id + (".data" if suffix in _TEXT_TYPES else "") + suffix


class AutonomousArtifactStore:
    def __init__(self, root: Path, *, account_id: int, scope: str, engine,
                 source_message_ids, allowed_input_roots=None, context_snapshot_factory=None) -> None:
        if type(account_id) is not int or not 1 <= account_id <= 2**63 - 1 or not callable(source_message_ids):
            raise ValueError("invalid artifact account or source supplier")
        self.private_user_id = None
        match = re.fullmatch(r"(?:group:([1-9][0-9]{0,18})|owner:group:([1-9][0-9]{0,18}):[1-9][0-9]{0,18})", scope) if isinstance(scope, str) else None
        if match and int(match[1] or match[2]) <= 2**63 - 1:
            self.group_id = int(match[1] or match[2])
        elif isinstance(scope, str) and (scope in {"public", "qzone_publish", "qzone_comment", "profile"}
                or re.fullmatch(r"owner:public:[1-9][0-9]{0,18}", scope)):
            self.group_id = None
        elif isinstance(scope, str) and re.fullmatch(r"private:[1-9][0-9]{0,18}", scope):
            self.group_id = None
            self.private_user_id = int(scope.split(':')[1])
            if self.private_user_id>2**63-1: raise ValueError('invalid private artifact scope')
        else:
            raise ValueError("invalid artifact scope")
        self.root = Path(root).resolve()
        self.account_id, self.scope, self.engine = account_id, scope, engine
        namespace = hashlib.sha256(f"{account_id}:{scope}".encode()).hexdigest()[:24]
        self._catalog = self.root / (".catalog-" + namespace + ".json")
        with _CATALOG_LOCKS_GUARD:
            self._catalog_lock = _CATALOG_LOCKS.setdefault(self._catalog, threading.RLock())
        self.source_message_ids = source_message_ids
        self.context_snapshot_factory = context_snapshot_factory
        self.allowed_input_roots = tuple(Path(value).resolve() for value in (allowed_input_roots if allowed_input_roots is not None else (self.root,)))
        if not self.allowed_input_roots:
            raise ValueError("artifact input roots must be explicit")

    def _sources_usable(self, sources) -> bool:
        if not isinstance(sources, (list, tuple)) or len(sources) > MAX_SOURCES or any(type(item) is not int or not 1 <= item <= 2**63 - 1 for item in sources):
            return False
        if len(set(sources)) != len(sources):
            return False
        if self.private_user_id is not None:
            with session_scope(self.engine) as session:
                rows = list(session.scalars(select(Message).where(Message.group_id.is_(None),
                    Message.user_id == self.private_user_id, Message.id.in_(sources))))
                return len(rows) == len(sources) and all(is_usable_memory_source(row, author_id=self.private_user_id)
                    and isinstance(row.raw_json,dict) and row.raw_json.get('direction') != 'outbound' for row in rows)
        if self.group_id is None:
            return not sources
        if not sources:
            return True
        with session_scope(self.engine) as session:
            rows = list(session.scalars(select(Message).where(Message.group_id == self.group_id, Message.id.in_(sources))))
            return len(rows) == len(sources) and all(is_usable_memory_source(row, group_id=self.group_id) for row in rows)

    def add(self, path: Path, origin: str, model: str = "") -> str:
        if not isinstance(path, Path) or not isinstance(origin, str) or not origin or len(origin) > 120 or not isinstance(model, str) or len(model) > 200:
            raise ValueError("invalid artifact metadata")
        if any(ord(char) < 32 for char in origin + model):
            raise ValueError("invalid artifact metadata")
        sources = tuple(self.source_message_ids())
        if not self._sources_usable(sources):
            raise ValueError("artifact source authority unavailable")
        actual = _local_file(path, self.allowed_input_roots)
        if actual is None:
            raise ValueError("artifact input is outside authorized local roots")
        with actual.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        media_type, suffix = _validated_type(raw, actual.suffix)
        artifact_id = "art-" + uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / _filename(artifact_id, suffix)
        metadata = {"id": artifact_id, "account_id": self.account_id, "scope": self.scope,
                    "file": target.name, "origin": origin, "model": model, "type": media_type,
                    "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                    "source_message_ids": list(sources), "created_ns": time.time_ns()}
        if self.context_snapshot_factory:
            metadata['context_snapshot'] = self.context_snapshot_factory()
        manifest = self.root / (artifact_id + ".meta.json")
        staging = self.root / (artifact_id + ".meta.tmp")
        try:
            target.write_bytes(raw)
            staging.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
            staging.replace(manifest)
        except Exception:
            target.unlink(missing_ok=True)
            staging.unlink(missing_ok=True)
            raise
        self._remember_id(artifact_id)
        return artifact_id

    def _catalog_ids(self) -> tuple[str, ...] | None:
        try:
            path = _local_file(self._catalog, (self.root,))
            if path is None:
                return None
            with path.open("rb") as stream:
                raw = stream.read(MAX_CATALOG_BYTES + 1)
            if len(raw) > MAX_CATALOG_BYTES:
                return None
            value = json.loads(raw)
            if (not isinstance(value, dict) or value.get("account_id") != self.account_id
                    or value.get("scope") != self.scope or not isinstance(value.get("ids"), list)
                    or len(value["ids"]) > MAX_CATALOG_IDS
                    or any(not isinstance(item, str) or not _ID.fullmatch(item) for item in value["ids"])
                    or len(set(value["ids"])) != len(value["ids"])):
                return None
            return tuple(value["ids"])
        except (OSError, ValueError, TypeError, RecursionError):
            return None

    def _fallback_ids(self) -> tuple[str, ...]:
        candidates = []
        try:
            # Count all directory entries, including turn logs and foreign scopes.
            # A glob limit would still scan an unbounded number of non-matches.
            with os.scandir(self.root) as entries:
                for entry in islice(entries, MAX_DIRECTORY_ENTRIES):
                    artifact_id = entry.name.removesuffix(".meta.json")
                    if not entry.name.endswith(".meta.json") or not _ID.fullmatch(artifact_id):
                        continue
                    try:
                        candidates.append((entry.stat(follow_symlinks=False).st_mtime_ns, artifact_id))
                    except OSError:
                        continue
        except OSError:
            return ()
        candidates.sort(reverse=True)
        return tuple(artifact_id for _, artifact_id in candidates[:MAX_LIST_CHECKS])

    def _remember_id(self, artifact_id: str) -> None:
        with self._catalog_lock:
            ids = self._catalog_ids()
            if ids is None:
                ids = self._fallback_ids()
            recent = (artifact_id, *(item for item in ids if item != artifact_id))[:MAX_CATALOG_IDS]
            staging = self.root / (".catalog-" + uuid4().hex + ".tmp")
            try:
                staging.write_text(json.dumps({"account_id": self.account_id, "scope": self.scope,
                                               "ids": recent}), encoding="utf-8")
                staging.replace(self._catalog)
            finally:
                staging.unlink(missing_ok=True)

    def _read(self, artifact_id) -> tuple[dict, Path] | None:
        if not isinstance(artifact_id, str) or not _ID.fullmatch(artifact_id):
            return None
        try:
            manifest = _local_file(self.root / (artifact_id + ".meta.json"), (self.root,))
            if manifest is None:
                return None
            with manifest.open("rb") as stream:
                raw_manifest = stream.read(MAX_MANIFEST_BYTES + 1)
            if len(raw_manifest) > MAX_MANIFEST_BYTES:
                return None
            metadata = json.loads(raw_manifest)
            if (not isinstance(metadata, dict) or metadata.get("id") != artifact_id
                    or type(metadata.get("account_id")) is not int or metadata["account_id"] != self.account_id
                    or metadata.get("scope") != self.scope or not self._sources_usable(metadata.get("source_message_ids"))):
                return None
            filename = metadata.get("file")
            if metadata.get('context_snapshot'):
                from app.agent.context_dependencies import context_snapshot_valid
                if not context_snapshot_valid(self.engine, account_id=self.account_id, scope=self.scope,
                        identifier=metadata['context_snapshot'], group_id=self.group_id,
                        private_user_id=self.private_user_id):
                    return None
            if not isinstance(filename, str) or Path(filename).name != filename:
                return None
            path = _local_file(self.root / filename, (self.root,))
            if path is None:
                return None
            with path.open("rb") as stream:
                raw = stream.read(MAX_BYTES + 1)
            media_type, suffix = _validated_type(raw, path.suffix)
            if (filename != _filename(artifact_id, suffix) or metadata.get("size") != len(raw)
                    or metadata.get("type") != media_type or metadata.get("sha256") != hashlib.sha256(raw).hexdigest()
                    or not isinstance(metadata.get("origin"), str) or len(metadata["origin"]) > 120):
                return None
            return metadata, path
        except (OSError, ValueError, TypeError, RecursionError):
            return None

    def get(self, artifact_id: str) -> Path | None:
        found = self._read(artifact_id)
        return found[1] if found else None

    @staticmethod
    def _description(metadata: dict) -> dict:
        return {"id": metadata["id"], "origin": metadata["origin"], "type": metadata["type"],
                "size": metadata["size"], "hash": metadata["sha256"],
                **({'content_sha256':metadata['content_sha256']} if metadata.get('content_sha256') else {}),
                **({'logical_media_id': metadata['logical_media_id']} if metadata.get('logical_media_id') else {})}

    def bind_media_identity(self, artifact_id: str, identity: str):
        """Trusted downloader metadata only; this never changes source authority."""
        if not isinstance(identity, str) or not 1 <= len(identity) <= 4096:
            raise ValueError('invalid_logical_media_identity')
        found = self._read(artifact_id)
        if not found:raise ValueError('artifact authority unavailable')
        metadata, _ = found
        if metadata.get('origin') not in {'web_page_image','web_page_video'}:
            raise ValueError('artifact is not downloaded media')
        digest = 'media-' + hashlib.sha256(identity.encode()).hexdigest()
        if metadata.get('logical_media_id') and metadata['logical_media_id'] != digest:
            raise ValueError('logical_media_identity_changed')
        metadata['logical_media_id'] = digest
        metadata['content_sha256'] = (json.loads(found[1].read_text(encoding='utf8')).get('sha256')
            if metadata['origin']=='web_page_video' else metadata['sha256'])
        target = self.root / (artifact_id + '.meta.json')
        temporary = self.root / (artifact_id + '.identity.tmp')
        try:
            temporary.write_text(json.dumps(metadata,ensure_ascii=False),encoding='utf8')
            temporary.replace(target)
        finally:temporary.unlink(missing_ok=True)
        return digest

    def describe(self, artifact_id: str) -> dict | None:
        found = self._read(artifact_id)
        return self._description(found[0]) if found else None

    def provenance(self, artifact_id: str) -> tuple[int, ...]:
        """Return validated origins; revoked authority must never become source-free."""
        found = self._read(artifact_id)
        if found is None:
            raise ValueError("artifact authority unavailable")
        return tuple(found[0]["source_message_ids"])

    def list(self) -> tuple[dict, ...]:
        if not self.root.is_dir():
            return ()
        with self._catalog_lock:
            ids = self._catalog_ids()
        if ids is None:
            ids = self._fallback_ids()
        results = []
        for artifact_id in ids[:MAX_LIST_CHECKS]:
            found = self._read(artifact_id)
            if found:
                metadata, _ = found
                results.append(self._description(metadata))
                if len(results) >= 12:
                    break
        return tuple(results)
