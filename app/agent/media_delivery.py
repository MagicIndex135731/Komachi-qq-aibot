"""Resolve ordered mixed media through scoped artifact and cache validation."""
from app.core.downloaded_media import DownloadedMediaCache


def resolve_media_bundle(store, settings, identifiers, *, before_read):
    if (not isinstance(identifiers, list) or not 1 <= len(identifiers) <= 20
            or any(not isinstance(value, str) for value in identifiers) or len(set(identifiers)) != len(identifiers)):
        raise ValueError('invalid_media_bundle')
    cache = DownloadedMediaCache.for_store(store, settings)
    cache.pin(identifiers)
    files = []
    for identifier in identifiers:
        path = before_read(identifier)
        description = store.describe(identifier)
        if description and description.get('origin') == 'web_page_video':
            path, _ = cache.video(store, identifier)
            kind = 'video'
        elif path is not None and path.suffix in {'.png', '.jpg', '.webp', '.gif'}:
            kind = 'image'
        else:
            raise ValueError('invalid_media_bundle_artifact')
        files.append((kind, str(path)))
    return tuple(files)
