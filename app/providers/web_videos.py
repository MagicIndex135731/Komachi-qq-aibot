"""Select only public H.264 MP4 streams belonging to the requested note."""
from __future__ import annotations
from app.core.video_file import MAX_VIDEO_BYTES, MAX_VIDEO_SECONDS
from app.core.url_policy import public_web_url
from app.providers.web_images import WebImageError


def parse_note_video(note: dict, page_url: str) -> dict:
    if note.get('type') != 'video':
        raise WebImageError('web_video_note_has_no_video')
    try:
        items = note.get('video', {}).get('media', {}).get('stream', {}).get('h264', [])
        if not isinstance(items, list):
            raise ValueError()
        candidates = []
        for item in items[:16]:
            if not isinstance(item, dict):
                continue
            size, duration = int(item.get('size') or 0), float(item.get('duration') or 0) / 1000
            if size > MAX_VIDEO_BYTES or duration > MAX_VIDEO_SECONDS:
                continue
            urls = [item.get('masterUrl'), *(item.get('backupUrls') or [])[:3]]
            for value in urls:
                try:
                    url = public_web_url(value)
                except (ValueError, TypeError, AttributeError):
                    continue
                if len(url) <= 4096 and '.m3u8' not in url.lower():
                    candidates.append({'url': url, 'page_url': page_url, 'title': str(note.get('title') or '')[:240],
                        'declared_bytes': size, 'duration_seconds': duration,
                        'width': int(item.get('width') or 0), 'height': int(item.get('height') or 0)})
                    break
        if not candidates:
            raise WebImageError('web_video_no_compatible_stream_within_budget')
        return max(candidates, key=lambda item: (item['width'] * item['height'], item['declared_bytes']))
    except (TypeError, ValueError, AttributeError) as exc:
        if isinstance(exc, WebImageError):
            raise
        raise WebImageError('web_video_state_invalid') from exc
