"""Select only public H.264 MP4 streams belonging to the requested note."""
from __future__ import annotations
from app.core.video_file import MAX_VIDEO_BYTES, MAX_VIDEO_SECONDS
from app.core.url_policy import public_web_url
from app.providers.web_images import WebImageError


def parse_note_video(note: dict, page_url: str) -> dict:
    if note.get('type') != 'video':
        raise WebImageError('web_video_note_has_no_video')
    try:
        media = note.get('video', {}).get('media', {})
        if (media.get('video') or {}).get('drmType') not in {None, 0}:
            raise WebImageError('web_video_protected_stream')
        streams = media.get('stream', {})
        # 2026-10 live XHS metadata uses EF4 for the AVC MP4 stream.
        # Its actual bytes are still required to pass ffprobe's H.264 check;
        # EF5/other codecs and covers are never silently substituted.
        groups = [streams.get('h264', []), streams.get('EF4', [])]
        if any(not isinstance(group, list) for group in groups):
            raise ValueError()
        items = [item for group in groups for item in group[:16]]
        candidates = []
        for item in items[:16]:
            if not isinstance(item, dict):
                continue
            if item.get('format') not in {None, '', 'mp4'}:continue
            size, duration = int(item.get('size') or 0), float(item.get('duration') or 0) / 1000
            if duration > MAX_VIDEO_SECONDS:continue
            urls = [item.get('masterUrl'), *(item.get('backupUrls') or [])[:3]]
            for value in urls:
                try:
                    url = public_web_url(value)
                except (ValueError, TypeError, AttributeError):
                    continue
                if len(url) <= 4096 and '.m3u8' not in url.lower():
                    candidates.append({'url': url, 'page_url': page_url, 'title': str(note.get('title') or '')[:240],
                        'declared_bytes': size, 'duration_seconds': duration,'codec':'h264',
                        'bitrate':int(item.get('avgBitrate') or item.get('videoBitrate') or 0),
                        'width': int(item.get('width') or 0), 'height': int(item.get('height') or 0)})
                    break
        if not candidates:
            raise WebImageError('web_video_no_compatible_stream_within_budget')
        from app.core.media_variants import native_variants
        choices=native_variants(candidates,max_bytes=MAX_VIDEO_BYTES)
        if not choices:raise WebImageError('web_video_no_compatible_stream_within_budget')
        return {**choices[0],'variants':candidates}
    except (TypeError, ValueError, AttributeError) as exc:
        if isinstance(exc, WebImageError):
            raise
        raise WebImageError('web_video_state_invalid') from exc
