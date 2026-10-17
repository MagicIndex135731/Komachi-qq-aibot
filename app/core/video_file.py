"""Bounded local MP4 validation. No remote protocols or shell execution."""
from __future__ import annotations
import json
import math
from pathlib import Path
import shutil
import subprocess

MAX_VIDEO_BYTES = 128 * 1024 * 1024
MAX_VIDEO_SECONDS = 1200


def validate_video(path: Path) -> dict:
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_VIDEO_BYTES:
        raise ValueError('web_video_size_budget')
    probe, decoder = shutil.which('ffprobe'), shutil.which('ffmpeg')
    if not probe or not decoder:
        raise ValueError('web_video_validator_unavailable')
    try:
        result = subprocess.run([probe, '-v', 'error', '-protocol_whitelist', 'file,pipe',
            '-show_entries', 'format=duration,format_name:stream=codec_type,codec_name,width,height',
            '-of', 'json', str(path)], capture_output=True, timeout=10, check=True)
        if len(result.stdout) > 65536:
            raise ValueError('web_video_invalid')
        data = json.loads(result.stdout)
        videos = [s for s in data.get('streams', []) if s.get('codec_type') == 'video']
        audio = [s for s in data.get('streams', []) if s.get('codec_type') == 'audio']
        duration = float(data.get('format', {}).get('duration', 0))
        if (len(videos) != 1 or videos[0].get('codec_name') != 'h264' or len(audio) > 1
                or any(s.get('codec_name') not in {'aac', 'mp3'} for s in audio)
                or 'mp4' not in data.get('format', {}).get('format_name', '').split(',')
                or not math.isfinite(duration) or not 0 < duration <= MAX_VIDEO_SECONDS
                or not 0 < int(videos[0].get('width', 0)) * int(videos[0].get('height', 0)) <= 8_294_400):
            raise ValueError('web_video_format_or_duration_invalid')
        subprocess.run([decoder, '-v', 'error', '-xerror', '-protocol_whitelist', 'file,pipe',
            '-i', str(path), '-frames:v', '1', '-an', '-f', 'null', '-'],
            capture_output=True, timeout=10, check=True)
        return {'duration_seconds': duration, 'width': videos[0]['width'], 'height': videos[0]['height'],
            'bytes': path.stat().st_size, 'codec': 'h264', 'mime_type': 'video/mp4'}
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith('web_video_'):
            raise
        raise ValueError('web_video_invalid') from exc
