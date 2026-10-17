import json
import pytest
from app.providers.web_videos import parse_note_video


def test_video_only_uses_current_note_h264_mp4():
    note = {'type': 'video', 'title': '视频', 'video': {'media': {'stream': {
        'h264': [{'masterUrl': 'https://cdn.example.com/a.mp4', 'size': 1000, 'duration': 54000, 'width': 720, 'height': 1280}],
        'h265': [{'masterUrl': 'https://cdn.example.com/wrong.mp4'}]}}}}
    result = parse_note_video(note, 'https://www.xiaohongshu.com/explore/0123456789abcdef01234567')
    assert result['url'] == 'https://cdn.example.com/a.mp4' and result['duration_seconds'] == 54


@pytest.mark.parametrize('change', ['image_note', 'private_url', 'oversize', 'long_video', 'no_h264'])
def test_invalid_video_is_explicitly_rejected(change):
    item = {'masterUrl': 'https://cdn.example.com/a.mp4', 'size': 1000, 'duration': 54000}
    note = {'type': 'video', 'video': {'media': {'stream': {'h264': [item]}}}}
    if change == 'image_note': note['type'] = 'normal'
    if change == 'private_url': item['masterUrl'] = 'http://127.0.0.1/a.mp4'
    if change == 'oversize': item['size'] = 128*1024*1024+1
    if change == 'long_video': item['duration'] = 1200001
    if change == 'no_h264': note['video']['media']['stream'] = {'h265': [item]}
    with pytest.raises(ValueError): parse_note_video(note, 'https://www.xiaohongshu.com/')


@pytest.mark.parametrize('size,duration',[(65*1024*1024,601000),(128*1024*1024,1200000)])
def test_video_above_old_limit_and_at_new_boundary_is_accepted(size,duration):
    note={'type':'video','video':{'media':{'stream':{'h264':[{
        'masterUrl':'https://cdn.example.com/a.mp4','size':size,'duration':duration}]}}}}
    result=parse_note_video(note,'https://www.xiaohongshu.com/')
    assert result['declared_bytes']==size and result['duration_seconds']==duration/1000
