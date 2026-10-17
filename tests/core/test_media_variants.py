from app.core.media_variants import native_variants,TransferBudget
import pytest


def test_declared_oversize_skipped_and_strict_quality_never_downgrades():
    values=[{'url':'4k','width':3840,'height':2160,'declared_bytes':200_000_000},
        {'url':'720','width':1280,'height':720,'declared_bytes':20_000_000}]
    result=native_variants(values,max_bytes=64*1024*1024)
    assert [i['url'] for i in result]==['720'] and result[0]['quality']=='lower_native'
    assert native_variants(values,max_bytes=64*1024*1024,original_only=True)==[]


def test_estimated_rate_duration_and_receive_bytes_share_budget():
    values=[{'url':'large','bitrate':20_000_000,'duration_seconds':60},
        {'url':'small','bitrate':1_000_000,'duration_seconds':60}]
    assert [i['url'] for i in native_variants(values,max_bytes=10_000_000)]==['small']
    budget=TransferBudget(10)
    budget.request();budget.charge(8)
    assert budget.remaining==2
    with pytest.raises(ValueError):budget.charge(3)
    assert budget.received==11


def test_child_transfer_keeps_one_local_deadline_and_charges_shared_parent():
    import time
    parent=TransferBudget(100,deadline=time.monotonic()+60)
    child=TransferBudget(80,deadline=time.monotonic()+30,parent=parent)
    child.request();child.charge(30)
    assert parent.received==child.received==30 and child.remaining==50
    assert child.deadline<parent.deadline
    with pytest.raises(ValueError):child.charge(51)
    assert parent.received==81


def test_video_limits_accept_128_mib_and_twenty_minutes_but_keep_upper_bounds():
    from app.core.video_file import MAX_VIDEO_BYTES,MAX_VIDEO_SECONDS
    assert MAX_VIDEO_BYTES==128*1024*1024 and MAX_VIDEO_SECONDS==1200
    variants=[{'url':'ordinary-long.mp4','codec':'h264','duration_seconds':1200,
        'declared_bytes':128*1024*1024}]
    assert len(native_variants(variants,max_bytes=MAX_VIDEO_BYTES))==1
    assert not native_variants([{**variants[0],'duration_seconds':1200.1}],max_bytes=MAX_VIDEO_BYTES)
    assert not native_variants([{**variants[0],'declared_bytes':MAX_VIDEO_BYTES+1}],max_bytes=MAX_VIDEO_BYTES)
