import copy

import pytest

from app.providers.x_posts import make_snapshot,parse_api_posts,parse_target

PID='900000000000000001';QID='900000000000000002'
URL=f'https://x.com/sample/status/{PID}'

def tweet(pid=PID,account='sample',kind='photo'):
    media={'type':kind,'media_url_https':'https://pbs.twimg.com/media/sample.jpg'}
    if kind!='photo':media['video_info']={'variants':[
        {'content_type':'video/mp4','bitrate':128,'url':'https://video.twimg.com/low.mp4'},
        {'content_type':'video/mp4','bitrate':512,'url':'https://video.twimg.com/high.mp4'}]}
    return {'rest_id':pid,'core':{'user_results':{'result':{'core':{'screen_name':account},
        'privacy':{'protected':False}}}},'legacy':{'id_str':pid,
        'created_at':'Fri Oct 16 05:00:00 +0000 2026','full_text':'synthetic post',
        'extended_entities':{'media':[media]}}}

def test_raw_api_projection_keeps_root_and_excludes_quotes():
    item=tweet();item['quoted_status_result']={'result':tweet(QID,'quoted')};item['secret']='must-not-leak'
    rows=parse_api_posts({'data':{'tweet_results':{'result':item}}})
    assert len(rows)==1 and rows[0]['post_id']==PID and rows[0]['author']=='sample'
    assert 'must-not-leak' not in str(rows) and rows[0]['media'][0]['url'].endswith('name=orig')

def test_retweet_wrong_identity_and_unknown_or_protected_user_are_rejected():
    retweet=tweet();retweet['legacy']['retweeted_status_result']={'result':tweet(QID)}
    wrong=tweet();wrong['legacy']['id_str']=QID
    unknown=tweet();unknown['core']['user_results']['result'].pop('privacy')
    private=tweet();private['core']['user_results']['result']['privacy']['protected']=True
    assert parse_api_posts([retweet,wrong,unknown,private])==[]

def test_video_keeps_highest_mp4_and_missing_stream_keeps_gap():
    assert parse_api_posts(tweet(kind='video'))[0]['media'][0]['url'].endswith('high.mp4')
    item=tweet(kind='video');item['legacy']['extended_entities']['media'][0]['video_info']['variants']=[]
    row=parse_api_posts(item)[0]
    assert not row['media'] and row['media_count']==1 and row['media_error']=='no_mp4_stream'

@pytest.mark.parametrize('target',['https://x.com/messages','https://x.com/sample?token=secret',
    'https://evil.example/sample','https://x.com/sample/status/12/likes'])
def test_restricted_targets(target):
    with pytest.raises(ValueError):parse_target(target)

def test_content_addressed_snapshot_does_not_claim_full_or_latest_coverage():
    target=parse_target(URL);row=parse_api_posts(tweet())[0]
    first=make_snapshot(target,[row],status='ok')
    assert first['snapshot_id']==make_snapshot(target,copy.deepcopy([row]),status='ok')['snapshot_id']
    assert first['posts'][0]['evidence_ref'].startswith(first['snapshot_id'])
    assert first['complete'] is False and first['latest_proven'] is False



def test_nullable_window_omits_lower_bound_and_zero_is_not_unlimited():
    from app.providers.x_posts import compile_search_targets,XSearchOptions
    from datetime import datetime,UTC
    now=datetime(2026,10,17,tzinfo=UTC)
    result=compile_search_targets('synthetic',XSearchOptions(),now,None)
    assert 'since:' not in result[0][1].identifier and 'until:' in result[0][1].identifier
    with pytest.raises(ValueError):compile_search_targets('synthetic',XSearchOptions(),now,0)
