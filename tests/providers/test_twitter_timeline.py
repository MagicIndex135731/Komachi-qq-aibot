from datetime import UTC, datetime
import json

import pytest

from app.providers.twitter_timeline import TwitterTimelineClient, parse_account, parse_timeline
from app.providers.web_images import WebImageError


def embedded(tweets, account='synthetic', **extra):
    data={'props':{'pageProps':{'headerProps':{'screenName':account},
        'timeline':{'entries':[{'type':'tweet','entry_id':'tweet-'+t['id_str'],
            'content':{'tweet':t}} for t in tweets]},**extra}}}
    return '<script id="__NEXT_DATA__" type="application/json">'+json.dumps(data)+'</script>'


def tweet(identifier, date, account='synthetic', **extra):
    return {'id_str':identifier,'user':{'screen_name':account},'created_at':date,'full_text':'作品',
        'extended_entities':{'media':[{'type':'photo'},{'type':'video'}]},**extra}


@pytest.mark.parametrize('value',['@synthetic','synthetic','https://x.com/synthetic','https://twitter.com/synthetic/'])
def test_account_normalization(value):
    assert parse_account(value)=='synthetic'


@pytest.mark.parametrize('value',['https://evil.example/synthetic','synthetic/status/123','../user','https://x.com/home','a'*16,'https://x.com/synthetic?token=secret'])
def test_only_accounts_are_accepted(value):
    with pytest.raises(ValueError):parse_account(value)


def test_embedded_timeline_validates_author_ignores_quotes_retweets_and_sorts_dates():
    quote=tweet('900000000000000099','Tue Oct 13 09:00:00 +0000 2026','other')
    html=embedded([
        tweet('900000000000000001','Mon Jan 01 00:00:00 +0000 2024'),
        tweet('900000000000000002','Thu Oct 15 08:00:00 +0000 2026',quoted_status=quote),
        tweet('900000000000000003','Fri Oct 16 01:00:00 +0000 2026','other'),
        tweet('900000000000000004','Fri Oct 16 01:00:00 +0000 2026',retweeted_status=quote),
        tweet('900000000000000005','Fri Oct 16 02:00:00 +0000 2026',in_reply_to_status_id_str='1'),
    ])
    result=parse_timeline(html,'synthetic')
    assert [p['post_id'] for p in result]==['900000000000000002','900000000000000001']
    assert result[0]['media_count']==2
    assert result[0]['media_roles']==['post_image','post_video']
    assert result[0]['source_url']=='https://x.com/synthetic/status/900000000000000002'
    assert result[0]['published_at']=='2026-10-15T08:00:00+00:00'


def test_wrong_profile_or_missing_structure_is_not_empty_success():
    for html in (embedded([],account='other'),'<html>login required</html>'):
        with pytest.raises(WebImageError):parse_timeline(html,'synthetic')


def test_client_limits_freshness_without_claiming_latest():
    seen=[]
    class Reader:
        def fetch(self,url,**kwargs):
            seen.append(url);kwargs['check']()
            return url,embedded([
                tweet('900000000000000001','Mon Jan 01 00:00:00 +0000 2024'),
                tweet('900000000000000002','Thu Oct 15 08:00:00 +0000 2026'),
                tweet('900000000000000003','Fri Oct 23 08:00:00 +0000 2026'),
            ]).encode(),'utf-8'
    client=TwitterTimelineClient(reader=Reader(),clock=lambda:datetime(2026,10,16,tzinfo=UTC))
    result=client.list_posts('@synthetic',limit=6,lookback_days=7)
    assert len(seen)==1 and 'screen-name/synthetic' in seen[0]
    assert [p['post_id'] for p in result['posts']]==['900000000000000002']
    assert result['latest_proven'] is False and result['complete'] is False
    assert result['excluded_old']==1 and result['excluded_invalid_date']==1


def test_empty_recent_window_reports_gap_not_no_posts():
    class Reader:
        def fetch(self,url,**kwargs):
            return url,embedded([tweet('900000000000000001','Mon Jan 01 00:00:00 +0000 2024')]).encode(),'utf-8'
    result=TwitterTimelineClient(reader=Reader(),clock=lambda:datetime(2026,10,16,tzinfo=UTC)).list_posts('synthetic',lookback_days=7)
    assert result['posts']==[] and result['status']=='partial'
    assert result['gap']=='no_verified_posts_in_requested_window'
    assert result['newest_visible_at'].startswith('2024-01-01')


def test_guard_revocation_prevents_returning_candidates():
    def check():raise RuntimeError('revoked')
    with pytest.raises(RuntimeError,match='revoked'):
        TwitterTimelineClient().list_posts('synthetic',check=check)
