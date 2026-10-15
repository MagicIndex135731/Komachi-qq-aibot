from dataclasses import replace

import pytest

from app.agent.observation import GroupObservationService
from app.core.message_mentions import addressing_evidence
from app.core.search_policy import detect_address_intent
from test_observation import make_router


@pytest.mark.parametrize('text,at_bot,expected', [
    ('你生成一张你的照片证明你比小町身材更好',False,False),
    ('你的照片比小町更好看吗',False,False),
    ('小町你看看这张照片',False,True),
    ('你生成一张你的照片证明你比小町身材更好',True,True),
])
def test_real_other_at_is_not_self_task(sqlite_engine,tmp_path,text,at_bot,expected):
    router,incoming,_=make_router(sqlite_engine,tmp_path)
    bot=router.runtime.settings.bot_qq
    event=replace(incoming,platform_msg_id='addressee-case',plain_text=text,mentioned_bot=at_bot,
        raw_payload={'message':[{'type':'at','data':{'qq':str(bot if at_bot else 900000999)}},
            {'type':'text','data':{'text':text}}]})
    router.ingest_live_group_message(event)
    from app.storage.repositories import MessageRepository
    from app.storage.db import session_scope
    service=GroupObservationService(router)
    with session_scope(sqlite_engine) as session:
        messages=MessageRepository(session)
        assert service._direct(event,messages,service._recent(messages,event.group_id)) is expected


def test_reply_to_other_is_not_a_name_grant_or_recent_followup():
    evidence=addressing_evidence({'message':[]},bot_id=900000001,
        text='你怎么比小町还傻',bot_names={'小町'},reply_author_id=900000002)
    result=detect_address_intent(text='你怎么比小町还傻',bot_names={'小町'},reply_to_bot=False,
        quoted_bot=False,bot_recently_participated=True,recent_bot_message_count=1,
        other_addressee=evidence['other_addressee'])
    assert not result.is_addressed and result.reason=='other_addressee'
