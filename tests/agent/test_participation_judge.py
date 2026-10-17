import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.context_policy import estimate_text_tokens
from app.agent.participation_judge import judge_packet, parse_decision


def row(n, text='普通讨论'):
    return SimpleNamespace(id=n, user_id=900000100+n, plain_text=text, mentioned_bot=False,
        reply_to_msg_id=None, platform_msg_id=str(n), raw_json={})


def test_judge_packet_bounds_long_window_and_keeps_addressing():
    rows=[row(n, '很长的群聊内容'*300) for n in range(1,31)]
    rows[-1].reply_to_msg_id='29'
    rows[-1].raw_json={'message':[{'type':'at','data':{'qq':'900000129'}}]}
    packet=judge_packet(rows,bot_id=900000001)
    assert estimate_text_tokens(json.dumps(packet,ensure_ascii=False))<=1500
    last=packet['messages'][-1]
    assert last['id']==30 and last['reply_to']==29 and last['mentions_other']
    assert not last['mentions_bot'] and packet['messages'][0]['id']>=1


def test_judge_decision_cannot_invent_target_or_execute_instructions():
    assert parse_decision('{"decision":"participate","source_id":12,"reason":"natural_hook"}',{12})==(12,'natural_hook')
    for value in ('yes','{"decision":"participate","source_id":99}',
            '{"decision":"participate","source_id":true}',
            '{"decision":"silent","source_id":12}', 'not JSON'):
        assert parse_decision(value,{12})[0] is None
