import json

import pytest

from app.agent.context_recovery import (
    application_turn_marker, safe_closed_items, should_rotate_context,
)
from app.agent.context_policy import RequestContextPolicy


@pytest.mark.parametrize('protocol,image_type,text_type,key', [
    ('openai-responses', 'input_image', 'input_text', 'input'),
    ('openai-completions', 'image_url', 'text', 'messages'),
])
def test_old_pixels_leave_request_but_current_and_tool_pixels_survive(protocol,image_type,text_type,key):
    body={key:[{'role':'user','content':[{'type':text_type,'text':'old topic'},
        {'type':image_type,'image_url':'OLD_PIXELS'}]},
        {'role':'assistant','content':'old successful description'},
        {'role':'user','content':[{'type':text_type,'text':application_turn_marker('fresh')+'this picture'},
        {'type':image_type,'image_url':'CURRENT_PIXELS'}]},
        {'role':'user','content':[{'type':image_type,'image_url':'CURRENT_TOOL_PIXELS'}]}]}
    projected,metrics=RequestContextPolicy().project(body,protocol=protocol,current_call_ids=set(),current_turn_id='fresh')
    rendered=json.dumps(projected)
    assert 'OLD_PIXELS' not in rendered
    assert 'CURRENT_PIXELS' in rendered and 'CURRENT_TOOL_PIXELS' in rendered
    assert 'old successful description' in rendered
    assert 'OLD_PIXELS' in json.dumps(body)
    assert metrics['historical_media_removed']==1


def test_compaction_never_reintroduces_image_pixels():
    body={'input':[{'role':'user','content':[{'type':'input_text','text':application_turn_marker('fresh')},
        {'type':'input_image','image_url':'CURRENT_PIXELS'}]}]}
    projected,_=RequestContextPolicy().project(body,protocol='openai-responses',current_call_ids=set(),
        current_turn_id='fresh',include_current_media=False)
    assert 'CURRENT_PIXELS' not in json.dumps(projected)


def test_rejected_input_and_unknown_turns_do_not_bootstrap_into_new_session():
    items=[{'role':'user','content':'normal request'},{'role':'assistant','content':'normal reply','status':'sent'},
        {'role':'user','content':'REJECTED_PRIVATE_INPUT'},
        {'role':'assistant','content':json.dumps({'status':'failed','trace':[]})},
        {'role':'user','content':'UNKNOWN_WRITE_INPUT'},{'role':'assistant','content':'','status':'uncertain'}]
    assert safe_closed_items(items)==items[:2]
    assert items[2]['content']=='REJECTED_PRIVATE_INPUT'
    assert should_rotate_context({'context_reusable':False,'receipts':[{'status':'uncertain'}]})
    assert should_rotate_context({'last_status':'failed'})
    assert not should_rotate_context({'last_status':'sent'})
