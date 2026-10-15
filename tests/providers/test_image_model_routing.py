"""One shared transport with explicit generation/edit models and real references."""
import base64
import json

import httpx
import pytest

from app.config import AppSettings
from app.core.message_content import ImageAttachment
from app.providers.llm_client import LlmClient
from app.runtime.providers import build_group_image_llm_client


@pytest.mark.parametrize('edit,edit_model,expected', [
    (False,'gpt-image-2.5-sunburst','gpt-image-2.5-flare'),
    (True,'gpt-image-2.5-sunburst','gpt-image-2.5-sunburst'),
    (True,'','gpt-image-2.5-flare'),
])
def test_selected_image_model_is_operation_specific_without_losing_reference_authority(tmp_path,edit,edit_model,expected):
    requests=[]
    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json={'id':'synthetic-response','output':[{
            'type':'image_generation_call','id':'synthetic-image','result':base64.b64encode(b'synthetic-result').decode()}]})
    http=httpx.Client(transport=httpx.MockTransport(handle))
    client=LlmClient(base_url='https://synthetic.invalid/v1',api_key='synthetic',model='gpt-6.1-sol',
        image_responses_model='gpt-6.1-sol',image_generation_model='gpt-image-2.5-flare',
        image_edit_model=edit_model,http_client=http)
    reference=tmp_path/'synthetic.png';reference.write_bytes(b'synthetic-reference-bytes')
    options={'prompt':'Keep the synthetic subject and change only its color.','model':'gpt-image-2.5-flare','max_attempts':1}
    try:
        if edit:result=client.edit_image(**options,images=[ImageAttachment(url='',local_path=str(reference),reference_subject='synthetic subject')])
        else:result=client.generate_image(**options)
    finally:http.close()
    assert len(requests)==1 and requests[0]['model']=='gpt-6.1-sol'
    assert requests[0]['tools']==[{'type':'image_generation','model':expected}]
    assert client.resolved_image_model(edit=edit)==expected
    content=requests[0]['input'][0]['content']
    if edit:
        assert content[0]['type']=='input_text' and 'synthetic subject' in content[0]['text']
        assert content[1]['type']=='input_image'
        assert base64.b64decode(content[1]['image_url'].split(',',1)[1])==reference.read_bytes()
        assert content[-1]['text']==options['prompt']
    else:assert not any(p['type']=='input_image' for p in content)
    assert result.artifacts[0].b64_json==base64.b64encode(b'synthetic-result').decode()


def test_shared_group_private_image_factory_uses_chat_credentials_and_both_configured_models():
    settings=AppSettings(_env_file=None,BOT_QQ=900000101,OWNER_QQ=900000102,
        NAPCAT_WS_URL='ws://synthetic.invalid/ws',
        LLM_BASE_URL='https://synthetic.invalid/v1',LLM_API_KEY='synthetic-chat-key',
        GROUP_IMAGE_TRANSPORT='responses',GROUP_IMAGE_CHAT_BASE_URL='',GROUP_IMAGE_CHAT_API_KEY='',
        GROUP_IMAGE_MODEL='gpt-image-2.5-flare',GROUP_IMAGE_EDIT_MODEL='gpt-image-2.5-sunburst',
        GROUP_IMAGE_RESPONSES_MODEL='gpt-6.1-sol')
    primary=LlmClient(base_url=settings.llm_base_url,api_key=settings.llm_api_key,model='gpt-6.1-sol')
    try:
        image=build_group_image_llm_client(settings=settings,engine=None,llm_client=primary)
        assert image.base_url==primary.base_url and image.api_key==primary.api_key
        assert image.http_client is primary.http_client
        assert image.image_generation_model=='gpt-image-2.5-flare'
        assert image.image_edit_model=='gpt-image-2.5-sunburst'
    finally:primary.http_client.close()


def test_edit_failure_keeps_selected_model_and_never_drops_the_reference(tmp_path):
    requests=[]
    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(503,json={'error':{'message':'synthetic failure'}})
    http=httpx.Client(transport=httpx.MockTransport(handle))
    client=LlmClient(base_url='https://synthetic.invalid/v1',api_key='synthetic',model='gpt-6.1-sol',
        image_responses_model='gpt-6.1-sol',image_generation_model='gpt-image-2.5-flare',
        image_edit_model='gpt-image-2.5-sunburst',http_client=http)
    source=tmp_path/'reference.png';source.write_bytes(b'synthetic-reference')
    try:
        with pytest.raises(ValueError,match='responses image request failed after retries'):
            client.edit_image(prompt='Change the color.',model='gpt-image-2.5-flare',images=[ImageAttachment(url='',local_path=str(source))],max_attempts=1)
    finally:http.close()
    assert len(requests)==1 and requests[0]['tools'][0]['model']=='gpt-image-2.5-sunburst'
    assert any(p['type']=='input_image' for p in requests[0]['input'][0]['content'])


def test_edit_model_requires_explicit_standard_responses_transport():
    with pytest.raises(ValueError,match='requires explicit Responses'):
        LlmClient(base_url='https://synthetic.invalid/v1',api_key='synthetic',model='gpt-image-2.5-flare',
            image_responses_model='gpt-image-2.5-flare',image_edit_model='gpt-image-2.5-sunburst')
