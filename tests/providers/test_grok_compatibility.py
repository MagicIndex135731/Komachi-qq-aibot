"""Provider switch contracts: source roles, tool identity and independent search."""
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from app.providers.llm_client import LlmClient, default_auxiliary_reasoning_effort
from app.runtime.providers import build_web_search_client


def sse(events):
    return ''.join('data: '+json.dumps(event)+'\n\n' for event in events)


@pytest.mark.parametrize('model,expected_role', [('grok-4.7','system'),('gpt-6.1-sol','developer')])
def test_transport_preserves_runtime_instruction_authority_without_rewriting_stored_history(model,expected_role):
    requests=[]
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json={'output':[]})
    http=httpx.Client(transport=httpx.MockTransport(handler))
    client=LlmClient(base_url='https://synthetic.invalid/v1',api_key='synthetic',model=model,
                     responses_model=model,http_client=http)
    body={'model':model,'input':[{'role':'developer','content':[{'type':'input_text','text':'runtime rule'}]},
                                {'role':'user','content':'synthetic question'}]}
    original=copy.deepcopy(body)
    try:client._read_responses_text_response(payload=body)
    finally:http.close()
    if model.startswith('grok-'):
        assert requests[0]['instructions']=='runtime rule'
        assert requests[0]['input']==[original['input'][1]]
    else:
        assert requests[0]['input'][0]['role']==expected_role
        assert requests[0]['input'][1]==original['input'][1]
    assert body==original


@pytest.mark.parametrize('arguments_before_item', [False, True])
def test_distinct_response_item_and_tool_call_ids_execute_once_and_continue_with_canonical_id(arguments_before_item):
    requests=[];executed=[]
    item={'type':'function_call','id':'fc_synthetic_1','call_id':'call_synthetic_1','name':'read_probe','arguments':'{"key":"alpha"}'}
    added={'type':'response.output_item.added','output_index':0,'item':{**item,'arguments':'','status':'in_progress'}}
    arguments={'type':'response.function_call_arguments.done','item_id':'fc_synthetic_1','name':'read_probe','arguments':item['arguments']}
    events=([arguments,added] if arguments_before_item else [added,arguments])+[
        {'type':'response.output_item.done','output_index':0,'item':item},
        {'type':'response.completed','response':{'id':'response_synthetic_1','output':[item]}}]
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,text=sse(events) if len(requests)==1 else sse([
            {'type':'response.completed','response':{'id':'response_synthetic_2','output':[{
                'type':'message','role':'assistant','content':[{'type':'output_text','text':'value=7691'}]}]}}]),
            headers={'content-type':'text/event-stream'})
    http=httpx.Client(transport=httpx.MockTransport(handler))
    client=LlmClient(base_url='https://synthetic.invalid/v1',api_key='synthetic',model='grok-4.7',
                     responses_model='grok-4.7',responses_only=True,http_client=http,request_max_attempts=1)
    def execute(name,args):
        executed.append((name,args));return '{"value":7691}'
    try:
        text=client.generate_text_with_tools(['Target message: read alpha'],tools=[{
             'type':'function','name':'read_probe','parameters':{'type':'object','properties':{'key':{'type':'string'}},'required':['key']}}],
             tool_executor=execute,max_tool_rounds=2)
    finally:http.close()
    assert text=='value=7691'
    assert executed==[('read_probe',{'key':'alpha'})]
    assert [x['call_id'] for x in requests[1]['input'] if x.get('type')=='function_call_output']==['call_synthetic_1']


def test_wyzlab_search_can_keep_its_verified_model_and_key_independent_of_grok_chat():
    settings=SimpleNamespace(llm_base_url='https://shell.wyzlab.ai/v1',llm_text_endpoint='responses',
        llm_api_key='synthetic-grok-key',llm_model='grok-4.7',llm_web_search_model='gpt-6.1-sol',
        llm_web_search_api_key='synthetic-search-key',agent_web_search_timeout_seconds=45,web_search_reasoning_effort='low')
    client=build_web_search_client(settings)
    try:
        assert client.provider=='wyzlab_alpha'
        assert client.model=='gpt-6.1-sol'
        assert client.api_key=='synthetic-search-key'
    finally:client.http_client.close()


def test_grok_auxiliary_jobs_use_low_without_changing_other_families():
    assert default_auxiliary_reasoning_effort('grok-4.7')=='low'
    assert default_auxiliary_reasoning_effort('gpt-6.1-sol')=='low'
    assert default_auxiliary_reasoning_effort('offline-agent')=='medium'


def test_switching_grok_models_separates_native_reasoning_replay_from_other_models():
    from app.agent.harness_runtime import CONTEXT_FORMAT_VERSION, model_context_format_version
    def client(model):return SimpleNamespace(responses_model=model,model=model)
    assert model_context_format_version(client('gpt-6.1-sol'))==CONTEXT_FORMAT_VERSION
    assert model_context_format_version(client('grok-4.7'))!=CONTEXT_FORMAT_VERSION
    assert model_context_format_version(client('grok-4.7'))!=model_context_format_version(client('grok-4.6'))
    assert model_context_format_version(object())==CONTEXT_FORMAT_VERSION


def test_grok_overlay_is_accepted_by_the_actual_settings_contract():
    from pathlib import Path
    from app.config import AppSettings
    settings=AppSettings(_env_file=Path('infra/wsl/model-profiles/grok-4.7.env.example'),
        BOT_QQ=900000201,OWNER_QQ=900000202,LLM_API_KEY='synthetic',
        LLM_BASE_URL='https://synthetic.invalid/v1',NAPCAT_WS_URL='ws://synthetic.invalid/ws')
    assert settings.llm_model==settings.llm_vision_model=='grok-4.7'
    assert settings.llm_fallback_model==''
    assert settings.llm_web_search_model=='gpt-6.1-sol'
    assert settings.memory_episode_topic_judge_max_output_tokens==512
    assert settings.memory_episode_post_segment_max_output_tokens==2048


@pytest.mark.parametrize('model', ['grok-4.7','gpt-6.1-sol'])
def test_native_large_integer_schema_bounds_are_transport_only_and_do_not_relax_execution(model):
    from app.agent.tool_agent import validate_arguments
    schema={'type':'object','properties':{'source_ids':{'type':'array','items':{
        'type':'integer','minimum':1,'maximum':9223372036854776000}},'count':{
        'type':'integer','minimum':0,'maximum':24},'amount':{'type':'number','maximum':1e20},
        'negative':{'type':'integer','minimum':-9223372036854776000,'maximum':-1}},
        'required':['source_ids','count','amount','negative'],'additionalProperties':False}
    body={'model':model,'instructions':'synthetic existing instructions','input':[{'role':'user','content':'synthetic'}],
          'tools':[{'type':'function','name':'read_probe','parameters':schema}]}
    original=copy.deepcopy(body)
    client=LlmClient(base_url='https://synthetic.invalid/v1',api_key='synthetic',model=model,responses_model=model)
    try:prepared=client.prepare_responses_payload(body)
    finally:client.http_client.close()
    outgoing=prepared['tools'][0]['parameters']
    assert body==original and schema==original['tools'][0]['parameters']
    if model.startswith('grok-'):
        assert outgoing['properties']['source_ids']['items']=={'type':'integer','minimum':1}
        assert outgoing['properties']['negative']=={'type':'integer','maximum':-1}
        assert outgoing['properties']['count']==schema['properties']['count']
        assert outgoing['properties']['amount']==schema['properties']['amount']
    else:assert prepared is body
    assert not validate_arguments({'source_ids':[0],'count':1,'amount':2,'negative':-1},schema)
    assert not validate_arguments({'source_ids':[9223372036854776001],'count':1,'amount':2,'negative':-1},schema)
