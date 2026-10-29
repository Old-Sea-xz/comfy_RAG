import copy
import json
import httpx
import pytest
from comfyrouter.agentic import AgentOutputError, AgentProviderError, OpenAICompatibleClient
from comfyrouter.prompt_evidence import compact_prompt_history
from comfyrouter.result_protocol import RESULT_TOOL, parse_result_call, result_tool, strict_compatible

def response(arguments='{"status":"ok"}', **overrides):
    call = {'type': 'function_call', 'name': RESULT_TOOL, 'call_id': 'call_1', 'arguments': arguments, 'status': 'completed', **overrides}
    return {'output': [{'type': 'reasoning', 'id': 'rs_1', 'summary': []}, call]}

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr('comfyrouter.agentic.config.load_env', lambda: None)
    monkeypatch.setenv('COMFYROUTER_LLM_RESULT_PROTOCOL', 'function_calling')
    monkeypatch.setenv('COMFYROUTER_LLM_CLASSIFICATION_MODEL', 'test-model')
    monkeypatch.setenv('COMFYROUTER_SERIAL_API', '0')
    monkeypatch.setenv('COMFYROUTER_LLM_STREAM', '0')
    c = OpenAICompatibleClient('https://example.invalid/v1', 'test-key', 'test-model')
    yield c
    c.http_client.close()

def wire(client, monkeypatch, results):
    bodies = []
    responses = iter(results)

    def send(**kwargs):
        bodies.append(json.loads(kwargs['body']))
        return httpx.Response(200, request=httpx.Request('POST', 'https://example.invalid'), json=next(responses))
    monkeypatch.setattr(client, '_send_httpx', send)
    return bodies

def test_native_request_and_multi_turn_call_outputs(client, monkeypatch):
    bodies = wire(client, monkeypatch, [response(), response(call_id='call_2')])
    schema = {'name': 'test', 'schema': {'type': 'object', 'additionalProperties': False, 'properties': {'status': {'type': 'string'}}, 'required': ['status']}}
    contract = {'rules': ['Keep exact node IDs.'], 'required_output': ['status']}
    payload = {'case_id': 'one', 'request': 'first', '_response_json_schema': schema}
    assert client.complete_json('generation_agent', contract, payload) == {'status': 'ok'}
    assert client.complete_json('generation_agent', contract, {**payload, 'request': 'second'}) == {'status': 'ok'}
    first, second = bodies
    assert first['tools'][0]['parameters'] == schema['schema']
    assert first['tools'][0]['strict'] is True
    assert first['tool_choice'] == {'type': 'function', 'name': RESULT_TOOL}
    assert first['parallel_tool_calls'] is False and 'text' not in first
    assert first['input'][0] == second['input'][0]
    assert [item.get('type') for item in second['input'][2:5]] == ['reasoning', 'function_call', 'function_call_output']
    assert second['input'][3]['call_id'] == second['input'][4]['call_id'] == 'call_1'
    assert second['input'][4]['output'] == '{"accepted":true}'
    assert 'stable_cache_anchor' not in json.dumps(first)
    assert json.loads(first['input'][0]['content'][0]['text'])['required_output'] == []

@pytest.mark.parametrize('result', [{'output_text': '{"status":"ok"}'}, response(name='other_tool'), response(arguments='[]'), response(arguments='{broken'), response(call_id=''), response(status='in_progress'), {'output': response()['output'] + response(call_id='call_2')['output']}, {'output': [{'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'No'}]}]}])
def test_rejects_text_wrong_multiple_and_malformed_calls(result):
    with pytest.raises(ValueError):
        parse_result_call(result)

def test_protocol_failure_retries_keep_native_choice_and_final_evidence(client, monkeypatch):
    bad = response(arguments='{broken')
    bodies = wire(client, monkeypatch, [bad, bad, bad])
    with pytest.raises(AgentOutputError) as caught:
        client.complete_json('generation_agent', {}, {'request': 'edit'})
    assert len(bodies) == 3
    assert all((body['tool_choice']['name'] == RESULT_TOOL for body in bodies))
    assert '{broken' in caught.value.raw_response
    assert 'no Markdown, prose, or tool calls' not in json.dumps(bodies[-1])

def test_incomplete_provider_result_is_not_accepted(client, monkeypatch):
    bodies = wire(client, monkeypatch, [{**response(), 'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'}}])
    with pytest.raises(AgentProviderError):
        client.complete_json('generation_agent', {}, {})
    assert len(bodies) == 1

def test_non_strict_legacy_contract_and_nullable_objects_do_not_change_shape():
    assert result_tool(None)['strict'] is False
    schema = {'schema': {'type': 'object', 'properties': {'optional': {'type': 'string'}}, 'required': [], 'additionalProperties': False}}
    original = copy.deepcopy(schema)
    assert result_tool(schema)['strict'] is False
    assert schema == original
    assert strict_compatible({'type': ['object', 'null'], 'properties': {'x': {'type': 'string'}}, 'required': [], 'additionalProperties': False}) is False

def test_sse_terminal_response_keeps_real_function_calls():
    result = response()
    stream = httpx.Response(200, text='data: ' + json.dumps({'type': 'response.completed', 'response': result}) + '\n\n')
    parsed = OpenAICompatibleClient._read_httpx_sse_completed_response(stream)
    assert parse_result_call(parsed)[0] == {'status': 'ok'}

def test_history_eviction_drops_whole_native_turn():
    old = [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'x' * 2000}]}]
    old.extend(parse_result_call(response())[2])
    current = [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'current'}]}]
    current.extend(parse_result_call(response(call_id='call_2'))[2])
    result = compact_prompt_history(old + current, limit=1000)
    calls = [item['call_id'] for item in result if item.get('type') == 'function_call']
    outputs = [item['call_id'] for item in result if item.get('type') == 'function_call_output']
    assert calls == outputs == ['call_2']
    assert result[-len(current):] == current
