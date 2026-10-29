import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from comfyrouter.agentic import AgentOutputError
from comfyrouter.rag_capability_analysis import analyze_capabilities
from comfyrouter.whole_request_selection import select_whole_request
from tools import run_anonymous_md_comparison as routing


def test_analysis_does_not_see_catalog_or_create_subtasks():
    client = Mock()
    client.complete_json.return_value = {'analysis': 'Generate an image from text, then upscale 2x.'}
    task = {'instruction': 'Generate a dog image and upscale 2x.', 'resources': [], 'input_types': []}
    assert '2x' in analyze_capabilities(client, task)
    _, contract, payload = client.complete_json.call_args.args
    assert 'candidates' not in payload and 'steps' not in payload
    assert payload['instruction'] == task['instruction']
    assert client.complete_json.call_count == 1


@pytest.mark.parametrize('blocked', [False, True])
def test_single_retrieval_joint_selection_and_honest_failure(tmp_path, monkeypatch, blocked):
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    write(tmp_path/'protocol.json', {'rag_profile': 'capability_single_pass', 'rag_top_k': 20})
    task = {'instruction': 'Generate a dog image.', 'modality': 't2i', 'resources': [], 'input_types': []}
    write(tmp_path/'task_inputs.json', {'001_v0': task})
    docs = [{'workflow_id': str(i), 'description': 'Generate an image.'} for i in range(25)]
    write(tmp_path/'workflow_catalog.json', docs)
    np.save(tmp_path/'workflow_vectors.npy', np.tile([[1., 0.]], (25, 1)))
    client = Mock()
    client.complete_json.return_value = {'analysis': 'Text-to-image generation, no input image.'}
    monkeypatch.setattr(routing, 'configure', lambda run: None)
    monkeypatch.setattr(routing, 'client_type', lambda: lambda: client)
    encode = Mock(return_value=np.array([[1., 0.]], dtype=np.float32))
    monkeypatch.setattr('comfyrouter.semantic_retrieval.E5Encoder',
                        lambda path: SimpleNamespace(encode=encode))
    selected = Mock()
    if blocked:
        selected.side_effect = AgentOutputError('selection blocked: required input missing')
    else:
        selected.return_value = (SimpleNamespace(), SimpleNamespace(), [], {'workflows': [], 'reason': 'fixture'})
    monkeypatch.setattr('comfyrouter.whole_request_selection.select_whole_request', selected)
    routing.route_worker(tmp_path, 'semantic', '001_v0')
    result = json.loads((tmp_path/'routing/001_v0/route.json').read_text())['methods']['semantic']
    assert client.complete_json.call_count == encode.call_count == selected.call_count == 1
    assert 'Analysis: Text-to-image' in encode.call_args.args[0][0]
    assert len(selected.call_args.args[1]) == 20
    assert selected.call_args.kwargs['retrieval_feedback'] is False
    assert selected.call_args.kwargs['prompt_profile'] == 'rag_single_pass'
    assert result['retrieval_evidence']['query_count'] == 1
    assert result['status'] == ('routing_failed' if blocked else 'routed')
    if not blocked:
        assert result['iteration_budget']['execution_attempts_remaining'] == 3
    routing.route_worker(tmp_path, 'semantic', '001_v0')
    assert selected.call_count == 1  # completed route, including failure, is final


def test_empty_joint_selection_is_a_failure_without_fallback():
    client = Mock()
    client.complete_json.return_value = {'workflows': [], 'reason': 'No real mask can be produced.'}
    with pytest.raises(AgentOutputError, match='selection blocked'):
        select_whole_request(client, [{'workflow_id': 'A', 'description': 'Masked edit.'}],
                             'Edit an image.', 'i2i', ['image.png'], ['image'], 'semantic',
                             anonymous_descriptions=True, prompt_profile='rag_single_pass')
    assert client.complete_json.call_count == 1
