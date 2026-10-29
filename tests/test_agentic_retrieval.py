import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from comfyrouter.agentic import AgentOutputError
from comfyrouter.agentic_retrieval import RepairRetriever
from tools import run_anonymous_md_comparison as routing


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('searches,blocked', [(1, False), (2, False), (3, False), (3, True)])
def test_agent_controls_followup_queries_and_stops_at_limit(tmp_path, monkeypatch, searches, blocked):
    write(tmp_path/'protocol.json', {'rag_profile': 'agentic_feedback', 'rag_top_k': 2, 'rag_query_limit': 3})
    task = {'instruction': 'original request', 'modality': 't2i', 'resources': [], 'input_types': []}
    write(tmp_path/'task_inputs.json', {'001_v0': task})
    write(tmp_path/'workflow_catalog.json', [
        {'workflow_id': 'A', 'description': 'first'}, {'workflow_id': 'B', 'description': 'second'}])
    np.save(tmp_path/'workflow_vectors.npy', np.eye(2))
    encode = Mock(return_value=np.array([[1., 0.]]))
    monkeypatch.setattr(routing, 'configure', lambda _: None)
    monkeypatch.setattr(routing, 'client_type', lambda: Mock)
    monkeypatch.setattr('comfyrouter.semantic_retrieval.E5Encoder', lambda _: SimpleNamespace(encode=encode))
    answers = [(None, None, None, {'search_needed': True, 'search_query': f'missing {i}',
                                  'workflows': [], 'reason': 'missing capability'})
               for i in range(searches if blocked else searches-1)]
    if not blocked:
        answers.append((SimpleNamespace(), SimpleNamespace(), [],
                        {'search_needed': False, 'search_query': '', 'workflows': [], 'reason': 'sufficient'}))
    select = Mock(side_effect=answers)
    monkeypatch.setattr('comfyrouter.whole_request_selection.select_whole_request', select)
    routing.route_worker(tmp_path, 'semantic', '001_v0')
    result = json.loads((tmp_path/'routing/001_v0/route.json').read_text())['methods']['semantic']
    assert result['status'] == ('routing_failed' if blocked else 'routed')
    assert select.call_count == encode.call_count == searches
    assert encode.call_args_list[0].args[0] == ['original request']
    if searches > 1:
        assert encode.call_args_list[1].args[0] == ['missing 0']
    assert all(c.kwargs['retrieval_feedback'] for c in select.call_args_list)
    assert result['retrieval_evidence']['query_count'] == searches


@pytest.mark.parametrize('required', [True, False])
def test_repair_decision_is_optional_and_checkpointed(tmp_path, required):
    retriever = RepairRetriever(tmp_path, tmp_path/'audit')
    client = Mock()
    client.complete_json.return_value = {'retrieval_required': required,
        'queries': ['mask conditioning'] if required else [], 'reason': 'based on observed failure'}
    retriever.search = Mock(return_value={'query': 'mask conditioning', 'candidates': [{'workflow_id': 'W1'}]})
    kwargs = {'instruction': 'edit', 'graph': {}, 'repair_evidence': {'execution_error': 'mask mismatch'}}
    result = retriever.augment(client, kwargs)
    assert result['repair_evidence']['workflow_rag']['decision']['retrieval_required'] == required
    assert retriever.search.call_count == int(required)
    assert retriever.query_count == int(required)
    assert 'workflow_rag' not in kwargs['repair_evidence']
    retriever.augment(client, kwargs)
    assert client.complete_json.call_count == 1
    assert retriever.search.call_count == int(required)
    # A new failed iteration may independently choose retrieval again.
    retriever.augment(client, {**kwargs, 'repair_evidence': {'quality_error': 'identity changed'}})
    assert client.complete_json.call_count == 2


def test_repair_reference_contains_actual_frozen_graph_and_checks_hash(tmp_path):
    graph = tmp_path/'graph.json'
    write(graph, {'1': {'class_type': 'ImageScale', 'inputs': {}}})
    write(tmp_path/'workflow_catalog.json', [{'workflow_id': 'W1', 'description': 'resize',
        'workflow_path': str(graph), 'sha256': hashlib.sha256(graph.read_bytes()).hexdigest()}])
    write(tmp_path/'protocol.json', {'rag_top_k': 20})
    np.save(tmp_path/'workflow_vectors.npy', np.array([[1., 0.]]))
    retriever = RepairRetriever(tmp_path, tmp_path/'audit')
    retriever.encoder = SimpleNamespace(encode=lambda *a, **k: np.array([[1., 0.]]))
    assert retriever.search('resize')['candidates'][0]['reference_graph']['1']['class_type'] == 'ImageScale'
    graph.write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        retriever.search('resize')


def test_invalid_decision_cannot_trigger_retrieval(tmp_path):
    client = Mock()
    client.complete_json.return_value = {'retrieval_required': False, 'queries': ['unexpected'], 'reason': 'x'}
    retriever = RepairRetriever(tmp_path, tmp_path/'audit')
    retriever.search = Mock()
    with pytest.raises(AgentOutputError):
        retriever.augment(client, {})
    retriever.search.assert_not_called()
