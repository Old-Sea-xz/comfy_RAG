"""Check standalone preparation and deployment boundaries without live services."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.prepare_run import prepare
from tools.run_routing_comparison import verify_frozen

ROOT = Path(__file__).resolve().parents[1]


def inputs(tmp_path, tasks=None):
    tasks = tasks or {'test_001': {'instruction': 'Generate a watercolor sailboat.',
                                 'modality': 't2i', 'category': 'vanilla', 'resources': []}}
    task_file = tmp_path/'tasks.json'
    task_file.write_text(json.dumps(tasks))
    resources = tmp_path/'resources'
    resources.mkdir()
    model = tmp_path/'encoder'
    model.mkdir()
    registry = tmp_path/'registry.json'
    registry.write_text(json.dumps({'SaveImage': {'input': {'required': {}}}}))
    return task_file, resources, model, registry


@pytest.mark.parametrize('attempts', [1, 3])
def test_fresh_run_is_independent_and_hashes_its_exact_graphs(tmp_path, attempts):
    tasks, resources, model, registry = inputs(tmp_path)
    run = prepare(tasks,resources,ROOT/'library',model,tmp_path/'run',registry=registry,
                  max_attempts=attempts)
    protocol = json.loads((run/'protocol.json').read_text())
    assert protocol['max_attempts'] == attempts
    assert protocol['rag_query_limit'] == 3 and protocol['rag_top_k'] == 20
    assert protocol['serial_api'] and protocol['stage_fifo_queue']
    assert protocol['native_queue_window'] == 3
    assert not list((run/'execution').rglob('outcome.json'))
    assert not list((run/'routing').rglob('route.json'))
    verify_frozen(run)
    catalog = json.loads((run/'workflow_catalog.json').read_text())
    assert len(catalog) == 66
    assert all(Path(row['workflow_path']).is_relative_to(run/'frozen') for row in catalog)
    graph = Path(catalog[0]['workflow_path'])
    graph.write_text('{}')
    with pytest.raises(RuntimeError, match='frozen library changed'):
        verify_frozen(run)


def test_preparation_cannot_overwrite_a_run(tmp_path):
    tasks, resources, model, registry = inputs(tmp_path)
    run = tmp_path/'run'
    run.mkdir()
    marker = run/'existing.json'
    marker.write_text('keep')
    with pytest.raises(ValueError, match='destination exists'):
        prepare(tasks,resources,ROOT/'library',model,run,registry=registry)
    assert marker.read_text() == 'keep'


def test_task_cannot_read_resources_outside_its_declared_directory(tmp_path):
    tasks, resources, model, registry = inputs(tmp_path, {
        'test_001': {'instruction': 'Edit this image.', 'modality': 'i2i',
                     'resources': ['../private.png']}})
    (tmp_path/'private.png').write_bytes(b'not published')
    with pytest.raises(ValueError, match='outside resource directory'):
        prepare(tasks,resources,ROOT/'library',model,tmp_path/'run',registry=registry)
    assert not (tmp_path/'run').exists()


def test_published_descriptions_and_dense_vectors_are_aligned():
    library = ROOT/'library'
    manifest = json.loads((library/'embedding_manifest.json').read_text())
    catalog = json.loads((library/'catalog.json').read_text())
    assert [row['workflow_id'] for row in catalog] == manifest['workflow_ids']
    assert hashlib.sha256((library/'workflow_vectors.npy').read_bytes()).hexdigest() == manifest['vectors_sha256']
    vectors = np.load(library/'workflow_vectors.npy', allow_pickle=False)
    assert vectors.shape == (66, 384) and np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
    for row, expected in zip(catalog, manifest['description_sha256'], strict=True):
        assert hashlib.sha256(row['description'].encode()).hexdigest() == expected
        assert hashlib.sha256((library/row['workflow_path']).read_bytes()).hexdigest() == row['sha256']


def test_executor_uses_configured_comfyui_paths(tmp_path, monkeypatch):
    from comfyrouter import config
    from comfyrouter.closed_loop import ComfyUIExecutor
    monkeypatch.setattr(config,'COMFYUI_ROOT',tmp_path)
    monkeypatch.setenv('COMFYROUTER_COMFYUI_API','http://localhost:18188')
    executor = ComfyUIExecutor()
    assert executor.output_root == tmp_path/'output'
    assert executor.comfyui_input_root == tmp_path/'input'
    assert executor.api == 'http://localhost:18188'


def test_rag_release_excludes_tree_and_decomposition_implementations():
    for name in ('router.py', 'jev.py', 'taxonomy_builder.py', 'classification_prompt.py'):
        assert not (ROOT/'comfyrouter'/name).exists()
    assert not (ROOT/'instructions/DECOMPOSITION_GATE_PROMPT.md').exists()
    from comfyrouter.agentic import AgentOrchestrator
    for name in ('decide_decomposition', 'classify', 'classify_models', 'bootstrap_taxonomy'):
        assert not hasattr(AgentOrchestrator, name)


def test_rag_generation_initializes_and_edits_selected_graph(tmp_path):
    from unittest.mock import Mock
    from comfyrouter.agentic import AgentOrchestrator
    from comfyrouter.types import RequirementSpec
    client = Mock()
    client.complete_json.return_value = {
        'skill_ids': [], 'edits': [{'action': 'set_param', 'target': '1',
                                  'key': 'text', 'value': 'watercolor dog'}],
        'rationale': 'Use the selected workflow.',
    }
    agent = AgentOrchestrator(client, max_corrections=0)
    plan = agent.generate_graph_edit_plan(
        RequirementSpec('t2i', 'anonymous', ('W1',)),
        [{'operation': 'W1', 'selected_profile': 'W1', 'base_path': str(tmp_path)}],
        graph={'1': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'dog'}}},
        instruction='Generate a watercolor dog.')
    assert plan['edits'][0]['value'] == 'watercolor dog'
    assert client.complete_json.call_count == 1
