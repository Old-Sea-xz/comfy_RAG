import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from comfyrouter import case_execution as runner
from comfyrouter.agentic import AgentOutputError
from comfyrouter.execution_contract import CaseExecutionRequest, ExecutionHooks, execution_context
from comfyrouter.quality_policy import POLICY_ID


def test_tolerant_temporal_preflight_accepts_boundary_but_rejects_large_loss(monkeypatch):
    graph = {'1': {'class_type': 'EmptyHunyuanVideo15Latent', 'inputs': {'length': 49}},
             '2': {'class_type': 'VAEDecode', 'inputs': {'samples': ['1', 0]}},
             '3': {'class_type': 'VHS_VideoCombine', 'inputs': {'images': ['2', 0], 'frame_rate': 24}}}
    # Exact planner-added counts formerly blocked a visually valid 2.04-second clip.
    monkeypatch.delenv('COMFYROUTER_QUALITY_POLICY', raising=False)
    with pytest.raises(AgentOutputError):
        runner.validate_temporal_parameters(graph, 'Generate exactly 48 frames at 24 fps for 2 seconds')
    monkeypatch.setenv('COMFYROUTER_QUALITY_POLICY', POLICY_ID)
    assert runner.validate_temporal_parameters(graph, 'Generate exactly 48 frames at 24 fps for 2 seconds')
    graph['1']['inputs']['length'] = 25
    with pytest.raises(AgentOutputError):
        runner.validate_temporal_parameters(graph, 'Generate exactly 48 frames at 24 fps for 2 seconds')


@pytest.mark.parametrize('composed', [False, True])
def test_rubric_regeneration_submits_next_round_without_reset_or_old_repair(tmp_path, monkeypatch, composed):
    monkeypatch.setattr(runner, 'write_operation_notebook', lambda *a: None)
    monkeypatch.setattr(runner, 'composition_handoff_inputs', lambda *a: ({}, []))
    out = tmp_path/'raw'/'case001'
    out.mkdir(parents=True)
    audit = tmp_path/'audit'
    audit.mkdir()
    spec = {'modality': 't2i', 'task_family': 'Generation', 'operations': ['Generation/text_to_image']}
    graph = {'1': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'same'}}}
    artifact = out/'first.png'
    artifact.write_bytes(b'original')
    prior = {'attempt': 1, 'passed': False, 'graph': graph, 'primary_output': str(artifact),
             'outputs': [str(artifact)], 'execution': {'status': 'success'},
             'comfybench': {'comfybench': {'resolved': False}}}
    r = {'case_id': '001', 'instruction': 'generate', 'max_attempts': 3, 'attempts': [prior],
         'trace': {'case_output': str(out), 'agent_audit': str(audit)}, 'case_resources': [],
         'resource_policy': {'vram_limit_bytes': 123}, 'requirement_spec': spec,
         'implementation_plan': [{'operation': 'Generation/text_to_image'}], 'base_graph': graph}
    if composed:
        stage = {'stage_id': 's1', 'instruction': 'generate', 'purpose': 'generate',
                 'input_type': 'TEXT', 'output_type': 'IMAGE', 'modality': 't2i', 'depends_on': []}
        r['composition'] = {'status': 'ready', 'stages': [{'stage': stage, 'requirement_spec': spec,
                            'implementation': {'operation': 'Generation/text_to_image'}, 'preflight_graph': graph}]}
        r['composition']['stages'].append({**r['composition']['stages'][0], 'stage': {**stage, 'stage_id': 's2'}})
        prior['stages'] = [{**prior, 'stage_id': sid} for sid in ['s1', 's2']]
    report_path = tmp_path/'report.json'
    report_path.write_text(json.dumps(r))
    invoked = Mock()
    monkeypatch.setattr(runner, 'run_composition' if composed else 'run_single_case', invoked)
    req = CaseExecutionRequest('001', {'instruction': 'generate', 'modality': 't2i'}, tmp_path/'raw',
                              resume_report=report_path, regenerate_on_resume=True)
    with execution_context(req, hooks=ExecutionHooks(agent_factory=lambda _: Mock(), executor_factory=lambda: Mock())):
        resumed = runner.resume_evaluated_case(req)
    assert invoked.call_args.kwargs['resume_after_evaluation'] is False
    if composed:
        assert invoked.call_args.kwargs['start_attempt'] == 2
        assert invoked.call_args.kwargs['initial_cache'] == {}
    assert resumed['attempts'] == [prior] and resumed['max_attempts'] == 3
    assert artifact.read_bytes() == b'original'
    assert resumed['quality_resume_history'][-1]['regenerate_on_resume'] is True


def test_regeneration_cannot_be_used_without_valid_resume(tmp_path):
    with pytest.raises(ValueError, match='regeneration requires'):
        CaseExecutionRequest('001', {'instruction': 'generate', 'modality': 't2i'}, tmp_path,
                             regenerate_on_resume=True)
