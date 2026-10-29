"""Real regressions from multi-reader video handoff and preflight repair."""
import ast
import copy
import json
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from comfyrouter import case_execution as runner
from comfyrouter.agentic import AgentOutputError
from comfyrouter.prompt_interfaces import compact_interface_context


def test_new_video_readers_follow_one_artifact_without_losing_frame_offsets():
    graph = {
        '4': {'class_type': 'LoadVideo', 'inputs': {'file': 'new.mp4'},
              '_meta': {'comfyrouter_handoff': {'media_type': 'VIDEO', 'slot': 'node:4:file'}}},
        '12': {'class_type': 'VHS_LoadVideo', 'inputs': {'video': 'old.mp4', 'skip_first_frames': 0},
               '_meta': {'comfyrouter_handoff': {'media_type': 'VIDEO', 'slot': 'node:12:video'}}},
        '13': {'class_type': 'VHS_LoadVideo', 'inputs': {'video': 'old.mp4', 'skip_first_frames': 5}},
    }
    stage = SimpleNamespace(input_type='VIDEO', input_types=('VIDEO',), modality='v2v')
    by_type, _ = runner.bind_composition_preflight_inputs(graph, stage, [])
    groups = runner.handoff_input_groups(graph, 'VIDEO', by_type['VIDEO'])
    assert len(groups) == 1 and len(groups[0]) == 3
    for node, key in groups[0]:
        graph[node]['inputs'][key] = 'current.mp4'
    assert {graph[node]['inputs'][key] for node, key in by_type['VIDEO']} == {'current.mp4'}
    assert graph['13']['inputs']['skip_first_frames'] == 5


def test_previous_proposed_interface_and_real_video_enum_reach_compacted_request(monkeypatch):
    monkeypatch.delenv('COMFYROUTER_BASE_ONLY_ABLATION', raising=False)
    registry = {
        'LoadVideo': {'input': {'required': {'file': [['other.mp4', 'taylor_swift.mp4']]}}, 'output': ['VIDEO']},
        'ImageScaleBy': {'input': {'required': {'image': ['IMAGE'], 'scale_by': ['FLOAT']}}, 'output': ['IMAGE']},
    }
    executor = SimpleNamespace(ensure_validator=lambda: SimpleNamespace(node_registry=registry))
    agent = SimpleNamespace(generate_graph_edit_plan=Mock(return_value={}))
    evidence = {'preflight_failures': [{'error': 'invalid video enum'}], 'previous_plan': {
        'edits': [{'action': 'add_node', 'node_id': '10', 'node': {'class_type': 'ImageScaleBy', 'inputs': {}}}]}}
    runner.generate_plan_with_repair(agent, executor, None, [], graph={
        '1': {'class_type': 'LoadVideo', 'inputs': {'file': 'VIDEO_INPUT.mp4'}}},
        case_resource_paths=[Path('/media/taylor_swift.mp4')], repair_evidence=evidence,
        allow_topology_repair=True)
    sent = compact_interface_context(agent.generate_graph_edit_plan.call_args.kwargs)
    encoded = json.dumps(sent)
    assert 'ImageScaleBy' in encoded and 'taylor_swift.mp4' in encoded
    interfaces = agent.generate_graph_edit_plan.call_args.kwargs['repair_evidence']['installed_node_interfaces']
    assert {row['class_type'] for row in interfaces} == set(registry)
    assert 'case_resources' not in evidence


def test_interpolation_endpoint_count_rejects_impossible_slice(monkeypatch):
    graph = {'1': {'class_type': 'WanFirstLastFrameToVideo', 'inputs': {'length': 49}},
             '2': {'class_type': 'VAEDecode', 'inputs': {'samples': ['1', 0]}},
             '3': {'class_type': 'PrimitiveInt', 'inputs': {'value': 3}},
             '4': {'class_type': 'FrameInterpolate', 'inputs': {'images': ['2', 0], 'multiplier': ['3', 0]}},
             '5': {'class_type': 'GetImageRangeFromBatch', 'inputs': {'images': ['4', 0], 'start_index': 0, 'num_frames': 147}},
             '6': {'class_type': 'CreateVideo', 'inputs': {'images': ['5', 0], 'fps': 72}}}
    with pytest.raises(AgentOutputError, match='exceeds 145 input frames'):
        runner.validate_temporal_parameters(graph, 'Interpolate 3x')
    graph['5']['inputs'].update(start_index=-1, num_frames=145)
    assert runner.validate_temporal_parameters(graph, 'Interpolate 3x')['outputs'][0]['frames'] == 145


def test_parameter_migration_retains_quality_failure_and_cached_stage(tmp_path):
    video = tmp_path / 'generated.mp4'
    video.write_bytes(b'keep this artifact')
    first = {'attempt': 1, 'passed': False, 'comfybench': {'comfybench': {'resolved': False}}}
    def failed(attempt, message):
        return {'attempt': attempt, 'stages': [
            {'stage_id': 'stage_01', 'execution': {'status': 'reused'}, 'outputs': [str(video)]},
            {'stage_id': 'stage_02', 'execution': {'status': 'error', 'history': {'error': message}}}]}
    report = {'attempts': [first, failed(2, 'VHS failed to extract audio: Output file does not contain any stream'),
                          failed(3, 'GetImageRangeFromBatch: End index is out of range')]}
    assert runner.recover_runtime_parameter_failure(report) == 2
    assert report['attempts'] == [first]
    assert len(report['parameter_failures']) == 2
    saved = report['parameter_checkpoint']
    assert saved['attempt'] == 2 and saved['comfyui_attempt_consumed'] is False
    assert saved['stages'][0]['primary_output'] == str(video)
    assert saved['stages'][0]['artifact_sha256']
    assert runner.runtime_parameter_failure({'execution': {'status': 'error', 'error': 'CUDA out of memory'}}) is None
    assert runner.runtime_parameter_failure({'execution': {'status': 'success'}, 'feedback': 'GetImageRangeFromBatch: End index is out of range'}) is None


def test_parameter_checkpoint_resumes_repair_with_two_normal_rounds_left(tmp_path, monkeypatch):
    from comfyrouter.execution_contract import CaseExecutionRequest, ExecutionHooks, execution_context
    output = tmp_path / 'raw/case001'
    output.mkdir(parents=True)
    video = output / 'retained.mp4'
    video.write_bytes(b'cached upstream video')
    graph = {'1': {'class_type': 'LoadVideo', 'inputs': {'file': 'retained.mp4'}}}
    spec = {'modality': 'v2v', 'task_family': 'Enhancement', 'operations': ['Enhancement/frame_interpolation']}
    stages = [{'stage_id': 's1', 'purpose': 'upscale', 'instruction': 'upscale', 'modality': 'v2v',
               'input_type': 'VIDEO', 'output_type': 'VIDEO', 'depends_on': []},
              {'stage_id': 's2', 'purpose': 'interpolate', 'instruction': 'interpolate', 'modality': 'v2v',
               'input_type': 'VIDEO', 'output_type': 'VIDEO', 'depends_on': ['s1']}]
    report = {'case_id': '001', 'instruction': 'upscale then interpolate', 'max_attempts': 3,
              'trace': {'case_output': str(output), 'agent_audit': str(tmp_path / 'audit')},
              'resource_policy': {'vram_limit_bytes': 123}, 'case_resources': [],
              'attempts': [{'attempt': 1, 'passed': False}],
              'composition': {'status': 'ready', 'stages': [
                  {'stage': stage, 'requirement_spec': spec, 'implementation': {}, 'preflight_graph': graph}
                  for stage in stages]}}
    failed = {'attempt': 2, 'stages': [
        {'stage_id': 's1', 'execution': {'status': 'success'}, 'primary_output': str(video), 'graph': graph},
        {'stage_id': 's2', 'execution': {'status': 'error', 'error': 'GetImageRangeFromBatch: End index is out of range'},
         'graph': graph}]}
    runner.record_runtime_parameter_failure(report, failed)
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(report))
    loop = Mock()
    monkeypatch.setattr(runner, 'run_composition', loop)
    request = CaseExecutionRequest('001', {'instruction': report['instruction'], 'modality': 'v2v'}, tmp_path / 'raw', interrupted_report=path)
    with execution_context(request=request, hooks=ExecutionHooks(agent_factory=lambda _: Mock(), executor_factory=lambda: Mock())):
        resumed = runner.resume_evaluated_case(request)
    assert len(resumed['attempts']) == 1
    assert loop.call_args.kwargs['start_attempt'] == 2
    assert loop.call_args.kwargs['resume_failed_parameters'] is True
    assert loop.call_args.kwargs['resume_after_evaluation'] is False
    assert loop.call_args.kwargs['initial_cache']['s1']['output'] == str(video)


def test_vhs_forced_rate_count_matches_the_installed_reader(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    import torch
    from imageio_ffmpeg import get_ffmpeg_exe
    video = tmp_path / 'source.mp4'
    subprocess.run([get_ffmpeg_exe(), '-v', 'error', '-f', 'lavfi', '-i',
                    'color=black:s=16x16:r=24.5:d=2', '-c:v', 'libx264', '-y', str(video)], check=True)
    source = runner.config.COMFYUI_ROOT / 'custom_nodes/ComfyUI-VideoHelperSuite/videohelpersuite/load_video_nodes.py'
    if not source.is_file():
        pytest.skip('optional local ComfyUI-VideoHelperSuite source is unavailable')
    tree = ast.parse(source.read_text())
    code = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'cv_frame_generator'], type_ignores=[])
    scope = dict(cv2=cv2, np=np, torch=torch, ProgressBar=lambda _: SimpleNamespace(update_absolute=lambda *a: None))
    exec(compile(code, str(source), 'exec'), scope)
    monkeypatch.setattr(runner, 'COMFYUI_INPUT', tmp_path)
    for rate, skip, every, cap in [(25, 0, 1, 0), (12, 3, 2, 0), (50, 2, 1, 15)]:
        reader = scope['cv_frame_generator'](str(video), rate, cap, skip, every)
        next(reader)  # metadata
        count = sum(1 for _ in reader)
        graph = {'1': {'class_type': 'VHS_LoadVideo', 'inputs': {
            'video': video.name, 'force_rate': rate, 'skip_first_frames': skip,
            'select_every_nth': every, 'frame_load_cap': cap}},
            '2': {'class_type': 'CreateVideo', 'inputs': {'images': ['1', 0], 'fps': rate}}}
        assert runner.validate_temporal_parameters(graph, 'process video')['outputs'][0]['frames'] == count


def test_vhs_optional_audio_accepts_silent_video_but_not_invalid_media(tmp_path):
    import torch
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
    source = runner.config.COMFYUI_ROOT / 'custom_nodes/ComfyUI-VideoHelperSuite/videohelpersuite/utils.py'
    if not source.is_file():
        pytest.skip('optional local ComfyUI-VideoHelperSuite source is unavailable')
    tree = ast.parse(source.read_text())
    isolated = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                               and n.name in {'get_audio', 'LazyAudioMap'}], type_ignores=[])
    scope = dict(torch=torch, re=re, subprocess=subprocess, ffmpeg_path=ffmpeg, ENCODE_ARGS=('utf-8', 'replace'), Mapping=Mapping)
    exec(compile(isolated, str(source), 'exec'), scope)
    silent = tmp_path / 'silent.mp4'
    audio = tmp_path / 'audio.mp4'
    for path, sound in [(silent, False), (audio, True)]:
        args = [ffmpeg, '-v', 'error', '-f', 'lavfi', '-i', 'color=black:s=16x16:r=5:d=0.4']
        if sound:
            args += ['-f', 'lavfi', '-i', 'sine=frequency=440:duration=0.4', '-c:a', 'aac']
        subprocess.run(args + ['-c:v', 'libx264', '-y', str(path)], check=True, capture_output=True)
    assert dict(scope['LazyAudioMap'](str(silent), 0, 0)) == {}
    assert scope['get_audio'](str(audio))['waveform'].numel() > 0
    invalid = tmp_path / 'corrupt.mp4'
    invalid.write_bytes(b'not video')
    with pytest.raises(Exception, match='VHS failed to extract audio'):
        scope['get_audio'](str(invalid))
