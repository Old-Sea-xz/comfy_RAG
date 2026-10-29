from __future__ import annotations
import unittest
import os
import copy
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from comfyrouter.agentic import AgentOrchestrator, AgentOutputError
from comfyrouter.closed_loop import ClosedLoopError
from comfyrouter.types import RequirementSpec
from tools.run_comfybench_case import apply_mask_handoff_adapter, bind_composition_preflight_inputs, bind_runtime_resource, composition_handoff_inputs, generate_plan_with_repair, local_registry_repair_candidates, handoff_input_groups, media_input_nodes, preflight_failure_record, repair_mode_for_context, run_composition, validate_stage_handoff_contract

class TestUpstreamMediaHandoff(unittest.TestCase):

    def test_runtime_fanout_binds_all_readers_and_preserves_executed_stage_on_error(self):
        for force_error in (False, True):
            with self.subTest(force_error=force_error), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                output = root / 'executor'
                output.mkdir()
                (root / 'input').mkdir()
                (output / 'generated.mp4').write_bytes(b'test-video')
                (output / 'edited.mp4').write_bytes(b'test-video')
                calls = []

                class Executor:
                    output_root = output
                    validator = None

                    def execute(self, graph):
                        calls.append(copy.deepcopy(graph))
                        name = 'generated.mp4' if len(calls) == 1 else 'edited.mp4'
                        return SimpleNamespace(as_dict=lambda: {'status': 'success', 'outputs': [name]})
                first = SimpleNamespace(stage_id='stage_01', instruction='generate', input_type='TEXT', output_type='VIDEO', depends_on=(), modality='t2v')
                second = SimpleNamespace(stage_id='stage_02', instruction='edit', input_type='VIDEO', output_type='VIDEO', depends_on=('stage_01',), modality='v2v')
                graph = {'1': {'class_type': 'VHS_LoadVideo', 'inputs': {'video': 'INPUT_VIDEO.mp4', 'skip_first_frames': 0}}, '2': {'class_type': 'VHS_LoadVideo', 'inputs': {'video': 'INPUT_VIDEO.mp4', 'skip_first_frames': 24}}}
                bind_composition_preflight_inputs(graph, second, [])
                states = []
                for stage, candidate in [(first, {}), (second, graph)]:
                    nodes = [] if stage is first else [('1', 'video'), ('2', 'video')]
                    states.append({'typed': stage, 'spec': RequirementSpec(stage.modality, 'Generation', ('Generation/test',)), 'implementation': {'operation': 'Generation/test'}, 'base_graph': candidate, 'graph': candidate, 'handoff_inputs': nodes, 'handoff_inputs_by_type': {'VIDEO': nodes}})
                report = {'run_id': 'handoff-test', 'attempts': []}
                with patch('tools.run_comfybench_case.COMFYUI_INPUT', root / 'input'), patch('tools.run_comfybench_case.comfyui_serial_execution', side_effect=lambda *a: nullcontext()), patch('tools.run_comfybench_case.release_comfyui_models', return_value={'status': 'released'}), patch('tools.run_comfybench_case.normalize_exact_image_geometry', return_value=None), patch('tools.run_comfybench_case.inspect_runtime_media_inputs', return_value={'status': 'mocked_decoder'}), patch('tools.run_comfybench_case.official_check', return_value={'comfybench': {'resolved': True}}):
                    if force_error:
                        with patch('tools.run_comfybench_case.require_runtime_resources_bound', side_effect=ClosedLoopError('unbound test input')):
                            run_composition('test', 'generate then edit', states, SimpleNamespace(), Executor(), root / 'results', report, lambda: None, 1, candidate_validation_only=True)
                    else:
                        run_composition('test', 'generate then edit', states, SimpleNamespace(), Executor(), root / 'results', report, lambda: None, 1)
                self.assertEqual(len(report['attempts']), 0 if force_error else 1)
                row = report['parameter_checkpoint'] if force_error else report['attempts'][0]
                self.assertTrue(row['stages'][0]['execution_passed'])
                if force_error:
                    self.assertEqual(len(calls), 1)
                    self.assertFalse(row['logical_iteration_consumed'])
                    self.assertEqual(row['stages'][1]['status'], 'handoff_failed')
                    self.assertIn('unbound test input', row['stages'][1]['error'])
                else:
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(calls[1]['1']['inputs']['video'], calls[1]['2']['inputs']['video'])
                    self.assertNotEqual(calls[1]['2']['inputs']['video'], 'INPUT_VIDEO.mp4')
                    self.assertEqual(calls[1]['2']['inputs']['skip_first_frames'], 24)
                    self.assertTrue(report['attempts'][0]['passed'])

    def test_two_image_slots_survive_preflight_binding_to_same_source(self):
        graph = {'9': {'class_type': 'LoadImage', 'inputs': {'image': 'START_IMAGE.png'}}, '10': {'class_type': 'LoadImage', 'inputs': {'image': 'END_IMAGE.png'}}}
        stage = SimpleNamespace(input_type='IMAGE', input_types=('IMAGE',), modality='i2v')
        bind_composition_preflight_inputs(graph, stage, [Path('family_trip.jpg')])
        self.assertEqual(graph['9']['inputs']['image'], 'family_trip.jpg')
        self.assertEqual(graph['10']['inputs']['image'], 'family_trip.jpg')
        self.assertEqual(media_input_nodes(graph, 'IMAGE'), [('9', 'image'), ('10', 'image')])
        groups = handoff_input_groups(graph, 'IMAGE')
        self.assertEqual(groups, [[('9', 'image')], [('10', 'image')]])
        for nodes, path in zip(groups, ['child.png', 'elderly.png']):
            for node_id, key in nodes:
                graph[node_id]['inputs'][key] = path
        self.assertEqual([graph[n]['inputs']['image'] for n in ['9', '10']], ['child.png', 'elderly.png'])
        validate_stage_handoff_contract(graph, {'producer_dependencies': [{'media_type': 'IMAGE'}, {'media_type': 'IMAGE'}]})

    def test_start_and_end_slots_do_not_depend_on_graph_dictionary_order(self):
        graph = {'10': {'class_type': 'LoadImage', 'inputs': {'image': 'END_IMAGE.png'}}, '9': {'class_type': 'LoadImage', 'inputs': {'image': 'START_IMAGE.png'}}}
        stage = SimpleNamespace(input_type='IMAGE', input_types=('IMAGE',), modality='i2v')
        bind_composition_preflight_inputs(graph, stage, [Path('family_trip.jpg')])
        self.assertEqual(handoff_input_groups(graph, 'IMAGE'), [[('9', 'image')], [('10', 'image')]])

    def test_one_video_artifact_can_feed_two_frame_range_loaders(self):
        graph = {'1': {'class_type': 'VHS_LoadVideo', 'inputs': {'video': 'INPUT_VIDEO.mp4', 'skip_first_frames': 0}}, '2': {'class_type': 'VHS_LoadVideo', 'inputs': {'video': 'INPUT_VIDEO.mp4', 'skip_first_frames': 24}}}
        stage = SimpleNamespace(input_type='VIDEO', input_types=('VIDEO',), modality='v2v')
        bind_composition_preflight_inputs(graph, stage, [])
        groups = handoff_input_groups(graph, 'VIDEO')
        self.assertEqual(groups, [[('1', 'video'), ('2', 'video')]])
        for node_id, key in groups[0]:
            graph[node_id]['inputs'][key] = 'generated.mp4'
        self.assertEqual(handoff_input_groups(graph, 'VIDEO'), groups)
        self.assertEqual([graph[n]['inputs']['skip_first_frames'] for n in ['1', '2']], [0, 24])
        validate_stage_handoff_contract(graph, {'producer_dependencies': [{'media_type': 'VIDEO'}]})
        with self.assertRaisesRegex(AgentOutputError, 'needs 2'):
            validate_stage_handoff_contract(graph, {'producer_dependencies': [{'media_type': 'VIDEO'}, {'media_type': 'VIDEO'}]})

    def test_two_independent_video_slots_stay_separate(self):
        graph = {'1': {'class_type': 'VHS_LoadVideo', 'inputs': {'video': 'INPUT_VIDEO_A.mp4'}}, '2': {'class_type': 'VHS_LoadVideo', 'inputs': {'video': 'INPUT_VIDEO_B.mp4'}}}
        stage = SimpleNamespace(input_type='VIDEO', input_types=('VIDEO',), modality='v2v')
        bind_composition_preflight_inputs(graph, stage, [Path('source.mp4')])
        self.assertEqual(handoff_input_groups(graph, 'VIDEO'), [[('1', 'video')], [('2', 'video')]])

    def test_mask_loader_is_not_displaced_by_high_scoring_registry_candidates(self):
        registry = {f'MaskBatchCompositeSubtractShape{i}': {'input': {'required': {'mask': ['MASK']}}, 'output': ['IMAGE', 'MASK']} for i in range(100)}
        registry.update({'LoadImageMask': {'input': {'required': {'image': [['MASK_INPUT.png']], 'channel': [['red', 'alpha']]}}, 'output': ['MASK']}, 'ImageToMask': {'input': {'required': {'image': ['IMAGE']}}, 'output': ['MASK']}, 'ImageCompositeMasked': {'input': {'required': {'mask': ['MASK']}}, 'output': ['IMAGE']}})
        executor = SimpleNamespace(ensure_validator=lambda: SimpleNamespace(node_registry=registry))
        diagnosis = {'setup_failure': 'handoff_failed: required MASK from upstream needs 1 compatible external loader(s), but the selected graph exposes 0'}
        candidates = local_registry_repair_candidates(executor, diagnosis, limit=3)
        self.assertEqual([r['class_type'] for r in candidates], ['LoadImageMask', 'ImageToMask', 'ImageCompositeMasked'])

    def test_geometry_repair_exposes_installed_padding_despite_registry_noise(self):
        registry = {f'ImageBatchMaskComposite{i}': {'input': {'required': {'image': ['IMAGE']}}, 'output': ['IMAGE']} for i in range(100)}
        registry.update({name: {'input': {'required': {'image': ['IMAGE']}}, 'output': ['IMAGE']} for name in ['ImageScale', 'ImagePadForOutpaint', 'EmptyImage', 'ImageCompositeMasked']})
        executor = SimpleNamespace(ensure_validator=lambda: SimpleNamespace(node_registry=registry))
        candidates = local_registry_repair_candidates(executor, {'error': 'portrait aspect ratio needs contain padding for square canvas'}, limit=4)
        self.assertEqual([r['class_type'] for r in candidates], ['ImageScale', 'ImagePadForOutpaint', 'EmptyImage', 'ImageCompositeMasked'])
        self.assertNotIn('ImageScaleBy', {r['class_type'] for r in candidates})

    def test_live_interfaces_reach_generation_and_reconstruction_without_graph_mutation(self):
        registry = {'LoadImage': {'input': {'required': {'image': [['source.png']]}}, 'output': ['IMAGE', 'MASK']}, 'LoadImageMask': {'input': {'required': {'image': [['MASK_INPUT.png']], 'channel': [['red', 'alpha']]}}, 'output': ['MASK']}, 'ImageCompositeMasked': {'input': {'required': {'mask': ['MASK']}}, 'output': ['IMAGE']}}
        executor = SimpleNamespace(ensure_validator=lambda: SimpleNamespace(node_registry=registry))
        calls = []
        agent = SimpleNamespace(client=object(), generate_graph_edit_plan=lambda *a, **kw: calls.append(kw) or {})
        graph = {'1': {'class_type': 'LoadImage', 'inputs': {'image': 'source.png'}}}
        error = 'handoff_failed: required MASK from upstream needs 1 compatible external loader(s), but the selected graph exposes 0'
        contract = {'required_input_types': ['IMAGE', 'MASK'], 'producer_dependencies': [{'media_type': 'MASK'}], 'mask_requirement': {'status': 'required'}}
        kwargs = dict(graph=graph, allow_topology_repair=True, handoff_contract_context=contract, repair_evidence={'setup_failure': error, 'preflight_failures': [{'error': error}]})
        with patch('tools.run_comfybench_case.config.base_only_ablation', return_value=False):
            generate_plan_with_repair(agent, executor, None, [], **kwargs)
            interfaces = {row['class_type']: row for row in calls[0]['repair_evidence']['installed_node_interfaces']}
            self.assertEqual(interfaces['LoadImageMask']['input'], registry['LoadImageMask']['input'])
            self.assertEqual(interfaces['LoadImageMask']['output'], ['MASK'])
            with patch('tools.run_comfybench_case.reconstruct_plan', return_value={}) as reconstruct:
                kwargs['repair_evidence']['preflight_failures'] *= 3
                generate_plan_with_repair(agent, executor, None, [], **kwargs)
                names = {row['class_type'] for row in reconstruct.call_args.kwargs['interfaces']}
                self.assertIn('LoadImageMask', names)
                self.assertIn('ImageCompositeMasked', names)
        self.assertEqual(graph, {'1': {'class_type': 'LoadImage', 'inputs': {'image': 'source.png'}}})

    def test_required_video_mask_conversion_and_audio_interfaces_are_included(self):
        registry = {'LoadVideo': {'input': {'required': {'file': ['STRING']}}, 'output': ['VIDEO']}, 'GetVideoComponents': {'input': {'required': {'video': ['VIDEO']}}, 'output': ['IMAGE', 'AUDIO', 'FLOAT']}, 'ImageToMask': {'input': {'required': {'image': ['IMAGE']}}, 'output': ['MASK']}, 'CreateVideo': {'input': {'required': {'images': ['IMAGE']}}, 'output': ['VIDEO']}, 'LoadAudio': {'input': {'required': {'audio': ['STRING']}}, 'output': ['AUDIO']}}
        executor = SimpleNamespace(ensure_validator=lambda: SimpleNamespace(node_registry=registry))
        contract = {'required_input_types': ['VIDEO', 'MASK', 'AUDIO'], 'mask_requirement': {'status': 'required'}}
        candidates = local_registry_repair_candidates(executor, {}, handoff_contract=contract)
        self.assertEqual({r['class_type'] for r in candidates}, set(registry))
        self.assertNotIn('LoadImageMask', {r['class_type'] for r in candidates})

    def test_shared_fixture_video_and_mask_are_bound_to_distinct_nodes(self):
        for loader, key in (('LoadVideo', 'file'), ('VHS_LoadVideo', 'video')):
            with self.subTest(loader=loader):
                graph = {'1': {'class_type': loader, 'inputs': {key: 'comfyrouter_fixture.mp4'}}, '27': {'class_type': loader, 'inputs': {key: 'comfyrouter_fixture.mp4'}, '_meta': {'comfyrouter_adapter': 'external_mask_video_loader'}}}
                contract = {'producer_dependencies': [{'media_type': 'VIDEO'}, {'media_type': 'MASK'}], 'mask_requirement': {'status': 'required'}}
                validate_stage_handoff_contract(graph, contract)
                by_type, _ = composition_handoff_inputs(graph, SimpleNamespace(input_type='VIDEO', input_types=('VIDEO', 'MASK')))
                self.assertEqual(by_type, {'VIDEO': [('1', key)], 'MASK': [('27', key)]})
                bind_runtime_resource(graph, Path('INPUT_VIDEO.mp4'), 'MASK')
                bind_runtime_resource(graph, Path('mountain.mp4'), 'VIDEO')
                self.assertEqual(graph['1']['inputs'][key], 'mountain.mp4')
                self.assertEqual(graph['27']['inputs'][key], 'INPUT_VIDEO.mp4')
                validate_stage_handoff_contract(graph, contract)

    def test_existing_video_mask_adapter_retains_role_after_placeholder_binding(self):
        graph = {'1': {'class_type': 'LoadVideo', 'inputs': {'file': 'INPUT_VIDEO.mp4'}}, '27': {'class_type': 'LoadVideo', 'inputs': {'file': 'MASK_INPUT.mp4'}}}
        adapted, _ = apply_mask_handoff_adapter(graph, 'Background_Editing/replacement')
        for node in adapted.values():
            node['inputs']['file'] = 'comfyrouter_fixture.mp4'
        self.assertEqual(media_input_nodes(adapted, 'VIDEO'), [('1', 'file')])
        self.assertEqual(media_input_nodes(adapted, 'MASK'), [('27', 'file')])

    def test_unmarked_duplicate_video_inputs_remain_ambiguous(self):
        graph = {'1': {'class_type': 'LoadVideo', 'inputs': {'file': 'fixture.mp4'}}, '2': {'class_type': 'LoadVideo', 'inputs': {'file': 'fixture.mp4'}}}
        self.assertEqual(media_input_nodes(graph, 'VIDEO'), [])
        self.assertEqual(media_input_nodes(graph, 'MASK'), [])

    def test_reference_image_gap_is_repaired_before_video_execution(self):
        graph = {'1': {'class_type': 'LoadVideo', 'inputs': {'file': 'INPUT_VIDEO.mp4'}}}
        contract = {'producer_dependencies': [{'stage_id': 'stage_01', 'media_type': 'IMAGE'}], 'mask_requirement': {'status': 'not_needed'}}
        with self.assertRaisesRegex(AgentOutputError, 'handoff_failed: required IMAGE from upstream') as raised:
            validate_stage_handoff_contract(graph, contract)
        failure = preflight_failure_record(raised.exception, 'base_handoff_validation')
        mode = repair_mode_for_context(failures=[failure], operation='Motion_Cloning/speaking', input_types=('VIDEO', 'IMAGE'), execution_failed=True)
        self.assertTrue(mode['allow_topology_repair'])
        graph['2'] = {'class_type': 'LoadImage', 'inputs': {'image': 'INPUT_IMAGE.png'}}
        validate_stage_handoff_contract(graph, contract)

    def test_multiple_image_dependencies_need_distinct_input_slots(self):
        graph = {'1': {'class_type': 'LoadImage', 'inputs': {'image': 'INPUT_IMAGE.png'}}}
        contract = {'producer_dependencies': [{'stage_id': 'stage_01', 'media_type': 'IMAGE'}, {'stage_id': 'stage_02', 'media_type': 'IMAGE'}]}
        with self.assertRaisesRegex(AgentOutputError, 'needs 2'):
            validate_stage_handoff_contract(graph, contract)
        graph['2'] = {'class_type': 'LoadImage', 'inputs': {'image': 'END_IMAGE.png'}}
        validate_stage_handoff_contract(graph, contract)

    def test_video_and_audio_dependencies_are_checked(self):
        for media_type, loader, input_name, filename in (('VIDEO', 'LoadVideo', 'file', 'INPUT_VIDEO.mp4'), ('AUDIO', 'LoadAudio', 'audio', 'INPUT_AUDIO.wav')):
            with self.subTest(media_type=media_type):
                contract = {'producer_dependencies': [{'media_type': media_type}]}
                with self.assertRaisesRegex(AgentOutputError, media_type):
                    validate_stage_handoff_contract({}, contract)
                validate_stage_handoff_contract({'1': {'class_type': loader, 'inputs': {input_name: filename}}}, contract)

    def test_single_video_internal_reference_does_not_require_image_loader(self):
        validate_stage_handoff_contract({}, {'required_input_types': ['VIDEO'], 'producer_dependencies': [], 'mask_requirement': {'status': 'not_needed'}})

    def test_explicit_mask_without_upstream_still_requires_loader(self):
        with self.assertRaisesRegex(AgentOutputError, 'required MASK'):
            validate_stage_handoff_contract({}, {'mask_requirement': {'status': 'required'}})
if __name__ == '__main__':
    unittest.main()
