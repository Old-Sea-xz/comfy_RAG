from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from comfyrouter.agentic import AgentOrchestrator, AgentOutputError, AgentTransportError
from comfyrouter.types import RequirementSpec
from tools.run_comfybench_case import (
    composition_mask_role,
    composition_diagnosis_media,
    failure_analysis_required_repair_stage,
    enforce_forward_stage_ownership,
    final_failure_status,
    merge_runtime_repair_inputs,
    reusable_linear_composition_diagnosis,
    run_composition,
    select_primary_output,
    split_primary_and_diagnostic_outputs,
)


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_json(self, agent_id, contract, payload):
        self.calls.append((agent_id, contract, payload))
        return self.response


def orchestrator(response):
    agent = object.__new__(AgentOrchestrator)
    agent.client = FakeClient(response)
    agent.max_corrections = 0
    agent.image_max_dimension = 0
    return agent


def conditioning_attribution():
    return {
        key: {
            "likelihood": "unknown",
            "evidence": [],
            "counterfactual": "A completed output comparison would resolve this channel.",
        }
        for key in (
            "text_conditioning", "source_image_conditioning", "mask_conditioning",
            "negative_conditioning", "sampler_conditioning",
        )
    }


class TestCompositionForwardDiagnosis(unittest.TestCase):
    def test_later_independent_pass_cannot_discard_concrete_earliest_failure(self):
        response = {
            'earliest_failing_stage_id': 'stage_01', 'repair_stage_id': 'stage_01',
            'contributing_stage_ids': [],
            'stage_assessments': [
                {'stage_id': 'stage_01', 'status': 'fail', 'observations': ['adult unchanged'], 'reason': 'not child'},
                {'stage_id': 'stage_02', 'status': 'pass', 'observations': ['elderly generated'], 'reason': 'independent result valid'},
            ],
            'conditioning_attribution': conditioning_attribution(),
            'rationale': 'The child frame needs repair.', 'evidence': ['Child frame still depicts adult.'],
        }
        result = orchestrator(response).diagnose_composition_failure(
            instruction='generate age keyframes', stages=[{'stage_id': 'stage_01'}, {'stage_id': 'stage_02'}],
            final_evaluation={}, media_paths=[])
        self.assertEqual(result['repair_stage_id'], 'stage_01')
        self.assertEqual(result['stage_assessments'][1]['status'], 'deferred_due_to_upstream_failure')
        self.assertEqual(response['stage_assessments'][1]['status'], 'pass')
        self.assertEqual(result['diagnosis_protocol_normalizations'][0]['original_status'], 'pass')

    def test_runtime_repair_inputs_do_not_cross_adapter_class_boundary(self):
        candidate = {
            "1": {"class_type": "LoadVideo", "inputs": {"file": "source.mp4"}},
        }
        runtime_graph = {
            "1": {
                "class_type": "VHS_LoadVideo",
                "inputs": {
                    "video": "source.mp4",
                    "force_rate": 1.5,
                    "frame_load_cap": 48,
                    "custom_width": 0,
                    "custom_height": 0,
                    "format": "AnimateDiff",
                },
            },
        }

        merged = merge_runtime_repair_inputs(candidate, runtime_graph)

        self.assertEqual(merged, candidate)
        self.assertNotIn("frame_load_cap", merged["1"]["inputs"])

    def test_runtime_repair_inputs_merge_for_unchanged_node_class(self):
        candidate = {
            "1": {"class_type": "VHS_LoadVideo", "inputs": {"video": "source.mp4"}},
        }
        runtime_graph = {
            "1": {
                "class_type": "VHS_LoadVideo",
                "inputs": {
                    "video": "source.mp4",
                    "force_rate": 1.5,
                    "frame_load_cap": 48,
                },
            },
        }

        merged = merge_runtime_repair_inputs(candidate, runtime_graph)

        self.assertEqual(merged["1"]["inputs"]["force_rate"], 1.5)
        self.assertEqual(merged["1"]["inputs"]["frame_load_cap"], 48)
        self.assertNotIn("frame_load_cap", candidate["1"]["inputs"])

    def test_passed_upstream_is_frozen_when_later_stage_fails(self):
        states = [
            {"typed": SimpleNamespace(stage_id="stage_01")},
            {"typed": SimpleNamespace(stage_id="stage_02")},
            {"typed": SimpleNamespace(stage_id="stage_03")},
        ]
        attempt = {"stages": [
            {"stage_id": "stage_01"},
            {"stage_id": "stage_02"},
            {"stage_id": "stage_03"},
        ]}
        diagnosis = {
            "earliest_failing_stage_id": "stage_03",
            # Simulate an over-eager model trying to reopen a passed mask.
            "repair_stage_id": "stage_01",
            "stage_assessments": [
                {"stage_id": "stage_01", "status": "pass", "observations": ["mask is correct"], "reason": "pass"},
                {"stage_id": "stage_02", "status": "pass", "observations": ["removal is correct"], "reason": "pass"},
                {"stage_id": "stage_03", "status": "fail", "observations": ["seam is wrong"], "reason": "fail"},
            ],
        }
        result = enforce_forward_stage_ownership(diagnosis, attempt, states)
        self.assertEqual(result["repair_stage_id"], "stage_03")
        self.assertTrue(attempt["stages"][0]["artifact_frozen"])
        self.assertTrue(attempt["stages"][1]["artifact_frozen"])
        self.assertFalse(attempt["stages"][2]["artifact_frozen"])
        self.assertEqual(attempt["stages"][2]["failure_owner"], "stage_03")

    def test_legacy_removal_consumer_implies_edit_region_mask(self):
        implementation = [
            {
                "operation": "Mask_Processing/generation",
                "task_family": "Mask_Processing",
                "decomposition_step": {
                    "step": "1",
                    "description": "Generate a white mask selecting the plate and food.",
                    "output_type": "mask",
                },
            },
            {
                "operation": "Object_Editing/removal",
                "task_family": "Object_Editing",
                "decomposition_step": {
                    "step": "2",
                    "description": "Remove the plate and food.",
                    "output_type": "image",
                },
            },
        ]
        stage = SimpleNamespace(
            stage_id="stage_02", depends_on=("stage_01",),
            instruction="Remove the plate and food.", purpose="Remove objects",
        )
        self.assertEqual(composition_mask_role(stage, implementation), "edit_region")

    def test_legacy_ambiguous_mask_contract_still_fails_closed(self):
        implementation = [
            {
                "operation": "Mask_Processing/generation",
                "task_family": "Mask_Processing",
                "decomposition_step": {
                    "step": "1", "description": "Generate a selected-region mask.",
                    "output_type": "mask",
                },
            },
            {
                "operation": "Style_Editing/repainting",
                "task_family": "Style_Editing",
                "decomposition_step": {
                    "step": "2", "description": "Produce the requested result.",
                    "output_type": "image",
                },
            },
        ]
        stage = SimpleNamespace(
            stage_id="stage_02", depends_on=("stage_01",),
            instruction="Produce the requested result.", purpose="Transform image",
        )
        with self.assertRaisesRegex(Exception, "explicit edit_region/preserve_region"):
            composition_mask_role(stage, implementation)

    def test_retry_reuses_linear_final_stage_diagnosis(self):
        typed = [
            SimpleNamespace(stage_id="stage_01", depends_on=()),
            SimpleNamespace(stage_id="stage_02", depends_on=("stage_01",)),
        ]
        states = [{"typed": item} for item in typed]
        previous = {
            "attempt": 1,
            "stage_failure_diagnosis": {"status": "selected", "decision": {
                "earliest_failing_stage_id": "stage_02",
                "repair_stage_id": "stage_02",
                "stage_assessments": [
                    {"stage_id": "stage_01", "status": "pass",
                     "observations": ["valid source"], "reason": "passes"},
                    {"stage_id": "stage_02", "status": "fail",
                     "observations": ["old failure"], "reason": "fails"},
                ],
                "conditioning_attribution": conditioning_attribution(),
            }},
        }
        current = {
            "attempt": 2,
            "stages": [{"status": "reused"}, {"execution": {"status": "success"}}],
            "comfybench": {"comfybench": {
                "resolved": False,
                "analysis": "The newly generated final stage still misses the requested visual change.",
            }},
        }
        diagnosis = reusable_linear_composition_diagnosis(
            current, previous, states,
            {"diagnosis": "final transformation remains incomplete"},
        )
        self.assertEqual(diagnosis["repair_stage_id"], "stage_02")
        self.assertEqual(diagnosis["reused_from_attempt"], 1)
        self.assertEqual(diagnosis["stage_assessments"][0]["status"], "pass")

    def test_final_failure_status_preserves_specific_runner_failure(self):
        self.assertEqual(final_failure_status({
            "attempts": [{"stages": [{"status": "handoff_failed"}]}],
        }), "handoff_failed")
        self.assertEqual(final_failure_status({
            "status": "repair_preflight_failed", "attempts": [],
        }), "repair_preflight_failed")
        self.assertEqual(final_failure_status({
            "attempts": [{"passed": False, "stages": []}],
        }), "quality_not_passed")

    def test_final_failure_status_detects_single_leaf_execution_failure(self):
        self.assertEqual(final_failure_status({
            "attempts": [{"execution": {"status": "executor_error"}}],
        }), "execution_failed")

    def test_final_failure_status_uses_terminal_attempt(self):
        self.assertEqual(final_failure_status({
            "attempts": [
                {"stages": [{"status": "handoff_failed"}]},
                {"passed": False, "stages": [
                    {"execution": {"status": "success"}},
                ]},
            ],
        }), "quality_not_passed")

    def test_prompt_only_repeat_detection_uses_latest_repair(self):
        graph = {
            "23": {"class_type": "CLIPTextEncode", "inputs": {"text": "street"}},
            "26": {"class_type": "FluxGuidance", "inputs": {"guidance": 30}},
        }
        history = [{
            "stages": [{
                "repair_generation_plans": [{
                    "plan": {"edits": [{
                        "action": "set_param", "node_id": "23", "param": "text", "value": "road",
                    }]},
                }],
            }],
        }]
        self.assertTrue(AgentOrchestrator._latest_repair_was_prompt_only(history, graph))
        history[0]["stages"][0]["repair_generation_plans"][0]["plan"]["edits"] = [{
            "action": "set_param", "node_id": "26", "param": "guidance", "value": 10,
        }]
        self.assertFalse(AgentOrchestrator._latest_repair_was_prompt_only(history, graph))

    def test_diagnosis_can_repair_a_contributing_upstream_stage(self):
        response = {
            "earliest_failing_stage_id": "stage_02",
            "repair_stage_id": "stage_01",
            "contributing_stage_ids": ["stage_01"],
            "stage_assessments": [
                {"stage_id": "stage_01", "status": "pass",
                 "observations": ["mask covers target"], "reason": "atomic mask contract passes"},
                {"stage_id": "stage_02", "status": "fail",
                 "observations": ["object is reconstructed"], "reason": "removal fails"},
            ],
            "conditioning_attribution": conditioning_attribution(),
            "rationale": "The valid mask still supplies a harmful silhouette prior.",
            "evidence": ["The output repeats the mask silhouette."],
        }
        agent = orchestrator(response)
        result = agent.diagnose_composition_failure(
            instruction="remove object",
            stages=[{"stage_id": "stage_01"}, {"stage_id": "stage_02"}],
            final_evaluation={}, media_paths=[],
        )
        self.assertEqual(result["repair_stage_id"], "stage_01")

    def test_mask_intermediate_is_not_selected_for_scoring_or_handoff(self):
        outputs = [
            Path("mask_intermediate__object_removal_00001_.png"),
            Path("final_result_00001_.png"),
        ]
        primary, diagnostics = split_primary_and_diagnostic_outputs(outputs)
        self.assertEqual(primary, [outputs[1]])
        self.assertEqual(diagnostics, [outputs[0]])
        self.assertEqual(select_primary_output(outputs), outputs[1])

    def test_only_mask_intermediate_has_no_primary_output(self):
        outputs = [Path("mask_intermediate__object_removal_00001_.png")]
        self.assertIsNone(select_primary_output(outputs))

    def test_execution_agent_selects_first_failure_and_receives_images(self):
        response = {
            "earliest_failing_stage_id": "stage_02",
            "repair_stage_id": "stage_02",
            "contributing_stage_ids": [],
            "stage_assessments": [
                {"stage_id": "stage_01", "status": "pass",
                 "observations": ["position 1 satisfies removal"], "reason": "atomic edit is complete"},
                {"stage_id": "stage_02", "status": "fail",
                 "observations": ["position 2 lacks target style"], "reason": "repainting is absent"},
            ],
            "rationale": "The first stage passes and the second is the earliest failure.",
            "evidence": ["Visual positions 1 and 2 show the transition."],
            "conditioning_attribution": conditioning_attribution(),
        }
        agent = orchestrator(response)
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "one.png"
            second = Path(temp) / "two.png"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            result = agent.diagnose_composition_failure(
                instruction="remove then repaint",
                stages=[
                    {"stage_id": "stage_01", "atomic_instruction": "remove object"},
                    {"stage_id": "stage_02", "atomic_instruction": "repaint image"},
                ],
                final_evaluation={"judgment": "False"},
                media_paths=[str(first), str(second)],
            )
        self.assertEqual(result["earliest_failing_stage_id"], "stage_02")
        agent_id, contract, payload = agent.client.calls[0]
        self.assertEqual(agent_id, "execution_diagnosis_agent")
        self.assertEqual(payload["_prompt_history_mode"], "latest_only")
        self.assertEqual(payload["_max_output_tokens"], 2000)
        self.assertEqual(len(payload["_image_data_urls"]), 2)
        self.assertTrue(any(
            "Walk stages strictly from first to last" in rule for rule in contract["rules"]
        ))

    def test_rejects_selection_when_an_earlier_stage_is_not_pass(self):
        response = {
            "earliest_failing_stage_id": "stage_02",
            "repair_stage_id": "stage_02",
            "contributing_stage_ids": [],
            "stage_assessments": [
                {"stage_id": "stage_01", "status": "fail",
                 "observations": ["object remains"], "reason": "removal failed"},
                {"stage_id": "stage_02", "status": "fail",
                 "observations": ["style absent"], "reason": "repainting failed"},
            ],
            "rationale": "select stage two", "evidence": ["outputs"],
            "conditioning_attribution": conditioning_attribution(),
        }
        agent = orchestrator(response)
        with self.assertRaisesRegex(AgentOutputError, "before the selected stage must pass"):
            agent.diagnose_composition_failure(
                instruction="remove then repaint",
                stages=[{"stage_id": "stage_01"}, {"stage_id": "stage_02"}],
                final_evaluation={}, media_paths=[],
            )

    def test_image_stage_results_are_persisted_in_visual_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "first.png"
            second = root / "second.png"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            attempt = {
                "attempt": 1,
                "stages": [
                    {"outputs": [str(first)], "execution": {"status": "success"}, "handoff": None,
                     "graph": {"1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}}}},
                    {"outputs": [str(second)], "execution": {"status": "success"}, "handoff": {},
                     "graph": {"2": {"class_type": "LoadImageMask", "inputs": {"image": "bound-mask.png"}}}},
                ],
            }
            states = [
                {"typed": SimpleNamespace(stage_id="stage_01", instruction="first task",
                                           input_type="IMAGE", output_type="IMAGE"),
                 "implementation": {"operation": "A/first"}},
                {"typed": SimpleNamespace(stage_id="stage_02", instruction="second task",
                                           input_type="IMAGE", output_type="IMAGE"),
                 "implementation": {"operation": "B/second"}},
            ]
            descriptors, paths, manifest = composition_diagnosis_media(
                attempt, states, root / "attempt", [first]
            )
            self.assertEqual(paths, [str(first), str(first), str(second)])
            self.assertEqual(manifest[0]["role"], "original_input")
            self.assertEqual(descriptors[0]["visual_inputs"][0]["position"], 2)
            self.assertEqual(descriptors[1]["visual_inputs"][0]["position"], 3)
            self.assertEqual(
                descriptors[1]["executed_graph"]["2"]["inputs"]["image"],
                "bound-mask.png",
            )
            self.assertEqual(attempt["stages"][1]["diagnosis_media"]["visual_inputs"][0]["path"], str(second))

    def test_combined_diagnosis_compacts_graph_but_keeps_runtime_handoff(self):
        response = {
            "earliest_failing_stage_id": "stage_01",
            "repair_stage_id": "stage_01",
            "contributing_stage_ids": [],
            "stage_assessments": [{
                "stage_id": "stage_01", "status": "fail",
                "observations": ["wrong result"], "reason": "conditioning failed",
            }],
            "conditioning_attribution": conditioning_attribution(),
            "rationale": "The only stage failed.",
            "evidence": ["The output does not satisfy the instruction."],
        }
        agent = orchestrator(response)
        graph = {
            "1": {"class_type": "LoadImageMask", "inputs": {
                "image": "runtime-bound-mask.png", "channel": "red", "upload": "image",
            }, "_meta": {"title": "large duplicate metadata"}},
            "2": {"class_type": "KSampler", "inputs": {
                "model": ["9", 0], "positive": ["3", 0], "latent_image": ["4", 0],
                "steps": 20, "cfg": 7.0, "denoise": 0.8,
                "control_after_generate": "randomize",
            }},
        }
        agent.diagnose_composition_failure(
            instruction="edit image",
            stages=[{"stage_id": "stage_01", "executed_graph": graph,
                     "handoff": {"bound_file": "runtime-bound-mask.png"}}],
            final_evaluation={}, media_paths=[],
        )
        agent_id, _, payload = agent.client.calls[0]
        compact = payload["ordered_stages"][0]
        self.assertEqual(agent_id, "execution_diagnosis_agent")
        self.assertEqual(compact["handoff"]["bound_file"], "runtime-bound-mask.png")
        self.assertEqual(compact["executed_graph"]["1"]["inputs"]["image"],
                         "runtime-bound-mask.png")
        self.assertEqual(compact["executed_graph"]["2"]["inputs"]["model"], ["9", 0])
        self.assertNotIn("upload", compact["executed_graph"]["1"]["inputs"])
        self.assertNotIn("_meta", compact["executed_graph"]["1"])

    def test_resume_honors_explicit_upstream_repair_stage_without_affected_leaves(self):
        states = [
            {"typed": SimpleNamespace(stage_id="stage_01")},
            {"typed": SimpleNamespace(stage_id="stage_02")},
            {"typed": SimpleNamespace(stage_id="stage_03")},
        ]
        attempt = {
            "stage_failure_diagnosis": {
                "decision": {
                    "earliest_failing_stage_id": "stage_02",
                    "repair_stage_id": "stage_02",
                },
            },
            "stages": [{"stage_id": "stage_01"}, {"stage_id": "stage_02"},
                       {"stage_id": "stage_03", "status": "executor_error"}],
        }
        selected = failure_analysis_required_repair_stage(
            {"failure_class": "quality", "affected_leaves": []},
            states,
            attempt,
        )
        self.assertEqual(selected, "stage_02")

    def test_reused_stage_reuses_prior_diagnosis_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "stage.mp4"
            output.write_bytes(b"video")
            frame = root / "prior_frame.png"
            frame.write_bytes(b"frame")
            prior = {
                "attempt": 1,
                "stages": [{
                    "stage_id": "stage_01",
                    "diagnosis_media": {
                        "visual_inputs": [{"path": str(frame)}],
                        "video_metadata": {"num_frames": 49},
                        "media_error": None,
                    },
                }],
            }
            current = {
                "attempt": 2,
                "stages": [{
                    "stage_id": "stage_01", "status": "reused",
                    "reused_from_attempt": 1, "outputs": [str(output)],
                    "execution": {"status": "reused"}, "graph": {},
                }],
            }
            states = [{
                "typed": SimpleNamespace(
                    stage_id="stage_01", instruction="task",
                    input_type="VIDEO", output_type="VIDEO",
                ),
                "implementation": {"operation": "A/video"},
            }]
            with patch("tools.run_comfybench_case.sample_video") as sample:
                descriptors, paths, _ = composition_diagnosis_media(
                    current, states, root / "attempt_02", prior_attempts=[prior],
                )
            sample.assert_not_called()
            self.assertEqual(paths, [str(frame)])
            self.assertEqual(descriptors[0]["video_metadata"]["num_frames"], 49)
            self.assertEqual(
                current["stages"][0]["diagnosis_media"]["reused_from_attempt"], 1,
            )

    def test_composition_loop_repairs_only_selected_stage(self):
        class Evidence:
            def __init__(self, payload):
                self.payload = payload

            def as_dict(self):
                return dict(self.payload)

        class Executor:
            def __init__(self, output_root, names):
                self.output_root = output_root
                self.names = iter(names)
                self.validator = None
                self.executions = 0

            def execute(self, graph):
                self.executions += 1
                return Evidence({"status": "success", "outputs": [next(self.names)]})

            def static_validate(self, graph, **kwargs):
                return None

        class Agent:
            def __init__(self):
                self.repaired_operations = []
                self.diagnosis_requests = []

            def diagnose_composition_failure(self, **kwargs):
                self.diagnosis_requests.append(kwargs)
                return {
                    "earliest_failing_stage_id": "stage_02",
                    "repair_stage_id": "stage_02",
                    "contributing_stage_ids": [],
                    "stage_assessments": [
                        {"stage_id": "stage_01", "status": "pass",
                         "observations": ["first output is correct"], "reason": "passes"},
                        {"stage_id": "stage_02", "status": "fail",
                         "observations": ["second transformation is absent"], "reason": "fails"},
                    ],
                    "rationale": "stage two is the earliest failure",
                    "evidence": ["ordered outputs"],
                    "conditioning_attribution": conditioning_attribution(),
                    "failure_class": "quality",
                    "root_causes": ["stage two did not apply its transformation"],
                    "next_action": "parameter_retry",
                    "repair_focus": ["repair stage two parameters"],
                    "needs_search": False,
                    "search_queries": [],
                }

            def evaluate_visual_output(self, *args, **kwargs):
                return {"status": "pass", "completion_score": 100,
                        "task_alignment": "passes", "issues": [], "evidence": []}

            def decide_repair_search(self, **kwargs):
                self.fail("composition loop must not make a second repair-search call")
                return {"diagnosis": "stage two failed", "repair_focus": ["repair stage two"],
                        "needs_search": False, "search_queries": [], "evidence": ["output"]}

            def generate_graph_edit_plan(self, spec, implementation, **kwargs):
                self.repaired_operations.append(implementation[0]["operation"])
                return {
                    "skill_ids": ["test-replacement-geometry"],
                    "selected_workflow": {
                        "1": {"class_type": "LoadImage", "inputs": {
                            "strength": 1, "image": "INPUT_IMAGE.png",
                        }},
                    },
                    "edits": [{"action": "set_param", "target": "1", "key": "strength", "value": 2}],
                    "requires_composition_replan": False,
                }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output_root = root / "executor"
            output_root.mkdir()
            (root / "input").mkdir()
            names = ["a1s1.png", "a1s2.png", "a2s2.png"]
            for name in names:
                (output_root / name).write_bytes(name.encode())
            typed = [
                SimpleNamespace(stage_id="stage_01", instruction="first", input_type="TEXT",
                                output_type="IMAGE", depends_on=()),
                SimpleNamespace(stage_id="stage_02", instruction="second", input_type="IMAGE",
                                output_type="IMAGE", depends_on=("stage_01",)),
            ]
            states = []
            for index, operation in enumerate(("A/first", "B/second")):
                base_graph = {"1": {"class_type": "Dummy", "inputs": {
                    "strength": 1, "image": "INPUT_IMAGE.png",
                }}}
                states.append({
                    "typed": typed[index],
                    "spec": RequirementSpec("t2i" if index == 0 else "i2i", "A" if index == 0 else "B", (operation,)),
                    "implementation": {"operation": operation},
                    "base_graph": base_graph,
                    "graph": copy.deepcopy(base_graph),
                    "handoff_inputs": [] if index == 0 else [("1", "image")],
                    "handoff_inputs_by_type": {} if index == 0 else {"IMAGE": [("1", "image")]},
                })
            report = {"run_id": "test", "attempts": []}
            checks = [
                {"comfybench": {"resolved": False}},
                {"comfybench": {"resolved": True}},
            ]
            agent = Agent()
            executor = Executor(output_root, names)
            with patch("tools.run_comfybench_case.copy_case_resources", return_value=[]), \
                    patch("tools.run_comfybench_case.COMFYUI_INPUT", root / "input"), \
                    patch("tools.run_comfybench_case.release_comfyui_models",
                          return_value={"status": "released", "attempt": 1}), \
                    patch("tools.run_comfybench_case.require_runtime_resources_bound"), \
                    patch("tools.run_comfybench_case.inspect_runtime_media_inputs", return_value={"status": "mocked_decoder"}), \
                    patch("tools.run_comfybench_case.official_check", side_effect=checks):
                run_composition(
                    "x", "first then second", states, agent,
                    executor, root / "results", report,
                    lambda: None, 2,
                )
            self.assertEqual(agent.repaired_operations, ["B/second"])
            self.assertEqual(len(agent.diagnosis_requests), 1)
            self.assertTrue(agent.diagnosis_requests[0]["include_repair_decision"])
            self.assertEqual(
                [item["role"] for item in agent.diagnosis_requests[0]["visual_manifest"]],
                ["stage_output", "stage_output"],
            )
            self.assertEqual(len(agent.diagnosis_requests[0]["media_paths"]), 2)
            self.assertEqual(states[0]["graph"]["1"]["inputs"]["strength"], 1)
            self.assertEqual(states[1]["graph"]["1"]["inputs"]["strength"], 2)
            self.assertEqual(states[1]["graph"]["1"]["class_type"], "LoadImage")
            first_attempt = report["attempts"][0]
            self.assertEqual(first_attempt["stages"][0]["repair_decision"], "unchanged_by_forward_diagnosis")
            self.assertEqual(first_attempt["stages"][1]["repair_decision"], "selected_causal_repair_stage")
            self.assertEqual(executor.executions, 3)
            self.assertEqual(report["attempts"][1]["stages"][0]["status"], "reused")
            self.assertEqual(report["attempts"][1]["stages"][0]["reused_from_attempt"], 1)
            self.assertEqual(
                first_attempt["cache_invalidation"]["invalidated_stage_ids"], ["stage_02"]
            )

    def test_composition_repair_transport_failure_is_checkpointed_without_name_error(self):
        class Evidence:
            def as_dict(self):
                return {"status": "success", "outputs": ["stage.png"]}

        class Executor:
            def __init__(self, output_root):
                self.output_root = output_root
                self.validator = None

            def execute(self, graph):
                return Evidence()

            def static_validate(self, graph):
                return None

        class Agent:
            def evaluate_visual_output(self, *args, **kwargs):
                return {"status": "pass", "completion_score": 100,
                        "task_alignment": "passes", "issues": [], "evidence": []}

            def diagnose_composition_failure(self, **kwargs):
                return {
                    "earliest_failing_stage_id": "stage_01",
                    "repair_stage_id": "stage_01",
                    "contributing_stage_ids": [],
                    "stage_assessments": [{
                        "stage_id": "stage_01", "status": "fail",
                        "observations": ["output is wrong"], "reason": "fails",
                    }],
                    "rationale": "stage one needs repair",
                    "evidence": ["output"],
                    "conditioning_attribution": conditioning_attribution(),
                    "failure_class": "quality",
                    "root_causes": ["stage one did not apply the requested change"],
                    "next_action": "parameter_retry",
                    "repair_focus": ["repair stage one"],
                    "needs_search": False,
                    "search_queries": [],
                }

            def generate_graph_edit_plan(self, *args, **kwargs):
                raise AgentTransportError("three request attempts exhausted")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output_root = root / "executor"
            output_root.mkdir()
            (output_root / "stage.png").write_bytes(b"image")
            typed = SimpleNamespace(
                stage_id="stage_01", instruction="edit", input_type="TEXT",
                output_type="IMAGE", depends_on=(), modality="t2i",
            )
            states = [{
                "typed": typed,
                "spec": RequirementSpec("t2i", "A", ("A/edit",)),
                "implementation": {"operation": "A/edit"},
                "graph": {"1": {"class_type": "Dummy", "inputs": {"strength": 1}}},
                "handoff_inputs": [],
                "handoff_inputs_by_type": {},
            }]
            report = {"run_id": "test", "attempts": []}
            checkpoints = []
            with patch("tools.run_comfybench_case.copy_case_resources", return_value=[]), \
                    patch("tools.run_comfybench_case.release_comfyui_models",
                          return_value={"status": "released"}), \
                    patch("tools.run_comfybench_case.official_check",
                          return_value={"comfybench": {"resolved": False}}):
                run_composition(
                    "x", "edit", states, Agent(), Executor(output_root),
                    root / "results", report,
                    lambda: checkpoints.append(report.get("status")), 2,
                )
            self.assertEqual(report["status"], "agent_transport_failed")
            self.assertEqual(report["failure_stage"], "composition_repair_transport")
            self.assertIn("three request attempts exhausted", report["error"])
            plans = report["attempts"][0]["stages"][0]["repair_generation_plans"]
            self.assertEqual(plans[-1]["error_type"], "AgentTransportError")
            self.assertFalse(report["logical_iteration_consumed"])
            self.assertFalse(report["comfyui_attempt_consumed"])

    def test_terminal_composition_failure_does_not_plan_another_repair(self):
        class Evidence:
            def as_dict(self):
                return {"status": "success", "outputs": ["stage.png"]}

        class Executor:
            def __init__(self, output_root):
                self.output_root = output_root
                self.validator = None

            def execute(self, graph):
                return Evidence()

        class Agent:
            def diagnose_composition_failure(self, **kwargs):
                return {
                    "earliest_failing_stage_id": "stage_01",
                    "repair_stage_id": "stage_01",
                    "contributing_stage_ids": [],
                    "stage_assessments": [{
                        "stage_id": "stage_01", "status": "fail",
                        "observations": ["quality requirement is missing"],
                        "reason": "output fails the atomic stage contract",
                    }],
                    "rationale": "stage one has a quality failure",
                    "evidence": ["official evaluation rejected the image"],
                    "conditioning_attribution": conditioning_attribution(),
                    "failure_class": "quality",
                    "root_causes": ["the requested visual change is missing"],
                    "next_action": "topology_repair",
                    "repair_focus": ["repair stage one"],
                    "needs_search": False,
                    "search_queries": [],
                }

            def generate_graph_edit_plan(self, *args, **kwargs):
                raise AssertionError(
                    "terminal bounded attempt must not plan an unexecutable repair"
                )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output_root = root / "executor"
            output_root.mkdir()
            (output_root / "stage.png").write_bytes(b"image")
            typed = SimpleNamespace(
                stage_id="stage_01", instruction="edit", input_type="TEXT",
                output_type="IMAGE", depends_on=(), modality="t2i",
            )
            states = [{
                "typed": typed,
                "spec": RequirementSpec("t2i", "A", ("A/edit",)),
                "implementation": {"operation": "A/edit"},
                "graph": {"1": {"class_type": "Dummy", "inputs": {"strength": 1}}},
                "handoff_inputs": [],
                "handoff_inputs_by_type": {},
            }]
            report = {
                "run_id": "test", "attempts": [],
                "self_evolution": {"enabled": False, "candidates": []},
            }
            with patch("tools.run_comfybench_case.release_comfyui_models",
                       return_value={"status": "released"}), \
                    patch("tools.run_comfybench_case.official_check",
                          return_value={"comfybench": {
                              "resolved": False, "judgment": "False",
                              "analysis": "The requested edit is visibly missing.",
                          }}), \
                    patch("tools.run_comfybench_case.composition_diagnosis_media",
                          return_value=([{"stage_id": "stage_01"}], [], [])), \
                    patch("tools.run_comfybench_case.self_evolution_disabled",
                          return_value=True):
                run_composition(
                    "x", "edit", states, Agent(), Executor(output_root),
                    root / "results", report, lambda: None, 1,
                )

        self.assertEqual(len(report["attempts"]), 1)
        self.assertEqual(report["status"], "quality_not_passed")
        self.assertEqual(report["attempts"][0]["terminal_diagnosis"],
                         "skipped_no_remaining_execution_budget")
        self.assertNotIn("failure_analysis", report["attempts"][0])
        self.assertNotIn(
            "repair_generation_plans", report["attempts"][0]["stages"][0],
        )

    def test_valid_mask_continues_without_intermediate_semantic_scoring(self):
        class Evidence:
            def __init__(self, output):
                self.output = output

            def as_dict(self):
                return {"status": "success", "outputs": [self.output]}

        class Executor:
            def __init__(self, output_root):
                self.output_root = output_root
                self.validator = None
                self.executions = 0

            def execute(self, graph):
                self.executions += 1
                return Evidence("mask.png" if self.executions == 1 else "image.png")

        class Agent:
            pass

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output_root = root / "executor"
            output_root.mkdir()
            from PIL import Image
            Image.new("L", (32, 32), 255).save(output_root / "mask.png")
            Image.new("RGB", (32, 32)).save(output_root / "image.png")
            case_input = root / "input.png"
            Image.new("RGB", (32, 32)).save(case_input)
            states = [{
                "typed": SimpleNamespace(
                    stage_id="stage_01", instruction="create prospective mask",
                    input_type="IMAGE", output_type="MASK", depends_on=(), modality="i2i",
                ),
                "spec": RequirementSpec("i2i", "Mask_Processing", ("Mask_Processing/generation",)),
                "implementation": {
                    "operation": "Mask_Processing/generation",
                    "decomposition_step": {
                        "mask_role": "edit_region", "target_presence": "absent",
                        "mask_construction": "prospective_placement",
                        "geometry_requirement": "single_regular",
                        "protected_regions": ["subject"],
                    },
                },
                "graph": {"1": {"class_type": "LoadImage", "inputs": {"image": "INPUT_IMAGE.png"}}},
                "handoff_inputs": [], "handoff_inputs_by_type": {},
            }, {
                "typed": SimpleNamespace(
                    stage_id="stage_02", instruction="edit inside mask",
                    input_type="IMAGE", output_type="IMAGE",
                    depends_on=("stage_01",), modality="i2i",
                ),
                "spec": RequirementSpec("i2i", "Object_Editing", ("Object_Editing/addition",)),
                "implementation": {"operation": "Object_Editing/addition"},
                "graph": {
                    "17": {"class_type": "LoadImage", "inputs": {"image": "INPUT.png"}},
                    "49": {"class_type": "LoadImageMask", "inputs": {
                        "image": "MASK_INPUT.png", "channel": "red",
                    }},
                },
                "handoff_inputs": [("17", "image")],
                "handoff_inputs_by_type": {
                    "IMAGE": [("17", "image")],
                    "MASK": [("49", "image")],
                },
            }]
            report = {
                "run_id": "test", "attempts": [],
                "self_evolution": {"enabled": False, "candidates": []},
            }
            (root / "input").mkdir()
            with patch("tools.run_comfybench_case.COMFYUI_INPUT", root / "input"), \
                    patch("tools.run_comfybench_case.release_comfyui_models",
                       return_value={"status": "released"}), \
                    patch("tools.run_comfybench_case.inspect_runtime_media_inputs", return_value={"status": "mocked_decoder"}), \
                    patch("tools.run_comfybench_case.composition_diagnosis_media",
                          return_value=([{"stage_id": "stage_01"}], [], [])), \
                    patch("tools.run_comfybench_case.self_evolution_disabled",
                          return_value=True), \
                    patch("tools.run_comfybench_case.official_check",
                          return_value={"comfybench": {
                              "resolved": True, "judgment": "True",
                          }}) as official:
                executor = Executor(output_root)
                run_composition(
                    "x", "add absent object", states, Agent(), executor,
                    root / "results", report, lambda: None, 1,
                    case_resources=[case_input],
                )

        official.assert_called_once()
        self.assertEqual(executor.executions, 2)
        self.assertEqual(len(report["attempts"][0]["stages"]), 2)
        self.assertTrue(report["attempts"][0]["passed"])
        self.assertEqual(report["status"], "passed")


if __name__ == "__main__":
    unittest.main()
