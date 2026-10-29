from __future__ import annotations

import copy
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from comfyrouter.agentic import AgentOrchestrator, AgentOutputError, AgentTransportError
from comfyrouter.types import RequirementSpec
from tests.test_composition_forward_diagnosis import conditioning_attribution, orchestrator
from tools.run_comfybench_case import (
    composition_repair_targets, repair_mode_for_context, run_composition,
)


def independent_repair():
    return {
        "stage_id": "stage_02", "evidence_basis": "graph_contract", "node_ids": ["1"],
        "contract": "The face must become a cartoon avatar.",
        "upstream_independence": "Correcting the mask cannot remove the downstream cartoon exclusion.",
        "evidence": ["Node 1 negative text excludes cartoon."],
        "failure_class": "quality", "root_causes": ["The prompt forbids the requested cartoon style."],
        "next_action": "parameter_retry", "repair_focus": ["Remove cartoon from the negative text."],
    }


def diagnosis():
    return {
        "earliest_failing_stage_id": "stage_01", "repair_stage_id": "stage_01",
        "contributing_stage_ids": [],
        "stage_assessments": [
            {"stage_id": "stage_01", "status": "fail", "observations": ["Wrong face mask."],
             "reason": "The background was selected."},
            {"stage_id": "stage_02", "status": "fail", "observations": ["Cartoon is excluded."],
             "reason": "The prompt contradicts the task independently of the mask."},
        ],
        "conditioning_attribution": conditioning_attribution(),
        "rationale": "Fix the mask and independently contradictory downstream prompt together.",
        "evidence": ["The mask selects the background and node 1 excludes cartoon."],
        "failure_class": "quality", "root_causes": ["Wrong mask tracking prompt."],
        "next_action": "parameter_retry", "repair_focus": ["Select the foreground face."],
        "needs_search": False, "search_queries": [],
        "independent_stage_repairs": [independent_repair()],
    }


def descriptors():
    return [
        {"stage_id": "stage_01", "executed_graph": {}, "output_files": []},
        {"stage_id": "stage_02", "executed_graph": {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "cartoon"}},
        }, "output_files": []},
    ]


class TestIndependentCompositionDiagnosis(unittest.TestCase):
    def diagnose(self, response=None, stages=None):
        return orchestrator(response or diagnosis()).diagnose_composition_failure(
            instruction="Create a face mask and cartoonize that face.",
            stages=stages or descriptors(), final_evaluation={}, media_paths=[],
            include_repair_decision=True,
        )

    def test_independent_downstream_failure_survives_normalization(self):
        response = diagnosis()
        result = self.diagnose(response)
        self.assertEqual(result["stage_assessments"][1]["status"], "fail")
        self.assertEqual(result["independent_stage_repairs"], response["independent_stage_repairs"])
        self.assertNotIn("diagnosis_protocol_normalizations", result)

    def test_blocked_visual_judgment_can_still_have_independent_graph_repair(self):
        response = diagnosis()
        response["stage_assessments"][1]["status"] = "deferred_due_to_upstream_failure"
        self.assertEqual(len(self.diagnose(response)["independent_stage_repairs"]), 1)

    def test_unsupported_downstream_visual_failure_is_still_deferred(self):
        response = diagnosis()
        response["independent_stage_repairs"] = []
        result = self.diagnose(response)
        self.assertEqual(result["stage_assessments"][1]["status"], "deferred_due_to_upstream_failure")

    def test_rejects_unanchored_independent_evidence(self):
        variants = [
            ("node_ids", ["missing"], "existing executed graph"),
            ("evidence", [], "requires evidence"),
            ("upstream_independence", "", "requires upstream_independence"),
            ("evidence_basis", "final_visual_judgment", "evidence basis"),
            ("next_action", "routing_correction", "parameter_retry or topology_repair"),
        ]
        for key, value, error in variants:
            with self.subTest(key=key):
                response = diagnosis()
                response["independent_stage_repairs"][0][key] = value
                with self.assertRaisesRegex(AgentOutputError, error):
                    self.diagnose(response)

    def test_rejects_duplicate_targets_and_reopening_passed_stages(self):
        response = diagnosis()
        response["independent_stage_repairs"] *= 2
        with self.assertRaisesRegex(AgentOutputError, "unique stages"):
            self.diagnose(response)
        response = diagnosis()
        response["stage_assessments"][1]["status"] = "pass"
        with self.assertRaisesRegex(AgentOutputError, "passed stage"):
            self.diagnose(response)
        response = diagnosis()
        response["independent_stage_repairs"][0]["stage_id"] = "stage_01"
        with self.assertRaisesRegex(AgentOutputError, "after the earliest failure"):
            self.diagnose(response)

    def test_decoded_frame_violation_is_independent_of_mask_content(self):
        response = diagnosis()
        response["independent_stage_repairs"][0].update(
            evidence_basis="decoded_media_contract", node_ids=[],
            contract="At most 3 seconds and 100 frames.",
            upstream_independence="Mask pixel content cannot change the frame expansion topology.",
            evidence=["The decoded video has 216 frames at 24 fps, lasting 9 seconds."],
            root_causes=["Video batches are expanded when composited."],
            next_action="topology_repair", repair_focus=["Join the edited batches before masked compositing."],
        )
        stages = descriptors()
        with self.assertRaisesRegex(AgentOutputError, "decoded stage metadata"):
            self.diagnose(response, stages)
        stages[1]["video_metadata"] = {"num_frames": 216, "fps": 24, "duration": 9}
        self.assertEqual(self.diagnose(response, stages)["independent_stage_repairs"][0]["next_action"],
                         "topology_repair")

    def test_execution_error_must_be_concrete(self):
        response = diagnosis()
        response["independent_stage_repairs"][0].update(evidence_basis="execution_error", node_ids=[])
        stages = descriptors()
        with self.assertRaisesRegex(AgentOutputError, "concrete stage execution error"):
            self.diagnose(response, stages)
        stages[1]["execution"] = {"status": "execution_failed", "error": "Required mask connection missing"}
        self.assertEqual(len(self.diagnose(response, stages)["independent_stage_repairs"]), 1)

    def test_additional_stage_keeps_its_own_topology_repair_mode(self):
        response = diagnosis()
        response["independent_stage_repairs"][0]["next_action"] = "topology_repair"
        states = [{"typed": SimpleNamespace(stage_id=s)} for s in ("stage_01", "stage_02")]
        targets = composition_repair_targets(response, states, descriptors())
        scoped = targets[1]["diagnosis"]
        mode = repair_mode_for_context(
            diagnosis={"stage_failure_diagnosis": scoped,
                       "failure_analysis": {"next_action": scoped["next_action"], "failure_class": "quality"}},
            execution_failed=True,
        )
        self.assertTrue(mode["allow_topology_repair"])
        self.assertEqual(targets[0]["diagnosis"]["next_action"], "parameter_retry")

    def test_independent_handoff_failure_does_not_escalate_primary_parameter_repair(self):
        response = diagnosis()
        response["independent_stage_repairs"][0].update(
            failure_class="handoff", next_action="topology_repair",
        )
        states = [{"typed": SimpleNamespace(stage_id=s)} for s in ("stage_01", "stage_02")]
        primary = composition_repair_targets(response, states, descriptors())[0]["diagnosis"]
        mode = repair_mode_for_context(
            diagnosis={"stage_failure_diagnosis": primary,
                       "failure_analysis": {"next_action": "parameter_retry", "failure_class": "quality"}},
            execution_failed=True,
        )
        self.assertFalse(mode["allow_topology_repair"])


class TestIndependentRepairExecution(unittest.TestCase):
    def run_scenario(self, *, fail_second=False, never_pass=False, max_attempts=3):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            output_root = root / "executor"
            output_root.mkdir()
            (root / "input").mkdir()
            names = [f"a{a}s{s}.png" for a in range(1, 4) for s in (1, 2)]
            for name in names:
                (output_root / name).write_bytes(b"mock media")

            class Executor:
                validator = None
                output_root = root / "executor"

                def __init__(self):
                    self.names = iter(names)
                    self.executed = []

                def execute(self, graph):
                    self.executed.append(copy.deepcopy(graph))
                    result = {"status": "success", "outputs": [next(self.names)]}
                    return SimpleNamespace(as_dict=lambda: result)

                def static_validate(self, graph, **kwargs):
                    return None

            states = []
            for index in range(2):
                graph = {"1": {"class_type": "LoadImage", "inputs": {
                    "image": "INPUT_IMAGE.png", "strength": 1,
                }}}
                typed = SimpleNamespace(
                    stage_id=f"stage_0{index+1}", instruction="generate" if index == 0 else "edit",
                    input_type="TEXT" if index == 0 else "IMAGE", output_type="IMAGE",
                    depends_on=() if index == 0 else ("stage_01",),
                )
                states.append({
                    "typed": typed, "spec": RequirementSpec("t2i" if index == 0 else "i2i", "A", ("A/edit",)),
                    "implementation": {"operation": f"A/stage{index+1}"},
                    "base_graph": copy.deepcopy(graph), "graph": graph,
                    "handoff_inputs": [] if index == 0 else [("1", "image")],
                    "handoff_inputs_by_type": {} if index == 0 else {"IMAGE": [("1", "image")]},
                })
            executor = Executor()
            planned = []
            call_counts_at_execution = []

            def generate(agent, executor, spec, implementations, *, graph, repair_evidence, **kwargs):
                stage_id = repair_evidence["forward_stage_diagnosis"]["repair_stage_id"]
                planned.append(stage_id)
                if fail_second and stage_id == "stage_02":
                    raise AgentTransportError("Repair service unavailable")
                return {"skill_ids": [], "requires_composition_replan": False,
                        "edits": [{"action": "set_param", "target": "1", "key": "strength",
                                   "value": graph["1"]["inputs"]["strength"] + 1}]}

            def checked(case_id, result, attempt_dir):
                call_counts_at_execution.append(len(planned))
                return {"comfybench": {"resolved": not never_pass and len(call_counts_at_execution) == 2}}

            agent = SimpleNamespace(diagnose_composition_failure=lambda **kwargs: diagnosis())
            report = {"run_id": "test", "attempts": [], "self_evolution": {"enabled": False}}
            replacements = {
                "COMFYUI_INPUT": root / "input",
                "copy_case_resources": lambda *args, **kwargs: [],
                "release_comfyui_models": lambda *args, **kwargs: {"status": "released"},
                "require_runtime_resources_bound": lambda *args, **kwargs: None,
                "inspect_runtime_media_inputs": lambda *args, **kwargs: {},
                "official_check": checked,
                "generate_plan_with_repair": generate,
                "self_evolution_disabled": lambda: True,
            }
            releases = None
            for name, value in replacements.items():
                mocked = stack.enter_context(patch(f"tools.run_comfybench_case.{name}", value))
                if name == "release_comfyui_models":
                    releases = stack.enter_context(patch(f"tools.run_comfybench_case.{name}", wraps=value))
            run_composition("test", "Generate then edit", states, agent, executor,
                            root / "results", report, lambda: None, max_attempts)
            return report, states, executor.executed, planned, call_counts_at_execution, releases.call_count

    def test_both_repairs_are_ready_before_the_next_execution(self):
        report, states, executed, planned, observed, releases = self.run_scenario()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(planned, ["stage_01", "stage_02"])
        self.assertEqual(observed, [0, 2])
        self.assertEqual(len(executed), 4)
        self.assertEqual([g["1"]["inputs"]["strength"] for g in executed], [1, 1, 2, 2])
        self.assertEqual(releases, 8)  # before and after each executed stage
        attempt = report["attempts"][0]
        self.assertEqual(attempt["repair_batch"]["validated_stage_ids"], ["stage_01", "stage_02"])
        self.assertEqual(attempt["cache_invalidation"]["invalidated_stage_ids"], ["stage_01", "stage_02"])
        self.assertEqual(attempt["stages"][1]["repair_decision"], "independent_evidence_repair_stage")

    def test_failed_second_plan_rolls_back_without_partial_execution(self):
        report, states, executed, planned, observed, releases = self.run_scenario(fail_second=True)
        self.assertEqual(report["status"], "agent_transport_failed")
        self.assertEqual(len(executed), 2)
        self.assertEqual(len(report["attempts"]), 1)
        self.assertEqual([s["graph"]["1"]["inputs"]["strength"] for s in states], [1, 1])
        self.assertEqual(report["attempts"][0]["repair_batch"]["status"], "rejected_before_execution")
        self.assertNotIn("cache_invalidation", report["attempts"][0])

    def test_three_execution_rounds_remain_the_absolute_limit(self):
        report, states, executed, planned, observed, releases = self.run_scenario(never_pass=True)
        self.assertEqual(len(report["attempts"]), 3)
        self.assertEqual(len(executed), 6)
        self.assertEqual(planned, ["stage_01", "stage_02"] * 2)
        self.assertNotIn("repair_batch", report["attempts"][-1])
        self.assertEqual(report["status"], "quality_not_passed")

    def test_terminal_round_stops_without_diagnosis_or_an_extra_repair(self):
        report, states, executed, planned, observed, releases = self.run_scenario(never_pass=True, max_attempts=1)
        self.assertEqual(len(report["attempts"]), 1)
        self.assertEqual(planned, [])
        self.assertEqual(len(executed), 2)
        self.assertEqual(report["attempts"][0]["terminal_diagnosis"],
                         "skipped_no_remaining_execution_budget")
        self.assertNotIn("stage_failure_diagnosis", report["attempts"][0])


if __name__ == "__main__":
    unittest.main()
