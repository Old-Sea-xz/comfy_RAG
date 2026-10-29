"""Shared preflight repair policy and transactional graph reconstruction."""
from __future__ import annotations

import copy
import os
from collections import Counter
from typing import Any

from . import config
from .agentic import AgentOrchestrator, AgentOutputError, VIDEO_TIMING_POLICY
from .closed_loop import apply_graph_edits, validate_no_new_dead_nodes
from .prompt_interfaces import scoped_interface_classes


def repair_plan_limit() -> int:
    """Planning failures do not consume the three inference attempts."""
    return max(3, min(12, int(os.getenv("COMFYROUTER_REPAIR_PLAN_LIMIT", "6"))))


def failure_kind(error: str) -> str:
    text = error.casefold()
    if "context_length_exceeded" in text:
        return "context_limit"
    if "llm provider failure" in text:
        return "provider_failure"
    if "non-json" in text:
        return "response_format"
    if any(token in text for token in ("requires a graph edit", "at least one", "blocked")):
        return "missing_executable_repair"
    if "composition replan" in text:
        return "composition_reconstruction"
    if "temporal constraint" in text:
        return "temporal_contract"
    if any(token in text for token in ("handoff", "loader", "missing_required_input", "unbound_media_placeholder")):
        return "media_interface"
    return "graph_validation"


def graph_transaction(before: dict, after: dict) -> list[dict]:
    """Compile a complete candidate into edits accepted by the same validator.

    Validate the final transaction, not invalid intermediate graph states.
    Parameter deletion has no typed edit, so replace that node explicitly.
    """
    if not isinstance(after, dict) or not after or any(
        not isinstance(node_id, str) or not isinstance(node, dict)
        or not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), dict)
        for node_id, node in after.items()
    ):
        raise AgentOutputError("repair candidate must be a complete ComfyUI API graph")
    changed = {
        node_id for node_id in before.keys() & after.keys()
        if before[node_id].get("class_type") != after[node_id].get("class_type")
        or before[node_id].get("inputs", {}).keys() - after[node_id].get("inputs", {}).keys()
    }
    removed = before.keys() - after.keys() | changed
    added = after.keys() - before.keys() | changed
    edits = [{"action": "remove_node", "target": node_id,
              "reason": "Replace or remove the obsolete node in the repair transaction"}
             for node_id in sorted(removed)]
    edits += [{"action": "add_node", "node_id": node_id, "node": copy.deepcopy(after[node_id]),
               "reason": "Install the explicitly supplied repair node"}
              for node_id in sorted(added)]
    for node_id in sorted(after.keys() - added):
        for key, value in after[node_id]["inputs"].items():
            # Removing a source deletes its outgoing links; restore these even
            # when the requested connection is textually unchanged.
            edge = isinstance(value, list) and len(value) == 2 \
                and isinstance(value[0], str) and isinstance(value[1], int)
            restore = edge and value[0] in removed
            if value != before[node_id].get("inputs", {}).get(key) or restore:
                edits.append({"action": "set_param", "target": node_id, "key": key,
                              "value": copy.deepcopy(value),
                              "reason": "Apply the candidate's declared input or connection"})
    # Metadata is preserved for existing nodes; it cannot secretly change a
    # graph's runtime adapter contract through reconstruction.
    return edits


def reconstruct_plan(agent, executor, spec, implementation, graph: dict, evidence: dict,
                     *, instruction: str, interfaces: list[dict]) -> dict:
    """Repair this stage from full interfaces, without publishing/evolving assets."""
    if config.base_only_ablation():
        raise AgentOutputError("Base-only ablation forbids graph reconstruction")
    contract = {
        "role": "Repair the selected workflow's executable graph from concrete failures.",
        "video_timing_policy": VIDEO_TIMING_POLICY,
        "required_output": ["workflow_api", "rationale"],
        "rules": [
            "Return one JSON object containing the complete repaired workflow_api and a concise rationale.",
            "Use only the supplied installed node interfaces, enum values, models and fixed stage media boundary.",
            "Preserve the task, selected model and all unaffected paths. A composed stage may have multiple internal nodes, but cannot silently change its business operation.",
            "A request to recompose must include the executable graph; a prose proposal or empty edits cannot resolve the failure.",
            "Use all previous errors. Remove obsolete nodes explicitly and keep every required external loader connected to its consumer.",
            "For an upstream white-selects PNG mask use LoadImageMask with channel=red and MASK_INPUT.png, not the opaque alpha channel. For a per-frame video mask use MASK_INPUT.mp4 in a separate video loader marked _meta.comfyrouter_adapter=external_mask_video_loader, then decode frames and convert their red channel to MASK. Connect the mask to the edit or a composite preserving the original outside the edit region; do not add an unused loader.",
            "For memory failures preserve the entire video's duration, frame order and audio. A frame cap alone is truncation, not chunk processing.",
            "Do not publish Skills, write experience, select unlisted models, or claim execution success.",
        ],
    }
    request_evidence = dict(evidence)
    if request_evidence.get("installed_node_interfaces") == interfaces:
        # The complete interfaces already have their own request field.
        # Keep distinct diagnostic interfaces, but send identical tables once.
        request_evidence.pop("installed_node_interfaces")
    payload = {"instruction": instruction, "implementation_plan": implementation,
               "current_graph": graph, "repair_evidence": request_evidence,
               "installed_node_interfaces": interfaces,
               "_max_output_tokens": 8000, "_timeout_seconds": 180}
    raw = agent.client.complete_json("workflow_repair_agent", contract, payload)
    candidate = raw.get("workflow_api")
    if not isinstance(candidate, dict):
        raise AgentOutputError("workflow reconstruction supplied no graph: " + str(
            raw.get("reason") or raw.get("rationale") or raw.get("missing_interfaces") or raw
        )[:2000])
    edits = graph_transaction(graph, candidate)
    supplied_classes = {item["class_type"] for item in interfaces}
    interface_scope = scoped_interface_classes(payload)
    if interface_scope is not None:
        supplied_classes &= interface_scope
    unknown = {node["class_type"] for node in candidate.values()} - supplied_classes
    if unknown:
        raise AgentOutputError("reconstruction used unsupplied node interfaces: " + ", ".join(sorted(unknown)))
    model_fields = {"ckpt_name", "unet_name", "vae_name", "clip_name", "clip_name1",
                    "clip_name2", "clip_name3", "lora_name", "model_name"}
    def model_bindings(workflow):
        return Counter((key, value) for node in workflow.values()
                       for key, value in node.get("inputs", {}).items()
                       if key in model_fields and isinstance(value, str))
    if model_bindings(graph) - model_bindings(candidate):
        raise AgentOutputError("reconstruction must preserve the selected model bindings")
    if not edits:
        raise AgentOutputError("workflow reconstruction returned an unchanged graph")
    # This is still the existing review/apply/validate path, not a bypass.
    patched = apply_graph_edits(graph, edits, executor.validator,
                                validate_after_each=False, allow_external_placeholders=True)
    executor.static_validate(patched, allow_external_placeholders=True)
    validate_no_new_dead_nodes(graph, patched, executor.validator)
    plan = AgentOrchestrator._normalize_generation_plan({
        "skill_ids": [], "edits": edits, "rationale": str(raw.get("rationale") or "Reconstruct the failing stage"),
    })
    AgentOrchestrator._validate_generation_plan(plan, set(), require_edit=True)
    AgentOrchestrator._validate_bound_profile_plan(implementation, graph, plan)
    plan.update({"requires_composition_replan": False, "selected_workflow": None,
                 "repair_strategy": "validated_stage_reconstruction", "repair_transaction": True})
    return plan
