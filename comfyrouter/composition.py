"""Typed multi-stage workflow-composition planning primitives.

Composition never selects a model directly.  Each stage is routed through the
normal task and model classifiers, then executed with its own bounded attempt
budget.  This module validates the plan before any workflow is loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class CompositionPlanError(ValueError):
    pass


@dataclass(frozen=True)
class CompositionStage:
    stage_id: str
    purpose: str
    instruction: str
    input_type: str
    input_types: tuple[str, ...]
    output_type: str
    modality: str
    depends_on: tuple[str, ...]


# Public operation aliases emitted by the generation agent.  The planner is
# deliberately conservative: unknown operations are rejected instead of being
# silently treated as a prompt-only edit.
_OPERATION_TYPES = {
    "text_to_image": ("TEXT", "IMAGE", "t2i"),
    "image_to_image": ("IMAGE", "IMAGE", "i2i"),
    "image_to_video": ("IMAGE", "VIDEO", "i2v"),
    "text_to_video": ("TEXT", "VIDEO", "t2v"),
    "video_to_video": ("VIDEO", "VIDEO", "v2v"),
    "video_interpolation": ("VIDEO", "VIDEO", "v2v"),
    "video_frame_interpolation": ("VIDEO", "VIDEO", "v2v"),
    "video_upscale": ("VIDEO", "VIDEO", "v2v"),
    "video_super_resolution": ("VIDEO", "VIDEO", "v2v"),
    "image_upscale": ("IMAGE", "IMAGE", "i2i"),
    "text_overlay": ("IMAGE", "IMAGE", "i2i"),
    "image_text_overlay": ("IMAGE", "IMAGE", "i2i"),
    "watermark_overlay": ("IMAGE", "IMAGE", "i2i"),
    "background_replace": ("IMAGE", "IMAGE", "i2i"),
    "image_background_replacement": ("IMAGE", "IMAGE", "i2i"),
    "foreground_segmentation": ("IMAGE", "IMAGE", "i2i"),
    "mask_generation": ("IMAGE", "MASK", "i2i"),
    "text_render": ("IMAGE", "IMAGE", "i2i"),
    "handwriting_render": ("IMAGE", "IMAGE", "i2i"),
    "video_background_replace": ("VIDEO", "VIDEO", "v2v"),
    "video_background_replacement": ("VIDEO", "VIDEO", "v2v"),
    "video_foreground_segmentation": ("VIDEO", "VIDEO", "v2v"),
    "motion_control": ("TEXT", "VIDEO", "t2v"),
    "trajectory_control": ("VIDEO", "VIDEO", "i2v"),
}


def typed_stage_proposal(raw: Any) -> list[dict[str, Any]]:
    """Convert the agent's compact operation proposal to typed stages.

    The generation contract intentionally keeps proposals short.  Execution
    requires explicit media handoffs, so this conversion is the single gate
    between the two contracts.
    """
    if not isinstance(raw, list):
        raise CompositionPlanError("composition proposal must be a list")
    result = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CompositionPlanError(f"composition stage {index + 1} must be an object")
        operation = item.get("operation")
        if operation not in _OPERATION_TYPES:
            raise CompositionPlanError(f"unsupported composition operation: {operation}")
        input_type, output_type, modality = _OPERATION_TYPES[operation]
        result.append({
            "stage_id": item.get("stage_id") or f"stage_{index + 1:02d}",
            "purpose": item.get("purpose"),
            "instruction": item.get("instruction") or item.get("purpose"),
            "input_type": item.get("input_type") or input_type,
            "output_type": item.get("output_type") or output_type,
            "modality": item.get("modality") or modality,
            "operation": operation,
        })
    return result


def validate_composition_plan(raw: Any) -> tuple[CompositionStage, ...]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise CompositionPlanError("composition requires at least two ordered stages")
    stages = []
    seen = set()
    output_types: dict[str, str] = {}
    available: dict[str, list[str]] = {"IMAGE": [], "VIDEO": [], "TEXT": [], "MASK": []}
    allowed_media = {"TEXT", "IMAGE", "VIDEO", "MASK", "IMAGE_FRAMES", "VIDEO_FRAMES"}
    allowed_modalities = {"t2i", "i2i", "t2v", "i2v", "v2v"}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CompositionPlanError(f"stage {index + 1} must be an object")
        stage_id = item.get("stage_id") or f"stage_{index + 1:02d}"
        purpose = item.get("purpose")
        instruction = item.get("instruction") or purpose
        input_type = item.get("input_type")
        input_types_raw = item.get("input_types", [input_type])
        output_type = item.get("output_type")
        modality = item.get("modality")
        if not all(isinstance(x, str) and x.strip()
                   for x in (stage_id, purpose, instruction, input_type, output_type, modality)):
            raise CompositionPlanError(f"stage {index + 1} is missing typed routing fields")
        if stage_id in seen:
            raise CompositionPlanError(f"duplicate stage_id: {stage_id}")
        if input_type not in allowed_media or output_type not in allowed_media:
            raise CompositionPlanError(f"stage {stage_id} has an unsupported media type")
        if (not isinstance(input_types_raw, (list, tuple)) or not input_types_raw
                or not all(value in allowed_media for value in input_types_raw)
                or input_type not in input_types_raw):
            raise CompositionPlanError(
                f"stage {stage_id} input_types must include input_type and contain supported media types"
            )
        input_types = tuple(input_types_raw)
        if modality not in allowed_modalities:
            raise CompositionPlanError(f"stage {stage_id} has an unsupported modality")
        declared = item.get("depends_on")
        if declared is not None and (
            not isinstance(declared, (list, tuple))
            or not all(isinstance(value, str) and value in seen for value in declared)
        ):
            raise CompositionPlanError(
                f"stage {stage_id} depends_on must reference earlier stage IDs"
            )
        dependencies = tuple(dict.fromkeys(
            declared if declared is not None else available.get(input_type, [])
        ))
        mismatched = [
            dependency for dependency in dependencies
            if output_types.get(dependency) not in input_types
        ]
        if mismatched:
            raise CompositionPlanError(
                f"stage {stage_id} dependency media mismatch: " + ", ".join(mismatched)
            )
        # Explicit empty dependencies declare an external-media root. Several
        # independent image edits may share the same case resource; requiring
        # the second root to consume the first output invents a dependency.
        if input_type != "TEXT" and not dependencies and stages and declared is None:
            raise CompositionPlanError(f"stage {stage_id} has no {input_type} dependency")
        for dependency in dependencies:
            for outputs in available.values():
                if dependency in outputs:
                    outputs.remove(dependency)
        seen.add(stage_id)
        output_types[stage_id] = output_type
        available[output_type].append(stage_id)
        stages.append(CompositionStage(stage_id, purpose, instruction,
                                       input_type, input_types, output_type, modality, dependencies))
    return tuple(stages)
