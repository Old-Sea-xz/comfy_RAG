"""Visually grounded placement masks and deterministic artifact checks."""
from __future__ import annotations

import math
import hashlib
import json
import re
import os
import shutil
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from .agentic import AgentOutputError


def region_pixels(size: tuple[int, int], box: list) -> tuple[int, int, int, int]:
    width, height = size
    x, y = int(box[0] * width), int(box[1] * height)
    return x, y, math.ceil(box[2] * width) - x, math.ceil(box[3] * height) - y


def inspect_mask(path: Path, source: Path | None = None) -> dict:
    try:
        with Image.open(path) as image:
            mask = image.convert("L")
        histogram = mask.histogram()
        selected = sum(histogram[1:])
        binary = mask.point(lambda value: 255 if value > 0 else 0)
        errors = []
        if not selected:
            errors.append("empty_mask")
        if source is not None:
            with Image.open(source) as image:
                if image.size != mask.size:
                    errors.append("source_mask_size_mismatch")
        return {"passed": not errors, "errors": errors, "width": mask.width,
                "height": mask.height, "selected_fraction": selected / (mask.width * mask.height),
                "bbox": binary.getbbox()}
    except (OSError, ValueError) as exc:
        return {"passed": False, "errors": ["unreadable_mask"], "error": str(exc)}


def render_regions(size: tuple[int, int], regions: list, protected: list) -> Image.Image:
    width, height = size

    def draw(items):
        canvas = Image.new("L", size)
        painter = ImageDraw.Draw(canvas)
        for item in items:
            if item["shape"] == "polygon":
                painter.polygon([(round(x*(width-1)),round(y*(height-1)))
                                 for x,y in item["points"]], fill=255)
                continue
            x, y, w, h = region_pixels(size, item["box"])
            pixels = (x, y, x + w - 1, y + h - 1)
            if item["shape"] == "ellipse":
                # KJNodes draws inclusive endpoints around an integer center.
                pixels = (x, y, x + 2 * (w // 2), y + 2 * (h // 2))
                painter.ellipse(pixels, fill=255)
            else:
                painter.rectangle(pixels, fill=255)
        return canvas

    return ImageChops.subtract(draw(regions), draw(protected))


def lighting_effect_region_contract(proposal: dict, overall_instruction: str) -> dict:
    """Scope inferred protection outside requested light footprints.

    Explicit exclusions in the original task remain authoritative. A blanket
    inferred hull polygon must not subtract the very illumination envelope
    the task asks to add; the resulting envelope still receives visual review.
    """
    explicit_exclusion = re.search(
        r'must remain black|never edit|do not (?:edit|illuminate|cover|overlap)|'
        r'(?:keep|leave).{0,40}(?:unlit|black)|禁止|不得|不能覆盖|必须.{0,12}黑',
        overall_instruction, re.I)
    if explicit_exclusion:
        return proposal
    return {**proposal, 'protected_regions': [],
            'inferred_protection_regions': proposal.get('protected_regions', []),
            'protection_scope': 'outside_requested_lighting_envelopes',
            'protection_reason': 'Preserve support geometry; permit the requested illumination over surface pixels.'}


def validate_regions(raw: dict) -> None:
    if not isinstance(raw, dict) or raw.get("status") != "propose":
        raise AgentOutputError("mask placement cannot be grounded: " + str(raw.get("reason") if isinstance(raw, dict) else raw))
    confidence = raw.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.7 <= confidence <= 1:
        raise AgentOutputError("mask placement confidence is too low")
    if (not isinstance(raw.get("reason"), str) or not raw["reason"].strip()
            or not isinstance(raw.get("anchors"), list) or not raw["anchors"]):
        raise AgentOutputError("mask placement requires visible anchors and rationale")
    for anchor in raw["anchors"]:
        if isinstance(anchor, str) and anchor.strip():
            continue
        if isinstance(anchor, dict) and isinstance(anchor.get("name"), str) and anchor["name"].strip():
            point = anchor.get("point")
            if (isinstance(point, list) and len(point) == 2 and all(
                    not isinstance(value, bool) and isinstance(value, (int, float))
                    and math.isfinite(value) and 0 <= value <= 1 for value in point)):
                continue
        raise AgentOutputError("mask placement anchor must name visible content and use normalized coordinates")
    for key in ("regions", "protected_regions"):
        items = raw.get(key)
        if not isinstance(items, list) or len(items) > 12 or (key == "regions" and not items):
            raise AgentOutputError("mask placement requires bounded region lists")
        for item in items:
            if not isinstance(item, dict) or item.get("shape") not in {"rectangle", "ellipse", "polygon"}:
                raise AgentOutputError("unsupported mask region shape")
            if item["shape"] == "polygon":
                points = item.get("points")
                if (not isinstance(points,list) or not 3<=len(points)<=32
                        or any(not isinstance(p,list) or len(p)!=2 or any(
                            isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
                            or not 0<=v<=1 for v in p) for p in points)):
                    raise AgentOutputError("polygon points must be bounded normalized source coordinates")
                area=abs(sum(points[i][0]*points[(i+1)%len(points)][1]
                             -points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points))))/2
                if area<=1e-6:raise AgentOutputError("polygon mask has zero area")
                continue
            box = item.get("box")
            if not isinstance(box, list) or len(box) != 4 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1 for value in box
            ) or box[0] >= box[2] or box[1] >= box[3]:
                raise AgentOutputError("mask coordinates must be ordered normalized source coordinates")


def placement_graph(source: Path, size: tuple[int, int], proposal: dict, registry: dict) -> dict:
    if any(item["shape"]=="polygon" for item in proposal["regions"]+proposal["protected_regions"]):
        # Render the reviewed geometry in the runtime input layer. This is an
        # internal derived artifact, not another external benchmark resource.
        # Standard installed nodes still execute and save the stage MASK.
        required={"LoadImage","LoadImageMask","GetImageSize","SolidMask","MaskComposite","MaskToImage","SaveImage"}
        if required-registry.keys():raise AgentOutputError("freeform mask missing installed interfaces: "+", ".join(sorted(required-registry.keys())))
        candidate=render_regions(size,proposal["regions"],proposal["protected_regions"])
        digest=hashlib.sha256(source.read_bytes()+json.dumps(proposal,sort_keys=True).encode()).hexdigest()[:24]
        artifact=source.parent/f"ComfyRouter_placement_geometry_{digest}.png"
        candidate.save(artifact)
        check=inspect_mask(artifact,source)
        if not check["passed"]:raise AgentOutputError("invalid derived geometry mask: "+str(check))
        proposal["derived_mask_artifact"]={"path":str(artifact),"sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),
                                           "source":str(source),"role":"runtime_generated_geometry"}
        return {
            "1":{"class_type":"LoadImage","inputs":{"image":source.name}},
            "2":{"class_type":"GetImageSize","inputs":{"image":["1",0]}},
            "3":{"class_type":"SolidMask","inputs":{"value":0.0,"width":["2",0],"height":["2",1]}},
            "4":{"class_type":"LoadImageMask","inputs":{"image":artifact.name,"channel":"red"},
                 "_meta":{"comfyrouter_adapter":"runtime_generated_geometry_mask"}},
            "5":{"class_type":"MaskComposite","inputs":{"destination":["3",0],"source":["4",0],"x":0,"y":0,"operation":"add"}},
            "6":{"class_type":"MaskToImage","inputs":{"mask":["5",0]}},
            "7":{"class_type":"SaveImage","inputs":{"images":["6",0],"filename_prefix":"ComfyRouter/estimated_placement_mask"}},
        }
    required = {"LoadImage", "GetImageSize", "SolidMask", "MaskComposite", "MaskToImage", "SaveImage"}
    if any(item["shape"] == "ellipse" for item in proposal["regions"] + proposal["protected_regions"]):
        required.add("CreateShapeMask")
    if required - registry.keys():
        raise AgentOutputError("placement mask missing installed interfaces: " + ", ".join(sorted(required - registry.keys())))
    width, height = size
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": source.name}},
        "2": {"class_type": "GetImageSize", "inputs": {"image": ["1", 0]}},
        "3": {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": ["2", 0], "height": ["2", 1]}},
    }
    current = "3"

    def add(class_type, inputs):
        node_id = str(len(graph) + 1)
        graph[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    for key, operation in (("regions", "add"), ("protected_regions", "subtract")):
        for item in proposal[key]:
            x, y, w, h = region_pixels(size, item["box"])
            if item["shape"] == "ellipse":
                shape = add("CreateShapeMask", {"shape": "circle", "frames": 1,
                    "location_x": x + w // 2, "location_y": y + h // 2, "grow": 0,
                    "frame_width": width, "frame_height": height, "shape_width": w, "shape_height": h})
                x, y = 0, 0
            else:
                shape = add("SolidMask", {"value": 1.0, "width": w, "height": h})
            current = add("MaskComposite", {"destination": [current, 0], "source": [shape, 0],
                                               "x": x, "y": y, "operation": operation})
    image = add("MaskToImage", {"mask": [current, 0]})
    add("SaveImage", {"images": [image, 0], "filename_prefix": "ComfyRouter/estimated_placement_mask"})
    return graph


def estimate_placement_plan(agent, executor, graph: dict, implementation: list,
                            source: Path, instruction: str, evidence: dict) -> dict:
    from .repair import graph_transaction

    contract = {"role": "Estimate a bounded edit footprint from actual source pixels.", "rules": [
        "Return status=propose or blocked, confidence in [0,1], reason, anchors, regions, protected_regions.",
        "Each region uses normalized SOURCE coordinates: rectangle/ellipse with box [left,top,right,bottom], or polygon with 3–32 ordered boundary points [[x,y],...].",
        "Choose the smallest coherent placement envelope permitting the requested new object, not an exact absent-object cutout.",
        "Infer placement from visible anchors and the instruction. Never segment the absent target or select the whole support surface.",
        "Use a union of up to 12 regions, subtract protected regions, and account for object attachments/contact area. Use polygons for diagonal beams and freeform protected boundaries; polygons are supported by a verified runtime rasterizer.",
        "Preserve specified content; if its boundary cannot be represented safely, return blocked. Never invent permission to overwrite it.",
    ]}
    decomposition = implementation[0].get("decomposition_step") or {}
    overall_instruction = evidence.get("overall_instruction", "")
    lighting_effect = bool(re.search(r"light[- ]?beams?|glow|illumination|光束|发光|照明",
                                    instruction + "\n" + overall_instruction, re.I))
    lighting_guidance = (
        "For requested light-beam/glow effects, preserving the source structure means retaining its "
        "geometry in the edited result. A narrow continuous beam footprint may overlay illuminated "
        "surface pixels between the visible opening and its exterior footprint. Do not invent a rule "
        "that every hull/support pixel must be black: protect only pixels outside the requested effect "
        "envelope or areas the original task explicitly forbids editing. Subtracting the entire support "
        "must not sever that envelope. Include separately bounded particle/stardust footprints when requested."
    ) if lighting_effect else None
    if lighting_guidance:
        contract["rules"].append(lighting_guidance)
    payload = {"instruction": instruction, "mask_contract": decomposition,
               "overall_instruction": overall_instruction,
               "failure_evidence": evidence.get("setup_failure"),
               "prior_mask_feedback": evidence.get("prior_mask_feedback", []), "_max_output_tokens": 2200,
               **agent._image_evidence_payload([str(source)])}
    region_schema = {"type": "object", "additionalProperties": False,
        "required": ["shape", "box", "points"], "properties": {
            "shape": {"type": "string", "enum": ["rectangle", "ellipse", "polygon"]},
            "box": {"type": "array", "minItems": 4, "maxItems": 4,
                    "items": {"type": "number", "minimum": 0, "maximum": 1}},
            "points":{"type":"array","maxItems":32,"items":{"type":"array","minItems":2,"maxItems":2,
                      "items":{"type":"number","minimum":0,"maximum":1}}}}}
    contract["rules"].append("Every region has shape, box and points. For polygon, box is its bounding box; for rectangle/ellipse use points=[].")
    payload["_response_json_schema"] = {"name": "grounded_placement_mask", "schema": {
        "type": "object", "additionalProperties": False,
        "required": ["status", "confidence", "reason", "anchors", "regions", "protected_regions"],
        "properties": {
            "status": {"type": "string", "enum": ["propose", "blocked"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
            "anchors": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                "required": ["name", "point"], "properties": {"name": {"type": "string"},
                    "point": {"type": "array", "minItems": 2, "maxItems": 2,
                              "items": {"type": "number", "minimum": 0, "maximum": 1}}}}},
            "regions": {"type": "array", "maxItems": 12, "items": region_schema},
            "protected_regions": {"type": "array", "maxItems": 12, "items": region_schema},
        }}}
    contract["rules"].append("Each anchor is {name: visible content, point: [x,y]} in normalized source coordinates.")
    proposal = agent.client.complete_json("mask_region_estimation_agent", contract, payload)
    validate_regions(proposal)
    if lighting_effect and overall_instruction:
        proposal = lighting_effect_region_contract(proposal, overall_instruction)
    with Image.open(source) as image:
        size = image.size
        candidate = render_regions(size, proposal["regions"], proposal["protected_regions"])
        pixels = sum(candidate.histogram()[128:])
        if not 0.0001 <= pixels / (size[0] * size[1]) <= 0.65:
            raise AgentOutputError("estimated placement mask is empty or too broad")
        preview = Image.blend(image.convert("RGB"), Image.new("RGB", size, (255, 64, 64)), 0.5)
        overlay = Image.composite(preview, image.convert("RGB"), candidate)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="comfyrouter-mask-review-") as directory:
        path = Path(directory) / "overlay.png"
        overlay.save(path)
        review = agent.client.complete_json("mask_region_review_agent", {
            "role": "Review the proposed red edit region against the original source and task.",
            "rules": ["Return passed: boolean, reason: non-empty string.",
                      "Approve only if the region supports the complete requested placement and preserves protected content.",
                      "The red overlay is a candidate edit envelope, not the generated object; allow a bounded envelope."]
                      + ([lighting_guidance] if lighting_guidance else [])},
            {"instruction": instruction, "overall_instruction": overall_instruction,
             "mask_contract": decomposition, "proposal": proposal,
             "_max_output_tokens": 1000,
             "_response_json_schema": {"name": "placement_mask_review", "schema": {
                 "type": "object", "additionalProperties": False, "required": ["passed", "reason"],
                 "properties": {"passed": {"type": "boolean"}, "reason": {"type": "string"}}}},
             **agent._image_evidence_payload([str(source), str(path)])})
    if (not isinstance(review, dict) or type(review.get("passed")) is not bool
            or not isinstance(review.get("reason"), str) or not review["reason"].strip()):
        raise AgentOutputError("invalid mask visual review: " + str(review))
    from comfyrouter.execution_contract import current_execution
    context = current_execution()
    preview_advisory = os.getenv('COMFYROUTER_MASK_PREVIEW_ADVISORY', '') == '1'
    if context and context.policy.mask_preview_advisory is not None:
        preview_advisory = context.policy.mask_preview_advisory
    if review['passed'] is False and not preview_advisory:
        raise AgentOutputError("estimated mask failed visual review: " + str(review))
    # Repair evidence may live under the run's output directory. ComfyUI
    # LoadImage/LoadImageMask enumerate the input directory, not that output.
    input_root = Path(getattr(executor, 'comfyui_input_root', source.parent))
    input_root.mkdir(parents=True, exist_ok=True)
    runtime_source = source
    if source.parent.resolve() != input_root.resolve():
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:24]
        runtime_source = input_root / f'ComfyRouter_mask_source_{digest}{source.suffix}'
        shutil.copy2(source, runtime_source)
        proposal['runtime_source_artifact'] = {
            'original': str(source), 'path': str(runtime_source),
            'sha256': hashlib.sha256(runtime_source.read_bytes()).hexdigest()}
    candidate_graph = placement_graph(runtime_source, size, proposal, executor.ensure_validator().node_registry)
    if runtime_source != source or proposal.get("derived_mask_artifact"):
        executor.validator = None  # New runtime artifact must enter the live filename enum.
    executor.static_validate(candidate_graph)
    plan = agent._normalize_generation_plan({"skill_ids": [], "edits": graph_transaction(graph, candidate_graph),
                                             "rationale": proposal["reason"]})
    plan.update({"requires_composition_replan": False, "selected_workflow": None,
                 "repair_transaction": True, "mask_fallback": {"proposal": proposal, "review": review,
                     "preview_gate": "advisory_final_quality_evaluates_pixels" if preview_advisory else "blocking",
                     "source_size": size, "selected_fraction": pixels / (size[0] * size[1]),
                     "method": "visually_grounded_placement_envelope"}})
    return plan
