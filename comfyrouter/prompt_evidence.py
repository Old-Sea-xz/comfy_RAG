"""Bound runtime telemetry sent to Agents; full execution reports stay on disk."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .execution_diagnostics import normalize_legacy_execution_error, legacy_error_details


_GRAPHS = {"graph", "current_graph", "base_graph", "workflow_api", "final_graph",
           "preflight_graph", "previous_plan", "edits", "node_registry",
           "installed_node_interfaces", "_response_json_schema", "partial_graph",
           "runtime_graph", "post_edit_graph", "initial_current_graph"}
_EVIDENCE = {"repair_evidence", "execution", "execution_history", "attempt_history",
             "attempts", "trace", "stages", "resource_monitor", "resource_usage",
             "base_execution_history", "failure_analyses", "diagnostic_outputs",
             "current_execution", "execution_result", "attempt_trace", "full_trace",
             "complete_initial_trace", "validation_feedback", "retrieval_feedback", "ordered_stages"}
_FILE_OUTPUTS = {"outputs", "output_files", "diagnostic_outputs", "selected_stage_result_files"}


def _encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_encoded(value).encode()).hexdigest()


def _indices(count: int, limit: int = 3) -> list[int]:
    return sorted({round(i * (count - 1) / (limit - 1)) for i in range(limit)}) if count else []


def _pose_summary(value: Any) -> dict:
    frames = value
    # ComfyUI wraps OPENPOSE_JSON in an output list, often containing a single
    # serialized array. Unwrap both encodings without sending raw coordinates.
    if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], str):
        frames = frames[0]
    if isinstance(frames, str):
        try:
            frames = json.loads(frames)
        except (ValueError, TypeError):
            frames = None
    if isinstance(frames, dict):
        frames = [frames]
    result = {"summary_type": "openpose_frames", "sha256": _digest(value),
              "raw_coordinates_omitted": True}
    if not isinstance(frames, list):
        result["parse_status"] = "unrecognized_pose_encoding"
        return result
    result["frame_count"] = len(frames)
    result["frames_with_people"] = sum(
        bool(frame.get("people")) for frame in frames if isinstance(frame, dict))
    result["representative_frames"] = []
    for index in _indices(len(frames)):
        frame = frames[index]
        if not isinstance(frame, dict):
            continue
        people = frame.get("people", [])
        result["representative_frames"].append({
            "frame_index": index,
            "canvas_width": frame.get("canvas_width"),
            "canvas_height": frame.get("canvas_height"),
            "people_count": len(people) if isinstance(people, list) else None,
            "keypoints": [{key: {"value_count": len(points),
                                  "nonzero_confidence_count": sum(
                                      isinstance(c, (int, float)) and c > 0
                                      for c in points[2::3])}
                           for key, points in person.items()
                           if "keypoints" in key and isinstance(points, list)}
                          for person in (people[:4] if isinstance(people, list) else [])
                          if isinstance(person, dict)],
        })
    return result


def compact_agent_payload(payload: dict) -> dict:
    """Copy only transport evidence, preserving executable graphs and errors.

    Repeated bulky runtime fields reference the first summary by JSON path.
    No document is loaded and neither the source report nor caller is mutated.
    """
    seen: dict[str, str] = {}

    def walk(value: Any, path: str, evidence: bool = False, raw_output: bool = False):
        if isinstance(value, dict):
            # ComfyUI also embeds executable graphs at unnamed positions,
            # e.g. history.prompt[2]. Their `samples` inputs are node links,
            # not resource-monitor samples.
            nodes = [item for key, item in value.items() if not str(key).startswith("__")]
            if nodes and all(isinstance(node, dict)
                             and isinstance(node.get("class_type"), str)
                             and isinstance(node.get("inputs"), dict) for node in nodes):
                return value
            result = {}
            for key, item in value.items():
                child_path = f"{path}/{key}"
                if key in _GRAPHS:
                    result[key] = item
                    continue
                active = evidence or key in _EVIDENCE
                if (key in _FILE_OUTPUTS and (active or key != "outputs") and isinstance(item, list)
                        and len(item) > 12 and all(isinstance(path, str) for path in item)):
                    paths = [item[index] for index in _indices(len(item))]
                    if value.get("primary_output") in item and value["primary_output"] not in paths:
                        paths.append(value["primary_output"])
                    result[key] = {"summary_type": "output_files", "file_count": len(item),
                                   "representative_paths": paths, "sha256": _digest(item), "paths_omitted": True}
                    continue
                if active and key == "error":
                    result[key] = normalize_legacy_execution_error(value)
                    if result[key] != item and not isinstance(value.get("history"), dict):
                        details = legacy_error_details(value)
                        if details is not None:
                            result["runtime_error_details"] = details
                    continue
                resource_rows = isinstance(item, list) and all(isinstance(row, dict) for row in item)
                bulky = active and (key == "openpose_json" or
                                   key in {"samples", "resource_samples"} and resource_rows)
                if bulky and isinstance(item, dict) and ("summary_type" in item or "evidence_ref" in item):
                    result[key] = item
                    continue
                if bulky:
                    digest = _digest(item)
                    if digest in seen:
                        result[key] = {"evidence_ref": seen[digest], "sha256": digest}
                        continue
                    seen[digest] = child_path
                    if key == "openpose_json":
                        result[key] = _pose_summary(item)
                    else:
                        result[key] = {"summary_type": "resource_samples", "sample_count": len(item),
                                       "sha256": digest,
                                       "representative_samples": [item[i] for i in _indices(len(item), 5)]}
                        # Retain peaks, including an exceptional short-lived spike.
                        numeric_keys = {k for row in item if isinstance(row, dict)
                                        for k, v in row.items()
                                        if isinstance(v, (float, int)) and not isinstance(v, bool)}
                        result[key]["maxima"] = {
                            k: max(row[k] for row in item if isinstance(row, dict)
                                   and isinstance(row.get(k), (float, int))) for k in numeric_keys}
                else:
                    result[key] = walk(item, child_path, active, raw_output or active and key in {"outputs", "current_inputs"})
            return result
        if isinstance(value, list):
            if raw_output and len(value) > 128 and all(isinstance(v, (int, float)) for v in value):
                return {"summary_type": "numeric_array", "value_count": len(value),
                        "sha256": _digest(value), "values_omitted": True}
            return [walk(item, f"{path}/{i}", evidence, raw_output) for i, item in enumerate(value)]
        if raw_output and isinstance(value, str) and len(value) > 8192:
            return {"summary_type": "large_runtime_output", "character_count": len(value),
                    "sha256": _digest(value), "preview": value[:2048], "content_omitted": True}
        return value

    return walk(payload, "")


def compact_prompt_history(messages: list, limit: int = 120_000) -> list:
    """Compact old serialized user evidence and bound cached turn accumulation."""
    result = []
    prefix = "Current request suffix:\n"
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            result.append(message)
            continue
        contents = []
        for content in message.get("content", []):
            text = content.get("text", "") if isinstance(content, dict) else ""
            if text.startswith(prefix):
                try:
                    original = json.loads(text[len(prefix):])
                    compact = compact_agent_payload(original)
                    if compact != original:
                        content = {**content, "text": prefix + _encoded(compact)}
                except (ValueError, TypeError):
                    pass
            contents.append(content)
        result.append({**message, "content": contents})
    omitted = 0
    while result and len(_encoded(result)) > limit:
        # Drop complete oldest turns; the authoritative current request keeps
        # its explicit attempt history and current executable graph.
        # A native result turn can contain reasoning, a function call and its
        # output. Evict it as a unit so replay never leaves an orphan call_id.
        count = 1
        if result[0].get("role") == "user":
            while count < len(result) and result[count].get("role") != "user":
                count += 1
        result = result[count:]
        omitted += count
    if omitted:
        result.insert(0, {"role": "user", "content": [{"type": "input_text", "text":
            f"{omitted} older cached messages omitted to bound context. Use the current request's explicit attempt history as authoritative evidence."}]})
    return result
