"""Keep exact node contracts while excluding unrelated uploaded media and UI text."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .execution_diagnostics import normalize_legacy_execution_error


_MEDIA_FIELDS = {"image", "video", "file", "audio", "audio_file"}
_LIVE_GRAPHS = {"current_graph", "current_candidate_graph", "post_edit_graph"}
_MEDIA_LISTS = {"case_resources", "visual_input_files", "media_paths", "output_paths", "outputs"}
_MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif",
                     ".mp4", ".mov", ".webm", ".mkv", ".avi", ".wav", ".mp3", ".flac", ".ogg", ".m4a"}
INTERFACE_CANDIDATE_LIMIT = 24


def scoped_interface_classes(payload: dict) -> set[str] | None:
    """Expand current nodes and the ranked repair shortlist, not every search hit."""
    graph = payload.get("current_graph")
    evidence = payload.get("repair_evidence")
    candidates = evidence.get("local_registry_candidates") if isinstance(evidence, dict) else None
    if not isinstance(graph, dict) or not isinstance(candidates, list):
        return None
    classes = {node["class_type"] for node in graph.values() if isinstance(node, dict) and isinstance(node.get("class_type"), str)}
    exact_errors = []
    for row in evidence.get("latest_stage_results", []) + evidence.get("preflight_failures", [])[-1:]:
        if isinstance(row, dict):
            exact_errors.append(normalize_legacy_execution_error(row))
    if isinstance(evidence.get("execution"), dict):
        exact_errors.append(normalize_legacy_execution_error(evidence["execution"]))
    limit = 8 if sum(len(error) for error in exact_errors if isinstance(error, str)) > 40000 else INTERFACE_CANDIDATE_LIMIT
    classes.update(row["class_type"] for row in candidates[:limit]
                   if isinstance(row, dict) and isinstance(row.get("class_type"), str))
    previous = evidence.get("previous_plan") or {}
    if isinstance(previous, dict):
        for edit in previous.get("edits", []):
            if isinstance(edit, dict) and isinstance(edit.get("node"), dict) and isinstance(edit["node"].get("class_type"), str):
                classes.add(edit["node"]["class_type"])
    return classes


def compact_interface_context(payload: dict, *, agent_id: str | None = None) -> dict:
    """Make a transport-only copy; the validator and audit retain full schemas.

    Model names, samplers, port types, and numerical constraints stay exact.
    Media dropdowns describe current case files, not every previous upload.
    """
    media = set()
    referenced = False
    retrieval_notes = []
    current_classes = {node.get("class_type") for node in payload.get("current_graph", {}).values()
                       if isinstance(node, dict)} if isinstance(payload.get("current_graph"), dict) else set()
    scope = scoped_interface_classes(payload)
    root_interfaces = payload.get("installed_node_interfaces")
    if isinstance(root_interfaces, list) and isinstance(payload.get("repair_evidence"), dict):
        if payload["repair_evidence"].get("installed_node_interfaces") == root_interfaces:
            payload = {**payload, "repair_evidence": {k: v for k, v in payload["repair_evidence"].items()
                                                      if k != "installed_node_interfaces"}}

    def add(value):
        if isinstance(value, str):
            media.add(value)
            media.add(Path(value).name)

    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in _LIVE_GRAPHS and isinstance(item, dict):
                    for node in item.values():
                        if isinstance(node, dict):
                            for field, binding in node.get("inputs", {}).items():
                                if field in _MEDIA_FIELDS:
                                    add(binding)
                elif key in _MEDIA_LISTS and isinstance(item, list):
                    for path in item:
                        add(path)
                elif key not in {"attempts", "attempt_history", "operation_notebook",
                                 "base_graph", "initial_current_graph", "installed_node_interfaces"}:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(payload)
    # A future handoff is a signed runtime binding, not another uploaded case.
    # Keep these placeholders visible to the repair planner even before the
    # generated file appears in the live filename dropdown.
    def deferred_placeholders(value):
        if isinstance(value, dict):
            dependencies = value.get("producer_dependencies")
            if isinstance(dependencies, list):
                types = {str(row.get("media_type", "")).upper()
                         for row in dependencies if isinstance(row, dict)}
                if "IMAGE" in types:
                    media.update({"INPUT_IMAGE.png", "IMAGE_INPUT.png", "START_IMAGE.png", "END_IMAGE.png"})
                if "MASK" in types:
                    media.update({"MASK_INPUT.png", "MASK_INPUT.mp4"})
                if "VIDEO" in types:
                    media.update({"INPUT_VIDEO.mp4", "INPUT_VIDEO_A.mp4", "INPUT_VIDEO_B.mp4"})
            for key, item in value.items():
                if key not in {"installed_node_interfaces", "current_graph", "base_graph"}:
                    deferred_placeholders(item)
        elif isinstance(value, list):
            for item in value:
                deferred_placeholders(item)
    deferred_placeholders(payload)

    def schema(row):
        result = dict(row)
        inputs = row.get("input")
        if not isinstance(inputs, dict):
            return result
        result["input"] = {}
        omitted = {}
        for section, fields in inputs.items():
            if section not in {"required", "optional"} or not isinstance(fields, dict):
                # Hidden inputs such as PROMPT/UNIQUE_ID are not editable.
                continue
            result["input"][section] = {}
            for field, declaration in fields.items():
                if not isinstance(declaration, (list, tuple)) or not declaration:
                    result["input"][section][field] = declaration
                    continue
                parts = list(declaration)
                options = dict(parts[1]) if len(parts) > 1 and isinstance(parts[1], dict) else None
                choices = parts[0] if isinstance(parts[0], list) else (options or {}).get("options")
                file_choices = [v for v in choices if v not in {"None", "none", ""}] if (
                    isinstance(choices, list) and all(isinstance(v, str) for v in choices)) else []
                is_media = field in _MEDIA_FIELDS and (
                    options and any(options.get(key) for key in ("image_upload", "video_upload", "audio_upload"))
                    or file_choices and all(Path(v).suffix.lower() in _MEDIA_EXTENSIONS for v in file_choices))
                if is_media:
                    if isinstance(choices, list) and all(isinstance(v, str) for v in choices):
                        selected = [v for v in choices if v in media or v in {"None", "none", ""}]
                        placeholders = media & {"INPUT_IMAGE.png", "IMAGE_INPUT.png", "START_IMAGE.png", "END_IMAGE.png",
                                                "MASK_INPUT.png", "MASK_INPUT.mp4", "INPUT_VIDEO.mp4",
                                                "INPUT_VIDEO_A.mp4", "INPUT_VIDEO_B.mp4"}
                        suffixes = {Path(v).suffix.lower() for v in file_choices}
                        if (options or {}).get("image_upload") or suffixes & {".jpg", ".jpeg", ".png", ".webp"}:
                            suffixes.update({".jpg", ".jpeg", ".png", ".webp"})
                        if (options or {}).get("video_upload"):
                            suffixes.add(".mp4")
                        selected.extend(sorted(v for v in placeholders if v not in selected
                                               and Path(v).suffix.lower() in suffixes))
                        if selected != choices:
                            omitted_count = len(set(choices) - set(selected))
                            if omitted_count:
                                omitted[f"{section}.{field}"] = omitted_count
                            if isinstance(parts[0], list):
                                parts[0] = selected
                            else:
                                options["options"] = selected
                            if options and options.get("default") in choices and options["default"] not in selected:
                                options.pop("default")
                if options is not None:
                    # Widget descriptions do not change accepted inputs.
                    options = {k: v for k, v in options.items() if k not in {"tooltip", "dynamicPrompts"}}
                    parts[1] = options
                result["input"][section][field] = parts
        if omitted:
            result["media_options_scope"] = "current case bindings; omitted uploads remain in the live validator"
            result["omitted_media_options"] = omitted
        if isinstance(result.get("description"), str) and len(result["description"]) > 160:
            result["description"] = result["description"][:159] + "…"
        return result

    def pointer(value):
        return str(value).replace("~", "~0").replace("/", "~1")

    def table(rows, path):
        nonlocal referenced
        if scope is not None:
            rows = [row for row in rows if not isinstance(row, dict) or row.get("class_type") in scope]
        result = [schema(row) if isinstance(row, dict) else row for row in rows]
        enums = {}
        for index, row in enumerate(result):
            if not isinstance(row, dict):
                continue
            for section, fields in row.get("input", {}).items():
                for field, declaration in fields.items():
                    if not isinstance(declaration, list) or not declaration or not isinstance(declaration[0], list):
                        continue
                    encoded = json.dumps(declaration[0], ensure_ascii=False, sort_keys=True)
                    if len(encoded) < 256:
                        continue
                    enum_path = f"{path}/{index}/input/{pointer(section)}/{pointer(field)}/0"
                    if encoded in enums:
                        referenced = True
                        options = dict(declaration[1]) if len(declaration) > 1 and isinstance(declaration[1], dict) else {}
                        options["options_ref"] = enums[encoded]
                        fields[field] = ["COMBO", options, *declaration[2:]]
                    else:
                        enums[encoded] = enum_path
        return result

    def candidate_refs(candidates, rows, path):
        nonlocal referenced
        interfaces = {row.get("class_type"): (index, row) for index, row in enumerate(rows) if isinstance(row, dict)}
        result = []
        for candidate in candidates:
            if isinstance(candidate, dict) and scope is not None and candidate.get("class_type") not in scope:
                result.append({key: candidate[key] for key in ("class_type", "matched_terms") if key in candidate}
                              | {"interface_details_omitted": True})
                continue
            if not isinstance(candidate, dict) or candidate.get("class_type") not in interfaces:
                result.append(candidate)
                continue
            index, interface = interfaces[candidate["class_type"]]
            compact = dict(candidate)
            for label, section in (("required_inputs", "required"), ("optional_inputs", "optional")):
                fields = interface.get("input", {}).get(section, {})
                types = {name: (value[0] if isinstance(value[0], str) else "COMBO") for name, value in fields.items()
                         if isinstance(value, (list, tuple)) and value}
                if candidate.get(label) == types:
                    compact.pop(label, None)
            if candidate.get("outputs") == interface.get("output"):
                compact.pop("outputs", None)
            if compact != candidate:
                referenced = True
                compact["interface_ref"] = f"{path}/{index}"
            result.append(compact)
        return result

    def deduplicate_retrieval(value, path):
        seen = {}

        def visit(item, current):
            nonlocal referenced
            if isinstance(item, dict) and any(key in item for key in ("evidence_ref", "options_ref")):
                return item
            if isinstance(item, dict) and item and all(
                isinstance(node, dict) and isinstance(node.get("class_type"), str)
                and isinstance(node.get("inputs"), dict) for node in item.values()):
                return item
            if isinstance(item, (dict, list, str)):
                encoded = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if len(encoded) >= 512:
                    if encoded in seen:
                        referenced = True
                        return {"evidence_ref": seen[encoded]}
                    seen[encoded] = current
            if isinstance(item, dict):
                return {key: child if key in {"error", "exception_message", "traceback", "current_graph", "workflow_api"}
                        else visit(child, current + "/" + pointer(key)) for key, child in item.items()}
            if isinstance(item, list):
                return [visit(child, current + f"/{index}") for index, child in enumerate(item)]
            return item

        return visit(value, path)

    def walk(value: Any, path="", retrieval=False):
        nonlocal referenced
        if isinstance(value, dict):
            nodes = [node for key, node in value.items() if not str(key).startswith("__")]
            if nodes and all(isinstance(node, dict) and isinstance(node.get("class_type"), str)
                             and isinstance(node.get("inputs"), dict) for node in nodes):
                return value
            result = {}
            for key, item in value.items():
                if key in {"installed_node_interfaces", "node_registry", "_response_json_schema"}:
                    if key == "installed_node_interfaces" and isinstance(item, list):
                        result[key] = table(item, path + "/" + pointer(key))
                    else:
                        result[key] = item
                elif key == "installed_implementation" and isinstance(item, list):
                    references = []
                    class_name = str(value.get("class_type") or "")
                    def source_rank(source):
                        if not isinstance(source, dict):
                            return (3, 1)
                        match = str(source.get("match") or "")
                        definition = bool(class_name and re.search(r"\bclass\s+" + re.escape(class_name) + r"\s*[:(]", match))
                        return (0 if definition else 1, "/tests/" in str(source.get("path") or ""))
                    for source in sorted(item, key=source_rank)[:1]:
                        if not isinstance(source, dict):
                            references.append(source)
                            continue
                        reference = dict(source)
                        excerpt = reference.get("implementation_excerpt")
                        if isinstance(excerpt, str) and len(excerpt) > 512:
                            reference.pop("implementation_excerpt")
                            reference["excerpt_sha256"] = hashlib.sha256(excerpt.encode()).hexdigest()
                            reference["excerpt_omitted"] = True
                        references.append(reference)
                    result[key] = references
                    if len(item) > 1:
                        result["implementation_match_count"] = value.get("implementation_match_count", len(item))
                elif retrieval and key in {"raw", "official_excerpt", "page_content"} and isinstance(item, str) and len(item) > 2048:
                    # Search pages are discovery evidence, not a live node
                    # interface. Preserve sources and matching lines instead
                    # of sending GitHub navigation and entire search batches.
                    terms = value.get("matched_agent_terms") or []
                    lines = [line for line in item.splitlines() if any(str(term).casefold() in line.casefold() for term in terms)]
                    preview = "\n".join(lines[:5]) or item[:768]
                    urls = list(dict.fromkeys(re.findall(r'https?://[^\s<>\)\]\"\x27]+', item)))
                    result[key] = {"summary_type": "retrieval_page", "preview": preview[:768],
                                   "urls": urls[:8], "source_url_count": len(urls),
                                   "character_count": len(item), "sha256": hashlib.sha256(item.encode()).hexdigest(),
                                   "content_omitted": True}
                elif retrieval and key == "nearest_interface_mismatches" and isinstance(item, list):
                    result[key] = [{name: row[name] for name in (
                        "class_type", "inputs_outputs", "required_contract", "interface_compatible",
                        "role_compatible", "role_reasons") if name in row} for row in item if isinstance(row, dict)]
                    result["rejected_candidate_details_omitted"] = True
                else:
                    child_path = path + "/" + pointer(key)
                    current_retrieval = (payload.get("validation_feedback") or {}).get("capability_retrieval") if isinstance(payload.get("validation_feedback"), dict) else None
                    old_discovery = isinstance(current_retrieval, dict) and any(part in path for part in (
                        "/_prompt_static_context", "/full_trace", "/attempt_trace", "/complete_initial_trace"))
                    if ((agent_id == "workflow_failure_analysis_agent" and current_classes or old_discovery)
                            and key in {"capability_retrieval", "latest_capability_retrieval"}
                            and isinstance(item, dict) and item.get("summary_type") != "historical_retrieval"):
                        verified = item.get("locally_verified_node_candidates", [])
                        retained = [row for row in verified if isinstance(row, dict) and row.get("class_type") in current_classes]
                        candidates = [row.get("class_type") for row in verified if isinstance(row, dict)]
                        requests = (item.get("retrieval_plan") or {}).get("retrieval_requests", [])
                        capabilities = [row.get("capability") for row in requests if isinstance(row, dict)]
                        retrieval_notes.append({"operation": "历史修复/检索", "input": "; ".join(str(v) for v in capabilities)[:180],
                                                "output": "; ".join(str(v) for v in candidates)[:180],
                                                "result": str(item.get("retrieval_status") or "检索结果已记录")[:180]})
                        item = {**{name: item[name] for name in ("source", "retrieval_status", "return_code", "error", "capability_scope") if name in item},
                                "summary_type": "historical_retrieval", "verified_candidate_count": len(verified),
                                "locally_verified_node_candidates": retained,
                                "policy": "Historical repair discovery is in the notebook. Diagnose current_graph and its latest exact failure."}
                    result[key] = walk(item, child_path, retrieval or key in {
                        "capability_retrieval", "latest_capability_retrieval", "retrieval_feedback", "anysearch_evidence"})
                    if key in {"capability_retrieval", "latest_capability_retrieval", "retrieval_feedback"}:
                        result[key] = deduplicate_retrieval(result[key], child_path)
            if isinstance(result.get("local_registry_candidates"), list):
                rows = result.get("installed_node_interfaces")
                if isinstance(rows, list):
                    result["local_registry_candidates"] = candidate_refs(result["local_registry_candidates"], rows, path + "/installed_node_interfaces")
                    if scope is not None:
                        result["omitted_interface_candidates"] = sum(isinstance(row, dict) and row.get("interface_details_omitted") is True
                                                                     for row in result["local_registry_candidates"])
                elif isinstance(root_interfaces, list):
                    result["local_registry_candidates"] = candidate_refs(result["local_registry_candidates"], table(root_interfaces, "/installed_node_interfaces"), "/installed_node_interfaces")
            diagnostics = result.get("diagnostics")
            if isinstance(diagnostics, list):
                for number, diagnostic in enumerate(diagnostics):
                    if not isinstance(diagnostic, dict):
                        continue
                    allowed = diagnostic.get("allowed")
                    field = diagnostic.get("input_name") or diagnostic.get("param")
                    if (media and field in _MEDIA_FIELDS and isinstance(allowed, list) and len(allowed) > 12
                            and all(isinstance(v, str) and (v in {"None", "none", ""} or Path(v).suffix.lower() in _MEDIA_EXTENSIONS) for v in allowed)):
                        selected = [v for v in allowed if v in media or v in {"None", "none", ""}]
                        diagnostic["allowed"] = selected
                        diagnostic["allowed_scope"] = "current case media; complete list is in the original error/report"
                        diagnostic["omitted_allowed_values"] = len(allowed) - len(selected)
                    for enum in result.get("allowed_enums", []) if isinstance(result.get("allowed_enums"), list) else []:
                        if isinstance(enum, dict) and enum.get("allowed") == allowed and len(json.dumps(allowed, ensure_ascii=False)) > 4096:
                            enum["allowed"] = {"evidence_ref": path + f"/diagnostics/{number}/allowed"}
                            referenced = True
            if isinstance(result.get("reviewed_failure_analysis"), dict) and result.get("failure_analyses") == [result["reviewed_failure_analysis"]]:
                result["failure_analyses"] = [{"evidence_ref": path + "/reviewed_failure_analysis"}]
                referenced = True
            current_feedback = payload.get("validation_feedback")
            if (path and isinstance(current_feedback, dict) and isinstance(current_feedback.get("error"), str)
                    and len(current_feedback["error"]) > 4096 and result.get("error") == current_feedback["error"]
                    and any(part in path for part in ("/full_trace", "/attempt_trace", "/complete_initial_trace"))):
                result["error"] = {"evidence_ref": "/validation_feedback/error"}
                referenced = True
            return result
        if isinstance(value, list):
            return [walk(item, path + f"/{index}", retrieval) for index, item in enumerate(value)]
        return value

    result = walk(payload)
    evidence = result.get("repair_evidence")
    if isinstance(evidence, dict):
        failures = evidence.get("preflight_failures")
        if isinstance(failures, list) and failures and isinstance(failures[-1], dict):
            latest = failures[-1]
            path = f"/repair_evidence/preflight_failures/{len(failures) - 1}"
            error = latest.get("error")
            if isinstance(error, str) and len(error) > 4096 and evidence.get("setup_failure") == error:
                evidence["setup_failure"] = {"evidence_ref": path + "/error"}
                referenced = True
            enums = latest.get("allowed_enums")
            original_failures = payload.get("repair_evidence", {}).get("preflight_failures", [])
            original_enums = original_failures[-1].get("allowed_enums") if original_failures else enums
            if isinstance(enums, list) and len(json.dumps(original_enums, ensure_ascii=False)) > 4096:
                if evidence.get("allowed_enums") == original_enums:
                    evidence["allowed_enums"] = {"evidence_ref": path + "/allowed_enums"}
                    referenced = True
                for index, enum in enumerate(enums):
                    if not isinstance(enum, dict) or not isinstance(enum.get("allowed"), list):
                        continue
                    for number, diagnostic in enumerate(latest.get("diagnostics", [])):
                        if isinstance(diagnostic, dict) and diagnostic.get("allowed") == enum["allowed"]:
                            enums[index] = {**enum, "allowed": {"evidence_ref": path + f"/diagnostics/{number}/allowed"}}
                            break
    if referenced:
        result["interface_options_policy"] = (
            "options_ref/interface_ref/evidence_ref are JSON pointers to exact supplied values; resolve them before editing. "
            "_prompt_static_context denotes the developer session_static_context. Media options are scoped to current case files."
        )
    if scope is not None:
        result["interface_scope_policy"] = (
            "Expanded interfaces include every current/previous_plan node and up to 24 ranked repair candidates; "
            "8 candidates when exact errors exceed 40000 characters. "
            "Other class names are discovery hints; retrieve their exact interfaces before using them."
        )
    if retrieval_notes:
        from .operation_notebook import notebook_agent_payload
        records = list(result.get("operation_notebook") or [])
        for note in retrieval_notes:
            if note not in records:
                records.append(note)
        result["operation_notebook"] = records
        result = notebook_agent_payload(result)
    return result
