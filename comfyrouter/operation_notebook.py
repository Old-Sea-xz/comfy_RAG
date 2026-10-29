"""Four short fields per generation, evaluation, or repair operation."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


NOTEBOOK_WINDOW = 12
_HISTORIES = {"attempts", "attempt_history"}
_PLAN_HISTORIES = {"generation_plan_attempts", "workflow_modification_attempts", "previous_attempts"}
_GRAPH_FIELDS = {"graph", "current_graph", "base_graph", "runtime_graph", "partial_graph",
                 "workflow_api", "final_graph", "preflight_graph", "post_edit_graph",
                 "initial_current_graph", "previous_plan", "edits", "node_registry",
                 "installed_node_interfaces", "_response_json_schema"}
_CONTEXT_COPIES = {"trace", "full_trace", "attempt_trace", "complete_initial_trace",
                   "full_report", "_prompt_static_context"}
_OLD_GRAPHS = {"graph", "current_graph", "base_graph", "runtime_graph", "final_graph",
               "preflight_graph", "initial_current_graph", "initial_graph", "current_candidate_graph"}


def _short(value: Any, limit: int = 180) -> str:
    text = str(value if value is not None else "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _files(values: Any) -> str:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return ""
    return _short("; ".join(dict.fromkeys(Path(v).name for v in values if isinstance(v, str))))


def _inputs(row: dict, fallback: Any) -> str:
    graph = row.get("runtime_graph") or row.get("graph") or {}
    files = []
    for node in graph.values() if isinstance(graph, dict) else []:
        if isinstance(node, dict):
            files.extend(v for k, v in node.get("inputs", {}).items()
                         if k in {"image", "video", "file", "audio"} and isinstance(v, str))
    return _files(files) or _files(fallback) or _short(row.get("operation")) or "工作流"


def _error(value: Any) -> str:
    if isinstance(value, dict):
        return _short(value.get("exception_message") or value.get("message") or value.get("error") or "")
    if isinstance(value, list):
        value = value[-1] if value else ""
    # The final line normally contains the exception, rather than its traceback.
    return _short(str(value or "").strip().splitlines()[-1] if str(value or "").strip() else "")


def _changes(plan: Any) -> str:
    if not isinstance(plan, dict):
        return ""
    changes = []
    for edit in plan.get("edits", []):
        if not isinstance(edit, dict):
            continue
        action = edit.get("action") or edit.get("op") or "edit"
        node = edit.get("target") or edit.get("node_id") or ""
        key = edit.get("key") or edit.get("param") or ""
        if action == "set_param":
            changes.append(f"{node}.{key}={_short(edit.get('value'), 70)}")
        elif action in {"connect", "disconnect"}:
            changes.append(f"{action} {edit.get('source')}→{edit.get('destination')}")
        else:
            changes.append(f"{action} {node}")
    return _short("; ".join(changes)) or _short(plan.get("rationale"))


def operation_records(attempts: list, inputs: Any = None) -> list[dict[str, str]]:
    """Read results already recorded by the runner; never infer a quality pass."""
    records = []

    def add(operation, input_value, output, result):
        records.append({"operation": _short(operation, 80), "input": _short(input_value),
                        "output": _short(output), "result": _short(result)})

    for index, attempt in enumerate(attempts, 1):
        if not isinstance(attempt, dict):
            continue
        label = f"第{attempt.get('attempt', index)}轮"
        stages = attempt.get("stages") or [attempt]
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            name = label + (f"/{stage['stage_id']}" if stage.get("stage_id") else "")
            execution = stage.get("execution") or {}
            if isinstance(execution, dict) and (execution or stage.get("status")):
                status = execution.get("status") or stage.get("status") or "unknown"
                error = _error(execution.get("error") or stage.get("error"))
                result = {"success": "执行成功", "reused": "复用已有输出"}.get(status, str(status))
                if error:
                    result += ": " + error
                add(name + "/生成", _inputs(stage, inputs),
                    _files(stage.get("outputs") or execution.get("outputs") or stage.get("primary_output")), result)

        evaluation = attempt.get("comfybench")
        if isinstance(evaluation, dict) and evaluation.get("status") not in {"no_output", "stage_execution_failed"}:
            judgment = evaluation.get("comfybench", evaluation)
            if not isinstance(judgment, dict):
                judgment = {}
            passed = judgment.get("resolved")
            output = "通过" if passed is True else "未通过" if passed is False else "未取得评测结果"
            reason = judgment.get("analysis") or evaluation.get("error") or evaluation.get("status") or ""
            add(label + "/评测", _files(attempt.get("outputs") or attempt.get("primary_output")), output, _error(reason))

        for stage in stages:
            if not isinstance(stage, dict):
                continue
            name = label + (f"/{stage['stage_id']}" if stage.get("stage_id") else "")
            # A single-leaf runner may record a plan, then a failure for the
            # same try. Keep the actual final outcome of that planning try.
            repairs = {}
            for number, repair in enumerate(stage.get("repair_generation_plans", []), 1):
                if isinstance(repair, dict):
                    repairs[repair.get("try", number)] = repair
            for number, repair in repairs.items():
                plan = repair.get("plan") or repair.get("previous_plan") or {}
                error = _error(repair.get("error"))
                result = ("失败: " + error) if error else (
                    "静态验证通过，待执行" if repair.get("status") == "validated" else "已提出计划，待验证")
                analysis = attempt.get("failure_analysis") or {}
                reason = stage.get("retry_reason") or (analysis.get("diagnosis") if isinstance(analysis, dict) else analysis) or "修复前次失败"
                add(name + f"/修复{number}", reason, _changes(plan), result)
        evolution = attempt.get("workflow_evolution")
        if isinstance(evolution, dict) and evolution:
            error = _error(evolution.get("error"))
            result = ("失败: " + error) if error else (
                "候选已记录，待执行" if evolution.get("candidate_path")
                else _short(evolution.get("status")) or "修复结果未记录")
            add(label + "/修复工作流", evolution.get("trigger") or "修复前次失败",
                _files(evolution.get("candidate_path")), result)
    return records


def write_operation_notebook(report: dict, path: Path) -> None:
    """Replace the short derived notebook atomically on each checkpoint."""
    records = operation_records(report.get("attempts", []), report.get("case_resources"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                                 for row in records), encoding="utf-8")
    temporary.replace(path)


def notebook_agent_payload(payload: dict) -> dict:
    """Use short operation records instead of recursively replaying attempts."""
    if not isinstance(payload.get("current_graph"), dict):
        for container in (payload, *(payload.get(key) for key in ("attempt_trace", "full_trace", "trace"))):
            if not isinstance(container, dict):
                continue
            current = container.get("current_candidate_graph") or container.get("current_graph")
            if isinstance(current, dict) and current and all(isinstance(node, dict) and isinstance(node.get("class_type"), str)
                                                            and isinstance(node.get("inputs"), dict) for node in current.values()):
                payload = {**payload, "current_graph": current}
                break
    existing = payload.get("operation_notebook")
    records = list(existing) if isinstance(existing, list) else []
    notebook_present = isinstance(existing, list)
    older_failures = []
    changed = False
    live_graph_present = isinstance(payload.get("current_graph"), dict)

    def walk(value, historical=False):
        nonlocal changed, notebook_present
        if isinstance(value, dict):
            nodes = [item for key, item in value.items() if not str(key).startswith("__")]
            if nodes and all(isinstance(node, dict) and isinstance(node.get("class_type"), str)
                             and isinstance(node.get("inputs"), dict) for node in nodes):
                return value
            result = {}
            for key, item in value.items():
                if historical and live_graph_present and key in _OLD_GRAPHS:
                    changed = True
                elif key in _GRAPH_FIELDS or key == "operation_notebook":
                    result[key] = item
                elif (key == "preflight_failures" and isinstance(item, list) and len(item) > 1
                      and all(isinstance(failure, dict) for failure in item)):
                    older_failures.extend(item[:-1])
                    result[key] = [walk(item[-1], historical)]
                    changed = True
                elif key == "composition_attempt" and isinstance(item, dict):
                    short_records = operation_records([item], payload.get("case_resources"))
                    if short_records:
                        records.extend(short_records)
                        # These are current failures, not historical graph
                        # copies. Preserve their exact messages for repair.
                        result["latest_stage_results"] = [{
                            "stage_id": stage.get("stage_id"),
                            "status": (stage.get("execution") or {}).get("status") or stage.get("status"),
                            "error": (stage.get("execution") or {}).get("error") or stage.get("error"),
                            "outputs": stage.get("outputs", []),
                        } for stage in item.get("stages", []) if isinstance(stage, dict)]
                        changed = True
                    else:
                        result[key] = walk(item, historical)
                elif key in _HISTORIES and isinstance(item, list):
                    short_records = operation_records(item, payload.get("case_resources"))
                    if short_records or not item:
                        records.extend(short_records)
                        notebook_present = True
                        changed = True
                    else:
                        # An unknown list of planner/tool attempts is not a
                        # generation history. Preserve evidence we cannot read.
                        result[key] = walk(item, historical)
                elif key in _PLAN_HISTORIES and isinstance(item, list) and all(isinstance(row, dict) for row in item):
                    for number, row in enumerate(item, 1):
                        plan = row.get("plan") or row.get("previous_plan") or {}
                        error = _error(row.get("error"))
                        outcome = "失败: " + error if error else row.get("status") or "已记录，未确认执行成功"
                        records.append({"operation": f"历史修复/{key}/{row.get('attempt', number)}",
                                        "input": "修复当前工作流", "output": _changes(plan) or _files(row.get("candidate_path")),
                                        "result": _short(outcome)})
                        if row.get("execution_successful") is not None:
                            records.append({"operation": f"历史生成/{row.get('attempt', number)}",
                                            "input": _files(row.get("candidate_path")), "output": _files(row.get("execution_path")),
                                            "result": "执行成功" if row["execution_successful"] is True else "执行未成功"})
                        evaluation = row.get("official_evaluation")
                        if isinstance(evaluation, dict) and evaluation.get("status") != "not_run":
                            records.append({"operation": f"历史评测/{row.get('attempt', number)}",
                                            "input": _files(row.get("evaluation_path")), "output": "",
                                            "result": _error(evaluation.get("analysis") or evaluation.get("status"))})
                    if item:
                        latest = item[-1]
                        # Keep the newest precise planning failure outside
                        # the notebook; older plans are historical context.
                        result[f"latest_{key}_diagnostic"] = {
                            name: latest[name] for name in ("error", "diagnostics", "raw_response", "previous_plan")
                            if name in latest}
                    notebook_present = True
                    changed = True
                elif key == "failure_analyses" and isinstance(item, list) and len(item) > 1 and all(isinstance(row, dict) for row in item):
                    for number, row in enumerate(item[:-1], 1):
                        records.append({"operation": f"历史修复分析/{number}", "input": "前次失败",
                                        "output": _short(row.get("next_action")),
                                        "result": _error(row.get("diagnosis") or row.get("root_causes") or row.get("failure_class"))})
                    result[key] = [walk(item[-1], historical)]
                    notebook_present = True
                    changed = True
                elif key == "_workflow_evolution_static_context":
                    # This is a cached copy of context, not another operation.
                    changed = True
                else:
                    result[key] = walk(item, historical or key in _CONTEXT_COPIES)
            failures = value.get("preflight_failures")
            if isinstance(failures, list) and len(failures) > 1 and all(isinstance(row, dict) for row in failures):
                for key in ("allowed_enums", "registry_hints"):
                    mirrors = [entry for row in failures for entry in row.get(key, [])]
                    if value.get(key) == mirrors:
                        result[key] = failures[-1].get(key, [])
            return result
        if isinstance(value, list):
            return [walk(item, historical) for item in value]
        return value

    result = walk(payload)
    for index, failure in enumerate(older_failures, 1):
        if not isinstance(failure, dict):
            continue
        output = _changes(failure.get("previous_plan"))
        outcome = _short("失败: " + _error(failure.get("error")))
        if not any(row["output"] == output and row["result"] == outcome for row in records):
            records.append({"operation": f"预检/修复{index}", "input": "修改当前工作流",
                            "output": output, "result": outcome})
    if records or notebook_present:
        unique = list({json.dumps(row, ensure_ascii=False, sort_keys=True): row for row in records}.values())
        result["operation_notebook"] = unique[-NOTEBOOK_WINDOW:]
        omitted = len(unique) - len(result["operation_notebook"])
        if omitted:
            result["notebook_earlier_records"] = omitted
        result["_prompt_history_mode"] = "latest_only"
        if isinstance(payload.get("current_graph"), dict):
            if records:
                result.pop("base_graph", None)
            result["current_workflow_policy"] = (
                "Continue edits on current_graph. Preserve existing parameters unless explicitly changed. "
                "Notebook entries describe past operations; never restore their graphs or the original Base."
            )
    return result if changed or records or notebook_present else payload
