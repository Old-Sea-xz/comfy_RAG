"""Compact output contract after the workflow has already been selected."""
from __future__ import annotations

from typing import Any


def named_parameter_slots(graph: dict, registry=None, *, freeze_generation_budget=False) -> dict:
    """Keep node bindings locally; expose short, unambiguous parameter names."""
    from .parameter_generation import parameter_slots
    slots = parameter_slots(graph, registry, freeze_generation_budget=freeze_generation_budget)
    reserved = {record["parameter"] for record in slots.values()}
    named = {}
    for record in slots.values():
        base = record["parameter"]
        name = base
        index = 2
        while name in named or (name != base and name in reserved):
            name = f"{base}_{index}"
            index += 1
        named[name] = {**record, "slot": name}
    return named


def parameter_prompt_context(slots: dict, *, graph_supplied=False) -> dict:
    """Send defaults or graph bindings once; types and limits live in schema."""
    if graph_supplied:
        return {name: [record["node_id"], record["parameter"]] for name, record in slots.items()}
    return {name: {key: value for key, value in record.items()
                   if key in {"class_type", "parameter", "current_value", "title"} and value != ""}
            for name, record in slots.items()}


def expand_parameter_edits(raw: Any, slots: dict, selection: dict | None = None) -> Any:
    """Restore runner metadata without asking the model to repeat it."""
    if not isinstance(raw, dict) or raw.get("status") == "blocked":
        return raw
    result = dict(raw)
    if "parameters" in result:
        from .parameter_generation import compile_parameters
        result["edits"] = compile_parameters(result, slots)
        result.pop("parameters")
    result.setdefault("skill_ids", list(selection["skill_ids"]) if selection else [])
    result.setdefault("rationale", "Apply parameter assignments to the selected workflow.")
    if selection:
        result.setdefault("skill_assessments", [row for row in selection["skill_assessments"]
                                               if row["skill_id"] in selection["skill_ids"]])
        result.setdefault("base_decision", selection["base_decision"])
    return result
