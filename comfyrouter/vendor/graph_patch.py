"""Typed graph patch implementation extracted from the local leaf-builder helpers."""
from __future__ import annotations
import copy
from typing import Any

def api_graph_nodes(graph: Any) -> dict[str, dict[str, Any]]:
    """Normalize API-format and UI-format graphs to an API-like node mapping."""
    if not isinstance(graph, dict):
        return {}
    if isinstance(graph.get("nodes"), list):
        result = {}
        for node in graph["nodes"]:
            if isinstance(node, dict) and isinstance(node.get("id"), (int, str)):
                result[str(node["id"])] = node
        return result
    return {str(k): v for k, v in graph.items() if isinstance(v, dict)}

def validate_api_graph(graph: Any) -> list[str]:
    errors: list[str] = []
    nodes = api_graph_nodes(graph)
    if not nodes:
        return ["workflow is not a non-empty node graph"]
    for node_id, node in nodes.items():
        if not isinstance(node.get("class_type"), str) or not node["class_type"]:
            errors.append(f"node {node_id}: missing class_type")
        if not isinstance(node.get("inputs", {}), dict):
            errors.append(f"node {node_id}: inputs must be an object")
    return errors

def patch_graph(base: dict, increments: list[dict], model_name: str = "") -> dict:
    graph=copy.deepcopy(base); errors=validate_api_graph(graph)
    if errors: raise ValueError("invalid Base graph: " + "; ".join(errors))
    nodes={str(k):v for k,v in graph.items() if isinstance(v,dict)}
    numeric=[int(k) for k in nodes if k.isdigit()]; next_id=max(numeric,default=0)+1
    for inc in increments:
        op=inc.get("op")
        if op=="modify_param":
            node=str(inc.get("node_id")); key=inc.get("input")
            if node not in nodes or not isinstance(key,str): raise ValueError(f"invalid modify_param target: {inc}")
            nodes[node].setdefault("inputs",{})[key]=inc.get("value")
        elif op=="append_prompt":
            node=str(inc.get("node_id")); key=str(inc.get("input","text")); suffix=str(inc.get("text","")).strip()
            if node not in nodes or not suffix: raise ValueError(f"invalid append_prompt target: {inc}")
            old=str(nodes[node].setdefault("inputs",{}).get(key,"")); nodes[node]["inputs"][key]=(old+", "+suffix).strip(", ")
        elif op=="insert_node":
            node=copy.deepcopy(inc.get("node"));
            if not isinstance(node,dict) or not isinstance(node.get("class_type"),str) or not isinstance(node.get("inputs",{}),dict): raise ValueError("insert_node requires class_type and inputs")
            requested=str(inc.get("node_id",next_id));
            while requested in nodes: requested=str(next_id); next_id+=1
            nodes[requested]=node
        else: raise ValueError(f"unsupported operation: {op}")
    return nodes
