"""Opt-in prompt reductions, kept separate from execution and quality policy.

The parameter representation is reversible: connections, fixed inputs, current
values and semantic node titles are retained. No repair evidence is summarized
here. Flags remain off until the paired validation gate is satisfied.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any


def enabled(feature: str) -> bool:
    return os.getenv('COMFYROUTER_COMPACT_' + feature, '0').casefold() in {'1', 'true', 'yes', 'on'}


EVALUATOR_INSTRUCTIONS = (
    'Evaluate the supplied media using exactly the supplied task and evaluation criteria. '
    'Follow their requested analysis and judgment output format.'
)


def evaluator_instructions() -> dict[str, str]:
    return {'instructions': EVALUATOR_INSTRUCTIONS} if enabled('EVALUATOR') else {}


def parameter_workflow(graph: dict, bindings: dict) -> tuple[dict, dict]:
    """Encode graph inputs once, keeping executable content and semantic titles."""
    parameters = {}
    covered = set()
    for name, binding in bindings.items():
        if not isinstance(binding, list) or len(binding) != 2:
            raise ValueError('parameter binding must be [node_id,input_name]')
        node_id, field = binding
        if (node_id, field) in covered:
            raise ValueError('duplicate parameter binding')
        value = graph[node_id]['inputs'][field]
        parameters[name] = [node_id, field, copy.deepcopy(value)]
        covered.add((node_id, field))
    nodes = {}
    titles = {}
    for node_id, node in graph.items():
        nodes[node_id] = [node['class_type'], {k: copy.deepcopy(v) for k, v in node['inputs'].items()
                                            if (node_id, k) not in covered}]
        title = (node.get('_meta') or {}).get('title')
        if title:
            titles[node_id] = title
    workflow = {'nodes': nodes}
    if titles:
        workflow['titles'] = titles
    return workflow, parameters


def restore_parameter_workflow(workflow: dict, parameters: dict) -> dict:
    """For validation: recover class, every input and semantic title exactly."""
    graph = {nid: {'class_type': item[0], 'inputs': copy.deepcopy(item[1])}
             for nid, item in workflow['nodes'].items()}
    for nid, field, value in parameters.values():
        graph[nid]['inputs'][field] = copy.deepcopy(value)
    for nid, title in workflow.get('titles', {}).items():
        graph[nid]['_meta'] = {'title': title}
    return graph


def compact_parameter_payload(payload: dict, contract: dict) -> dict:
    """Only selected-workflow value filling; never structural repair/selection."""
    if (not enabled('PARAMETER_INPUT') or contract.get('required_output') != ['parameters']
            or not isinstance(payload.get('current_graph'), dict)
            or not isinstance(payload.get('editable_parameters'), dict)):
        return payload
    # gen_038 violated prompt-only repair scope in paired v3 validation.
    # Do not alter the representation or duplicate evidence during repair:
    # an existing graph reference can carry scope even when values match.
    if (payload.get('repair_evidence') or payload.get('operation_notebook')
            or 'repair' in str(payload.get('planning_mode', '')).casefold()):
        return payload
    graph = payload['current_graph']
    try:
        parameter_workflow(graph, payload['editable_parameters'])  # Validate exact bindings.
    except (KeyError, ValueError, TypeError):
        return payload
    # A table-only prototype changed sampler decisions in paired validation.
    # Keep the familiar executable graph verbatim; only de-duplicate exact
    # copies and remove ComfyUI UI metadata other than semantic titles.
    trimmed = copy.deepcopy(graph)
    for node in trimmed.values():
        if isinstance(node.get('_meta'), dict):
            title = node['_meta'].get('title')
            if title:
                node['_meta'] = {'title': title}
            else:
                node.pop('_meta')
    def walk(value, root=False):
        if isinstance(value, dict):
            if not root and value == graph:
                return {'$ref': '#/current_graph'}
            return {key: (trimmed if root and key == 'current_graph' else walk(item))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value
    result = walk(payload, root=True)
    if result.get('base_graph') == {'$ref': '#/current_graph'}:
        result.pop('base_graph')
    encoded = lambda v: json.dumps(v, ensure_ascii=False, separators=(',', ':'))
    if len(encoded(result)) >= len(encoded(payload)):
        return payload  # Small graphs can already be more compact in their original form.
    return result


def compact_diagnosis_request(contract: dict, payload: dict) -> tuple[dict, dict]:
    if not enabled('DIAGNOSIS'):
        return contract, payload
    # A global shorthand rule changed earliest-stage attribution in the
    # gen_065 paired pilot. Only abbreviate channels that Python can prove
    # absent in a known, closed text-generation graph. Multi-stage and edited
    # media retain their original diagnosis contract exactly.
    graph = payload.get('current_graph')
    known = {
        'UNETLoader', 'DualCLIPLoader', 'CLIPLoader', 'VAELoader', 'CheckpointLoaderSimple',
        'CLIPTextEncode', 'EmptyLatentImage', 'EmptySD3LatentImage',
        'EmptyHunyuanLatentVideo', 'EmptyHunyuanLatentVideo15', 'EmptyLTXVLatentVideo',
        'ModelSamplingAuraFlow', 'ModelSamplingSD3', 'ModelSamplingFlux', 'FluxGuidance',
        'KSampler', 'KSamplerAdvanced', 'VAEDecode', 'SaveImage', 'CreateVideo', 'SaveVideo',
        'VHS_VideoCombine', 'ConditioningZeroOut',
    }
    empty = {'EmptyLatentImage', 'EmptySD3LatentImage', 'EmptyHunyuanLatentVideo',
             'EmptyHunyuanLatentVideo15', 'EmptyLTXVLatentVideo'}
    if not isinstance(graph, dict) or not graph or payload.get('ordered_stages'):
        return contract, payload
    classes = {node.get('class_type') for node in graph.values() if isinstance(node, dict)}
    if len(classes) == 0 or not classes <= known or not classes & empty:
        return contract, payload
    if any(not isinstance(node, dict) or not isinstance(node.get('inputs'), dict) for node in graph.values()):
        return contract, payload
    contract, payload = copy.deepcopy(contract), copy.deepcopy(payload)
    contract.setdefault('rules', []).append(
        'This known empty-latent workflow has no source-image or mask conditioning. Return the literal '
        'string "not_applicable" for source_image_conditioning and mask_conditioning. '
        'Keep all other channels with their original likelihood, concrete evidence and counterfactual; '
        'do not abbreviate any uncertain cause. Keep failure class, repair decision and evidence complete.'
    )
    schema = payload.get('output_schema', {})
    if isinstance(schema.get('conditioning_attribution'), dict):
        for key in ('source_image_conditioning', 'mask_conditioning'):
            if key in schema['conditioning_attribution']:
                schema['conditioning_attribution'][key] = 'not_applicable'
    return contract, payload


def expand_conditioning_attribution(raw: Any) -> Any:
    """Restore the legacy downstream shape; never turn uncertainty into absence."""
    if not isinstance(raw, dict) or not isinstance(raw.get('conditioning_attribution'), dict):
        return raw
    if not any(v == 'not_applicable' for v in raw['conditioning_attribution'].values()):
        return raw
    result = copy.deepcopy(raw)
    for key, value in result['conditioning_attribution'].items():
        if value == 'not_applicable':
            result['conditioning_attribution'][key] = {
                'likelihood': 'not_applicable', 'evidence': [],
                'counterfactual': 'Not applicable to the supplied executed workflow.',
            }
    return result
