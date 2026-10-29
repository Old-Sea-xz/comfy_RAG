from __future__ import annotations
from typing import Any

class ResourceEnvelopeError(ValueError):
    pass

def validate_runtime_resource_envelope(graph: dict[str, Any], envelope: dict[str, Any]) -> None:
    """Reject a runtime graph that exceeds the leaf template's recorded limits."""
    bounded = {'width', 'height', 'num_frames', 'batch_size', 'amount', 'frame_load_cap', 'steps'}
    limits: dict[str, list[float]] = {}
    for key, value in envelope.items():
        param = str(key).rsplit('.', 1)[-1]
        if param in bounded and isinstance(value, (int, float)):
            limits.setdefault(param, []).append(value)
    runtime = _template_resource_envelope(graph)
    for key, value in runtime.items():
        param = key.rsplit('.', 1)[-1]
        if param not in limits or not isinstance(value, (int, float)):
            continue
        limit = max(limits[param])
        if value > limit:
            raise ResourceEnvelopeError(f'runtime resource envelope exceeded: {key}={value} > {limit}')

def _template_resource_envelope(graph: dict[str, Any]) -> dict[str, Any]:
    keys = {'width', 'height', 'num_frames', 'batch_size', 'amount', 'frame_load_cap', 'steps'}
    envelope: dict[str, Any] = {}
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        for key, value in node.get('inputs', {}).items():
            if key in keys and isinstance(value, (int, float)):
                envelope[f'{node_id}.{key}'] = value
    return envelope

def require_bound_sources(graph):
    """Never submit an unresolved media placeholder."""
    if any(node.get('_meta', {}).get('m3a_pending_source') or '__M3A_PENDING_SOURCE__' in str(node.get('inputs', {})) for node in graph.values()):
        raise ResourceEnvelopeError('Geometry source must be bound before submission')
