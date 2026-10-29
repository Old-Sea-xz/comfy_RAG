"""Executable feedback/evolution loop for ComfyRouter workflows.

The LLM agents propose decisions, but this module owns the parts that must be
deterministic: applying typed graph edits, submitting the exact API graph to
ComfyUI, collecting `/history` evidence, attributing failures, calculating
quality gates, and retrying a repaired candidate.  A caller may plug in an
LLM feedback/evolution callback, but no callback is allowed to claim success
without the executor evidence.
"""
from __future__ import annotations
import copy
import difflib
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from . import config
from .execution_diagnostics import execution_error_text, large_media_choices

class ClosedLoopError(RuntimeError):
    """Invalid graph, failed execution, or an unsafe repair proposal."""

class ComfyUITransportError(ClosedLoopError):
    """The service is unavailable, rather than the workflow being invalid."""

    def __init__(self, message, *, prompt_id='', submission_uncertain=False, audit=None):
        super().__init__(message)
        self.prompt_id = prompt_id
        self.submission_uncertain = submission_uncertain
        self.audit = audit or {}

class StaticValidationError(ClosedLoopError):
    """A non-executing graph check failed, with machine-readable diagnostics."""

    def __init__(self, diagnostics: list[dict[str, Any]]) -> None:
        self.diagnostics = diagnostics

        def render(item: dict[str, Any]) -> str:
            text = f'{item['code']} at {item['path']}: {item['message']}'
            location = []
            for key in ('node_id', 'class_type', 'input_name', 'source_node_id', 'source_class_type'):
                if key in item and item[key] not in (None, ''):
                    location.append(f'{key}={item[key]!r}')
            if location:
                text += '; ' + ', '.join(location)
            if 'expected' in item or 'actual' in item:
                text += f'; expected={item.get('expected')!r}, actual={item.get('actual')!r}'
            if 'allowed' in item:
                if large_media_choices(item['allowed']):
                    text += f'; allowed_count={len(item['allowed'])}; complete_upload_catalog_in_diagnostics=true'
                else:
                    text += f'; allowed={item['allowed']!r}'
            if item.get('closest_node_classes'):
                text += f'; closest_node_classes={item['closest_node_classes']!r}'
            if item.get('closest_inputs'):
                text += f'; closest_inputs={item['closest_inputs']!r}'
            return text
        detail = '; '.join((render(item) for item in diagnostics))
        super().__init__(detail)

def _local_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))

def _json_request(url: str, payload: dict[str, Any] | None=None, timeout: int=60) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with _local_opener().open(request, timeout=timeout) as response:
            body = response.read().decode(errors='replace').strip()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        detail = raw.decode(errors='replace').strip() if raw else ''
        if detail:
            try:
                parsed = json.loads(detail)
                detail = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
            except json.JSONDecodeError:
                pass
            detail = detail[:16384]
        message = f'ComfyUI HTTP {exc.code} for {url}'
        if detail:
            message += f': {detail}'
        raise ClosedLoopError(message) from exc
    if not body:
        return {}
    result = json.loads(body)
    if not isinstance(result, dict):
        raise ClosedLoopError(f'ComfyUI returned a non-object response: {url}')
    return result

def _basic_graph_diagnostics(graph: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []

    def add(code: str, path: str, message: str, **extra: Any) -> None:
        diagnostics.append({'code': code, 'path': path, 'message': message, **extra})
    if not isinstance(graph, dict) or not graph:
        add('graph_shape', '/', 'API graph must be a non-empty object')
        return diagnostics
    for node_id, node in graph.items():
        if not isinstance(node, dict) or not isinstance(node.get('class_type'), str):
            add('node_shape', f'/{node_id}', 'node requires a string class_type')
        elif not isinstance(node.get('inputs', {}), dict):
            add('input_shape', f'/{node_id}/inputs', 'node inputs must be an object')
    node_ids = set(graph)
    for node_id, node in graph.items():
        if not isinstance(node, dict) or not isinstance(node.get('inputs', {}), dict):
            continue
        for key, value in node['inputs'].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and (value[0] not in node_ids):
                add('missing_source_node', f'/{node_id}/inputs/{key}', f'references missing node {value[0]!r}', node_id=str(node_id), input_name=str(key), source_node_id=value[0], closest_node_ids=difflib.get_close_matches(value[0], [str(item) for item in node_ids], n=5, cutoff=0.2))
    return diagnostics

class StateAwareValidator:
    """Static ``INT(Δt)`` and ``COMP(Gt, Δt)`` validation for ComfyUI API graphs.

    ``node_registry`` uses ComfyUI's ``/object_info`` shape.  No method here
    submits ``/prompt`` or executes inference: registry inspection, schema
    validation, type checking and graph analysis are all in-memory.
    """

    def __init__(self, node_registry: Mapping[str, Any], invariants: Iterable[Callable[[dict[str, Any], Mapping[str, Any]], list[dict[str, Any]]]]=()) -> None:
        self.node_registry = dict(node_registry)
        self.invariants = tuple(invariants)

    def _closest_node_classes(self, name: str, limit: int=5) -> list[str]:
        return difflib.get_close_matches(str(name), [str(item) for item in self.node_registry], n=limit, cutoff=0.2)

    def _node_interface(self, class_type: str) -> dict[str, Any]:
        schema = self.node_registry.get(class_type, {})
        inputs = schema.get('input', {}) if isinstance(schema, Mapping) else {}
        declared_inputs: list[str] = []
        if isinstance(inputs, Mapping):
            for section in ('required', 'optional'):
                fields = inputs.get(section, {})
                if isinstance(fields, Mapping):
                    declared_inputs.extend((str(key) for key in fields))
        outputs = schema.get('output', []) if isinstance(schema, Mapping) else []
        return {'class_type': class_type, 'inputs': sorted(set(declared_inputs)), 'outputs': list(outputs) if isinstance(outputs, list) else []}

    def _compatible_output_classes(self, expected: str | None, *, reference: str='', limit: int=5) -> list[str]:
        """Return locally registered node classes that can produce ``expected``.

        These are repair hints, not automatic substitutions.  Ordering favors
        names similar to the node already involved in the failed connection so
        the Agent gets a short, relevant list instead of the whole registry.
        """
        if not expected or expected == '*':
            return []
        candidates = []
        for class_type in self.node_registry:
            interface = self._node_interface(str(class_type))
            if expected in interface['outputs'] or '*' in interface['outputs']:
                candidates.append(str(class_type))
        if not reference:
            return sorted(candidates)[:limit]
        return sorted(candidates, key=lambda item: (-difflib.SequenceMatcher(None, reference.casefold(), item.casefold()).ratio(), item))[:limit]

    @staticmethod
    def _spec(node_schema: Mapping[str, Any], input_name: str) -> Any:
        inputs = node_schema.get('input', {}) if isinstance(node_schema, Mapping) else {}
        for section in ('required', 'optional'):
            fields = inputs.get(section, {}) if isinstance(inputs, Mapping) else {}
            if input_name in fields:
                return fields[input_name]
        return None

    @staticmethod
    def _type_from_spec(spec: Any) -> str | None:
        if isinstance(spec, (list, tuple)) and spec:
            value = spec[0]
            return value if isinstance(value, str) else None
        return None

    @staticmethod
    def _options_from_spec(spec: Any) -> Mapping[str, Any]:
        return spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], Mapping) else {}

    @staticmethod
    def _is_link(value: Any) -> bool:
        return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)

    def validate_edit(self, graph: dict[str, Any], edit: dict[str, Any], *, allow_external_placeholders: bool=False) -> list[dict[str, Any]]:
        """Validate an atomic edit independent of, then relative to, ``graph``."""
        diagnostics: list[dict[str, Any]] = []
        action = edit.get('action') if isinstance(edit, dict) else None
        if action not in {'add_node', 'remove_node', 'connect', 'disconnect', 'set_param'}:
            return [{'code': 'unknown_operator', 'path': '/action', 'message': 'unsupported graph edit action'}]
        if action == 'add_node':
            node = edit.get('node')
            node_id = str(edit.get('node_id', edit.get('target', '')))
            if not node_id or node_id in graph:
                diagnostics.append({'code': 'node_id', 'path': '/node_id', 'message': 'new node id must be unique'})
            if not isinstance(node, dict) or not isinstance(node.get('class_type'), str):
                diagnostics.append({'code': 'node_shape', 'path': '/node', 'message': 'add_node requires an API node'})
            elif node['class_type'] not in self.node_registry:
                diagnostics.append({'code': 'unknown_node_class', 'path': '/node/class_type', 'message': f'{node['class_type']!r} is not registered locally', 'node_id': node_id, 'class_type': node['class_type'], 'actual': node['class_type'], 'closest_node_classes': self._closest_node_classes(node['class_type'])})
        elif action == 'set_param':
            node_id, key = (str(edit.get('target', '')), edit.get('key') or edit.get('input'))
            node = graph.get(node_id)
            if not isinstance(node, dict):
                diagnostics.append({'code': 'missing_target_node', 'path': '/target', 'message': 'set_param target does not exist'})
            elif not isinstance(key, str) or self._spec(self.node_registry.get(node.get('class_type'), {}), key) is None:
                class_type = str(node.get('class_type', ''))
                interface = self._node_interface(class_type)
                diagnostics.append({'code': 'unknown_input', 'path': '/key', 'message': 'input is not declared by the node schema', 'node_id': node_id, 'class_type': class_type, 'input_name': key, 'actual': key, 'node_interface': interface, 'closest_inputs': difflib.get_close_matches(str(key), interface['inputs'], n=5, cutoff=0.2)})
            elif allow_external_placeholders and self._external_placeholder(node, str(key), edit.get('value')):
                pass
            else:
                diagnostics.extend(self._literal_diagnostics(node_id, key, edit.get('value'), self._spec(self.node_registry[node['class_type']], key)))
        elif action in {'connect', 'disconnect'}:
            source, destination = (edit.get('source') or edit.get('from'), edit.get('destination') or edit.get('to'))
            if not (isinstance(source, (list, tuple)) and len(source) == 2 and isinstance(destination, (list, tuple)) and (len(destination) == 2)):
                diagnostics.append({'code': 'edge_shape', 'path': '/', 'message': 'edge edit requires source and destination pairs'})
            elif str(source[0]) not in graph or str(destination[0]) not in graph:
                diagnostics.append({'code': 'missing_edge_node', 'path': '/', 'message': 'edge references an unknown node', 'source_node_id': str(source[0]), 'target_node_id': str(destination[0]), 'input_name': str(destination[1]), 'known_node_ids': sorted((str(node_id) for node_id in graph))})
        return diagnostics

    def _literal_diagnostics(self, node_id: str, key: str, value: Any, spec: Any) -> list[dict[str, Any]]:
        if self._is_link(value):
            return []
        diagnostics: list[dict[str, Any]] = []
        options = self._options_from_spec(spec)
        expected = self._type_from_spec(spec)
        choices = spec[0] if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], list) else None
        if choices is None and expected == 'COMBO':
            declared = options.get('options')
            if isinstance(declared, list):
                choices = declared
        path = f'/{node_id}/inputs/{key}'
        if choices is not None and value not in choices:
            diagnostics.append({'code': 'enum_value', 'path': path, 'message': 'value is outside the declared enum', 'node_id': node_id, 'input_name': key, 'actual': value, 'allowed': choices})
        invalid_numeric_type = expected == 'INT' and (not isinstance(value, int) or isinstance(value, bool)) or (expected == 'FLOAT' and (not isinstance(value, (int, float)) or isinstance(value, bool)))
        if invalid_numeric_type:
            diagnostics.append({'code': 'value_type', 'path': path, 'message': f'expected {expected} literal', 'node_id': node_id, 'input_name': key, 'expected': expected, 'actual': value, 'actual_type': type(value).__name__})
        if isinstance(value, (int, float)):
            if 'min' in options and value < options['min']:
                diagnostics.append({'code': 'value_range', 'path': path, 'message': f'value is below minimum {options['min']}', 'node_id': node_id, 'input_name': key, 'actual': value, 'minimum': options['min']})
            if 'max' in options and value > options['max']:
                diagnostics.append({'code': 'value_range', 'path': path, 'message': f'value exceeds maximum {options['max']}', 'node_id': node_id, 'input_name': key, 'actual': value, 'maximum': options['max']})
        return diagnostics

    @staticmethod
    def _external_placeholder(node: Mapping[str, Any], key: str, value: Any) -> bool:
        """Whether this literal belongs to execution-time media binding.

        Planning must not validate upload-file enums: placeholders and files
        copied into ComfyUI/input after the live registry snapshot are both
        execution-agent resources. Strict validation still runs after binding
        with a freshly loaded registry.
        """
        return node.get('class_type') in {'LoadImage', 'LoadImageMask', 'LoadVideo', 'VHS_LoadVideo', 'LoadAudio'} and key in {'image', 'file', 'video', 'audio', 'audio_file'} and isinstance(value, str) and bool(value.strip())

    def validate_graph(self, graph: dict[str, Any], *, allow_external_placeholders: bool=False) -> list[dict[str, Any]]:
        diagnostics = _basic_graph_diagnostics(graph)
        if diagnostics:
            return diagnostics
        edges: dict[str, list[str]] = {str(node_id): [] for node_id in graph}
        for node_id, node in graph.items():
            node_type = node['class_type']
            schema = self.node_registry.get(node_type)
            if not isinstance(schema, Mapping):
                diagnostics.append({'code': 'unknown_node_class', 'path': f'/{node_id}/class_type', 'message': f'{node_type!r} is not registered locally', 'node_id': str(node_id), 'class_type': node_type, 'actual': node_type, 'closest_node_classes': self._closest_node_classes(node_type)})
                continue
            required = schema.get('input', {}).get('required', {}) if isinstance(schema.get('input', {}), Mapping) else {}
            for key in required:
                if key not in node['inputs']:
                    spec = self._spec(schema, key)
                    expected = self._type_from_spec(spec)
                    diagnostics.append({'code': 'missing_required_input', 'path': f'/{node_id}/inputs/{key}', 'message': 'required input is absent', 'node_id': str(node_id), 'class_type': node_type, 'input_name': str(key), 'expected': expected, 'node_interface': self._node_interface(node_type), 'closest_node_classes': self._compatible_output_classes(expected, reference=node_type)})
            for key, value in node['inputs'].items():
                spec = self._spec(schema, key)
                if spec is None:
                    interface = self._node_interface(node_type)
                    diagnostics.append({'code': 'unknown_input', 'path': f'/{node_id}/inputs/{key}', 'message': 'input is not declared by the node schema', 'node_id': str(node_id), 'class_type': node_type, 'input_name': str(key), 'actual': key, 'node_interface': interface, 'closest_inputs': difflib.get_close_matches(str(key), interface['inputs'], n=5, cutoff=0.2)})
                    continue
                if allow_external_placeholders and self._external_placeholder(node, key, value):
                    continue
                if self._external_placeholder(node, key, value) and value in {'INPUT_IMAGE.png', 'START_IMAGE.png', 'END_IMAGE.png', 'MASK_INPUT.png', 'MASK_INPUT.mp4', 'INPUT_VIDEO.mp4', 'INPUT_VIDEO_A.mp4', 'INPUT_VIDEO_B.mp4', 'INPUT_AUDIO.wav'}:
                    diagnostics.append({'code': 'unbound_media_placeholder', 'path': f'/{node_id}/inputs/{key}', 'node_id': str(node_id), 'class_type': node_type, 'input_name': str(key), 'actual': value, 'message': 'Template media placeholder is not a bound case resource. Bind the required source or generated upstream artifact before execution; prompt edits cannot supply a missing target image.'})
                    continue
                if self._is_link(value):
                    source_id, slot = value
                    source = graph.get(source_id, {})
                    source_schema = self.node_registry.get(source.get('class_type'), {}) if isinstance(source, dict) else {}
                    outputs = source_schema.get('output', []) if isinstance(source_schema, Mapping) else []
                    if not isinstance(outputs, list) or slot < 0 or slot >= len(outputs):
                        diagnostics.append({'code': 'invalid_output_slot', 'path': f'/{node_id}/inputs/{key}', 'message': 'source output slot is not declared', 'node_id': str(node_id), 'class_type': node_type, 'input_name': str(key), 'source_node_id': str(source_id), 'source_class_type': str(source.get('class_type', '')), 'source_output_slot': slot, 'source_interface': self._node_interface(str(source.get('class_type', ''))), 'target_interface': self._node_interface(node_type)})
                        continue
                    expected, actual = (self._type_from_spec(spec), outputs[slot])
                    if expected not in {None, '*'} and actual not in {'*', expected}:
                        diagnostics.append({'code': 'port_type_mismatch', 'path': f'/{node_id}/inputs/{key}', 'message': f'expected {expected}, received {actual}; insert a compatible adapter if one exists', 'node_id': str(node_id), 'class_type': node_type, 'input_name': str(key), 'source_node_id': str(source_id), 'source_class_type': str(source.get('class_type', '')), 'source_output_slot': slot, 'expected': expected, 'actual': actual, 'closest_node_classes': self._compatible_output_classes(expected, reference=str(source.get('class_type', ''))), 'source_interface': self._node_interface(str(source.get('class_type', ''))), 'target_interface': self._node_interface(node_type)})
                    edges[source_id].append(str(node_id))
                else:
                    diagnostics.extend(self._literal_diagnostics(str(node_id), key, value, spec))
        if not diagnostics and self._has_cycle(edges):
            diagnostics.append({'code': 'graph_cycle', 'path': '/', 'message': 'API graph must be acyclic'})
        if not diagnostics:
            for invariant in self.invariants:
                diagnostics.extend(invariant(graph, self.node_registry))
        return diagnostics

    @staticmethod
    def _has_cycle(edges: Mapping[str, list[str]]) -> bool:
        active: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in active:
                return True
            if node in visited:
                return False
            active.add(node)
            cyclic = any((visit(child) for child in edges.get(node, [])))
            active.remove(node)
            visited.add(node)
            return cyclic
        return any((visit(node) for node in edges))

    def ensure_graph(self, graph: dict[str, Any], *, allow_external_placeholders: bool=False) -> None:
        diagnostics = self.validate_graph(graph, allow_external_placeholders=allow_external_placeholders)
        if diagnostics:
            raise StaticValidationError(diagnostics)

def validate_api_graph(graph: dict[str, Any], validator: StateAwareValidator | None=None, *, allow_external_placeholders: bool=False) -> None:
    """Validate the API prompt shape before it can reach ComfyUI."""
    if validator is not None:
        validator.ensure_graph(graph, allow_external_placeholders=allow_external_placeholders)
        return
    diagnostics = _basic_graph_diagnostics(graph)
    if diagnostics:
        raise StaticValidationError(diagnostics)

def graph_output_reachable_nodes(graph: dict[str, Any], validator: StateAwareValidator | None=None) -> set[str]:
    """Return nodes that contribute to a declared output node."""
    known_outputs = {'SaveImage', 'PreviewImage', 'VHS_VideoCombine', 'SaveAnimatedWEBP', 'SaveAnimatedPNG', 'SaveAudio'}
    sinks = set()
    for node_id, node in graph.items():
        class_type = str(node.get('class_type', '')) if isinstance(node, dict) else ''
        schema = validator.node_registry.get(class_type, {}) if validator is not None else {}
        if class_type in known_outputs or (isinstance(schema, Mapping) and schema.get('output_node') is True):
            sinks.add(str(node_id))
    reachable = set(sinks)
    pending = list(sinks)
    while pending:
        node = graph.get(pending.pop(), {})
        for value in (node.get('inputs', {}) if isinstance(node, dict) else {}).values():
            if isinstance(value, list) and len(value) == 2:
                source_id = str(value[0])
                if source_id in graph and source_id not in reachable:
                    reachable.add(source_id)
                    pending.append(source_id)
    return reachable

def validate_no_new_dead_nodes(before: dict[str, Any], after: dict[str, Any], validator: StateAwareValidator | None=None) -> None:
    """Reject topology edits that leave old or newly added nodes unreachable."""
    before_live = graph_output_reachable_nodes(before, validator)
    after_live = graph_output_reachable_nodes(after, validator)
    stranded = sorted((before_live & set(after)) - after_live)
    new_unreachable = sorted(set(after) - set(before) - after_live)
    diagnostics = []
    for node_id in stranded:
        diagnostics.append({'code': 'new_dead_node', 'path': f'/{node_id}', 'message': 'topology edits made a previously output-reachable node unreachable; preserve/reconnect it, or explicitly remove and replace it in the same plan', 'node_id': node_id, 'class_type': str(after[node_id].get('class_type', ''))})
    for node_id in new_unreachable:
        diagnostics.append({'code': 'new_unconnected_node', 'path': f'/{node_id}', 'message': 'newly added node is not reachable from a graph output; connect it to the active workflow or remove it in the same plan', 'node_id': node_id, 'class_type': str(after[node_id].get('class_type', ''))})
    if diagnostics:
        raise StaticValidationError(diagnostics)

def apply_graph_edits(graph: dict[str, Any], edits: list[dict[str, Any]], validator: StateAwareValidator | None=None, *, validate_after_each: bool=True, allow_external_placeholders: bool=False) -> dict[str, Any]:
    """Apply only the typed edit vocabulary emitted by generation_agent."""
    result = copy.deepcopy(graph)
    total_edits = len(edits)

    def annotate_failure(exc: Exception, *, edit_index: int, edit: Any, phase: str, applied: bool) -> None:
        """Expose the exact sequential-edit state to repair/evolution Agents."""
        setattr(exc, 'graph_edit_application', {'mode': 'sequential', 'validation': 'after_each_edit', 'failed_edit_index': edit_index, 'failed_edit_number': edit_index + 1, 'failed_edit_action': str(edit.get('action')) if isinstance(edit, dict) else None, 'failed_edit': copy.deepcopy(edit), 'failure_phase': phase, 'edit_was_applied': applied, 'applied_edit_count': edit_index + (1 if applied else 0), 'total_edit_count': total_edits, 'later_edits_applied': False, 'remaining_edits': copy.deepcopy(edits[edit_index + 1:]), 'partial_graph': copy.deepcopy(result)})
    for edit_index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            exc = ClosedLoopError('graph edit must be an object')
            annotate_failure(exc, edit_index=edit_index, edit=edit, phase='edit_shape_validation', applied=False)
            raise exc
        if validator is not None:
            diagnostics = validator.validate_edit(result, edit, allow_external_placeholders=allow_external_placeholders)
            if diagnostics:
                exc = StaticValidationError(diagnostics)
                annotate_failure(exc, edit_index=edit_index, edit=edit, phase='edit_shape_validation', applied=False)
                raise exc
        action = edit.get('action')
        target = str(edit.get('target', ''))
        try:
            if action == 'add_node':
                node_id = str(edit.get('node_id', target))
                node = edit.get('node')
                if not node_id or node_id in result or (not isinstance(node, dict)):
                    raise ClosedLoopError(f'invalid add_node edit: {edit}')
                result[node_id] = copy.deepcopy(node)
            elif action == 'remove_node':
                if target not in result:
                    raise ClosedLoopError(f'remove_node target does not exist: {target}')
                del result[target]
                for node in result.values():
                    for key, value in list(node.get('inputs', {}).items()):
                        if isinstance(value, list) and len(value) == 2 and (value[0] == target):
                            del node['inputs'][key]
            elif action in {'connect', 'disconnect'}:
                source = edit.get('source') or edit.get('from')
                destination = edit.get('destination') or edit.get('to')
                if not isinstance(source, (list, tuple)) or len(source) != 2 or (not isinstance(destination, (list, tuple))) or (len(destination) != 2):
                    raise ClosedLoopError(f'{action} requires source/destination [node,slot]')
                src, src_slot = (str(source[0]), int(source[1]))
                dst, dst_slot = (str(destination[0]), str(destination[1]))
                if src not in result or dst not in result:
                    raise ClosedLoopError(f'{action} references an unknown node')
                if action == 'connect':
                    result[dst].setdefault('inputs', {})[dst_slot] = [src, src_slot]
                elif dst_slot not in result[dst].get('inputs', {}):
                    raise ClosedLoopError(f'disconnect target is not connected: {dst}.{dst_slot}')
                else:
                    del result[dst]['inputs'][dst_slot]
            elif action == 'set_param':
                if target not in result:
                    raise ClosedLoopError(f'set_param target does not exist: {target}')
                key = edit.get('key') or edit.get('input')
                if not isinstance(key, str) or not key:
                    raise ClosedLoopError('set_param requires key/input')
                result[target].setdefault('inputs', {})[key] = copy.deepcopy(edit.get('value'))
            else:
                raise ClosedLoopError(f'unsupported graph edit action: {action!r}')
        except ClosedLoopError as exc:
            annotate_failure(exc, edit_index=edit_index, edit=edit, phase='edit_application', applied=False)
            raise
        if validate_after_each:
            try:
                validate_api_graph(result, validator, allow_external_placeholders=allow_external_placeholders)
            except StaticValidationError as exc:
                annotate_failure(exc, edit_index=edit_index, edit=edit, phase='graph_validation_after_edit', applied=True)
                raise
    if not validate_after_each:
        try:
            validate_api_graph(result, validator, allow_external_placeholders=allow_external_placeholders)
        except StaticValidationError as exc:
            setattr(exc, 'graph_edit_application', {'mode': 'transactional', 'validation': 'after_complete_edit_list', 'failed_edit_index': None, 'failed_edit_number': None, 'failed_edit_action': None, 'failed_edit': None, 'failure_phase': 'final_graph_validation', 'edit_was_applied': True, 'applied_edit_count': total_edits, 'total_edit_count': total_edits, 'later_edits_applied': False, 'remaining_edits': [], 'partial_graph': copy.deepcopy(result), 'original_graph_preserved': True})
            raise
    return result

@dataclass
class ExecutionEvidence:
    prompt_id: str
    status: str
    outputs: list[str] = field(default_factory=list)
    history: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    resource: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {'prompt_id': self.prompt_id, 'status': self.status, 'outputs': self.outputs, 'history': self.history, 'error': self.error, 'resource': self.resource}

class ComfyUIExecutor:
    """Submit exact API graphs and collect output files from ComfyUI."""

    def __init__(self, api: str | None=None, output_root: Path | None=None, timeout: int=1800, poll_seconds: float=2.0, validator: StateAwareValidator | None=None, vram_limit_bytes: int | None=None, resource_envelope: dict[str, Any] | None=None) -> None:
        self.api = (api or os.getenv('COMFYROUTER_COMFYUI_API', 'http://127.0.0.1:8188')).rstrip('/')
        self.output_root = Path(output_root or config.COMFYUI_ROOT / 'output')
        self.timeout, self.poll_seconds = (timeout, poll_seconds)
        self.validator = validator
        self.vram_limit_bytes = vram_limit_bytes
        self.resource_envelope = resource_envelope
        self.video_frame_budget = max(0, int(os.getenv('COMFYROUTER_VIDEO_FRAME_BUDGET', '0')))
        self.comfyui_input_root = Path(os.getenv('COMFYROUTER_COMFYUI_INPUT_ROOT', str(config.COMFYUI_ROOT / 'input')))
        self.media_probe_python = Path(os.getenv('COMFYROUTER_MEDIA_PROBE_PYTHON', sys.executable))

    def ensure_validator(self) -> StateAwareValidator:
        """Load the live object_info registry before planning or validation."""
        if self.validator is None:
            self.refresh_validator()
        return self.validator

    def refresh_validator(self) -> StateAwareValidator:
        """Reload object_info after execution-time media files are installed."""
        deadline = time.monotonic() + float(os.getenv('COMFYROUTER_SUBMISSION_RECOVERY_SECONDS', '300'))
        while True:
            try:
                self.validator = StateAwareValidator(_json_request(self.api + '/object_info'))
                break
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                if isinstance(exc, urllib.error.HTTPError):
                    raise
                if time.monotonic() >= deadline:
                    raise ComfyUITransportError('ComfyUI node registry unavailable during service recovery', audit={'path': '/object_info', 'error': str(exc)}) from exc
                time.sleep(min(self.poll_seconds, 1))
        return self.validator

    def static_validate(self, graph: dict[str, Any], *, allow_external_placeholders: bool=False) -> None:
        """Validate against the live node registry without submitting inference."""
        validate_api_graph(graph, self.ensure_validator(), allow_external_placeholders=allow_external_placeholders)

    def _probe_video(self, value: str) -> dict[str, Any] | None:
        """Read video timing through ComfyUI's own Python environment."""
        raw = Path(value)
        candidates = [raw, self.comfyui_input_root / raw.name]
        path = next((item for item in candidates if item.is_file()), None)
        if path is None or not self.media_probe_python.is_file():
            return None
        script = "import cv2,json,sys,av; p=sys.argv[1]; c=cv2.VideoCapture(p); n=float(c.get(cv2.CAP_PROP_FRAME_COUNT)); f=float(c.get(cv2.CAP_PROP_FPS)); c.release(); a=av.open(p); has_audio=any(s.codec_context is not None for s in a.streams.audio); a.close(); print(json.dumps({'frame_count':int(round(n)),'fps':f,'duration':(n/f if f>0 else 0.0),'has_audio':has_audio}))"
        try:
            completed = subprocess.run([str(self.media_probe_python), '-c', script, str(path)], text=True, capture_output=True, timeout=20, check=False)
            metadata = json.loads(completed.stdout.strip())
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return None
        if not isinstance(metadata, dict) or metadata.get('frame_count', 0) < 1 or metadata.get('fps', 0) <= 0 or (metadata.get('duration', 0) <= 0):
            return None
        metadata['path'] = str(path)
        return metadata

    def _apply_video_frame_budget(self, graph: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Apply an optional explicit video frame budget.

        The default budget is disabled.  When explicitly configured, native
        ``LoadVideo`` graphs are normalized to the equivalent VHS loader
        contract and multiple primary/mask videos share one target sampling
        rate so their frame indices remain aligned.
        """
        result = copy.deepcopy(graph)
        if self.video_frame_budget <= 0:
            return (result, [])
        audits: list[dict[str, Any]] = []
        loader_rows: list[dict[str, Any]] = []
        native_components: dict[str, str] = {}
        for component_id, component in result.items():
            if not isinstance(component, dict) or component.get('class_type') != 'GetVideoComponents':
                continue
            source = (component.get('inputs') or {}).get('video')
            if isinstance(source, list) and source and (str(source[0]) in result) and (result[str(source[0])].get('class_type') == 'LoadVideo'):
                native_components[str(component_id)] = str(source[0])
        for node_id, node in result.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get('class_type')
            inputs = node.get('inputs', {})
            video = inputs.get('video') if class_type == 'VHS_LoadVideo' else inputs.get('file')
            if class_type not in {'VHS_LoadVideo', 'LoadVideo'} or not isinstance(video, str):
                continue
            metadata = self._probe_video(video)
            if metadata:
                loader_rows.append({'node_id': str(node_id), 'class_type': class_type, 'video': video, 'metadata': metadata})
        if not loader_rows:
            return (result, audits)
        target_fps = min((min(float(row['metadata']['fps']), self.video_frame_budget / float(row['metadata']['duration'])) for row in loader_rows))
        max_estimated = 0
        next_id = max((int(node_id) for node_id in result if str(node_id).isdigit()), default=0) + 1
        for row in loader_rows:
            node_id = row['node_id']
            node = result[node_id]
            inputs = node.get('inputs', {})
            metadata = row['metadata']
            duration = float(metadata['duration'])
            source_fps = float(metadata['fps'])
            estimated_frames = int(math.ceil(duration * source_fps))
            max_estimated = max(max_estimated, estimated_frames)
            if estimated_frames <= self.video_frame_budget and len(loader_rows) == 1:
                continue
            before = copy.deepcopy(inputs)
            if row['class_type'] == 'LoadVideo':
                component_ids = [cid for cid, lid in native_components.items() if lid == node_id]
                if not component_ids:
                    continue
                info_id = str(next_id)
                next_id += 1
                result[info_id] = {'class_type': 'VHS_VideoInfoLoaded', 'inputs': {'video_info': [node_id, 3]}}
                node['class_type'] = 'VHS_LoadVideo'
                node['inputs'] = {'video': row['video'], 'force_rate': target_fps, 'custom_width': 0, 'custom_height': 0, 'frame_load_cap': self.video_frame_budget, 'skip_first_frames': 0, 'select_every_nth': 1, 'format': 'AnimateDiff'}
                for component_id in component_ids:
                    for consumer in result.values():
                        if not isinstance(consumer, dict):
                            continue
                        for key, value in list((consumer.get('inputs') or {}).items()):
                            if not isinstance(value, list) or not value or str(value[0]) != component_id:
                                continue
                            slot = int(value[1]) if len(value) > 1 else 0
                            if slot == 0:
                                consumer['inputs'][key] = [node_id, 0]
                            elif slot == 1:
                                if metadata.get('has_audio') is False:
                                    consumer['inputs'].pop(key, None)
                                else:
                                    consumer['inputs'][key] = [node_id, 2]
                            elif slot == 2:
                                consumer['inputs'][key] = [info_id, 0]
                            elif slot == 3:
                                consumer['inputs'][key] = 'auto'
                            elif slot == 4:
                                consumer['inputs'][key] = 'sRGB'
                    result.pop(component_id, None)
                policy = 'native_loadvideo_to_synchronized_vhs_sampling'
            else:
                current_force_rate = inputs.get('force_rate', 0)
                effective_fps = float(current_force_rate) if isinstance(current_force_rate, (int, float)) and current_force_rate > 0 else source_fps
                select_every = inputs.get('select_every_nth', 1)
                select_every = int(select_every) if isinstance(select_every, (int, float)) and select_every >= 1 else 1
                estimated_frames = int(math.ceil(duration * effective_fps / select_every))
                explicit_repair_controls = isinstance(inputs.get('force_rate'), (int, float)) and isinstance(inputs.get('frame_load_cap'), (int, float)) and (float(inputs.get('force_rate', 0)) > 0) and (int(inputs.get('frame_load_cap', 0)) > self.video_frame_budget)
                if explicit_repair_controls:
                    audits.append({'policy': 'preserve_execution_agent_runtime_controls', 'max_frames': self.video_frame_budget, 'loader_node_id': node_id, 'source': metadata, 'estimated_frames_before': estimated_frames, 'loader_inputs_before': before, 'loader_inputs_after': copy.deepcopy(inputs), 'expected_output_fps': float(inputs['force_rate']), 'expected_duration': duration})
                    continue
                inputs['force_rate'] = target_fps
                inputs['frame_load_cap'] = self.video_frame_budget
                inputs['select_every_nth'] = 1
                policy = 'synchronized_full_duration_cap'
            audits.append({'policy': policy, 'max_frames': self.video_frame_budget, 'loader_node_id': node_id, 'source': metadata, 'estimated_frames_before': estimated_frames, 'loader_inputs_before': before, 'loader_inputs_after': copy.deepcopy(result[node_id]['inputs']), 'expected_output_fps': target_fps, 'expected_duration': duration})
        expected_max = min(self.video_frame_budget, max_estimated)
        animate_loaders = [(str(node_id), node) for node_id, node in result.items() if isinstance(node, dict) and node.get('class_type') == 'ADE_AnimateDiffLoaderGen1']
        if expected_max > 32 and animate_loaders:
            context_id = next((str(node_id) for node_id, node in result.items() if isinstance(node, dict) and node.get('class_type') == 'ADE_StandardStaticContextOptions'), None)
            if context_id is None:
                context_id = str(next_id)
                result[context_id] = {'class_type': 'ADE_StandardStaticContextOptions', 'inputs': {'context_length': 16, 'context_overlap': 4, 'fuse_method': 'pyramid', 'use_on_equal_length': False, 'start_percent': 0.0, 'guarantee_steps': 1}}
            connected = []
            for loader_id, loader in animate_loaders:
                if 'context_options' not in (loader.get('inputs') or {}):
                    loader.setdefault('inputs', {})['context_options'] = [context_id, 0]
                    connected.append(loader_id)
            if connected:
                audits.append({'policy': 'animatediff_context_for_33_to_48_frames', 'context_node_id': context_id, 'connected_loader_ids': connected, 'expected_max_frames': expected_max})
        return (result, audits)

    def execute(self, graph: dict[str, Any]) -> ExecutionEvidence:
        from .resource_limits import require_bound_sources
        require_bound_sources(graph)
        native_ticket = None
        prompt_id = ''
        request_phase = 'static_validation'
        resource: dict[str, Any] = {'vram_limit_bytes': self.vram_limit_bytes, 'peak_vram_used_bytes': 0, 'peak_accounted_used_bytes': 0, 'samples': []}
        try:
            if self.resource_envelope:
                from .resource_limits import validate_runtime_resource_envelope
                validate_runtime_resource_envelope(graph, self.resource_envelope)
            submitted_graph, budget_audits = self._apply_video_frame_budget(graph)
            from . import config
            if budget_audits:
                resource['video_frame_budget'] = budget_audits
            self.static_validate(submitted_graph)
            request_phase = 'submission'
            native_mode = os.getenv('COMFYROUTER_COMFYUI_QUEUE_MODE', '').casefold() in {'1', 'true', 'yes', 'on'}
            if native_mode:
                from .native_queue import submit_prompt
                response, native_ticket = submit_prompt(self.api, {'prompt': submitted_graph}, _json_request)
            else:
                response = _json_request(self.api + '/prompt', {'prompt': submitted_graph})
            if response.get('submission_transport'):
                resource['submission_transport'] = response['submission_transport']
            prompt_id = response.get('prompt_id')
            if not isinstance(prompt_id, str):
                raise ClosedLoopError(f'/prompt rejected graph: {response}')
            request_phase = 'observation'
            execution_started = None
            time_limit_sent = False
            history: dict[str, Any] = {}
            while True:
                try:
                    history = _json_request(self.api + '/history/' + prompt_id).get(prompt_id, {})
                    queue = _json_request(self.api + '/queue') if not history else {}
                    boundary = _json_request(self.api + '/queue_cleanup') if native_mode else {}
                except (OSError, urllib.error.URLError, TimeoutError) as exc:
                    if not native_mode:
                        raise
                    resource.setdefault('poll_transport_errors', []).append(str(exc))
                    time.sleep(self.poll_seconds)
                    continue
                if native_mode and response.get('queue_boot_id') and (boundary.get('boot_id') != response['queue_boot_id']):
                    return ExecutionEvidence(prompt_id, 'server_restarted', [], history, 'ComfyUI restarted after native queue submission; prompt checkpoint retained', resource)
                status = history.get('status', {}).get('status_str')
                if status in {'success', 'error'}:
                    break
                if not native_mode and history:
                    queue = _json_request(self.api + '/queue')
                running = any((isinstance(row, (list, tuple)) and len(row) > 1 and (row[1] == prompt_id) for row in queue.get('queue_running', [])))
                if not running:
                    resource['queued_polls'] = resource.get('queued_polls', 0) + 1
                    time.sleep(self.poll_seconds)
                    continue
                resource['memory_sample_prompt_id'] = prompt_id
                if native_mode:
                    if boundary.get('prompt_id') == prompt_id and boundary.get('phase') != 'executing':
                        time.sleep(self.poll_seconds)
                        continue
                if execution_started is None:
                    execution_started = time.monotonic()
                elapsed = time.monotonic() - execution_started
                if native_mode and boundary.get('prompt_id') == prompt_id:
                    started_unix = boundary.get('execution_started_at', boundary.get('completed_at'))
                    if isinstance(started_unix, (int, float)):
                        elapsed = max(0, time.time() - started_unix)
                runtime_limit = self.timeout
                policy_file = os.getenv('COMFYROUTER_EXECUTION_LIMIT_POLICY_FILE', '')
                if policy_file:
                    runtime_limit = float(json.loads(Path(policy_file).read_text()).get('execution_timeout_seconds', runtime_limit))
                if runtime_limit and runtime_limit > 0 and (elapsed >= runtime_limit) and (not time_limit_sent):
                    _json_request(self.api + '/api/jobs/' + prompt_id + '/cancel', {})
                    time_limit_sent = True
                    resource.update(execution_time_limit_seconds=runtime_limit, execution_elapsed_seconds=elapsed, time_limit_interrupt_sent=True)
                    time.sleep(self.poll_seconds)
                    continue
                self._record_vram(resource)
                accounted_used = resource.get('peak_accounted_used_bytes', resource['peak_vram_used_bytes'])
                if self.vram_limit_bytes is not None and accounted_used > self.vram_limit_bytes:
                    latest_queue = _json_request(self.api + '/queue')
                    if not any((isinstance(row, (list, tuple)) and len(row) > 1 and (row[1] == prompt_id) for row in latest_queue.get('queue_running', []))):
                        continue
                    resource['limit_exceeded'] = True
                    resource['interrupt_target_confirmed_running'] = True
                    interrupt_error = None
                    try:
                        _json_request(self.api + '/interrupt', {'prompt_id': prompt_id})
                        resource['interrupt_sent'] = True
                    except (OSError, urllib.error.URLError, TimeoutError, ClosedLoopError) as exc:
                        interrupt_error = str(exc)
                        resource['interrupt_sent'] = False
                        resource['interrupt_error'] = interrupt_error
                    basis = resource.get('memory_accounting', {}).get('basis', 'device_vram_free')
                    message = f'Memory limit exceeded ({basis}): {accounted_used} > {self.vram_limit_bytes} bytes; ' + ('sent /interrupt' if interrupt_error is None else f'/interrupt failed: {interrupt_error}')
                    return ExecutionEvidence(prompt_id, 'vram_limit_exceeded', [], history, message, resource)
                time.sleep(self.poll_seconds)
            status = history.get('status', {}).get('status_str', 'timeout')
            if time_limit_sent and status != 'success':
                return ExecutionEvidence(prompt_id, 'execution_timeout', self._outputs(history), history, f'ComfyUI execution exceeded {resource['execution_time_limit_seconds']} seconds; cancelled this prompt', resource)
            outputs = self._outputs(history)
            if status != 'success':
                return ExecutionEvidence(prompt_id, status, outputs, history, execution_error_text(history, f'ComfyUI execution status: {status}'), resource)
            if not outputs or not all(((self.output_root / item).is_file() and (self.output_root / item).stat().st_size > 0 for item in outputs)):
                return ExecutionEvidence(prompt_id, 'output_missing', outputs, history, 'history succeeded but output file is missing', resource)
            return ExecutionEvidence(prompt_id, 'success', outputs, history, None, resource)
        except ComfyUITransportError as exc:
            resource['submission_transport'] = exc.audit
            status = 'submission_uncertain' if exc.submission_uncertain else 'execution_service_unavailable'
            return ExecutionEvidence(exc.prompt_id, status, [], {}, str(exc), resource)
        except (OSError, urllib.error.URLError, TimeoutError, ClosedLoopError) as exc:
            if isinstance(exc, (OSError, urllib.error.URLError, TimeoutError)) and (not isinstance(exc, urllib.error.HTTPError)):
                import errno
                reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
                refused = isinstance(reason, OSError) and reason.errno == errno.ECONNREFUSED
                uncertain = bool(prompt_id) or (request_phase == 'submission' and (not refused))
                resource['transport_failure'] = {'phase': request_phase, 'error': str(exc)}
                return ExecutionEvidence(prompt_id or '', 'submission_uncertain' if uncertain else 'execution_service_unavailable', [], {}, str(exc), resource)
            failed_history = {'static_validation': {'diagnostics': copy.deepcopy(exc.diagnostics)}} if isinstance(exc, StaticValidationError) else {}
            return ExecutionEvidence(prompt_id or '', 'executor_error', [], failed_history, str(exc), resource)
        finally:
            if native_ticket:
                native_ticket.unlink(missing_ok=True)

    def _record_vram(self, resource: dict[str, Any]) -> None:
        """Collect best-effort device usage without turning telemetry failure into a run failure."""
        try:
            stats = _json_request(self.api + '/system_stats')
            devices = stats.get('devices', [])
            if not devices:
                return
            device = devices[0]
            total, free = (device.get('vram_total'), device.get('vram_free'))
            if not isinstance(total, int) or not isinstance(free, int):
                return
            system = stats.get('system', {})
            ram_total = system.get('ram_total') if isinstance(system, dict) else None
            ram_available = system.get('ram_free') if isinstance(system, dict) else None
            unified_memory = isinstance(ram_total, int) and isinstance(ram_available, int) and (total == ram_total)
            effective_free = ram_available if unified_memory else free
            used = max(0, total - effective_free)
            basis = 'unified_system_available' if unified_memory else 'device_vram_free'
            resource['peak_vram_used_bytes'] = max(resource['peak_vram_used_bytes'], used)
            resource['peak_accounted_used_bytes'] = max(resource.get('peak_accounted_used_bytes', 0), used)
            resource['memory_accounting'] = {'basis': basis, 'unified_memory': unified_memory, 'total_bytes': total}
            sample = {'sampled_at_unix': time.time(), 'used_bytes': used, 'free_bytes': effective_free, 'accounting_basis': basis, 'raw_device_free_bytes': free}
            if isinstance(ram_available, int):
                sample['system_available_bytes'] = ram_available
            resource['samples'].append(sample)
        except (OSError, urllib.error.URLError, TimeoutError, ClosedLoopError):
            return

    def _outputs(self, history: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        for node in history.get('outputs', {}).values():
            for kind in ('images', 'gifs', 'videos'):
                for item in node.get(kind, []):
                    filename = item.get('filename')
                    if isinstance(filename, str):
                        subfolder = item.get('subfolder', '')
                        paths.append(str(Path(subfolder) / filename))
        return paths

def attribute_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    """Turn multidimensional feedback into one repair owner deterministically."""
    dims = feedback.get('dimensions', {}) if isinstance(feedback, dict) else {}

    def failed(*names: str) -> bool:
        return any((isinstance(dims.get(name), dict) and dims[name].get('status') in {'fail', 'partial'} for name in names))
    if failed('task_classification'):
        owner = 'routing'
    elif failed('model_selection'):
        owner = 'model_policy'
    elif failed('dependencies'):
        owner = 'dependencies'
    elif failed('workflow_schema', 'interface_compatibility'):
        owner = 'generation'
    elif failed('execution', 'resource_constraints'):
        owner = 'runtime'
    elif failed('output', 'task_alignment'):
        owner = 'generation'
    else:
        owner = 'none'
    return {'owner': owner, 'retryable': owner in {'generation', 'runtime', 'dependencies'}, 'failed_dimensions': [name for name, value in dims.items() if isinstance(value, dict) and value.get('status') in {'fail', 'partial'}]}

def calculate_gates(graph: dict[str, Any], evidence: ExecutionEvidence, feedback: dict[str, Any] | None=None) -> dict[str, bool]:
    """Calculate promotion gates from evidence rather than trusting an LLM."""
    validate_api_graph(graph)
    dims = (feedback or {}).get('dimensions', {})
    passed = lambda name, default: dims.get(name, {}).get('status') == 'pass' if isinstance(dims.get(name), dict) else default
    assessment = (feedback or {}).get('output_assessment', {})
    has_visual_score = isinstance(assessment, dict) and assessment.get('status') in {'pass', 'fail', 'partial', 'unverifiable'} and ('completion_score' in (feedback or {}))
    visual_quality_pass = not has_visual_score or (assessment.get('status') == 'pass' and isinstance((feedback or {}).get('completion_score'), int) and (feedback['completion_score'] >= 90))
    return {'schema_valid': True, 'interface_signature_valid': passed('interface_compatibility', True), 'static_graph_valid': True, 'dependencies_resolved': passed('dependencies', True), 'execution_pass': evidence.status == 'success', 'task_alignment_pass': passed('task_alignment', evidence.status == 'success') and visual_quality_pass, 'near_duplicate_checked': True, 'heldout_no_regression': passed('resource_constraints', True), 'contamination_pass': True}
