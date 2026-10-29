"""Dataset-neutral case execution: planning, media handoff, stages and feedback.

ComfyBench, ComfyBench2 and comparisons inject data/evaluation dependencies
into execute_case(). CLI compatibility lives in tools/run_comfybench_case.py.
"""
from __future__ import annotations
import copy
import fcntl
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4
from contextlib import contextmanager
from types import SimpleNamespace
from comfyrouter.execution_contract import CaseExecutionRequest, ExecutionHooks, ExecutionPolicy, current_execution, execution_context
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from comfyrouter import config
from comfyrouter.network import configure_external_proxy
from comfyrouter.agentic import AgentOrchestrator, AgentDecision, AgentOutputError, AgentTransportError, AgentProviderError, OpenAICompatibleClient
from comfyrouter.api_serial import serialized_api_request
from comfyrouter.operation_notebook import write_operation_notebook
from comfyrouter.composition import CompositionPlanError, validate_composition_plan
from comfyrouter.closed_loop import ComfyUIExecutor, ComfyUITransportError, ClosedLoopError, StaticValidationError, apply_graph_edits, validate_no_new_dead_nodes
from comfyrouter.media_inputs import MediaInputGate, MediaInputError, ordered_stage_resources
from comfyrouter.types import RequirementSpec
from comfyrouter.repair import repair_plan_limit as _default_repair_plan_limit, failure_kind, reconstruct_plan
from tools.run_dataset_direction_smoke import sample_video
config.load_env()
configure_external_proxy()
DATASET = config.COMFYBENCH_ROOT / 'dataset/benchmark'
REPORT_ROOT = config.EVOLUTION_DIR / 'comfybench_runs'
COMFYUI_INPUT = Path(os.getenv('COMFYROUTER_COMFYUI_INPUT_ROOT', str(config.COMFYUI_ROOT / 'input')))
COMFYUI_ROOT = config.COMFYUI_ROOT
COMFYUI_API = os.getenv('COMFYROUTER_COMFYUI_API', 'http://127.0.0.1:8188')
COMFYUI_SERIAL_LOCK = Path(os.getenv('COMFYROUTER_COMFYUI_LOCK', '/tmp/comfyrouter-comfyui.serial.lock'))
ANYSEARCH_CLI = Path(os.getenv('COMFYROUTER_ANYSEARCH_CLI', '/opt/anysearch/scripts/anysearch_cli.py'))
MASK_INTERMEDIATE_MARKER = 'mask_intermediate__'
UPSCALE_MODELS_BY_FACTOR = {2: 'RealESRGAN_x2plus.pth', 4: 'RealESRGAN_x4plus.pth'}

def repair_plan_limit() -> int:
    context = current_execution()
    if context and context.policy.planning_attempt_limit is not None:
        return context.policy.planning_attempt_limit
    return _default_repair_plan_limit()

def initialize_self_evolution_trial(report: dict) -> dict:
    """Start a fresh candidate ledger without inheriting stale trial rows."""
    state = report.setdefault('self_evolution', {})
    state.update({'enabled': not True, 'planner_called': False, 'mutation_called': False, 'execution_agent_required': True})
    state['candidates'] = []
    return state

def self_evolution_disabled() -> bool:
    return True

def comfyui_queue_mode() -> bool:
    """Use ComfyUI's native prompt queue for cross-case serialization."""
    return os.getenv('COMFYROUTER_COMFYUI_QUEUE_MODE', '').casefold() in {'1', 'true', 'yes', 'on'}

def notify_pipeline_ready(case_id: str, operation: str) -> None:
    """Tell an ordered pipeline that this case finished API preparation."""
    target = os.getenv('COMFYROUTER_PIPELINE_READY_FILE', '').strip()
    if not target:
        return
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps({'case_id': str(case_id), 'operation': operation, 'pid': os.getpid(), 'ready_at': time.time()}, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)

@contextmanager
def comfyui_serial_execution(case_id: str, operation: str):
    """Serialize the complete ComfyUI transaction across parallel case workers.

    The batch coordinator runs complete cases sequentially. This lock covers
    memory cleanup, prompt submission, output collection, and post-run cleanup
    so a worker cannot release another case's models or copy its outputs.
    ``flock`` is process-safe and is released automatically if a worker exits.
    """
    execution_order = os.getenv('COMFYROUTER_EXECUTION_ORDER_FILE', '').strip()
    if execution_order:
        order_path = Path(execution_order)
        while True:
            state = json.loads(order_path.read_text(encoding='utf-8'))
            if str(state.get('case_id')) == str(case_id):
                break
            if state.get('state') == 'finished':
                raise ClosedLoopError(f'execution queue finished before case {case_id}')
            time.sleep(1)
    if comfyui_queue_mode():
        yield
        return
    COMFYUI_SERIAL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    queue_dir_text = os.getenv('COMFYROUTER_STAGE_QUEUE_DIR', '').strip()
    ticket = None
    if queue_dir_text:
        queue_dir = Path(queue_dir_text)
        queue_dir.mkdir(parents=True, exist_ok=True)
        ticket = queue_dir / f'{time.time_ns():020d}_{os.getpid()}.json'
        temporary = ticket.with_suffix('.tmp')
        temporary.write_text(json.dumps({'case_id': case_id, 'operation': operation, 'pid': os.getpid(), 'queued_at': time.time(), 'status': 'waiting', 'destination': os.getenv('COMFYROUTER_STAGE_QUEUE_DESTINATION', '')}) + '\n')
        temporary.replace(ticket)
    with COMFYUI_SERIAL_LOCK.open('a+', encoding='utf-8') as handle:
        if ticket:
            while True:
                for pending in sorted(queue_dir.glob('*.json')):
                    try:
                        owner = json.loads(pending.read_text())['pid']
                        proc = Path(f'/proc/{owner}')
                        if not proc.exists() or proc.joinpath('stat').read_text().split(') ')[1].split()[0] in {'Z', 'X'}:
                            stale = queue_dir / 'abandoned'
                            stale.mkdir(exist_ok=True)
                            pending.replace(stale / pending.name)
                    except (OSError, ValueError, KeyError):
                        continue
                waiting = sorted(queue_dir.glob('*.json'))
                if waiting and waiting[0] == ticket:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        pass
                time.sleep(0.5)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({'case_id': str(case_id), 'operation': operation, 'pid': os.getpid(), 'started_at': time.time()}, ensure_ascii=False) + '\n')
        handle.flush()
        if ticket:
            row = json.loads(ticket.read_text())
            row.update(status='running', started_at=time.time())
            ticket.write_text(json.dumps(row) + '\n')
        try:
            yield
        finally:
            handle.seek(0)
            handle.truncate()
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            if ticket:
                ticket.unlink(missing_ok=True)

def append_evolution_candidate(report: dict, record: dict) -> None:
    """Record candidates safely on both success and exception paths."""
    report.setdefault('self_evolution', {}).setdefault('candidates', []).append(record)

def diagnostic_image_save(graph: dict, node_id: str) -> bool:
    """Recognize auxiliary images by graph semantics even after file renaming."""
    node = graph[node_id]
    meta = node.get('_meta') or {}
    if str(meta.get('output_role', '')).casefold() in {'mask', 'diagnostic', 'preview'}:
        return True
    if str(meta.get('title', '')).casefold().startswith('diagnostic:'):
        return True
    prefix = str(node.get('inputs', {}).get('filename_prefix', '')).casefold()
    if any((token in prefix for token in (MASK_INTERMEDIATE_MARKER, 'mask_preview', 'diagnostic_mask', 'intermediate_mask', 'control_preview', 'depth_preview'))):
        return True
    link = node.get('inputs', {}).get('images')
    seen = set()
    while isinstance(link, list) and len(link) == 2 and (str(link[0]) not in seen):
        upstream_id = str(link[0])
        seen.add(upstream_id)
        upstream = graph.get(upstream_id, {})
        kind = upstream.get('class_type')
        if kind == 'MaskToImage':
            return True
        passthrough = {'ImageScale': 'image', 'ImageScaleBy': 'image', 'ImageCrop': 'image'}
        if kind not in passthrough:
            break
        link = upstream.get('inputs', {}).get(passthrough[kind])
    return False

def split_primary_and_diagnostic_outputs(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Separate saved mask evidence from outputs eligible for scoring/handoff."""
    diagnostic_tokens = (MASK_INTERMEDIATE_MARKER, 'mask_preview', 'diagnostic_mask', 'intermediate_mask', 'control_preview', 'depth_preview')
    roles_by_directory = {}
    for path in paths:
        directory = path.parent
        if directory not in roles_by_directory:
            try:
                roles_by_directory[directory] = json.loads((directory / 'output_roles.json').read_text())
            except (OSError, ValueError):
                roles_by_directory[directory] = {}
    diagnostics = [path for path in paths if roles_by_directory[path.parent].get(path.name) == 'diagnostic' or any((token in path.name.casefold() for token in diagnostic_tokens))]
    primary = [path for path in paths if path not in diagnostics]
    return (primary, diagnostics)

def select_primary_output(paths: list[Path]) -> Path | None:
    primary, _ = split_primary_and_diagnostic_outputs(paths)
    output = primary[-1] if primary else None
    context = current_execution()
    if output is not None and context and context.hooks.output_observer:
        context.hooks.output_observer(output)
    return output

def parse_runtime_params(values: list[str]) -> list[dict]:
    """Parse explicit experiment-only parameter overrides."""
    edits = []
    for value in values:
        target, separator, raw = value.partition('=')
        node_id, dot, input_name = target.partition('.')
        if not separator or not dot or (not node_id) or (not input_name):
            raise ValueError(f'invalid --runtime-param {value!r}; expected NODE.INPUT=JSON_VALUE')
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f'invalid JSON value in --runtime-param {value!r}: {exc}') from exc
        edits.append({'op': 'set_param', 'action': 'set_param', 'node_id': node_id, 'target': node_id, 'input': input_name, 'key': input_name, 'value': parsed})
    return edits

def parse_stage_runtime_params(values: list[str]) -> list[dict]:
    """Parse explicit per-stage overrides for composed workflows."""
    edits = []
    for value in values:
        stage_id, separator, expression = value.partition(':')
        if not separator or not re.fullmatch('stage_[0-9]{2}', stage_id):
            raise ValueError(f'invalid --stage-param {value!r}; expected stage_XX:NODE.INPUT=JSON_VALUE')
        for edit in parse_runtime_params([expression]):
            edits.append({**edit, 'stage_id': stage_id})
    return edits

def apply_runtime_params(graph: dict, edits: list[dict], executor) -> dict:
    if not edits:
        return graph
    candidate = apply_graph_edits(graph, edits, executor.validator)
    executor.static_validate(candidate)
    return candidate

def requested_upscale_factor(instruction: str) -> int | None:
    """Return one explicit supported spatial scale such as 2x or 4倍."""
    factors = {int(match) for match in re.findall('(?<![\\d.])([24])\\s*(?:x|×|倍)(?![a-z0-9])', instruction, flags=re.IGNORECASE)}
    if len(factors) > 1:
        raise AgentOutputError('upscaling request contains conflicting explicit factors: ' + ', '.join((f'{factor}x' for factor in sorted(factors))))
    return next(iter(factors), None)

def normalize_upscale_model(graph: dict, instruction: str) -> list[dict]:
    """Bind an explicit 2x/4x request to the matching model in one leaf."""
    context = current_execution()
    if context and context.policy.frozen_graph:
        return []
    factor = requested_upscale_factor(instruction)
    if factor is None:
        return []
    model_name = UPSCALE_MODELS_BY_FACTOR[factor]
    changes = []
    for node_id, node in graph.items():
        if not isinstance(node, dict) or node.get('class_type') != 'UpscaleModelLoader':
            continue
        inputs = node.setdefault('inputs', {})
        previous = inputs.get('model_name')
        if previous == model_name:
            continue
        inputs['model_name'] = model_name
        changes.append({'node_id': str(node_id), 'factor': factor, 'previous_model': previous, 'model_name': model_name})
    return changes

def _geometry_adapter_node(graph: dict, title: str) -> tuple[str, dict] | None:
    """Find one explicitly marked geometry-adapter node."""
    matches = [(str(node_id), node) for node_id, node in graph.items() if isinstance(node, dict) and str((node.get('_meta') or {}).get('title', '')) == title]
    if len(matches) > 1:
        raise ClosedLoopError(f'geometry adapter title is ambiguous: {title!r}')
    return matches[0] if matches else None

def _aligned_canvas(size: int, multiple: int) -> tuple[int, int, int]:
    """Return aligned size and balanced padding before/after the content."""
    if size <= 0 or multiple <= 0:
        raise ValueError('geometry dimensions and alignment multiple must be positive')
    aligned = (size + multiple - 1) // multiple * multiple
    before = (aligned - size) // 2
    return (aligned, before, aligned - size - before)
FIXED_IMAGE_LATENT_CLASSES = {'EmptyFlux2LatentImage', 'EmptySD3LatentImage', 'EmptyLatentImage'}

def exact_image_geometry_requirement(instruction: str) -> dict:
    """Identify explicit geometry requirements that justify exact-size mode."""
    evidence = []
    if re.search('\\b\\d+\\s*(?:pixels?|px)\\b|\\d+\\s*像素', instruction, flags=re.IGNORECASE):
        evidence.append('explicit_pixel_count')
    factor = requested_upscale_factor(instruction)
    if factor is not None:
        evidence.append('explicit_upscale_factor')
    if re.search('\\b\\d{2,5}\\s*[x×]\\s*\\d{2,5}\\b', instruction, flags=re.IGNORECASE):
        evidence.append('explicit_resolution')
    if re.search('\\b(?:exact(?:ly)?|exact size|original dimensions?|same dimensions?)\\b|精确|严格尺寸|保持原(?:始)?尺寸|尺寸不变', instruction, flags=re.IGNORECASE):
        evidence.append('explicit_exactness_language')
    return {'required': bool(evidence), 'evidence': evidence, 'factor': factor}

def normalize_empty_alpha_mask_canvases(graph: dict) -> list[dict]:
    """Expand ComfyUI's 64x64 no-alpha sentinel to its image canvas.

    LoadImage's secondary MASK is a 64x64 zero tensor for RGB media, not a
    full-image mask. Only replace this proven empty sentinel when used as
    the destination of a spatial MaskComposite. Existing alpha masks and
    other nonempty segmentation outputs retain their semantics.
    """
    from PIL import Image
    audits = []
    canvases = {}
    for composite_id, node in list(graph.items()):
        if not isinstance(node, dict) or node.get('class_type') != 'MaskComposite':
            continue
        destination = node.get('inputs', {}).get('destination')
        if not (isinstance(destination, list) and len(destination) == 2 and (destination[1] == 1)):
            continue
        loader_id = str(destination[0])
        loader = graph.get(loader_id, {})
        if loader.get('class_type') != 'LoadImage':
            continue
        filename = loader.get('inputs', {}).get('image')
        if not isinstance(filename, str) or not filename:
            continue
        path = COMFYUI_INPUT / filename
        try:
            with Image.open(path) as image:
                if 'A' in image.getbands() or 'transparency' in image.info:
                    continue
                width, height = image.size
        except (OSError, ValueError):
            continue
        if (width, height) == (64, 64):
            continue
        if loader_id not in canvases:
            canvas_id = str(max([int(k) for k in graph if str(k).isdigit()] or [0]) + 1)
            graph[canvas_id] = {'class_type': 'SolidMask', 'inputs': {'value': 0.0, 'width': width, 'height': height}, '_meta': {'title': 'Full-size empty alpha mask canvas'}}
            canvases[loader_id] = canvas_id
        node['inputs']['destination'] = [canvases[loader_id], 0]
        audits.append({'node_id': str(composite_id), 'source_loader': loader_id, 'canvas_node': canvases[loader_id], 'width': width, 'height': height, 'reason': 'RGB LoadImage MASK is a 64x64 empty sentinel; use the source image canvas'})
    return audits

def normalize_exact_image_geometry(graph: dict, source: Path | None, instruction: str, *, multiple: int=8, overall_instruction: str | None=None, stage_operation: str | None=None, mask_source: Path | None=None) -> dict | None:
    """Align local-edit image/mask canvases; opt in to other exact-size modes."""
    context = current_execution()
    if context and context.policy.frozen_graph:
        return None
    empty_mask_canvases = normalize_empty_alpha_mask_canvases(graph)
    requirement = exact_image_geometry_requirement('\n'.join(filter(None, (instruction, overall_instruction))))
    image_pad = _geometry_adapter_node(graph, 'Pad image to VAE multiple')
    mask_canvas = _geometry_adapter_node(graph, 'Padded mask canvas')
    mask_align = _geometry_adapter_node(graph, 'Align mask with padded image')
    source_crop = _geometry_adapter_node(graph, 'Restore source dimensions')
    edit_nodes = (image_pad, mask_canvas, mask_align, source_crop)
    if any(edit_nodes) and (not all(edit_nodes)):
        raise ClosedLoopError('incomplete exact-geometry adapter in image-edit graph')
    if not requirement['required'] and (not all(edit_nodes)):
        return {'mode': 'normalize_empty_alpha_mask_canvas', 'corrections': empty_mask_canvases} if empty_mask_canvases else None
    geometry = media_file_geometry_contract(source, 'IMAGE')
    if not geometry:
        if all(edit_nodes):
            raise ClosedLoopError('local-edit geometry requires a readable source image')
        return None
    source_width = int(geometry['width'])
    source_height = int(geometry['height'])
    if all(edit_nodes):
        if mask_source is not None:
            mask_geometry = media_file_geometry_contract(mask_source, 'IMAGE')
            if not mask_geometry or (int(mask_geometry['width']), int(mask_geometry['height'])) != (source_width, source_height):
                raise ClosedLoopError(f'local-edit mask must match source-image dimensions {source_width}x{source_height}: {mask_source}')
        canvas_width, left, right = _aligned_canvas(source_width, multiple)
        canvas_height, top, bottom = _aligned_canvas(source_height, multiple)
        image_pad[1].setdefault('inputs', {}).update({'left': left, 'top': top, 'right': right, 'bottom': bottom, 'feathering': 0})
        mask_canvas[1].setdefault('inputs', {}).update({'width': canvas_width, 'height': canvas_height})
        mask_align[1].setdefault('inputs', {}).update({'x': left, 'y': top})
        source_crop[1].setdefault('inputs', {}).update({'width': source_width, 'height': source_height, 'x': left, 'y': top})
        return {'mode': 'preserve_source', 'alignment_reason': 'mandatory_local_edit_image_mask_alignment', 'requirement_evidence': requirement['evidence'], 'source_width': source_width, 'source_height': source_height, 'target_width': source_width, 'target_height': source_height, 'canvas_width': canvas_width, 'canvas_height': canvas_height, 'alignment_multiple': multiple, 'padding': {'left': left, 'top': top, 'right': right, 'bottom': bottom}}
    factor = requested_upscale_factor(instruction)
    if factor is not None and stage_operation is not None and (not stage_operation.endswith('/upscaling')):
        if stage_operation.endswith('/text_overlay'):
            return {'mode': 'preserve_stage_input', 'requirement_evidence': ['text_overlay_preserves_input_geometry'], 'stage_operation': stage_operation, 'source_width': source_width, 'source_height': source_height, 'target_width': source_width, 'target_height': source_height, 'referenced_upstream_factor': factor}
        factor = None
    if factor is not None:
        return {'mode': 'upscale', 'requirement_evidence': requirement['evidence'], 'source_width': source_width, 'source_height': source_height, 'target_width': source_width * factor, 'target_height': source_height * factor, 'factor': factor}
    output_crop = _geometry_adapter_node(graph, 'Restore requested output dimensions')
    outpaint_nodes = [(str(node_id), node) for node_id, node in graph.items() if isinstance(node, dict) and node.get('class_type') == 'ImagePadForOutpaint' and (str((node.get('_meta') or {}).get('title', '')) != 'Pad image to VAE multiple')]
    fixed_latents = [(str(node_id), node) for node_id, node in graph.items() if isinstance(node, dict) and node.get('class_type') in FIXED_IMAGE_LATENT_CLASSES]
    if fixed_latents and (not outpaint_nodes):
        if len(fixed_latents) != 1:
            raise ClosedLoopError('exact generation geometry requires one fixed image latent node')
        latent_id, latent_node = fixed_latents[0]
        canvas_width, left, right = _aligned_canvas(source_width, multiple)
        canvas_height, top, bottom = _aligned_canvas(source_height, multiple)
        latent_node.setdefault('inputs', {}).update({'width': canvas_width, 'height': canvas_height})
        if any((left, top, right, bottom)):
            if output_crop is None:
                primary_saves = [(str(node_id), node) for node_id, node in graph.items() if isinstance(node, dict) and node.get('class_type') == 'SaveImage' and (not diagnostic_image_save(graph, str(node_id)))]
                if len(primary_saves) != 1:
                    raise ClosedLoopError('exact generation geometry requires one primary SaveImage node')
                save_id, save_node = primary_saves[0]
                image_link = copy.deepcopy(save_node.setdefault('inputs', {}).get('images'))
                if not (isinstance(image_link, list) and len(image_link) == 2):
                    raise ClosedLoopError(f'primary SaveImage node {save_id!r} has no image connection')
                numeric_ids = [int(node_id) for node_id in graph if str(node_id).isdigit()]
                crop_id = str(max(numeric_ids, default=0) + 1)
                while crop_id in graph:
                    crop_id = str(int(crop_id) + 1)
                crop_node = {'class_type': 'ImageCrop', 'inputs': {'image': image_link}, '_meta': {'title': 'Restore requested output dimensions'}}
                graph[crop_id] = crop_node
                save_node['inputs']['images'] = [crop_id, 0]
                output_crop = (crop_id, crop_node)
            output_crop[1].setdefault('inputs', {}).update({'width': source_width, 'height': source_height, 'x': left, 'y': top})
        return {'mode': 'preserve_source_generation', 'requirement_evidence': requirement['evidence'], 'source_width': source_width, 'source_height': source_height, 'target_width': source_width, 'target_height': source_height, 'canvas_width': canvas_width, 'canvas_height': canvas_height, 'alignment_multiple': multiple, 'padding': {'left': left, 'top': top, 'right': right, 'bottom': bottom}, 'latent_node_id': latent_id, 'crop_node_id': output_crop[0] if output_crop else None}
    if outpaint_nodes:
        if len(outpaint_nodes) != 1:
            raise ClosedLoopError('exact extension geometry requires one ImagePadForOutpaint node')
        pad_id, pad_node = outpaint_nodes[0]
        pad_inputs = pad_node.setdefault('inputs', {})
        requested = {key: int(pad_inputs.get(key, 0)) for key in ('left', 'top', 'right', 'bottom')}
        positive_extensions = [value for value in requested.values() if value > 0]
        requested_feathering = int(pad_inputs.get('feathering', 0))
        effective_feathering = requested_feathering
        if positive_extensions and effective_feathering <= 0:
            effective_feathering = min(40, min(positive_extensions))
        pad_inputs['feathering'] = effective_feathering
        target_width = source_width + requested['left'] + requested['right']
        target_height = source_height + requested['top'] + requested['bottom']
        canvas_width, extra_left, extra_right = _aligned_canvas(target_width, multiple)
        canvas_height, extra_top, extra_bottom = _aligned_canvas(target_height, multiple)
        pad_inputs.update({'left': requested['left'] + extra_left, 'top': requested['top'] + extra_top, 'right': requested['right'] + extra_right, 'bottom': requested['bottom'] + extra_bottom})
        if output_crop is None:
            primary_saves = [(str(node_id), node) for node_id, node in graph.items() if isinstance(node, dict) and node.get('class_type') == 'SaveImage' and (not diagnostic_image_save(graph, str(node_id)))]
            if len(primary_saves) != 1:
                raise ClosedLoopError('exact extension geometry requires one primary SaveImage node')
            save_id, save_node = primary_saves[0]
            image_link = copy.deepcopy(save_node.setdefault('inputs', {}).get('images'))
            if not (isinstance(image_link, list) and len(image_link) == 2):
                raise ClosedLoopError(f'primary SaveImage node {save_id!r} has no image connection')
            numeric_ids = [int(node_id) for node_id in graph if str(node_id).isdigit()]
            crop_id = str(max(numeric_ids, default=0) + 1)
            while crop_id in graph:
                crop_id = str(int(crop_id) + 1)
            crop_node = {'class_type': 'ImageCrop', 'inputs': {'image': image_link}, '_meta': {'title': 'Restore requested output dimensions'}}
            graph[crop_id] = crop_node
            save_node['inputs']['images'] = [crop_id, 0]
            output_crop = (crop_id, crop_node)
        output_crop[1].setdefault('inputs', {}).update({'width': target_width, 'height': target_height, 'x': extra_left, 'y': extra_top})
        return {'mode': 'extension', 'requirement_evidence': requirement['evidence'], 'source_width': source_width, 'source_height': source_height, 'target_width': target_width, 'target_height': target_height, 'canvas_width': canvas_width, 'canvas_height': canvas_height, 'alignment_multiple': multiple, 'requested_extension': requested, 'requested_feathering': requested_feathering, 'effective_feathering': effective_feathering, 'alignment_padding': {'left': extra_left, 'top': extra_top, 'right': extra_right, 'bottom': extra_bottom}, 'pad_node_id': pad_id, 'crop_node_id': output_crop[0]}
    return None

def exact_image_output_validation(path: Path | None, contract: dict | None) -> dict | None:
    """Validate a materialized image before it can be scored or handed off."""
    if contract is None or contract.get('mode') == 'normalize_empty_alpha_mask_canvas':
        return None
    expected = (int(contract['target_width']), int(contract['target_height']))
    actual_geometry = media_file_geometry_contract(path, 'IMAGE')
    actual = None if not actual_geometry else (int(actual_geometry['width']), int(actual_geometry['height']))
    return {'passed': actual == expected, 'expected_width': expected[0], 'expected_height': expected[1], 'actual_width': actual[0] if actual else None, 'actual_height': actual[1] if actual else None, 'error': None if actual == expected else f'exact output geometry mismatch: expected {expected[0]}x{expected[1]}, ' + (f'got {actual[0]}x{actual[1]}' if actual else 'output is unreadable')}

def modality_for_step(step: dict, fallback: str) -> str:
    """Resolve a decomposed stage from its own media contract."""
    return {(None, 'image'): 't2i', (None, 'video'): 't2v', ('text', 'image'): 't2i', ('text', 'video'): 't2v', ('image', 'image'): 'i2i', ('image', 'video'): 'i2v', ('video', 'image'): 'v2i', ('video', 'video'): 'v2v', ('image', 'mask'): 'i2i', ('mask', 'image'): 'i2i', ('video', 'mask'): 'v2v', ('mask', 'video'): 'v2v'}.get((step.get('input_type'), step.get('output_type')), fallback)

def implementation_from_report(report: dict) -> tuple[AgentDecision, RequirementSpec, list[dict]]:
    """Reuse a persisted decomposition/classification checkpoint without new routing calls."""
    raw = report.get('classification') or {}
    decision = AgentDecision(task_family=str(raw.get('task_family', '')), subworkflow_sequence=tuple(raw.get('subworkflow_sequence') or ()), capabilities=tuple(raw.get('capabilities') or ()), priorities=tuple(raw.get('priorities') or ()), constraints=tuple(raw.get('constraints') or ()), explicit_model=raw.get('explicit_model'), confidence=str(raw.get('confidence', '')), proposal_ids=tuple(raw.get('proposal_ids') or ()), reason=str(raw.get('reason', '')))
    spec = RequirementSpec(str((report.get('requirement_spec') or {}).get('modality', '')), str((report.get('requirement_spec') or {}).get('task_family', decision.task_family)), tuple((report.get('requirement_spec') or {}).get('operations') or decision.subworkflow_sequence), tuple((report.get('requirement_spec') or {}).get('capabilities') or decision.capabilities), tuple((report.get('requirement_spec') or {}).get('priorities') or decision.priorities), tuple((report.get('requirement_spec') or {}).get('constraints') or decision.constraints), (report.get('requirement_spec') or {}).get('explicit_model', decision.explicit_model))
    implementation = copy.deepcopy(report.get('implementation_plan') or [])
    if not implementation:
        raise ClosedLoopError('checkpoint has no implementation_plan')
    return (decision, spec, implementation)

def spec_for_stage(step: dict) -> RequirementSpec:
    return RequirementSpec(str(step['modality']), str(step['task_family']), (str(step['operation']),), tuple(step.get('capabilities') or (step['operation'],)), tuple(step.get('priorities') or ()), tuple(step.get('constraints') or ()), step.get('explicit_model'))

def composition_stages(implementation: list[dict]) -> tuple:
    raw = []
    step_ids = {str((step.get('decomposition_step') or {}).get('step') or f'stage_{index:02d}'): f'stage_{index:02d}' for index, step in enumerate(implementation, 1)}
    for index, step in enumerate(implementation, 1):
        decomposition = step.get('decomposition_step') or {}
        item = {'stage_id': f'stage_{index:02d}', 'purpose': decomposition.get('description') or step['operation'], 'instruction': decomposition.get('description') or step['operation'], 'input_type': str(decomposition.get('input_type', 'image') or 'text').upper(), 'input_types': [str(value or 'text').upper() for value in decomposition.get('input_types', [decomposition.get('input_type', 'image')])], 'output_type': str(decomposition.get('output_type', 'image')).upper(), 'modality': step['modality']}
        if 'depends_on' in decomposition:
            item['depends_on'] = [step_ids.get(str(dependency), str(dependency)) for dependency in decomposition.get('depends_on') or ()]
        raw.append(item)
    return validate_composition_plan(raw)
MEDIA_INPUT_ADAPTERS = {'IMAGE': (('LoadImage', 'image'),), 'MASK': (('LoadImageMask', 'image'), ('LoadVideo', 'file'), ('VHS_LoadVideo', 'video')), 'VIDEO': (('LoadVideo', 'file'), ('VHS_LoadVideo', 'video')), 'AUDIO': (('LoadAudio', 'audio'), ('LoadAudio', 'audio_file'), ('LoadAudio', 'file'))}
MEDIA_PLACEHOLDERS = {'IMAGE': {'INPUT_IMAGE.png', 'START_IMAGE.png', 'END_IMAGE.png'}, 'MASK': {'MASK_INPUT.png', 'MASK_INPUT.mp4'}, 'VIDEO': {'INPUT_VIDEO.mp4', 'INPUT_VIDEO_A.mp4', 'INPUT_VIDEO_B.mp4'}, 'AUDIO': {'INPUT_AUDIO.wav'}}
MASK_SELECTED_SUBJECT_CONTRACT = {'mask_role': 'selected_subject', 'white_means': 'selected_subject', 'black_means': 'everything_else'}

def _next_graph_node_id(graph: dict) -> str:
    numeric = [int(node_id) for node_id in graph if str(node_id).isdigit()]
    return str(max(numeric, default=0) + 1)

def apply_mask_handoff_adapter(graph: dict, operation: str, mask_role: str='preserve_region') -> tuple[dict, dict | None]:
    """Adapt the canonical white-subject mask to a consumer's edit-mask contract."""
    context = current_execution()
    if False or (context and context.policy.frozen_graph):
        return (graph, None)
    if str(operation).casefold() != 'background_editing/replacement':
        return (graph, None)
    candidate = copy.deepcopy(graph)
    mask_loaders = media_input_nodes(candidate, 'MASK')
    image_loaders = media_input_nodes(candidate, 'IMAGE')
    video_loaders = media_input_nodes(candidate, 'VIDEO')
    stale_image_mask_loaders = [(node_id, input_name) for node_id, input_name in mask_loaders if candidate[node_id].get('class_type') == 'LoadImageMask']
    if video_loaders and (not image_loaders) and (len(stale_image_mask_loaders) == 1):
        invert_nodes = [(node_id, node) for node_id, node in candidate.items() if isinstance(node, dict) and node.get('class_type') == 'InvertMask']
        composite_nodes = [(node_id, node) for node_id, node in candidate.items() if isinstance(node, dict) and node.get('class_type') == 'ImageCompositeMasked']
        if len(invert_nodes) != 1 or len(composite_nodes) != 1:
            raise ClosedLoopError('video background replacement mask adapter requires one InvertMask and one ImageCompositeMasked node')
        loader_id, _ = stale_image_mask_loaders[0]
        old_ref = [str(loader_id), 0]
        candidate[loader_id] = {'class_type': 'LoadVideo', 'inputs': {'file': 'MASK_INPUT.mp4'}, '_meta': {'comfyrouter_adapter': 'external_mask_video_loader'}}
        components_id = _next_graph_node_id(candidate)
        mask_id = str(int(components_id) + 1)
        candidate[components_id] = {'class_type': 'GetVideoComponents', 'inputs': {'video': [loader_id, 0]}, '_meta': {'comfyrouter_adapter': 'external_mask_video_frames'}}
        candidate[mask_id] = {'class_type': 'ImageToMask', 'inputs': {'image': [components_id, 0], 'channel': 'red'}, '_meta': {'comfyrouter_adapter': 'external_selected_subject_mask'}}
        for node_id, node in candidate.items():
            if node_id in {loader_id, components_id, mask_id} or not isinstance(node, dict):
                continue
            for key, value in node.get('inputs', {}).items():
                if value == old_ref:
                    node['inputs'][key] = [mask_id, 0]
        invert_id, invert_node = invert_nodes[0]
        composite_id, composite_node = composite_nodes[0]
        invert_node.setdefault('inputs', {})['mask'] = [mask_id, 0]
        composite_node.setdefault('inputs', {})['mask'] = [mask_id, 0]
        return (candidate, {'status': 'applied', 'adapter': 'stale_image_mask_to_video_mask_handoff', 'source_contract': dict(MASK_SELECTED_SUBJECT_CONTRACT), 'consumer_contract': {'mask_role': 'editable_background', 'white_means': 'replace'}, 'nodes': {'mask_video_loader': loader_id, 'mask_video_components': components_id, 'selected_subject_mask': mask_id, 'invert_to_background': str(invert_id), 'restore_original_subject': str(composite_id)}})
    if not mask_loaders and (not image_loaders) and video_loaders:
        invert_nodes = [(node_id, node) for node_id, node in candidate.items() if isinstance(node, dict) and node.get('class_type') == 'InvertMask']
        composite_nodes = [(node_id, node) for node_id, node in candidate.items() if isinstance(node, dict) and node.get('class_type') == 'ImageCompositeMasked']
        if len(invert_nodes) != 1 or len(composite_nodes) != 1:
            raise ClosedLoopError('video background replacement mask adapter requires one InvertMask and one ImageCompositeMasked node')
        loader_id = _next_graph_node_id(candidate)
        components_id = str(int(loader_id) + 1)
        mask_id = str(int(loader_id) + 2)
        candidate[loader_id] = {'class_type': 'LoadVideo', 'inputs': {'file': 'MASK_INPUT.mp4'}, '_meta': {'comfyrouter_adapter': 'external_mask_video_loader'}}
        candidate[components_id] = {'class_type': 'GetVideoComponents', 'inputs': {'video': [loader_id, 0]}, '_meta': {'comfyrouter_adapter': 'external_mask_video_frames'}}
        candidate[mask_id] = {'class_type': 'ImageToMask', 'inputs': {'image': [components_id, 0], 'channel': 'red'}, '_meta': {'comfyrouter_adapter': 'external_selected_subject_mask'}}
        invert_id, invert_node = invert_nodes[0]
        composite_id, composite_node = composite_nodes[0]
        invert_node.setdefault('inputs', {})['mask'] = [mask_id, 0]
        composite_node.setdefault('inputs', {})['mask'] = [mask_id, 0]
        return (candidate, {'status': 'applied', 'adapter': 'video_internal_mask_to_external_mask_handoff', 'source_contract': dict(MASK_SELECTED_SUBJECT_CONTRACT), 'consumer_contract': {'mask_role': 'editable_background', 'white_means': 'replace'}, 'nodes': {'mask_video_loader': loader_id, 'mask_video_components': components_id, 'selected_subject_mask': mask_id, 'invert_to_background': str(invert_id), 'restore_original_subject': str(composite_id)}})
    if len(mask_loaders) == 1 and (not image_loaders) and video_loaders:
        candidate[mask_loaders[0][0]].setdefault('_meta', {})['comfyrouter_adapter'] = 'external_mask_video_loader'
        return (candidate, {'status': 'already_applied', 'adapter': 'external_mask_video_handoff', 'source_contract': dict(MASK_SELECTED_SUBJECT_CONTRACT), 'consumer_contract': {'mask_role': 'editable_background', 'white_means': 'replace'}, 'nodes': {'mask_loader': str(mask_loaders[0][0]), 'source_video_loader': str(video_loaders[0][0])}})
    if len(mask_loaders) != 1 or not image_loaders:
        raise ClosedLoopError('background replacement mask adapter requires one MASK loader and one IMAGE loader')
    mask_node, _ = mask_loaders[0]
    image_node, _ = image_loaders[0]
    if mask_role == 'edit_region':
        return (candidate, {'status': 'direct_contract', 'adapter': 'selected_edit_region_to_inpaint', 'source_contract': {'mask_role': 'edit_region', 'white_means': 'replace'}, 'consumer_contract': {'mask_role': 'editable_region', 'white_means': 'replace'}, 'nodes': {'selected_edit_mask': str(mask_node)}})
    if mask_role != 'preserve_region':
        raise ClosedLoopError(f'unsupported background replacement mask_role: {mask_role}')
    if any((node.get('class_type') == 'InvertMask' and node.get('_meta', {}).get('comfyrouter_adapter') == 'selected_subject_to_editable_background' for node in candidate.values() if isinstance(node, dict))):
        return (candidate, {'status': 'already_applied', 'source_contract': dict(MASK_SELECTED_SUBJECT_CONTRACT), 'consumer_contract': {'mask_role': 'editable_region', 'white_means': 'replace'}})
    invert_id = _next_graph_node_id(candidate)
    candidate[invert_id] = {'class_type': 'InvertMask', 'inputs': {'mask': [mask_node, 0]}, '_meta': {'comfyrouter_adapter': 'selected_subject_to_editable_background'}}
    for node_id, node in candidate.items():
        if node_id == invert_id or not isinstance(node, dict):
            continue
        for key, value in node.get('inputs', {}).items():
            if value == [mask_node, 0]:
                node['inputs'][key] = [invert_id, 0]
    feather_id = _next_graph_node_id(candidate)
    candidate[feather_id] = {'class_type': 'FeatherMask', 'inputs': {'mask': [mask_node, 0], 'left': 4, 'top': 4, 'right': 4, 'bottom': 4}, '_meta': {'comfyrouter_adapter': 'subject_edge_feather'}}
    save_nodes = [(node_id, node) for node_id, node in candidate.items() if isinstance(node, dict) and node.get('class_type') == 'SaveImage' and (not diagnostic_image_save(candidate, str(node_id)))]
    if len(save_nodes) != 1:
        raise ClosedLoopError('background replacement mask adapter requires one primary SaveImage')
    save_id, save_node = save_nodes[0]
    generated = save_node.get('inputs', {}).get('images')
    if not isinstance(generated, list) or len(generated) != 2:
        raise ClosedLoopError('background replacement SaveImage has no generated image input')
    composite_id = _next_graph_node_id(candidate)
    candidate[composite_id] = {'class_type': 'ImageCompositeMasked', 'inputs': {'destination': generated, 'source': [image_node, 0], 'x': 0, 'y': 0, 'resize_source': False, 'mask': [feather_id, 0]}, '_meta': {'comfyrouter_adapter': 'restore_original_selected_subject'}}
    save_node['inputs']['images'] = [composite_id, 0]
    return (candidate, {'status': 'applied', 'source_contract': dict(MASK_SELECTED_SUBJECT_CONTRACT), 'consumer_contract': {'mask_role': 'editable_region', 'white_means': 'replace'}, 'nodes': {'invert_mask': invert_id, 'feather_subject_mask': feather_id, 'restore_original_subject': composite_id, 'save_image': save_id}})

def composition_mask_role(typed_stage, implementation: list[dict]) -> str:
    """Resolve the semantic role of an upstream MASK without guessing polarity."""
    by_stage = {str((item.get('decomposition_step') or {}).get('step') or f'stage_{index:02d}'): item for index, item in enumerate(implementation, 1)}
    stage_ids = [f'stage_{index:02d}' for index in range(1, len(implementation) + 1)]
    candidates = []
    try:
        consumer = implementation[stage_ids.index(str(typed_stage.stage_id))]
    except (ValueError, IndexError):
        consumer = by_stage.get(str(typed_stage.stage_id)) or {}
    consumer_operation = str(consumer.get('operation') or '').strip().casefold()
    consumer_family = str(consumer.get('task_family') or '').strip().casefold()
    for dependency in typed_stage.depends_on:
        try:
            producer = implementation[stage_ids.index(str(dependency))]
        except (ValueError, IndexError):
            producer = by_stage.get(str(dependency))
        step = (producer or {}).get('decomposition_step') or {}
        if str(step.get('output_type') or '').casefold() != 'mask':
            continue
        role = str(step.get('mask_role') or '').strip().casefold()
        if role in {'edit_region', 'preserve_region'}:
            candidates.append(role)
            continue
        producer_text = str(step.get('description') or '').casefold()
        consumer_text = str(typed_stage.instruction or typed_stage.purpose or '').casefold()
        preserve_markers = ('preserve', 'keep unchanged', 'retain', 'subject to preserve', 'foreground subject')
        edit_markers = ('replace the selected', 'edit the selected', 'remove the selected', 'change the selected', 'selected area', 'selected region')
        if any((marker in producer_text or marker in consumer_text for marker in preserve_markers)):
            candidates.append('preserve_region')
        elif any((marker in consumer_text for marker in edit_markers)) or consumer_operation.endswith('/removal') or consumer_operation in {'object_removal', 'remove_object'} or (consumer_family == 'object_editing' and 'removal' in consumer_operation):
            candidates.append('edit_region')
        else:
            raise ClosedLoopError('MASK dependency is missing an explicit edit_region/preserve_region contract')
    if not candidates:
        raise ClosedLoopError('MASK-dependent stage has no MASK semantic contract')
    if len(set(candidates)) != 1:
        raise ClosedLoopError('MASK dependencies have conflicting semantic roles')
    return candidates[0]
AUXILIARY_INPUT_MARKERS = {'mask', 'reference', 'identity', 'style', 'pose', 'driving', 'control', 'depth'}

def media_input_nodes(graph: dict, media_type: str) -> list[tuple[str, str]]:
    """Resolve the primary external media input without binding auxiliaries."""
    adapters = set(MEDIA_INPUT_ADAPTERS.get(media_type, ()))
    placeholders = MEDIA_PLACEHOLDERS.get(media_type, set())
    candidates = []
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get('inputs', {})
        if not isinstance(inputs, dict):
            continue
        class_type = str(node.get('class_type', ''))
        for adapter_class, input_name in adapters:
            if class_type != adapter_class:
                continue
            value = inputs.get(input_name)
            if isinstance(value, str):
                marker = str((node.get('_meta') or {}).get('comfyrouter_adapter', ''))
                handoff = (node.get('_meta') or {}).get('comfyrouter_handoff') or {}
                if handoff.get('media_type') and handoff['media_type'] != media_type:
                    continue
                if media_type == 'VIDEO' and (marker == 'external_mask_video_loader' or value in MEDIA_PLACEHOLDERS['MASK']):
                    continue
                if media_type == 'MASK' and marker == 'external_video_loader':
                    continue
                candidates.append((str(node_id), input_name, value))
    role_marker = {'MASK': 'external_mask_video_loader', 'VIDEO': 'external_video_loader'}.get(media_type)
    marked = [(node_id, key) for node_id, key, _ in candidates if ((graph[node_id].get('_meta') or {}).get('comfyrouter_handoff') or {}).get('media_type') == media_type or (role_marker and (graph[node_id].get('_meta') or {}).get('comfyrouter_adapter') == role_marker)]
    if marked:
        if media_type == 'MASK':
            return marked
        values = {graph[node_id]['inputs'][key] for node_id, key in marked}
        return [(node_id, key) for node_id, key, value in candidates if (node_id, key) in marked or value in values]
    exact = [(node_id, key) for node_id, key, value in candidates if value in placeholders]
    if exact:
        return exact
    if media_type == 'MASK':
        if len(candidates) == 1 and graph[candidates[0][0]].get('class_type') == 'LoadImageMask':
            node_id, key, _ = candidates[0]
            return [(node_id, key)]
        return []
    primary = [(node_id, key) for node_id, key, value in candidates if not any((marker in value.casefold() for marker in AUXILIARY_INPUT_MARKERS))]
    if len(primary) == 1:
        return primary
    if len(candidates) == 1:
        node_id, key, _ = candidates[0]
        return [(node_id, key)]
    return []

def composition_handoff_inputs(graph: dict, typed_stage) -> tuple[dict, list]:
    """Discover the current graph's media inputs after any topology edits."""
    by_type = {media_type: media_input_nodes(graph, media_type) for media_type in stage_input_types(typed_stage)}
    return (by_type, by_type.get(typed_stage.input_type, []))

def handoff_input_groups(graph: dict, media_type: str, nodes: list[tuple[str, str]] | None=None) -> list[list[tuple[str, str]]]:
    """Group file endpoints by artifact slot, not by loader count.

    INPUT_VIDEO.mp4 can be read twice with distinct frame ranges. START_IMAGE
    and END_IMAGE must instead receive different upstream artifacts. Persist
    that distinction before substituting real filenames during preflight.
    """
    groups = {}
    for node_id, key in media_input_nodes(graph, media_type) if nodes is None else nodes:
        node = graph[node_id]
        handoff = (node.get('_meta') or {}).get('comfyrouter_handoff') or {}
        value = node.get('inputs', {}).get(key)
        slot = handoff.get('slot')
        if not slot:
            slot = value if value in MEDIA_PLACEHOLDERS.get(media_type, set()) else f'node:{node_id}:{key}'
        groups.setdefault(slot, []).append((node_id, key))
    slot_order = {'START_IMAGE.png': 0, 'END_IMAGE.png': 1, 'INPUT_VIDEO_A.mp4': 0, 'INPUT_VIDEO_B.mp4': 1}
    slot_order.update({f'{media_type}_declared_slot_{index:02d}': index - 1 for index in range(1, len(groups) + 1)})
    return [nodes for _, nodes in sorted(groups.items(), key=lambda item: slot_order.get(item[0], 0))]

def bind_composition_preflight_inputs(graph: dict, typed_stage, case_resources: list[Path], *, preserve_upstream_mask_placeholder: bool=False) -> tuple[dict, list]:
    """Bind real local enum values while generated handoff artifacts do not exist."""
    case_resources = ordered_stage_resources(case_resources, getattr(typed_stage, 'instruction', '') or '')
    by_type, primary = composition_handoff_inputs(graph, typed_stage)
    for media_type, nodes in by_type.items():
        declared_slots = list(stage_input_types(typed_stage)).count(media_type)
        groups = handoff_input_groups(graph, media_type, nodes)
        if declared_slots == 1 and len(groups) > 1 and all((not ((graph[node_id].get('_meta') or {}).get('comfyrouter_handoff') or {}).get('slot') or str(((graph[node_id].get('_meta') or {}).get('comfyrouter_handoff') or {})['slot']).startswith('node:') or ((graph[node_id].get('_meta') or {}).get('comfyrouter_handoff') or {}).get('slot') in {f'{media_type}_declared_slot_01', 'INPUT_VIDEO.mp4', 'INPUT_IMAGE.png', 'IMAGE_INPUT.png'} for node_id, _ in nodes)) and (not any((graph[node_id]['inputs'][key] in {'START_IMAGE.png', 'END_IMAGE.png', 'INPUT_VIDEO_A.mp4', 'INPUT_VIDEO_B.mp4'} for node_id, key in nodes))):
            for node_id, _ in nodes:
                graph[node_id].setdefault('_meta', {})['comfyrouter_handoff'] = {'media_type': media_type, 'slot': f'{media_type}_declared_slot_01', 'slot_origin': 'single_stage_input_fanout'}
        if declared_slots > 1 and len(nodes) == declared_slots and (len(groups) == 1):
            ordered = sorted(nodes, key=lambda row: ((0, int(row[0])) if str(row[0]).isdigit() else (1, str(row[0])), row[1]))
            for index, (node_id, _) in enumerate(ordered, 1):
                graph[node_id].setdefault('_meta', {})['comfyrouter_handoff'] = {'media_type': media_type, 'slot': f'{media_type}_declared_slot_{index:02d}', 'slot_origin': 'explicit_stage_input_multiplicity'}
        for group in handoff_input_groups(graph, media_type, nodes):
            first_id, first_key = group[0]
            value = graph[first_id].get('inputs', {}).get(first_key)
            slot = value if value in MEDIA_PLACEHOLDERS.get(media_type, set()) else f'node:{first_id}:{first_key}'
            for node_id, input_name in group:
                graph[node_id].setdefault('_meta', {}).setdefault('comfyrouter_handoff', {'media_type': media_type, 'slot': slot})
    source, source_type = primary_case_resource(case_resources, str(getattr(typed_stage, 'modality', '')))
    if source is not None and source_type:
        for media_type, nodes in by_type.items():
            if media_type not in {source_type, 'MASK'}:
                continue
            if media_type == 'MASK' and preserve_upstream_mask_placeholder:
                continue
            for node_id, input_name in nodes:
                graph[node_id].setdefault('inputs', {})[input_name] = source.name
    elif by_type.get('MASK'):
        fixture = COMFYUI_INPUT / 'comfyrouter_fixture.mp4'
        if fixture.is_file():
            for node_id, input_name in by_type['MASK']:
                if graph[node_id].get('class_type') == 'LoadVideo':
                    graph[node_id].setdefault('inputs', {})[input_name] = fixture.name
    return (by_type, primary)
GEOMETRY_NODE_PRIORITY = {'Wan22ImageToVideoLatent': 100, 'EmptyHunyuanVideo15Latent': 100, 'EmptyFlux2LatentImage': 90, 'EmptySD3LatentImage': 90, 'EmptyLatentImage': 90, 'ImageScale': 80}

def graph_geometry_contract(graph: dict) -> dict | None:
    """Return the strongest explicit canvas contract declared by a graph."""
    candidates = []
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get('inputs') or {}
        width, height = (inputs.get('width'), inputs.get('height'))
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            continue
        if width <= 0 or height <= 0:
            continue
        class_type = str(node.get('class_type', ''))
        crop = inputs.get('crop')
        candidates.append((GEOMETRY_NODE_PRIORITY.get(class_type, 50), {'node_id': str(node_id), 'class_type': class_type, 'width': width, 'height': height, 'aspect_ratio': round(float(width) / float(height), 6), 'resize_or_crop_behavior': str(crop) if crop is not None else 'fixed_canvas_model_input; source fit behavior must be verified'}))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]['node_id']))
    return candidates[0][1]

def cover_crop_risk(upstream: dict | None, downstream: dict | None) -> dict | None:
    """Estimate content discarded if a mismatched handoff uses cover crop."""
    if not upstream or not downstream:
        return None
    r_up = float(upstream['aspect_ratio'])
    r_down = float(downstream['aspect_ratio'])
    if abs(r_up - r_down) <= 0.01:
        return {'aspect_ratio_mismatch': False, 'retained_width_fraction': 1.0, 'retained_height_fraction': 1.0, 'estimated_total_crop_pixels': {'horizontal': 0, 'vertical': 0}}
    retained_width = min(1.0, r_down / r_up)
    retained_height = min(1.0, r_up / r_down)
    return {'aspect_ratio_mismatch': True, 'formula': 'r_up=W_up/H_up; r_down=W_down/H_down; cover retains min(1,r_down/r_up) width and min(1,r_up/r_down) height', 'retained_width_fraction': round(retained_width, 6), 'retained_height_fraction': round(retained_height, 6), 'estimated_total_crop_pixels': {'horizontal': round(float(upstream['width']) * (1.0 - retained_width), 2), 'vertical': round(float(upstream['height']) * (1.0 - retained_height), 2)}, 'risk': 'A cover/center crop can remove edge content; align canvases or preserve content with contain/padding and safe margins.'}

def handoff_geometry_contexts(typed, bases: list[dict], geometry_overrides: list[dict | None] | None=None, planned_stage_ids: set[str] | None=None) -> list[dict]:
    """Build direct upstream/downstream geometry contracts for every stage."""
    overrides = geometry_overrides or [None] * len(bases)
    geometries = [override or graph_geometry_contract(base) for base, override in zip(bases, overrides)]
    stage_indexes = {stage.stage_id: index for index, stage in enumerate(typed)}
    contexts = []
    for index, stage in enumerate(typed):
        upstream = []
        for dependency in stage.depends_on:
            source_index = stage_indexes[dependency]
            source_geometry = geometries[source_index]
            upstream.append({'stage_id': dependency, 'output_type': typed[source_index].output_type, 'output_geometry': source_geometry, 'cover_crop_risk_estimate': cover_crop_risk(source_geometry, geometries[index])})
        downstream = []
        for consumer_index, consumer in enumerate(typed):
            if stage.stage_id not in consumer.depends_on:
                continue
            consumer_geometry = geometries[consumer_index]
            downstream.append({'stage_id': consumer.stage_id, 'input_type': consumer.input_type, 'target_geometry': consumer_geometry, 'geometry_authority': 'planned_consumer' if planned_stage_ids is None or consumer.stage_id in planned_stage_ids else 'unplanned_template_hint', 'cover_crop_risk_estimate': cover_crop_risk(geometries[index], consumer_geometry)})
        contexts.append({'current_stage_id': stage.stage_id, 'current_input_type': stage.input_type, 'current_output_type': stage.output_type, 'current_output_geometry': geometries[index], 'upstream_producers': upstream, 'downstream_consumers': downstream, 'required_checks': ['Keep required subject, head, feet, hands, text, logos, and boundary objects inside frame.', 'Use visible safety margin on all edges for bounded subjects.', 'Inspect the actual post-handoff output, not only action or keyword compliance.']})
    return contexts

def media_file_geometry_contract(path: Path | None, media_type: str | None) -> dict | None:
    """Read the geometry of an external media input before graph planning."""
    if path is None or not path.is_file() or media_type not in {'IMAGE', 'VIDEO'}:
        return None
    if media_type == 'IMAGE':
        try:
            from PIL import Image
            with Image.open(path) as image:
                width, height = image.size
        except (OSError, ValueError):
            return None
    else:
        _, metadata, error = sample_video(path, 'handoff-geometry', 1)
        if error:
            return None
        width, height = (metadata.get('width'), metadata.get('height'))
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        return None
    if width <= 0 or height <= 0:
        return None
    return {'source': str(path), 'media_type': media_type, 'width': width, 'height': height, 'aspect_ratio': round(float(width) / float(height), 6), 'resize_or_crop_behavior': 'external source; preserve required content before model canvas fitting'}

def external_handoff_geometry_context(graph: dict, source: Path | None, media_type: str | None, stage_id: str='stage_01') -> dict | None:
    source_geometry = media_file_geometry_contract(source, media_type)
    if not source_geometry:
        return None
    target_geometry = graph_geometry_contract(graph)
    return {'current_stage_id': stage_id, 'current_input_type': media_type, 'current_output_type': None, 'current_output_geometry': target_geometry, 'upstream_producers': [{'stage_id': 'external_input', 'output_type': media_type, 'output_geometry': source_geometry, 'cover_crop_risk_estimate': cover_crop_risk(source_geometry, target_geometry)}], 'downstream_consumers': [], 'required_checks': ['Preserve every required visible part of the external source during canvas fitting.', 'Do not crop faces, heads, feet, hands, text, logos, or boundary objects.', 'Match the model canvas to the source aspect ratio or use explicit contain/padding.']}

def composition_external_geometry_context(graph, stage, typed, case_resources, context):
    """Expose original media geometry alongside auxiliary upstream masks.

    A dependency on MASK does not imply that the stage's IMAGE comes from
    upstream. Only a same-type producer prevents using the bound case media.
    """
    result = copy.deepcopy(context)
    producers = {item.stage_id: item for item in typed}
    for media_type in ('IMAGE', 'VIDEO'):
        if media_type not in stage_input_types(stage):
            continue
        if any((str(producers[dep].output_type).upper() == media_type for dep in stage.depends_on if dep in producers)):
            continue
        bound_names = {Path(str(graph[node_id].get('inputs', {}).get(key, ''))).name for node_id, key in media_input_nodes(graph, media_type)}
        source = next((Path(path) for path in case_resources if Path(path).name in bound_names), None)
        external = external_handoff_geometry_context(graph, source, media_type, stage.stage_id)
        if external:
            result['upstream_producers'] = external['upstream_producers'] + result.get('upstream_producers', [])
            result['required_checks'] = list(dict.fromkeys(result.get('required_checks', []) + external['required_checks']))
    return result

def copy_case_resources(instruction: str) -> list[str]:
    copied = []
    resource_root = DATASET / 'resource'
    COMFYUI_INPUT.mkdir(parents=True, exist_ok=True)
    for name in dict.fromkeys(re.findall('`([^`]+)`', instruction)):
        source = resource_root / Path(name).name
        if not source.is_file():
            continue
        target = COMFYUI_INPUT / source.name
        shutil.copy2(source, target)
        copied.append(str(target))
    return copied

def _is_comfyui_server_process(process_dir: Path) -> bool:
    """Recognize ComfyUI whether main.py was passed as an absolute or relative path."""
    argv = [part for part in (process_dir / 'cmdline').read_bytes().split(b'\x00') if part]
    if not argv:
        return False
    main_path = str(COMFYUI_ROOT / 'main.py').encode()
    if main_path in argv:
        return True
    if b'main.py' not in argv:
        return False
    return (process_dir / 'cwd').resolve() == COMFYUI_ROOT.resolve()

def _comfyui_server_rss_bytes() -> int | None:
    """Return the resident memory of the local ComfyUI server, if discoverable."""
    page_size = os.sysconf('SC_PAGE_SIZE')
    for process_dir in Path('/proc').iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            if not _is_comfyui_server_process(process_dir):
                continue
            resident_pages = int((process_dir / 'statm').read_text().split()[1])
            return resident_pages * page_size
        except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
            continue
    return None

def _comfyui_server_pids() -> list[int]:
    pids = []
    for process_dir in Path('/proc').iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            if _is_comfyui_server_process(process_dir):
                pids.append(int(process_dir.name))
        except (FileNotFoundError, PermissionError, OSError, ValueError):
            continue
    return pids

def _wait_comfyui_idle(opener, timeout: int=90) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with opener.open(COMFYUI_API + '/queue', timeout=10) as response:
                queue = json.loads(response.read().decode('utf-8'))
            if not queue.get('queue_running') and (not queue.get('queue_pending')):
                return
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(1)
    raise ClosedLoopError('ComfyUI queue did not become idle before model release')

def _restart_comfyui_for_clean_memory(rss_limit: int) -> dict:
    old_pids = _comfyui_server_pids()
    for pid in old_pids:
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while any((Path(f'/proc/{pid}').exists() for pid in old_pids)):
        if time.monotonic() >= deadline:
            raise ClosedLoopError('ComfyUI did not stop for clean-memory restart')
        time.sleep(0.5)
    log_path = ROOT.parent / 'output3' / 'comfyui_auto_restart.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('ab', buffering=0) as log_handle:
        process = subprocess.Popen([str(COMFYUI_ROOT / '.venv/bin/python'), str(COMFYUI_ROOT / 'main.py'), '--listen', '127.0.0.1', '--port', '8188'], cwd=COMFYUI_ROOT, stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            with opener.open(COMFYUI_API + '/system_stats', timeout=5) as response:
                response.read()
            rss = _comfyui_server_rss_bytes()
            if rss is not None and rss <= rss_limit:
                return {'pid': process.pid, 'rss_bytes': rss, 'log': str(log_path)}
        except (OSError, urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise ClosedLoopError('ComfyUI restart did not reach the clean-memory RSS target')

def _release_comfyui_models_once(retries: int=6, release_passes: int=3) -> dict:
    """Try one unload/restart cycle and return only after reaching the RSS target."""
    errors = []
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = json.dumps({'unload_models': True, 'free_memory': True}).encode()
    rss_limit = int(float(os.getenv('COMFYROUTER_RELEASE_RSS_LIMIT_GIB', '10')) * 1024 ** 3)
    successful_passes = 0
    last_attempt = 0
    rss = _comfyui_server_rss_bytes()
    _wait_comfyui_idle(opener)
    for release_pass in range(1, release_passes + 1):
        for attempt in range(1, retries + 1):
            last_attempt = attempt
            request = urllib.request.Request(COMFYUI_API + '/free', data=payload, headers={'Content-Type': 'application/json'}, method='POST')
            try:
                with opener.open(request, timeout=30) as response:
                    response.read()
                successful_passes = release_pass
                time.sleep(2)
                rss = _comfyui_server_rss_bytes()
                break
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                errors.append(str(exc))
                time.sleep(min(2 * attempt, 10))
        else:
            raise ClosedLoopError(f'ComfyUI model release failed after {retries} attempts: ' + ' | '.join(errors))
        if rss is not None and rss <= rss_limit:
            return {'status': 'released', 'attempt': last_attempt, 'release_passes': successful_passes, 'rss_bytes': rss, 'rss_limit_bytes': rss_limit}
    restart = _restart_comfyui_for_clean_memory(rss_limit)
    return {'status': 'restarted', 'attempt': last_attempt, 'release_passes': successful_passes, 'rss_bytes': restart['rss_bytes'], 'rss_limit_bytes': rss_limit, 'restart': restart}

def release_comfyui_models(retries: int=6, release_passes: int=3) -> dict:
    """Keep recovering ComfyUI until it is idle and below the RSS target.

    A transient /free or restart failure must delay the serial run, not stop it.
    """
    recovery_errors = []
    recovery_round = 0
    while True:
        recovery_round += 1
        try:
            result = _release_comfyui_models_once(retries, release_passes)
            result['recovery_rounds'] = recovery_round
            if recovery_errors:
                result['recovery_errors'] = recovery_errors
            return result
        except (ClosedLoopError, OSError, urllib.error.URLError, TimeoutError) as exc:
            recovery_errors.append(str(exc))
            try:
                rss_limit = int(float(os.getenv('COMFYROUTER_RELEASE_RSS_LIMIT_GIB', '10')) * 1024 ** 3)
                _restart_comfyui_for_clean_memory(rss_limit)
            except (ClosedLoopError, OSError, urllib.error.URLError, TimeoutError) as restart_exc:
                recovery_errors.append(f'restart: {restart_exc}')
            time.sleep(10)

def bind_runtime_resource(graph: dict, source: Path | None, media_type: str) -> list[dict]:
    """Replace compatible Base placeholders after Agent edits and before execution."""
    if source is None:
        return []
    bindings = []
    for node_id, input_name in media_input_nodes(graph, media_type):
        old = graph[node_id].setdefault('inputs', {}).get(input_name)
        graph[node_id]['inputs'][input_name] = source.name
        bindings.append({'node_id': node_id, 'input': input_name, 'old': old, 'value': source.name})
    return bindings

def require_runtime_resources_bound(graph: dict, sources: list[Path], media_type: str) -> None:
    adapters = set(MEDIA_INPUT_ADAPTERS.get(media_type, ()))
    values = []
    for node in graph.values():
        if not isinstance(node, dict) or not isinstance(node.get('inputs'), dict):
            continue
        for class_type, input_name in adapters:
            if node.get('class_type') == class_type and isinstance(node['inputs'].get(input_name), str):
                values.append(node['inputs'][input_name])
    missing = [source.name for source in sources if source.name not in values]
    if missing:
        raise ClosedLoopError(f'runtime input binding failed: {', '.join(missing)} is not bound to a {media_type} loader')
    unresolved = sorted(set(values) & MEDIA_PLACEHOLDERS.get(media_type, set()))
    if unresolved:
        raise ClosedLoopError('runtime input binding left unresolved placeholders: ' + ', '.join(unresolved))

def require_runtime_resource_bound(graph: dict, source: Path, media_type: str) -> None:
    require_runtime_resources_bound(graph, [source], media_type)

def primary_case_resource(resources: list[Path], modality: str) -> tuple[Path | None, str | None]:
    input_kind = modality.split('2', 1)[0].lower()
    if input_kind == 'i':
        extensions, media_type = ({'.png', '.jpg', '.jpeg', '.webp', '.bmp'}, 'IMAGE')
    elif input_kind == 'v':
        extensions, media_type = ({'.mp4', '.webm', '.mov', '.avi', '.gif'}, 'VIDEO')
    elif input_kind == 'a':
        extensions, media_type = ({'.wav', '.mp3', '.flac', '.m4a'}, 'AUDIO')
    else:
        return (None, None)
    return (next((path for path in resources if path.suffix.lower() in extensions), None), media_type)

def call_with_three_retries(callback):
    errors = []
    for attempt in range(1, 4):
        try:
            return callback()
        except (AgentTransportError, AgentProviderError):
            raise
        except AgentOutputError as exc:
            errors.append(str(exc))
            if attempt < 3:
                time.sleep(3)
    raise AgentOutputError('Agent call failed after three attempts: ' + ' | '.join(errors))

def repair_search_with_fallback(callback):
    """Keep a completed generation attempt reportable when search is unavailable."""
    try:
        return call_with_three_retries(callback)
    except (AgentTransportError, AgentOutputError) as exc:
        return {'failure_class': 'execution', 'root_causes': ['Combined visual diagnosis Agent was unavailable.'], 'next_action': 'parameter_retry', 'diagnosis': 'Repair-search Agent unavailable after bounded retries.', 'conditioning_attribution': {}, 'repair_focus': [], 'needs_search': False, 'search_queries': [], 'evidence': [], 'error': str(exc)}

def combined_failure_analysis(decision: dict) -> dict:
    """Project one visual execution diagnosis into the retained trace schema."""
    return {'failure_class': str(decision.get('failure_class') or 'execution'), 'root_causes': list(decision.get('root_causes') or [str(decision.get('diagnosis') or 'Execution result did not satisfy the task.')]), 'evidence': list(decision.get('evidence') or []), 'next_action': str(decision.get('next_action') or 'parameter_retry'), 'affected_nodes': [], 'affected_leaves': [], 'comparison': {}, 'confidence': 'high' if decision.get('evidence') else 'medium', 'source': 'combined_visual_execution_diagnosis'}

def combined_search_decision(decision: dict) -> dict:
    """Project the same diagnosis into the optional retrieval contract."""
    return {'diagnosis': str(decision.get('diagnosis') or decision.get('rationale') or ''), 'conditioning_attribution': copy.deepcopy(decision.get('conditioning_attribution') or {}), 'repair_focus': list(decision.get('repair_focus') or []), 'needs_search': bool(decision.get('needs_search')), 'search_queries': list(decision.get('search_queries') or []), 'evidence': list(decision.get('evidence') or []), 'source': 'combined_visual_execution_diagnosis'}

def composition_repair_targets(diagnosis: dict, states: list[dict], descriptors: list[dict]) -> list[dict]:
    """Plan the primary repair plus independent later errors before the next execution."""
    stage_ids = [state['typed'].stage_id for state in states]
    primary = diagnosis['repair_stage_id']
    assessments = {row['stage_id']: row for row in diagnosis['stage_assessments']}
    AgentOrchestrator._validate_independent_repair_evidence(diagnosis, descriptors)
    primary_diagnosis = {**diagnosis, 'independent_stage_repairs': []}
    targets = [{'stage_id': primary, 'diagnosis': primary_diagnosis, 'assessment': assessments[primary], 'independent': False}]
    seen = {primary}
    for repair in diagnosis.get('independent_stage_repairs', []):
        stage_id = repair['stage_id']
        if stage_id in seen or stage_id not in stage_ids or stage_ids.index(stage_id) <= stage_ids.index(primary) or (assessments.get(stage_id, {}).get('status') == 'pass'):
            raise AgentOutputError('independent repair cannot duplicate the primary or reopen passed/earlier stages')
        seen.add(stage_id)
        scoped = {**diagnosis, **repair, 'repair_stage_id': stage_id, 'independent_stage_repairs': [], 'independent_repair': repair}
        assessment = {**assessments[stage_id], 'status': 'fail', 'observations': list(repair['evidence']), 'reason': repair['contract']}
        targets.append({'stage_id': stage_id, 'diagnosis': scoped, 'assessment': assessment, 'independent': True})
    return sorted(targets, key=lambda target: stage_ids.index(target['stage_id']))

def single_failure_visual_inputs(case_resources: list[Path], primary_output: Path | None, diagnostic_outputs: list[Path]) -> tuple[list[str], list[dict]]:
    """Order the actual input, effective masks, and failed image for one-stage diagnosis."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    paths = []
    manifest = []
    original = next((path for path in case_resources if path.is_file() and path.suffix.lower() in image_extensions), None)
    if original is not None:
        paths.append(str(original))
        manifest.append({'position': len(paths), 'role': 'original_input', 'path': str(original)})
    for path in diagnostic_outputs[:3]:
        if path.is_file() and path.suffix.lower() in image_extensions:
            paths.append(str(path))
            manifest.append({'position': len(paths), 'role': 'effective_mask', 'path': str(path)})
    if primary_output is not None and primary_output.is_file() and (primary_output.suffix.lower() in image_extensions):
        paths.append(str(primary_output))
        manifest.append({'position': len(paths), 'role': 'failed_output', 'path': str(primary_output)})
    return (paths, manifest)

def run_anysearch(queries: list[str]) -> list[dict]:
    context = current_execution()
    if context and context.policy.disable_search:
        raise ClosedLoopError('search is disabled by the execution policy')
    'Run only the searches explicitly requested by the repair decision Agent.'
    results = []
    if not queries:
        return results
    if not ANYSEARCH_CLI.is_file():
        return [{'status': 'unavailable', 'error': f'missing AnySearch CLI: {ANYSEARCH_CLI}'}]
    for query in queries[:3]:
        errors = []
        for search_attempt in range(1, 4):
            try:
                with serialized_api_request():
                    proc = subprocess.run([sys.executable, str(ANYSEARCH_CLI), 'search', query, '--max_results', '5'], text=True, capture_output=True, timeout=90)
                payload = None
                if proc.stdout.strip():
                    try:
                        payload = json.loads(proc.stdout)
                    except json.JSONDecodeError:
                        payload = proc.stdout[-12000:]
                if proc.returncode == 0:
                    results.append({'query': query, 'status': 'ok', 'attempt': search_attempt, 'result': payload, 'stderr': proc.stderr[-4000:]})
                    break
                errors.append(proc.stderr[-4000:] or f'exit {proc.returncode}')
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(str(exc))
            if search_attempt < 3:
                time.sleep(3)
        else:
            results.append({'query': query, 'status': 'error', 'attempts': 3, 'errors': errors})
    return results

def require_parameter_only(plan: dict, require_edit: bool=False) -> None:
    edits = plan.get('edits', [])
    if require_edit and (not edits):
        raise AgentOutputError('failed ComfyBench attempt requires at least one parameter edit')
    invalid = sorted({str(edit.get('action')) for edit in edits if isinstance(edit, dict) and edit.get('action') != 'set_param'})
    if invalid:
        raise AgentOutputError('ComfyBench leaf adaptation is parameter-only; invalid edits: ' + ', '.join(invalid))

def graph_selected_by_execution_agent(plan: dict, base: dict, executor: ComfyUIExecutor) -> tuple[dict, dict | None]:
    """Return the graph the execution Agent actually selected.

    ``available_workflow_skills`` is advisory until the Agent returns a
    concrete ``skill_ids`` choice.  An empty choice keeps the exact Base;
    a selected lightweight Skill supplies its complete ``workflow_api.json``
    candidate.  The candidate is validated before any edits are applied.
    """
    context = current_execution()
    if context and context.policy.frozen_graph:
        require_parameter_only(plan)
        if plan.get('skill_ids') or plan.get('selected_workflow'):
            raise AgentOutputError('frozen graph policy forbids workflow replacement')
        return (base, None)
    selected = plan.get('selected_workflow')
    skill_ids = [str(value) for value in plan.get('skill_ids', [])]
    if not skill_ids:
        if plan.get('selected_workflow_source') == 'metadata_selected_base':
            if not isinstance(selected, dict) or not selected:
                raise AgentOutputError('metadata-selected Base requires its complete API graph')
            executor.static_validate(selected, allow_external_placeholders=True)
            return (copy.deepcopy(selected), {'skill_ids': [], 'selection': 'use_base', 'workflow_source': 'metadata_selected_base'})
        return (base, None)
    if not isinstance(selected, dict) or not selected:
        raise AgentOutputError('execution Agent selected incremental Skill(s) but supplied no complete workflow candidate')
    executor.static_validate(selected, allow_external_placeholders=True)
    workflow_source = str(plan.get('selected_workflow_source') or 'selected_skill.workflow_template')
    return (copy.deepcopy(selected), {'skill_ids': skill_ids, 'selection': 'use_increment', 'workflow_source': workflow_source})

def repair_plan_base_graph(plan: dict, base: dict, current: dict) -> dict:
    """Keep unselected repairs cumulative across execution attempts.

    ``skill_assessments`` only records alternatives the Agent considered; it
    does not authorize resetting validated edits by returning to the original
    Base.  A newly selected complete Workflow Skill is handled separately by
    ``graph_selected_by_execution_agent``.  Otherwise every new edit applies
    to the previous candidate, and can explicitly reverse an inherited value
    by setting that parameter itself.
    """
    return current

class RepairPreflightError(ClosedLoopError):
    """A bounded repair plan could not produce a statically valid graph."""

def stage_input_types(typed_stage) -> tuple[str, ...]:
    """Read current and legacy typed-stage input contracts uniformly."""
    declared = getattr(typed_stage, 'input_types', None)
    if declared is None:
        declared = (getattr(typed_stage, 'input_type', None),)
    elif isinstance(declared, str):
        declared = (declared,)
    return tuple((value for value in declared if value))

def deferred_stage_handoff_types(typed_stage, all_stages) -> tuple[str, ...]:
    """Return media boundaries whose producer has not run yet.

    A reusable Base may legitimately retain a signed runtime placeholder such
    as ``MASK_INPUT.png`` until its upstream stage materializes the artifact.
    That placeholder is allowed during graph planning only; execution still
    performs strict binding and validation after the file is copied into the
    ComfyUI input directory.
    """
    by_id = {str(stage.stage_id): stage for stage in all_stages}
    deferred = []
    for dependency in getattr(typed_stage, 'depends_on', ()):
        producer = by_id.get(str(dependency))
        output_type = str(getattr(producer, 'output_type', '')).upper()
        if output_type in MEDIA_PLACEHOLDERS:
            deferred.append(output_type)
    return tuple(dict.fromkeys(deferred))

def build_handoff_contract_context(typed_stage, all_stages, implementation, graph: dict | None=None, artifact: dict | None=None) -> dict:
    """Build explicit media-boundary evidence for Skill selection.

    The selected Base is not the handoff contract.  A composition stage may
    consume a MASK produced upstream while its Base has no ``LoadImageMask``;
    a complete conditional Skill can replace that Base and provide the loader.
    Conversely, a face-restoration graph with native face detection must not be
    forced to accept an external MASK merely because the word appears in an
    old trace.
    """
    input_types = [str(value).upper() for value in stage_input_types(typed_stage)]
    producers = []
    stage_by_id = {str(stage.stage_id): stage for stage in all_stages}
    implementation_by_stage = {str(stage.stage_id): implementation[index] for index, stage in enumerate(all_stages) if index < len(implementation) and isinstance(implementation[index], dict)}
    for dependency in getattr(typed_stage, 'depends_on', ()):
        producer = stage_by_id.get(str(dependency))
        if producer is None:
            continue
        producers.append({'stage_id': str(producer.stage_id), 'media_type': str(producer.output_type).upper(), 'status': 'declared_upstream'})
    has_mask_dependency = any((row['media_type'] == 'MASK' for row in producers))
    mask_required = 'MASK' in input_types or has_mask_dependency
    native_mask_classes = {'FaceRestoreModelLoader', 'FaceRestoreCFWithModel', 'ReActorFaceSwap', 'ReActorRestoreFace', 'RestoreFaceWithModel'}
    graph_classes = {str(node.get('class_type')) for node in (graph or {}).values() if isinstance(node, dict)}
    if mask_required:
        mask_requirement = {'status': 'required', 'role': 'edit_region', 'source': 'external_upstream' if has_mask_dependency else 'stage_contract', 'polarity': 'white_selects', 'alignment': 'same_image_size'}
    elif graph_classes & native_mask_classes:
        mask_requirement = {'status': 'native_model_control', 'role': 'face_detection_internal', 'source': 'model_native', 'alignment': 'model_internal'}
    else:
        mask_requirement = {'status': 'not_needed', 'role': None, 'source': 'none'}
    context = {'required_input_types': input_types, 'required_output_type': str(typed_stage.output_type).upper(), 'producer_dependencies': producers, 'downstream_consumers': [{'stage_id': str(consumer.stage_id), 'operation': str(implementation_by_stage.get(str(consumer.stage_id), {}).get('operation') or ''), 'task_family': str(implementation_by_stage.get(str(consumer.stage_id), {}).get('task_family') or ''), 'input_types': [str(value).upper() for value in stage_input_types(consumer)]} for consumer in all_stages if str(typed_stage.stage_id) in {str(dependency) for dependency in getattr(consumer, 'depends_on', ())}], 'mask_requirement': mask_requirement, 'current_graph_inputs': {media_type: [{'node_id': node_id, 'input': input_name} for node_id, input_name in media_input_nodes(graph or {}, media_type)] for media_type in ('IMAGE', 'MASK', 'VIDEO', 'AUDIO')}, 'selection_rule': 'Match complete Skills by SKILL.md semantic trigger and this contract. A required upstream MASK is available evidence even when current_graph_inputs.MASK is empty; the selected complete Skill must then expose a compatible MASK loader.' if mask_required else 'Do not add or require an external MASK for this stage. Use a Skill only if its SKILL.md matches the non-mask task and its documented inputs.'}
    if artifact:
        context['upstream_artifact'] = copy.deepcopy(artifact)
    return context

def media_input_gate() -> MediaInputGate:
    return MediaInputGate(nodes=media_input_nodes, groups=handoff_input_groups, adapters=MEDIA_INPUT_ADAPTERS, input_root=COMFYUI_INPUT)

def validate_stage_handoff_contract(graph: dict, handoff_contract: dict, *, case_resources: list[Path] | None=None, validator=None) -> dict:
    try:
        return media_input_gate().preflight(graph, handoff_contract, case_resources=case_resources, validator=validator)
    except MediaInputError as exc:
        raise AgentOutputError(str(exc)) from exc

def bind_external_composition_inputs(graph: dict, typed_stage, resources: list[Path], used_by_type: dict[str, int]) -> list[dict]:
    return media_input_gate().bind_external(graph, stage_input_types(typed_stage), resources, used_by_type, instruction=getattr(typed_stage, 'instruction', '') or '')

def inspect_runtime_media_inputs(graph: dict, media_types) -> dict:
    return media_input_gate().inspect_materialized(graph, media_types)

def single_stage_input_contract(implementation, media_type: str | None) -> dict:
    decomposition = implementation[0].get('decomposition_step') or {} if implementation else {}
    declared = decomposition.get('input_types')
    if not declared:
        declared = [decomposition.get('input_type') or media_type or 'TEXT']
    return {'required_input_types': [str(value).upper() for value in declared], 'producer_dependencies': []}

def prepare_single_media_inputs(graph: dict, implementation, resources: list[Path], media_type: str | None, modality: str, validator=None) -> dict:
    """Use the same gate for single leaves, including mixed and repeated inputs."""
    contract = single_stage_input_contract(implementation, media_type)
    types = tuple(contract['required_input_types'])
    description = (implementation[0].get('decomposition_step') or {}).get('description', '') if implementation else ''
    resources = ordered_stage_resources(resources, description)
    stage = SimpleNamespace(input_type=media_type or types[0], input_types=types, modality=modality, instruction=description)
    bind_composition_preflight_inputs(graph, stage, resources)
    interface = validate_stage_handoff_contract(graph, contract, case_resources=resources, validator=validator)
    bindings = media_input_gate().bind_external(graph, types, resources, {})
    for kind in dict.fromkeys((item['media_type'] for item in bindings)):
        require_runtime_resources_bound(graph, [Path(item['input']) for item in bindings if item['media_type'] == kind], kind)
    materialized = inspect_runtime_media_inputs(graph, types)
    return {'interface': interface, 'bindings': bindings, 'materialized': materialized}

def _base_has_complete_mask_skill(base_path: str | None) -> bool:
    """Check for a complete, local MASK-aware replacement Skill.

    This is deliberately a file/interface check, not a profile allow-list.
    Semantic selection remains the Agent's job; the presence of a complete
    candidate prevents us from pruning a valid upstream MASK before selection.
    """
    if not base_path:
        return False
    root = Path(base_path).resolve().parent / 'workflow_skill'
    if not root.is_dir():
        return False
    for skill_dir in root.iterdir():
        if not skill_dir.is_dir():
            continue
        skill_doc = skill_dir / 'SKILL.md'
        workflow = skill_dir / 'workflow_api.json'
        if not skill_doc.is_file() or not workflow.is_file():
            continue
        text = skill_doc.read_text(encoding='utf-8').casefold()
        if 'mask' in text and any((token in text for token in ('loadimagemask', 'aligned mask', 'mask handoff', 'image and mask', 'source `mask`', 'mask artifact'))):
            return True
    return False

def _base_declares_image_only(base_path: str | None) -> bool:
    """Read the exact leaf's SKILL.md for a documented single-image input."""
    if not base_path:
        return False
    document = Path(base_path).resolve() / 'SKILL.md'
    if not document.is_file():
        return False
    text = document.read_text(encoding='utf-8').casefold()
    image_only_markers = ('input: one image', 'input: a single image', 'one image containing', 'single-image input', 'single image input', 'image-only input')
    if not any((marker in text for marker in image_only_markers)):
        return False
    mask_markers = ('loadimagemask', 'external mask', 'mask handoff', 'image and mask')
    return not any((marker in text for marker in mask_markers))

def _prune_unconsumed_mask_stages(implementation: list[dict], bases: list[dict]) -> list[dict]:
    """Remove an unnecessary generated-MASK stage for an image-only Base.

    Decomposition may conservatively add a localization stage. It must not be
    forced into a downstream Base whose documented interface is image-only.
    Conversely, a complete MASK Skill at that exact leaf keeps the branch so
    the generation Agent can select it semantically.
    """
    context = current_execution()
    if context and context.policy.frozen_graph:
        return implementation
    if len(implementation) != len(bases):
        return implementation
    mask_stage_ids = {str((step.get('decomposition_step') or {}).get('step') or f'stage_{i:02d}') for i, step in enumerate(implementation, 1) if str((step.get('decomposition_step') or {}).get('output_type') or '').casefold() == 'mask'}
    if not mask_stage_ids:
        return implementation
    changed = False
    kept = []
    for index, step in enumerate(implementation):
        decomposition = copy.deepcopy(step.get('decomposition_step') or {})
        input_types = [str(value).casefold() for value in decomposition.get('input_types', [decomposition.get('input_type')])]
        dependencies = [str(value) for value in decomposition.get('depends_on') or []]
        consumes_mask = bool(mask_stage_ids.intersection(dependencies)) or 'mask' in input_types
        explicit_local_region = any((token in str(decomposition.get('description', '')).casefold() for token in ('outside the mask', 'outside a mask', '遮罩外', 'mask-limited')))
        explicit_local_region = explicit_local_region or any((str((producer.get('decomposition_step') or {}).get('step')) in dependencies and (producer.get('decomposition_step') or {}).get('mask_role') == 'edit_region' for producer in implementation))
        if consumes_mask and (not explicit_local_region) and _base_declares_image_only(step.get('base_path')) and (not _base_has_complete_mask_skill(step.get('base_path'))):
            decomposition['input_types'] = [value for value in input_types if value != 'mask'] or [str(decomposition.get('input_type') or 'image').casefold()]
            decomposition['depends_on'] = [value for value in dependencies if value not in mask_stage_ids]
            step = {**step, 'decomposition_step': decomposition}
            changed = True
        kept.append(step)
    if not changed:
        return implementation
    referenced = {str(value) for step in kept for value in (step.get('decomposition_step') or {}).get('depends_on') or []}
    normalized = [step for step in kept if str((step.get('decomposition_step') or {}).get('output_type') or '').casefold() != 'mask' or str((step.get('decomposition_step') or {}).get('step') or '') in referenced]
    return normalized

def _internal_motion_target_evidence(graph: dict) -> dict | None:
    """Prove a text-generated image and a separate video drive reach the output.

    Unknown generator/encoder interfaces deliberately leave the plan alone.
    A missing loader, a node title, or a disconnected sampler is not evidence.
    """
    if any((str(node.get('class_type', '')).startswith('LoadImage') for node in graph.values() if isinstance(node, dict))):
        return None
    video_bindings = media_input_nodes(graph, 'VIDEO')
    video_inputs = {node_id for node_id, _ in video_bindings}
    video_files = {graph[node_id]['inputs'][key] for node_id, key in video_bindings}
    if not video_inputs or len(video_files) != 1:
        return None
    parents = {str(node_id): {str(value[0]) for value in node.get('inputs', {}).values() if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int) and (str(value[0]) in graph)} for node_id, node in graph.items() if isinstance(node, dict)}

    def ancestors(node_id):
        seen, pending = (set(), [node_id])
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(parents.get(current, ()))
        return seen
    outputs = []
    for node_id, node in graph.items():
        if node.get('class_type') not in {'CreateVideo', 'VHS_VideoCombine'}:
            continue
        images = node.get('inputs', {}).get('images')
        if isinstance(images, list) and len(images) == 2 and (str(images[0]) in graph):
            outputs.append(ancestors(str(images[0])))
    if not outputs:
        return None
    samplers = {'KSampler', 'KSamplerAdvanced', 'SamplerCustom', 'SamplerCustomAdvanced'}
    for decoder_id, decoder in graph.items():
        if decoder.get('class_type') != 'VAEDecode':
            continue
        samples = decoder.get('inputs', {}).get('samples')
        if not isinstance(samples, list) or len(samples) != 2:
            continue
        sampler_id = str(samples[0])
        if graph.get(sampler_id, {}).get('class_type') not in samplers:
            continue
        sampling_inputs = graph[sampler_id].get('inputs', {})
        positive = sampling_inputs.get('positive')
        guider = sampling_inputs.get('guider')
        if positive is None and isinstance(guider, list) and (len(guider) == 2):
            positive = graph.get(str(guider[0]), {}).get('inputs', {}).get('positive')
        if not isinstance(positive, list) or len(positive) != 2 or str(positive[0]) not in graph:
            continue
        text_nodes = [node_id for node_id in ancestors(str(positive[0])) if graph[node_id].get('class_type') == 'CLIPTextEncode' and str(graph[node_id].get('inputs', {}).get('text', '')).strip()]
        if not text_nodes or not all((str(decoder_id) in path for path in outputs)):
            continue
        for merge_id, incoming in parents.items():
            if merge_id == str(decoder_id) or not all((merge_id in path for path in outputs)):
                continue
            branches = [ancestors(parent) for parent in incoming]
            if any((str(decoder_id) in branch for branch in branches)) and any((video_inputs.intersection(branch) and str(decoder_id) not in branch for branch in branches)):
                return {'image_decoder': str(decoder_id), 'image_sampler': sampler_id, 'text_nodes': sorted(text_nodes), 'motion_merge': merge_id, 'video_inputs': sorted(video_inputs)}
    return None

def _collapse_internal_motion_target(implementation, bases, instruction, case_resources=None, stage_runtime_param_edits=None):
    """Fold only an auxiliary text-to-image target into a self-contained motion Base.

    This is a workflow-boundary correction, including in Base-only mode; it
    neither edits Base topology nor reads workflow skills/experience documents.
    Explicit intermediate artifacts, sequencing and external references win.
    """
    context = current_execution()
    if context and context.policy.frozen_graph:
        return (implementation, None)
    if len(implementation) != 2 or len(bases) != 2 or stage_runtime_param_edits:
        return (implementation, None)
    producer, consumer = implementation
    first = producer.get('decomposition_step') or {}
    last = consumer.get('decomposition_step') or {}
    if producer.get('operation') != 'Generation/text_to_image' or producer.get('modality') != 't2i' or producer.get('explicit_model') or (first.get('input_type') != 'text') or (first.get('input_types', ['text']) != ['text']) or (first.get('output_type') != 'image') or first.get('depends_on') or (consumer.get('task_family') != 'Motion_Cloning') or (consumer.get('modality') != 'v2v') or (last.get('input_type') != 'video') or (last.get('output_type') != 'video') or (set(last.get('input_types') or []) != {'video', 'image'}) or (list(last.get('depends_on') or []) != [str(first.get('step'))]):
        return (implementation, None)
    explicit_artifact = re.search('\\b(image|images|photo|photos|photograph|picture|pictures|portrait|still|first|then|after|afterwards|subsequently|before|next|clips|segments|both|separately)\\b|\\b(?:generate|create)\\b[^.!?]*\\band\\s+animate\\b|\\b(?:two|three|four|multiple|several|\\d+)\\s+(?:\\w+\\s+){0,2}videos\\b|先|再|然后|接着|之后|最后|图片|照片|静态图|画像|单独|分别|片段|多个视频', instruction, re.I)
    if explicit_artifact or primary_case_resource(list(case_resources or []), 'i2i')[0] is not None:
        return (implementation, None)
    evidence = _internal_motion_target_evidence(bases[-1])
    if evidence is None:
        return (implementation, None)
    normalized = copy.deepcopy(consumer)
    normalized['decomposition_step'] = {**copy.deepcopy(last), 'description': instruction, 'input_type': 'video', 'input_types': ['video'], 'depends_on': []}
    return ([normalized], {'action': 'fold_internal_target_image_generation', 'reason': 'The motion Base already generates its target image from text and consumes it with the driving video; no independent image artifact was requested.', 'removed_step': copy.deepcopy(producer), 'original_consumer': copy.deepcopy(consumer), 'graph_evidence': evidence, 'remaining_operations': [consumer['operation']]})

def classification_for_collapsed_plan(decision, spec, implementation):
    """Keep reports and single-Base execution aligned with the physical plan."""
    final = implementation[-1]
    operations = tuple((step['operation'] for step in implementation))
    decision = AgentDecision(**{**decision.__dict__, 'task_family': final['task_family'], 'subworkflow_sequence': operations})
    spec = RequirementSpec(final['modality'], final['task_family'], operations, spec.capabilities, spec.priorities, spec.constraints, final.get('explicit_model', spec.explicit_model))
    return (decision, spec)
STRUCTURAL_DIAGNOSTIC_CODES = {'edge_shape', 'graph_cycle', 'invalid_output_slot', 'missing_edge_node', 'missing_required_input', 'missing_source_node', 'missing_target_node', 'node_id', 'node_shape', 'port_type_mismatch', 'unknown_input', 'unknown_node_class'}

def agent_request_failure_status(exc: Exception) -> str:
    return 'agent_provider_failed' if isinstance(exc, AgentProviderError) else 'agent_transport_failed'

def preflight_failure_record(exc: Exception, phase: str, previous_plan: dict | None=None) -> dict:
    diagnostics = copy.deepcopy(getattr(exc, 'diagnostics', []))
    allowed_enums = [{'path': item.get('path'), 'actual': item.get('actual'), 'allowed': copy.deepcopy(item.get('allowed'))} for item in diagnostics if isinstance(item, dict) and isinstance(item.get('allowed'), list)]
    registry_hints = [{key: copy.deepcopy(item[key]) for key in ('path', 'code', 'actual', 'closest_node_classes', 'closest_inputs', 'node_interface', 'source_interface', 'target_interface') if key in item} for item in diagnostics if isinstance(item, dict) and any((key in item for key in ('closest_node_classes', 'closest_inputs', 'node_interface', 'source_interface', 'target_interface')))]
    return {'phase': phase, 'error_type': type(exc).__name__, 'failure_kind': failure_kind(str(exc)), 'raw_response': getattr(exc, 'raw_response', None), 'parse_error': getattr(exc, 'parse_error', None), 'provider_error': getattr(exc, 'provider_error', None), 'provider_code': getattr(exc, 'provider_code', None), 'error': str(exc), 'diagnostics': diagnostics, 'graph_edit_application': copy.deepcopy(getattr(exc, 'graph_edit_application', None)), 'allowed_enums': allowed_enums, 'registry_hints': registry_hints, 'previous_plan': copy.deepcopy(previous_plan)}

def current_graph_edit_contract(graph: dict | None, validator=None) -> dict:
    """Describe the only node ids and inputs a repair may edit."""
    nodes = []
    for node_id, node in sorted((graph or {}).items(), key=lambda item: str(item[0])):
        if not isinstance(node, dict):
            continue
        class_type = str(node.get('class_type', ''))
        declared_inputs = []
        if validator is not None and class_type:
            try:
                declared_inputs = list(validator._node_interface(class_type).get('inputs', []))
            except Exception:
                declared_inputs = []
        nodes.append({'node_id': str(node_id), 'class_type': class_type, 'declared_inputs': declared_inputs, 'current_inputs': sorted((str(key) for key in node.get('inputs') or {}))})
    return {'editable_node_ids': [row['node_id'] for row in nodes], 'nodes': nodes, 'policy': 'Only these current node ids are valid existing edit targets. A node id from previous_plan is invalid unless it is added again in the same plan. For set_param, use only declared_inputs; if the required control is absent, choose a statically supported topology repair instead of inventing an input.'}

def sanitize_previous_plan_for_graph(previous_plan: dict | None, graph: dict | None, validator=None) -> tuple[dict | None, list[dict]]:
    """Remove stale or interface-invalid edits before retrying an Agent plan."""
    if not isinstance(previous_plan, dict):
        return (None, [])
    if graph is None:
        return (copy.deepcopy(previous_plan), [])
    plan = copy.deepcopy(previous_plan)
    edits = plan.get('edits')
    if not isinstance(edits, list):
        return (plan, [])
    existing = {str(node_id) for node_id in graph or {}}
    added = {str(edit.get('node_id', edit.get('target', ''))) for edit in edits if isinstance(edit, dict) and str(edit.get('action') or edit.get('op')) == 'add_node'} - {''}
    valid_ids = existing | added
    kept = []
    rejected = []
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            rejected.append({'index': index, 'reason': 'edit_not_object', 'edit': copy.deepcopy(edit)})
            continue
        action = str(edit.get('action') or edit.get('op') or '')
        reason = None
        if action == 'set_param':
            target = edit.get('target', edit.get('node_id'))
            key = edit.get('key') or edit.get('input') or edit.get('param')
            if not isinstance(target, (str, int)) or str(target) not in existing:
                reason = 'set_param_target_not_in_current_graph'
            elif validator is not None and isinstance(key, str):
                node = (graph or {}).get(str(target), {})
                class_type = str(node.get('class_type', ''))
                try:
                    declared = set(validator._node_interface(class_type).get('inputs', []))
                except Exception:
                    declared = set()
                if declared and key not in declared:
                    reason = 'set_param_input_not_declared'
        elif action == 'remove_node':
            target = edit.get('target', edit.get('node_id'))
            if not isinstance(target, (str, int)) or str(target) not in existing:
                reason = 'remove_target_not_in_current_graph'
        elif action in {'connect', 'disconnect'}:
            source = edit.get('source') or edit.get('from')
            destination = edit.get('destination') or edit.get('to')
            refs = []
            if isinstance(source, (list, tuple)) and source:
                refs.append(str(source[0]))
            if isinstance(destination, (list, tuple)) and destination:
                refs.append(str(destination[0]))
            if len(refs) != 2 or any((node_id not in valid_ids for node_id in refs)):
                reason = 'edge_node_not_in_current_or_same_plan_graph'
        if reason:
            rejected.append({'index': index, 'reason': reason, 'edit': copy.deepcopy(edit)})
        else:
            kept.append(copy.deepcopy(edit))
    plan['edits'] = kept
    return (plan, rejected)

def preflight_retry_evidence(failures: list[dict], previous_plan: dict | None, policy: str, *, graph: dict | None=None, validator=None) -> dict | None:
    if not failures:
        return None
    sanitized_plan, rejected_edits = sanitize_previous_plan_for_graph(previous_plan, graph, validator)
    sanitized_failures = copy.deepcopy(failures)
    for failure in sanitized_failures:
        if not isinstance(failure, dict) or not isinstance(failure.get('previous_plan'), dict):
            continue
        failure['previous_plan'], failure['rejected_previous_edits'] = sanitize_previous_plan_for_graph(failure['previous_plan'], graph, validator)
    return {'setup_failure': failures[-1]['error'], 'preflight_failures': sanitized_failures, 'allowed_enums': [enum for failure in failures[-1:] for enum in failure.get('allowed_enums', [])], 'registry_hints': [hint for failure in failures[-1:] for hint in failure.get('registry_hints', [])], 'previous_plan': sanitized_plan, 'rejected_previous_edits': rejected_edits, 'current_graph_edit_contract': current_graph_edit_contract(graph, validator), 'policy': policy}

def generate_plan_with_repair(agent, executor, spec, implementation, **kwargs) -> dict:
    """Use the same progressive feedback policy for all setup/repair paths."""
    if callable(getattr(type(agent), 'augment_repair_context', None)):
        kwargs = agent.augment_repair_context(kwargs)
    evidence = copy.deepcopy(kwargs.get('repair_evidence') or {})
    resources = kwargs.pop('case_resource_paths', None)
    if resources:
        evidence['case_resources'] = [str(path) for path in resources]
    elif kwargs.get('visual_context_paths'):
        evidence['case_resources'] = list(kwargs['visual_context_paths'])
    if evidence and kwargs.get('handoff_contract_context'):
        evidence['upstream_handoff_contract'] = copy.deepcopy(kwargs['handoff_contract_context'])
    failures = evidence.get('preflight_failures', [])
    decomposition = implementation[0].get('decomposition_step') or {} if len(implementation) == 1 else {}
    if len(failures) >= 2 and (not False) and kwargs.get('allow_topology_repair') and (decomposition.get('mask_construction') == 'prospective_placement') and (not decomposition.get('depends_on') or evidence.get('materialized_upstream_source')) and getattr(agent, 'client', None):
        source = next((Path(path) for path in kwargs.get('visual_context_paths') or [] if Path(path).is_file() and Path(path).suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}), None)
        if source is not None:
            from comfyrouter.mask_fallback import estimate_placement_plan
            return estimate_placement_plan(agent, executor, kwargs['graph'], implementation, source, kwargs.get('instruction', ''), evidence)
    if evidence:
        evidence['stage_instruction'] = kwargs.get('instruction', '')
        evidence['overall_instruction'] = kwargs.get('overall_instruction', '')
        kinds = [failure_kind(str(row.get('error') or '')) for row in failures]
        evidence['repair_progress'] = {'planning_attempt': len(failures) + 1, 'planning_limit': repair_plan_limit(), 'failure_kinds': kinds, 'inference_attempt_consumed': False, 'instruction': 'Correct the exact latest failure; do not repeat an unchanged blocked response. Missing information requires checking the supplied interfaces, not inventing nodes or truncating the result.'}
        if 'mask' in json.dumps(evidence, ensure_ascii=False).casefold():
            evidence['mask_repair_guidance'] = "Check saved mask pixels, not only node execution status. An all-black mask selects nothing and must be repaired before editing. MaskFromRGBCMYBW+ uses near-pure RGB thresholds; a blue photo object is not necessarily pure blue. Inspect the source and mask together, adjust or remove a destructive color intersection, and preserve reliable semantic segmentation. For an existing object's precise recoloring or restoration, do not substitute an arbitrary placement rectangle. An image-only restorer can be followed by ImageCompositeMasked with original destination, restored source, and the edit mask to preserve outside pixels."
        if decomposition.get('mask_construction') == 'prospective_placement' and re.search('light[- ]?beams?|glow|illumination|光束|发光|照明', kwargs.get('instruction', '') + '\n' + kwargs.get('overall_instruction', ''), re.I):
            evidence['effect_mask_contract_clarification'] = 'Preserve the base/support geometry in the final edit, not every original surface pixel. The requested light effect permits a narrow beam envelope to overlay illuminated surfaces. A previous review demanding both connected beam footprints and an entirely black intervening hull introduced a conflicting extra restriction. Follow the original task: protect source pixels outside the bounded beam and particle envelopes and preserve structure. Do not repeat a blocked response based on that unrequested whole-hull-black restriction.'
        evidence['invalid_response_evidence'] = [{'error': row.get('error'), 'raw_response': row.get('raw_response')} for row in failures[-1:] if row.get('raw_response')]
        ensure_validator = getattr(executor, 'ensure_validator', None)
        validator = ensure_validator() if callable(ensure_validator) else getattr(executor, 'validator', None)
        registry = getattr(validator, 'node_registry', {})
        class_names = {node.get('class_type') for node in (kwargs.get('graph') or {}).values() if isinstance(node, dict)}
        previous = evidence.get('previous_plan') or {}
        for edit in previous.get('edits', []):
            node = edit.get('node') or {}
            name = node.get('class_type') or edit.get('class_type')
            if name in registry:
                class_names.add(name)
        topology = kwargs.get('allow_topology_repair', False) and (not False)
        if topology and callable(ensure_validator):
            shortlist = local_registry_repair_candidates(executor, evidence, limit=24, handoff_contract=kwargs.get('handoff_contract_context'))
            evidence['local_registry_candidates'] = shortlist
            class_names.update((row['class_type'] for row in shortlist))
        interfaces = []
        for class_name in sorted((str(name) for name in class_names if name)):
            schema = registry.get(class_name, {})
            input_schema = copy.deepcopy(schema.get('input', {}))
            interfaces.append({'class_type': class_name, 'input': input_schema, 'output': schema.get('output', []), 'description': str(schema.get('description') or '')[:1000]})
        evidence['installed_node_interfaces'] = interfaces
        kwargs['repair_evidence'] = evidence
        if topology and len(failures) >= 3 and getattr(agent, 'client', None):
            return reconstruct_plan(agent, executor, spec, implementation, kwargs['graph'], evidence, instruction=kwargs.get('instruction', ''), interfaces=interfaces)
    plan = agent.generate_graph_edit_plan(spec, implementation, **kwargs)
    if evidence and plan.get('requires_composition_replan') and (not False) and getattr(agent, 'client', None):
        evidence['composition_proposal'] = plan.get('workflow_composition_assessment')
        if not kwargs.get('allow_topology_repair'):
            raise AgentOutputError('composition replan requires a structural repair: ' + str(evidence['composition_proposal'])[:1500])
        plan = reconstruct_plan(agent, executor, spec, implementation, kwargs['graph'], evidence, instruction=kwargs.get('instruction', ''), interfaces=interfaces)
    if evidence and plan.get('repair_blocked_reason') and (not plan.get('edits')) and (not plan.get('skill_ids')):
        raise AgentOutputError('repair blocked by missing executable capability: ' + plan['repair_blocked_reason'][:2500])
    return plan

def local_registry_repair_candidates(executor, diagnosis: dict | None, *, limit: int=24, handoff_contract: dict | None=None) -> list[dict]:
    """Return a compact live-registry shortlist for a structural repair.

    The Agent must not conclude that a capability is unavailable merely
    because the current graph does not contain it.  This shortlist exposes
    only locally installed structural operators relevant to the diagnosed
    media contract; it is evidence for Agent selection, never an automatic
    graph mutation.
    """
    validator = executor.ensure_validator()
    registry = validator.node_registry
    handoff_types = {str(row.get('media_type', '')).upper() for row in (handoff_contract or {}).get('producer_dependencies', []) if isinstance(row, dict)}
    if handoff_contract:
        handoff_types.update((str(value).upper() for value in handoff_contract.get('required_input_types', [])))
        handoff_types.add(str(handoff_contract.get('required_output_type', '')).upper())
        if (handoff_contract.get('mask_requirement') or {}).get('status') == 'required':
            handoff_types.add('MASK')
    failure_evidence = json.dumps(diagnosis or {}, ensure_ascii=False)
    handoff_types.update(re.findall('handoff_failed: required (IMAGE|MASK|VIDEO|AUDIO)\\b', failure_evidence, flags=re.IGNORECASE))
    handoff_types = {value.upper() for value in handoff_types}
    priority_classes = []
    if any((marker in failure_evidence.casefold() for marker in ('animatediff', 'context_options', 'context window', 'context_length'))):
        priority_classes.extend(['ADE_StandardUniformContextOptions', 'ADE_StandardStaticContextOptions'])
    if 'VIDEO' in handoff_types or 'video' in failure_evidence.casefold():
        priority_classes.extend(['LoadVideo', 'GetVideoComponents', 'VHS_LoadVideo', 'VHS_VideoInfoLoaded', 'CreateVideo', 'SaveVideo', 'VHS_VideoCombine'])
    if any((word in failure_evidence.casefold() for word in ('upscal', 'scale frames', 'scale video', '放大'))):
        priority_classes.extend(['ImageScaleBy', 'ImageScale', 'ImageUpscaleWithModel', 'UpscaleModelLoader'])
    for media_type in ('MASK', 'IMAGE', 'VIDEO', 'AUDIO'):
        if media_type in handoff_types:
            priority_classes.extend((name for name, _ in MEDIA_INPUT_ADAPTERS[media_type]))
    if 'MASK' in handoff_types:
        priority_classes.extend(['ImageToMask', 'ImageCompositeMasked'])
        if 'VIDEO' in handoff_types:
            priority_classes.extend(['GetVideoComponents', 'CreateVideo'])
    repair_text = failure_evidence.casefold()
    if any((word in repair_text for word in ('semantic', 'segment', 'source-aligned', 'wall', 'canopy', 'recolor', 'face', '墙', '篷', '改色', '人脸', '分割'))):
        priority_classes.extend(['GroundingDinoModelLoader (segment anything)', 'SAMModelLoader (segment anything)', 'GroundingDinoSAMSegment (segment anything)', 'ImageCompositeMasked'])
        if any((word in repair_text for word in ('face', '人脸', '面部'))):
            priority_classes.extend(['FaceAnalysisModels', 'FaceSegmentation'])
    if any((word in repair_text for word in ('saturation', 'hue', 'recolor', 'colour', 'color', '饱和', '改色', '蓝色'))):
        priority_classes.extend(['LayerColor: HSV', 'JWImageSaturation', 'ImageCompositeMasked'])
    if 'mask' in failure_evidence.casefold():
        priority_classes.extend(['SolidMask', 'MaskComposite', 'MaskToImage', 'ImageCompositeMasked', 'CreateShapeMask', 'MaskRectAreaAdvanced', 'GetImageSize', 'CropMask', 'GrowMask', 'InvertMask'])
    if any((word in failure_evidence.casefold() for word in ('title', '文字', '标题', 'glyph', 'text render', 'text overlay', 'typography'))):
        priority_classes.extend(['CR Overlay Text', 'CR Draw Text', 'CreateTextMask', 'Text To Image (mtb)', 'TextOverlay', 'GetImageSize', 'ImageCompositeMasked'])
    if any((word in failure_evidence.casefold() for word in ('watercolor', 'watercolour', 'source-conditioned', 'chunk', 'reassembl', '水彩'))):
        priority_classes.extend(['VHS_LoadVideo', 'VHS_VideoCombine', 'VHS_BatchManager', 'VAEEncode', 'VAEDecode', 'KSampler', 'CheckpointLoaderSimple', 'CLIPTextEncode', 'ADE_LoadAnimateDiffModel', 'ADE_ApplyAnimateDiffModelSimple', 'ADE_UseEvolvedSampling', 'ADE_StandardUniformContextOptions'])
    if 'mask' in failure_evidence.casefold() and any((word in failure_evidence.casefold() for word in ('refine', 'whole', 'multiple', 'exclude', 'over-inclusive', 'overinclusive', 'full-person', 'full-subject', 'spatial selection', 'mask-intersection'))):
        priority_classes.extend(['ImageCrop', 'CropMask', 'MaskComposite', 'GrowMask', 'InvertMask', 'MaskToImage'])
    if any((word in failure_evidence.casefold() for word in ('frame-trim', 'frame trim', 'temporal constraint', 'frame count', 'one-frame trim', 'endpoint frame', 'append', 'interpolat', '视频帧', '裁帧'))):
        priority_classes.extend(['GetImageRangeFromBatch', 'ImageFromBatch', 'ImageBatch', 'RepeatImageBatch'])
    if any((word in failure_evidence.casefold() for word in ('geometry', 'aspect ratio', 'canvas', 'contain', 'letterbox', 'padding', '宽高比', '画布', '补边'))):
        priority_classes.extend(['ImageScale', 'ImageScaleBy', 'ImagePadForOutpaint', 'EmptyImage', 'ImageCompositeMasked'])
    priority_order = {name: i for i, name in enumerate(dict.fromkeys(priority_classes))}
    evidence = json.dumps(diagnosis or {}, ensure_ascii=False).casefold()
    if 'concatenate' in evidence or 'concatenation' in evidence:
        evidence += ' concat join merge'
    if any((marker in evidence for marker in ('handoff', 'loader', 'decode', 'encode', 'reassembl'))):
        evidence += ' load decode encode save combine'
    if 'mask' in evidence and any((marker in evidence for marker in ('protect', 'exclude', 'subtract', 'prospective', 'footprint', 'topology'))):
        evidence += ' mask subtract composite shape rectangle ellipse union add'
    structural_terms = ('batch', 'split', 'concat', 'join', 'merge', 'slice', 'range', 'select', 'mask', 'subtract', 'composite', 'shape', 'rectangle', 'ellipse', 'union', 'add', 'image', 'video', 'context', 'window', 'animatediff', 'load', 'decode', 'encode', 'save', 'combine')
    active_terms = [term for term in structural_terms if term in evidence]
    if not active_terms:
        active_terms = ['batch', 'split', 'concat', 'join', 'merge', 'slice']

    def declared_type(spec):
        if isinstance(spec, (list, tuple)) and spec:
            first = spec[0]
            if isinstance(first, str):
                return first
            if isinstance(first, list):
                return 'COMBO'
        return None
    rows = []
    for class_type, schema in registry.items():
        if not isinstance(schema, dict):
            continue
        haystack = ' '.join((str(class_type), str(schema.get('display_name', '')), str(schema.get('category', '')), str(schema.get('description', '')))).casefold()
        operator_hits = [term for term in active_terms if term in haystack and term not in {'image', 'video'}]
        if not operator_hits and str(class_type) not in priority_order:
            continue
        inputs = schema.get('input', {}) if isinstance(schema.get('input'), dict) else {}
        required = inputs.get('required', {}) if isinstance(inputs.get('required'), dict) else {}
        optional = inputs.get('optional', {}) if isinstance(inputs.get('optional'), dict) else {}
        outputs = list(schema.get('output') or [])
        media_types = {'IMAGE', 'MASK', 'VIDEO'}
        diagnosed_types = {str(value).upper() for value in outputs if str(value).casefold() in evidence or str(value).replace('_', ' ').casefold() in evidence}
        typed_inputs = {str(name): declared_type(spec) for name, spec in {**required, **optional}.items()}
        if not (str(class_type) in priority_order or diagnosed_types or media_types.intersection((str(value).upper() for value in outputs)) or media_types.intersection((str(value).upper() for value in typed_inputs.values() if value))):
            continue
        class_name = str(class_type).casefold()
        score = 3 * len(operator_hits)
        score += 10 * sum((token in class_name for token in ('split', 'concat', 'join', 'merge', 'slice', 'subtract', 'composite', 'shape', 'rectangle', 'ellipse')))
        score += 4 if 'batch' in class_name else 0
        score += sum((2 for value in outputs if str(value).upper() in media_types))
        score += 12 * sum((1 for value in outputs if str(value).upper() in diagnosed_types))
        score += sum((1 for value in typed_inputs.values() if value and str(value).upper() in media_types))
        score -= 6 * sum((token in class_name for token in ('prompt', 'ipadapter', 'latent', 'crop', 'grid', 'schedule')))
        rows.append((score, str(class_type), {'class_type': str(class_type), 'category': schema.get('category'), 'required_inputs': {str(name): declared_type(spec) for name, spec in required.items()}, 'optional_inputs': {str(name): declared_type(spec) for name, spec in optional.items()}, 'outputs': outputs, 'matched_terms': operator_hits}))
    rows.sort(key=lambda item: (0 if item[1] in priority_order else 1, priority_order.get(item[1], 0), -item[0], item[1].casefold()))
    return [row for _, _, row in rows[:limit]]

def repair_mode_for_context(*, failures: list[dict] | None=None, operation: str='', input_types=(), diagnosis: dict | None=None, retry_index: int=1, execution_failed: bool=False) -> dict:
    """Classify a retry and escalate after the first executed repair fails.

    Retry one is parameter/prompt-only. Later retries may be a controlled
    multi-edit topology plan when the accumulated trace identifies a real
    structural or capability issue.
    """
    context = current_execution()
    if context and context.policy.frozen_graph:
        return {'mode': 'parameter', 'allow_topology_repair': False, 'requires_evolution': False, 'reasons': ['frozen graph policy']}
    reasons = []
    codes = {str(item.get('code')) for failure in failures or [] for item in failure.get('diagnostics', []) if isinstance(item, dict)}
    structural = sorted(codes & STRUCTURAL_DIAGNOSTIC_CODES)
    if structural:
        reasons.append('static diagnostics require graph structure repair: ' + ', '.join(structural))
    normalized_inputs = {str(value).upper() for value in input_types or ()}
    repair_context = bool(failures or diagnosis)
    diagnosis_text = json.dumps({'diagnosis': diagnosis or {}, 'failures': failures or []}, ensure_ascii=False).casefold()
    failure_text = json.dumps(failures or [], ensure_ascii=False).casefold()
    structured_structural = False
    structured_reasons = []

    def inspect_structural(value):
        nonlocal structured_structural
        if isinstance(value, dict):
            status = str(value.get('status', '')).casefold()
            failure_class = str(value.get('failure_class', '')).casefold()
            owner = str(value.get('owner', '')).casefold()
            problem_type = str(value.get('problem_type', '')).casefold()
            next_action = str(value.get('next_action', '')).casefold()
            structural_status = status in {'handoff_failed', 'topology_failed', 'connection_failed'}
            structural_class = failure_class in {'handoff', 'workflow', 'topology'}
            structural_problem = any((token in problem_type for token in ('handoff', 'topology', 'connection', 'contract')))
            if structural_status or structural_class or structural_problem:
                structured_structural = True
                structured_reasons.append(f'structured diagnosis identifies {status or failure_class or problem_type or owner}')
            for child in value.values():
                inspect_structural(child)
        elif isinstance(value, list):
            for child in value:
                inspect_structural(child)
    inspect_structural(diagnosis or {})
    if structured_structural:
        reasons.extend(structured_reasons[:3])
    capability_failures = [failure for failure in failures or [] if failure_kind(str(failure.get('error') or '')) in {'missing_executable_repair', 'composition_reconstruction', 'media_interface'}]
    blocked_structure = any(('repair blocked by missing executable capability' in str(row.get('error', '')).casefold() and any((marker in str(row.get('error', '')).casefold() for marker in ('requires a topology', 'requires topology', 'topology-level', 'topology repair is disallowed', 'topology repair is prohibited', 'topology repair is required', 'requires a structural', 'cannot express', '需要拓扑', '需要结构'))) for row in failures or []))
    if blocked_structure:
        reasons.append('blocked parameter plan explicitly requires structural repair')
        structured_structural = True
    if len(capability_failures) >= 2:
        reasons.append('repeated preflight failures require an executable structural repair')
        structured_structural = True
    mask_structure_markers = ('mask handoff failed', 'no free mask', 'missing mask input', 'unbound mask', 'mask contract missing', 'mask interface mismatch', 'required mask has no compatible external loader', '无法接收 mask', 'mask 接口错误', 'mask 连接错误')
    mask_handoff_marker = repair_context and 'MASK' in normalized_inputs and any((marker in diagnosis_text for marker in mask_structure_markers))
    if mask_handoff_marker:
        reasons.append('stage consumes an explicit MASK handoff')
    if any((marker in failure_text for marker in ('handoff_failed', 'required mask has no compatible external loader', 'required external loader', 'missing mask input'))):
        reasons.append('preflight handoff failure requires topology repair')
        structured_structural = True
    contract_requirement = (diagnosis or {}).get('handoff_contract', {}) if isinstance(diagnosis, dict) else {}
    if isinstance(contract_requirement, dict):
        mask_requirement = contract_requirement.get('mask_requirement', {})
        if isinstance(mask_requirement, dict) and str(mask_requirement.get('status', '')).casefold() == 'required':
            reasons.append('handoff contract requires an external MASK interface')
    operation_leaf = str(operation).casefold().rstrip('/').rsplit('/', 1)[-1]
    if repair_context and operation_leaf in {'color_modification', 'video_attribute_edit'}:
        reasons.append('deterministic color replacement may require mask/composite topology')
    if repair_context and any((marker in diagnosis_text for marker in ('deterministic color', 'deterministic recolor', 'colour replacement', 'color selector', 'recolor node', 'recolour node', '确定性换色', '换色节点'))):
        reasons.append('diagnosis requires deterministic color topology')
    if any((marker in diagnosis_text for marker in ('missing node', 'missing connection', 'connection error', 'topology', 'cannot express', 'mask handoff'))):
        reasons.append('diagnosis identifies a structural capability gap')
    diagnosis_action = str((diagnosis or {}).get('next_action') or '').casefold() if isinstance(diagnosis, dict) else ''
    diagnosis_class = str((diagnosis or {}).get('failure_class') or '').casefold() if isinstance(diagnosis, dict) else ''
    nested_analysis = (diagnosis or {}).get('failure_analysis') if isinstance(diagnosis, dict) else None
    if isinstance(nested_analysis, dict):
        diagnosis_action = str(nested_analysis.get('next_action') or diagnosis_action).casefold()
        diagnosis_class = str(nested_analysis.get('failure_class') or diagnosis_class).casefold()
    explicit_parameter_retry = diagnosis_action == 'parameter_retry'
    if diagnosis_action in {'topology_repair', 'composition_repair'}:
        reasons.append(f'combined visual diagnosis requires {diagnosis_action}')
    if explicit_parameter_retry and (not structural) and (not structured_structural) and (diagnosis_class not in {'handoff', 'workflow'}):
        reasons = []
    confirmed_structural = bool(reasons) and (diagnosis_action.startswith('evolve_workflow') or diagnosis_action in {'topology_repair', 'composition_repair'} or diagnosis_class in {'handoff', 'workflow'} or bool(structural) or structured_structural or mask_handoff_marker)
    topology = bool(reasons) and (execution_failed or bool(failures)) and (confirmed_structural or retry_index >= 2)
    return {'mode': 'topology' if topology else 'parameter', 'requires_evolution': bool(reasons), 'allow_topology_repair': topology, 'reasons': reasons or ['failure is expressible through existing node parameters'], 'escalation_policy': 'the supplied diagnosis establishes a structural failure; this retry may add or reconnect multiple nodes in one validated transaction' if topology else 'parameter/prompt-only repair until execution evidence establishes a structural failure'}

def require_controlled_topology_repair(plan: dict, require_edit: bool=False) -> None:
    edits = plan.get('edits', [])
    if require_edit and (not edits) and (not plan.get('requires_composition_replan')):
        raise AgentOutputError('failed repair requires a graph edit or composition replan')
    allowed = {'set_param', 'add_node', 'remove_node', 'connect', 'disconnect'}
    invalid = sorted({str(edit.get('action')) if isinstance(edit, dict) else type(edit).__name__ for edit in edits if not isinstance(edit, dict) or edit.get('action') not in allowed})
    if invalid:
        raise AgentOutputError('controlled topology repair contains invalid edits: ' + ', '.join(invalid))
QUANTIZED_VIDEO_LENGTH_NODES = {'EmptyHunyuanLatentVideo', 'EmptyHunyuanVideo15Latent', 'Wan22ImageToVideoLatent', 'WanFirstLastFrameToVideo'}

def requested_duration_seconds(instruction: str) -> float | None:
    pattern = '(?:for\\s+)?(\\d+(?:\\.\\d+)?)\\s*(?:-\\s*)?(?:seconds?|secs?|s\\b|秒)'
    matches = list(re.finditer(pattern, instruction, re.IGNORECASE))
    candidates = []
    for match in matches:
        before = instruction[max(0, match.start() - 90):match.start()].casefold()
        after = instruction[match.end():match.end() + 70].casefold()
        if re.search('(?:no more than|at most|not exceeding|不超过|最多|≤|<=)\\s*$', before):
            continue
        final = bool(re.search('(?:final|resulting|result|total|produce|output|最终|总时长|输出)[^.;。]{0,65}$', before))
        auxiliary = bool(re.match('\\s*(?:overlap|transition|fade|转场|重叠)', after)) or bool(re.search('(?:overlap|transition|fade|转场|重叠)[^.;。]{0,30}$', before))
        if auxiliary and (not re.match('\\s*(?:video|视频)', after)):
            continue
        component = bool(re.match('\\s*(?:video\\s+)?clips?\\b', after))
        candidates.append((float(match.group(1)), final and (not component), component))
    final_durations = [value for value, final, _ in candidates if final]
    if final_durations:
        return final_durations[-1]
    whole_durations = [value for value, _, component in candidates if not component]
    if len(set(whole_durations)) == 1:
        return whole_durations[0]
    if not whole_durations and len(candidates) == 1:
        return candidates[0][0]
    return None

def requested_fps(instruction: str) -> float | None:
    match = re.search('(\\d+(?:\\.\\d+)?)\\s*(?:fps(?![A-Za-z0-9_])|frames?\\s*(?:per|/)\\s*second|帧\\s*/?\\s*秒)', instruction, re.IGNORECASE)
    return float(match.group(1)) if match else None

def requested_output_frames(instruction: str) -> int | None:
    counts = set()
    for match in re.finditer('(?<![\\d.])(\\d+)\\s*(?:-\\s*)?(?:frames?\\b|帧(?!率))', instruction, re.I):
        before = instruction[max(0, match.start() - 80):match.start()].casefold()
        after = instruction[match.end():match.end() + 25].casefold()
        if re.search('(?:\\bstage|\\bstep|阶段|步骤)\\s*[-_ ]?\\s*$', before):
            continue
        if re.match('\\s*(?:per|/)\\s*second', after):
            continue
        if re.search('(?:no more than|at most|not exceeding|within|up to|fewer than|less than|under|below|不超过|最多|至多|上限(?:为)?|≤|<=)(?:\\s*(?:\\d+(?:\\.\\d+)?|three)\\s*(?:seconds?|secs?|s\\b|秒)\\s*(?:and|or|以及|和|及|与|或|[,、，])\\s*)?\\s*$', before):
            continue
        components = list(re.finditer('(?:each(?:\\s+(?:input|clip|branch))?|per\\s+(?:input|clip|branch)|每路|每段|各段|输入片段)', before))
        finals = list(re.finditer('(?:\\b(?:final|output|total)\\b|输出|最终|合计|总共)', before))
        if components and (not finals or components[-1].start() > finals[-1].start()):
            continue
        counts.add(int(match.group(1)))
    return next(iter(counts)) if len(counts) == 1 else None

def quantized_video_frame_count(length: int | float) -> int:
    """Return the decoded frame count for Wan/Hunyuan temporal latents."""
    value = max(1, int(length))
    return 4 * ((value - 1) // 4) + 1

def nearest_quantized_video_length(target: int | float) -> int:
    """Return the nearest decodable 4n+1 frame count, preferring longer on ties."""
    value = max(1, float(target))
    lower = max(1, 4 * int((value - 1) // 4) + 1)
    upper = lower if lower >= value else lower + 4
    return upper if upper - value <= value - lower else lower

def bounded_quantized_video_length(target, instruction, fps):
    """Use one realizable target for both normalization and validation."""
    duration_caps = re.findall('(?:no more than|at most|not exceeding|within|up to|不超过|最多|≤|<=)\\s*(\\d+(?:\\.\\d+)?)\\s*(?:seconds?|secs?|s\\b|秒)', instruction, re.IGNORECASE)
    duration_cap = min(map(float, duration_caps)) if duration_caps else None
    frame_caps = re.findall('(?:no more than|at most|not exceeding|within|up to|不超过|最多|≤|<=)\\s*(\\d+)\\s*(?:frames?|帧)', instruction, re.IGNORECASE)
    if os.environ.get('COMFYROUTER_VIDEO_FRAME_BUDGET'):
        frame_caps.append(os.environ['COMFYROUTER_VIDEO_FRAME_BUDGET'])
    limits = [int(value) for value in frame_caps]
    if duration_cap is not None and fps:
        limits.append(int(duration_cap * fps + 1e-09))
    replacement = nearest_quantized_video_length(target)
    return min(replacement, quantized_video_frame_count(min(limits))) if limits else replacement

def normalize_quantized_video_timing(graph: dict, instruction: str) -> list[dict]:
    """Normalize Wan/Hunyuan lengths and encoder FPS to realizable timing."""
    duration = requested_duration_seconds(instruction)
    explicit_fps = requested_fps(instruction)
    encoder_nodes = [(str(node_id), node) for node_id, node in graph.items() if isinstance(node, dict) and node.get('class_type') in {'VHS_VideoCombine', 'CreateVideo'}]
    encoder_fps = None
    for _, node in encoder_nodes:
        inputs = node.get('inputs', {})
        value = inputs.get('frame_rate', inputs.get('fps'))
        if isinstance(value, (int, float)) and value > 0:
            encoder_fps = float(value)
            break
    target_frames = None
    if duration is not None:
        basis_fps = explicit_fps or encoder_fps
        if basis_fps is not None:
            target_frames = nearest_quantized_video_length(round(duration * basis_fps))
    audits = []
    normalized_frames = None
    for node_id, node in graph.items():
        if not isinstance(node, dict) or node.get('class_type') not in QUANTIZED_VIDEO_LENGTH_NODES:
            continue
        inputs = node.get('inputs', {})
        length = inputs.get('length')
        if not isinstance(length, (int, float)) or length <= 0:
            continue
        basis_fps = explicit_fps or encoder_fps
        replacement = bounded_quantized_video_length(target_frames or length, instruction, basis_fps)
        decoded_before = quantized_video_frame_count(length)
        normalized_frames = replacement
        if length != replacement or decoded_before != int(length):
            inputs['length'] = replacement
            audits.append({'node_id': str(node_id), 'class_type': node.get('class_type'), 'parameter': 'length', 'before': length, 'after': replacement, 'decoded_frames_before': decoded_before, 'reason': 'Wan/Hunyuan decoded frame counts must be 4n+1'})
    rates = {float(value) for value in re.findall('(\\d+(?:\\.\\d+)?)\\s*fps(?![A-Za-z0-9_])', instruction, re.IGNORECASE)}
    if explicit_fps is not None and len(rates) == 1 and (normalized_frames is not None):
        for node_id, node in encoder_nodes:
            inputs = node.get('inputs', {})
            parameter = 'frame_rate' if 'frame_rate' in inputs else 'fps'
            before = inputs.get(parameter)
            if isinstance(before, (int, float)) and abs(float(before) - explicit_fps) > 1e-09:
                inputs[parameter] = explicit_fps
                audits.append({'node_id': node_id, 'class_type': node.get('class_type'), 'parameter': parameter, 'before': before, 'after': explicit_fps, 'reason': 'honor the unambiguous explicit stage frame rate'})
    if duration is not None and explicit_fps is None and (normalized_frames is not None):
        exact_fps = normalized_frames / duration
        for node_id, node in encoder_nodes:
            inputs = node.get('inputs', {})
            parameter = 'frame_rate' if 'frame_rate' in inputs else 'fps'
            before = inputs.get(parameter)
            if isinstance(before, (int, float)) and abs(float(before) - exact_fps) > 1e-09:
                inputs[parameter] = exact_fps
                audits.append({'node_id': node_id, 'class_type': node.get('class_type'), 'parameter': parameter, 'before': before, 'after': exact_fps, 'reason': 'preserve the requested duration with a realizable 4n+1 frame count'})
    return audits

def validate_temporal_parameters(graph: dict, instruction: str) -> dict:
    """Validate calculable frame-count invariants before submitting a video graph."""
    from .quality_policy import visual_tolerance_enabled, temporal_near
    tolerant = visual_tolerance_enabled()
    duration = requested_duration_seconds(instruction)
    explicit_fps = requested_fps(instruction)
    explicit_output_frames = requested_output_frames(instruction)
    memo: dict[str, int | None] = {}

    def scalar(value):
        if isinstance(value, list) and value:
            source = graph.get(str(value[0]), {})
            if source.get('class_type') in {'PrimitiveInt', 'PrimitiveFloat'}:
                return source.get('inputs', {}).get('value')
        return value

    def frame_count(node_id: str) -> int | None:
        node_id = str(node_id)
        if node_id in memo:
            return memo[node_id]
        memo[node_id] = None
        node = graph.get(node_id, {})
        inputs = node.get('inputs', {}) if isinstance(node, dict) else {}
        class_type = node.get('class_type') if isinstance(node, dict) else None
        result = None
        if class_type in {'VHS_LoadVideo', 'LoadVideo'}:
            cap = inputs.get('frame_load_cap')
            filename = inputs.get('video', inputs.get('file'))
            if isinstance(filename, str) and filename not in MEDIA_PLACEHOLDERS.get('VIDEO', set()):
                path = COMFYUI_INPUT / filename
                if path.is_file():
                    try:
                        import cv2
                        capture = cv2.VideoCapture(str(path))
                        try:
                            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                            source_fps = capture.get(cv2.CAP_PROP_FPS)
                        finally:
                            capture.release()
                        force_rate = inputs.get('force_rate', 0)
                        if count > 0 and class_type == 'VHS_LoadVideo' and (source_fps > 0):
                            target = 1 / (float(force_rate) if force_rate else source_fps)
                            offset, source_index, selected, evaluated = (target, 1, 0, 0)
                            skip = int(inputs.get('skip_first_frames', 0))
                            every = max(1, int(inputs.get('select_every_nth', 1)))
                            for _ in range(10000000):
                                if offset < target:
                                    source_index += 1
                                    if source_index > count:
                                        result = selected
                                        break
                                    offset += 1 / source_fps
                                if offset < target:
                                    continue
                                offset -= target
                                evaluated += 1
                                if evaluated > skip and (evaluated - skip - 1) % every == 0:
                                    selected += 1
                                    if isinstance(cap, (int, float)) and cap > 0 and (selected >= cap):
                                        result = selected
                                        break
                        elif count > 0 and (not force_rate):
                            result = max(0, count - int(inputs.get('skip_first_frames', 0)))
                            if isinstance(cap, (int, float)) and cap > 0:
                                result = min(result, int(cap))
                    except (ImportError, OSError, ValueError):
                        pass
        elif class_type in QUANTIZED_VIDEO_LENGTH_NODES:
            length = inputs.get('length')
            result = quantized_video_frame_count(length) if isinstance(length, (int, float)) and length > 0 else None
        elif class_type in {'KSampler', 'KSamplerAdvanced'}:
            source = inputs.get('latent_image')
            result = frame_count(str(source[0])) if isinstance(source, list) and source else None
        elif class_type in {'VAEDecode', 'VAEDecodeTiled'}:
            source = inputs.get('samples')
            result = frame_count(str(source[0])) if isinstance(source, list) and source else None
        elif class_type == 'GetVideoComponents':
            source = inputs.get('video')
            result = frame_count(str(source[0])) if isinstance(source, list) and source else None
        elif class_type == 'FrameInterpolate':
            source = inputs.get('images')
            count = frame_count(str(source[0])) if isinstance(source, list) and source else None
            multiplier = scalar(inputs.get('multiplier'))
            if count is not None and isinstance(multiplier, int) and (multiplier >= 1):
                result = (count - 1) * multiplier + 1 if count >= 2 else count
        elif class_type == 'GetImageRangeFromBatch':
            count = inputs.get('num_frames')
            start = inputs.get('start_index', 0)
            source = inputs.get('images')
            source_count = frame_count(str(source[0])) if isinstance(source, list) and source else None
            if source_count is not None and isinstance(count, (int, float)) and (count > 0) and isinstance(start, int):
                start = source_count - int(count) if start == -1 else start
                if start < 0 or start >= source_count or start + int(count) > source_count:
                    raise AgentOutputError(f'temporal constraint failed at node {node_id}: GetImageRangeFromBatch range [{start}, {start + int(count)}) exceeds {source_count} input frames; the node raises an out-of-range error rather than clamping')
                result = int(count)
        elif class_type == 'ImageFromBatch':
            source = inputs.get('image')
            source_count = frame_count(str(source[0])) if isinstance(source, list) and source else None
            start, count = (inputs.get('batch_index', 0), inputs.get('length', 1))
            if source_count is not None and isinstance(start, int) and isinstance(count, int):
                result = min(count, max(0, source_count - start))
        elif class_type == 'ImageBatch':
            sources = [inputs.get('image1'), inputs.get('image2')]
            counts = [frame_count(str(source[0])) for source in sources if isinstance(source, list) and source]
            result = sum(counts) if len(counts) == 2 and all((x is not None for x in counts)) else None
        elif class_type == 'CrossFadeImages':
            first, second = (inputs.get('images_1'), inputs.get('images_2'))
            first_count = frame_count(str(first[0])) if isinstance(first, list) and first else None
            second_count = frame_count(str(second[0])) if isinstance(second, list) and second else None
            start = inputs.get('transition_start_index')
            span = inputs.get('transitioning_frames')
            if first_count is not None and second_count is not None:
                if first_count != second_count:
                    raise AgentOutputError(f'temporal constraint failed at node {node_id}: CrossFadeImages inputs have {first_count} and {second_count} frames; coupled inputs must be equal')
                if isinstance(start, int) and isinstance(span, int) and (start + span > first_count):
                    raise AgentOutputError(f'temporal constraint failed at node {node_id}: transition end {start + span} exceeds input length {first_count}')
                result = second_count
        memo[node_id] = result
        return result
    outputs = []
    for node_id, node in graph.items():
        if not isinstance(node, dict) or node.get('class_type') not in {'VHS_VideoCombine', 'CreateVideo'}:
            continue
        inputs = node.get('inputs', {})
        source = inputs.get('images')
        count = frame_count(str(source[0])) if isinstance(source, list) and source else None
        fps = inputs.get('frame_rate', inputs.get('fps'))
        outputs.append({'node_id': str(node_id), 'frames': count, 'fps': fps})
        if explicit_output_frames is not None and count is not None and (count != explicit_output_frames) and (not (tolerant and abs(count - explicit_output_frames) <= 2)):
            raise AgentOutputError(f'temporal constraint failed at output node {node_id}: task requires {explicit_output_frames} frames, graph produces {count}')
        if duration is not None and count is not None and isinstance(fps, (int, float)) and (fps > 0):
            target = round(duration * float(fps))
            if explicit_output_frames is not None:
                target = explicit_output_frames
            elif explicit_fps is not None and (not any((isinstance(item, dict) and item.get('class_type') in {'GetImageRangeFromBatch', 'ImageFromBatch'} for item in graph.values()))) and any((isinstance(item, dict) and item.get('class_type') in QUANTIZED_VIDEO_LENGTH_NODES for item in graph.values())):
                target = bounded_quantized_video_length(target, instruction, float(fps))
            if count != target and (not (tolerant and temporal_near(count / float(fps), target / float(fps), float(fps)))):
                raise AgentOutputError(f'temporal constraint failed at output node {node_id}: task requires {duration:g}s * {float(fps):g}fps = {target} frames, graph produces {count}')
    return {'requested_duration_seconds': duration, 'outputs': outputs}

def base_graph(store, spec, implementation, *, agent=None, instruction: str=''):
    if not implementation:
        raise ClosedLoopError('empty implementation plan')
    context = current_execution()
    if context and context.hooks.base_provider:
        return context.hooks.base_provider(store, spec, implementation, agent=agent, instruction=instruction)
    raise ClosedLoopError('RAG execution requires the selected frozen workflow provider')

def audit_usage(audit_dir: Path) -> dict:
    calls = []
    totals = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    by_agent: dict[str, dict[str, int]] = {}
    for path in sorted(audit_dir.glob('*.json'), key=lambda item: item.stat().st_mtime):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        usage = record.get('response', {}).get('usage') or {}
        row = {'agent_id': record.get('agent_id'), 'input_tokens': int(usage.get('input_tokens') or 0), 'output_tokens': int(usage.get('output_tokens') or 0), 'total_tokens': int(usage.get('total_tokens') or 0), 'api_attempts': 1 + len(record.get('request_errors') or []), 'audit_file': str(path)}
        calls.append(row)
        bucket = by_agent.setdefault(str(row['agent_id']), {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0})
        bucket['calls'] += 1
        for key in totals:
            totals[key] += row[key]
            bucket[key] += row[key]
    return {'calls': calls, 'by_agent': by_agent, 'totals': totals}

def notify_evaluation_context(attempt: dict, result: Path, attempt_dir: Path, output_types: dict | None=None) -> None:
    """Expose actual outputs, including reused artifacts, before evaluation.

    This observer receives no selected workflows or reference decomposition.
    Existing three-argument evaluator callbacks remain compatible.
    """
    context = current_execution()
    if not context or not context.hooks.evaluation_context_observer:
        return
    output_types = output_types or {}
    records = attempt.get('stages') or [attempt]
    stages = []
    for record in records:
        stage_id = record.get('stage_id')
        stages.append({'stage_id': stage_id, 'output_type': output_types.get(stage_id), 'status': 'reused' if record.get('status') == 'reused' else record.get('execution', {}).get('status'), 'outputs': copy.deepcopy(record.get('outputs', [])), 'primary_output': record.get('primary_output'), 'diagnostic_outputs': copy.deepcopy(record.get('diagnostic_outputs', [])), 'reused_from_attempt': record.get('reused_from_attempt')})
    context.hooks.evaluation_context_observer({'attempt': attempt.get('attempt'), 'attempt_dir': str(attempt_dir), 'result': str(result), 'stages': stages})

def official_check(case_id: str, result: Path, attempt_dir: Path) -> dict:
    context = current_execution()
    if context and context.hooks.evaluator:
        return context.hooks.evaluator(case_id, result, attempt_dir)
    out = attempt_dir / 'comfybench_evaluation.json'
    evaluator_python = sys.executable
    for candidate in (ROOT.parent / 'ComfyUI' / '.venv' / 'bin' / 'python', ROOT.parent / 'venv' / 'bin' / 'python'):
        if candidate.is_file():
            probe = subprocess.run([str(candidate), '-c', 'import cv2'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if probe.returncode == 0:
                evaluator_python = str(candidate)
                break
    cmd = [evaluator_python, str(ROOT / 'tools' / 'evaluate_comfybench_case.py'), '--case-id', case_id, '--result', str(result), '--out', str(out)]
    last = None
    for evaluation_try in range(1, 2):
        proc = subprocess.run(cmd, cwd=config.COMFYBENCH_ROOT, text=True, capture_output=True)
        (attempt_dir / 'comfybench_evaluator_stdout.log').write_text(proc.stdout or '', encoding='utf-8')
        (attempt_dir / 'comfybench_evaluator_stderr.log').write_text(proc.stderr or '', encoding='utf-8')
        if proc.returncode == 0:
            try:
                evaluation = json.loads(out.read_text(encoding='utf-8'))
                official = evaluation.get('comfybench', {})
                analysis = official.get('analysis')
                judgment = official.get('judgment')
                if not isinstance(analysis, str) or not analysis.strip():
                    raise ValueError('official evaluator returned empty analysis')
                if judgment not in {'True', 'False'}:
                    raise ValueError('official evaluator returned invalid judgment')
                return evaluation
            except (OSError, json.JSONDecodeError) as exc:
                last = {'status': 'evaluator_error', 'error': str(exc), 'stdout': proc.stdout[-4000:]}
            except (TypeError, ValueError) as exc:
                last = {'status': 'evaluator_error', 'error': str(exc), 'stdout': proc.stdout[-4000:]}
        else:
            last = {'status': 'evaluator_error', 'error': proc.stderr[-4000:], 'stdout': proc.stdout[-4000:]}
    return {**(last or {'status': 'evaluator_error'}), 'evaluation_retries': 3}

def prepare_composition(agent, store, executor, implementation, report, checkpoint, case_resources=None, stage_runtime_param_edits=None):
    typed = composition_stages(implementation)
    case_resources = list(case_resources or [])
    stage_runtime_param_edits = list(stage_runtime_param_edits or [])
    unknown_stages = {edit['stage_id'] for edit in stage_runtime_param_edits} - {stage.stage_id for stage in typed}
    if unknown_stages:
        raise ClosedLoopError('stage parameter override references unknown stages: ' + ', '.join(sorted(unknown_stages)))
    stage_specs = [spec_for_stage(step) for step in implementation]
    bases = [base_graph(store, spec, [step], agent=agent, instruction=str((step.get('decomposition_step') or {}).get('description') or report.get('instruction') or '')) for spec, step in zip(stage_specs, implementation)]
    normalized, normalization = _collapse_internal_motion_target(implementation, bases, str(report.get('instruction') or ''), case_resources, stage_runtime_param_edits)
    if normalization:
        report.setdefault('decomposition_normalizations', []).append(normalization)
        implementation[:] = normalized
        report['_collapsed_to_single_leaf'] = True
        return None
    normalized_implementation = _prune_unconsumed_mask_stages(implementation, bases)
    if normalized_implementation is not implementation:
        implementation[:] = normalized_implementation
        report.setdefault('mask_contract_normalizations', []).append({'action': 'remove_unconsumed_mask_stage', 'reason': 'The downstream leaf documents a single-image input and has no complete MASK-aware replacement Skill; the generated mask stage is not part of its contract.', 'remaining_operations': [str(item.get('operation')) for item in implementation]})
        if len(normalized_implementation) == 1:
            report['_collapsed_to_single_leaf'] = True
            return None
        return prepare_composition(agent, store, executor, implementation, report, checkpoint, case_resources, stage_runtime_param_edits)
    mask_adapters: list[dict | None] = [None] * len(bases)
    for index, (typed_stage, step) in enumerate(zip(typed, implementation)):
        has_mask_dependency = any((next((producer.output_type for producer in typed if producer.stage_id == dependency), None) == 'MASK' for dependency in typed_stage.depends_on))
        if has_mask_dependency:
            mask_role = composition_mask_role(typed_stage, implementation)
            bases[index], mask_adapters[index] = apply_mask_handoff_adapter(bases[index], str(step.get('operation', '')), mask_role=mask_role)
    planned_graphs = list(bases)
    geometry_overrides: list[dict | None] = [None] * len(bases)
    states = []
    report['composition'] = {'status': 'preparing', 'trigger': 'multi_operation_classification', 'stages': [], 'execution_order': [stage.stage_id for stage in typed], 'attempt_policy': 'Up to three bounded attempts. Reuse successful unaffected DAG stages; after a repair, rerun only the selected failing stage and its descendants.'}
    for index, (typed_stage, step, stage_spec, base) in enumerate(zip(typed, implementation, stage_specs, bases), 1):
        deferred_handoff_types = deferred_stage_handoff_types(typed_stage, typed)
        handoff_inputs_by_type, handoff_inputs = bind_composition_preflight_inputs(base, typed_stage, case_resources, preserve_upstream_mask_placeholder=any((str(next((producer.output_type for producer in typed if producer.stage_id == dependency), '')).upper() == 'MASK' for dependency in typed_stage.depends_on)))
        preflight_source, preflight_media_type = primary_case_resource(ordered_stage_resources(case_resources, typed_stage.instruction), typed_stage.modality)
        defer_grounded_mask = typed_stage.output_type == 'MASK' and bool(typed_stage.depends_on) and ((step.get('decomposition_step') or {}).get('mask_construction') == 'prospective_placement') and (not False)
        if defer_grounded_mask:
            preflight_source = None
        source = None
        media_type = None
        if not typed_stage.depends_on and handoff_inputs:
            source, media_type = primary_case_resource(ordered_stage_resources(case_resources, typed_stage.instruction), typed_stage.modality)
            source_geometry = media_file_geometry_contract(source, media_type)
            if source_geometry and typed_stage.input_type == typed_stage.output_type and (graph_geometry_contract(base) is None):
                geometry_overrides[index - 1] = source_geometry
        geometry_context = handoff_geometry_contexts(typed, planned_graphs, geometry_overrides, planned_stage_ids={state['typed'].stage_id for state in states})[index - 1]
        if source is not None:
            external_context = external_handoff_geometry_context(base, source, media_type, typed_stage.stage_id)
            if external_context:
                external_context['current_output_geometry'] = geometry_context['current_output_geometry']
                external_context['downstream_consumers'] = geometry_context['downstream_consumers']
                geometry_context = external_context
        else:
            geometry_context = composition_external_geometry_context(base, typed_stage, typed, case_resources, geometry_context)
        current = None
        errors = []
        stage_overrides = [{key: value for key, value in edit.items() if key != 'stage_id'} for edit in stage_runtime_param_edits if edit['stage_id'] == typed_stage.stage_id]
        preflight_failures = []
        previous_plan = None
        plans = []
        handoff_contract = build_handoff_contract_context(typed_stage, typed, implementation, graph=base)
        try:
            executor.static_validate(base, allow_external_placeholders=bool(deferred_handoff_types))
        except StaticValidationError as exc:
            errors.append(str(exc))
            failure = preflight_failure_record(exc, 'base_static_validation')
            preflight_failures.append(failure)
            plans.append({'try': 0, **failure})
        try:
            validate_stage_handoff_contract(base, handoff_contract, case_resources=case_resources, validator=executor.validator)
        except AgentOutputError as exc:
            failure = preflight_failure_record(exc, 'base_handoff_validation')
            preflight_failures.append(failure)
            plans.append({'try': 0, **failure})
        if (defer_grounded_mask or None) and (not preflight_failures):
            current = copy.deepcopy(base)
            plans.append({'try': 0, 'status': 'visual_planning_deferred_until_source_materialized', 'reason': 'Mask or geometry coordinates require actual upstream pixels.'})
        for plan_try in range(1, repair_plan_limit() + 1) if current is None else []:
            try:
                repair_mode = repair_mode_for_context(failures=preflight_failures, operation=str(step.get('operation', '')), input_types=stage_input_types(typed_stage), execution_failed=bool(preflight_failures))
                topology_repair = repair_mode.get('allow_topology_repair', False)
                plan = generate_plan_with_repair(agent, executor, stage_spec, [step], case_resource_paths=case_resources, graph=base if current is None else current, instruction=typed_stage.instruction, overall_instruction=str(report.get('instruction', '')), handoff_geometry_context=geometry_context, handoff_contract_context=handoff_contract, visual_context_paths=[str(preflight_source)] if preflight_source is not None else [], repair_evidence=preflight_retry_evidence(preflight_failures, previous_plan, 'Repair the diagnosed handoff/topology contract. You may add and connect the minimum required input/adapter nodes in one transaction; preserve all unrelated graph paths.' if topology_repair else 'Repair only existing parameter and prompt bindings.', graph=base if current is None else current, validator=executor.validator), parameter_only=not topology_repair, allow_topology_repair=topology_repair)
                previous_plan = plan
                if topology_repair or plan.get('repair_transaction'):
                    require_controlled_topology_repair(plan, require_edit=not bool(plan.get('skill_ids')))
                else:
                    require_parameter_only(plan, require_edit=bool(errors))
                if plan['requires_composition_replan']:
                    raise AgentOutputError('controlled composition replan was requested but no executable replacement stage graph was supplied')
                plan_base, selection = graph_selected_by_execution_agent(plan, base if current is None else current, executor)
                if selection:
                    plan['execution_graph_selection'] = selection
                selected_contract = build_handoff_contract_context(typed_stage, typed, implementation, graph=plan_base)
                candidate = apply_graph_edits(plan_base, plan['edits'], executor.validator, validate_after_each=not (topology_repair or plan.get('repair_transaction')), allow_external_placeholders=True)
                stage_overrides = [{key: value for key, value in edit.items() if key != 'stage_id'} for edit in stage_runtime_param_edits if edit['stage_id'] == typed_stage.stage_id]
                if stage_overrides:
                    candidate = apply_graph_edits(candidate, stage_overrides, executor.validator, allow_external_placeholders=bool(deferred_handoff_types))
                temporal_instruction = typed_stage.instruction
                if index == len(typed) and requested_duration_seconds(temporal_instruction) is None:
                    temporal_instruction = str(report.get('instruction', ''))
                timing_normalizations = normalize_quantized_video_timing(candidate, temporal_instruction)
                if timing_normalizations:
                    plan['deterministic_timing_normalizations'] = timing_normalizations
                upscale_normalizations = normalize_upscale_model(candidate, temporal_instruction)
                if upscale_normalizations:
                    plan['deterministic_upscale_normalizations'] = upscale_normalizations
                bind_composition_preflight_inputs(candidate, typed_stage, case_resources, preserve_upstream_mask_placeholder=any((str(next((producer.output_type for producer in typed if producer.stage_id == dependency), '')).upper() == 'MASK' for dependency in typed_stage.depends_on)))
                input_preflight = validate_stage_handoff_contract(candidate, selected_contract, case_resources=case_resources, validator=executor.validator)
                executor.static_validate(candidate, allow_external_placeholders=bool(deferred_handoff_types))
                validate_temporal_parameters(candidate, temporal_instruction)
                current = candidate
                plans.append({'try': plan_try, 'repair_mode': repair_mode, 'plan': plan})
                break
            except (AgentTransportError, AgentProviderError):
                raise
            except ComfyUITransportError:
                raise
            except (AgentOutputError, StaticValidationError, ClosedLoopError) as exc:
                errors.append(str(exc))
                failure = preflight_failure_record(exc, 'composition_stage_preflight', previous_plan)
                preflight_failures.append(failure)
                plans.append({'try': plan_try, 'repair_mode': repair_mode, **failure})
        if current is None:
            report['composition']['status'] = 'repair_preflight_failed'
            report['repair_terminal'] = {'reason': 'planning_budget_exhausted', 'planning_limit': repair_plan_limit(), 'stage_id': typed_stage.stage_id, 'last_failure': preflight_failures[-1] if preflight_failures else None}
            raise RepairPreflightError(f'composition stage {typed_stage.stage_id} preflight failed: ' + (errors[-1] if errors else 'unknown error'))
        handoff_inputs_by_type, handoff_inputs = composition_handoff_inputs(current, typed_stage)
        planned_graphs[index - 1] = current
        effective_geometry = graph_geometry_contract(current)
        if effective_geometry:
            geometry_overrides[index - 1] = effective_geometry
        stage_record = {'stage': typed_stage.__dict__, 'requirement_spec': stage_spec.__dict__, 'implementation': step, 'base_graph': base, 'generation_plan_attempts': plans, 'preflight_graph': current, 'handoff_geometry_context': geometry_context, 'input_preflight': input_preflight, 'handoff_inputs': [{'node_id': node, 'input': key} for node, key in handoff_inputs], 'handoff_inputs_by_type': {media_type: [{'node_id': node, 'input': key} for node, key in nodes] for media_type, nodes in handoff_inputs_by_type.items()}, 'mask_handoff_adapter': mask_adapters[index - 1], 'runtime_parameter_overrides': stage_overrides, 'deferred_grounded_mask': defer_grounded_mask}
        report['composition']['stages'].append(stage_record)
        states.append({'typed': typed_stage, 'spec': stage_spec, 'implementation': step, 'base_graph': base, 'graph': current, 'handoff_inputs': handoff_inputs, 'handoff_inputs_by_type': handoff_inputs_by_type, 'handoff_geometry_context': geometry_context, 'mask_handoff_adapter': mask_adapters[index - 1], 'deferred_grounded_mask': defer_grounded_mask})
        checkpoint()
    report['composition']['status'] = 'ready'
    checkpoint()
    return states

def composition_diagnosis_media(attempt_row, states, attempt_dir, case_resources=None, prior_attempts=None):
    """Persist visual evidence for ordered, Agent-owned stage diagnosis."""
    media_paths = []
    visual_manifest = []
    descriptors = []
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    video_extensions = {'.gif', '.mp4', '.webm', '.mov'}
    for source in list(case_resources or []):
        source = Path(source)
        if source.suffix.lower() in image_extensions and source.is_file():
            media_paths.append(str(source))
            visual_manifest.append({'position': len(media_paths), 'role': 'original_input', 'path': str(source)})
            break
    for index, (stage_row, state) in enumerate(zip(attempt_row.get('stages', []), states), 1):
        visual_inputs = []
        metadata = {}
        media_error = None
        outputs = [Path(path) for path in stage_row.get('outputs', [])]
        primary_outputs, diagnostic_outputs = split_primary_and_diagnostic_outputs(outputs)
        reused_media = None
        reused_from = stage_row.get('reused_from_attempt')
        if stage_row.get('status') == 'reused' and reused_from is not None:
            source_attempt = next((row for row in reversed(list(prior_attempts or [])) if row.get('attempt') == reused_from), None)
            if source_attempt:
                source_stage = next((row for row in source_attempt.get('stages', []) if row.get('stage_id') == stage_row.get('stage_id')), None)
                candidate = (source_stage or {}).get('diagnosis_media')
                paths = [Path(str(item.get('path'))) for item in (candidate or {}).get('visual_inputs', []) if isinstance(item, dict) and item.get('path')]
                if paths and all((path.is_file() for path in paths)):
                    reused_media = candidate
                    visual_inputs = paths
                    metadata = copy.deepcopy(candidate.get('video_metadata') or {})
                    media_error = candidate.get('media_error')
        images = [path for path in primary_outputs if path.suffix.lower() in image_extensions and path.is_file()]
        if reused_media is not None:
            pass
        elif images:
            visual_inputs = images[-1:]
        else:
            videos = [path for path in primary_outputs if path.suffix.lower() in video_extensions and path.is_file()]
            if videos:
                sampled, metadata, media_error = sample_video(videos[-1], f'composition-a{attempt_row['attempt']:02d}-s{index:02d}', 5)
                frame_dir = attempt_dir / f'stage_{index:02d}' / 'diagnosis_frames'
                frame_dir.mkdir(parents=True, exist_ok=True)
                for frame_index, frame in enumerate(sampled, 1):
                    source = Path(frame)
                    if source.is_file():
                        target = frame_dir / f'frame_{frame_index:02d}.png'
                        shutil.copy2(source, target)
                        visual_inputs.append(target)
            elif not primary_outputs:
                media_error = 'stage produced no output file'
            else:
                media_error = 'stage output is not supported visual media'
        visual_inputs.extend((path for path in diagnostic_outputs[:3] if path.suffix.lower() in image_extensions and path.is_file()))
        positions = []
        for media in visual_inputs:
            media_paths.append(str(media))
            role = 'effective_mask' if str(state['typed'].output_type).upper() == 'MASK' else 'stage_output'
            item = {'position': len(media_paths), 'role': role, 'stage_id': state['typed'].stage_id, 'path': str(media)}
            positions.append(item)
            visual_manifest.append(item)
        execution = stage_row.get('execution', {})
        runtime_graph = stage_row.get('runtime_graph')
        if not isinstance(runtime_graph, dict):
            history = execution.get('history', {}) if isinstance(execution, dict) else {}
            prompt = history.get('prompt') if isinstance(history, dict) else None
            if isinstance(prompt, list) and len(prompt) >= 3 and isinstance(prompt[2], dict):
                runtime_graph = copy.deepcopy(prompt[2])
        if not isinstance(runtime_graph, dict):
            runtime_graph = copy.deepcopy(stage_row.get('graph', {}))
        stage_row['runtime_graph'] = copy.deepcopy(runtime_graph)
        descriptors.append({'stage_id': state['typed'].stage_id, 'stage_index': index, 'operation': state['implementation']['operation'], 'atomic_instruction': state['typed'].instruction, 'input_type': state['typed'].input_type, 'output_type': state['typed'].output_type, 'handoff': stage_row.get('handoff'), 'executed_graph': copy.deepcopy(runtime_graph), 'output_files': [str(path) for path in outputs], 'primary_output': str(primary_outputs[-1]) if primary_outputs else None, 'diagnostic_outputs': [str(path) for path in diagnostic_outputs], 'execution': {'status': execution.get('status'), 'error': execution.get('error') or (stage_row.get('geometry_validation') or {}).get('error'), 'node_errors': execution.get('node_errors')}, 'execution_passed': stage_row.get('execution_passed'), 'geometry_validation': copy.deepcopy(stage_row.get('geometry_validation')), 'geometry_normalization': copy.deepcopy(stage_row.get('geometry_normalization')), 'visual_inputs': positions, 'video_metadata': metadata, 'media_error': media_error})
        stage_row['diagnosis_media'] = {'visual_inputs': positions, 'video_metadata': metadata, 'media_error': media_error, 'reused_from_attempt': reused_from if reused_media is not None else None}
    return (descriptors, media_paths, visual_manifest)

def merge_runtime_repair_inputs(candidate: dict, runtime_graph: dict) -> dict:
    """Expose executor-injected causal inputs on the next repair candidate.

    Runtime adapters may add video sampling controls while materializing the
    ComfyUI prompt. Copy those controls only when the runtime node retains the
    Base class. An adapter that changes the class also changes its input
    schema, so copying its inputs onto the Base class would persist an invalid
    hybrid node. The complete runtime graph remains available as diagnosis
    evidence without becoming the next repair candidate.
    """
    if not isinstance(candidate, dict) or not isinstance(runtime_graph, dict):
        return candidate
    runtime_keys = {'frame_load_cap', 'force_rate', 'select_every_nth', 'skip_first_frames', 'custom_width', 'custom_height', 'format', 'loop_count', 'pingpong'}
    merged = copy.deepcopy(candidate)
    for node_id, runtime_node in runtime_graph.items():
        base_node = merged.get(str(node_id))
        if not isinstance(base_node, dict) or not isinstance(runtime_node, dict):
            continue
        if base_node.get('class_type') != runtime_node.get('class_type'):
            continue
        runtime_inputs = runtime_node.get('inputs')
        base_inputs = base_node.setdefault('inputs', {})
        if not isinstance(runtime_inputs, dict) or not isinstance(base_inputs, dict):
            continue
        for key in runtime_keys:
            if key in runtime_inputs:
                base_inputs[key] = copy.deepcopy(runtime_inputs[key])
    return merged

def official_failure_analysis(attempt_row: dict) -> str:
    evaluation = attempt_row.get('comfybench', {})
    result = evaluation.get('comfybench', {}) if isinstance(evaluation, dict) else {}
    if not isinstance(result, dict) or result.get('resolved') is not False:
        return ''
    analysis = result.get('analysis')
    return analysis.strip() if isinstance(analysis, str) else ''

def failure_analysis_required_repair_stage(failure_analysis: dict, states: list[dict], attempt_row: dict | None=None) -> str | None:
    """Return the earliest stage that fresh trace evidence says must change.

    A previous final-stage diagnosis is only reusable while new evidence does
    not identify an upstream leaf or adapter as causal.  Prefer explicit
    ``leaf_change_required`` findings; otherwise use a stage-scoped owner
    finding when its evidence describes the currently observed failure.
    """
    if not isinstance(failure_analysis, dict):
        return None
    stage_order = {state['typed'].stage_id: index for index, state in enumerate(states)}
    diagnosis_record = (attempt_row or {}).get('stage_failure_diagnosis', {})
    diagnosis = diagnosis_record.get('decision', {}) if isinstance(diagnosis_record, dict) else {}
    explicit_stage = str(diagnosis.get('repair_stage_id') or diagnosis.get('earliest_failing_stage_id') or '')
    if explicit_stage in stage_order:
        return explicit_stage
    if str(failure_analysis.get('failure_class', '')).casefold() == 'handoff':
        failed_consumers = [str(stage.get('stage_id')) for stage in (attempt_row or {}).get('stages', []) if isinstance(stage, dict) and str(stage.get('status', '')).casefold() == 'handoff_failed' and (str(stage.get('stage_id')) in stage_order)]
        if failed_consumers:
            return min(failed_consumers, key=stage_order.get)
    candidates: list[tuple[int, str, int]] = []
    for leaf in failure_analysis.get('affected_leaves', []):
        if not isinstance(leaf, dict):
            continue
        stage_id = str(leaf.get('stage_id') or '')
        if stage_id not in stage_order:
            continue
        evidence = leaf.get('evidence')
        has_evidence = isinstance(evidence, list) and any((isinstance(item, str) and item.strip() for item in evidence))
        if not has_evidence:
            continue
        if leaf.get('leaf_change_required') is True:
            priority = 0
        elif str(leaf.get('owner') or '') in {'composition_adapter', 'runner', 'routing_or_decomposition'}:
            priority = 1
        else:
            continue
        candidates.append((priority, stage_order[stage_id], stage_id))
    if not candidates:
        return None
    candidates.sort()
    best_priority = candidates[0][0]
    return min((item for item in candidates if item[0] == best_priority), key=lambda item: item[1])[2]

def enforce_forward_stage_ownership(diagnosis: dict, attempt_row: dict, states: list[dict]) -> dict:
    """Make the ordered stage contract executable and freeze passed artifacts.

    Diagnosis is advisory, but the execution boundary is deterministic: once a
    stage passed its own atomic contract its artifact is frozen.  A later
    quality defect therefore belongs to the first later failing stage unless
    the trace directly marks an earlier stage as failed.
    """
    if not isinstance(diagnosis, dict):
        return diagnosis
    ordered_ids = [state['typed'].stage_id for state in states]
    rows = {str(row.get('stage_id')): row for row in diagnosis.get('stage_assessments', []) if isinstance(row, dict)}
    for stage in attempt_row.get('stages', []):
        gate = stage.get('mask_quality_gate')
        if isinstance(gate, dict) and gate.get('passed') is False:
            row = rows.get(str(stage.get('stage_id')))
            if row is None:
                row = {'stage_id': stage['stage_id']}
                diagnosis.setdefault('stage_assessments', []).append(row)
                rows[str(stage['stage_id'])] = row
            row.update({'status': 'fail', 'reason': 'Mask artifact gate: ' + ', '.join(gate['errors'])})
    first_failed = next((stage_id for stage_id in ordered_ids if rows.get(stage_id, {}).get('status') in {'fail', 'unverifiable'}), None)
    if first_failed is None:
        first_failed = ordered_ids[-1] if ordered_ids else None
    if first_failed is None:
        return diagnosis
    selected = str(diagnosis.get('repair_stage_id') or '')
    selected_status = rows.get(selected, {}).get('status')
    if selected_status == 'pass' or selected != first_failed:
        diagnosis.setdefault('evidence', []).append(f'Forward ownership guard selected {first_failed}; upstream passed artifacts remain frozen.')
        diagnosis['repair_stage_id'] = first_failed
        diagnosis['earliest_failing_stage_id'] = first_failed
        diagnosis['forward_ownership_override'] = {'previous_repair_stage_id': selected or None, 'selected_repair_stage_id': first_failed, 'policy': 'passed_upstream_artifact_frozen'}
    assessments = rows
    for index, stage_id in enumerate(ordered_ids):
        row = assessments.get(stage_id)
        if row is None:
            continue
        stage_row = next((item for item in attempt_row.get('stages', []) if item.get('stage_id') == stage_id), None)
        if stage_row is None:
            continue
        status = str(row.get('status') or 'unverifiable')
        stage_row['quality_passed'] = status == 'pass'
        stage_row['artifact_frozen'] = status == 'pass'
        stage_row['failure_owner'] = None if status == 'pass' else stage_id
        stage_row['retry_reason'] = None if status == 'pass' else str(row.get('reason') or '')
        stage_row['rollback_target'] = None if status == 'pass' else stage_id
    return diagnosis

def reusable_linear_composition_diagnosis(attempt_row: dict, prior_attempt: dict, states: list[dict], search_decision: dict, failure_analysis: dict | None=None) -> dict | None:
    """Reuse proven upstream attribution when only the final linear stage was rerun."""
    if len(states) < 2 or len(attempt_row.get('stages', [])) != len(states):
        return None
    stage_ids = [state['typed'].stage_id for state in states]
    if states[0]['typed'].depends_on or any((tuple(state['typed'].depends_on) != (stage_ids[index - 1],) for index, state in enumerate(states[1:], 1))):
        return None
    if not all((row.get('status') == 'reused' for row in attempt_row['stages'][:-1])):
        return None
    fresh_repair_stage = failure_analysis_required_repair_stage(failure_analysis or {}, states, attempt_row)
    if fresh_repair_stage is not None and fresh_repair_stage != stage_ids[-1]:
        return None
    prior_record = prior_attempt.get('stage_failure_diagnosis', {})
    prior = prior_record.get('decision', {}) if isinstance(prior_record, dict) else {}
    final_id = stage_ids[-1]
    if prior.get('earliest_failing_stage_id') != final_id or prior.get('repair_stage_id') != final_id:
        return None
    prior_assessments = {row.get('stage_id'): row for row in prior.get('stage_assessments', []) if isinstance(row, dict)}
    if any((prior_assessments.get(stage_id, {}).get('status') != 'pass' for stage_id in stage_ids[:-1])):
        return None
    analysis = official_failure_analysis(attempt_row)
    if len(analysis) < 40:
        return None
    assessments = [copy.deepcopy(prior_assessments[stage_id]) for stage_id in stage_ids[:-1]]
    assessments.append({'stage_id': final_id, 'status': 'fail', 'observations': [analysis], 'reason': str(search_decision.get('diagnosis') or analysis)})
    attribution = search_decision.get('conditioning_attribution')
    if not isinstance(attribution, dict):
        attribution = copy.deepcopy(prior.get('conditioning_attribution', {}))
    return {'earliest_failing_stage_id': final_id, 'repair_stage_id': final_id, 'contributing_stage_ids': [], 'stage_assessments': assessments, 'conditioning_attribution': attribution, 'rationale': 'Reused the prior final-stage attribution because every upstream stage was reused unchanged and the fresh official evaluation still reports failure.', 'evidence': [analysis], 'reused_from_attempt': prior_attempt.get('attempt')}

def deterministic_composition_diagnosis(attempt_row: dict, descriptors: list[dict], search_decision: dict, contract_error: Exception) -> dict:
    """Keep an executed quality failure actionable after diagnosis JSON failure."""
    if not descriptors:
        raise AgentOutputError('composition diagnosis fallback requires executed stages')
    failed_index = next((index for index, row in enumerate(descriptors) if (row.get('execution') or {}).get('status') not in {'success', 'reused'}), len(descriptors) - 1)
    failed_id = str(descriptors[failed_index]['stage_id'])
    official = official_failure_analysis(attempt_row)
    assessments = []
    for index, row in enumerate(descriptors):
        if index < failed_index:
            status = 'pass'
            reason = 'Stage execution completed; no earlier stage failure is identified by the retained trace.'
        elif index == failed_index:
            status = 'fail'
            reason = official or str(search_decision.get('diagnosis') or contract_error)
        else:
            status = 'deferred_due_to_upstream_failure'
            reason = 'Deferred after the earliest retained failure.'
        assessments.append({'stage_id': str(row['stage_id']), 'status': status, 'observations': [reason], 'reason': reason})
    unknown = {'likelihood': 'unknown', 'evidence': ['Diagnosis Agent contract failed; attribution is intentionally conservative.'], 'counterfactual': 'A valid stage diagnosis response would refine this attribution.'}
    return {'earliest_failing_stage_id': failed_id, 'repair_stage_id': failed_id, 'contributing_stage_ids': [], 'stage_assessments': assessments, 'conditioning_attribution': {key: copy.deepcopy(unknown) for key in ('text_conditioning', 'source_image_conditioning', 'mask_conditioning', 'negative_conditioning', 'sampler_conditioning')}, 'rationale': 'Deterministic fallback selected the earliest execution failure, or the final transformation stage when all stages executed and only final quality failed.', 'evidence': [official or str(contract_error)], 'fallback_source': 'executed_stage_status_and_official_evaluation', 'diagnosis_contract_error': str(contract_error)}

def legacy_unsubmitted_service_stage(row):
    """Require affirmative evidence before resubmitting a legacy transaction."""
    stages = row.get('stages') or [row]
    failed = [stage for stage in stages if stage.get('execution', {}).get('status') not in {'success', 'reused'}]
    if len(failed) != 1:
        return None
    execution = failed[0].get('execution', {})
    resource = execution.get('resource') or {}
    if execution.get('status') != 'executor_error' or execution.get('prompt_id') or execution.get('outputs') or execution.get('history') or failed[0].get('outputs') or resource.get('samples') or resource.get('memory_sample_prompt_id'):
        return None
    error = execution.get('error') or ''
    refused = '[Errno 111] Connection refused' in error
    reconciled = execution.get('submission_reconciliation') or {}
    if not refused and (not (('Connection reset by peer' in error or 'timed out' in error) and reconciled.get('unsubmitted') is True and reconciled.get('evidence'))):
        return None
    return failed[0]

def legacy_unobserved_submission_stage(row):
    """Identify lost legacy admission responses, without claiming rejection."""
    stages = row.get('stages') or [row]
    failed = [stage for stage in stages if stage.get('execution', {}).get('status') not in {'success', 'reused'}]
    if len(failed) != 1:
        return None
    stage = failed[0]
    execution = stage.get('execution', {})
    resource = execution.get('resource') or {}
    if execution.get('status') == 'executor_error' and (not execution.get('prompt_id')) and (not execution.get('history')) and (not execution.get('outputs')) and (not stage.get('outputs')) and (not resource.get('samples')) and (not resource.get('memory_sample_prompt_id')) and ('Connection reset by peer' in (execution.get('error') or '')):
        return stage
    return None

def recover_unsubmitted_service_failure(report):
    """Exclude every verified transport-only row, retaining genuine iterations.

    An unknown POST result needs reconciliation first. Lack of a prompt ID
    alone is insufficient: old workers also lost IDs during history polling.
    """
    attempts = report.get('attempts', [])
    if not attempts or legacy_unsubmitted_service_stage(attempts[-1]) is None:
        raise ValueError('failure does not prove an unsubmitted connection refusal or reconciled outage')
    removed = [row for row in attempts if legacy_unsubmitted_service_stage(row) is not None or legacy_unobserved_submission_stage(row) is not None]
    retained = [row for row in attempts if row not in removed]
    report.setdefault('legacy_submission_failures', []).extend(copy.deepcopy(removed))
    attempts[:] = retained
    for index, row in enumerate(attempts, 1):
        if row.get('attempt') != index:
            row['legacy_attempt_number'] = row.get('attempt')
            row['attempt'] = index
    for original in removed:
        row = copy.deepcopy(original)
        failed = legacy_unsubmitted_service_stage(row)
        uncertain = failed is None
        if uncertain:
            failed = legacy_unobserved_submission_stage(row)
        failed['execution']['status'] = 'submission_uncertain' if uncertain else 'execution_service_unavailable'
        row['legacy_attempt_number'] = row.get('attempt')
        row['attempt'] = len(attempts) + 1
        record_execution_service_failure(report, row, lambda: None)
    report.pop('repair_terminal', None)
    report.pop('failure_stage', None)
    return len(removed)

def freeze_checkpoint_artifacts(row):
    """Persist complete artifact references for successful and reused stages."""
    import hashlib
    for stage in row.get('stages', []):
        if stage.get('execution', {}).get('status') not in {'success', 'reused'}:
            continue
        artifact = stage.get('primary_output') or next(iter(stage.get('outputs') or []), None)
        if not artifact or not Path(artifact).is_file():
            raise ValueError('completed checkpoint stage has no existing artifact')
        digest = hashlib.sha256(Path(artifact).read_bytes()).hexdigest()
        if stage.get('artifact_sha256') and stage['artifact_sha256'] != digest:
            raise ValueError('completed checkpoint stage artifact changed')
        stage['primary_output'] = str(artifact)
        stage['artifact_sha256'] = digest

def record_execution_service_failure(report, row, checkpoint):
    """Keep an infrastructure checkpoint outside the quality/repair budget."""
    import hashlib
    stages = row.get('stages') or [row]
    failed = next((stage for stage in stages if stage.get('execution', {}).get('status') in {'execution_service_unavailable', 'submission_uncertain', 'server_restarted'}), None)
    if failed is None:
        return False
    saved = copy.deepcopy(row)
    saved.update(passed=None, comfyui_attempt_consumed=False, recorded_at_unix=time.time())
    freeze_checkpoint_artifacts(saved)
    report.setdefault('infrastructure_failures', []).append(saved)
    report['service_checkpoint'] = saved
    report.update(status='submission_uncertain' if failed['execution']['status'] == 'submission_uncertain' else 'execution_service_unavailable', error=failed['execution'].get('error'), completed_attempts=len(report['attempts']))
    checkpoint()
    return True

def runtime_parameter_failure(row):
    """Recognize proven preparation/input errors, never pixel quality failures.

    A composite can have already generated an upstream artifact when a later
    stage fails before submission. Retain that artifact, but do not consume a
    complete attempt for the downstream preflight failure.
    """
    stages = row.get('stages') or [row]
    for stage in stages:
        execution = stage.get('execution', {})
        if execution.get('status') == 'media_budget_failed':
            checks = execution.get('history', {}).get('technical_media_checks', [])
            failed_checks = [item for item in checks if item.get('passed') is False]
            timing_errors = {'video_duration_budget_exceeded', 'video_frame_budget_exceeded'}
            if failed_checks and all((item.get('failures') and set(item['failures']) <= timing_errors for item in failed_checks)):
                return 'video_timing_budget_exceeded'
        if execution.get('prompt_id') or execution.get('history') or execution.get('outputs') or stage.get('outputs') or stage.get('primary_output'):
            continue
        if execution.get('status') == 'preflight_failed':
            return 'stage_preflight_failed'
        if stage.get('status') == 'handoff_failed' and (not execution.get('status')):
            return 'handoff_preflight_failed'
    failed = [stage for stage in stages if stage.get('execution', {}).get('status') == 'error']
    for stage in failed:
        text = json.dumps(stage['execution'], ensure_ascii=False)
        if 'GetImageRangeFromBatch' in text and ('End index is out of range' in text or 'Start index is out of range' in text):
            return 'frame_range_out_of_bounds'
        if 'VHS failed to extract audio' in text and 'Output file does not contain any stream' in text:
            return 'optional_audio_absent'
    return None

def record_runtime_parameter_failure(report, row):
    """Retain the graph, actual errors and completed upstream artifacts for repair."""
    import hashlib
    kind = runtime_parameter_failure(row)
    if kind is None:
        return False
    row.update(comfyui_attempt_consumed=False, logical_iteration_consumed=False, parameter_failure_kind=kind)
    freeze_checkpoint_artifacts(row)
    report.setdefault('parameter_failures', []).append(row)
    report['parameter_checkpoint'] = row
    report.update(status='execution_parameter_failed', completed_attempts=len(report['attempts']))
    return True

def recover_runtime_parameter_failure(report):
    """Migrate only a contiguous suffix of proven parameter errors from old reports."""
    attempts = report.get('attempts', [])
    removed = []
    while attempts and runtime_parameter_failure(attempts[-1]):
        removed.insert(0, attempts.pop())
    if not removed:
        raise ValueError('no verified trailing runtime parameter errors')
    report.setdefault('legacy_parameter_attempts', []).extend(copy.deepcopy(removed))
    for row in removed:
        row['legacy_attempt_number'] = row.get('attempt')
        row['attempt'] = len(attempts) + 1
        record_runtime_parameter_failure(report, row)
    report.pop('repair_terminal', None)
    report.pop('failure_stage', None)
    return len(removed)

def final_failure_status(report: dict) -> str:
    preserved = {'agent_transport_failed', 'agent_provider_failed', 'agent_contract_failed', 'static_preflight_failed', 'composition_diagnosis_contract_failed', 'evaluation_unavailable', 'execution_failed', 'handoff_failed', 'repair_preflight_failed', 'workflow_evolution_failed', 'workflow_unresolved', 'setup_failed', 'workflow_composition_unsupported', 'execution_service_unavailable', 'submission_uncertain', 'execution_parameter_failed', 'parameter_repair_exhausted'}
    if report.get('status') in preserved:
        return str(report['status'])
    attempts = report.get('attempts', [])
    if not attempts:
        return 'quality_not_passed'
    terminal_attempt = attempts[-1]
    terminal_stages = terminal_attempt.get('stages', [])
    if any((stage.get('status') == 'handoff_failed' for stage in terminal_stages)):
        return 'handoff_failed'
    if any((stage.get('repair_failed') for stage in terminal_stages)) or terminal_attempt.get('repair_error'):
        return 'repair_preflight_failed'
    single_execution = terminal_attempt.get('execution', {}).get('status')
    if single_execution not in {None, 'success', 'reused'} or any((stage.get('execution', {}).get('status') not in {None, 'success', 'reused'} for stage in terminal_stages)):
        return 'execution_failed'
    return 'quality_not_passed'

def execute_stage(executor, graph: dict, destination: Path, *, case_id: str, operation: str) -> dict:
    """One shared GPU transaction for single and composed cases.

    The process lock owns preparation, cleanup, submission and collection.
    Cleanup after submission also runs when the executor raises.
    """
    destination.mkdir(parents=True, exist_ok=True)
    notify_pipeline_ready(case_id, operation)
    from comfyrouter.stage_pipeline import transition
    transition('gpu', operation=operation, gpu_ready_at=time.time(), api_status='idle')
    if os.getenv('COMFYROUTER_STAGE_QUEUE_DIR'):
        os.environ['COMFYROUTER_STAGE_QUEUE_DESTINATION'] = str(destination)
        os.environ['COMFYROUTER_STAGE_CASE_ID'] = str(case_id)
        os.environ['COMFYROUTER_STAGE_OPERATION'] = operation
    with comfyui_serial_execution(case_id, operation):
        from tools.vace_material_video import prepare_material_video, restore_material_video
        graph, material_video_context = prepare_material_video(graph, COMFYUI_INPUT)
        cleanup = {'status': 'deferred_to_comfyui_queue'} if comfyui_queue_mode() else release_comfyui_models()
        try:
            context = current_execution()
            if context and context.hooks.before_submit:
                context.hooks.before_submit(graph)
            execution = executor.execute(graph).as_dict()
        finally:
            cleanup_after = {'status': 'deferred_to_comfyui_queue'} if comfyui_queue_mode() else release_comfyui_models()
        copied = []
        if comfyui_queue_mode() and execution.get('history', {}).get('queue_memory_cleanup'):
            server_cleanup = execution['history']['queue_memory_cleanup']
            cleanup = server_cleanup['before']
            cleanup_after = server_cleanup['after']
        for item in execution.get('outputs', []):
            source = executor.output_root / item
            if source.is_file():
                target = destination / source.name
                shutil.copy2(source, target)
                copied.append(target)
        output_roles = {}
        history = execution.get('history') or {}
        submitted = (history.get('prompt') or [None, None, graph])[2]
        for node_id, outputs in history.get('outputs', {}).items():
            node = submitted.get(str(node_id), {})
            if node.get('class_type') != 'SaveImage':
                continue
            role = 'diagnostic' if diagnostic_image_save(submitted, str(node_id)) else 'primary'
            for image in outputs.get('images', []):
                output_roles[Path(image['filename']).name] = role
        (destination / 'output_roles.json').write_text(json.dumps(output_roles, indent=2) + '\n')
        if material_video_context and execution.get('status') == 'success':
            sampled_output = select_primary_output(copied)
            target = destination / 'material_edit_full_timeline.mp4'
            restore_material_video(sampled_output, material_video_context, target)
            (destination / 'material_video_sampling.json').write_text(json.dumps(material_video_context, indent=2) + '\n', encoding='utf-8')
            copied = [target]
    transition('stage_complete', operation=operation, gpu_completed_at=time.time(), prompt_id=execution.get('prompt_id'))
    return {'graph': graph, 'execution': execution, 'outputs': copied, 'pre_execution_cleanup': cleanup, 'post_execution_cleanup': cleanup_after, 'material_video_sampling': material_video_context}

def run_composition(case_id, instruction, states, agent, executor, output_dir, report, checkpoint, max_attempts, case_resources=None, initial_cache=None, start_attempt=1, candidate_validation_only=False, resume_after_evaluation=False, resume_failed_parameters=False):
    context = current_execution()
    if context and context.hooks.composition_observer:
        context.hooks.composition_observer(states)
    case_resources = list(case_resources or [])
    cached_outputs: dict[str, dict] = copy.deepcopy(initial_cache or {})
    stages_by_id = {state['typed'].stage_id: state['typed'] for state in states}

    def invalidate_stage_and_descendants(stage_id: str) -> list[str]:
        invalidated = {stage_id}
        changed = True
        while changed:
            changed = False
            for state in states:
                current_id = state['typed'].stage_id
                if current_id not in invalidated and any((dependency in invalidated for dependency in state['typed'].depends_on)):
                    invalidated.add(current_id)
                    changed = True
        for current_id in invalidated:
            cached_outputs.pop(current_id, None)
        return sorted(invalidated)
    resume_pending = resume_after_evaluation or resume_failed_parameters
    if resume_after_evaluation and (not report.get('attempts') or len(report['attempts']) >= max_attempts or report['attempts'][-1].get('passed') is not False):
        raise ValueError('composition continuation requires False and remaining attempts')
    evolution_attempted = False
    attempt = len(report['attempts']) if resume_after_evaluation else start_attempt
    while resume_pending or attempt <= max_attempts:
        attempt_dir = output_dir / f'attempt_{attempt:02d}'
        if report.get('parameter_failures'):
            attempt_dir = output_dir / f'attempt_{attempt:02d}_parameter_repair_{len(report['parameter_failures']):02d}'
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if resume_pending:
            resume_pending = False
            attempt_row = report['parameter_checkpoint'] if resume_failed_parameters else report['attempts'][-1]
            resume_failed_parameters = False
            all_stages_succeeded = all((s.get('execution', {}).get('status') in {'success', 'reused'} for s in attempt_row.get('stages', [])))
        else:
            attempt_row = {'attempt': attempt, 'stages': [], 'attempt_dir': str(attempt_dir)}
            stage_outputs: dict[str, Path] = {}
            all_stages_succeeded = True
            for index, state in enumerate(states, 1):
                stage_id = state['typed'].stage_id
                cached = cached_outputs.get(stage_id)
                if cached is not None and Path(cached['output']).is_file():
                    stage_outputs[stage_id] = Path(cached['output'])
                    attempt_row['stages'].append({'stage_id': stage_id, 'operation': state['implementation']['operation'], 'graph': copy.deepcopy(state['graph']), 'handoff': cached.get('handoff'), 'status': 'reused', 'reused_from_attempt': cached['attempt'], 'execution': {'status': 'reused', 'outputs': [Path(cached['output']).name]}, 'outputs': [cached['output']], 'primary_output': cached['output'], 'execution_passed': True, 'quality_passed': True, 'artifact_frozen': True, 'failure_owner': None, 'retry_reason': None, 'rollback_target': None})
                    continue
                graph = copy.deepcopy(state['graph'])
                handoff = None
                dependency_sources = [(dependency, stage_outputs[dependency], stages_by_id[dependency].output_type) for dependency in state['typed'].depends_on if dependency in stage_outputs]
                if len(dependency_sources) != len(state['typed'].depends_on):
                    all_stages_succeeded = False
                    missing = sorted(set(state['typed'].depends_on) - set(stage_outputs))
                    attempt_row['stages'].append({'stage_id': state['typed'].stage_id, 'status': 'handoff_failed', 'error': 'missing dependency outputs: ' + ', '.join(missing)})
                    break
                if dependency_sources:
                    bound = []
                    used_by_type: dict[str, int] = {}
                    binding_error = None
                    for dependency_index, (dependency, source, source_type) in enumerate(dependency_sources, 1):
                        input_groups = handoff_input_groups(graph, source_type, state['handoff_inputs_by_type'].get(source_type, []))
                        input_index = used_by_type.get(source_type, 0)
                        if input_index >= len(input_groups):
                            binding_error = f'downstream Base has no free {source_type} input for dependency {dependency}'
                            break
                        used_by_type[source_type] = input_index + 1
                        suffix = source.suffix or '.png'
                        handoff_name = f'ComfyRouter_{report['run_id']}_a{attempt:02d}_to_s{index:02d}_input{dependency_index:02d}{suffix}'
                        handoff_path = COMFYUI_INPUT / handoff_name
                        shutil.copy2(source, handoff_path)
                        for node_id, input_name in input_groups[input_index]:
                            graph[node_id].setdefault('inputs', {})[input_name] = handoff_name
                            bound.append({'source': str(source), 'input': str(handoff_path), 'node_id': node_id, 'input_name': input_name, 'media_type': source_type, 'dependency': dependency})
                    if binding_error:
                        all_stages_succeeded = False
                        attempt_row['stages'].append({'stage_id': state['typed'].stage_id, 'status': 'handoff_failed', 'error': binding_error})
                        break
                    handoff = {'dependencies': list(state['typed'].depends_on), 'bindings': bound}
                    executor.validator = None
                try:
                    external_bindings = bind_external_composition_inputs(graph, state['typed'], case_resources, used_by_type if dependency_sources else {})
                    if external_bindings:
                        if handoff is None:
                            handoff = {'bindings': [], 'dependencies': []}
                        handoff['bindings'].extend(external_bindings)
                        executor.validator = None
                except ClosedLoopError as exc:
                    all_stages_succeeded = False
                    attempt_row['stages'].append({'stage_id': stage_id, 'status': 'handoff_failed', 'error': str(exc), 'graph': copy.deepcopy(graph), 'handoff': handoff})
                    break
                if handoff is not None:
                    try:
                        if 'input' in handoff:
                            require_runtime_resources_bound(graph, [Path(handoff['input'])], state['typed'].input_type)
                        else:
                            for media_type in dict.fromkeys((item.get('media_type', state['typed'].input_type) for item in handoff.get('bindings', []))):
                                typed_sources = [Path(item['input']) for item in handoff.get('bindings', []) if item.get('media_type', state['typed'].input_type) == media_type]
                                require_runtime_resources_bound(graph, typed_sources, media_type)
                        handoff['input_validation'] = inspect_runtime_media_inputs(graph, [item.get('media_type', state['typed'].input_type) for item in handoff.get('bindings', [])])
                    except ClosedLoopError as exc:
                        all_stages_succeeded = False
                        attempt_row['stages'].append({'stage_id': stage_id, 'status': 'handoff_failed', 'error': str(exc), 'graph': copy.deepcopy(graph), 'handoff': handoff})
                        break
                runtime_image_source = None
                if isinstance(handoff, dict) and handoff.get('kind') == 'case_resource':
                    runtime_image_source = Path(handoff['input'])
                elif isinstance(handoff, dict):
                    image_binding = next((item for item in handoff.get('bindings', []) if item.get('media_type') == 'IMAGE'), None)
                    if image_binding:
                        runtime_image_source = Path(image_binding['input'])
                if state.get('deferred_grounded_mask'):
                    try:
                        if runtime_image_source is None or not runtime_image_source.is_file():
                            raise AgentOutputError('grounded mask planning requires the materialized upstream IMAGE')
                        from comfyrouter.mask_fallback import estimate_placement_plan
                        plan = estimate_placement_plan(agent, executor, graph, [state['implementation']], runtime_image_source, state['typed'].instruction, {'setup_failure': 'Prospective mask planning deferred until actual source pixels exist.', 'overall_instruction': instruction, 'prior_mask_feedback': [{'attempt': prior.get('attempt'), 'error': row.get('error')} for prior in [*report.get('attempts', []), *report.get('parameter_failures', [])] for row in prior.get('stages', []) if row.get('stage_id') == stage_id and row.get('error')]})
                        graph = apply_graph_edits(graph, plan['edits'], executor.validator, validate_after_each=False)
                        require_runtime_resources_bound(graph, [runtime_image_source], 'IMAGE')
                        contract = build_handoff_contract_context(state['typed'], [s['typed'] for s in states], [s['implementation'] for s in states], graph=graph)
                        validate_stage_handoff_contract(graph, contract, case_resources=case_resources, validator=executor.validator)
                        executor.static_validate(graph)
                        state['graph'] = copy.deepcopy(graph)
                        record = report['composition']['stages'][index - 1]
                        record.setdefault('runtime_grounded_mask_plans', []).append({'attempt': attempt, 'source': str(runtime_image_source), 'plan': plan})
                        record['preflight_graph'] = copy.deepcopy(graph)
                        checkpoint()
                    except ComfyUITransportError:
                        raise
                    except (AgentOutputError, StaticValidationError, ClosedLoopError) as exc:
                        all_stages_succeeded = False
                        attempt_row['stages'].append({'stage_id': stage_id, 'status': 'execution_failed', 'execution': {'status': 'preflight_failed', 'error': str(exc)}, 'error': str(exc), 'graph': copy.deepcopy(graph), 'handoff': handoff})
                        break
                geometry_normalization = normalize_exact_image_geometry(graph, runtime_image_source, state['typed'].instruction, overall_instruction=instruction, stage_operation=state['implementation']['operation'], mask_source=next((Path(item['input']) for item in (handoff or {}).get('bindings', []) if item.get('media_type') == 'MASK'), None))
                transaction = execute_stage(executor, graph, attempt_dir / f'stage_{index:02d}', case_id=case_id, operation=f'composition_stage_{index:02d}_attempt_{attempt:02d}')
                graph = transaction['graph']
                execution = transaction['execution']
                copied = transaction['outputs']
                cleanup = transaction['pre_execution_cleanup']
                cleanup_after = transaction['post_execution_cleanup']
                material_video_context = transaction['material_video_sampling']
                primary_output = select_primary_output(copied)
                _, diagnostic_outputs = split_primary_and_diagnostic_outputs(copied)
                geometry_validation = exact_image_output_validation(primary_output, geometry_normalization)
                stage_row = {'stage_id': state['typed'].stage_id, 'operation': state['implementation']['operation'], 'graph': graph, 'runtime_graph': None, 'handoff': handoff, 'pre_execution_cleanup': cleanup, 'post_execution_cleanup': cleanup_after, 'execution': execution, 'outputs': [str(path) for path in copied], 'primary_output': str(primary_output) if primary_output else None, 'diagnostic_outputs': [str(path) for path in diagnostic_outputs], 'geometry_normalization': geometry_normalization, 'geometry_validation': geometry_validation, 'material_video_sampling': material_video_context, 'execution_passed': execution.get('status') == 'success' and primary_output is not None and (not (geometry_validation and (not geometry_validation['passed']))), 'quality_passed': None, 'artifact_frozen': False, 'failure_owner': None, 'retry_reason': None, 'rollback_target': None}
                history_prompt = execution.get('history', {}).get('prompt') if isinstance(execution.get('history'), dict) else None
                if isinstance(history_prompt, list) and len(history_prompt) >= 3 and isinstance(history_prompt[2], dict):
                    stage_row['runtime_graph'] = copy.deepcopy(history_prompt[2])
                attempt_row['stages'].append(stage_row)
                if record_execution_service_failure(report, attempt_row, checkpoint):
                    return
                if not stage_row['execution_passed']:
                    all_stages_succeeded = False
                    break
                if state['typed'].output_type == 'MASK' and primary_output.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                    from comfyrouter.mask_fallback import inspect_mask
                    gate = inspect_mask(primary_output, runtime_image_source)
                    stage_row['mask_quality_gate'] = gate
                    if not gate['passed']:
                        stage_row.update({'status': 'mask_quality_failed', 'quality_passed': False, 'error': 'mask_quality_failed: ' + ', '.join(gate['errors'])})
                        all_stages_succeeded = False
                        break
                stage_outputs[state['typed'].stage_id] = primary_output
                cached_outputs[state['typed'].stage_id] = {'output': str(primary_output), 'attempt': attempt, 'handoff': handoff}
            final_output = stage_outputs.get(states[-1]['typed'].stage_id) if states else None
            if all_stages_succeeded and final_output is not None:
                attempt_row['outputs'] = [str(final_output)]
                notify_evaluation_context(attempt_row, final_output, attempt_dir, {state['typed'].stage_id: state['typed'].output_type for state in states})
                attempt_row['comfybench'] = official_check(case_id, final_output, attempt_dir)
                if attempt_row['comfybench'].get('status') == 'evaluator_error':
                    attempt_row['passed'] = None
                    attempt_row['evaluation_status'] = 'unavailable'
                    report['attempts'].append(attempt_row)
                    report['status'] = 'evaluation_unavailable'
                    report['error'] = 'official ComfyBench evaluation failed after three retries'
                    checkpoint()
                    break
                official_passed = bool(attempt_row['comfybench'].get('comfybench', {}).get('resolved') is True)
                attempt_row['official_passed'] = official_passed
                attempt_row['passed'] = official_passed
            else:
                attempt_row['outputs'] = []
                attempt_row['comfybench'] = {'status': 'stage_execution_failed', 'passed': False}
                attempt_row['passed'] = False
            if not record_runtime_parameter_failure(report, attempt_row):
                report['attempts'].append(attempt_row)
                report.pop('parameter_checkpoint', None)
                if report.get('status') == 'execution_parameter_failed':
                    report['status'] = 'running'
            checkpoint()
        parameter_failed = attempt_row.get('comfyui_attempt_consumed') is False and bool(attempt_row.get('parameter_failure_kind'))
        if attempt_row['passed']:
            for completed_stage in attempt_row.get('stages', []):
                if completed_stage.get('execution_passed') or completed_stage.get('status') == 'reused':
                    if completed_stage.get('quality_passed') is not False:
                        completed_stage['quality_passed'] = True
                    completed_stage['artifact_frozen'] = True
                    completed_stage['failure_owner'] = None
                    completed_stage['retry_reason'] = None
                    completed_stage['rollback_target'] = None
            checkpoint()
            break
        if candidate_validation_only:
            report['status'] = 'candidate_composition_quality_failed' if all_stages_succeeded else 'candidate_composition_execution_failed'
            report['candidate_validation_only'] = True
            checkpoint()
            break
        if len(report['attempts']) >= max_attempts and True:
            attempt_row['terminal_diagnosis'] = 'skipped_no_remaining_execution_budget'
            checkpoint()
            break
        descriptors, diagnosis_paths, visual_manifest = composition_diagnosis_media(attempt_row, states[:len(attempt_row['stages'])], attempt_dir, case_resources, prior_attempts=report.get('attempts', []) if parameter_failed else report.get('attempts', [])[:-1])
        fallback_search_decision = {'diagnosis': 'Combined visual diagnosis output was invalid.', 'conditioning_attribution': {}, 'repair_focus': [], 'needs_search': False, 'search_queries': [], 'evidence': []}
        try:
            diagnosis = agent.diagnose_composition_failure(instruction=instruction, stages=descriptors, final_evaluation={'comfybench': attempt_row['comfybench']}, media_paths=diagnosis_paths, visual_manifest=visual_manifest, include_repair_decision=True)
            AgentOrchestrator._validate_independent_repair_evidence(diagnosis, descriptors)
            diagnosis_status = 'selected_combined_visual_diagnosis'
        except (AgentTransportError, AgentProviderError) as exc:
            attempt_row['stage_failure_diagnosis'] = {'status': agent_request_failure_status(exc), 'error': str(exc), 'stages': descriptors}
            report['status'] = agent_request_failure_status(exc)
            report['failure_stage'] = 'composition_failure_diagnosis'
            report['logical_iteration_consumed'] = False
            report['comfyui_attempt_consumed'] = False
            checkpoint()
            break
        except AgentOutputError as exc:
            diagnosis = deterministic_composition_diagnosis(attempt_row, descriptors, fallback_search_decision, exc)
            diagnosis.update({'failure_class': 'execution' if not all_stages_succeeded else 'quality', 'root_causes': [diagnosis['rationale']], 'next_action': 'parameter_retry', 'repair_focus': [diagnosis['rationale']], 'needs_search': False, 'search_queries': []})
            diagnosis_status = 'deterministic_contract_fallback'
        diagnosis = enforce_forward_stage_ownership(diagnosis, attempt_row, states)
        failure_analysis = combined_failure_analysis(diagnosis)
        report.setdefault('failure_analyses', []).append({'attempt': attempt, 'analysis': failure_analysis})
        attempt_row['failure_analysis'] = failure_analysis
        search_decision = combined_search_decision(diagnosis)
        search_evidence = run_anysearch(search_decision['search_queries']) if search_decision['needs_search'] else []
        attempt_row['repair_search_decision'] = search_decision
        attempt_row['anysearch_evidence'] = search_evidence
        repair_evidence = {'composition_attempt': attempt_row, 'comfybench': attempt_row['comfybench'], 'repair_search_decision': search_decision, 'anysearch_evidence': search_evidence, 'attempt_history': report['attempts'], 'parameter_failure_history': report.get('parameter_failures', []), 'instruction': instruction, 'policy': 'Repair only parameters or prompts in this stage; preserve every Base topology.'}
        analysis_repair_stage = failure_analysis_required_repair_stage(attempt_row.get('failure_analysis', {}), states, attempt_row)
        if analysis_repair_stage is not None and diagnosis.get('repair_stage_id') != analysis_repair_stage:
            prior_stage = diagnosis.get('repair_stage_id')
            diagnosis['repair_stage_id'] = analysis_repair_stage
            diagnosis['earliest_failing_stage_id'] = analysis_repair_stage
            diagnosis.setdefault('evidence', []).append(f'Fresh complete-trace failure analysis superseded the prior/reused repair stage {prior_stage!r} and selected the earliest evidence-backed stage {analysis_repair_stage!r}.')
            diagnosis['failure_analysis_stage_override'] = {'previous_repair_stage_id': prior_stage, 'selected_repair_stage_id': analysis_repair_stage, 'policy': 'fresh_trace_analysis_supersedes_reused_downstream_attribution'}
            diagnosis_status = 'fresh_failure_analysis_override'
        selected_id = diagnosis['repair_stage_id']
        earliest_failure_id = diagnosis['earliest_failing_stage_id']
        selected_index = next((index for index, state in enumerate(states) if state['typed'].stage_id == selected_id))
        attempt_row['stage_failure_diagnosis'] = {'status': diagnosis_status, 'decision': diagnosis, 'selected_stage_index': selected_index + 1, 'earliest_failing_stage_id': earliest_failure_id, 'repair_stage_id': selected_id, 'visual_input_files': diagnosis_paths, 'visual_input_order': visual_manifest}
        repair_targets = composition_repair_targets(diagnosis, states, descriptors)
        repair_target_ids = [target['stage_id'] for target in repair_targets]
        attempt_row['stage_failure_diagnosis']['repair_stage_ids'] = repair_target_ids
        for index, stage_row in enumerate(attempt_row['stages']):
            stage_row['repair_decision'] = 'selected_causal_repair_stage' if index == selected_index else 'independent_evidence_repair_stage' if stage_row['stage_id'] in repair_target_ids else 'unchanged_by_forward_diagnosis'
        state = states[selected_index]
        selected_stage_row = attempt_row.get('stages', [])[selected_index] if selected_index < len(attempt_row.get('stages', [])) else {}
        runtime_graph = selected_stage_row.get('runtime_graph')
        if isinstance(runtime_graph, dict):
            state['graph'] = merge_runtime_repair_inputs(state['graph'], runtime_graph)
        selected_assessment = next((row for row in diagnosis['stage_assessments'] if row['stage_id'] == selected_id))
        evolution_context = repair_mode_for_context(failures=[], operation=str(state['implementation'].get('operation', '')), input_types=stage_input_types(state['typed']), diagnosis={'stage_failure_diagnosis': diagnosis, 'failure_analysis': attempt_row.get('failure_analysis', {})}, execution_failed=True)
        composition_parameter_only = bool(isinstance(report.get('repair_policy'), dict) and report['repair_policy'].get('parameter_only') is True)
        if composition_parameter_only:
            evolution_context['mode'] = 'parameter'
            evolution_context['allow_topology_repair'] = False
            evolution_context['requires_evolution'] = False
        if len(report['attempts']) >= max_attempts:
            break
        if parameter_failed and len(report.get('parameter_failures', [])) >= repair_plan_limit():
            report['status'] = 'parameter_repair_exhausted'
            checkpoint()
            break
        primary_repair_id = selected_id
        batch_snapshots = [{key: copy.deepcopy(state.get(key)) for key in ('graph', 'handoff_inputs', 'handoff_inputs_by_type')} for state in states]
        composition_snapshot = copy.deepcopy(report.get('composition'))
        cache_snapshot = copy.deepcopy(cached_outputs)
        validated_target_ids = []
        invalidated_ids = set()
        attempt_row['repair_batch'] = {'primary_stage_id': primary_repair_id, 'requested_stage_ids': repair_target_ids, 'validated_stage_ids': validated_target_ids, 'status': 'planning', 'execution_budget_increment': 0}
        for target in repair_targets:
            selected_id = target['stage_id']
            selected_index = next((index for index, item in enumerate(states) if item['typed'].stage_id == selected_id))
            state = states[selected_index]
            selected_assessment = target['assessment']
            target_diagnosis = target['diagnosis']
            target_failure_analysis = combined_failure_analysis(target_diagnosis)
            runtime_graph = attempt_row['stages'][selected_index].get('runtime_graph')
            if isinstance(runtime_graph, dict):
                state['graph'] = merge_runtime_repair_inputs(state['graph'], runtime_graph)
            repaired = False
            repair_records = []
            repair_failures = []
            previous_repair_plan = None
            for repair_try in range(1, repair_plan_limit() + 1):
                try:
                    repair_mode = repair_mode_for_context(failures=repair_failures, operation=str(state['implementation'].get('operation', '')), input_types=stage_input_types(state['typed']), diagnosis={'stage_failure_diagnosis': target_diagnosis, 'failure_analysis': target_failure_analysis}, retry_index=attempt, execution_failed=True)
                    if composition_parameter_only:
                        repair_mode['mode'] = 'parameter'
                        repair_mode['allow_topology_repair'] = False
                        repair_mode['requires_evolution'] = False
                    stage_repair_evidence = {**repair_evidence, 'forward_stage_diagnosis': target_diagnosis, 'batch_repair_stage_ids': repair_target_ids, 'independent_repair': target_diagnosis.get('independent_repair'), 'failure_analysis': target_failure_analysis, 'repair_focus': target_diagnosis.get('repair_focus', []), 'selected_stage_assessment': selected_assessment, 'selected_stage_result_files': descriptors[selected_index]['output_files'], 'preflight_failures': copy.deepcopy(repair_failures), 'allowed_enums': [enum for failure in repair_failures for enum in failure.get('allowed_enums', [])], 'previous_plan': copy.deepcopy(previous_repair_plan), 'repair_mode': repair_mode, 'policy': 'Modify only repair_stage_id. Use existing node parameters only.' if repair_mode['mode'] == 'parameter' else 'Modify only the current repair_stage_id using the smallest statically validated topology edits justified by its evidence.'}
                    stage_repair_evidence['previous_plan'], stage_repair_evidence['rejected_previous_edits'] = sanitize_previous_plan_for_graph(previous_repair_plan, state['graph'], executor.validator)
                    stage_repair_evidence['current_graph_edit_contract'] = current_graph_edit_contract(state['graph'], executor.validator)
                    if repair_mode.get('allow_topology_repair'):
                        stage_repair_evidence['local_registry_candidates'] = local_registry_repair_candidates(executor, {'failure_analysis': target_failure_analysis, 'stage_failure_diagnosis': target_diagnosis, 'repair_focus': selected_assessment})
                        stage_repair_evidence['local_registry_policy'] = 'These are installed live /object_info candidates. Assess their exact typed interfaces and select the smallest executable repair; do not claim the capability is unavailable without considering this shortlist.'
                    stage_repair_evidence['repair_visual_inputs'] = visual_manifest
                    handoff_contract = build_handoff_contract_context(state['typed'], [item['typed'] for item in states], [item['implementation'] for item in states], graph=state['graph'])
                    plan = generate_plan_with_repair(agent, executor, state['spec'], [state['implementation']], graph=state['graph'], instruction=state['typed'].instruction, overall_instruction=instruction, handoff_geometry_context=state.get('handoff_geometry_context'), handoff_contract_context=handoff_contract, repair_evidence=stage_repair_evidence, parameter_only=repair_mode['mode'] == 'parameter', allow_topology_repair=repair_mode.get('allow_topology_repair', False))
                    previous_repair_plan = plan
                    if repair_mode.get('allow_topology_repair') or plan.get('repair_transaction'):
                        require_controlled_topology_repair(plan, require_edit=True)
                    else:
                        require_parameter_only(plan, require_edit=True)
                    if plan['requires_composition_replan']:
                        raise AgentOutputError('composition replan requested without an executable replacement stage graph')
                    repair_base = repair_plan_base_graph(plan, state.get('base_graph', state['graph']), state['graph'])
                    plan_base, selection = graph_selected_by_execution_agent(plan, repair_base, executor)
                    if selection:
                        plan['execution_graph_selection'] = selection
                    candidate = apply_graph_edits(plan_base, plan['edits'], executor.validator, validate_after_each=not (repair_mode.get('allow_topology_repair', False) or plan.get('repair_transaction')), allow_external_placeholders=True)
                    temporal_instruction = state['typed'].instruction
                    if selected_index == len(states) - 1 and requested_duration_seconds(temporal_instruction) is None:
                        temporal_instruction = instruction
                    timing_normalizations = normalize_quantized_video_timing(candidate, temporal_instruction)
                    if timing_normalizations:
                        plan['deterministic_timing_normalizations'] = timing_normalizations
                    upscale_normalizations = normalize_upscale_model(candidate, temporal_instruction)
                    if upscale_normalizations:
                        plan['deterministic_upscale_normalizations'] = upscale_normalizations
                    repaired_handoff_inputs, repaired_primary_inputs = bind_composition_preflight_inputs(candidate, state['typed'], case_resources, preserve_upstream_mask_placeholder=any((str(next((producer['typed'].output_type for producer in states if producer['typed'].stage_id == dependency), '')).upper() == 'MASK' for dependency in state['typed'].depends_on)))
                    validate_no_new_dead_nodes(plan_base, candidate, executor.validator)
                    for media_type, existing_inputs in state['handoff_inputs_by_type'].items():
                        if not repaired_handoff_inputs.get(media_type):
                            repaired_handoff_inputs[media_type] = existing_inputs
                    repaired_primary_inputs = repaired_handoff_inputs.get(state['typed'].input_type, repaired_primary_inputs)
                    validate_stage_handoff_contract(candidate, handoff_contract, case_resources=case_resources, validator=executor.validator)
                    executor.static_validate(candidate, allow_external_placeholders=bool(deferred_stage_handoff_types(state['typed'], [item['typed'] for item in states])))
                    validate_temporal_parameters(candidate, temporal_instruction)
                    state['graph'] = candidate
                    state['handoff_inputs_by_type'] = repaired_handoff_inputs
                    state['handoff_inputs'] = repaired_primary_inputs
                    composition_rows = report.get('composition', {}).get('stages', [])
                    if selected_index < len(composition_rows):
                        composition_stage = composition_rows[selected_index]
                        composition_stage['preflight_graph'] = copy.deepcopy(candidate)
                        composition_stage['handoff_inputs'] = [{'node_id': node_id, 'input': input_name} for node_id, input_name in repaired_primary_inputs]
                        composition_stage['handoff_inputs_by_type'] = {media_type: [{'node_id': node_id, 'input': input_name} for node_id, input_name in nodes] for media_type, nodes in repaired_handoff_inputs.items()}
                    repair_records.append({'try': repair_try, 'repair_mode': repair_mode, 'plan': plan, 'status': 'validated'})
                    invalidated_ids.update(invalidate_stage_and_descendants(selected_id))
                    attempt_row['cache_invalidation'] = {'selected_stage_id': primary_repair_id, 'selected_stage_ids': list(validated_target_ids) + [selected_id], 'invalidated_stage_ids': sorted(invalidated_ids), 'policy': 'Reuse successful unaffected DAG stages; rerun the repaired stage and its descendants only.'}
                    repaired = True
                    break
                except (AgentTransportError, AgentProviderError) as exc:
                    failure = preflight_failure_record(exc, 'composition_repair_transport', previous_repair_plan)
                    repair_records.append({'try': repair_try, 'repair_mode': repair_mode, **failure})
                    report['status'] = agent_request_failure_status(exc)
                    report['failure_stage'] = 'composition_repair_transport'
                    report['error'] = str(exc)
                    report['logical_iteration_consumed'] = False
                    report['comfyui_attempt_consumed'] = False
                    break
                except ComfyUITransportError:
                    raise
                except (AgentOutputError, StaticValidationError, ClosedLoopError) as exc:
                    failure = preflight_failure_record(exc, 'composition_repair_preflight', previous_repair_plan)
                    repair_failures.append(failure)
                    repair_records.append({'try': repair_try, 'repair_mode': repair_mode, **failure})
            attempt_row['stages'][selected_index]['repair_generation_plans'] = repair_records
            if report.get('status') in {'agent_transport_failed', 'agent_provider_failed'}:
                break
            if not repaired:
                attempt_row['stages'][selected_index]['repair_failed'] = True
                report['status'] = 'repair_preflight_failed'
                report['repair_terminal'] = {'reason': 'planning_budget_exhausted', 'planning_limit': repair_plan_limit(), 'stage_id': selected_id, 'last_failure': repair_failures[-1] if repair_failures else None}
                report['failure_stage'] = selected_id
                break
            validated_target_ids.append(selected_id)
        if len(validated_target_ids) != len(repair_targets):
            for state, snapshot in zip(states, batch_snapshots):
                state.update(snapshot)
            if composition_snapshot is not None:
                report['composition'] = composition_snapshot
            cached_outputs.clear()
            cached_outputs.update(cache_snapshot)
            attempt_row['repair_batch']['status'] = 'rejected_before_execution'
            attempt_row.pop('cache_invalidation', None)
            checkpoint()
            break
        attempt_row['repair_batch']['status'] = 'validated_for_next_attempt'
        checkpoint()
        if not parameter_failed:
            attempt += 1
    report['final_graphs'] = [state['graph'] for state in states]
    terminal_status = final_failure_status(report)
    report['status'] = 'passed' if any((row.get('passed') is True for row in report['attempts'])) else terminal_status if terminal_status in {'agent_transport_failed', 'agent_provider_failed', 'agent_contract_failed', 'static_preflight_failed', 'composition_diagnosis_contract_failed', 'evaluation_unavailable', 'execution_failed', 'handoff_failed', 'repair_preflight_failed', 'workflow_evolution_failed', 'setup_failed', 'workflow_composition_unsupported'} else 'workflow_unresolved' if evolution_attempted else terminal_status
    if report['status'] == 'passed':
        passed_attempt = next((row for row in report['attempts'] if row.get('passed')), None)
        report['success_class'] = 'evolved_workflow_passed' if evolution_attempted else 'first_pass' if passed_attempt and passed_attempt.get('attempt') == 1 else 'repair_passed'
    report['completed_attempts'] = len(report['attempts'])
    checkpoint()

def run_single_case(case_id, instruction, modality, spec, implementation, base, current, source, media_type, handoff_geometry_context, runtime_param_edits, agent, executor, output_dir, report, checkpoint, max_attempts, case_resources, resume_after_evaluation=False, resume_failed_parameters=False):
    """Execute the shared single-leaf repair loop, optionally from evaluated media."""
    if resume_after_evaluation:
        attempts = report.get('attempts', [])
        if not attempts or len(attempts) >= max_attempts or attempts[-1].get('passed') is not False or (not attempts[-1].get('comfybench')) or (attempts[-1]['comfybench'].get('status') == 'evaluator_error'):
            raise ValueError('resume requires a completed False evaluation and remaining attempts')
    attempt = len(report['attempts'])
    resume_pending = resume_after_evaluation or resume_failed_parameters
    execution_limit = max_attempts
    evolution_attempted = False
    while resume_pending or attempt < execution_limit:
        if resume_pending:
            resume_pending = False
            row = report['parameter_checkpoint'] if resume_failed_parameters else report['attempts'][-1]
            resume_failed_parameters = False
            attempt = row['attempt']
            execution = row['execution']
            copied = [Path(path) for path in row.get('outputs', [])]
            primary_output = Path(row['primary_output']) if row.get('primary_output') else None
            diagnostic_outputs = [Path(path) for path in row.get('diagnostic_outputs', [])]
        else:
            attempt = len(report['attempts']) + 1
            attempt_dir = output_dir / f'attempt_{attempt:02d}'
            if report.get('parameter_failures'):
                attempt_dir = output_dir / f'attempt_{attempt:02d}_parameter_repair_{len(report['parameter_failures']):02d}'
            attempt_dir.mkdir(parents=True, exist_ok=True)
            execution_graph = copy.deepcopy(current)
            try:
                input_validation = prepare_single_media_inputs(execution_graph, implementation, case_resources, media_type, modality, validator=executor.validator)
            except (AgentOutputError, ClosedLoopError) as exc:
                report.update(status='handoff_failed', error=str(exc))
                report['input_preflight_failure'] = {'phase': 'single_stage_materialized_input_validation', 'error': str(exc), 'comfyui_attempt_consumed': False, 'completed_attempts': len(report['attempts'])}
                checkpoint()
                return
            geometry_normalization = normalize_exact_image_geometry(execution_graph, source if media_type == 'IMAGE' else None, instruction)
            transaction = execute_stage(executor, execution_graph, attempt_dir, case_id=case_id, operation=f'single_stage_attempt_{attempt:02d}')
            execution_graph = transaction['graph']
            execution = transaction['execution']
            copied = transaction['outputs']
            cleanup = transaction['pre_execution_cleanup']
            cleanup_after = transaction['post_execution_cleanup']
            material_video_context = transaction['material_video_sampling']
            primary_output = select_primary_output(copied)
            _, diagnostic_outputs = split_primary_and_diagnostic_outputs(copied)
            geometry_validation = exact_image_output_validation(primary_output, geometry_normalization)
            row = {'attempt': attempt, 'graph': execution_graph, 'execution': execution, 'input_validation': input_validation, 'material_video_sampling': material_video_context, 'pre_execution_cleanup': cleanup, 'post_execution_cleanup': cleanup_after, 'outputs': [str(x) for x in copied], 'primary_output': str(primary_output) if primary_output else None, 'diagnostic_outputs': [str(x) for x in diagnostic_outputs], 'geometry_normalization': geometry_normalization, 'geometry_validation': geometry_validation}
            if record_execution_service_failure(report, row, checkpoint):
                return
            if geometry_validation and (not geometry_validation['passed']):
                row['comfybench'] = {'status': 'exact_geometry_failed', 'passed': False, 'error': geometry_validation['error']}
                row['passed'] = False
            elif execution.get('status') not in {'success', 'reused'}:
                row['comfybench'] = {'status': 'stage_execution_failed', 'passed': False, 'error': execution.get('error')}
                row['passed'] = False
            elif primary_output is not None:
                notify_evaluation_context(row, primary_output, attempt_dir)
                row['comfybench'] = official_check(case_id, primary_output, attempt_dir)
                if row['comfybench'].get('status') == 'evaluator_error':
                    row['evaluation_status'] = 'unavailable'
                    row['passed'] = None
                    report['attempts'].append(row)
                    report['status'] = 'evaluation_unavailable'
                    report['error'] = 'official ComfyBench evaluation failed after three retries'
                    checkpoint()
                    break
                cb = row['comfybench'].get('comfybench', {})
                row['passed'] = bool(cb.get('resolved') is True)
            else:
                row['comfybench'] = {'status': 'no_output', 'passed': False}
                row['passed'] = False
            if not record_runtime_parameter_failure(report, row):
                report['attempts'].append(row)
                report.pop('parameter_checkpoint', None)
                if report.get('status') == 'execution_parameter_failed':
                    report['status'] = 'running'
            checkpoint()
        parameter_failed = row.get('comfyui_attempt_consumed') is False and bool(row.get('parameter_failure_kind'))
        if row['passed']:
            break
        if not parameter_failed and attempt >= execution_limit and True:
            row['terminal_diagnosis'] = 'skipped_no_remaining_execution_budget'
            checkpoint()
            break
        diagnosis_paths, visual_manifest = single_failure_visual_inputs(case_resources, primary_output, diagnostic_outputs)
        search_decision = repair_search_with_fallback(lambda: agent.decide_repair_search(instruction=instruction, graph=current, execution=execution, official_evaluation=row['comfybench'], attempt_history=report['attempts'], media_paths=diagnosis_paths, visual_manifest=visual_manifest))
        search_evidence = run_anysearch(search_decision['search_queries']) if search_decision['needs_search'] else []
        row['repair_search_decision'] = search_decision
        row['anysearch_evidence'] = search_evidence
        repair_evidence = {'execution': execution, 'outputs': [str(x) for x in copied], 'primary_output': str(primary_output) if primary_output else None, 'diagnostic_outputs': [str(x) for x in diagnostic_outputs], 'mask_diagnosis_policy': 'Files in diagnostic_outputs are the effective masks consumed by the edit. Inspect them before changing segmentation prompt, threshold, or mask growth.', 'comfybench': row['comfybench'], 'repair_search_decision': search_decision, 'anysearch_evidence': search_evidence, 'visual_input_files': diagnosis_paths, 'visual_input_order': visual_manifest, 'attempt_history': report['attempts'], 'parameter_failure_history': report.get('parameter_failures', []), 'instruction': instruction, 'policy': 'Repair against the task requirement, original input, actual output, and ComfyBench feedback. Use search evidence only when requested.'}
        failure_analysis = combined_failure_analysis(search_decision)
        report.setdefault('failure_analyses', []).append({'attempt': attempt, 'analysis': failure_analysis})
        row['failure_analysis'] = failure_analysis
        evolution_context = repair_mode_for_context(failures=[], operation=str(implementation[0].get('operation', '')), input_types=('MASK',) if media_input_nodes(current, 'MASK') else (), diagnosis={'search_decision': search_decision, 'comfybench': row['comfybench'], 'execution': execution, 'failure_analysis': failure_analysis})
        if attempt >= execution_limit:
            if not parameter_failed:
                break
        if parameter_failed and len(report.get('parameter_failures', [])) >= repair_plan_limit():
            report['status'] = 'parameter_repair_exhausted'
            checkpoint()
            break
        repair_ok = False
        repair_errors = []
        repair_failures = []
        previous_repair_plan = None
        for repair_try in range(1, repair_plan_limit() + 1):
            try:
                repair_mode = repair_mode_for_context(failures=repair_failures, operation=str(implementation[0].get('operation', '')), input_types=('MASK',) if media_input_nodes(current, 'MASK') else (), diagnosis=search_decision, retry_index=attempt, execution_failed=True)
                current_repair_evidence = {**repair_evidence, 'preflight_failures': copy.deepcopy(repair_failures), 'allowed_enums': [enum for failure in repair_failures for enum in failure.get('allowed_enums', [])], 'previous_plan': copy.deepcopy(previous_repair_plan), 'repair_mode': repair_mode}
                if repair_mode.get('allow_topology_repair'):
                    current_repair_evidence['local_registry_candidates'] = local_registry_repair_candidates(executor, {'failure_analysis': failure_analysis, 'repair_search_decision': search_decision})
                    current_repair_evidence['local_registry_policy'] = 'These are installed live /object_info candidates. Assess their exact typed interfaces and select the smallest executable repair; do not claim the capability is unavailable without considering this shortlist.'
                repair_plan = generate_plan_with_repair(agent, executor, spec, implementation, graph=current, instruction=instruction, repair_evidence=current_repair_evidence, parameter_only=repair_mode['mode'] == 'parameter', allow_topology_repair=repair_mode.get('allow_topology_repair', False), handoff_geometry_context=handoff_geometry_context, visual_context_paths=[str(source)] if source is not None else [])
                previous_repair_plan = repair_plan
                if repair_mode.get('allow_topology_repair') or repair_plan.get('repair_transaction'):
                    require_controlled_topology_repair(repair_plan, require_edit=True)
                else:
                    require_parameter_only(repair_plan, require_edit=True)
                row.setdefault('repair_generation_plans', []).append({'try': repair_try, 'repair_mode': repair_mode, 'plan': repair_plan})
                if repair_plan['requires_composition_replan']:
                    raise AgentOutputError('composition replan requested without an executable replacement graph')
                repair_base = repair_plan_base_graph(repair_plan, base, current)
                plan_base, selection = graph_selected_by_execution_agent(repair_plan, repair_base, executor)
                if selection:
                    repair_plan['execution_graph_selection'] = selection
                candidate = apply_graph_edits(plan_base, repair_plan['edits'], executor.validator, validate_after_each=not (repair_mode.get('allow_topology_repair', False) or repair_plan.get('repair_transaction')), allow_external_placeholders=True)
                validate_no_new_dead_nodes(plan_base, candidate, executor.validator)
                source, media_type = primary_case_resource(case_resources, modality)
                repair_plan['input_validation'] = prepare_single_media_inputs(candidate, implementation, case_resources, media_type, modality, validator=executor.validator)
                executor.static_validate(candidate)
                candidate = apply_runtime_params(candidate, runtime_param_edits, executor)
                timing_normalizations = normalize_quantized_video_timing(candidate, instruction)
                if timing_normalizations:
                    repair_plan['deterministic_timing_normalizations'] = timing_normalizations
                upscale_normalizations = normalize_upscale_model(candidate, instruction)
                if upscale_normalizations:
                    repair_plan['deterministic_upscale_normalizations'] = upscale_normalizations
                executor.static_validate(candidate)
                validate_temporal_parameters(candidate, instruction)
                current = candidate
                repair_ok = True
                row['repair_generation_plans'][-1]['status'] = 'validated'
                break
            except (AgentTransportError, AgentProviderError) as exc:
                failure = preflight_failure_record(exc, 'single_leaf_repair_request', previous_repair_plan)
                row.setdefault('repair_generation_plans', []).append({'try': repair_try, **failure})
                report['status'] = agent_request_failure_status(exc)
                report['error'] = str(exc)
                report['provider_error'] = getattr(exc, 'provider_error', None)
                report['logical_iteration_consumed'] = False
                report['comfyui_attempt_consumed'] = False
                checkpoint()
                break
            except ComfyUITransportError:
                raise
            except (AgentOutputError, StaticValidationError, ClosedLoopError) as exc:
                repair_errors.append(str(exc))
                failure = preflight_failure_record(exc, 'single_leaf_repair_preflight', previous_repair_plan)
                repair_failures.append(failure)
                row.setdefault('repair_generation_plans', []).append({'try': repair_try, 'repair_mode': repair_mode, **failure})
                continue
        if report.get('status') in {'agent_transport_failed', 'agent_provider_failed'}:
            break
        if not repair_ok:
            row['repair_error'] = repair_errors
            report['status'] = 'repair_preflight_failed'
            report['repair_terminal'] = {'reason': 'planning_budget_exhausted', 'planning_limit': repair_plan_limit(), 'last_failure': repair_failures[-1] if repair_failures else None}
            report['failure_stage'] = 'single_leaf_repair'
            checkpoint()
            break
        checkpoint()
        if parameter_failed:
            attempt = len(report['attempts'])
    report['final_graph'] = current
    terminal_status = final_failure_status(report)
    report['status'] = 'passed' if any((x.get('passed') for x in report['attempts'])) else terminal_status if terminal_status in {'agent_transport_failed', 'agent_provider_failed', 'agent_contract_failed', 'static_preflight_failed', 'composition_diagnosis_contract_failed', 'evaluation_unavailable', 'execution_failed', 'handoff_failed', 'repair_preflight_failed', 'workflow_evolution_failed', 'setup_failed', 'workflow_composition_unsupported'} else 'workflow_unresolved' if evolution_attempted else terminal_status
    if report['status'] == 'passed':
        passed_attempt = next((row for row in report['attempts'] if row.get('passed')), None)
        report['success_class'] = 'evolved_workflow_passed' if evolution_attempted else 'first_pass' if passed_attempt and passed_attempt.get('attempt') == 1 else 'repair_passed'
    report['completed_attempts'] = len(report['attempts'])
    checkpoint()

def load_execution_resources(request: CaseExecutionRequest) -> list[Path]:
    """Resolve declared inputs consistently for either dataset; fail before GPU work."""
    context = current_execution()
    if context and context.hooks.resource_loader:
        return [Path(p) for p in context.hooks.resource_loader(request.case, COMFYUI_INPUT)]
    if 'resource' not in request.case and 'resources' not in request.case:
        return [Path(p) for p in copy_case_resources(request.case['instruction'])]
    names = request.case.get('resource', request.case.get('resources')) or []
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, (list, tuple)):
        raise ValueError('resource must be a filename, list or null')
    roots = request.resource_roots if request.resource_roots is not None else (DATASET / 'resource',)
    sources = []
    for name in names:
        path = Path(name)
        if path.is_absolute() or path.name != str(path):
            raise ValueError(f'input media must be a simple filename: {name}')
        matches = list(dict.fromkeys(((root / path).resolve() for root in roots if (root / path).is_file())))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one source for {name}, found {len(matches)}')
        sources.append(matches[0])
    COMFYUI_INPUT.mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = COMFYUI_INPUT / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    return [COMFYUI_INPUT / source.name for source in sources]

def execute_case(request: CaseExecutionRequest, *, hooks: ExecutionHooks | None=None, policy: ExecutionPolicy | None=None) -> dict:
    """The single execution API for all datasets and routing methods.

    Dataset adapters supply tasks/resources/evaluators, never another runner.
    ContextVar dependencies are restored on both normal and exceptional exits.
    GPU stages keep their shared serial lock and memory-release transaction.
    """
    audit_environment = os.environ.get('COMFYROUTER_AGENT_AUDIT_DIR')
    try:
        with execution_context(request, hooks=hooks, policy=policy):
            if request.resume_report or request.interrupted_report:
                return resume_evaluated_case(request)
            return _execute_case(request)
    finally:
        if audit_environment is None:
            os.environ.pop('COMFYROUTER_AGENT_AUDIT_DIR', None)
        else:
            os.environ['COMFYROUTER_AGENT_AUDIT_DIR'] = audit_environment

def resume_evaluated_case(request: CaseExecutionRequest) -> dict:
    """Restore quality or interrupted-stage checkpoints inside the shared loop."""
    import hashlib
    interrupted = bool(request.interrupted_report)
    report_path = Path(request.interrupted_report or request.resume_report).resolve()
    report = json.loads(report_path.read_text())
    attempts = report.get('attempts', [])
    service_recovery = interrupted and report.get('status') == 'execution_service_unavailable'
    parameter_recovery = interrupted and report.get('status') == 'execution_parameter_failed'
    if str(report.get('case_id')) != request.case_id or report.get('instruction') != request.case['instruction']:
        raise ValueError('checkpoint belongs to a different task')
    output_dir = request.output_root.resolve() / f'case{request.case_id}'
    if Path(report['trace']['case_output']).resolve() != output_dir:
        raise ValueError('checkpoint belongs to a different output directory')
    if interrupted:
        recovery = report.get('parameter_checkpoint' if parameter_recovery else 'service_checkpoint' if service_recovery else 'interrupted_checkpoint', {})
        if service_recovery or parameter_recovery:
            if int(report.get('max_attempts', 0)) != request.max_attempts or recovery.get('attempt') != len(attempts) + 1 or len(attempts) >= request.max_attempts or (recovery.get('comfyui_attempt_consumed') is not False):
                raise ValueError('service recovery requires a non-consuming execution checkpoint')
            if parameter_recovery and (not runtime_parameter_failure(recovery)):
                raise ValueError('parameter recovery requires a proven input/preflight error')
        elif attempts or int(report.get('max_attempts', 0)) != request.max_attempts or report.get('composition', {}).get('status') != 'ready' or (recovery.get('attempt') != 1) or (not recovery.get('stages')):
            raise ValueError('interruption recovery requires a verified partial first composition attempt')
        for row in recovery.get('stages', []):
            if (service_recovery or parameter_recovery) and row.get('execution', {}).get('status') not in {'success', 'reused'}:
                continue
            artifact_ref = row.get('primary_output') or next(iter(row.get('outputs') or []), None)
            if not artifact_ref or not row.get('artifact_sha256'):
                raise ValueError('interrupted stage artifact reference or checksum missing')
            artifact = Path(artifact_ref)
            if not artifact.resolve().is_relative_to(output_dir) or not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != row['artifact_sha256']:
                raise ValueError('interrupted stage artifact changed or outside output directory')
    elif not attempts or len(attempts) >= request.max_attempts or int(report.get('max_attempts', 0)) != request.max_attempts or ([row.get('attempt') for row in attempts] != list(range(1, len(attempts) + 1))) or (attempts[-1].get('passed') is not False) or (not (attempts[-1].get('comfybench', {}).get('comfybench', {}).get('resolved') is False or attempts[-1].get('comfybench', {}).get('status') == 'stage_execution_failed')) or any((row.get('passed') for row in attempts)):
        raise ValueError('continuation requires a completed False and remaining original attempts')
    context = current_execution()
    audit_dir = Path(report['trace']['agent_audit'])
    os.environ['COMFYROUTER_AGENT_AUDIT_DIR'] = str(audit_dir)
    store = None
    agent = context.hooks.agent_factory(store) if context.hooks.agent_factory else AgentOrchestrator(OpenAICompatibleClient(), store)
    executor = context.hooks.executor_factory() if context.hooks.executor_factory else ComfyUIExecutor(timeout=float(os.getenv('COMFYROUTER_EXECUTION_TIMEOUT_SECONDS', '3600')), vram_limit_bytes=int(report['resource_policy']['vram_limit_bytes']))
    resources = [Path(path) for path in report.get('case_resources', [])]
    if not all((path.is_file() for path in resources)):
        raise ValueError('saved input resource missing')

    def checkpoint():
        report['token_usage'] = audit_usage(audit_dir)
        temp = report_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        temp.replace(report_path)
        write_operation_notebook(report, output_dir / 'operation_notebook.jsonl')
        manifest_path = output_dir / 'trace_manifest.json'
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            manifest.update(status=report['status'], updated_at_unix=time.time(), attempt_dirs=sorted((str(p.resolve()) for p in output_dir.glob('attempt_[0-9][0-9]'))))
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    def spec(raw):
        return RequirementSpec(str(raw['modality']), str(raw['task_family']), tuple(raw['operations']), tuple(raw.get('capabilities') or ()), tuple(raw.get('priorities') or ()), tuple(raw.get('constraints') or ()), raw.get('explicit_model'))
    report.update(status='resuming', execution_policy=context.policy.as_dict(), execution_entrypoint='comfyrouter.case_execution.execute_case')
    report.pop('error', None)
    report.setdefault('interruption_resume_history' if interrupted else 'quality_resume_history', []).append({'source_attempt': len(attempts), 'next_attempt': len(attempts) + 1, 'reused_official_evaluation': not interrupted and (not request.regenerate_on_resume), 'regenerate_on_resume': request.regenerate_on_resume, 'quality_policy': os.getenv('COMFYROUTER_QUALITY_POLICY', 'official_comfybench'), 'started_at_unix': time.time(), 'entrypoint': 'comfyrouter.case_execution.execute_case'})
    checkpoint()
    if report.get('composition', {}).get('status') == 'ready':
        composition = report['composition']['stages']
        typed = validate_composition_plan([row['stage'] for row in composition])
        restored = recovery['stages'] if interrupted else attempts[-1]['stages']
        latest = {row['stage_id']: row for row in restored}
        states, cache = ([], {})
        for stage_index, (stage, row) in enumerate(zip(typed, composition)):
            historical = latest.get(stage.stage_id, {}) if interrupted else next((s for a in reversed(attempts) for s in a.get('stages', []) if s.get('stage_id') == stage.stage_id and (s.get('runtime_graph') or s.get('graph'))), {})
            saved_graphs = report.get('final_graphs') or []
            fallback = saved_graphs[stage_index] if interrupted and stage_index < len(saved_graphs) else row['preflight_graph']
            graph = copy.deepcopy(historical.get('runtime_graph') or historical.get('graph') or fallback)
            by_type, primary = composition_handoff_inputs(graph, stage)
            states.append({'typed': stage, 'spec': spec(row['requirement_spec']), 'implementation': row['implementation'], 'graph': graph, 'handoff_inputs_by_type': by_type, 'handoff_inputs': primary, 'handoff_geometry_context': row.get('handoff_geometry_context'), 'deferred_grounded_mask': False})
            prior = latest.get(stage.stage_id, {})
            candidates = [prior.get('primary_output'), *(prior.get('outputs') or [])]
            output = next((str(p) for p in candidates if p and Path(p).is_file()), None)
            if not request.regenerate_on_resume and output and (prior.get('execution', {}).get('status') in {'success', 'reused'}):
                cache[stage.stage_id] = {'output': output, 'attempt': prior.get('reused_from_attempt', 1 if interrupted else len(attempts)), 'handoff': prior.get('handoff')}
        run_composition(request.case_id, report['instruction'], states, agent, executor, output_dir, report, checkpoint, request.max_attempts, resources, initial_cache=cache, start_attempt=len(attempts) + 1, resume_after_evaluation=not interrupted and (not request.regenerate_on_resume), resume_failed_parameters=parameter_recovery)
    else:
        requirement = spec(report['requirement_spec'])
        current = copy.deepcopy(recovery['graph'] if service_recovery or parameter_recovery else attempts[-1]['graph'])
        source, media_type = primary_case_resource(resources, requirement.modality)
        run_single_case(request.case_id, report['instruction'], requirement.modality, requirement, report['implementation_plan'], report['base_graph'], current, source, media_type, external_handoff_geometry_context(current, source, media_type), report.get('runtime_parameter_overrides', []), agent, executor, output_dir, report, checkpoint, request.max_attempts, resources, resume_after_evaluation=not (service_recovery or parameter_recovery or request.regenerate_on_resume), resume_failed_parameters=parameter_recovery)
    return report

def _execute_case(request: CaseExecutionRequest) -> dict:
    args = SimpleNamespace(case_id=request.case_id, output_root=Path(request.output_root), max_attempts=request.max_attempts, vram_limit_gib=request.vram_limit_gib, reuse_report=request.reuse_report)
    case = request.case
    runtime_param_edits = parse_runtime_params(list(request.runtime_params))
    stage_runtime_param_edits = parse_stage_runtime_params(list(request.stage_params))
    routing_correction = request.routing_correction
    if routing_correction is not None and (not isinstance(routing_correction, dict)):
        raise ValueError('routing_correction must contain one JSON object')
    if request.reuse_report:
        previous = json.loads(request.reuse_report.read_text())
        if previous.get('attempts'):
            raise ValueError('Planning-only restart cannot reset any existing execution attempts')
    run_id = f'comfybench-{args.case_id}-{uuid4().hex}'
    report_path = (request.report_root or REPORT_ROOT).resolve() / f'{run_id}.json'
    output_dir = args.output_root.resolve() / f'case{args.case_id}'
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = (request.audit_root or config.EVOLUTION_DIR / 'agent_audit').resolve() / run_id
    audit_dir.mkdir(parents=True, exist_ok=True)
    trace_links = {output_dir / 'agent_trace': audit_dir, output_dir / 'run_report.json': report_path}
    for link, target in trace_links.items():
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    os.environ['COMFYROUTER_AGENT_AUDIT_DIR'] = str(audit_dir)
    store = None
    context = current_execution()
    if context.hooks.agent_factory:
        agent = context.hooks.agent_factory(store)
    else:
        agent = AgentOrchestrator(OpenAICompatibleClient(), store)
    vram_limit_bytes = int(args.vram_limit_gib * 1024 ** 3)
    executor_factory = context.hooks.executor_factory or ComfyUIExecutor
    executor = executor_factory(timeout=float(os.getenv('COMFYROUTER_EXECUTION_TIMEOUT_SECONDS', '3600')), vram_limit_bytes=vram_limit_bytes)
    report = {'schema_version': 'comfybench-bounded-iteration-1', 'run_id': run_id, 'case_id': args.case_id, 'instruction': case['instruction'], 'trace': {'run_id': run_id, 'report': str(report_path), 'agent_audit': str(audit_dir), 'case_output': str(output_dir)}, 'self_evolution': {'enabled': not True, 'planner_called': False, 'mutation_called': False, 'execution_agent_required': True, 'candidates': []}, 'gate': 'official_comfybench_judgment_only', 'max_attempts': args.max_attempts, 'repair_policy': {'planning_attempt_limit': repair_plan_limit(), 'inference_attempt_limit': args.max_attempts, 'parameter_only': False or context.policy.frozen_graph, 'structural_repair': 'forbidden' if context.policy.frozen_graph else 'validated_transaction_after_concrete_failure', 'reconstruction_after_preflight_failures': 3}, 'resource_policy': {'vram_limit_gib': args.vram_limit_gib, 'vram_limit_bytes': vram_limit_bytes, 'action': 'interrupt_current_prompt', 'sample_scope': 'confirmed_running_prompt_only'}, 'runtime_parameter_overrides': runtime_param_edits, 'stage_parameter_overrides': stage_runtime_param_edits, 'attempts': []}
    report['execution_policy'] = context.policy.as_dict()
    report['execution_entrypoint'] = 'comfyrouter.case_execution.execute_case'
    case_resources = []
    report['case_resources'] = []

    def checkpoint():
        report['token_usage'] = audit_usage(audit_dir)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        notebook_path = output_dir / 'operation_notebook.jsonl'
        write_operation_notebook(report, notebook_path)
        attempt_dirs = sorted((str(path.resolve()) for path in output_dir.glob('attempt_[0-9][0-9]') if path.is_dir()))
        manifest = {'schema_version': 'comfybench-trace-manifest-1', 'case_id': args.case_id, 'run_id': run_id, 'status': report.get('status', 'running'), 'report': str(report_path.resolve()), 'operation_notebook': str(notebook_path.resolve()), 'agent_trace': str(audit_dir.resolve()), 'case_output': str(output_dir.resolve()), 'attempt_dirs': attempt_dirs, 'agent_trace_records': sum((1 for path in audit_dir.glob('*.json') if path.is_file())), 'updated_at_unix': time.time()}
        manifest_path = output_dir / 'trace_manifest.json'
        manifest_tmp = output_dir / f'.{manifest_path.name}.{run_id}.tmp'
        manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        manifest_tmp.replace(manifest_path)
    report['status'] = 'running'
    checkpoint()
    try:
        case_resources = load_execution_resources(request)
        report['case_resources'] = [str(path) for path in case_resources]
        from PIL import Image, ImageOps
        owner = getattr(agent, '_agent', agent)
        owner.decomposition_resources = []
        for path in case_resources:
            media = 'video' if path.suffix.lower() in {'.mp4', '.mov', '.avi', '.webm'} else 'image'
            entry = {'path': str(path), 'media_type': media}
            if media == 'image':
                with Image.open(path) as image:
                    width, height = ImageOps.exif_transpose(image).size
                    entry.update(width=width, height=height)
            owner.decomposition_resources.append(entry)
        checkpoint()
        if context.hooks.implementation_provider:
            decision, spec, implementation = context.hooks.implementation_provider(agent, case['instruction'], case['modality'], routing_correction=routing_correction)
        elif args.reuse_report:
            checkpoint_report = json.loads(args.reuse_report.read_text(encoding='utf-8'))
            decision, spec, implementation = implementation_from_report(checkpoint_report)
            report['checkpoint_source'] = str(args.reuse_report)
        else:
            decision, spec, implementation = _require_rag_route()
        report['setup_transport_retries'] = []
        report.update({'classification': decision.__dict__, 'requirement_spec': spec.__dict__, 'implementation_plan': implementation})
        checkpoint()
        if len(decision.subworkflow_sequence) > 1:
            states = prepare_composition(agent, store, executor, implementation, report, checkpoint, case_resources, stage_runtime_param_edits)
            if not report.pop('_collapsed_to_single_leaf', False):
                run_composition(args.case_id, case['instruction'], states, agent, executor, output_dir, report, checkpoint, args.max_attempts, case_resources)
                return report
            decision, spec = classification_for_collapsed_plan(decision, spec, implementation)
            report.update({'classification': decision.__dict__, 'requirement_spec': spec.__dict__, 'implementation_plan': implementation})
            checkpoint()
        base = base_graph(store, spec, implementation, agent=agent, instruction=case['instruction'])
        report['base_graph'] = base
        source, media_type = primary_case_resource(case_resources, case['modality'])
        handoff_geometry_context = external_handoff_geometry_context(base, source, media_type)
        current = None
        plan_error = None
        preflight_failures = []
        previous_plan = None
        try:
            executor.static_validate(base, allow_external_placeholders=True)
        except StaticValidationError as exc:
            plan_error = str(exc)
            preflight_failures.append(preflight_failure_record(exc, 'base_static_validation'))
            report['base_static_validation'] = {'status': 'repair_required', 'error': str(exc), 'diagnostics': exc.diagnostics}
        for plan_attempt in range(1, repair_plan_limit() + 1):
            try:
                repair_mode = repair_mode_for_context(failures=preflight_failures, operation=str(implementation[0].get('operation', '')))
                agent_plan = generate_plan_with_repair(agent, executor, spec, implementation, graph=base if current is None else current, instruction=case['instruction'], parameter_only=repair_mode['mode'] == 'parameter', allow_topology_repair=repair_mode.get('allow_topology_repair', False), handoff_geometry_context=handoff_geometry_context, handoff_contract_context=single_stage_input_contract(implementation, media_type), visual_context_paths=[str(source)] if source is not None else [], repair_evidence=preflight_retry_evidence(preflight_failures, previous_plan, 'Repair the parameter edit contract only; do not score or evolve memory.' if repair_mode['mode'] == 'parameter' else 'Repair only the statically diagnosed graph structure; do not score or evolve memory.', graph=base if current is None else current, validator=executor.validator))
                previous_plan = agent_plan
                if repair_mode.get('allow_topology_repair') or agent_plan.get('repair_transaction'):
                    require_controlled_topology_repair(agent_plan, require_edit=plan_error is not None)
                else:
                    require_parameter_only(agent_plan, require_edit=plan_error is not None)
                report.setdefault('generation_plan_attempts', []).append({'attempt': plan_attempt, 'repair_mode': repair_mode, 'plan': agent_plan})
                if agent_plan['requires_composition_replan']:
                    raise AgentOutputError('controlled composition replan was requested but no executable replacement graph was supplied')
                plan_base, selection = graph_selected_by_execution_agent(agent_plan, base if current is None else current, executor)
                if selection:
                    agent_plan['execution_graph_selection'] = selection
                candidate = apply_graph_edits(plan_base, agent_plan['edits'], executor.validator, validate_after_each=not (repair_mode.get('allow_topology_repair') or agent_plan.get('repair_transaction')), allow_external_placeholders=True)
                input_validation = prepare_single_media_inputs(candidate, implementation, case_resources, media_type, case['modality'], validator=executor.validator)
                runtime_bindings = input_validation['bindings']
                executor.static_validate(candidate)
                candidate = apply_runtime_params(candidate, runtime_param_edits, executor)
                timing_normalizations = normalize_quantized_video_timing(candidate, case['instruction'])
                if timing_normalizations:
                    agent_plan['deterministic_timing_normalizations'] = timing_normalizations
                upscale_normalizations = normalize_upscale_model(candidate, case['instruction'])
                if upscale_normalizations:
                    agent_plan['deterministic_upscale_normalizations'] = upscale_normalizations
                executor.static_validate(candidate)
                validate_temporal_parameters(candidate, case['instruction'])
                current = candidate
                if plan_attempt == 1:
                    report['initial_generation_plan'] = agent_plan
                else:
                    report['setup_repair_generation_plan'] = agent_plan
                report['runtime_input_bindings'] = runtime_bindings
                report['input_preflight'] = input_validation
                break
            except (AgentTransportError, AgentProviderError) as exc:
                plan_error = str(exc)
                preflight_failures.append(preflight_failure_record(exc, 'generation_plan_transport', previous_plan))
                report.setdefault('generation_transport_retries', []).append(plan_error)
                raise
            except ComfyUITransportError:
                raise
            except (AgentOutputError, ClosedLoopError, StaticValidationError) as exc:
                plan_error = str(exc)
                failure = preflight_failure_record(exc, 'generation_or_static_preflight', previous_plan)
                preflight_failures.append(failure)
                report.setdefault('generation_plan_attempts', []).append({'attempt': plan_attempt, 'repair_mode': repair_mode, **failure})
        if current is None:
            report['repair_terminal'] = {'reason': 'planning_budget_exhausted', 'planning_limit': repair_plan_limit(), 'last_failure': preflight_failures[-1] if preflight_failures else None}
            raise RepairPreflightError(plan_error or 'generation plan failed')
        report['preflight_graph'] = current
        checkpoint()
    except (AgentTransportError, AgentProviderError) as exc:
        report['status'] = agent_request_failure_status(exc)
        report['error_type'] = type(exc).__name__
        report['error'] = str(exc)
        report['provider_error'] = getattr(exc, 'provider_error', None)
        report['logical_iteration_consumed'] = False
        report['comfyui_attempt_consumed'] = False
        checkpoint()
        return report
    except ComfyUITransportError as exc:
        report.update(status='execution_service_unavailable', error=str(exc), comfyui_attempt_consumed=False, attempts_consumed=len(report.get('attempts', [])))
        report['setup_service_failure'] = {'audit': exc.audit, 'phase': 'preparation', 'recorded_at_unix': time.time()}
        checkpoint()
        return report
    except (AgentOutputError, CompositionPlanError, ClosedLoopError, StaticValidationError, ValueError) as exc:
        composition_unsupported = str(exc).startswith('workflow_composition_unsupported')
        report['status'] = 'repair_preflight_failed' if isinstance(exc, RepairPreflightError) else 'workflow_composition_unsupported' if composition_unsupported else 'setup_failed'
        report['error'] = str(exc)
        report['failure_stage'] = 'repair_preflight' if isinstance(exc, RepairPreflightError) else 'composition_planning' if str(exc).startswith('workflow_composition_unsupported') or isinstance(exc, CompositionPlanError) else 'generation_or_static_preflight'
        report['attempts_consumed'] = 0
        checkpoint()
        return report
    run_single_case(args.case_id, case['instruction'], case['modality'], spec, implementation, base, current, source, media_type, handoff_geometry_context, runtime_param_edits, agent, executor, output_dir, report, checkpoint, args.max_attempts, case_resources)
    return report

def _require_rag_route():
    raise AgentOutputError('RAG execution requires a joint workflow selection checkpoint')
