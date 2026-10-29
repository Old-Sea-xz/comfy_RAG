"""Prepare a fresh RAG batch from a local task file and workflow library."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_routing_comparison import read, write, verify_frozen


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(tasks_path, resources, library, model, run, *, registry=None,
            max_attempts=3, workers=8, queue_window=3, vram_gib=115,
            evaluation_workdir=None):
    """Copy definitions and rubrics only; never import routes or outcomes."""
    import numpy as np
    from comfyrouter import config
    config.load_env()
    tasks_path, resources, library, model, run = [Path(p).resolve()
                                                for p in (tasks_path, resources, library, model, run)]
    if run.exists():
        raise ValueError('destination exists; use tools/run.py to resume it')
    if not resources.is_dir() or not model.is_dir():
        raise ValueError('resource and local E5 model directories must exist')
    if max_attempts not in (1, 3) or workers < 1 or queue_window < 1:
        raise ValueError('attempts must be 1 or 3; workers and queue window must be positive')
    tasks = read(tasks_path)
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError('task file must be a nonempty object keyed by case ID')
    inputs, gold, source = {}, {}, {}
    for cid, task in tasks.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', cid):
            raise ValueError('case ID must be a simple filename component')
        instruction, modality = task['instruction'], task['modality']
        if not isinstance(instruction, str) or not instruction.strip() or modality not in {'t2i','i2i','i2v','t2v','v2v'}:
            raise ValueError('invalid instruction or modality for ' + cid)
        names = task.get('resources', task.get('resource')) or []
        names = [names] if isinstance(names, str) else names
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError('resources must be filenames for ' + cid)
        inferred = []
        for name in names:
            path = (resources / name).resolve()
            if not path.is_relative_to(resources) or not path.is_file():
                raise ValueError('input resource missing or outside resource directory: ' + name)
            inferred.append('video' if path.suffix.lower() in {'.mp4','.webm','.mov','.avi','.mkv','.gif'} else 'image')
        kinds = task.get('input_types', inferred)
        if len(kinds) != len(names) or any(kind not in {'image','video','mask'} for kind in kinds):
            raise ValueError('input_types must describe every supplied resource for ' + cid)
        inputs[cid+'_v0'] = dict(case_id=cid, variant=0, instruction=instruction,
                                modality=modality, resources=names, input_types=kinds)
        category = task.get('category', 'unspecified')
        source[cid] = dict(instruction=instruction, modality=modality, category=category, resource=names)
        gold[cid] = dict(case_id=cid, category=category, composite=bool(task.get('composite', False)))
    catalog = read(library/'catalog.json')
    manifest = read(library/'embedding_manifest.json')
    vectors_path = library/'workflow_vectors.npy'
    vectors = np.load(vectors_path, allow_pickle=False)
    if [row['workflow_id'] for row in catalog] != manifest['workflow_ids']:
        raise ValueError('catalog/embedding order mismatch')
    if vectors.ndim != 2 or len(vectors) != len(catalog) or not np.isfinite(vectors).all():
        raise ValueError('invalid workflow vectors')
    if sha256(vectors_path) != manifest['vectors_sha256']:
        raise ValueError('embedding file hash mismatch')
    for row, digest in zip(catalog, manifest['description_sha256'], strict=True):
        if hashlib.sha256(row['description'].encode()).hexdigest() != digest:
            raise ValueError('description/embedding mismatch')
        graph = (library/row['workflow_path']).resolve()
        if not graph.is_relative_to(library) or sha256(graph) != row['sha256']:
            raise ValueError('workflow file/hash mismatch')
    # Fetch live node definitions locally. No API credential is sent to ComfyUI.
    if registry is None:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        api = os.getenv('COMFYROUTER_COMFYUI_API', 'http://127.0.0.1:8188')
        with opener.open(api.rstrip('/')+'/object_info', timeout=30) as response:
            node_registry = json.load(response)
    else:
        node_registry = read(Path(registry))
    if not isinstance(node_registry, dict) or not node_registry:
        raise ValueError('node registry must be a nonempty object')
    frozen = run/'frozen'
    frozen.mkdir(parents=True)
    copied = []
    for row in catalog:
        target = frozen/'workflows'/row['workflow_id']/'workflow_api.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library/row['workflow_path'], target)
        copied.append({**row, 'workflow_path': str(target), 'base_path': str(target.parent)})
    shutil.copy2(ROOT/'data/agent_skills.json', frozen/'agent_skills.json')
    write(frozen/'node_registry.json', node_registry)
    (run/'models').mkdir()
    (run/'models/multilingual-e5-small').symlink_to(model, target_is_directory=True)
    shutil.copy2(vectors_path, run/'workflow_vectors.npy')
    shutil.copy2(library/'embedding_manifest.json', run/'embedding_manifest.json')
    write(run/'workflow_catalog.json', copied)
    write(run/'task_inputs.json', inputs)
    write(run/'gold.json', gold)
    write(run/'source_instructions.json', source)
    for cid, task in tasks.items():
        write(run/'baseline'/cid/'evaluation.json', {'evaluation_criteria': task.get('evaluation_criteria', [])})
    evaluation_root = Path(evaluation_workdir).resolve() if evaluation_workdir else run/'evaluation_workspace'
    if not evaluation_workdir:
        refs = evaluation_root/'dataset/benchmark/resource'
        refs.parent.mkdir(parents=True)
        refs.symlink_to(resources, target_is_directory=True)
    for name in ['logs','routing','execution','runtime_state','claims']:
        (run/name).mkdir()
    protocol = dict(schema_version='agentic-rag-1', rag_profile='agentic_feedback',
        case_ids=list(tasks), cases=len(tasks), rag_top_k=20, rag_query_limit=3, repair_rag=True,
        execution_methods=['semantic'], queue_routing=True, stage_pipeline=True, stage_fifo_queue=True,
        native_comfyui_pending_queue=True, native_queue_window=queue_window,
        comfyui_api=os.getenv('COMFYROUTER_COMFYUI_API', 'http://127.0.0.1:8188'),
        execution_worker_limit=workers, execution_seed=20261010,
        serial_api=True, api_serial_lock=os.getenv('COMFYROUTER_API_SERIAL_LOCK','/tmp/comfyrouter-api.serial.lock'),
        gpu_serial_lock=os.getenv('COMFYROUTER_COMFYUI_LOCK','/tmp/comfyrouter-comfyui.serial.lock'),
        max_attempts=max_attempts, vram_limit_gib=vram_gib, video_frame_budget=0,
        llm_transport_retries=2, execution_timeout_seconds=5400,
        resource_root=str(resources), evaluation_workdir=str(evaluation_root),
        evaluation_model=config.evaluation_model(), evaluation_api_style='responses',
        evaluation_uses_primary_key=not bool(os.getenv('COMFYROUTER_LLM_EVALUATION_API_KEY')),
        immediate_final_media_check=True, quality_policy='visual_task_tolerant_v2',
        transport_retry_policy=dict(enabled=True,initial_backoff_seconds=60,max_backoff_seconds=600,max_recoveries_per_case=5),
        frozen_files={str(p.relative_to(frozen)):sha256(p) for p in frozen.rglob('*') if p.is_file()})
    write(run/'protocol.json', protocol)
    write(run/'RUN_MANIFEST.json', dict(status='prepared', jobs=[['semantic',cid+'_v0'] for cid in tasks],
        code_hashes={str(p.relative_to(ROOT)):sha256(p) for folder in ['comfyrouter','tools'] for p in (ROOT/folder).rglob('*.py')}))
    verify_frozen(run)
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--resources', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--library', type=Path, default=ROOT/'library')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--registry', type=Path)
    p.add_argument('--max-attempts', type=int, choices=[1,3], default=3)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--queue-window', type=int, default=3)
    p.add_argument('--vram-gib', type=float, default=115)
    p.add_argument('--evaluation-workdir', type=Path)
    a = p.parse_args()
    print(prepare(a.tasks,a.resources,a.library,a.model,a.run,registry=a.registry,
        max_attempts=a.max_attempts,workers=a.workers,queue_window=a.queue_window,
        vram_gib=a.vram_gib,evaluation_workdir=a.evaluation_workdir))


if __name__ == '__main__':
    main()
