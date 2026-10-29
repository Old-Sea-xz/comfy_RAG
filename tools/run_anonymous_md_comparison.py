"""Full-request RAG/direct replacement; no independently decomposed inputs."""
import argparse, copy, fcntl, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_routing_comparison import read, write, configure, client_type, claim, PYTHON, verify_frozen

def execution_attempt_budget(retrieval_queries, iteration_limit=3):
    """Retrieval does not consume or reset the GPU execution iteration budget."""
    return iteration_limit

def read_route(path):
    return read(path) if path.exists() else {'methods': {}}

def save_route_prediction(path, method, prediction):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = read_route(path)
        result.setdefault('methods', {})[method] = prediction
        write(path, result)

def route_worker(run, method, key):
    configure(run)
    task = read(run / 'task_inputs.json')[key]
    protocol = read(run / 'protocol.json')
    single_pass = protocol.get('rag_profile') == 'capability_single_pass'
    path = run / 'routing' / key / 'route.json'
    if method in read_route(path)['methods']:
        return
    if not claim(run / 'claims/whole_request_routing' / (method + '_' + key)):
        return
    os.environ['COMFYROUTER_AGENT_AUDIT_DIR'] = str(run / 'routing' / key / (method + '_whole_request_trace'))
    os.environ['COMFYROUTER_LLM_PROMPT_CACHE_NAMESPACE'] = 'whole-request:' + method + ':' + key
    documents = read(run / 'workflow_catalog.json')
    retrieval = None
    transport = []
    checkpoint_path = run / 'routing' / key / (method + '_selection_checkpoint.json')
    try:
        import numpy as np
        from comfyrouter.semantic_retrieval import E5Encoder
        encoder = E5Encoder(run / 'models/multilingual-e5-small')
        vectors = np.load(run / 'workflow_vectors.npy')
        k = read(run / 'protocol.json')['rag_top_k']
        query_limit = 1 if single_pass else int(protocol.get('rag_query_limit', 3))
        if not 1 <= query_limit <= 3:
            raise ValueError('rag_query_limit must be between 1 and 3')
        retrieval = {'query_limit': query_limit, 'query_count': 0, 'top_k_per_query': k, 'candidate_window': 'current query only; no cross-round accumulation', 'modality_filter': False, 'rounds': []}
        from comfyrouter.whole_request_selection import select_whole_request
        from comfyrouter.agentic import AgentOutputError
        client = client_type()()
        from comfyrouter.agentic import AgentTransportError, AgentProviderError
        from comfyrouter.transport_retry import retryable_api_failure
        import numpy as np
        query_text = task['instruction']
        if single_pass:
            from comfyrouter.rag_capability_analysis import analyze_capabilities
            analysis_path = run / 'routing' / key / 'capability_analysis.json'
            checkpoint = read(checkpoint_path) if checkpoint_path.exists() else {}
            if checkpoint.get('status') == 'selection_pending':
                analysis = checkpoint['retrieval_evidence'].get('capability_analysis', '')
            elif analysis_path.exists():
                analysis = read(analysis_path)['analysis']
            else:
                for transport_attempt in range(3):
                    try:
                        analysis = analyze_capabilities(client, task)
                        break
                    except (AgentTransportError, AgentProviderError) as exc:
                        transport.append({'stage': 'analysis', 'attempt': transport_attempt + 1, 'error': str(exc)})
                        if transport_attempt == 2 or not retryable_api_failure(str(exc), type(exc).__name__):
                            raise
                        time.sleep(5 * (transport_attempt + 1))
                        client = client_type()()
                write(analysis_path, {'analysis': analysis, 'instruction': task['instruction']})
            query_text = f'Instruction: {task['instruction']}\n\nAnalysis: {analysis}'
            retrieval['capability_analysis'] = analysis
        current_docs = []
        checkpoint = read(checkpoint_path) if checkpoint_path.exists() else {}
        start_round = 1
        if checkpoint.get('status') == 'selection_pending':
            retrieval = checkpoint['retrieval_evidence']
            start_round = retrieval['query_count']
        for search_round in range(start_round, query_limit + 1):
            if checkpoint.get('status') == 'selection_pending' and search_round == start_round:
                by_id = {d['workflow_id']: d for d in documents}
                current_docs = [{**by_id[c['workflow_id']], 'similarity': c['similarity']} for c in retrieval['rounds'][-1]['candidates']]
            else:
                query = encoder.encode([query_text], query=True)[0]
                query = query / np.linalg.norm(query)
                scores = vectors @ query
                indices = sorted(range(len(documents)), key=lambda i: (-float(scores[i]), documents[i]['workflow_id']))[:k]
                current_docs = [{**documents[i], 'similarity': float(scores[i])} for i in indices]
                retrieval['query_count'] = search_round
                retrieval['rounds'].append({'round': search_round, 'query': query_text, 'candidates': [{'workflow_id': doc['workflow_id'], 'similarity': doc['similarity']} for doc in current_docs]})
            write(checkpoint_path, {'status': 'selection_pending', 'retrieval_evidence': retrieval})
            for transport_attempt in range(3):
                try:
                    decision, spec, implementation, raw = select_whole_request(client, current_docs, task['instruction'], task['modality'], task['resources'], task['input_types'], method, anonymous_descriptions=True, retrieval_feedback=not single_pass, retrieval_round=search_round, retrieval_limit=query_limit, prompt_profile='rag_single_pass' if single_pass else 'standard')
                    break
                except (AgentTransportError, AgentProviderError) as exc:
                    transport.append({'round': search_round, 'attempt': transport_attempt + 1, 'error': str(exc)})
                    if transport_attempt == 2 or not retryable_api_failure(str(exc), type(exc).__name__):
                        raise
                    time.sleep(5 * (transport_attempt + 1))
                    client = client_type()()
            retrieval['rounds'][-1]['agent_decision'] = raw
            if raw.get('search_needed'):
                if search_round == query_limit:
                    raise AgentOutputError(f'selection blocked after {query_limit} retrieval: ' + raw['reason'])
                query_text = raw['search_query'].strip()
                continue
            break
        else:
            raise AgentOutputError('selection blocked after retrieval limit')
        budget = execution_attempt_budget(retrieval['query_count'], int(protocol.get('max_attempts', 3)))
        prediction = {'status': 'routed', 'classification': decision.__dict__, 'requirement_spec': spec.__dict__, 'implementation_plan': implementation, 'whole_request_selection': raw, 'retrieval_evidence': retrieval, 'iteration_budget': {'maximum': budget, 'retrieval_queries': retrieval['query_count'], 'execution_attempts_remaining': budget}, 'transport_failures': transport, 'candidate_count': len(current_docs), 'independent_decomposition': False, 'selection_prompt_profile': 'rag_single_pass' if single_pass else 'standard', 'selection_decisions': retrieval['query_count'], 'routing_input': task}
    except Exception as exc:
        from comfyrouter.transport_retry import retryable_api_failure
        retryable = retryable_api_failure(str(exc), type(exc).__name__)
        api_error = isinstance(exc, (AgentTransportError, AgentProviderError)) if 'AgentTransportError' in locals() else False
        prediction = {'status': 'agent_transport_failed' if retryable else 'agent_provider_failed' if api_error else 'routing_failed', 'error': str(exc), 'error_type': type(exc).__name__, 'retryable': retryable, 'transport_failures': transport, 'failure_stage': 'routing', 'independent_decomposition': False, 'retrieval_evidence': retrieval}
    save_route_prediction(path, method, prediction)
    print(json.dumps({'method': method, 'key': key, 'status': prediction['status']}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--route-worker', nargs=2, required=True, metavar=('METHOD', 'CASE'))
    args = parser.parse_args()
    route_worker(args.run.resolve(), *args.route_worker)
