"""Optional local workflow retrieval at every setup and repair planning boundary."""
import copy
import hashlib
import json
from pathlib import Path

from .agentic import AgentOutputError


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class RepairRetriever:
    def __init__(self, run, audit):
        self.run, self.audit = Path(run), Path(audit)
        self.encoder = None

    @property
    def query_count(self):
        return sum(len(read(p).get('retrievals', [])) for p in self.audit.glob('*.json'))

    def augment(self, client, kwargs):
        context = {key: kwargs.get(key) for key in
                   ('instruction', 'overall_instruction', 'graph', 'repair_evidence',
                    'parameter_only', 'allow_topology_repair', 'handoff_contract_context')}
        digest = hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest()
        path = self.audit / (digest + '.json')
        record = read(path) if path.exists() else {'context_sha256': digest, 'retrievals': []}
        if 'decision' not in record:
            schema = {'type': 'object', 'additionalProperties': False,
                      'required': ['retrieval_required', 'queries', 'reason'],
                      'properties': {'retrieval_required': {'type': 'boolean'},
                                     'queries': {'type': 'array', 'maxItems': 3,
                                                 'items': {'type': 'string'}},
                                     'reason': {'type': 'string'}}}
            decision = client.complete_json('repair_workflow_retrieval_agent', {
                'role': 'Decide whether additional local workflow references would help the next generation or repair.',
                'rules': [
                    'Use the actual current graph, input contracts, preflight errors, execution errors and evaluation feedback.',
                    'Retrieve only when examples from the local workflow library would materially help resolve a concrete uncertainty or missing capability. Otherwise return retrieval_required=false and queries=[].',
                    'You may request retrieval at any setup, preflight repair, execution repair or quality iteration; previous routing retrieval does not forbid it.',
                    'Return up to three focused queries describing the missing capability, node usage or workflow pattern. Explain the decision.',
                    'Retrieval supplies reference evidence only. It does not install models, change the task, replace a running prompt or bypass graph validation.',
                ]}, {**context, '_max_output_tokens': 1200,
                     '_response_json_schema': {'name': 'repair_workflow_retrieval', 'schema': schema}})
            if (not isinstance(decision, dict)
                    or set(decision) != {'retrieval_required', 'queries', 'reason'}
                    or type(decision['retrieval_required']) is not bool
                    or not isinstance(decision['reason'], str) or not decision['reason'].strip()
                    or not isinstance(decision['queries'], list) or len(decision['queries']) > 3
                    or any(not isinstance(q, str) or not q.strip() for q in decision['queries'])
                    or decision['retrieval_required'] != bool(decision['queries'])):
                raise AgentOutputError('invalid repair workflow retrieval decision')
            record['decision'] = decision
            write(path, record)
        decision = record['decision']
        for query in dict.fromkeys(decision['queries']):
            if any(row['query'] == query for row in record['retrievals']):
                continue
            record['retrievals'].append(self.search(query))
            write(path, record)
        result = dict(kwargs)
        evidence = copy.deepcopy(kwargs.get('repair_evidence') or {})
        evidence['workflow_rag'] = record
        evidence['workflow_rag_policy'] = (
            'Use these existing workflow graphs as reference evidence for the current repair. '
            'Retain user inputs, task goals, successful stage contracts and the allowed repair mode. '
            'Models/nodes in references are not guaranteed installed. Validate all edits against live interfaces; '
            'do not silently switch the selected workflow chain or assume retrieved media is user input.')
        result['repair_evidence'] = evidence
        return result

    def search(self, query):
        import numpy as np
        from .semantic_retrieval import E5Encoder
        if self.encoder is None:
            self.encoder = E5Encoder(self.run / 'models/multilingual-e5-small')
        docs = read(self.run / 'workflow_catalog.json')
        vectors = np.load(self.run / 'workflow_vectors.npy', allow_pickle=False)
        vector = self.encoder.encode([query], query=True)[0]
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('invalid repair retrieval query vector')
        scores = vectors @ (vector / norm)
        k = int(read(self.run / 'protocol.json')['rag_top_k'])
        indices = sorted(range(len(docs)), key=lambda i: (-float(scores[i]), docs[i]['workflow_id']))[:k]
        candidates = []
        for rank, index in enumerate(indices):
            doc = docs[index]
            row = {'workflow_id': doc['workflow_id'], 'description': doc['description'],
                   'similarity': float(scores[index])}
            # Keep all Top-20 descriptions; include the three closest executable references.
            if rank < 3:
                graph_path = Path(doc['workflow_path'])
                if hashlib.sha256(graph_path.read_bytes()).hexdigest() != doc['sha256']:
                    raise ValueError('frozen repair reference hash mismatch: ' + doc['workflow_id'])
                row['reference_graph'] = read(graph_path)
            candidates.append(row)
        return {'query': query, 'candidates': candidates}
