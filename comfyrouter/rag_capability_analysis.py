"""One full-request capability summary; no catalog or per-step planning."""
from .agentic import AgentOutputError


def analyze_capabilities(client, task):
    result = client.complete_json('direct_semantic_selection_agent', {
        'role': 'Summarize the capabilities needed by the complete request for dense workflow retrieval.',
        'rules': [
            'Return one concise English paragraph describing operations, actual supplied inputs, '
            'required outputs, and explicit numeric constraints.',
            'Preserve reference conditioning, identity/style preservation and explicit ordering.',
            'Do not output a plan, numbered subtasks, workflow IDs, or individual retrieval queries.',
            'Do not add unstated requirements. A visual storm or thunder scene does not itself '
            'require generated audio unless the request explicitly asks for sound.',
            'Do not decide whether the request is feasible or promise generation quality.',
        ],
    }, {
        'stage': 'rag_full_request_capability_analysis', 'instruction': task['instruction'],
        'resources': [{'filename': name, 'type': kind}
                      for name, kind in zip(task['resources'], task['input_types'])],
        '_max_output_tokens': 2048, '_reasoning_effort': 'low',
        '_response_json_schema': {'name': 'rag_capability_analysis', 'schema': {
            'type': 'object', 'additionalProperties': False, 'required': ['analysis'],
            'properties': {'analysis': {'type': 'string'}},
        }},
    })
    if not isinstance(result, dict) or set(result) != {'analysis'} \
            or not isinstance(result['analysis'], str) or not result['analysis'].strip():
        raise AgentOutputError('invalid RAG capability analysis')
    return result['analysis'].strip()
