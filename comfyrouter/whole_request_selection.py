"""One full-request decision selecting existing workflows and their media edges."""
import hashlib
import json
import os
from .agentic import AgentDecision, AgentOutputError
from .types import RequirementSpec
from .composition import validate_composition_plan


def select_whole_request(client, documents, instruction, modality, resources, input_types, method,
                         *, anonymous_descriptions=False, retrieval_feedback=False,
                         retrieval_round=1, retrieval_limit=1, prompt_profile="standard"):
    """No gate, atomic-step input, per-step search, or taxonomy traversal."""
    if prompt_profile not in {"standard", "bare_semantic", "rag_single_pass"}:
        raise ValueError(f"unsupported whole-request prompt profile: {prompt_profile}")
    ids=[d['workflow_id'] for d in documents]
    fields={
        'workflow_id':{'type':'string','enum':ids},
        'instruction':{'type':'string'},
        'input_type':{'type':'string','enum':['text','image','video','mask']},
        'input_types':{'type':'array','items':{'type':'string','enum':['text','image','video','mask']},'minItems':1},
        'output_type':{'type':'string','enum':['image','video','mask']},
        'depends_on':{'type':'array','items':{'type':'integer','minimum':1,'maximum':8}},
        'mask_role':{'type':['string','null'],'enum':['edit_region','preserve_region',None]},
        'mask_construction':{'type':['string','null'],'enum':['segment_visible','prospective_placement',None]},
        'target_presence':{'type':['string','null'],'enum':['present','absent',None]},
        'geometry_requirement':{'type':['string','null'],'enum':['source_topology','consumer_target_canvas',None]},
        'protected_regions':{'type':'array','items':{'type':'string'}},
    }
    output_properties={'reason':{'type':'string'},'workflows':{
        'type':'array','minItems':0 if anonymous_descriptions else 1,'maxItems':8,'items':{
            'type':'object','additionalProperties':False,'required':list(fields),'properties':fields}}}
    output_required=['workflows','reason']
    if retrieval_feedback:
        output_properties={
            'search_needed':{'type':'boolean'},
            'search_query':{'type':'string'},
            **output_properties,
        }
        output_required=['search_needed','search_query','workflows','reason']
    candidates=[{k:d[k] for k in (['workflow_id','description'] if anonymous_descriptions else
                                  ['workflow_id','modality','description','node_types'])} for d in documents]
    template_rule=('Each supplied description states the function, inputs and outputs of an existing workflow.'
                   if anonymous_descriptions else 'Descriptions and node_types describe existing templates.')
    if prompt_profile == "bare_semantic":
        contract = {
            'role': 'Given the user request and the workflow descriptions, select suitable workflows and combine them.',
            'rules': [],
        }
    else:
        contract = {
            'role':'Read the complete user request and jointly select existing workflows and their dependencies in one decision.',
            'rules':[
            ('Judge whether this candidate set contains workflows that can complete every requested goal. If a required capability is absent, set search_needed=true, return no workflows, and provide a concise English search_query describing only the missing capability and its key input/output or model family. Do not request more candidates merely because you prefer another implementation. If the candidates are sufficient, set search_needed=false, provide an empty search_query, and select the complete workflow chain.' if retrieval_feedback else 'No separate task decomposition, taxonomy classification, search, or invented workflow. You receive only the complete request and this candidate catalog.'),
            'Select one or more supplied workflow_id values. Prefer a single existing workflow when it covers every goal; otherwise select an ordered executable chain.',
            'Every requested goal must be addressed. Decide the entire chain, per-workflow instruction, media contract and dependencies together in this response.',
            template_rule+' Start from the selected existing workflows. Parameters, prompts and media bindings may be adapted. The shared executor also permits validated structural repairs after concrete execution or quality failures. Do not assume arbitrary new models or unsupported capabilities.',
            'The shared executor can use installed ImageScale/ImageCrop nodes for deterministic output resizing or center cropping during validated repairs. Preserve such requested formatting in the selected stage instruction; it does not require a separate catalog workflow. This does not guarantee semantic generation quality.',
            'Do not require external media not supplied by the user. A masked template needs a real supplied or upstream-generated mask. A start/end video template needs both real frames.',
            'depends_on contains 1-based indices of earlier workflows producing required media. Empty depends_on means original user media or text. Include every consumed upstream artifact; image and mask can come from different workflows.',
            'Keep source images when consuming generated masks. MASK outputs must declare edit_region/preserve_region and appropriate mask_construction; do not replace a present object mask with an arbitrary placement region.',
            'Write instructions preserving numeric constraints, filenames, quoted text and scene preservation. Never omit a difficult goal or change the user request.',
            ('If the supplied candidates cannot form an executable chain using the supplied or upstream media, request another retrieval only when retrieval_feedback is enabled; otherwise return an empty workflows array and explain the missing capability. Do not invent inputs or force an unsuitable workflow.' if anonymous_descriptions else 'Select at least one workflow.'),
            ('Evaluate only the current candidate set afresh. Earlier missing-capability conclusions do not apply to new candidates. Check every current description before declaring a capability absent. Every input listed by the chosen description must be supplied or produced upstream; never omit a required mask from input_types.' if retrieval_feedback else 'Choose once; no subsequent workflow reselection.'),
            'Return exactly the fields defined by the result tool schema.'
        ]}
    if prompt_profile == 'rag_single_pass':
        contract['rules'].extend([
            'Distinguish executable media/input compatibility from uncertain visual quality. '
            'Select a supported chain that attempts every requested goal; semantic fidelity, '
            'identity preservation and aesthetic quality are evaluated after execution and '
            'are not prerequisites that must be guaranteed before selecting.',
            'Read the exact documented inputs. Do not invent a reference-image requirement '
            'for a workflow accepting only text and layout parameters.',
            'Only explicitly requested sound requires audio generation. Do not infer extra '
            'audio requirements from visual scene words such as thunder or rainstorm.',
            'There is exactly one retrieval and one joint selection. Return no workflows '
            'only if a genuine required capability or supplied/upstream input is missing. '
            'Never fabricate a missing input or silently omit an explicit goal.',
        ])
    raw=client.complete_json('direct_semantic_selection_agent', contract, {
            'stage':(f'anonymous_description_selection_round_{retrieval_round}' if retrieval_feedback else
                     'anonymous_description_selection_bare_semantic' if prompt_profile == 'bare_semantic' else
                     'anonymous_description_selection') if anonymous_descriptions else method+':whole_request',
            'instruction':instruction,
            **({} if anonymous_descriptions else {'overall_modality':modality}),
            'resources':[{'filename':name,'type':kind} for name,kind in zip(resources,input_types)],
            'candidates':candidates,
            '_max_output_tokens':8192,'_reasoning_effort':'low',
            '_retrieval_round':retrieval_round,
            '_retrieval_limit':retrieval_limit,
            **({'_prompt_history_mode':'latest_only',
                '_prompt_cache_namespace':os.getenv('COMFYROUTER_LLM_PROMPT_CACHE_NAMESPACE','whole-request')
                    + ':candidates:' + hashlib.sha256(json.dumps(candidates,ensure_ascii=False,
                        sort_keys=True).encode()).hexdigest()[:20]} if retrieval_feedback else {}),
            '_response_json_schema':{'name':'whole_request_workflows','schema':{
                'type':'object','additionalProperties':False,'required':output_required,
                'properties':output_properties}}})
    expected=set(output_required)
    if not isinstance(raw,dict) or set(raw)!=expected or not isinstance(raw['reason'],str):
        raise AgentOutputError('invalid complete-request selection response')
    if retrieval_feedback:
        if type(raw['search_needed']) is not bool or not isinstance(raw['search_query'],str):
            raise AgentOutputError('invalid retrieval feedback response')
        if raw['search_needed']:
            if raw['workflows'] or not raw['search_query'].strip():
                raise AgentOutputError('retrieval feedback must provide only a non-empty search query')
            return None,None,None,raw
        if raw['search_query'].strip():
            raise AgentOutputError('search_query must be empty when candidates are sufficient')
    rows=raw['workflows']
    if anonymous_descriptions and rows==[]:
        raise AgentOutputError('selection blocked: '+raw['reason'])
    if not isinstance(rows,list) or not 1<=len(rows)<=8:raise AgentOutputError('invalid workflow count')
    by_id={d['workflow_id']:d for d in documents};implementation=[];typed=[]
    for index,row in enumerate(rows,1):
        if set(row)!=set(fields) or row['workflow_id'] not in by_id:
            raise AgentOutputError('unknown workflow or invalid media contract fields')
        choice=by_id[row['workflow_id']]
        dependencies=row['depends_on']
        if any(type(dep)!=int or not 1<=dep<index for dep in dependencies):
            raise AgentOutputError('dependency must refer to an earlier selected workflow')
        if row['input_type'] not in row['input_types']:raise AgentOutputError('input_types omits primary input')
        if ('mask' in row['input_types'] and 'mask' not in input_types
                and not any(rows[dep-1]['output_type']=='mask' for dep in dependencies)):
            raise AgentOutputError(f'stage {index} requires a mask from supplied media or an earlier declared dependency')
        if not row['instruction'].strip():raise AgentOutputError('empty workflow instruction')
        stage={'step':str(index),'description':row['instruction'],
               **{k:row[k] for k in fields if k not in {'workflow_id','instruction','depends_on'}},
               'depends_on':[str(dep) for dep in dependencies],
               'selection_origin':'joint complete-request workflow selection; no independent decomposition'}
        implementation.append({**{k:choice[k] for k in ['workflow_id','workflow_path','base_path','modality','operation','task_family','selected_profile']},
            'decomposition_step':stage,'score':choice.get('similarity'), 'reasons':[raw['reason']],
            'capabilities':[choice['operation']],'priorities':[],'constraints':[]})
        typed.append({'stage_id':f'stage_{index:02d}','purpose':row['instruction'],'instruction':row['instruction'],
            'input_type':row['input_type'].upper(),'input_types':[t.upper() for t in row['input_types']],
            'output_type':row['output_type'].upper(),'modality':choice['modality'],
            'depends_on':[f'stage_{dep:02d}' for dep in dependencies]})
    if len(typed)>1:validate_composition_plan(typed)
    last=implementation[-1];operations=tuple(x['operation'] for x in implementation)
    decision=AgentDecision(last['task_family'],operations,operations,(),(),None,'medium',reason=raw['reason'])
    spec=RequirementSpec(modality,last['task_family'],operations,operations,(),(),None)
    return decision,spec,implementation,raw
