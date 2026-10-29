"""One-shot prompt/parameter filling without an LLM graph-edit interface."""
from __future__ import annotations

from typing import Any


def visual_inputs(paths):
    """Encode images and decoded video frames as actual supported images."""
    import base64,io
    from pathlib import Path
    from PIL import Image
    urls=[]
    def encode(image):
        image=image.convert('RGB');image.thumbnail((768,768))
        buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=90)
        urls.append('data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode())
    for path in map(Path,paths):
        if path.suffix.lower() in {'.mp4','.webm','.mov','.avi','.ogv'}:
            import cv2
            cap=cv2.VideoCapture(str(path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            try:
                for index in sorted({round(max(0,count-1)*t) for t in [0,.25,.5,.75,1]}):
                    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
                    if ok:encode(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
            finally:cap.release()
        else:
            with Image.open(path) as image:encode(image)
    return urls


def parameter_slots(graph: dict[str, Any], node_registry=None, *, freeze_generation_budget=True) -> dict[str, dict]:
    slots = {}
    fixed_names = {'ckpt_name','unet_name','clip_name','clip_name1','clip_name2',
                   'vae_name','model_name','lora_name','filename_prefix'}
    for node_id, node in graph.items():
        class_type = str(node.get('class_type', ''))
        for name, value in node.get('inputs', {}).items():
            # References and file/model bindings are owned by the fixed runner.
            if type(value) not in {str,int,float,bool} or name in fixed_names:
                continue
            if isinstance(value,str) and (
                name in {'image','video','mask','audio','file','path','filename'}
                or value.lower().endswith(('.safetensors','.ckpt','.pth','.pt','.gguf',
                                           '.bin','.png','.jpg','.jpeg','.mp4','.webm'))
            ):
                continue
            slot=f'p{len(slots)+1:04d}'
            slots[slot]={'slot':slot,'node_id':node_id,'class_type':class_type,
                         'parameter':name,'current_value':value,
                         'title':node.get('_meta',{}).get('title','')}
            schema=(node_registry or {}).get(class_type,{}).get('input',{})
            interface=next((schema.get(section,{}).get(name) for section in ['required','optional']
                            if name in schema.get(section,{})),None)
            if interface:
                declared=interface[0]
                if isinstance(declared,list):
                    slots[slot].update(declared_type='ENUM',allowed_values=declared)
                else:
                    slots[slot]['declared_type']=declared
                    if len(interface)>1 and isinstance(interface[1],dict):
                        slots[slot]['limits']={key:interface[1][key] for key in ['min','max'] if key in interface[1]}
            # Only generation/latent dimensions consume a generation budget.
            # Mask rectangle sizes, crop regions and video batch slicing do not.
            budget=(name in {'steps','batch_size','num_frames'}
                    or (name in {'width','height','length'} and
                        (class_type.startswith('Empty') or 'Latent' in class_type)))
            if freeze_generation_budget and budget and type(value) in {int,float}:
                slots[slot]['generation_budget_max']=value
    return slots


def compile_parameters(raw: dict, slots: dict[str, dict]) -> list[dict]:
    from .agentic import AgentOutputError
    if not isinstance(raw,dict) or not {'parameters'} <= set(raw) <= {'parameters','reason'}:
        raise AgentOutputError('parameter filler must return parameters only')
    if not isinstance(raw['parameters'],(list,dict)) or not isinstance(raw.get('reason',''),str):
        raise AgentOutputError('invalid parameter filler response')
    parameters=raw['parameters']
    if isinstance(parameters,dict):
        parameters=[{'slot':slot,'value':value} for slot,value in parameters.items()]
    edits=[];seen=set()
    for item in parameters:
        if not isinstance(item,dict) or set(item)!={'slot','value'}:
            raise AgentOutputError('only slot/value assignments are allowed')
        slot=item['slot']
        if not isinstance(slot,str) or slot not in slots or slot in seen:
            raise AgentOutputError('unknown or repeated parameter slot')
        seen.add(slot);record=slots[slot];value=item['value'];original=record['current_value']
        declared=record.get('declared_type')
        if declared=='FLOAT':
            valid=type(value) in {int,float}
        elif declared=='INT':
            valid=type(value) is int
        elif declared=='BOOLEAN':
            valid=type(value) is bool
        elif declared=='STRING':
            valid=type(value) is str
        elif type(original) is float:
            valid=type(value) in {int,float}
        else:valid=type(value) is type(original)
        if not valid:raise AgentOutputError('parameter type differs from current workflow')
        if 'allowed_values' in record and value not in record['allowed_values']:
            raise AgentOutputError(f"parameter {record['parameter']} must be one of {record['allowed_values']}")
        for limit,bound in record.get('limits',{}).items():
            if type(value) in {int,float} and ((limit=='min' and value<bound) or (limit=='max' and value>bound)):
                raise AgentOutputError(f"parameter {record['parameter']} violates declared {limit} {bound}")
        if 'generation_budget_max' in record and value>record['generation_budget_max']:
            raise AgentOutputError('parameter filler cannot increase the frozen generation budget')
        if value!=original:
            edits.append({'action':'set_param','target':record['node_id'],
                          'key':record['parameter'],'value':value})
    return edits


def parameter_response_schema(slots, *, sparse_object=False):
    import json
    assignments=[]
    groups={}
    properties={}
    for slot,record in slots.items():
        declared=record.get('declared_type')
        if 'allowed_values' in record:
            kind={str:'string',int:'integer',float:'number',bool:'boolean'}[type(record['current_value'])]
            value={'type':kind,'enum':record['allowed_values']}
        else:
            kind={'FLOAT':'number','INT':'integer','STRING':'string','BOOLEAN':'boolean'}.get(declared)
            if kind is None:
                kind={str:'string',int:'integer',float:'number',bool:'boolean'}[type(record['current_value'])]
            value={'type':kind}
            if kind in {'number','integer'}:
                limits=record.get('limits',{})
                if 'min' in limits and abs(limits['min'])<=2**53-1:value['minimum']=limits['min']
                maximum=limits.get('max')
                if 'generation_budget_max' in record:
                    maximum=min(maximum,record['generation_budget_max']) if maximum is not None else record['generation_budget_max']
                if maximum is not None and abs(maximum)<=2**53-1:value['maximum']=maximum
        signature=json.dumps(value,sort_keys=True)
        properties[slot]=value
        if signature not in groups:
            branch={'type':'object','additionalProperties':False,
                    'required':['slot','value'],'properties':{'slot':{'type':'string','enum':[]},'value':value}}
            assignments.append(branch);groups[signature]=branch
        groups[signature]['properties']['slot']['enum'].append(slot)
    return {'name':'existing_parameter_values','schema':{
        'type':'object','additionalProperties':False,'required':['parameters'],
        'properties':{'parameters':(
            {'type':'object','additionalProperties':False,'properties':properties,'required':[]}
            if sparse_object else {'type':'array','items':{'anyOf':assignments}})}}}


def generate_parameters(agent, spec, implementation_plan, graph, *, instruction='',
                        overall_instruction='', visual_context_paths=None, **kwargs):
    if not isinstance(graph,dict) or not graph:
        raise ValueError('parameter filling needs the selected existing graph')
    import json,os
    from pathlib import Path
    registry_path=os.getenv('COMFYROUTER_PARAMETER_REGISTRY','').strip()
    if registry_path:
        registry=json.loads(Path(registry_path).read_text())
    else:
        from .closed_loop import ComfyUIExecutor
        registry=ComfyUIExecutor().ensure_validator().node_registry
    from .generation_output import named_parameter_slots, parameter_prompt_context
    slots=named_parameter_slots(graph,registry,freeze_generation_budget=True)
    if slots:
        raw=agent.client.complete_json('generation_agent',{
            'role':'Fill existing workflow prompts and scalar parameters for one execution.',
            'rules':[
                'Return only changed prompt and parameter values as parameters={name:value}; '
                'use {} when no change is needed. Names come from editable_parameters. '
                'Do not output nodes, edits, actions, reasons or explanations.',
                'Adapt prompt text and necessary scalar values to the current atomic business step.',
                'Preserve the existing workflow. Do not add/remove nodes, change connections, select Skills, '
                'replace models/workflows, decompose again, search, diagnose failures, or propose repairs.',
                'Leave suitable parameters unchanged. Use the declared node input type, allowed_values and limits, '
                'rather than guessing a type from the current default value.',
                'Respect generation_budget_max only where supplied; mask region sizes are not generation dimensions. '
                'Use existing parameters for required geometry and timing. The runner binds input files but '
                'will not add crop, padding, resize or other nodes.',
                'For CR Overlay Text, align is vertical (top/center/bottom), justify is horizontal (left/center/right). '
                'For every other node, follow its supplied allowed_values.',
                'For multi-stage tasks, perform only the current step; other stages own the other changes.',
                'If previous_execution_feedback is provided, use its actual evaluation and previous '
                'parameter assignments to improve the current step with existing scalar parameters only. '
                'A parameter validation error is preparation feedback, not a normal execution attempt.',
            ]}, {
                'instruction':instruction,'overall_instruction':overall_instruction,
                'implementation_plan':implementation_plan,
                'parameter_validation_feedback':kwargs.get('repair_evidence'),
                'previous_execution_feedback':(
                    json.loads(Path(os.environ['COMFYROUTER_COMPARISON_FEEDBACK']).read_text())
                    if os.getenv('COMFYROUTER_COMPARISON_FEEDBACK') else None),
                'editable_parameters':parameter_prompt_context(slots),
                '_image_data_urls':visual_inputs(visual_context_paths or []),
                '_max_output_tokens':8192,
                '_reasoning_effort':'low',
                '_prompt_history_mode':'latest_only',
                '_response_json_schema':parameter_response_schema(slots,sparse_object=True)
            })
        edits=compile_parameters(raw,slots);reason=raw.get('reason','Apply parameter assignments to the selected workflow.')
    else:edits=[];reason='No editable scalar parameters; execute the selected workflow as supplied.'
    return {'edits':edits,'skill_ids':[],'requires_composition_replan':False,
            'repair_transaction':False,'rationale':reason,'parameter_only':True,
            'generation_policy':'node_interface_parameter_slots_v3',
            'editable_parameter_slots':list(slots.values())}
