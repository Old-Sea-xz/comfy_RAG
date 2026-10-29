"""Run the original ComfyBench visual rubric with exactly one HTTP request."""
import json, os, sys, time
from pathlib import Path
from types import ModuleType


def read_chat_evaluation_response(response,deadline,*,line_logger=None):
    """Require a completed chat response before accepting its judgment tags."""
    if 'text/event-stream' not in response.headers.get('content-type','').lower():
        response.read();data=response.json();choice=data['choices'][0]
        if choice.get('finish_reason')!='stop':raise RuntimeError('incomplete chat evaluator response')
        return choice['message']['content'],data.get('usage')
    parts=[];usage=None;finished=False
    for line in response.iter_lines():
        if line_logger is not None:line_logger(line)
        if time.monotonic()>deadline:raise TimeoutError('chat evaluation deadline exceeded')
        if not line.startswith('data:'):continue
        data=line[5:].strip()
        if data=='[DONE]':break
        chunk=json.loads(data)
        if chunk.get('error'):raise RuntimeError('chat evaluator stream returned provider error')
        usage=chunk.get('usage') or usage
        for choice in chunk.get('choices',[]):
            value=choice.get('delta',{}).get('content')
            if isinstance(value,str):parts.append(value)
            reason=choice.get('finish_reason')
            if reason and reason!='stop':raise RuntimeError('incomplete chat evaluator: '+str(reason))
            if reason=='stop':finished=True
    if not finished:raise RuntimeError('chat evaluation stream ended without completion')
    return ''.join(parts),usage


def evaluate(run,method,cid,result,attempt_dir):
    from tools.run_routing_comparison import read,write,client_type
    from tools.evaluate_comfybench_case import (validate_vision_evaluation_text,
                                               instruction_with_evaluation_criteria)
    bench=Path(os.getenv('COMFYROUTER_COMFYBENCH_ROOT', str(Path(__file__).resolve().parents[2]/'ComfyBench')));sys.path.insert(0,str(bench))
    os.chdir(read(run/'protocol.json').get('evaluation_workdir',str(bench)))
    calls=[]
    def invoke(messages):
        from comfyrouter.api_serial import serialized_api_request
        with serialized_api_request(lane="evaluate"):
            return invoke_locked(messages)

    def invoke_locked(messages):
        if calls:raise RuntimeError('second evaluation call prohibited')
        calls.append(True)
        client=client_type()()
        model=client.evaluation_model
        content=[]
        from comfyrouter.quality_policy import POLICY_ID, POLICY_TEXT
        if read(run/'protocol.json').get('quality_policy') == POLICY_ID:
            messages = [{**messages[0], 'content': [*messages[0]['content'],
                {'type': 'text', 'text': POLICY_TEXT}]}]
        for item in messages[0]['content']:
            if item['type']=='text':content.append({'type':'input_text','text':item['text']})
            elif item['type']=='image_url':content.append({'type':'input_image','image_url':item['image_url']['url']})
        payload={'model':model,'temperature':0,'input':[{'role':'user','content':content}],
                 'max_output_tokens':4096,'stream':True}
        style=os.getenv('COMFYROUTER_EVALUATION_API_STYLE','responses')
        if style=='chat':
            payload={'model':model,'temperature':0,'messages':messages,'max_tokens':4096,'stream':True}
        if not model.casefold().startswith('gemini-'):
            if style=='chat':payload['reasoning_effort']='low'
            else:payload['reasoning']={'effort':'low'}
        headers={'Content-Type':'application/json','Accept':'text/event-stream','User-Agent':client.user_agent}
        if client.evaluation_api_key:headers['Authorization']='Bearer '+client.evaluation_api_key
        write(attempt_dir/'vision_request_summary.json',{'model':model,'calls':1,
            'text':[c['text'] for c in content if c['type']=='input_text'],
            'image_count':sum(c['type']=='input_image' for c in content),'expected_route_sent':False,
            'rubric':'unmodified official ComfyBench modality prompt','api_style':style})
        deadline=time.monotonic()+180
        endpoint=client.base_url+('/chat/completions' if style=='chat' else '/responses')
        if style=='chat':
            import httpx
            request=client.http_client.build_request('POST',endpoint,headers=headers,
                content=json.dumps(payload).encode(),timeout=httpx.Timeout(connect=15,read=180,write=120,pool=15))
            response=client.http_client.send(request,stream=True)
        else:
            response=client._send_httpx(body=json.dumps(payload).encode(),headers=headers,timeout_seconds=180)
        response_lines=[]
        try:
            if response.status_code>=400:
                response.read()
                try:detail=response.json().get('error',{})
                except Exception:detail={'message':'non-JSON error response'}
                safe=json.dumps(detail,ensure_ascii=False)
                for key in [client.api_key,client.evaluation_api_key]:
                    if key:safe=safe.replace(key,'[redacted]')
                write(attempt_dir/'vision_http_error.json',{'status':response.status_code,'model':model,
                      'endpoint':endpoint,'error':safe})
            response.raise_for_status()
            if style=='chat':
                text,usage=read_chat_evaluation_response(response,deadline,line_logger=response_lines.append)
                raw={'usage':usage}
            else:
                raw=client._read_httpx_sse_completed_response(response,deadline=deadline,
                                                              line_logger=response_lines.append)
                text=client._response_output_text(raw)
        finally:
            # Preserve every HTTP response, including HTML gateway failures;
            # repeated evaluator requests must not overwrite the only evidence.
            if response_lines:
                response_body='\n'.join(response_lines)
            else:
                try:response_body=response.content.decode('utf-8',errors='replace')
                except Exception:response_body=''
            record={'http_status_code':response.status_code,'endpoint':endpoint,
                    'model':model,'response_headers':dict(response.headers),
                    'response_body_text':response_body}
            for key in [client.api_key,client.evaluation_api_key]:
                if key:
                    record=json.loads(json.dumps(record).replace(key,'[redacted]'))
            write(attempt_dir/f'vision_http_response_{time.time_ns()}.json',record)
            response.close()
        validate_vision_evaluation_text(text)
        write(attempt_dir/'vision_response.json',{'model':model,'usage':raw.get('usage'),'text':text})
        return text,raw.get('usage')
    model_module=ModuleType('utils.model');model_module.invoke_vision=invoke;sys.modules['utils.model']=model_module
    comfy_module=ModuleType('utils.comfy');comfy_module.execute_prompt=lambda *_:None;sys.modules['utils.comfy']=comfy_module
    from script import evaluation
    evaluation.invoke_vision=invoke
    trial_key=cid;task_key=cid if '_v' in cid else cid+'_v0'
    requested=read(run/'task_inputs.json')[task_key];cid=requested['case_id']
    task=read(run/'source_instructions.json')[cid].copy();task['instruction']=requested['instruction']
    baseline=read(run/'baseline'/cid/'evaluation.json')
    original=task['instruction'];criteria=baseline.get('evaluation_criteria',[])
    task['instruction']=instruction_with_evaluation_criteria(original,criteria);task['result']=str(result.resolve())
    function=getattr(evaluation,'evaluate_'+task['modality'])
    analysis,judgment=function(task);resolved=judgment.strip()=='True'
    payload={'schema_version':'comfybench-case-evaluation-1','case_id':cid,
        'category':task['category'],'modality':task['modality'],'instruction':original,
        'evaluation_criteria':criteria,'result':str(result),
        'quality_policy':read(run/'protocol.json').get('quality_policy','official_comfybench'),
        'comfybench':{'passed':result.is_file(),'resolved':resolved,'analysis':analysis,'judgment':judgment},
        'system_evaluation':None,'evaluation_calls':len(calls)}
    write(attempt_dir/'comfybench_evaluation.json',payload)
    quality={'resolved':resolved,'judgment':judgment,'analysis':analysis,
          'evaluator_model':client_type()().evaluation_model,'rubric':read(run/'protocol.json').get('quality_policy','official_comfybench'),'evaluation_calls':len(calls)}
    response=payload
    if (run/'protocol.json').exists() and read(run/'protocol.json').get('immediate_final_media_check',False):
        from tools.audit_final_media_contract import decoded_kind
        expected='video' if requested['modality'].endswith('v') else 'image'
        try:actual,error=decoded_kind(result),None
        except Exception as exc:actual,error='unreadable',str(exc)
        contract={'expected':expected,'actual':actual,'passed':expected==actual,
                  'artifact':str(result),'error':error,'raw_evaluator_resolved':resolved}
        write(run/'execution'/method/trial_key/'final_media_contract.json',contract)
        quality['final_media_contract']=contract
        if not contract['passed']:
            quality.update(resolved=False,judgment='False',raw_evaluator_resolved=resolved,
                           analysis=analysis+'\nFinal media contract failed: expected '+expected+', received '+actual)
            # Preserve the official raw record above; execution sees the corrected decision.
            import copy
            response=copy.deepcopy(payload)
            response['comfybench'].update(resolved=False,judgment='False',analysis=quality['analysis'])
    write(run/'execution'/method/trial_key/'quality.json',quality)
    return response
