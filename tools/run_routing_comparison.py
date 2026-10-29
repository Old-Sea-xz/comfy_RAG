"""Shared RAG configuration, checkpoints and request policy helpers."""
import copy, hashlib, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.getenv('COMFYROUTER_PYTHON', sys.executable))
sys.path.insert(0,str(ROOT))

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')
    temp.replace(path)

def execution_checkpoint(route):
    """Use the persisted executable stages as the single/compound authority."""
    checkpoint=copy.deepcopy(route)
    operations=[s['operation'] for s in checkpoint['implementation_plan']]
    checkpoint['classification']['subworkflow_sequence']=operations
    checkpoint['requirement_spec']['operations']=operations
    return checkpoint

def configure(run):
    protocol=read(run/'protocol.json')
    reconciliation=run/'worker_reconciliation'/'accepted_prompts.json'
    if reconciliation.is_file():
        os.environ['COMFYROUTER_SUBMISSION_RECONCILIATION_FILE']=str(reconciliation)
    else:
        os.environ.pop('COMFYROUTER_SUBMISSION_RECONCILIATION_FILE',None)
    os.environ.update(COMFYROUTER_DISABLE_SELF_EVOLUTION='1',
        COMFYROUTER_AGENT_CONTRACTS=str(run/'frozen/agent_skills.json'),
        COMFYROUTER_EVOLUTION_DIR=str(run/'runtime_state'),
        COMFYROUTER_LLM_RETRIES=str(protocol.get('llm_transport_retries',0)),
        COMFYROUTER_SERIAL_API='1' if protocol.get('serial_api',False) else '0',
        COMFYROUTER_VIDEO_FRAME_BUDGET=str(protocol.get('video_frame_budget',100)), COMFYROUTER_COMFYUI_QUEUE_MODE='0',
        COMFYROUTER_RELEASE_RSS_LIMIT_GIB='10')
    if protocol.get('stage_pipeline'):
        os.environ['COMFYROUTER_STAGE_PIPELINE_DIR']=str(run/'stage_pipeline')
        os.environ['COMFYROUTER_SKIP_TERMINAL_DIAGNOSIS']='1'
    if protocol.get('stage_fifo_queue'):
        os.environ['COMFYROUTER_STAGE_QUEUE_DIR']=str(run/'stage_queue')
    if protocol.get('native_comfyui_pending_queue'):
        os.environ['COMFYROUTER_COMFYUI_QUEUE_MODE']='1'
        os.environ['COMFYROUTER_NATIVE_QUEUE_WINDOW']=str(protocol.get('native_queue_window',3))
    os.environ['COMFYROUTER_QUALITY_POLICY']=protocol.get('quality_policy','')
    os.environ['COMFYROUTER_EXECUTION_TIMEOUT_SECONDS']=str(protocol.get('execution_timeout_seconds',3600))
    os.environ['COMFYROUTER_EXECUTION_LIMIT_POLICY_FILE']=str(run/'protocol.json')
    if protocol.get('serial_api',False):
        os.environ['COMFYROUTER_API_SERIAL_LOCK']=protocol.get('api_serial_lock','/tmp/comfyrouter-api.serial.lock')
        os.environ['COMFYROUTER_API_LOCK_AUDIT']=str(run/'api_lock_events.jsonl')
    # The experiment uses zero to mean no truncation, not a zero-frame cap.
    if float(protocol.get('video_frame_budget',100))<=0:
        os.environ.pop('COMFYROUTER_VIDEO_FRAME_BUDGET',None)
    from comfyrouter import config
    config.load_env()
    registry=run/'frozen/node_registry.json'
    if registry.is_file():
        os.environ['COMFYROUTER_NODE_REGISTRY']=str(registry)
        config.NODE_REGISTRY=registry
    if protocol.get('evaluation_model'):
        os.environ['COMFYROUTER_LLM_EVALUATION_MODEL']=protocol['evaluation_model']
        os.environ['COMFYROUTER_EVALUATION_API_STYLE']=protocol.get('evaluation_api_style','chat')
        if protocol.get('evaluation_uses_primary_key'):
            os.environ['COMFYROUTER_LLM_EVALUATION_API_KEY']=os.environ['COMFYROUTER_LLM_API_KEY']
    from comfyrouter.network import configure_external_proxy
    configure_external_proxy()

def verify_frozen(run):
    protocol=read(run/'protocol.json')
    for relative,digest in protocol['frozen_files'].items():
        if hashlib.sha256((run/'frozen'/relative).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('frozen library changed: '+relative)

def claim(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('x') as handle:
            handle.write(json.dumps({'pid':os.getpid(),'started_at':time.time()})+'\n')
        return True
    except FileExistsError:
        return False

def client_type():
    from comfyrouter.agentic import OpenAICompatibleClient, AgentOutputError
    class OneShotClient(OpenAICompatibleClient):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.retries=0;self.calls=set()
        def _complete_json(self,agent_id,contract,payload):
            if agent_id not in {'generation_agent',
                                'direct_semantic_selection_agent',
                                'visual_evaluation_agent'}:
                raise AgentOutputError('one-shot policy: only routing, parameter filling and final evaluation are allowed')
            if payload.get('_json_retry') or payload.get('contract_correction') or payload.get('correction'):
                raise AgentOutputError('one-shot policy: response correction is prohibited')
            # Repeated calls for different composition stages are distinct shots.
            impl=payload.get('implementation_plan') or []
            identity=[agent_id,payload.get('stage'),payload.get('request'),payload.get('instruction'),payload.get('operation'),
                      [(s.get('operation'),s.get('decomposition_step',{}).get('step'))
                       for s in impl if isinstance(s,dict)]]
            key=json.dumps(identity,sort_keys=True,default=str)
            if key in self.calls and not (agent_id=='generation_agent' and
                    os.getenv('COMFYROUTER_COMPARISON_ITERATIVE')=='1'):
                raise AgentOutputError('one-shot policy: second agent decision is prohibited')
            self.calls.add(key)
            return super()._complete_json(agent_id,contract,payload)
    return OneShotClient
