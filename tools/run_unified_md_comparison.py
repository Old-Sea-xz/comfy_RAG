"""Anonymous original-request comparison using the standard execute_case loop.

This module selects inputs, invokes the unified API once and observes records.
It contains no generation, repair or retry loop around execute_case.
"""
import argparse,collections,hashlib,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_routing_comparison import read,write,configure,verify_frozen,claim,execution_checkpoint,PYTHON
from tools.run_anonymous_md_comparison import route_worker,read_route
LABELS={'semantic':'RAG'}

def execution_methods(protocol):
    methods=protocol.get('execution_methods',['semantic'])
    if not methods or len(set(methods))!=len(methods) or any(m not in LABELS for m in methods):
        raise ValueError('execution_methods must select supported comparison methods')
    return methods

def memory_worker_capacity(active, worker_limit=200):
    """Size case workers from available host RAM, keeping room for ComfyUI."""
    try:
        worker_limit = max(1, int(worker_limit))
    except (TypeError, ValueError):
        worker_limit = 1
    try:
        available_kib=next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                           if line.startswith('MemAvailable:'))
        available=available_kib*1024
    except (OSError, StopIteration, ValueError, IndexError):
        return min(len(active), worker_limit)
    pss=[]
    for process,_ in active.values():
        try:
            pss.append(sum(int(line.split()[1]) for line in
                           Path(f'/proc/{process.pid}/smaps_rollup').read_text().splitlines()
                           if line.startswith('Pss:'))*1024)
        except (OSError, ValueError, IndexError):
            continue
    estimate=max(int(1.5*1024**3), sorted(pss)[len(pss)//2] if pss else 0)
    reserve=12*1024**3
    capacity = len(active)+(1 if available>=reserve+estimate else 0)
    return min(capacity, worker_limit)

def complete_attempt(attempt):
    stages=attempt.get('stages') or [attempt]
    for stage in stages:
        if stage.get('execution',{}).get('status') not in ['success','reused']:
            return False
        files=stage.get('outputs',[]) if stage.get('status')=='reused' else [stage.get('primary_output')]
        if not files or not all(p and Path(p).is_file() for p in files):
            return False
    return True

def queued_routing_enabled(protocol):
    return protocol.get('queue_routing',protocol.get('stage_pipeline',False))

def ensure_queued_route(run,method,key,protocol):
    path=run/'routing'/key/'route.json'
    if method in read_route(path)['methods']:return
    if not queued_routing_enabled(protocol):
        raise ValueError('queued execution requires completed routing '+method+'/'+key)
    from comfyrouter.stage_pipeline import transition
    transition('routing',api_status='idle')
    # Routing used to run in a separate process. Keep its trace/cache settings
    # local to this phase when continuing in the same case worker.
    names=('COMFYROUTER_AGENT_AUDIT_DIR','COMFYROUTER_LLM_PROMPT_CACHE_NAMESPACE',
           'COMFYROUTER_API_LOCK_AUDIT')
    previous={name:os.environ.get(name) for name in names}
    try:
        route_worker(run,method,key)
    finally:
        for name,value in previous.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value
    if method not in read_route(path)['methods']:
        raise RuntimeError('routing incomplete; existing claim requires review '+method+'/'+key)

def select_recovery_mode(report_path,resume,restart_preparation,interrupted):
    """Keep an existing execution checkpoint when a preparation restart is requested."""
    if restart_preparation:
        if not report_path.is_file():
            # A route-only transport checkpoint has never prepared a graph;
            # there is no execution report to reuse yet.
            return False,False
        previous=read(report_path) or {}
        if previous.get('attempts'):
            return True,False
    return resume,restart_preparation

def execute_worker(run,method,key,resume=False,restart_preparation=False,interrupted=False):
    configure(run)
    protocol=read(run/'protocol.json')
    dest=run/'execution'/method/key
    memory_retry=key.removesuffix('_v0') in protocol.get('memory_retry_cases',[])
    case_output_root=dest/'memory_retry' if memory_retry else dest/'raw'
    case_report=case_output_root/('case'+key.removesuffix('_v0'))/'run_report.json'
    os.environ['COMFYROUTER_STAGE_PIPELINE_KEY']=method+'_'+key
    os.environ['COMFYROUTER_STAGE_PIPELINE_REPORT']=str(case_report)
    ensure_queued_route(run,method,key,protocol)
    report_path=case_report
    resume,restart_preparation=select_recovery_mode(
        report_path,resume,restart_preparation,interrupted)
    claim_group='host_interruption_resume' if interrupted else 'gpt_quality_resume' if resume else 'preparation_restart' if restart_preparation else 'execution'
    if interrupted or restart_preparation:
        previous=read(report_path) if report_path.exists() else {}
        checkpoint=(previous.get('parameter_checkpoint') or previous.get('service_checkpoint')) if interrupted else {
            'run_id':previous.get('run_id'), 'repair_terminal':previous.get('repair_terminal')}
        claim_group += '_'+(hashlib.sha256(json.dumps(checkpoint,sort_keys=True).encode()).hexdigest()[:16]
                           if checkpoint else str(read(run/'protocol.json').get('service_recovery_generation',0)))
    if not claim(run/'claims'/claim_group/(method+'_'+key)):return
    task=read(run/'task_inputs.json')[key];dest=run/'execution'/method/key
    os.environ['COMFYROUTER_STAGE_PIPELINE_KEY']=method+'_'+key
    os.environ['COMFYROUTER_STAGE_PIPELINE_REPORT']=str(case_report)
    from comfyrouter.stage_pipeline import transition
    transition('preparing', api_status='idle')
    route=read(run/'routing'/key/'route.json')['methods'][method]
    if route['status']!='routed':
        write(dest/'outcome.json',{'status':route['status'],'resolved':False,'error':route.get('error'),
                                 'error_type':route.get('error_type'),'failure_stage':'routing',
                                 'retryable':route.get('retryable',False),
                                 'normal_attempts':0,'attempt_count':0,'first_normal_resolved':False})
        transition('finished',outcome=route['status'],api_status='idle')
        return
    from comfyrouter import case_execution as runner
    from comfyrouter.agentic import AgentOrchestrator,OpenAICompatibleClient
    from comfyrouter.execution_contract import CaseExecutionRequest,ExecutionHooks,ExecutionPolicy
    from tools.official_evaluation_once import evaluate
    os.environ['COMFYROUTER_API_LOCK_AUDIT']=str(dest/'api_lock_events.jsonl')
    def implementation_provider(*args,**kwargs):
        return runner.implementation_from_report(execution_checkpoint(route))
    def base_provider(store,spec,implementation,**kwargs):
        source=Path(implementation[0]['workflow_path'])
        if not source.resolve().is_relative_to((run/'frozen').resolve()):
            raise RuntimeError('initial selected workflow outside shared library')
        return read(source)
    def evaluator(case_id,result,attempt_dir):
        requests=[]
        # Transport-only evaluator recovery uses the same artifact. The unified
        # executor owns quality iterations; this is not a generation retry.
        for n in range(1,4):
            try:
                response=evaluate(run,method,key,Path(result),Path(attempt_dir))
                quality=read(dest/'quality.json')
                requests.append({'request':n,'status':'evaluated','artifact':str(result)})
                write(Path(attempt_dir)/'evaluation_requests.json',requests)
                write(Path(attempt_dir)/'quality_guarded.json',quality)
                return response
            except Exception as exc:
                requests.append({'request':n,'status':'unavailable','error':str(exc),'artifact':str(result)})
                write(Path(attempt_dir)/'evaluation_requests.json',requests)
        return {'status':'evaluator_error','error':requests[-1]['error']}
    audit_path=dest/('memory_retry_submission_audit.json' if memory_retry else 'submission_audit.json')
    submitted=read(audit_path) if (resume or interrupted or restart_preparation) and audit_path.exists() else []
    def before_submit(graph):
        reconciliation=os.getenv('COMFYROUTER_SUBMISSION_RECONCILIATION_FILE')
        destination=os.getenv('COMFYROUTER_STAGE_QUEUE_DESTINATION','')
        accepted=read(Path(reconciliation)).get(destination) if reconciliation else None
        if accepted:
            # Result collection is not another GPU submission. Keep the
            # original submission counts intact and audit the observer apart.
            observer_path=dest/'observer_collection_audit.json'
            observations=read(observer_path) if observer_path.exists() else []
            observations.append({'time':time.time(),'destination':destination,
                'prompt_id':accepted['prompt_id'],'pid':os.getpid(),
                'graph_sha256':hashlib.sha256(json.dumps(graph,sort_keys=True).encode()).hexdigest()})
            write(observer_path,observations)
            return
        submitted.append({'submission':len(submitted)+1,'time':time.time(),
            'graph_sha256':hashlib.sha256(json.dumps(graph,sort_keys=True).encode()).hexdigest(),
            'node_count':len(graph),'graph':graph})
        write(audit_path,submitted)
    iteration_budget=route.get('iteration_budget',{})
    max_attempts=min(3,max(1,int(iteration_budget.get('execution_attempts_remaining',3))))
    request=CaseExecutionRequest(case_id=task['case_id'],
        case={'name':task['case_id'],'instruction':task['instruction'],'modality':task['modality'],
              'resource':task['resources'] or None,'category':read(run/'gold.json')[task['case_id']]['category']},
        output_root=case_output_root,max_attempts=max_attempts,
        vram_limit_gib=float(protocol.get('vram_limit_gib',110)),
        resource_roots=(Path(protocol['resource_root']),),
        report_root=run/'runtime_state'/method/'comfybench_runs',
        audit_root=dest/('memory_retry_agent_trace' if memory_retry else 'agent_trace'),
        resume_report=case_report if resume else None,
        interrupted_report=case_report if interrupted else None,
        regenerate_on_resume=resume and task['case_id'] in protocol.get('rubric_regeneration_cases',[]),
        reuse_report=case_report if restart_preparation else None)
    from comfyrouter.agentic_retrieval import RepairRetriever
    repair_retriever = RepairRetriever(run, dest/'repair_rag') if protocol.get('repair_rag') else None
    def agent_factory(store):
        class RAGJsonAgent(AgentOrchestrator):
            def augment_repair_context(self, kwargs):
                return repair_retriever.augment(self.client, kwargs) if repair_retriever else kwargs
            def generate_graph_edit_plan(self,*args,**kwargs):
                kwargs['allow_active_skills']=False
                return super().generate_graph_edit_plan(*args,**kwargs)
        return RAGJsonAgent(OpenAICompatibleClient(),store)
    hooks=ExecutionHooks(evaluator=evaluator,implementation_provider=implementation_provider,
        base_provider=base_provider,agent_factory=agent_factory,
        before_submit=before_submit)
    # Standard diagnosis, notebook, current-graph repair and DAG caching stay
    # inside the common executor. Disable persistent evolution/extra slots;
    # both groups may use normal repair searches and structural repair.
    policy=ExecutionPolicy(name='standard_three_round_anonymous_comparison',frozen_graph=False,
                           disable_self_evolution=True,disable_search=False)
    contract={'entrypoint':'comfyrouter.case_execution.execute_case',
        'max_attempts':request.max_attempts,'iteration_budget':iteration_budget,
        'vram_limit_gib':request.vram_limit_gib,'policy':policy.as_dict(),
        'agent_factory':'standard AgentOrchestrator(OpenAICompatibleClient)',
        'outer_generation_retry_loop':False,'initial_routing_source':'fresh anonymous MD',
        'active_workflow_skill_selection':False,
        'initial_graph_source':'selected frozen workflow_api.json'}
    contract_name=('memory_retry_execution_contract.json' if memory_retry else
                   'continuation_contract.json' if resume or interrupted else 'execution_contract.json')
    write(dest/contract_name,contract)
    try:
        report=runner.execute_case(request,hooks=hooks,policy=policy)
        attempts=report.get('attempts',[]);normal=[]
        for a in attempts:
            if complete_attempt(a):normal.append(a)
        resolved=report.get('status')=='passed' and any(a.get('passed') is True for a in attempts)
        write(dest/'outcome.json',{'status':report.get('status','unknown'),'resolved':resolved,
            'normal_attempts':len(normal),'attempt_count':len(attempts),
            'retrieval_queries':iteration_budget.get('retrieval_queries',1),
            'repair_retrieval_queries':repair_retriever.query_count if repair_retriever else 0,
            'case_iteration_budget':iteration_budget.get('maximum',3),
            'execution_attempt_budget':request.max_attempts,
            'parameter_failures':len(report.get('parameter_failures',[])),
            'evaluation_model':protocol.get('evaluation_model','gemini-3.7-flash'),
            'first_normal_resolved':bool(normal and normal[0].get('passed')),
            'executed_graphs':len(submitted),'error':report.get('error'),
            'report':str(case_report),
            'execution_entrypoint':report.get('execution_entrypoint'),'execution_policy':report.get('execution_policy'),
            'route_reasonable':True if resolved else None,'route_review':'pass implies reasonable; unresolved pending review'})
    except BaseException as exc:
        write(dest/'outcome.json',{'status':'execution_worker_failed','resolved':False,'error':str(exc),
                                 'traceback':traceback.format_exc(),'executed_graphs':len(submitted)})
        raise
    transition('finished', outcome=read(dest/'outcome.json').get('status'), api_status='idle')
    print(json.dumps({'method':method,'key':key,**read(dest/'outcome.json')},ensure_ascii=False),flush=True)

def summarize(run):
    protocol=read(run/'protocol.json');rows=[];metrics={}
    methods=execution_methods(protocol)
    for method in methods:
        for cid in protocol['case_ids']:
            key=cid+'_v0';rp=run/'routing'/key/'route.json';route=read_route(rp)['methods'].get(method,{})
            outp=run/'execution'/method/key/'outcome.json';out=read(outp) if outp.exists() else {'status':'pending','resolved':False}
            from comfyrouter.transport_retry import read_state
            transport=read_state(run,method,key)
            if transport.get('status')=='parked' and not out.get('resolved'):
                out={**out,'status':'pending','transport_status':'parked','transport_error':out.get('error')}
            rows.append({'method':method,'case_id':cid,'composite':read(run/'gold.json')[cid]['composite'],
                         'routing_status':route.get('status','pending'),**out})
    for method in methods:
        group=[x for x in rows if x['method']==method]
        metrics[method]={'planned':len(group),'routing_complete':sum(x['routing_status']!='pending' for x in group),
            'conditions_finished':sum(x['status']!='pending' for x in group),'resolved':sum(x['resolved'] for x in group),
            'first_normal_resolved':sum(bool(x.get('first_normal_resolved')) for x in group),
            'normal_attempts':sum(x.get('normal_attempts',0) for x in group),
            'attempts':sum(x.get('attempt_count',0) for x in group),
            'graph_submissions':sum(x.get('executed_graphs',0) for x in group),
            'unresolved_statuses':dict(collections.Counter(x['status'] for x in group if x['status']!='pending' and not x['resolved']))}
    write(run/'progress.json',{'time':time.time(),'metrics':metrics,'conditions':rows,
                             'complete':all(x['status']!='pending' for x in rows)})
    return rows,metrics

def coordinate(run):
    verify_frozen(run);protocol=read(run/'protocol.json');manifest_path=run/'RUN_MANIFEST.json'
    jobs=[(m,cid+'_v0') for cid in protocol['case_ids'] for m in execution_methods(protocol)]
    if manifest_path.exists():manifest=read(manifest_path)
    else:
        manifest={'status':'pilot','created_at':time.time(),'source_snapshot':str(ROOT),'jobs':jobs,
            'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for d in ['comfyrouter','tools'] for p in (ROOT/d).rglob('*.py')},
            'input_hashes':{n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in
                ['task_inputs.json','gold.json','workflow_catalog.json','candidates.json','workflow_vectors.npy','protocol.json']}}
        write(manifest_path,manifest)
    env=os.environ.copy()
    for n in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy',
              'COMFYROUTER_EXTERNAL_PROXY_URL','COMFYROUTER_LLM_PROXY_URL']:pass  # inherit the operator's proxy settings
    def select(method,key):
        if method in read(run/'routing'/key/'route.json')['methods']:return
        with (run/'logs'/f'route_{method}_{key}.log').open('w') as f:
            code=subprocess.call([str(PYTHON),str(Path(__file__)),'--run',str(run),'--route-worker',method,key],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if method not in read(run/'routing'/key/'route.json')['methods']:raise RuntimeError(f'route missing {method}/{key} exit {code}')
        summarize(run)
    def execute(method,key,phase):
        if (run/'execution'/method/key/'outcome.json').exists():return
        if (run/'claims/execution'/(method+'_'+key)).exists():raise RuntimeError('interrupted worker requires evidence review '+method+'/'+key)
        write(run/'active_condition.json',{'method':method,'key':key,'phase':phase,'started_at':time.time()})
        with (run/'logs'/f'execute_{method}_{key}.log').open('w') as f:
            code=subprocess.call([str(PYTHON),str(Path(__file__)),'--run',str(run),'--execute-worker',method,key],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        path=run/'execution'/method/key/'outcome.json'
        if code or not path.exists() or read(path)['status']=='execution_worker_failed':
            raise RuntimeError(f'execution worker setup/uncaught failure; stop for review {method}/{key}, exit {code}')
        summarize(run);print('condition complete',method,key,flush=True)
        if read(path)['status'] in ['evaluation_unavailable','agent_transport_failed','agent_provider_failed']:
            raise RuntimeError('API infrastructure unavailable; retain artifact and stop for review '+method+'/'+key)
    pilot=[j for cid in protocol['pilot_case_ids'] for j in jobs if j[1]==cid+'_v0']
    try:
        for method,key in pilot:select(method,key);execute(method,key,'pilot')
        pilot_outcomes=[{'method':m,'key':k,**read(run/'execution'/m/k/'outcome.json')} for m,k in pilot]
        passed_runtime=all(x['normal_attempts']>0 or x['status']=='routing_failed' for x in pilot_outcomes) and any(x['normal_attempts']>0 for x in pilot_outcomes)
        write(run/'PILOT_CHECK.json',{'complete':True,'passed_runtime':passed_runtime,'conditions':pilot_outcomes})
        if not passed_runtime:raise RuntimeError('Pilot runtime needs review')
        manifest['status']='routing_remaining';write(manifest_path,manifest)
        remainder=[j for j in jobs if j not in pilot]
        for method,key in remainder:select(method,key)
        manifest['routing_hashes']={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'routing').glob('*/route.json')}
        manifest['status']='executing_remaining';write(manifest_path,manifest)
        random.Random(protocol['execution_seed']).shuffle(remainder)
        for method,key in remainder:execute(method,key,'full_batch')
        manifest['status']='execution_complete_pending_failure_route_review';manifest['completed_at']=time.time();write(manifest_path,manifest)
        summarize(run);write(run/'active_condition.json',{'phase':'complete','completed_at':time.time()})
    except Exception as exc:
        manifest.update(status='needs_review',error=str(exc),stopped_at=time.time());write(manifest_path,manifest);raise

def requeue_unsubmitted_failure(run, method, key):
    """Recover old workers that mistook a pre-submission outage for graph failure."""
    from comfyrouter.case_execution import recover_unsubmitted_service_failure,recover_runtime_parameter_failure
    dest=run/'execution'/method/key
    outcome=read(dest/'outcome.json')
    if outcome.get('status') not in {'repair_preflight_failed','execution_failed','execution_service_unavailable'}:return False
    report_path=(dest/'raw'/('case'+key.removesuffix('_v0'))/'run_report.json').resolve()
    report=read(report_path)
    recovery_mode='service'
    if (report.get('status') == 'execution_service_unavailable' and report.get('setup_service_failure')
            and not report.get('service_checkpoint')):
        attempts=report.get('attempts',[])
        if not attempts:
            recovery_mode='preparation'
        elif (len(attempts)<report.get('max_attempts',3) and attempts[-1].get('passed') is False
              and attempts[-1].get('comfybench',{}).get('comfybench',{}).get('resolved') is False):
            recovery_mode='quality'
        else:
            return False
    elif report.get('status') != 'execution_service_unavailable':
        try:
            recover_runtime_parameter_failure(report)
        except ValueError:
            try:
                recover_unsubmitted_service_failure(report)
            except ValueError:
                return False
    elif (len(report.get('attempts',[])) >= report.get('max_attempts',3)
          or report.get('service_checkpoint',{}).get('comfyui_attempt_consumed') is not False):
        return False
    archive=run/'service_recovery'/key/str(time.time_ns())
    write(archive/'run_report.json',read(report_path))
    write(archive/'outcome.json',outcome)
    audit=dest/'submission_audit.json'
    write(archive/'submission_audit.json',read(audit) if audit.exists() else [])
    write(report_path,report)
    manifest=dest/'raw'/('case'+key.removesuffix('_v0'))/'trace_manifest.json'
    trace=read(manifest);trace.update(status=report['status'],updated_at_unix=time.time());write(manifest,trace)
    (dest/'outcome.json').unlink()
    return recovery_mode


def recover_transport_checkpoint(run, method, key):
    from comfyrouter.comparison_transport_recovery import recover
    return recover(run,method,key)


def coordinate_queued(run, *, worker_script=None):
    """Prepare cases concurrently; their shared stage transactions enter FIFO.

    GPU ownership is per stage, never per whole case. Each worker still invokes
    execute_case once and owns that case's original three-round repair loop.
    """
    verify_frozen(run)
    protocol=read(run/'protocol.json')
    if not protocol.get('stage_fifo_queue'):
        raise ValueError('queued coordinator requires stage_fifo_queue')
    if queued_routing_enabled(protocol) and not (protocol.get('stage_pipeline') and protocol.get('serial_api')):
        raise ValueError('queued routing requires stage_pipeline and serial_api')
    def park_missing_worker(method,key,pid,exit_observation):
        # Accepted GPU prompts may outlive their observer. Keep their claims
        # and artifacts until reconciled; never blindly submit the case again.
        record={'method':method,'key':key,'pid':pid,'exit_observation':exit_observation,
                'reason':'worker disappeared without outcome; reconcile accepted prompts before resuming',
                'recorded_at':time.time()}
        write(run/'worker_reconciliation'/f'{method}_{key}.json',record)
        fresh=read(run/'protocol.json')
        fresh['checkpoint_review_cases']=list(dict.fromkeys(
            fresh.get('checkpoint_review_cases',[])+[key.removesuffix('_v0')]))
        fresh['adopted_workers']=[w for w in fresh.get('adopted_workers',[])
                                  if (w['method'],w['key'])!=(method,key)]
        write(run/'protocol.json',fresh)
        protocol.update(checkpoint_review_cases=fresh['checkpoint_review_cases'],
                        adopted_workers=fresh['adopted_workers'])
    for worker in list(protocol.get('adopted_workers',[])):
        m,k,pid=worker['method'],worker['key'],worker['pid']
        if (run/'execution'/m/k/'outcome.json').exists():continue
        try:
            alive=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0] not in {'Z','X'}
        except FileNotFoundError:
            alive=False
        if not alive:park_missing_worker(m,k,pid,'adopted process missing; original exit status unavailable')
    adopted_keys={(w['method'],w['key']) for w in protocol.get('adopted_workers',[])}
    from comfyrouter.transport_retry import ready_job_index, mark_started
    # A supervisor may have died after a worker wrote its transport outcome.
    # Reconcile that checkpoint before deriving the remaining job list.
    for c in protocol['case_ids']:
        for m in execution_methods(protocol):
            if (m,c+'_v0') not in adopted_keys and (run/'execution'/m/(c+'_v0')/'outcome.json').exists():
                recover_transport_checkpoint(run,m,c+'_v0')
    protocol=read(run/'protocol.json')
    jobs=[(m,c+'_v0') for c in protocol['case_ids'] for m in execution_methods(protocol)
          if (m,c+'_v0') not in adopted_keys and c not in protocol.get('draining_case_ids',[])
          and c not in protocol.get('checkpoint_review_cases',[])
          and not (run/'execution'/m/(c+'_v0')/'outcome.json').exists()]
    random.Random(protocol['execution_seed']).shuffle(jobs)
    recovery_cases=set(protocol.get('service_resume_cases',[])+protocol.get('preparation_resume_cases',[])
                       +protocol.get('quality_resume_cases',[]))
    execution_resumes=set(protocol.get('service_resume_cases',[])+protocol.get('quality_resume_cases',[]))
    memory_retries=set(protocol.get('memory_retry_cases',[]))
    observer_recoveries=set(protocol.get('observer_recovery_cases',[]))
    collection_priority=set(protocol.get('observer_collection_priority_cases',[]))
    jobs.sort(key=lambda job: (-3 if job[1].removesuffix('_v0') in collection_priority else
                              -2 if job[1].removesuffix('_v0') in observer_recoveries else
                              -1 if job[1].removesuffix('_v0') in memory_retries else
                              0 if job[1].removesuffix('_v0') in execution_resumes else
                              1 if job[1].removesuffix('_v0') in recovery_cases else 2))
    for m,k in jobs:
        if not queued_routing_enabled(protocol) and read_route(run/'routing'/k/'route.json')['methods'].get(m) is None:
            raise ValueError('queued execution requires completed routing '+m+'/'+k)
        resumable=protocol.get('service_resume_cases',[])+protocol.get('preparation_resume_cases',[])+protocol.get('quality_resume_cases',[])
        if (run/'claims/execution'/(m+'_'+k)).exists() and k.removesuffix('_v0') not in resumable and (m,k) not in adopted_keys:
            raise ValueError('unfinished claimed case requires checkpoint review '+m+'/'+k)
    manifest_path=run/'RUN_MANIFEST.json'
    manifest=read(manifest_path)
    manifest.update(status='executing_stage_queue',worker_limit=protocol.get('execution_worker_limit',4),
                    routing_schedule='per-case queued' if queued_routing_enabled(protocol) else 'precomputed')
    write(manifest_path,manifest)
    env=os.environ.copy()
    for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy',
                'COMFYROUTER_EXTERNAL_PROXY_URL','COMFYROUTER_LLM_PROXY_URL']:
        pass  # inherit the operator's proxy settings
    active={}
    # Explicit handover keeps already running case loops and GPU tickets intact.
    class AdoptedWorker:
        def __init__(self,pid,outcome):self.pid=pid;self.outcome=outcome
        def poll(self):
            try:
                if Path(f'/proc/{self.pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z':return None
            except FileNotFoundError:pass
            return 0 if self.outcome.exists() else 1
    for worker in protocol.get('adopted_workers',[]):
        m,k,pid=worker['method'],worker['key'],worker['pid']
        if (run/'execution'/m/k/'outcome.json').exists():continue
        try:
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        except FileNotFoundError:
            raise RuntimeError('adopted worker exited without outcome '+str(pid))
        if str(run) not in cmd or not any(f'{flag} {m} {k}' in cmd for flag in
                ['--execute-worker','--interrupted-worker','--resume-worker','--restart-preparation-worker']):
            raise RuntimeError('adopted worker identity mismatch '+str(pid))
        active[(m,k)]=(AdoptedWorker(pid,run/'execution'/m/k/'outcome.json'),
                       (run/'logs'/f'execute_{m}_{k}.log').open('ab'))
    def workers():
        return [{'method':m,'key':k,'pid':p.pid} for (m,k),(p,f) in active.items()]
    def worker_limit():
        return protocol.get('execution_worker_limit',4)
    def refresh_legacy():
        if not protocol.get('stage_pipeline'):return
        from comfyrouter.stage_pipeline import write as write_stage
        legacy_pids=set(protocol.get('pipeline_legacy_worker_pids',[]))
        for worker in workers():
            if worker['pid'] not in legacy_pids:continue
            dest=run/'execution'/worker['method']/worker['key']
            report_path=dest/'raw'/('case'+worker['key'].removesuffix('_v0'))/'run_report.json'
            report=read(report_path) if report_path.exists() else {}
            audit_path=dest/'submission_audit.json'
            submitted=read(audit_path) if audit_path.exists() else []
            phase='repairing' if report.get('attempts') else 'stage_complete' if submitted else 'preparing'
            write_stage(run/'stage_pipeline/states'/f"{worker['method']}_{worker['key']}.json",
                {'key':worker['method']+'_'+worker['key'],'pid':worker['pid'],'phase':phase,
                 'legacy':True,'api_status':'legacy','updated_at':time.time()})
    def can_prepare():
        return len(active)<memory_worker_capacity(active,worker_limit())
    def publish():
        refresh_legacy()
        write(run/'QUEUED_EXECUTION.json',{'updated_at':time.time(),'waiting_cases':len(jobs),
            'active_workers':[{'method':m,'key':k,'pid':p.pid} for (m,k),(p,f) in active.items()],
            'transport_retry_waiting':sum(ready_job_index(run,[job],time.time()) is None for job in jobs),
            'checkpoint_review_cases':protocol.get('checkpoint_review_cases',[]),
            'worker_limit':memory_worker_capacity(active,worker_limit()),
            'gpu_execution':('bounded native FIFO; server cleans between stages' if protocol.get('native_comfyui_pending_queue')
                             else 'FIFO stages; one at a time; cleanup before/after'),
            'api_serial_lock':protocol['api_serial_lock']})
    failed=None
    case_level_failures=[]
    draining=protocol.get('draining_case_ids',[])
    while jobs or active or any(not (run/'execution/direct'/(c+'_v0')/'outcome.json').exists() for c in draining):
        while jobs and not failed and can_prepare():
            ready_index=ready_job_index(run,jobs,time.time())
            if ready_index is None:break
            if not can_prepare():
                # Evaluated/service checkpoints enter repair, so they do not
                # consume a fresh routing/preparation slot.
                resumes=set(protocol.get('quality_resume_cases',[])+protocol.get('service_resume_cases',[]))
                ready_index=next((index for index,job in enumerate(jobs)
                    if job[1].removesuffix('_v0') in resumes
                    and ready_job_index(run,[job],time.time()) is not None),None)
                if ready_index is None:break
            m,k=jobs.pop(ready_index)
            mark_started(run,m,k)
            log=(run/'logs'/f'execute_{m}_{k}.log').open('ab')
            cid=k.removesuffix('_v0')
            flag=('--interrupted-worker' if cid in protocol.get('service_resume_cases',[]) else
                  '--restart-preparation-worker' if cid in protocol.get('preparation_resume_cases',[]) else
                  '--resume-worker' if cid in protocol.get('quality_resume_cases',[]) else '--execute-worker')
            p=subprocess.Popen([str(PYTHON),'-u',str(worker_script or Path(__file__)),'--run',str(run),flag,m,k],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            active[(m,k)]=(p,log)
            if protocol.get('stage_pipeline'):
                from comfyrouter.stage_pipeline import write as write_stage
                write_stage(run/'stage_pipeline/states'/f'{m}_{k}.json',
                    {'key':m+'_'+k,'pid':p.pid,
                     'phase':'repairing' if flag in {'--resume-worker','--interrupted-worker'} else
                         'preparing' if m in read_route(run/'routing'/k/'route.json')['methods'] else
                         'routing',
                     'api_status':'idle','updated_at':time.time()})
            print('case preparing',m,k,'pid',p.pid,flush=True)
        for (m,k),(p,f) in list(active.items()):
            code=p.poll()
            if code is None:continue
            f.close();del active[(m,k)]
            out=run/'execution'/m/k/'outcome.json'
            status=read(out)['status'] if out.exists() else 'missing_outcome'
            if status=='missing_outcome':
                park_missing_worker(m,k,p.pid,code)
                print('worker checkpoint review',m,k,'other cases continue',flush=True)
                summarize(run)
                continue
            transport=recover_transport_checkpoint(run,m,k) if out.exists() else None
            if transport:
                mode=transport.get('mode')
                if mode:
                    recovery_key={'service':'service_resume_cases','preparation':'preparation_resume_cases',
                                  'quality':'quality_resume_cases'}[mode]
                    fresh=read(run/'protocol.json')
                    fresh.setdefault(recovery_key,[]).append(k.removesuffix('_v0'))
                    fresh[recovery_key]=list(dict.fromkeys(fresh[recovery_key]))
                    write(run/'protocol.json',fresh)
                    for name in ['service_resume_cases','preparation_resume_cases','quality_resume_cases']:
                        protocol[name]=fresh.get(name,[])
                    jobs.append((m,k))
                print('transport checkpoint',m,k,transport.get('status'),
                      'retry_at',transport.get('next_retry_at'),flush=True)
                summarize(run)
                continue
            recovery_mode=requeue_unsubmitted_failure(run,m,k) if not code and out.exists() else False
            if recovery_mode:
                recovery_key={'service':'service_resume_cases','preparation':'preparation_resume_cases',
                              'quality':'quality_resume_cases'}[recovery_mode]
                protocol.setdefault(recovery_key,[]).append(k.removesuffix('_v0'))
                fresh=read(run/'protocol.json')
                fresh[recovery_key]=list(dict.fromkeys(protocol[recovery_key]+fresh.get(recovery_key,[])))
                write(run/'protocol.json',fresh)
                jobs.append((m,k))
                print('service recovery queued',m,k,'same workflow and iteration budget',flush=True)
                summarize(run)
                continue
            print('condition complete',m,k,status,flush=True)
            if status in {'evaluation_unavailable','agent_transport_failed','agent_provider_failed',
                          'execution_worker_failed'} or code:
                case_level_failures.append({'method':m,'key':k,'status':status,
                                            'error':read(out).get('error')})
                write(run/'CASE_LEVEL_FAILURES.json',case_level_failures)
            summarize(run)
        publish()
        if failed and not active:break
        time.sleep(1)
    manifest=read(manifest_path)
    from comfyrouter.transport_retry import read_state
    parked=[c for c in protocol['case_ids'] if any(
        read_state(run,m,c+'_v0').get('status') == 'parked'
        for m in execution_methods(protocol))]
    manifest.update(status=('needs_review' if failed else
        'execution_complete_pending_worker_reconciliation' if protocol.get('checkpoint_review_cases') else
        'execution_complete_pending_transport_recovery' if parked else
        'execution_complete_pending_failure_route_review'), parked_transport_cases=parked,
        case_level_failures=case_level_failures)
    if failed:manifest['error']=failed
    else:manifest['completed_at']=time.time()
    write(manifest_path,manifest)
    write(run/'active_condition.json',{'phase':'needs_review' if failed else 'transport_pending' if parked else 'complete'})
    summarize(run)
    if failed:raise RuntimeError(failed)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--route-worker',nargs=2);p.add_argument('--execute-worker',nargs=2);p.add_argument('--resume-worker',nargs=2);p.add_argument('--restart-preparation-worker',nargs=2);p.add_argument('--interrupted-worker',nargs=2)
    a=p.parse_args();run=a.run.resolve()
    if a.interrupted_worker:execute_worker(run,*a.interrupted_worker,interrupted=True)
    elif a.restart_preparation_worker:execute_worker(run,*a.restart_preparation_worker,restart_preparation=True)
    elif a.resume_worker:execute_worker(run,*a.resume_worker,resume=True)
    elif a.route_worker:route_worker(run,*a.route_worker)
    elif a.execute_worker:execute_worker(run,*a.execute_worker)
    elif read(run/'protocol.json').get('stage_fifo_queue'):coordinate_queued(run)
    else:coordinate(run)
