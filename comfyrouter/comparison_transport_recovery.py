"""Recover comparison-run API checkpoints without spending quality rounds."""
import json
from pathlib import Path
import shutil
import time

from .benchmark_transport_recovery import checkpoint_mode
from .transport_retry import read_state, retryable_api_failure, write_state


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def write(path, value):
    from tools.run_routing_comparison import write as atomic_write
    atomic_write(path, value)


def recovery_claims(run, method, key, routing):
    markers=[]
    for directory in (run/'claims').glob('*'):
        if directory.name=='whole_request_routing' and not routing:
            continue
        if not (directory.name in {'whole_request_routing','execution'} or
                directory.name.startswith(('host_interruption_resume','gpt_quality_resume','preparation_restart'))):
            continue
        marker=directory/(method+'_'+key)
        if not marker.is_file():
            continue
        pid=read(marker).get('pid')
        try:
            if Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()[0] not in {'Z','X'}:
                raise ValueError(f'recovery refused: worker {pid} is still alive')
        except (OSError,TypeError):
            pass
        markers.append(marker)
    return markers


def recover(run, method, key):
    run=Path(run)
    dest=run/'execution'/method/key
    out=dest/'outcome.json'
    outcome=read(out)
    policy=read(run/'protocol.json').get('transport_retry_policy',{})
    if not outcome or not policy.get('enabled') or outcome.get('resolved'):
        return None
    route_path=run/'routing'/key/'route.json'
    route_file=read(route_path)
    route=route_file.get('methods',{}).get(method,{})
    report_path=Path(outcome['report']) if outcome.get('report') else dest/'raw'/('case'+key.removesuffix('_v0'))/'run_report.json'
    report=read(report_path)
    routing=route.get('status')!='routed' and outcome.get('status') in {
        'routing_failed','agent_transport_failed','agent_provider_failed'}
    error=route.get('error') if routing else outcome.get('error')
    error_type=route.get('error_type') if routing else outcome.get('error_type')
    if routing:
        retryable=retryable_api_failure(error,error_type)
    else:
        retryable=outcome.get('status') in {'agent_transport_failed','agent_provider_failed','evaluation_unavailable'}
        if outcome.get('status')=='evaluation_unavailable':
            error=(report.get('attempts') or [{}])[-1].get('comfybench',{}).get('error',error)
        retryable=retryable and retryable_api_failure(error,error_type or 'AgentTransportError')
    if not retryable:
        return None
    state=read_state(run,method,key)
    if state.get('status')=='parked':
        return {**state,'mode':None}
    count=int(state.get('recovery_count',0))
    record={'case_id':key.removesuffix('_v0'),'method':method,'source_status':outcome['status'],
            'failure_stage':'routing' if routing else 'execution_or_evaluation',
            'recovery_count':count,'normal_attempt_budget_reset':False,'recorded_at':time.time()}
    if count>=int(policy.get('max_recoveries_per_case',5)):
        record.update(status='parked',reason='transport recovery limit reached',mode=None)
        write_state(run,method,key,record)
        return record
    original_report=read(report_path)
    try:
        markers=recovery_claims(run,method,key,routing)
        if routing and report.get('attempts'):
            raise ValueError('routing recovery cannot discard executed rounds')
        mode='preparation' if routing else checkpoint_mode(report) if report else 'preparation'
    except (ValueError,KeyError) as exc:
        record.update(status='parked',reason=str(exc),mode=None)
        write_state(run,method,key,record)
        return record
    archive=run/'transport_retries/archive'/f'{method}_{key}'/str(time.time_ns())
    archive.mkdir(parents=True)
    shutil.copy2(out,archive/'outcome.json')
    if report_path.is_file():
        write(archive/'run_report.json',original_report)
    if routing:
        # Keep retrieval progress so a network retry repeats only the pending
        # selection, rather than consuming an extra retrieval/quality slot.
        shutil.copy2(route_path,archive/'route.json')
        checkpoint=run/'routing'/key/(method+'_selection_checkpoint.json')
        if not checkpoint.exists() and route.get('retrieval_evidence',{}).get('rounds'):
            write(checkpoint,{'status':'selection_pending','retrieval_evidence':route['retrieval_evidence']})
        import fcntl
        with route_path.with_suffix('.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            route_file=read(route_path)
            route_file['methods'].pop(method,None)
            write(route_path,route_file)
    elif report_path.is_file():
        write(report_path,report)
    for marker in markers:
        shutil.copy2(marker,archive/(marker.parent.name+'.claim.json'))
        marker.unlink()
    delay=min(float(policy.get('initial_backoff_seconds',60))*2**count,
              float(policy.get('max_backoff_seconds',600)))
    record.update(status='waiting',recovery_count=count+1,mode=mode,
                  next_retry_at=time.time()+delay,backoff_seconds=delay,archive=str(archive))
    write_state(run,method,key,record)
    # A persistent per-run cooldown prevents an outage from sweeping through
    # the whole pending set. It never blocks another experiment's API lane.
    write(run/'transport_retries/provider_backoff.json',{
        'next_retry_at':record['next_retry_at'],'source':method+'_'+key,'reason':str(error)[:500]})
    protocol=read(run/'protocol.json')
    cid=key.removesuffix('_v0')
    field={'service':'service_resume_cases','preparation':'preparation_resume_cases','quality':'quality_resume_cases'}[mode]
    for name in ['service_resume_cases','preparation_resume_cases','quality_resume_cases']:
        protocol[name]=[c for c in protocol.get(name,[]) if c!=cid]
    protocol[field].append(cid)
    write(run/'protocol.json',protocol)
    out.unlink()
    return record
