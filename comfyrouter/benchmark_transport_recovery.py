"""Resume verified LLM/evaluator checkpoints without resetting quality rounds."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

from comfyrouter.transport_retry import read_state, write_state


def read(path, default=None):
    if not path or not Path(path).exists():
        return default
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def verify_artifacts(report, row):
    root = Path(report['trace']['case_output']).resolve()
    for stage in row.get('stages') or [row]:
        if stage.get('execution', {}).get('status') not in {'success', 'reused'}:
            continue
        ref = stage.get('primary_output') or next(iter(stage.get('outputs') or []), None)
        if not ref:
            raise ValueError('successful stage has no checkpoint artifact')
        artifact = Path(ref)
        if not artifact.is_file() or not artifact.resolve().is_relative_to(root):
            raise ValueError('checkpoint artifact missing or outside case output')
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if stage.get('artifact_sha256') and stage['artifact_sha256'] != digest:
            raise ValueError('checkpoint artifact changed')
        stage.update(primary_output=str(artifact), artifact_sha256=digest)


def checkpoint_mode(report):
    attempts = report.get('attempts', [])
    limit = int(report.get('max_attempts', 3))
    if attempts and report.get('status') == 'evaluation_unavailable':
        row = attempts[-1]
        if row.get('passed') is not None or row.get('comfybench', {}).get('status') != 'evaluator_error':
            raise ValueError('evaluation recovery lacks an unevaluated round')
        if not all(s.get('execution', {}).get('status') in {'success', 'reused'}
                   for s in row.get('stages') or [row]):
            raise ValueError('evaluation recovery cannot replay an unsuccessful GPU stage')
        verify_artifacts(report, row)
        saved = copy.deepcopy(row)
        saved.update(comfyui_attempt_consumed=False, logical_iteration_consumed=False,
                     reason='LLM evaluation transport retry; reuse the same generated round')
        report.setdefault('evaluation_transport_checkpoints', []).append(copy.deepcopy(row))
        attempts.pop()
        report['service_checkpoint'] = saved
        report['status'] = 'execution_service_unavailable'
        return 'service'
    for field, status in [('parameter_checkpoint', 'execution_parameter_failed'),
                          ('service_checkpoint', 'execution_service_unavailable')]:
        row = report.get(field, {})
        if (row and row.get('attempt') == len(attempts) + 1
                and row.get('comfyui_attempt_consumed') is False and len(attempts) < limit):
            verify_artifacts(report, row)
            report['status'] = status
            return 'service'
    if not attempts:
        return 'preparation'
    last = attempts[-1]
    if (len(attempts) < limit and last.get('passed') is False
            and (last.get('comfybench', {}).get('comfybench', {}).get('resolved') is False
                 or last.get('comfybench', {}).get('status') == 'stage_execution_failed')):
        verify_artifacts(report, last)
        return 'quality'
    raise ValueError('no resumable checkpoint within the original quality budget')


def recover(run, method, key):
    run = Path(run)
    cid = key.removesuffix('_v0')
    out = run / 'outcomes' / (cid + '.json')
    outcome = read(out, {})
    retryable = (outcome.get('status') in {'agent_transport_failed', 'evaluation_unavailable',
                                         'execution_service_unavailable'}
                 or outcome.get('status') == 'infrastructure_failed'
                 and outcome.get('error_type') == 'AgentTransportError')
    if outcome.get('completed') or not retryable:
        return None
    policy = read(run / 'protocol.json', {}).get('transport_retry_policy', {})
    if not policy.get('enabled'):
        return None
    report_path = Path(outcome['report']).resolve() if outcome.get('report') else None
    report = read(report_path, {})
    if outcome.get('status') == 'execution_service_unavailable':
        checkpoint = report.get('service_checkpoint', {})
        stages = checkpoint.get('stages') or [checkpoint]
        failures = [s.get('execution', {}) for s in stages
                    if s.get('execution', {}).get('status') not in {'success', 'reused'}]
        # Only retry a verified, unsubmitted stage. Accepted or uncertain
        # prompts require reconciliation, never an automatic second POST.
        if (not checkpoint or checkpoint.get('comfyui_attempt_consumed') is not False
                or checkpoint.get('attempt') != len(report.get('attempts', [])) + 1
                or len(report.get('attempts', [])) >= int(report.get('max_attempts', 3))
                or not failures or any(
                    e.get('status') != 'execution_service_unavailable' or e.get('prompt_id')
                    or e.get('submission_uncertain')
                    or any(t.get('submission_uncertain') for t in
                           e.get('resource', {}).get('submission_transport', {}).get('transport_errors', []))
                    for e in failures)):
            return None
    attempts = report.get('attempts', [])
    limit = int(report.get('max_attempts', 3))
    if (report.get('failure_stage') == 'composition_failure_diagnosis'
            and len(attempts) == limit and limit > 0
            and [a.get('attempt') for a in attempts] == list(range(1, limit + 1))
            and all(a.get('passed') is False for a in attempts)
            and (attempts[-1].get('comfybench', {}).get('comfybench', {}).get('resolved') is False
                 or attempts[-1].get('comfybench', {}).get('status') == 'stage_execution_failed')):
        archive = run / 'transport_retries' / 'archive' / cid / str(time.time_ns())
        archive.mkdir(parents=True)
        shutil.copy2(out, archive / 'outcome.json')
        shutil.copy2(report_path, archive / 'run_report.json')
        report.setdefault('terminal_transport_errors', []).append({
            'failure_stage': report.get('failure_stage'), 'error': report.get('error'),
            'archive': str(archive), 'recorded_at': time.time()})
        attempts[-1]['terminal_diagnosis'] = 'skipped_no_remaining_execution_budget'
        report['status'] = 'running'
        from comfyrouter.case_execution import final_failure_status
        report['status'] = final_failure_status(report)
        report.pop('failure_stage', None)
        report.pop('error', None)
        report['completed_attempts'] = len(attempts)
        write(report_path, report)
        outcome.update(status=report['status'], completed=True, task_pass=False, finished_at=time.time())
        write(out, outcome)
        record = {'case_id': cid, 'status': 'finished_budget_exhausted', 'mode': None,
                  'archive': str(archive), 'normal_attempt_budget_reset': False,
                  'diagnosis_retried': False, 'gpu_regenerated': False}
        write_state(run, method, key, record)
        return record
    state = read_state(run, method, key)
    if state.get('status') == 'parked':
        return {**state, 'mode': None}
    count = int(state.get('recovery_count', 0))
    record = {'case_id': cid, 'method': method, 'source_status': outcome['status'],
              'recovery_count': count, 'normal_attempt_budget_reset': False,
              'recorded_at': time.time()}
    if count >= int(policy.get('max_recoveries_per_case', 5)):
        record.update(status='parked', reason='transport recovery limit reached', mode=None)
        write_state(run, method, key, record)
        return record
    try:
        mode = checkpoint_mode(report) if report else 'preparation'
    except ValueError as exc:
        record.update(status='parked', reason=str(exc), mode=None)
        write_state(run, method, key, record)
        return record
    archive = run / 'transport_retries' / 'archive' / cid / str(time.time_ns())
    archive.mkdir(parents=True)
    shutil.copy2(out, archive / 'outcome.json')
    if report_path:
        shutil.copy2(report_path, archive / 'run_report.json')
    # Replace only this dead worker's recovery markers. Its route and outputs
    # stay in place, and no GPU prompt is interrupted or blindly resubmitted.
    for folder in ['media_failure_recovery', 'planning_failure_recovery', 'quality_transport_recovery']:
        marker = run / folder / (cid + '.json')
        if marker.exists():
            shutil.copy2(marker, archive / (folder + '.json'))
            marker.unlink()
    if report_path:
        write(report_path, report)
        folder = {'service': 'media_failure_recovery', 'preparation': 'planning_failure_recovery',
                  'quality': 'quality_transport_recovery'}[mode]
        write(run / folder / (cid + '.json'), {
            'case_id': cid, 'report': str(report_path), 'archive': str(archive),
            'reason': 'automatic service/LLM transport checkpoint recovery',
            'reboot_preparation': mode == 'preparation',
            'normal_attempts_retained': len(report.get('attempts', [])),
            'normal_attempt_budget_reset': False})
    delay = min(float(policy.get('initial_backoff_seconds', 60)) * 2 ** count,
                float(policy.get('max_backoff_seconds', 600)))
    record.update(status='waiting', recovery_count=count + 1, mode=mode,
                  next_retry_at=time.time() + delay, backoff_seconds=delay, archive=str(archive))
    write_state(run, method, key, record)
    protocol = read(run / 'protocol.json')
    field = {'service': 'service_resume_cases', 'preparation': 'preparation_resume_cases',
             'quality': 'quality_resume_cases'}[mode]
    # Clear stale modes so the scheduler chooses the current checkpoint flag.
    for name in ['service_resume_cases', 'preparation_resume_cases', 'quality_resume_cases']:
        protocol[name] = [c for c in protocol.get(name, []) if c != cid]
    protocol[field].append(cid)
    write(run / 'protocol.json', protocol)
    out.unlink()
    return record
