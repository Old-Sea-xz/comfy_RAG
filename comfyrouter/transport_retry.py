"""Persistent per-case transport backoff, independent of quality iterations."""
import json
import os
import re
from pathlib import Path


def retryable_api_failure(error, error_type=None):
    """Do not turn invalid HTTP requests or context limits into endless retries."""
    message = str(error)
    statuses = re.findall(r"(?:\bHTTP[ /:]*(?:status[ :=]*)?|(?:Client|Server) error ['\"]?)([45]\d\d)\b", message, re.I)
    if statuses:
        return all(int(status) in {408, 429, 500, 502, 503, 504, 524} for status in statuses)
    if any(code in message for code in ('context_length_exceeded', 'invalid_request_error',
                                       'max_output_tokens', 'content_filter')):
        return False
    return error_type == 'AgentTransportError' or any(code in message for code in (
        'server_error', 'internal_error', 'service_unavailable', 'overloaded_error',
        'rate_limit_exceeded', 'SSE stream ended', 'timed out', 'Connection refused'))


def state_path(run, method, key):
    return Path(run) / 'transport_retries' / f'{method}_{key}.json'


def read_state(run, method, key):
    path = state_path(run, method, key)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_state(run, method, key, record):
    path = state_path(run, method, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def ready_job_index(run, jobs, now):
    outage = Path(run) / 'transport_retries' / 'provider_backoff.json'
    if outage.exists() and json.loads(outage.read_text()).get('next_retry_at', 0) > now:
        return None
    for index, (method, key) in enumerate(jobs):
        state = read_state(run, method, key)
        if state.get('status') == 'parked':
            continue
        if state.get('status') != 'waiting' or state.get('next_retry_at', 0) <= now:
            return index
    return None


def mark_started(run, method, key):
    state = read_state(run, method, key)
    if state:
        state['status'] = 'running'
        write_state(run, method, key, state)
