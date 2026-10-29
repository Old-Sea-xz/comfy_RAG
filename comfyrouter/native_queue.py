"""FIFO admission to a bounded native ComfyUI prompt queue."""
import fcntl
import errno
import json
import os
from pathlib import Path
import time
import urllib.error
from uuid import uuid4
from .closed_loop import ClosedLoopError, ComfyUITransportError


def reconcile_accepted_prompt(api, payload, request):
    """Reattach a dead observer to its proven submission without a second POST."""
    manifest = os.getenv('COMFYROUTER_SUBMISSION_RECONCILIATION_FILE')
    if not manifest:
        return None
    destination = os.getenv('COMFYROUTER_STAGE_QUEUE_DESTINATION', '')
    row = json.loads(Path(manifest).read_text()).get(destination)
    if row is None:
        return None
    if (row['case_id'] != os.getenv('COMFYROUTER_STAGE_CASE_ID')
            or row['operation'] != os.getenv('COMFYROUTER_STAGE_OPERATION')):
        raise ClosedLoopError('accepted prompt checkpoint identity mismatch')
    if row['graph'] != payload.get('prompt'):
        raise ClosedLoopError('accepted prompt checkpoint graph mismatch; refusing duplicate submission')
    prompt_id = row['prompt_id']
    history = request(api + '/history/' + prompt_id).get(prompt_id)
    queued = None
    if history is None:
        queue = request(api + '/queue')
        queued = next((x for x in queue.get('queue_running', []) + queue.get('queue_pending', [])
                       if x[1] == prompt_id), None)
    graph = history['prompt'][2] if history is not None else queued[2] if queued else None
    if graph != row['graph']:
        raise ClosedLoopError('accepted prompt evidence missing or changed; refusing duplicate submission')
    capability = request(api + '/queue_cleanup')
    if capability.get('boot_id') != row['boot_id']:
        raise ClosedLoopError('accepted prompt server identity changed')
    queue_dir = Path(os.environ['COMFYROUTER_STAGE_QUEUE_DIR'])
    queue_dir.mkdir(parents=True, exist_ok=True)
    ticket = queue_dir / f'{time.time_ns():020d}_{os.getpid()}.json'
    state = {key: value for key, value in row.items() if key != 'graph'}
    state.update(pid=os.getpid(), status='submitted', observer_recovered_at=time.time())
    ticket.write_text(json.dumps(state) + '\n')
    return {'prompt_id': prompt_id, 'queue_boot_id': row['boot_id'],
            'observer_recovered': True}, ticket


def submit_prompt(api, payload, request):
    recovered = reconcile_accepted_prompt(api, payload, request)
    if recovered is not None:
        return recovered
    queue_dir = Path(os.environ["COMFYROUTER_STAGE_QUEUE_DIR"])
    queue_dir.mkdir(parents=True, exist_ok=True)
    ticket = queue_dir / f"{time.time_ns():020d}_{os.getpid()}.json"
    state = {"case_id": os.environ.get("COMFYROUTER_STAGE_CASE_ID", ""),
             "operation": os.environ.get("COMFYROUTER_STAGE_OPERATION", ""),
             "pid": os.getpid(), "status": "waiting", "queued_at": time.time(),
             "destination": os.environ.get("COMFYROUTER_STAGE_QUEUE_DESTINATION", "")}
    payload = dict(payload)
    payload.setdefault('prompt_id', str(uuid4()))
    payload['extra_data'] = {**payload.get('extra_data', {}), 'release_memory_between_prompts': True}
    state['requested_prompt_id'] = payload['prompt_id']
    recovery_limit = float(os.getenv('COMFYROUTER_SUBMISSION_RECOVERY_SECONDS', '300'))
    if recovery_limit <= 0:
        raise ValueError('submission recovery timeout must be positive')
    outage_started = None
    uncertain = False
    request_path = ''
    temp = ticket.with_suffix('.tmp')
    temp.write_text(json.dumps(state) + "\n")
    temp.replace(ticket)
    try:
        with (queue_dir / "native_submit.lock").open('a+') as lock:
            while True:
                with_admission = False
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    waiting = []
                    for path in sorted(queue_dir.glob('*.json')):
                        try:
                            row = json.loads(path.read_text())
                        except FileNotFoundError:
                            # Completed workers remove their tickets outside
                            # this admission lock, including after glob().
                            continue
                        try:
                            stat = Path(f"/proc/{row['pid']}/stat").read_text().rsplit(')', 1)[1].split()[0]
                        except FileNotFoundError:
                            stat = 'X'
                        if stat in {'Z', 'X'}:
                            abandoned = queue_dir / 'abandoned'
                            abandoned.mkdir(exist_ok=True)
                            try:
                                path.replace(abandoned / path.name)
                            except FileNotFoundError:
                                pass  # The owner finished cleanup first.
                        elif row['status'] == 'waiting':
                            waiting.append(path)
                    if waiting and waiting[0] == ticket:
                        request_path = '/queue_cleanup'
                        capability = request(api + '/queue_cleanup')
                        if capability.get('supported') is not True:
                            raise ClosedLoopError('ComfyUI does not support memory release between native queued prompts')
                        if capability.get('phase') in {'restart_required', 'restarting'}:
                            state['waiting_for_restart'] = True
                            outage_started = outage_started if outage_started is not None else time.monotonic()
                            temp.write_text(json.dumps(state) + '\n')
                            temp.replace(ticket)
                            raise ComfyUITransportError('ComfyUI is restarting; wait for its queue API to recover')
                        state.pop('waiting_for_restart', None)
                        request_path = '/queue'
                        queue = request(api + '/queue')
                        if uncertain:
                            # A lost POST response does not establish rejection.
                            # Locate the same UUID; never blindly resubmit it.
                            found = any(row[1] == payload['prompt_id'] for row in
                                        queue.get('queue_running', []) + queue.get('queue_pending', []))
                            request_path = '/history/' + payload['prompt_id']
                            history = request(api + request_path) if not found else {}
                            if found or payload['prompt_id'] in history:
                                response = {'prompt_id': payload['prompt_id'], 'submission_recovered': True}
                                with_admission = True
                        depth = len(queue.get('queue_running', [])) + len(queue.get('queue_pending', []))
                        window = int(os.environ.get('COMFYROUTER_NATIVE_QUEUE_WINDOW', '3'))
                        if window < 1:
                            raise ValueError('native queue window must be positive')
                        if not uncertain and depth < window:
                            request_path = '/prompt'
                            response = request(api + '/prompt', payload)
                            if not isinstance(response.get('prompt_id'), str):
                                raise ClosedLoopError(f'/prompt rejected graph: {response}')
                            with_admission = True
                        if with_admission:
                            state.update(status='submitted', prompt_id=response['prompt_id'], submitted_at=time.time())
                            response['queue_boot_id'] = capability.get('boot_id')
                            state['boot_id'] = response['queue_boot_id']
                            if state.get('transport_errors'):
                                response['submission_transport'] = state['transport_errors']
                            temp.write_text(json.dumps(state) + '\n')
                            temp.replace(ticket)
                        if not uncertain:
                            outage_started = None
                except (OSError, urllib.error.URLError, TimeoutError, ComfyUITransportError) as exc:
                    if isinstance(exc, urllib.error.HTTPError):
                        raise
                    # ECONNREFUSED occurs before the connection can send a POST.
                    # Other POST failures may have happened after acceptance.
                    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
                    refused = isinstance(reason, OSError) and reason.errno == errno.ECONNREFUSED
                    if request_path == '/prompt' and not refused:
                        uncertain = True
                    outage_started = outage_started if outage_started is not None else time.monotonic()
                    errors = state.setdefault('transport_errors', [])
                    errors.append({'time': time.time(), 'path': request_path, 'error': str(exc),
                                   'submission_uncertain': uncertain})
                    state['transport_errors'] = errors[-20:]
                    temp.write_text(json.dumps(state) + '\n')
                    temp.replace(ticket)
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                if with_admission:
                    return response, ticket
                if outage_started is not None and time.monotonic() - outage_started >= recovery_limit:
                    raise ComfyUITransportError(
                        'ComfyUI submission recovery timed out; retain the workflow and retry after service recovery',
                        prompt_id=payload['prompt_id'] if uncertain else '', submission_uncertain=uncertain,
                        audit={'recovery_seconds': recovery_limit, 'transport_errors': state.get('transport_errors', [])})
                time.sleep(0.5)
    except BaseException:
        ticket.unlink(missing_ok=True)
        raise
