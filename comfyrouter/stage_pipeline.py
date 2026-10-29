"""Durable stage state and fair admission to the one shared model API lock.

Case coroutines retain execute_case ownership. Admission slots belong to stages,
so evaluation and repair do not prevent fresh tasks preparing for the GPU.
"""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
from uuid import uuid4
LANES = ('route', 'prepare', 'prepare', 'evaluate', 'repair')

def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.{uuid4().hex}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False) + '\n')
    temp.replace(path)

def alive(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[0] not in {'Z', 'X'}
    except (OSError, ValueError, TypeError, IndexError):
        return False

def root():
    value = os.getenv('COMFYROUTER_STAGE_PIPELINE_DIR', '').strip()
    return Path(value) if value else None

def state_path():
    directory = root()
    key = os.getenv('COMFYROUTER_STAGE_PIPELINE_KEY', '')
    return directory / 'states' / f'{key}.json' if directory and key else None

def transition(phase, **details):
    path = state_path()
    if path is None:
        return
    previous = read(path, {})
    previous.update(key=os.getenv('COMFYROUTER_STAGE_PIPELINE_KEY'), pid=os.getpid(), phase=phase, updated_at=time.time(), **details)
    write(path, previous)

def infer_lane(agent_id=''):
    if agent_id == 'direct_semantic_selection_agent':
        return 'route'
    if agent_id == 'visual_evaluation_agent':
        return 'evaluate'
    path = os.getenv('COMFYROUTER_STAGE_PIPELINE_REPORT', '')
    report = read(path, {}) if path else {}
    if report.get('attempts') or report.get('parameter_failures') or report.get('service_failures'):
        return 'repair'
    return 'prepare'

def request_phase(lane, agent_id=''):
    return {'route': 'routing', 'prepare': 'preparing', 'evaluate': 'evaluating', 'repair': 'repairing'}[lane]

def phase_counts(workers, directory):
    """Count live owners only; stale states from a prior PID are not phases."""
    counts = {phase: 0 for phase in ('routing', 'preparing', 'evaluating', 'repairing', 'starting')}
    for worker in workers:
        if not alive(worker.get('pid')):
            continue
        state = read(Path(directory) / 'states' / f'{worker['method']}_{worker['key']}.json', {})
        phase = state.get('phase') if state.get('pid') == worker['pid'] else 'starting'
        if phase in counts:
            counts[phase] += 1
    return counts

def choose_ticket(tickets, cursor):
    """Weighted round robin between lanes, FIFO inside each lane."""
    for offset in range(len(LANES)):
        index = (cursor + offset) % len(LANES)
        options = [(path, row) for path, row in tickets if row.get('lane') == LANES[index]]
        if options:
            return (min(options, key=lambda pair: pair[0].name)[0], (index + 1) % len(LANES))
    return (None, cursor)

@contextmanager
def api_admission(handle, lane, agent_id=''):
    """Select a lane ticket and acquire the real API lock atomically.

    Old workers and other experiments can still use that same flock directly.
    A ticket never consumes the real API lock while waiting for its turn.
    """
    directory = root()
    directory.mkdir(parents=True, exist_ok=True)
    queue = directory / 'api_queue'
    queue.mkdir(exist_ok=True)
    ticket = queue / f'{time.time_ns():020d}_{os.getpid()}_{uuid4().hex}.json'
    queued = time.time()
    phase = request_phase(lane, agent_id)
    row = {'pid': os.getpid(), 'key': os.getenv('COMFYROUTER_STAGE_PIPELINE_KEY', ''), 'lane': lane, 'queued_at': queued, 'status': 'waiting'}
    write(ticket, row)
    transition(phase, api_status='waiting', api_queued_at=queued, api_lane=lane)
    acquired = False
    try:
        with (directory / 'admission.lock').open('a+') as admission:
            while not acquired:
                fcntl.flock(admission, fcntl.LOCK_EX)
                try:
                    tickets = []
                    for path in sorted(queue.glob('*.json')):
                        state = read(path, {})
                        if not alive(state.get('pid')):
                            path.unlink(missing_ok=True)
                        elif state.get('status') == 'waiting':
                            tickets.append((path, state))
                    cursor = read(directory / 'fair_cursor.json', {}).get('cursor', 0)
                    selected, next_cursor = choose_ticket(tickets, cursor)
                    if selected == ticket:
                        try:
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            if time.time() - queued >= 1.0:
                                fcntl.flock(handle, fcntl.LOCK_EX)
                                acquired = True
                        else:
                            acquired = True
                        if acquired:
                            write(directory / 'fair_cursor.json', {'cursor': next_cursor})
                            row.update(status='active', acquired_at=time.time())
                            write(ticket, row)
                            transition(phase, api_status='active', api_acquired_at=row['acquired_at'], api_wait_seconds=row['acquired_at'] - queued)
                finally:
                    fcntl.flock(admission, fcntl.LOCK_UN)
                if not acquired:
                    time.sleep(0.2)
            yield
    finally:
        ticket.unlink(missing_ok=True)
        if acquired:
            transition(phase, api_status='complete', api_completed_at=time.time())
            fcntl.flock(handle, fcntl.LOCK_UN)

def preparation_count(workers, directory):
    """Count only live preparation stages, never GPU/evaluation/repair waiters."""
    count = 0
    for worker in workers:
        if not alive(worker.get('pid')):
            continue
        state = read(Path(directory) / 'states' / f'{worker['method']}_{worker['key']}.json', {})
        if state.get('pid') != worker['pid'] or state.get('phase') in {'starting', 'routing', 'preparing'}:
            count += 1
    return count
