"""Preserve queued graphs and finished history across a local memory restart."""
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import psutil


def restart_with_queue_checkpoint(queue, server, item_id, history_result, status):
    # Admission is blocked by the queue mutex until exec replaces this process.
    with queue.mutex:
        running = list(queue.currently_running.values())
        pending = list(queue.queue)
        history = dict(queue.history)
        if history_result is not None:
            prompt = queue.currently_running[item_id]
            history[prompt[1]] = {'prompt': list(prompt[:5]), 'status': status,
                                  **history_result}
        else:
            pending.extend(running)
        checkpoint = {'boot_id': server.queue_cleanup_state['boot_id'],
                      'pending': pending, 'history': history, 'number': server.number}
        fd, name = tempfile.mkstemp(prefix='comfyui-queue-recovery-', suffix='.json')
        with os.fdopen(fd, 'w') as file:
            json.dump(checkpoint, file)
            file.flush()
            os.fsync(file.fileno())
        os.environ['COMFYROUTER_QUEUE_RECOVERY_FILE'] = name
        logging.warning('Restarting ComfyUI for memory release; preserving %d pending prompts and finished results', len(pending))
        os.execv(sys.executable, [sys.executable, *sys.argv])


def restore_queue_checkpoint(server):
    name = os.environ.pop('COMFYROUTER_QUEUE_RECOVERY_FILE', '')
    if not name:
        return
    path = Path(name)
    checkpoint = json.loads(path.read_text())
    # Restore before the worker starts and before HTTP admissions are accepted.
    server.queue_cleanup_state.update(boot_id=checkpoint['boot_id'], phase='recovered')
    rss = psutil.Process().memory_info().rss
    limit = int(float(os.getenv('COMFYROUTER_RELEASE_RSS_LIMIT_GIB', '10')) * 1024 ** 3)
    if rss > limit:
        raise RuntimeError(f'Restored ComfyUI baseline RSS {rss} exceeds {limit}; queue remains paused')
    for record in checkpoint['history'].values():
        cleanup = record.get('queue_memory_cleanup', {})
        if cleanup.get('after', {}).get('status') == 'restart_checkpoint':
            cleanup['after'] = {'status': 'released_after_restart', 'phase': 'released',
                               'rss_bytes': rss, 'rss_limit_bytes': limit,
                               'before_restart': cleanup['after']}
    server.prompt_queue.history.update(checkpoint['history'])
    for item in sorted(checkpoint['pending'], key=lambda item: item[0]):
        # HTTP /queue deliberately omits the sensitive sixth field. Local
        # recovery of those sanitized rows restores an empty sensitive map.
        if len(item) == 5:
            item = [*item, {}]
        server.prompt_queue.put(tuple(item))
    server.number = checkpoint['number']
    server.queue_cleanup_state['recovery_checkpoint'] = str(path)
    logging.info('Restored %d pending prompts after memory restart', len(checkpoint['pending']))
