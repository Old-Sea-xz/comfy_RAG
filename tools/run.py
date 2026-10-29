"""Run or resume a prepared RAG batch under an exclusive batch lock."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    from comfyrouter import config
    config.load_env()
    os.environ.setdefault('COMFYROUTER_BASE_ONLY_ABLATION', '0')
    from tools.run_routing_comparison import read, write, verify_frozen
    from tools.run_unified_md_comparison import coordinate_queued
    with (run / 'batch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_frozen(run)
        protocol = read(run / 'protocol.json')
        os.environ['COMFYROUTER_COMFYUI_LOCK'] = protocol['gpu_serial_lock']
        state = {'pid': os.getpid(), 'status': 'running', 'started_at': time.time()}
        write(run / 'active.json', state)
        try:
            coordinate_queued(run)
            state['status'] = 'finished' if read(run / 'progress.json').get('complete') else 'needs_review'
        except BaseException as exc:
            state.update(status='needs_review', error=str(exc))
            raise
        finally:
            state['updated_at'] = time.time()
            write(run / 'active.json', state)


if __name__ == '__main__':
    main()
