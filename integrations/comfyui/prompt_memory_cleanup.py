"""Local memory release between queued prompts."""
import ctypes
import gc
import logging
import os
import time

import psutil

_allocator = ctypes.CDLL(None)


class MemoryReleaseTimeout(RuntimeError):
    """A process restart is needed to release memory held outside node caches."""


def release_prompt_memory(executor, model_management, memory_management, state):
    limit = int(float(os.getenv("COMFYROUTER_RELEASE_RSS_LIMIT_GIB", "10")) * 1024 ** 3)
    started = time.time()
    deadline = time.monotonic() + float(os.getenv("COMFYROUTER_RELEASE_TIMEOUT_SECONDS", "30"))
    state.pop('completed_at', None)
    state.update(phase="releasing", started_at=started)
    while True:
        memory_management.set_ram_cache_release_state(None, 0)
        executor.reset()
        model_management.unload_all_models()
        gc.collect()
        model_management.soft_empty_cache(force=True)
        _allocator.malloc_trim(0)
        rss = psutil.Process().memory_info().rss
        state.update(rss_bytes=rss, rss_limit_bytes=limit)
        if rss <= limit:
            state.update(phase="released", completed_at=time.time())
            logging.info("Queued prompt memory released: %.2f GiB", rss / 1024 ** 3)
            return dict(state)
        # Pending jobs stay in the queue while this gate is closed.
        state["phase"] = "waiting_for_memory"
        if time.monotonic() >= deadline:
            state['phase'] = 'restart_required'
            raise MemoryReleaseTimeout(f'RSS remains {rss} bytes above {limit} after cleanup')
        logging.warning("Queue paused for memory release: %.2f GiB > %.2f GiB", rss / 1024 ** 3, limit / 1024 ** 3)
        time.sleep(5)
