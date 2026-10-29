"""Cross-process serialization for external model/API calls.

The ComfyBench rerun pipeline may keep more than one case worker alive so a
later case can prepare its request while an earlier case is executing in
ComfyUI.  This gate keeps the external Agent/evaluator API traffic strictly
one request at a time and enforces a small gap between logical requests.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def serial_api_enabled() -> bool:
    return os.getenv("COMFYROUTER_SERIAL_API", "").casefold() in {
        "1", "true", "yes", "on",
    }


def _interval_seconds() -> float:
    try:
        value = float(os.getenv("COMFYROUTER_API_INTERVAL_SECONDS", "1"))
    except ValueError:
        value = 1.0
    return max(0.0, value)


def _lock_path() -> Path:
    return Path(os.getenv(
        "COMFYROUTER_API_SERIAL_LOCK",
        "/tmp/comfyrouter-api.serial.lock",
    ))


def _audit_lock_event(event: str, lock_path: Path) -> None:
    audit = os.getenv('COMFYROUTER_API_LOCK_AUDIT', '').strip()
    if not audit:
        return
    path = Path(audit)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'event': event, 'pid': os.getpid(),
                                 'time': time.time(), 'lock': str(lock_path)}) + '\n')


@contextmanager
def serialized_api_request(*, lane: str | None = None, agent_id: str = "") -> Iterator[None]:
    """Serialize one logical external API request across worker processes."""
    if not serial_api_enabled():
        yield
        return

    lock_path = _lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        from .stage_pipeline import root as pipeline_root, api_admission, infer_lane
        admission = api_admission(handle, lane or infer_lane(agent_id), agent_id) if pipeline_root() else None
        if admission is not None:
            admission.__enter__()
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            raw = handle.read().strip()
            try:
                state = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                state = {}
            last_finished = state.get("last_finished_at")
            try:
                remaining = _interval_seconds() - (time.time() - float(last_finished))
            except (TypeError, ValueError):
                remaining = 0.0
            if remaining > 0:
                time.sleep(remaining)
            _audit_lock_event('acquired', lock_path)
            yield
        finally:
            _audit_lock_event('released', lock_path)
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({
                "last_finished_at": time.time(),
                "pid": os.getpid(),
            }) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if admission is not None:
                admission.__exit__(None, None, None)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
