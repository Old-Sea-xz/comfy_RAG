import ast
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from comfyrouter.native_queue import submit_prompt
from comfyrouter.closed_loop import ClosedLoopError, ComfyUIExecutor


def test_native_admission_bounds_window_and_preserves_ready_order(tmp_path, monkeypatch):
    monkeypatch.setenv('COMFYROUTER_STAGE_QUEUE_DIR', str(tmp_path))
    monkeypatch.setenv('COMFYROUTER_NATIVE_QUEUE_WINDOW', '2')
    pending = []
    submissions = []
    returned = []
    errors = []
    max_depth = 0
    guard = threading.Lock()

    def request(url, payload=None):
        nonlocal max_depth
        with guard:
            if url.endswith('/queue_cleanup'):
                return {'supported': True}
            if url.endswith('/queue'):
                return {'queue_running': [], 'queue_pending': list(pending)}
            assert url.endswith('/prompt')
            assert payload['extra_data']['release_memory_between_prompts'] is True
            case = payload['prompt']['case']
            pending.append(case)
            submissions.append(case)
            max_depth = max(max_depth, len(pending))
            return {'prompt_id': case}

    def worker(case):
        try:
            returned.append(submit_prompt('local', {'prompt': {'case': case}}, request))
        except BaseException as exc:
            errors.append(exc)

    # Fill the window, then establish the order of blocked requests.
    for case in ['a', 'b']:
        worker(case)
    threads = []
    for index, case in enumerate(['c', 'd']):
        thread = threading.Thread(target=worker, args=(case,), daemon=True)
        thread.start()
        threads.append(thread)
        deadline = time.time() + 5
        while len(list(tmp_path.glob('*.json'))) < 3 + index and time.time() < deadline:
            time.sleep(.01)
        assert len(list(tmp_path.glob('*.json'))) == 3 + index
    assert submissions == ['a', 'b']
    with guard:
        pending.pop(0)
    deadline = time.time() + 5
    while len(submissions) < 3 and time.time() < deadline:
        time.sleep(.01)
    assert submissions == ['a', 'b', 'c']
    with guard:
        pending.pop(0)
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert not errors
    assert submissions == ['a', 'b', 'c', 'd']
    assert max_depth == 2
    for response, ticket in returned:
        assert ticket.exists()
        ticket.unlink()


def test_native_admission_refuses_server_without_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv('COMFYROUTER_STAGE_QUEUE_DIR', str(tmp_path))
    request = Mock(return_value={'supported': False})
    with pytest.raises(ClosedLoopError, match='memory release'):
        submit_prompt('local', {}, request)
    assert not list(tmp_path.glob('*.json'))
    assert request.call_count == 1


@pytest.mark.parametrize('success', [True, False])
def test_server_cleanup_preserves_original_result_and_precedes_task_done(success):
    # Exercise the real server worker function without starting a GPU server.
    main = Path(__file__).resolve().parents[1] / 'integrations/comfyui/prompt_worker.py'
    tree = ast.parse(main.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'prompt_worker')
    class StopWorker(Exception):
        pass
    events = []
    result = {'outputs': {'6': {'videos': [{'filename': 'result.mp4'}]}}}
    executor = SimpleNamespace(success=success, status_messages=[('actual_status', {})], history_result=result)
    def execute(*args):
        executor.success = success
        executor.status_messages = [('actual_status', {})]
        events.append('execute')
    executor.execute = Mock(side_effect=execute)
    def release(executor, *args):
        events.append('release')
        executor.success = True
        executor.status_messages = []
        return {'phase': 'released', 'rss_bytes': 5, 'rss_limit_bytes': 10}
    def task_done(item_id, history, status, **kwargs):
        events.append('task_done')
        assert status.status_str == ('success' if success else 'error')
        assert status.completed is success
        assert status.messages == [('actual_status', {})]
        assert history['outputs'] == result['outputs']
        assert history['queue_memory_cleanup']['after']['phase'] == 'released'
    queue = Mock()
    item = (0, 'id', {}, {'release_memory_between_prompts': True}, [], {})
    queue.get.side_effect = [(item, 1), StopWorker()]
    queue.task_done.side_effect = task_done
    queue.get_flags.return_value = {}
    namespace = {
        'args': SimpleNamespace(cache_classic=True, cache_none=False, cache_lru=0),
        'execution': SimpleNamespace(CacheType=SimpleNamespace(RAM_PRESSURE=1, CLASSIC=2),
            PromptExecutor=lambda *a, **kw: executor,
            PromptQueue=SimpleNamespace(ExecutionStatus=lambda **kw: SimpleNamespace(**kw))),
        'comfy': SimpleNamespace(model_management=SimpleNamespace(soft_empty_cache=Mock()), memory_management=object()),
        'time': time, 'gc': SimpleNamespace(collect=Mock()), 'logging': SimpleNamespace(info=Mock()),
        'asset_seeder': SimpleNamespace(pause=Mock(), is_disabled=lambda: True, resume=Mock()),
        'hook_breaker_ac10a0': SimpleNamespace(restore_functions=Mock()),
        'release_prompt_memory': release,
        'MemoryReleaseTimeout': RuntimeError,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(main), 'exec'), namespace)
    server = SimpleNamespace(last_prompt_id=None, client_id=None, queue_cleanup_state={})
    with pytest.raises(StopWorker):
        namespace['prompt_worker'](queue, server)
    assert events == ['release', 'execute', 'release', 'task_done']


def test_memory_gate_waits_until_rss_is_below_limit(monkeypatch):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'integrations/comfyui/prompt_memory_cleanup.py'
    spec = importlib.util.spec_from_file_location('prompt_memory_cleanup_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('COMFYROUTER_RELEASE_RSS_LIMIT_GIB', '10')
    process = Mock()
    process.memory_info.side_effect = [SimpleNamespace(rss=11 * 1024 ** 3), SimpleNamespace(rss=5 * 1024 ** 3)]
    monkeypatch.setattr(module.psutil, 'Process', lambda: process)
    monkeypatch.setattr(module, '_allocator', SimpleNamespace(malloc_trim=Mock()))
    monkeypatch.setattr(module.gc, 'collect', Mock())
    sleeper = Mock()
    monkeypatch.setattr(module.time, 'sleep', sleeper)
    executor, models, memory = Mock(), Mock(), Mock()
    state = {'prompt_id': 'id'}
    result = module.release_prompt_memory(executor, models, memory, state)
    assert result['phase'] == 'released'
    assert result['rss_bytes'] == 5 * 1024 ** 3
    assert executor.reset.call_count == models.unload_all_models.call_count == 2
    memory.set_ram_cache_release_state.assert_called_with(None, 0)
    sleeper.assert_called_once_with(5)


def test_native_polling_waits_through_transient_transport_and_cleanup(tmp_path, monkeypatch):
    import comfyrouter.closed_loop as loop
    import comfyrouter.native_queue as admission
    monkeypatch.setenv('COMFYROUTER_COMFYUI_QUEUE_MODE', '1')
    ticket = tmp_path / 'ticket.json'
    ticket.touch()
    monkeypatch.setattr(admission, 'submit_prompt', lambda *args: ({'prompt_id': 'id', 'queue_boot_id': 'boot'}, ticket))
    executor = ComfyUIExecutor(output_root=tmp_path, poll_seconds=0)
    executor.static_validate = Mock()
    executor._record_vram = Mock()
    (tmp_path / 'result.png').write_bytes(b'output')
    polls = 0
    def request(url, *args):
        nonlocal polls
        if '/history/' in url:
            polls += 1
            if polls == 1:
                raise TimeoutError('temporary local API timeout')
            if polls >= 4:
                return {'id': {'status': {'status_str': 'success'}, 'outputs': {
                    '1': {'images': [{'filename': 'result.png', 'type': 'output'}]}}}}
            return {}
        if url.endswith('/queue'):
            return {'queue_running': [[0, 'id']], 'queue_pending': []}
        assert url.endswith('/queue_cleanup')
        return {'boot_id': 'boot', 'prompt_id': 'id', 'phase': 'releasing' if polls == 2 else 'executing'}
    monkeypatch.setattr(loop, '_json_request', request)
    evidence = executor.execute({})
    assert evidence.status == 'success'
    assert evidence.resource['poll_transport_errors']
    assert executor._record_vram.call_count == 1
    assert not ticket.exists()


def test_native_polling_reports_restart_instead_of_waiting_forever(tmp_path, monkeypatch):
    import comfyrouter.closed_loop as loop
    import comfyrouter.native_queue as admission
    monkeypatch.setenv('COMFYROUTER_COMFYUI_QUEUE_MODE', '1')
    ticket = tmp_path / 'ticket.json'
    ticket.touch()
    monkeypatch.setattr(admission, 'submit_prompt', lambda *args: ({'prompt_id': 'id', 'queue_boot_id': 'old'}, ticket))
    def request(url, *args):
        return {'boot_id': 'new'} if url.endswith('/queue_cleanup') else {}
    monkeypatch.setattr(loop, '_json_request', request)
    executor = ComfyUIExecutor(poll_seconds=0)
    executor.static_validate = Mock()
    assert executor.execute({}).status == 'server_restarted'
    assert not ticket.exists()
