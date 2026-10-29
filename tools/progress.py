"""Compact progress monitor for this fresh RAG ComfyBench200 run."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = None
LABELS = {
    'quality_not_passed': '质量不通过', 'routing_failed': '路由失败',
    'setup_failed': '配置失败', 'repair_preflight_failed': '修复预检失败',
    'execution_failed': '执行失败', 'parameter_generation_failed': '参数生成失败',
    'evaluation_unavailable': '评价失败', 'infrastructure_failed': '基础设施失败',
    'execution_worker_failed': '执行进程失败',
    'agent_transport_failed': '通信失败', 'agent_provider_failed': '接口请求失败',
}


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def alive(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[0] not in {'Z', 'X'}
    except (OSError, ValueError, TypeError, IndexError):
        return False


def snapshot():
    protocol = read(ROOT / 'protocol.json')
    cases = protocol['case_ids']
    outcomes = [read(ROOT / 'execution/semantic' / (cid + '_v0') / 'outcome.json')
                for cid in cases]
    retries = [read(ROOT / 'transport_retries' / f'semantic_{cid}_v0.json') for cid in cases]
    parked = sum(row.get('status') == 'parked' and not outcome.get('resolved')
                 for row, outcome in zip(retries, outcomes))
    review_cases = set(protocol.get('checkpoint_review_cases', []))
    parked += sum(cid in review_cases and not outcome and retry.get('status') != 'parked'
                  for cid, outcome, retry in zip(cases, outcomes, retries))
    outcomes = [{} if retry.get('status') == 'parked' and not outcome.get('resolved') else outcome
                for retry, outcome in zip(retries, outcomes)]
    retrying = sum(row.get('status') in {'waiting', 'running'} and not outcome
                   for row, outcome in zip(retries, outcomes))
    completed = sum(bool(row) for row in outcomes)
    passed = sum(bool(row.get('resolved')) for row in outcomes)
    failures = Counter(LABELS.get(row.get('status', 'unknown'), row.get('status', 'unknown'))
                       for row in outcomes if row and not row.get('resolved'))
    queue = read(ROOT / 'QUEUED_EXECUTION.json')
    workers = [row for row in queue.get('active_workers', []) if alive(row.get('pid'))]
    if completed == len(cases):
        current = '全部完成'
    else:
        phases = Counter()
        for worker in workers:
            state = read(ROOT / 'stage_pipeline/states' / f"semantic_{worker['key']}.json")
            phases[state.get('phase') if state.get('pid') == worker['pid'] else 'starting'] += 1
        gpu = 'GPU空闲'
        pending = '?'
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(protocol.get('comfyui_api','http://127.0.0.1:8188').rstrip('/')+'/queue', timeout=3) as response:
                remote = json.load(response)
            pending = len(remote.get('queue_pending', []))
            running = remote.get('queue_running', [])
            if running:
                ticket_roots = [ROOT / 'stage_queue']
                predecessor = protocol.get('start_after_run')
                if predecessor:
                    ticket_roots.append(Path(predecessor) / 'stage_queue')
                tickets = [read(path) for root in ticket_roots for path in root.rglob('*.json')]
                ticket = next((row for row in tickets if row.get('prompt_id') == running[0][1]), {})
                cid = ticket.get('case_id', '')
                is_predecessor = bool(cid and not any(
                    row.get('prompt_id') == running[0][1]
                    for row in [read(path) for path in (ROOT / 'stage_queue').rglob('*.json')]))
                gpu = ('GPU前序 gen_' if is_predecessor else 'GPU gen_') + str(cid).removeprefix('gen_') if cid else 'GPU其他任务'
        except (OSError, ValueError):
            gpu = 'Comfy状态暂不可读'
        active_state = read(ROOT / 'active.json')
        status = ('运行中' if workers or alive(active_state.get('pid')) else
                  '等待前序完成' if active_state.get('status') == 'waiting_for_predecessor' else '暂停待恢复')
        current = (f"{status} | {gpu} | Comfy排队 {pending} | RAG路由 {phases['routing']}"
                   f" | 生成准备 {phases['preparing']} | 评价/修复 {phases['evaluating'] + phases['repairing']}"
                   f" | GPU等待 {phases['gpu']}"
                   f" | 活跃worker {len(workers)} | 通信重试 {retrying} | 待恢复 {parked}")
    detail = '、'.join(f'{name} {count}' for name, count in failures.items())
    return (f'当前：{current} | 总体进度：{completed}/{len(cases)}（{completed/len(cases):.0%}）'
            f' | 通过 {passed} | 未通过 {completed-passed}' + (f'（{detail}）' if detail else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.run.resolve()
    try:
        while True:
            print(snapshot(), flush=True)
            if args.once:
                break
            time.sleep(10)
    except KeyboardInterrupt:
        sys.exit(0)
