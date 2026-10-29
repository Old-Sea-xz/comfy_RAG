"""Apply a uniform decoded final-media check while retaining raw evaluator records."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def decoded_kind(path):
    if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        return 'image'
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        decoded, frame = capture.read()
        if capture.isOpened() and decoded and frame is not None and frames > 1 and fps > 0:
            return 'video'
        return 'unknown'
    finally:
        capture.release()


def apply(run):
    tasks = read(run / 'task_inputs.json')
    rows = []
    for ledger_path in sorted((run / 'iteration_runs').glob('*/*/ledger.json')):
        ledger = read(ledger_path)
        original_ledger = json.loads(json.dumps(ledger))
        if not ledger.get('complete'):
            raise RuntimeError('Cannot apply final-media audit to an active condition: ' + str(ledger_path))
        method, key = ledger['method'], ledger['key']
        expected = 'video' if tasks[key]['modality'].endswith('v') else 'image'
        changed = False
        for entry in ledger['candidates']:
            dest = Path(entry['run']) / 'execution' / method / key
            evaluations = list((dest / 'raw').glob('case*/attempt_01/comfybench_evaluation.json'))
            if not evaluations:
                continue
            evaluation = read(evaluations[0])
            artifact = Path(evaluation['result'])
            try:
                actual, error = decoded_kind(artifact), None
            except Exception as exc:
                actual, error = 'unreadable', str(exc)
            passed = actual == expected
            record = dict(method=method, key=key, candidate=entry['candidate'], artifact=str(artifact),
                          expected=expected, actual=actual, passed=passed,
                          raw_evaluator_resolved=evaluation['comfybench']['resolved'], error=error)
            rows.append(record)
            write(dest / 'final_media_contract.json', record)
            if not passed and entry['resolved']:
                if ledger['normal_attempts'] < 3:
                    raise RuntimeError('Corrected false success requires remaining normal iterations: ' + str(ledger_path))
                original = ledger_path.parent / 'ledger_before_media_contract.json'
                if not original.exists():
                    write(original, original_ledger)
                entry.update(raw_evaluator_resolved=True, resolved=False, status='final_media_contract_failed')
                outcome_path = dest / 'outcome.json'
                outcome = read(outcome_path)
                if not (dest / 'outcome_before_media_contract.json').exists():
                    write(dest / 'outcome_before_media_contract.json', outcome)
                outcome.update(raw_evaluator_resolved=True, resolved=False,
                               status='final_media_contract_failed', final_media_contract=record)
                write(outcome_path, outcome)
                changed = True
        if changed:
            ledger['resolved'] = False
            write(ledger_path, ledger)
            final_path = run / 'execution' / method / key / 'outcome.json'
            outcome = read(final_path)
            if not (final_path.parent / 'outcome_before_media_contract.json').exists():
                write(final_path.parent / 'outcome_before_media_contract.json', outcome)
            selected = read(Path(ledger['selected_candidate']) / 'outcome.json')
            outcome.update(resolved=False, status=selected['status'], raw_evaluator_resolved=True,
                           first_normal_resolved=next((c['resolved'] for c in ledger['candidates']
                                                      if c['kind'] == 'normal_execution'), None),
                           final_media_contract=selected['final_media_contract'])
            write(final_path, outcome)
    payload = dict(conditions=len(list((run / 'iteration_runs').glob('*/*/ledger.json'))),
                   evaluated_candidates=len(rows),
                   mismatch_candidates=sum(not x['passed'] for x in rows),
                   corrected_false_successes=sum(not x['passed'] and x['raw_evaluator_resolved'] for x in rows),
                   all_candidates_checked=True, details=rows)
    write(run / 'final_media_contract_audit.json', payload)
    print({k: v for k, v in payload.items() if k != 'details'})
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    apply(parser.parse_args().run.resolve())
