"""Accuracy, natural-log NLL, multiclass Brier and top-label equal-width ECE."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from .dataset import MJevDataset


def score_row(row, prediction):
    labels = list(row['candidates'])
    if ('logits' in prediction) == ('probabilities' in prediction):
        raise ValueError('Supply exactly one of logits or probabilities')
    field = 'logits' if 'logits' in prediction else 'probabilities'
    values = prediction[field]
    if not isinstance(values, dict) or set(values) != set(labels):
        raise ValueError('Prediction must map every original candidate label exactly once')
    if any(isinstance(values[k], bool) or not isinstance(values[k], (int, float)) for k in labels):
        raise ValueError('Scores must be numbers')
    x = [float(values[k]) for k in labels]
    if not all(math.isfinite(v) for v in x):
        raise ValueError('Nonfinite scores')
    target = labels.index(row['label'])
    if field == 'logits':
        m = max(x)
        log_z_shifted = math.log(sum(math.exp(v-m) for v in x))
        p = [math.exp(v-m-log_z_shifted) for v in x]
        nll = (m-x[target]) + log_z_shifted
        winner = max(range(len(x)), key=x.__getitem__)
    else:
        if any(v < 0 or v > 1 for v in x) or not math.isclose(sum(x), 1, abs_tol=1e-6, rel_tol=0):
            raise ValueError('Probabilities must be in [0,1] and sum to 1; no silent renormalization')
        p = x
        nll = -math.log(p[target]) if p[target] else math.inf
        winner = max(range(len(p)), key=p.__getitem__)
    if 'prediction' in prediction and prediction['prediction'] != labels[winner]:
        raise ValueError('Predicted label disagrees with argmax/first-label tie rule')
    return {'correct': int(winner == target), 'nll': nll,
            'brier': sum((v-int(i == target))**2 for i, v in enumerate(p)),
            'confidence': p[winner]}


def aggregate(scores, bins):
    n = len(scores)
    buckets = [[] for _ in range(bins)]
    for s in scores:
        buckets[min(bins-1, int(s['confidence'] * bins))].append(s)
    calibration = []
    ece = 0.0
    for i, group in enumerate(buckets):
        acc = sum(s['correct'] for s in group)/len(group) if group else None
        conf = sum(s['confidence'] for s in group)/len(group) if group else None
        if group:
            ece += len(group)/n * abs(acc-conf)
        calibration.append({'lower': i/bins, 'upper': (i+1)/bins,
                            'count': len(group), 'accuracy': acc, 'confidence': conf})
    infinite = sum(not math.isfinite(s['nll']) for s in scores)
    return {'count': n, 'accuracy': sum(s['correct'] for s in scores)/n if n else None,
            'nll': None if infinite or not n else sum(s['nll'] for s in scores)/n,
            'nll_is_infinite': bool(infinite), 'zero_target_probability_count': infinite,
            'brier_score': sum(s['brier'] for s in scores)/n if n else None,
            'ece': ece if n else None, 'calibration_bins': calibration}


def evaluate(records, predictions, bins=15, allow_partial=False):
    if not isinstance(bins, int) or bins < 1:
        raise ValueError('bins must be a positive integer')
    rows = list(records)
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate dataset IDs')
    pred = {}
    for p in predictions:
        if p['id'] in pred:
            raise ValueError('Duplicate prediction IDs')
        if p['id'] not in by_id:
            raise ValueError('Prediction for unknown ID')
        pred[p['id']] = p
    missing = sorted(set(by_id)-set(pred))
    if missing and not allow_partial:
        raise ValueError(f'Missing {len(missing)} predictions; use explicit allow_partial for coverage reporting')
    groups = defaultdict(list)
    scored = []
    for row in rows:
        if row['id'] not in pred:
            continue
        s = score_row(row, pred[row['id']]); scored.append(s)
        for field in ('dataset', 'modality', 'task'):
            values = row[field] if isinstance(row[field], list) else [row[field]]
            for v in dict.fromkeys(values):
                groups[f'{field}:{v}'].append(s)
        groups[f'candidate_count:{len(row["candidates"])}'].append(s)
    return {'metrics': aggregate(scored, bins), 'expected_count': len(rows),
            'coverage': len(scored)/len(rows) if rows else 0,
            'missing_ids': missing, 'groups': {k: aggregate(v, bins) for k, v in sorted(groups.items())},
            'definitions': {'nll': 'mean negative natural log probability of original label; infinity encoded as null with nll_is_infinite=true',
                            'brier_score': 'mean sum over candidates of (p-one_hot(label)) squared; no division by candidate count',
                            'ece': f'top-label confidence, {bins} equal-width bins; left-closed, last bin includes 1',
                            'decision': 'argmax, ties resolved by original candidate order',
                            'scope': 'sampled per-question metrics, not official Video-MME-v2 group leaderboard score'}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True); p.add_argument('--bins', type=int, default=15)
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    ds = MJevDataset(args.manifest, require_media=False)
    predictions = [json.loads(x) for x in Path(args.predictions).read_text().splitlines() if x.strip()]
    result = evaluate(ds, predictions, args.bins, args.allow_partial)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['metrics'], allow_nan=False))

if __name__ == '__main__':
    main()
