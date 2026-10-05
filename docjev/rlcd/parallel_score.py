"""Eight independent native HF replicas for frozen references or evaluation."""
import argparse
import datetime
import json
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
from .common import read_jsonl, sha256, write_json, write_jsonl
from .model import Inputs, candidate_logits, load_model, move_batch
from .metrics import probability_metrics
from .train import assert_reference


def validation_metrics(rows):
    return {'original_order': probability_metrics(rows),
            'languages': {lang: probability_metrics([r for r in rows if r['language'] == lang])
                          for lang in ('zh', 'en')}}


def run(model_path, data, config_path, out, phase, references=None):
    rank, local_rank, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    device = torch.device('cuda', local_rank)
    root, dataset = Path(out), Path(data)
    config = json.loads(Path(config_path).read_text())
    if not (dataset / '_SUCCESS').exists():
        raise ValueError('Dataset preparation incomplete')
    if rank == 0:
        if root.exists():
            raise FileExistsError('Use a fresh scoring directory')
        root.mkdir(parents=True)
    dist.barrier()
    reference_meta = None
    if phase == 'after':
        if not (Path(references) / '_SUCCESS').exists():
            raise ValueError('Frozen references incomplete')
        reference_meta = json.loads((Path(references) / 'reference.json').read_text())
        if reference_meta['config_sha256'] != sha256(config_path) or reference_meta['dataset_manifest_sha256'] != sha256(dataset / 'manifest.json'):
            raise ValueError('Frozen cohort changed')
        for name, digest in reference_meta['files'].items():
            if sha256(Path(references) / name) != digest:
                raise ValueError('Reference bytes changed')
    model, loading = load_model(model_path, dtype=torch.bfloat16, device=device)
    model.eval()
    inputs = Inputs(model_path, dataset, config)
    started = time.perf_counter()
    splits = ('train', 'dev', 'test') if phase == 'reference' else ('dev',)
    for split in splits:
        source = dataset if phase == 'reference' else Path(references)
        rows = read_jsonl(source / (split + '.jsonl'))
        output = []
        for index in range(rank, len(rows), world):
            row = rows[index]
            batch, spec = inputs.prepare(row)
            if phase == 'after':
                assert_reference(row, spec)
            with torch.no_grad():
                logits = candidate_logits(model, move_batch(batch, device), spec['label_ids'])
                log_probs = logits.log_softmax(-1)
            target = spec['labels'].index(row['label'])
            result = {**row, **spec, 'raw_logits': logits.cpu().tolist(),
                'probabilities': log_probs.exp().cpu().tolist(), 'target_index': target,
                'decision': spec['labels'][int(logits.argmax())]}
            if phase == 'reference':
                result.update(reference_log_probs=log_probs.cpu().tolist(),
                              p_target=float(log_probs[target].exp()))
            output.append(result)
            if len(output) % 25 == 0:
                print(json.dumps({'phase': phase, 'split': split, 'rank': rank,
                    'completed': len(output), 'assigned': len(rows[rank::world]),
                    'elapsed_seconds': time.perf_counter() - started}), flush=True)
        write_jsonl(root / f'{split}.rank{rank}.jsonl', output)
        dist.barrier()
        if rank == 0:
            merged = [r for worker in range(world) for r in read_jsonl(root / f'{split}.rank{worker}.jsonl')]
            by_id = {r['id']: r for r in merged}
            if len(merged) != len(rows) or len(by_id) != len(rows) or set(by_id) != {r['id'] for r in rows}:
                raise ValueError('Missing or duplicate distributed predictions')
            write_jsonl(root / (split + '.jsonl'), [by_id[r['id']] for r in rows])
        dist.barrier()
    if rank == 0:
        if phase == 'reference':
            write_json(root / 'reference.json', {'model_id': config['model_id'],
                'model_revision': config['model_revision'], 'loading_info': loading,
                'config_sha256': sha256(config_path), 'dataset_manifest_sha256': sha256(dataset / 'manifest.json'),
                'coverage': {s: len(read_jsonl(root / (s + '.jsonl'))) for s in splits},
                'elapsed_seconds': time.perf_counter() - started, 'world_size': world,
                'action_space': 'candidate_label_softmax', 'numerics': 'native_bfloat16',
                'files': {s + '.jsonl': sha256(root / (s + '.jsonl')) for s in splits}})
        rows = read_jsonl(root / 'dev.jsonl')
        metadata = {'phase': 'before' if phase == 'reference' else 'after',
            'scope': config['scope'], 'metrics': {'dev': validation_metrics(rows)},
            'validation_pairs': len(rows) // 2, 'model_path': str(model_path),
            'config_sha256': sha256(config_path), 'dataset_manifest_sha256': sha256(dataset / 'manifest.json'),
            'reference_provenance': config['reference_provenance'], 'human_gt_accuracy_claim': False,
            'prediction_sha256': sha256(root / 'dev.jsonl')}
        if phase == 'after':
            metadata['reference_metadata_sha256'] = sha256(Path(references) / 'reference.json')
            # Independent exported-checkpoint reload vs the FSDP final forward witnesses.
            witnesses = read_jsonl(root.parent / 'train' / 'post_training_dev.jsonl')
            by_id = {r['id']: r for r in rows}
            errors = [max(abs(a - b) for a, b in zip(w['raw_logits'], by_id[w['id']]['raw_logits'], strict=True)) for w in witnesses]
            if not errors or max(errors) > .125:
                raise ValueError('Exported checkpoint differs from final FSDP witnesses')
            metadata['checkpoint_reload_witnesses'] = len(errors)
            metadata['checkpoint_reload_max_logit_error'] = max(errors)
        write_json(root / 'metrics.json', metadata)
        (root / '_SUCCESS').write_text('complete fixed-cohort distributed native HF scoring\n')
        print(json.dumps(metadata), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('model', 'data', 'config', 'out', 'phase'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--references')
    args = parser.parse_args()
    if args.phase not in ('reference', 'after') or (args.phase == 'after' and not args.references):
        parser.error('Use reference or after with --references')
    run(args.model, args.data, args.config, args.out, args.phase, args.references)
