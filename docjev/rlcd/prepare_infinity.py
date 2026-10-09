"""Freeze all eligible bilingual pairs and an exact image-grouped validation set."""
import argparse
import collections
import concurrent.futures
import json
import math
import multiprocessing
import random
import shutil
from pathlib import Path
from .common import read_jsonl, sha256, write_json, write_jsonl
from .prepare_docjev import components, paired_rows
from .model import Inputs

_inputs = None


def worker_init(model, root, config):
    global _inputs
    import torch
    torch.set_num_threads(1)
    _inputs = Inputs(model, Path(root), config)


def inspect_pair(pair):
    try:
        specs = [_inputs.prepare(row)[1] for row in pair]
    except ValueError as exc:
        return None, {'pair_id': pair[0]['pair_id'], 'reason': str(exc),
                      'both_languages_excluded': True}
    for row, spec in zip(pair, specs, strict=True):
        row['prepared_input_tokens'] = spec['input_tokens']
    return pair, None


def exact_validation_groups(groups, target):
    """Subset sum over complete media groups, in previously seeded random order."""
    reachable = {0: ()}
    for index, group in enumerate(groups):
        size = sum(len(doc['retained_pairs']) for doc in group)
        if not size:
            continue
        for total, chosen in list(reachable.items()):
            new = total + size
            if new <= target and new not in reachable:
                reachable[new] = chosen + (index,)
        if target in reachable:
            return set(reachable[target])
    raise ValueError('Cannot retain exactly the requested validation pairs as complete image groups')


def prepare(source, image_root, model, config_path, out):
    root, media = Path(out), Path(image_root).resolve()
    if root.exists():
        raise FileExistsError('Use a fresh frozen dataset directory')
    root.mkdir(parents=True)
    (root / 'media').mkdir()
    (root / 'source_documents.jsonl').write_bytes(Path(source).read_bytes())
    source_sha = sha256(root / 'source_documents.jsonl')
    docs = read_jsonl(root / 'source_documents.jsonl')
    config = json.loads(Path(config_path).read_text())
    jobs, owners = [], []
    for doc in docs:
        if doc['status'] != 'accepted':
            raise ValueError('Nonaccepted source image')
        src = (media / doc['image']).resolve()
        if media not in src.parents or not src.is_file() or sha256(src) != doc['image_sha256']:
            raise ValueError('Missing, unsafe or changed source media')
        relative = 'media/' + doc['image_sha256'] + src.suffix.lower()
        shutil.copyfile(src, root / relative)
        doc['media_path'] = relative
        doc['retained_pairs'] = []
        for pair in paired_rows(doc, relative, 'unassigned', source_sha, config['reference_provenance']):
            jobs.append(pair)
            owners.append(doc)
    exclusions = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=config['prepare_workers'],
            mp_context=multiprocessing.get_context('spawn'), initializer=worker_init,
            initargs=(model, str(root), config)) as pool:
        for index, ((pair, excluded), owner) in enumerate(zip(pool.map(inspect_pair, jobs, chunksize=8), owners, strict=True)):
            if pair:
                owner['retained_pairs'].append(pair)
            else:
                exclusions.append(excluded)
            if (index + 1) % 100 == 0:
                print(json.dumps({'phase': 'prepare', 'pairs_checked': index + 1,
                    'total_source_pairs': len(jobs), 'excluded_pairs': len(exclusions)}), flush=True)
    eligible = [d for d in docs if d['retained_pairs']]
    groups = components(eligible)
    random.Random(config['seed']).shuffle(groups)
    validation = exact_validation_groups(groups, config['validation_pairs'])
    splits = {'train': [], 'dev': [], 'test': []}
    images = []
    for index, group in enumerate(groups):
        split = 'dev' if index in validation else 'train'
        for doc in group:
            images.append({'image_sha256': doc['image_sha256'], 'paper_id': doc.get('paper_id'),
                'split': split, 'media_path': doc['media_path'], 'source_id': doc['id'],
                'retained_pairs': len(doc['retained_pairs'])})
            for pair in doc['retained_pairs']:
                for row in pair:
                    row['split'] = split
                splits[split].append(pair)
    image_sets = {s: {r['image_sha256'] for pair in values for r in pair} for s, values in splits.items()}
    paper_sets = {s: {r['paper_id'] for pair in values for r in pair if r['paper_id']} for s, values in splits.items()}
    for sets in (image_sets, paper_sets):
        if sets['train'] & sets['dev']:
            raise ValueError('Cross-split image/paper leakage')
    if len(splits['dev']) != config['validation_pairs'] or not splits['train']:
        raise ValueError('Unexpected training/validation cohort')
    for split, pairs in splits.items():
        random.Random(config['seed'] + {'train': 0, 'dev': 1, 'test': 2}[split]).shuffle(pairs)
        write_jsonl(root / (split + '.jsonl'), [row for pair in pairs for row in pair])
    rows = [row for pairs in splits.values() for pair in pairs for row in pair]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate question ID')
    config['steps'] = math.ceil(len(splits['train']) * 2 / config['world_size']) * config['epochs']
    write_json(config_path, config)
    write_jsonl(root / 'images.jsonl', images)
    write_jsonl(root / 'exclusions.jsonl', exclusions)
    write_json(root / 'manifest.json', {'scope': config['scope'], 'seed': config['seed'],
        'source_sha256': source_sha, 'source_images': len(docs), 'source_pairs': len(jobs),
        'images': {s: len(v) for s, v in image_sets.items()},
        'questions': {s: len(v) * 2 for s, v in splits.items()},
        'semantic_pairs': {s: len(v) for s, v in splits.items()},
        'excluded_pairs': len(exclusions), 'max_prepared_input_tokens': max(r['prepared_input_tokens'] for r in rows),
        'split_rule': 'seeded exact subset of complete paper/image connected groups; all bilingual pairs stay together',
        'image_and_paper_splits_disjoint': True, 'annotations_changed': False,
        'reference_provenance': config['reference_provenance'], 'human_ground_truth_claim': False,
        'near_duplicate_leakage_audit': 'exact image SHA256 and supplied paper IDs only; perceptual duplicates not assessed',
        'files': {p.name: sha256(p) for p in sorted(root.glob('*.jsonl'))}})
    (root / '_SUCCESS').write_text('frozen grouped bilingual data and native processor checks passed\n')
    print(json.dumps({'phase': 'prepared', 'training_steps': config['steps'],
        'images': {s: len(v) for s, v in image_sets.items()}, 'pairs': {s: len(v) for s, v in splits.items()},
        'excluded_pairs': len(exclusions)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['source', 'image-root', 'model', 'config', 'out']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    prepare(args.source, args.image_root, args.model, args.config, args.out)
