"""Check saved safetensors headers, independent of FSDP local parameter views."""
import argparse
import json
import math
from pathlib import Path
from .common import write_json


def inspect_index(checkpoint, expected_parameters, *, repair=False):
    checkpoint = Path(checkpoint)
    index_path = checkpoint / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    tensors = {}
    byte_count = element_count = 0
    for filename in sorted(set(index['weight_map'].values())):
        if Path(filename).name != filename:
            raise ValueError('Unsafe checkpoint shard filename')
        path = checkpoint / filename
        with path.open('rb') as f:
            length = int.from_bytes(f.read(8), 'little')
            if length <= 0 or length > 100_000_000:
                raise ValueError('Invalid safetensors header length')
            header = json.loads(f.read(length))
        for name, spec in header.items():
            if name == '__metadata__':
                continue
            if name in tensors or index['weight_map'].get(name) != filename:
                raise ValueError('Checkpoint index does not match shard headers')
            count = math.prod(spec['shape'])
            start, stop = spec['data_offsets']
            if spec['dtype'] != 'BF16' or stop - start != count * 2:
                raise ValueError('Expected a BF16 deployment checkpoint')
            if start < 0 or stop < start or 8 + length + stop > path.stat().st_size:
                raise ValueError('Invalid checkpoint tensor offsets')
            tensors[name] = count
            byte_count += stop - start
            element_count += count
    if set(tensors) != set(index['weight_map']):
        raise ValueError('Checkpoint index references missing tensors')
    config = json.loads((checkpoint / 'config.json').read_text())
    tied = config.get('tie_word_embeddings', False)
    unique_parameters = element_count
    if tied and 'lm_head.weight' in tensors:
        if tensors['lm_head.weight'] != tensors['model.language_model.embed_tokens.weight']:
            raise ValueError('Tied embedding/head shape mismatch')
        unique_parameters -= tensors['lm_head.weight']
    if unique_parameters != expected_parameters:
        raise ValueError('Serialized unique parameter count differs from the full base model')
    if index['metadata']['total_size'] != byte_count:
        raise ValueError('Checkpoint total_size differs from shard headers')
    previous = index['metadata']['total_parameters']
    if previous != unique_parameters:
        if not repair:
            raise ValueError('Checkpoint total_parameters is a local FSDP shard count')
        index['metadata']['total_parameters'] = unique_parameters
        write_json(index_path, index)
    return {'unique_parameters': unique_parameters, 'stored_elements': element_count,
            'stored_bytes': byte_count, 'stored_tensors': len(tensors),
            'tied_lm_head': tied, 'previous_total_parameters': previous,
            'metadata_repaired': previous != unique_parameters,
            'index_matches_safetensors_headers': True}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--expected-parameters', required=True, type=int)
    p.add_argument('--report', required=True)
    p.add_argument('--repair', action='store_true')
    a = p.parse_args()
    report = inspect_index(a.checkpoint, a.expected_parameters, repair=a.repair)
    write_json(a.report, report)
    print(json.dumps(report), flush=True)
