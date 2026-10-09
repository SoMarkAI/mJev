"""Persist a CPU byte comparison of the official base and exported weights."""
import argparse
import contextlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from .common import write_json


def compare(base, checkpoint, report_path):
    base, checkpoint = Path(base), Path(checkpoint)
    base_map = json.loads((base / 'model.safetensors.index.json').read_text())['weight_map']
    export_map = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    if set(base_map) - set(export_map) or set(export_map) - set(base_map) - {'lm_head.weight'}:
        raise ValueError('Base/export tensor key mismatch')
    records = []
    with contextlib.ExitStack() as stack:
        base_files = {name: stack.enter_context(safe_open(base / name, framework='pt', device='cpu'))
                      for name in set(base_map.values())}
        export_files = {name: stack.enter_context(safe_open(checkpoint / name, framework='pt', device='cpu'))
                        for name in set(export_map.values())}
        for name in sorted(base_map):
            source = base_files[base_map[name]].get_tensor(name)
            saved = export_files[export_map[name]].get_tensor(name)
            if source.shape != saved.shape or source.dtype != saved.dtype:
                raise ValueError('Base/export tensor shape or dtype mismatch')
            equal = torch.equal(source.view(torch.uint8), saved.view(torch.uint8))
            record = {'name': name, 'dtype': str(source.dtype), 'shape': list(source.shape),
                      'bytes': source.numel() * source.element_size(), 'byte_identical': equal,
                      'visual': name.startswith('model.visual.')}
            if record['visual'] and not equal:
                raise ValueError('Export changed a frozen visual tensor')
            records.append(record)
        embedding_name = 'model.language_model.embed_tokens.weight'
        embedding = export_files[export_map[embedding_name]].get_tensor(embedding_name)
        head = export_files[export_map['lm_head.weight']].get_tensor('lm_head.weight')
        tied_equal = torch.equal(embedding.view(torch.uint8), head.view(torch.uint8))
        if not tied_equal:
            raise ValueError('Exported tied embedding and LM head differ')
    visual = [r for r in records if r['visual']]
    language = [r for r in records if not r['visual']]
    changed = sum(not r['byte_identical'] for r in language)
    if changed == 0:
        raise ValueError('Exported language tensors are identical to the base')
    summary = {'common_tensors': len(records), 'visual_tensors': len(visual),
               'visual_bytes': sum(r['bytes'] for r in visual), 'visual_all_byte_identical': True,
               'language_tensors': len(language), 'changed_language_tensors': changed,
               'tied_head_equals_embedding': tied_equal, 'device': 'cpu',
               'comparison': 'uint8 byte equality, not approximate float equality'}
    write_json(report_path, {'summary': summary, 'tensors': records})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['base', 'checkpoint', 'report']:
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    compare(a.base, a.checkpoint, a.report)
