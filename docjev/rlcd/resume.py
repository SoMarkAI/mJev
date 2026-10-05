"""Exact same-world FSDP restart: FP32 master model plus rank-local Adam/RNG."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from .common import sha256, write_json


def layout(model):
    return [(n, list(p.shape), str(p.dtype), p.requires_grad) for n, p in model.named_parameters()]


def inspect_resume(path, expected):
    path = Path(path)
    if not (path / '_SUCCESS').is_file():
        raise ValueError('Resume checkpoint is incomplete')
    metadata = json.loads((path / 'manifest.json').read_text())
    if metadata['contract'] != expected:
        raise ValueError('Resume world/config/data/reference contract differs')
    for name, digest in metadata['files'].items():
        if sha256(path / name) != digest:
            raise ValueError('Resume artifact hash differs: ' + name)
    return metadata


def save_resume(model, optimizer, sampling, out, step, contract):
    rank, world = dist.get_rank(), dist.get_world_size()
    path = Path(out) / ('resume-step' + str(step))
    if rank == 0:
        path.mkdir()
    dist.barrier()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        state = model.state_dict()
    if rank == 0:
        torch.save(state, path / 'model-fp32.pt')
    torch.save({'optimizer': optimizer.state_dict(), 'layout': layout(model),
        'sampling_rng': sampling.get_state(), 'torch_cpu_rng': torch.get_rng_state(),
        'torch_cuda_rng': torch.cuda.get_rng_state(), 'step': step}, path / f'rank{rank}.pt')
    dist.barrier()
    if rank == 0:
        names = ['model-fp32.pt'] + [f'rank{r}.pt' for r in range(world)]
        write_json(path / 'manifest.json', {'step': step, 'contract': contract,
            'format': 'full FP32 master state plus deterministic rank-local AdamW/RNG; same FSDP world/layout required',
            'files': {name: sha256(path / name) for name in names}})
        (path / '_MODEL_OPTIMIZER_READY').write_text('model, optimizer and RNG files saved; continuation witness pending\n')
    dist.barrier()
    return path


def restore_rank(path, model, optimizer, sampling):
    state = torch.load(Path(path) / f'rank{dist.get_rank()}.pt', map_location='cpu', weights_only=True)
    if state['layout'] != layout(model):
        raise ValueError('FSDP local parameter layout changed')
    optimizer.load_state_dict(state['optimizer'])
    sampling.set_state(state['sampling_rng'].cpu())
    torch.set_rng_state(state['torch_cpu_rng'].cpu())
    torch.cuda.set_rng_state(state['torch_cuda_rng'].cpu())
    if not optimizer.state or not all('step' in value for value in optimizer.state.values()):
        raise ValueError('Adam optimizer state did not restore')
    if max(float(v['step']) for v in optimizer.state.values()) != state['step']:
        raise ValueError('Restored Adam step counter differs from saved step')
    return state['step']
