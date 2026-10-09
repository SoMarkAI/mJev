"""FSDP full-language-parameter, candidate-restricted RLCD trainer."""
import argparse
import datetime
import functools
import json
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, MixedPrecision,
                                    ShardingStrategy, StateDictType, FullStateDictConfig)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
from .common import read_jsonl, sha256, write_json, write_jsonl
from .checkpoint_index import inspect_index
from .model import (Inputs, candidate_logits, cast_parameters_preserve_buffers,
                    load_model, move_batch, position_buffer_digest, visual_digest)
from .objective import advantages, rlcd_loss, rewards, sample_group
from .resume import inspect_resume, restore_rank, save_resume
from .distributed_eval import evaluation_schedule


def clean_name(name):
    return name.replace('_fsdp_wrapped_module.', '')


def group_name(name):
    if 'visual' in name:
        return 'visual'
    if 'embed_tokens' in name:
        return 'embedding'
    if 'lm_head' in name:
        return 'lm_head'
    return 'language_backbone'


def assert_reference(row, spec):
    for key in ['label_ids', 'labels', 'input_tokens', 'tensor_hashes', 'prompt_sha256']:
        if row[key] != spec[key]:
            raise ValueError(f'{row["id"]}: frozen reference input mismatch: {key}')
    probability = torch.tensor(row['reference_log_probs']).exp()
    if not torch.isclose(probability.sum(), torch.tensor(1.), atol=1e-5):
        raise ValueError('Invalid frozen reference distribution')
    if abs(float(probability[row['target_index']]) - row['p_target']) > 1e-6:
        raise ValueError('p_target does not match the frozen reference')


def train(model_path, data_root, references, config_path, out, resume=None, stop_after=None):
    config = json.loads(Path(config_path).read_text())
    if config['reward_scaling'] or config['lora'] or not config['freeze_visual']:
        raise ValueError('Config violates agreed training contract')
    if config['attention'] != 'causal' or config['prefix_cache']:
        raise ValueError('This training prototype uses uncached causal forwards')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != config.get('world_size', 4):
        raise ValueError('FSDP world differs from the frozen configuration')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=15))
    torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(out)
    if rank == 0:
        if out.exists() and not resume:
            raise FileExistsError('Use a fresh training output directory')
        if resume and (not out.exists() or (out / '_TRAIN_COMPLETE').exists()):
            raise ValueError('Resume requires an unfinished training directory')
        out.mkdir(parents=True, exist_ok=bool(resume))
    dist.barrier()
    if not (Path(references) / '_SUCCESS').is_file():
        raise ValueError('Frozen reference scoring did not complete')
    reference_meta = json.loads((Path(references) / 'reference.json').read_text())
    if reference_meta['config_sha256'] != sha256(config_path):
        raise ValueError('Reference configuration changed')
    if reference_meta['dataset_manifest_sha256'] != sha256(Path(data_root) / 'manifest.json'):
        raise ValueError('Dataset changed after reference scoring')
    for name, digest in reference_meta['files'].items():
        if sha256(Path(references) / name) != digest:
            raise ValueError('Frozen reference file changed')
    contract = {'world': world, 'config_sha256': sha256(config_path),
        'dataset_manifest_sha256': sha256(Path(data_root) / 'manifest.json'),
        'reference_metadata_sha256': sha256(Path(references) / 'reference.json')}
    resume_metadata = inspect_resume(resume, contract) if resume else None
    rows = read_jsonl(Path(references) / 'train.jsonl')
    dev = read_jsonl(Path(references) / 'dev.jsonl')
    if config.get('post_training_witness_rows'):
        dev = dev[:config['post_training_witness_rows']]
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('Nontraining sample entered the optimizer')
    inputs = Inputs(model_path, data_root, config)
    # FP32 master parameters and optimizer state; BF16 forward/backward compute.
    native, loading = load_model(model_path, dtype=torch.float32)
    position_before = position_buffer_digest(native)
    if resume:
        state = torch.load(Path(resume) / 'model-fp32.pt', map_location='cpu', weights_only=True)
        native.load_state_dict(state, strict=True)
        del state
        if position_buffer_digest(native) != position_before:
            raise ValueError('Resume changed official position buffers')
    native.model.visual.requires_grad_(False)
    cast_parameters_preserve_buffers(native.model.visual, device=device, dtype=torch.bfloat16)
    for module in native.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    if config['gradient_checkpointing']:
        native.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    counts = {'trainable': sum(p.numel() for p in native.parameters() if p.requires_grad),
              'frozen': sum(p.numel() for p in native.parameters() if not p.requires_grad)}
    if any('lora' in n.lower() for n, _ in native.named_parameters()):
        raise ValueError('Unexpected LoRA parameters')
    if any(p.requires_grad for p in native.model.visual.parameters()):
        raise ValueError('Visual tower/aligners are not fully frozen')
    frozen_before = visual_digest(native) if rank == 0 else None
    wrap = functools.partial(transformer_auto_wrap_policy,
                             transformer_layer_cls={Qwen3VLTextDecoderLayer})
    model = FSDP(native, auto_wrap_policy=wrap, ignored_modules=[native.model.visual],
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                               reduce_dtype=torch.float32,
                                               buffer_dtype=torch.float32),
                 device_id=device, use_orig_params=True, limit_all_gathers=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'], foreach=False)
    # Sample values from EVERY local parameter shard for actual update witnesses.
    before = {n: p.detach().flatten()[:64].clone() for n, p in model.named_parameters()
              if p.requires_grad and p.numel()}
    before_deployment = {n: value.to(torch.bfloat16) for n, value in before.items()}
    sampling = torch.Generator(device=device).manual_seed(config['seed'] + 1000 + rank)
    start_step = restore_rank(resume, model, optimizer, sampling) if resume else 0
    if resume and start_step != resume_metadata['step']:
        raise ValueError('Rank resume step differs')
    records = read_jsonl(out / 'steps.jsonl') if resume and rank == 0 else []
    groups = read_jsonl(out / 'rollouts.jsonl') if resume and rank == 0 else []
    if rank == 0 and resume and (len(records) != start_step or len(groups) != start_step * world):
        raise ValueError('Recorded rollouts/steps differ from resume checkpoint')
    end_step = min(config['steps'], stop_after) if stop_after else config['steps']
    if not start_step < end_step <= config['steps']:
        raise ValueError('Invalid training segment')
    if rank == 0:
        write_json(out / ('resume-validation.json' if resume else 'start-validation.json'), {
            'start_step': start_step, 'end_step': end_step, 'optimizer_restored': bool(resume),
            'rng_restored': bool(resume), 'rank_local_layout_verified': bool(resume),
            'contract': contract})
    started = time.perf_counter()
    for step in range(start_step, end_step):
        model.train()
        native.model.visual.eval()
        from .distributed_eval import training_position
        index, is_padding = training_position(len(rows), step, rank, world)
        row = rows[index]
        batch, spec = inputs.prepare(row)
        assert_reference(row, spec)
        batch = move_batch(batch, device)
        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        with torch.no_grad():
            old_logits = candidate_logits(model, batch, spec['label_ids'])
            actions, old_logp = sample_group(old_logits, config['group_size'], sampling)
        if resume and step == start_step:
            witness = json.loads((Path(resume) / f'continuation-rank{rank}.json').read_text())
            error = float((old_logits.cpu() - torch.tensor(witness['raw_logits'])).abs().max())
            if row['id'] != witness['id'] or error > .125 or actions.cpu().tolist() != witness['actions']:
                raise ValueError('Resume policy/data cursor/RNG continuation differs')
            write_json(out / f'resume-witness-rank{rank}.json', {'id': row['id'],
                'max_logit_error': error, 'sampled_actions_match': True,
                'optimizer_state_restored': True, 'data_cursor_match': True})
        reward = rewards(actions, row['target_index'], row['p_target'])
        advantage = advantages(reward)
        optimizer.zero_grad(set_to_none=True)
        logits = candidate_logits(model, batch, spec['label_ids'])
        reference_logp = torch.tensor(row['reference_log_probs'], device=device)
        loss, details = rlcd_loss(logits, actions, old_logp, advantage, reference_logp,
                                  clip_epsilon=config['clip_epsilon'], kl_beta=config['kl_beta'])
        if is_padding:
            # All ranks participate in FSDP, but padded rows have zero gradient.
            loss = loss * 0.
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite RLCD loss')
        loss.backward()
        gradient_norm = model.clip_grad_norm_(config['max_grad_norm'])
        if not torch.isfinite(gradient_norm):
            raise ValueError('Nonfinite gradient norm')
        norms = {k: torch.zeros((), device=device) for k in
                 ['visual', 'embedding', 'lm_head', 'language_backbone']}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                value = parameter.grad.detach().norm().float().square()
                norms[group_name(clean_name(name))] += value
        for value in norms.values():
            dist.all_reduce(value)
        if norms['visual'].item() != 0 or any(p.grad is not None for p in native.model.visual.parameters()):
            raise ValueError('Frozen visual parameter acquired a gradient')
        optimizer.step()
        torch.cuda.synchronize(device)
        local = {'step': step + 1, 'rank': rank, 'id': row['id'], 'is_padding': is_padding,
                 'image_sha256': row['image_sha256'], 'input_tokens': spec['input_tokens'],
                 'p_target': row['p_target'], 'correct_label': row['label'],
                 'sampled_labels': [spec['labels'][i] for i in actions.cpu().tolist()],
                 'old_action_log_probs': old_logp.cpu().tolist(),
                 'raw_rewards': reward.cpu().tolist(), 'advantages': advantage.cpu().tolist(),
                 'zero_advantage_group': bool(torch.all(advantage == 0)),
                 'loss': float(loss.detach()), **details,
                 'gradient_norm_before_clip': float(gradient_norm),
                 'gradient_norms_after_clip': {k: float(v.sqrt()) for k, v in norms.items()},
                 'seconds': time.perf_counter() - step_started,
                 'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device)}
        gathered = [None] * world if rank == 0 else None
        dist.gather_object(local, gathered, dst=0)
        if rank == 0:
            groups.extend(gathered)
            summary = {'step': step + 1, 'mean_loss': sum(v['loss'] for v in gathered) / world,
                       'mixed_reward_groups': sum(not v['zero_advantage_group'] for v in gathered),
                       'gradient_norm': local['gradient_norm_before_clip'],
                       'gradient_norms': local['gradient_norms_after_clip'],
                       'seconds': max(v['seconds'] for v in gathered)}
            records.append(summary)
            write_jsonl(out / 'steps.jsonl', records)
            write_jsonl(out / 'rollouts.jsonl', groups)
            print(json.dumps(summary), flush=True)
    if end_step < config['steps']:
        if not config['save_optimizer']:
            raise ValueError('Stopping early requires optimizer checkpoint saving')
        saved = save_resume(model, optimizer, sampling, out, end_step, contract)
        continuation = rows[training_position(len(rows), end_step, rank, world)[0]]
        batch, spec = inputs.prepare(continuation)
        assert_reference(continuation, spec)
        with torch.no_grad():
            next_logits = candidate_logits(model, move_batch(batch, device), spec['label_ids'])
        next_sampling = torch.Generator(device=device)
        next_sampling.set_state(sampling.get_state())
        next_actions, _ = sample_group(next_logits, config['group_size'], next_sampling)
        write_json(saved / f'continuation-rank{rank}.json', {'id': continuation['id'],
            'raw_logits': next_logits.cpu().tolist(), 'actions': next_actions.cpu().tolist()})
        dist.barrier()
        if rank == 0:
            metadata = json.loads((saved / 'manifest.json').read_text())
            for r in range(world):
                name = f'continuation-rank{r}.json'
                metadata['files'][name] = sha256(saved / name)
            write_json(saved / 'manifest.json', metadata)
            (saved / '_SUCCESS').write_text('model, optimizer, RNG and continuation witnesses completed\n')
            write_json(out / 'partial-run.json', {'steps_completed': end_step,
                'target_steps': config['steps'], 'resume_checkpoint': str(saved),
                'elapsed_seconds': time.perf_counter() - started})
            (out / '_RESUME_READY').write_text('intentional stop; resume checkpoint saved\n')
            print('RESUME_CHECKPOINT_READY', flush=True)
        dist.barrier(); dist.destroy_process_group()
        return
    changed = torch.zeros((), device=device, dtype=torch.int64)
    changed_deployment = torch.zeros((), device=device, dtype=torch.int64)
    update_squared = torch.zeros((), device=device)
    for name, parameter in model.named_parameters():
        if name in before:
            delta = parameter.detach().flatten()[:64] - before[name]
            changed += (delta != 0).sum()
            changed_deployment += (parameter.detach().flatten()[:64].to(torch.bfloat16)
                                   != before_deployment[name]).sum()
            update_squared += delta.norm().float().square()
    dist.all_reduce(changed)
    dist.all_reduce(changed_deployment)
    dist.all_reduce(update_squared)
    if changed.item() == 0:
        raise ValueError('No language parameter changed')
    if changed_deployment.item() == 0:
        raise ValueError('No witnessed update survives deployment BF16 casting')
    frozen_after = visual_digest(native) if rank == 0 else None
    if rank == 0 and frozen_before != frozen_after:
        raise ValueError('Frozen visual tensors changed')
    position_after = position_buffer_digest(native)
    if position_after != position_before:
        raise ValueError('Official position encoding buffers changed')
    # Record fresh post-training forwards for independent checkpoint reload checks.
    model.eval()
    local_dev = []
    # Every rank must execute the same number of FSDP forwards. Pad the last
    # collective batch with a real row, but never retain padding predictions.
    for row, retain in evaluation_schedule(dev, rank, world):
        batch, spec = inputs.prepare(row)
        assert_reference(row, spec)
        with torch.no_grad():
            logits = candidate_logits(model, move_batch(batch, device), spec['label_ids'])
        if retain:
            local_dev.append({'id': row['id'], 'label_ids': spec['label_ids'],
                          'labels': spec['labels'], 'raw_logits': logits.cpu().tolist(),
                          'probabilities': logits.softmax(-1).cpu().tolist(),
                          'decision': spec['labels'][int(logits.argmax())]})
    gathered_dev = [None] * world if rank == 0 else None
    dist.gather_object(local_dev, gathered_dev, dst=0)
    if rank == 0:
        write_jsonl(out / 'post_training_dev.jsonl', sorted(
            [v for values in gathered_dev for v in values], key=lambda r: r['id']))
    del optimizer, before, before_deployment
    torch.cuda.empty_cache()
    save_started = time.perf_counter()
    state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config):
        state = model.state_dict()
    if rank == 0:
        checkpoint = out / 'checkpoint-final'
        deployment = {k: v.to(dtype=torch.bfloat16) if v.is_floating_point() else v
                      for k, v in state.items()}
        native.save_pretrained(checkpoint, state_dict=deployment, max_shard_size='2GB', safe_serialization=True)
        inputs.processor.save_pretrained(checkpoint)
        # save_pretrained counts this rank's sharded parameters; headers are full.
        index_report = inspect_index(checkpoint, sum(counts.values()), repair=True)
        write_json(out / 'checkpoint-index-validation.json', index_report)
        write_json(out / 'run.json', {
            'scope': config['scope'], 'training_status': 'completed',
            'base_model': config['model_id'], 'base_revision': config['model_revision'],
            'training_method': 'candidate_restricted_single_step_RLCD',
            'implementation': f'HF/PyTorch FSDP{world}; no generate, vLLM, LoRA, or new head',
            'parameter_counts': counts, 'parameter_master_dtype': 'float32',
            'lm_head_tied_to_embedding': native.config.tie_word_embeddings,
            'compute_dtype': 'bfloat16', 'checkpoint_dtype': 'bfloat16',
            'loading_info': loading, 'config': config, 'config_sha256': sha256(config_path),
            'reference_metadata_sha256': sha256(Path(references) / 'reference.json'),
            'dataset_manifest_sha256': sha256(Path(data_root) / 'manifest.json'),
            'optimizer_steps': len(records), 'questions_used': sum(not g.get('is_padding', False) for g in groups),
            'distinct_training_questions': len({g['id'] for g in groups if not g.get('is_padding', False)}),
            'padding_forward_count': sum(g.get('is_padding', False) for g in groups),
            'rollout_labels': len(groups) * config['group_size'],
            'mixed_reward_groups': sum(not v['zero_advantage_group'] for v in groups),
            'changed_parameter_witness_values': int(changed),
            'changed_deployment_parameter_witness_values': int(changed_deployment),
            'parameter_witness_update_l2': float(update_squared.sqrt()),
            'visual_sha256_before': frozen_before, 'visual_sha256_after': frozen_after,
            'position_buffer_sha256_before': position_before,
            'position_buffer_sha256_after': position_after,
            'position_buffer_dtype': 'float32 (official initialization preserved)',
            'optimizer_saved': bool(resume), 'optimizer_checkpoint_step': start_step if resume else None,
            'resume_performed': bool(resume), 'update_witness_scope': 'final training segment',
            'checkpoint': 'checkpoint-final',
            'elapsed_seconds': time.perf_counter() - started,
            'save_seconds': time.perf_counter() - save_started,
            'test_split_used_for_optimizer_or_model_selection': False,
            'reference_provenance': config.get('reference_provenance', 'model_generated_reference')})
        (out / '_TRAIN_COMPLETE').write_text(f'{len(records)} training steps and checkpoint save completed\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for argument in ['model', 'data', 'references', 'config', 'out']:
        parser.add_argument('--' + argument, required=True)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    train(args.model, args.data, args.references, args.config, args.out, args.resume, args.stop_after)
