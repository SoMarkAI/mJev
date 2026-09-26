"""Opt-in vLLM 0.25.1 hooks. No scheduler, model weights, or MRoPE changes."""
import contextvars
import functools
import hashlib
import json
import os
from threading import RLock

_ACTIVE = contextvars.ContextVar('jev_runner', default=None)
_INSTALLED = False
_INSTALL_ERROR = None
_INSTALL_LOCK = RLock()

def spec_of(request):
    params = getattr(request, 'pooling_params', None)
    return (getattr(params, 'extra_kwargs', None) or {}).get('mjev')

def batch_route(specs):
    if not any(specs):
        return 'stock'
    if not all(specs):
        raise RuntimeError('Mixed mJev/non-mJev batch unsupported')
    modes = {spec['mode'] for spec in specs}
    if 'stock' in modes and len(modes) != 1:
        raise RuntimeError('Stock baseline cannot be co-batched with custom attention')
    return 'stock' if modes == {'stock'} else 'custom'

def visibility(torch, qpositions, nkeys, spans, mode='isolated'):
    keys = torch.arange(nkeys, device=qpositions.device)
    mask = keys[None, :] <= qpositions[:, None]
    if mode == 'isolated':
        common_end = spans[0][0]
        for start, end in spans:
            rows = (qpositions >= start) & (qpositions < end)
            allowed = (keys < common_end) | ((keys >= start) & (keys < end))
            mask &= ~rows[:, None] | allowed[None, :]
    elif mode not in ('causal', 'stock'):
        raise ValueError(mode)
    return mask

def install():
    global _INSTALLED, _INSTALL_ERROR
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        if _INSTALL_ERROR is not None:
            raise RuntimeError('Previous mJev hook installation failed; restart the process before retrying') from _INSTALL_ERROR
        try:
            _install_hooks()
        except BaseException as exc:
            # Partial monkey-patches cannot safely be applied a second time.
            # Fail closed, never report a partially initialized process as ready.
            _INSTALL_ERROR = exc
            raise
        _INSTALLED = True


def _install_hooks():
    import vllm
    if vllm.__version__ != '0.25.1':
        raise RuntimeError('mJev hooks require inspected vLLM 0.25.1')
    from vllm.model_executor.models import ModelRegistry
    ModelRegistry.register_model('MJevThinker', 'mjev.model:MJevThinker')
    ModelRegistry.register_model('MJevVL', 'mjev.model:MJevVL')
    from vllm.v1.engine.input_processor import InputProcessor
    original_process = InputProcessor.process_inputs
    @functools.wraps(original_process)
    def process(self, *args, **kwargs):
        request = original_process(self, *args, **kwargs)
        spec = spec_of(request)
        if spec:
            spec = dict(spec)
            raw = spec.pop('raw_ids')
            start = spec['spans'][0][0]
            delta = len(request.prompt_token_ids) - len(raw)
            if request.prompt_token_ids[start + delta:] != raw[start:]:
                raise ValueError('Official multimodal expansion changed candidate suffix')
            spec['spans'] = [[a + delta, b + delta] for a, b in spec['spans']]
            spec['mask_hash'] = hashlib.sha256(json.dumps(
                [spec['spans'], spec['mode'], 'mjev-v1'], separators=(',', ':')
            ).encode()).hexdigest()
            request.pooling_params.extra_kwargs = {'mjev': spec}
        return request
    InputProcessor.process_inputs = process

    from vllm.v1.core import kv_cache_utils as kv
    old_need, old_keys = kv.need_extra_keys, kv.generate_block_hash_extra_keys
    def need(request):
        return bool(spec_of(request)) or old_need(request)
    def keys(request, start_token_idx, end_token_idx, start_mm_idx):
        extras, mmidx = old_keys(request, start_token_idx, end_token_idx, start_mm_idx)
        spec = spec_of(request)
        if spec and end_token_idx > spec['spans'][0][0]:
            extras = (*(extras or ()), ('mjev-mask-v1', spec['mask_hash']))
        if spec and (trace_path := os.environ.get('MJEV_CACHE_TRACE_PATH')):
            event = {'request_id': request.request_id, 'mode': spec['mode'],
                     'start': start_token_idx, 'end': end_token_idx,
                     'candidate_start': spec['spans'][0][0],
                     'mask_hash': spec['mask_hash'],
                     'extra_keys_digest': hashlib.sha256(repr(extras).encode()).hexdigest()}
            with open(trace_path, 'a') as trace:
                trace.write(json.dumps(event) + '\n')
        return extras, mmidx
    kv.need_extra_keys, kv.generate_block_hash_extra_keys = need, keys

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    old_execute = GPUModelRunner.execute_model
    @functools.wraps(old_execute)
    def execute(self, *args, **kwargs):
        token = _ACTIVE.set((self, {}))
        try:
            return old_execute(self, *args, **kwargs)
        finally:
            _ACTIVE.reset(token)
    GPUModelRunner.execute_model = execute

    import torch
    import torch.nn.functional as F
    # vLLM's batch-invariant RMSNorm still delegates its residual branch to
    # fused_add_rms_norm, whose reduction varies with packed batch shape.
    # Preserve the unrounded FP32 sum for normalization, then store the
    # residual in its original dtype, matching the fused operation's contract.
    from vllm import envs
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant
    original_norm = RMSNorm.forward_cuda
    def norm(self, x, residual=None):
        if envs.VLLM_BATCH_INVARIANT and residual is not None:
            summed=x.float()+residual.float()
            normalized=rms_norm_batch_invariant(summed,self.weight.data,self.variance_epsilon)
            return normalized.to(x.dtype),summed.to(residual.dtype)
        return original_norm(self,x,residual)
    RMSNorm.forward_cuda=norm
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    old_forward = TritonAttentionImpl.forward
    @functools.wraps(old_forward)
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        active = _ACTIVE.get()
        if active is None or attn_metadata is None or self.attn_type != 'decoder':
            return old_forward(self, layer, query, key, value, kv_cache,
                               attn_metadata, output, output_scale, output_block_scale)
        runner, memo = active
        req_ids = runner.input_batch.req_ids
        specs = [spec_of(runner.requests[r]) for r in req_ids]
        if batch_route(specs) == 'stock':
            return old_forward(self, layer, query, key, value, kv_cache,
                               attn_metadata, output, output_scale, output_block_scale)
        if self.kv_cache_dtype not in ('auto', 'bfloat16', 'float16') or self.sliding_window != (-1, -1):
            raise RuntimeError('Only unquantized, full-context KV supported')
        assert output_scale is None and output_block_scale is None
        assert not attn_metadata.use_cascade
        if 'layout' not in memo:
            memo['layout'] = (attn_metadata.query_start_loc.cpu().tolist(),
                              attn_metadata.seq_lens.cpu().tolist())
            trace_path = os.environ.get('MJEV_BATCH_TRACE_PATH')
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if trace_path and rank == 0:
                starts, lengths = memo['layout']
                with open(trace_path, 'a') as trace:
                    trace.write(json.dumps({'request_ids': list(req_ids),
                        'query_tokens': [b-a for a,b in zip(starts, starts[1:])],
                        'sequence_lengths': lengths}) + '\n')
        starts, lens = memo['layout']
        kc, vc = kv_cache.unbind(1)
        block_size = kc.shape[1]
        for i, spec in enumerate(specs):
            a, b, n = starts[i], starts[i + 1], lens[i]
            if a == b:
                continue
            memo.setdefault('custom_layers', {}).setdefault(req_ids[i], set()).add(layer.layer_name)
            memo_key = (i, query.device, block_size)
            if memo_key not in memo:
                positions = torch.arange(n, device=query.device)
                blocks = attn_metadata.block_table[i, positions // block_size].long()
                slots = positions % block_size
                mask = visibility(torch, torch.arange(n - (b - a), n, device=query.device),
                                  n, spec['spans'], spec['mode'])
                memo[memo_key] = blocks, slots, mask
            blocks, slots, mask = memo[memo_key]
            k = kc[blocks, slots].transpose(0, 1).unsqueeze(0)
            v = vc[blocks, slots].transpose(0, 1).unsqueeze(0)
            q = query[a:b].transpose(0, 1).unsqueeze(0)
            if envs.VLLM_BATCH_INVARIANT:
                from .stable_attention import aligned_attention
                z=aligned_attention(q,k,v,mask,n-(b-a),self.scale)
            else:
                z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                        dropout_p=0.0, is_causal=False, scale=self.scale, enable_gqa=True)
            output[a:b].copy_(z.squeeze(0).transpose(0, 1).reshape_as(output[a:b]))
        return output
    TritonAttentionImpl.forward = forward
