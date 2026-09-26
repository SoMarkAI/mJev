import asyncio
import uuid
import copy
import hashlib
import json
import os
from pathlib import Path
import torch
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from .protocol import Protocol
from .families import detect, VL

class MJevEngine:
    def __init__(self, model, *, enable_av=False, enable_prefix_caching=True, max_model_len=None, mm_processor_cache_gb=None, tensor_parallel_size=None):
        # Spawned vLLM workers must load this checkout's opt-in hooks, not an
        # older installed package or a container's original working directory.
        root=str(Path(__file__).resolve().parents[1])
        paths=[p for p in os.environ.get('PYTHONPATH','').split(os.pathsep) if p and p!=root]
        os.environ['PYTHONPATH']=os.pathsep.join([root,*paths])
        os.environ['MJEV_ENABLE']='1'
        os.environ['VLLM_USE_V2_MODEL_RUNNER']='0'
        from .patch import install
        install()
        kind=detect(model)
        self.model_family=kind
        self.protocol = Protocol(model)
        tp=tensor_parallel_size if tensor_parallel_size is not None else (1 if kind==VL else 4)
        args = AsyncEngineArgs(
            model=model, runner='pooling', convert='none',
            hf_overrides={'architectures': ['MJevVL' if kind==VL else 'MJevThinker']},
            tensor_parallel_size=tp, dtype='bfloat16',
            max_model_len=max_model_len or (8192 if enable_av else 4096), max_num_seqs=8, max_num_batched_tokens=8192,
            gpu_memory_utilization=0.88, enforce_eager=True,
            enable_prefix_caching=enable_prefix_caching, enable_chunked_prefill=False,
            async_scheduling=False, attention_backend='TRITON_ATTN',
            limit_mm_per_prompt=({'image':1,'video':int(enable_av)} if kind==VL else {'image':1,'audio':int(enable_av),'video':int(enable_av)}),
            mm_processor_kwargs={'min_pixels': 3136, 'max_pixels': 262144},
            disable_log_stats=False,
        )
        if mm_processor_cache_gb is not None:args.mm_processor_cache_gb=mm_processor_cache_gb
        self.engine = AsyncLLM.from_engine_args(args)
        # Guard: any accidental generation call fails immediately.
        def forbidden(*args, **kwargs):
            raise AssertionError('generate() is forbidden in this prototype')
        self.engine.generate = forbidden

    async def score(self, image, question, candidates, context='', mode='causal', debug=False):
        prepared = self.protocol.build(image, context, question, candidates, mode, debug)
        return await self.score_prepared(prepared, question, candidates, mode=mode, debug=debug)

    async def score_media(self, media_path, question, candidates, *, modality, context='',
                          mode='causal', video_options=None, debug=False):
        prepared = self.protocol.build(media_path, context, question, candidates, mode, debug,
                                       modality=modality, video_options=video_options)
        return await self.score_prepared(prepared, question, candidates, mode=mode, debug=debug)

    async def score_prepared(self, prepared, question, candidates, *, mode='causal', debug=False):
        from .inputs import normalize_question
        candidates = normalize_question({'question': question, 'candidates': candidates})['candidates']
        original_prompt, original_params, labels, token_ids = prepared
        prompt = {**original_prompt, 'multi_modal_data': dict(original_prompt['multi_modal_data'])}
        params = copy.deepcopy(original_params)
        params.extra_kwargs['mjev']['mode'] = mode
        params.extra_kwargs['mjev']['debug'] = debug
        params.skip_reading_prefix_cache = bool(params.skip_reading_prefix_cache or debug)
        request_id = 'mjev-' + uuid.uuid4().hex
        output = None
        async for output in self.engine.encode(prompt, params, request_id):
            pass
        if output is None:
            raise RuntimeError('No pooling output')
        values = output.outputs.data.float().cpu()
        count = len(candidates)
        raw = values[:count]
        probabilities = torch.softmax(raw, dim=0)
        from .decision import choose
        result = {'model_family':self.model_family,'request_id': request_id, 'question': question,
                  'numerics': 'stable' if os.environ.get('VLLM_BATCH_INVARIANT')=='1' else 'native',
                  'mode': mode, 'num_cached_tokens': int(values[count]),
                  'candidates': [dict(label=label, text=text, token_id=token,
                      raw_logit=float(logit), probability=float(prob))
                      for label, text, token, logit, prob in
                      zip(labels, candidates, token_ids, raw, probabilities)],
                  'probability_sum': float(probabilities.sum()),
                  'prompt_tokens': len(output.prompt_token_ids),
                  'prompt_token_ids_sha256': hashlib.sha256(json.dumps(list(output.prompt_token_ids),separators=(',',':')).encode()).hexdigest(),
                  'candidate_spans': [[a + len(output.prompt_token_ids) - len(params.extra_kwargs['mjev']['raw_ids']),
                                       b + len(output.prompt_token_ids) - len(params.extra_kwargs['mjev']['raw_ids'])]
                                      for a,b in params.extra_kwargs['mjev']['spans']]}
        result['decision'] = choose(result['candidates'])
        if debug:
            result['candidate_hidden'] = values[count + 1:].reshape(count, -1).tolist()
        return result

    async def score_questions(self, image, questions, context='', mode='causal'):
        from .inputs import normalize_questions
        questions = normalize_questions(questions)
        return await asyncio.gather(*(self.score(image, q['question'], q['candidates'], context, mode=mode)
                                      for q in questions))
    def close(self):
        self.engine.shutdown()
