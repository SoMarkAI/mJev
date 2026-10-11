# Numerics and prefix caching

[简体中文](stability.zh-CN.md) · [HF](hf.md) · [vLLM](vllm.md) · [Measurements](cache_scaling.md)

## HF

General HF defaults to `stable`; the document model defaults to `native`, one question and no cache. Stable numerics preserve loaded parameter dtype, use FP32 text computation and align attention/projection blocks. Results can differ from native BF16. Hold numerics, projection and media settings fixed in every comparison.

```bash
python demo_hf.py --model /path/to/model --input examples/multiple.json \
  --numerics stable --projection full --question-batch-size 2 --prefix-cache
```

Each question occupies an independent batch row with its own copied prefix KV branch. Questions cannot read other question answers. Prefix reuse saves computation but consumes cache memory; it is not zero-copy. The document HTTP service also supports worker-local cross-request caching; see [serving](docjev/serving.md).

`causal` is the default within-question attention mode. `isolated` explicitly restricts visibility between candidates of the same question. Independent question branches and within-question candidate isolation are distinct mechanisms.

## vLLM

Set `VLLM_BATCH_INVARIANT=1` before starting the process. mJev hooks combine stable linear/MoE kernels, RMSNorm and aligned SDPA while preserving scheduling and candidate masks.

```bash
VLLM_BATCH_INVARIANT=1 python demo.py --model /path/to/model \
  --input examples/multiple.json --mode causal
```

HF and vLLM use different execution paths; cross-backend bitwise equality is not guaranteed. Concurrent cold requests do not necessarily share cache hits.

## Validation

[Development tests](development.md#tests) check tiny random-model mechanisms. The [cache probe](hf.md#evaluation-and-validation) compares serial, batch and cache execution with real weights. Report maximum logit/probability deltas, decision agreement and timing boundaries, rather than decision agreement alone.

The [fixed historical regression record](https://github.com/SoMarkAI/mJev/tree/a313f135041cf893db27a81f40052b9e66b2b5a2/docs/stability.md) retains its original model, dependencies, samples and hardware scope, not a claim about every deployment.
