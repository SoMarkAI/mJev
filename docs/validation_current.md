**English** &nbsp;&nbsp; [简体中文](validation_current.zh-CN.md)

# Results and validation scope

## Model evaluation

| Evaluation | Base Qwen | Trained | Details |
| --- | ---: | ---: | --- |
| mJev-Doc: 3,024 Chinese/English records | 82.04% | 84.36% | [Protocol and results](docjev/results.md), [breakdown](validation_evidence/mjev-doc-unified1512.json) |
| General mJev: 195 questions | 77.95% | 80.00% | [Training and evaluation](training.md) |

The rows use different datasets. The document study contains 1,512 bilingual question pairs, separate from the 500-record training-validation comparison. It uses identical processed inputs, causal/native BF16, full LM Head projection and temperature 1. The languages are paired observations, not independent samples.

## Inference and performance

| Path | Existing checks | Scope |
| --- | --- | --- |
| HF image service | Real tiny Qwen3-VL forwards, HTTP/spawn, dynamic batching and mixed-length prefix caches | CPU mechanism checks; full 4B service throughput and Windows CUDA remain untested. [Service guide](docjev/serving.md) |
| HF main API and vLLM pooling | Qwen3-VL-4B image/video, cache, batches and Omni HF regression | Applies to the models and configurations in each [validation record](qwen3_vl_validation.md). |
| HF cache performance | 63/2,266-token prefixes; 1/3/8/16 questions; five repeats per configuration | Long-prefix, 16-question case: 32.62→6.29 seconds. Single-question cases are slower; peak memory increases. [Measurements](cache_scaling.md) |
| Experimental vLLM HTTP / Tree-KV | Separate implementation and tests | Main API results do not establish full-model validation for this path. [Experimental runtime](integrated.md) |

`stable` and `native` are different numerical profiles. Fix inputs and timing boundaries when comparing profiles, backends or cache strategies. The historical native-cache acceptance failure remains documented in the [HF guide](hf.md); tiny-model tests do not replace full-weight validation.

## Historical records

These reports retain their original environments and limitations. They are not new GPU evaluations of the current version.

- [Paired backend evaluation](paired_backends.md), [numerical stability](stability.md): some historical model revisions and full environment locks remain unverified; do not assign current defaults to them.
- [Linux installation and official-model verification](clean_linux_validation.md), [earlier macOS installation checks](installation_validation.md)
- [Software regression and fixes](reliability_review.md)
- [Original dataset integrity checks](benchmark_validation.md), [audio/video pilot](av_pilot.md), [first release validation](validation.md)

To rerun checks, use the [test commands](testing.md) or [public evaluation workflow](reproduce.md).
