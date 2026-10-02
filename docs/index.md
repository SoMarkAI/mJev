**English** · [简体中文](index.zh-CN.md)

# mJev · Multimodal Decisions

One shared media context. Multiple questions. Direct candidate probabilities.

mJev provides a common scoring interface for image, video and audio questions with dynamic candidate sets. Start with the released **mJev-Qwen3-VL-4B-RLCD** for image/video; choose official Qwen3-Omni Thinker weights for audio. Candidate-label scores come directly from the native LM Head.

## When to use it

- Answer multiple fixed-choice questions about the same image, video or audio.
- Obtain raw scores, normalized candidate probabilities and a reproducible decision rule.
- Compare ordinary causal attention, candidate isolation and prefix caching under controlled conditions.

Reuse the shared media prefix while keeping each question's scoring independent. Check [current validation](validation_current.md) for the exact model and configuration behind recorded results.

## Get started

- [README and Quick Start](../README.md#quick-start)
- [Model selection and deployment](models.md)
- [Runnable examples and recorded outputs](../examples/README.md)
- [HF deployment without vLLM](hf.md)
- [vLLM Docker deployment](vllm.md)
- [Developer guide](development.md)
- [Numerical stability and validation scope](stability.md)
- [Contribution guide](../CONTRIBUTING.md)

## Reproduction and current validation

- [Public mini evaluation: prepare → infer → report](reproduce.md)
- [Current validation and backend boundaries](validation_current.md)
- [pytest suites and test entry points](testing.md)
- [Isolated installation checks and limitations](installation_validation.md)

## Explore further

- [Released model: training and reward design](training.md)
- [Cache scaling: timings and measurement conditions](cache_scaling.md)
- [Experimental HTTP / Tree-KV](integrated.md)
- [Historical AV tools and Docker image](../experiments/av/README.md)
