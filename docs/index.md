**English** · [简体中文](index.zh-CN.md)

# mJev · Multimodal Decisions

One shared media context. Multiple questions. Direct candidate probabilities.

mJev uses the official Qwen3-VL-4B and Qwen3-Omni Thinker models to provide a common scoring interface for image, video and audio questions with dynamic candidate sets. It reads candidate-label scores directly from the original LM Head.

## When to use it

- Answer multiple fixed-choice questions about the same image, video or audio.
- Obtain raw scores, normalized candidate probabilities and a reproducible decision rule.
- Compare ordinary causal attention, candidate isolation and prefix caching under controlled conditions.

Start with the Qwen3-VL-4B image/video example validated on one GPU; switch to Omni when you need audio. Reuse the shared media prefix while keeping each question's scoring independent.

## Get started

- [README and Quick Start](../README.md#quick-start)
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
