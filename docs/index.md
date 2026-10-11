**English** &nbsp;&nbsp; [简体中文](index.zh-CN.md)

# mJev documentation

mJev scores candidates against images, documents and video. mJev-Doc focuses on document classification, field verification and explicit rules. Both models share the inference interface; download their weights separately from Hugging Face.

## Use

| Task | Guide |
| --- | --- |
| Run the document model | [mJev-Doc](docjev/README.md) |
| Prepare inputs and read outputs | [Document API](docjev/input-output.md), [examples](../examples/README.md) |
| Process multiple images and questions | [HF service, dynamic batching and prefix caching](docjev/serving.md) |
| Process general images, video or audio | [Model selection](models.md), [HF deployment](hf.md) |
| Use vLLM | [vLLM deployment](vllm.md) |

## Evaluate and train

- [DocJev-bench: tasks and business scenarios](docjev/benchmark.md)
- [Document evaluation, candidate permutations and model comparison](docjev/evaluation.md)
- [mJev-Doc training](docjev/training.md), [general mJev training](training.md)
- [Public mini evaluation](reproduce.md)
- [Results and validation scope](validation_current.md), [cache performance](cache_scaling.md)

## Develop

- [Code entry points and implementation constraints](development.md)
- [Test commands](testing.md), [numerical stability](stability.md)
- [Contributing](../CONTRIBUTING.md)
- [Experimental vLLM HTTP / Tree-KV](integrated.md), [audio/video tools](../experiments/av/README.md)
