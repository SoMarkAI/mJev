# DocJev · Give every page a decision

[English](README.md) · [简体中文](README.zh-CN.md) · [Back to mJev](../../README.md)

Bring a document image, ask several questions and define meaningful choices. **DocJev-Qwen3-VL-4B-RLCD** specializes the Jev workflow for document understanding: visible attributes, document categories, content relationships and decisions under explicit criteria.

## One project, two checkpoints

| Model | Focus | Weights |
| --- | --- | --- |
| mJev-Qwen3-VL-4B-RLCD | General visual candidate decisions | [Hugging Face](https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD) |
| DocJev-Qwen3-VL-4B-RLCD | Document-image candidate decisions | Complete local exported checkpoint |

Both start from Qwen3-VL-4B-Instruct. The DocJev adapter imports the shared `mjev/` HF core rather than maintaining another copy. It preserves the document study's RGB preprocessing, pixel limits and native BF16 readout. Checkpoint names identify different training runs; they are not aliases.

## Try a document

Use Linux, Python 3.11+ and an NVIDIA GPU. Install a driver-compatible PyTorch 2.11 wheel first; the recorded document setup used CUDA 13.0 and a 96 GB Blackwell GPU.

```bash
python -m pip install -e '.[docjev]'
hf download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --local-dir models/Qwen3-VL-4B-Instruct
python demo_docjev.py --model models/Qwen3-VL-4B-Instruct \
  --input examples/docjev/multiple.json --check-only
python demo_docjev.py --model models/Qwen3-VL-4B-Instruct \
  --input examples/docjev/multiple.json --output outputs/docjev-base.json
```

To use trained DocJev, replace `--model` with your complete `DocJev-Qwen3-VL-4B-RLCD` checkpoint directory. This repository does not distribute that checkpoint. The example includes one real validation table, eight bilingual language records and a recorded trained-model output. [Browse the example →](../../examples/docjev/README.md)

## Keep asking

```bash
python demo_docjev.py --model /path/to/DocJev-Qwen3-VL-4B-RLCD \
  --input examples/docjev/multiple.json --numerics stable \
  --prefix-cache --question-batch-size 2 --output outputs/docjev-cached.json
```

The shared media/context prefix is prefetched once, then independent question branches are scored. Cache/batch comparisons use `stable` numerics; reported accuracy studies use `native`, serial, uncached inference. Keep these protocols distinct.

## Learn and evaluate

The first DocJev training run consumed **15,658 language records / 7,829 bilingual question pairs / 1,223 images** over one epoch on eight GPUs. It freezes vision and updates language parameters with candidate-restricted GRPO, without LoRA or a new decision head.

| Cohort | Base | DocJev |
| --- | ---: | ---: |
| First validation: 500 language records | 73.0% | 77.8% |
| Multitask diagnostic: 2,984 language records | 82.21% | 84.65% |
| Rotation / paper diagnostic: 40 language records | 70.0% | 62.5% |

These are separate cohorts and reference-label agreement measurements. The visual diagnostic combines deterministic rotation targets with generated-and-screened paper references. The bilingual versions are paired observations, not independent questions. [Detailed results and visual diagnostics →](results.md)

- [Installation](installation.md) · [Input/output and API](input-output.md)
- [Shared attention, logits and caching](architecture.md)
- [Evaluation, candidate permutations and paired model comparison](evaluation.md)
- [RLCD reward, stages and resume](training.md)

DocJev-Bench: [Immortal-Zhang/DocJev-Bench](https://huggingface.co/datasets/Immortal-Zhang/DocJev-Bench). Code is Apache-2.0; the example table keeps its [CC-BY-2.0 terms](../../THIRD_PARTY_LICENSES.md).
