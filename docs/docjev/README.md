# mjev-doc · Give every page a decision

<!-- mjev-doc-publishing -->
Model configuration and documentation are available; weight publication is pending. Run the download and inference commands once all five shards are present.
<!-- /mjev-doc-publishing -->

[English](README.md) · [简体中文](README.zh-CN.md) · [Back to mJev](../../README.md)

Bring a document image, ask several questions and define meaningful choices. **mjev-doc** specializes the Jev workflow for document understanding: visible attributes, document categories, content relationships and decisions under explicit criteria.

## One project, two checkpoints

| Model | Focus | Weights |
| --- | --- | --- |
| mJev-Qwen3-VL-4B-RLCD | General visual candidate decisions | [Hugging Face](https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD) |
| mjev-doc | Document-image candidate decisions | [Hugging Face](https://huggingface.co/SoMarkAI/mjev-doc) |

Both start from Qwen3-VL-4B-Instruct. The mjev-doc adapter imports the shared `mjev/` HF core rather than maintaining another copy. It preserves the document study's RGB preprocessing, pixel limits and native BF16 readout. Checkpoint names identify different training runs; they are not aliases.

## Try a document

Use Linux, Python 3.11+ and an NVIDIA GPU. Install a driver-compatible PyTorch 2.11 wheel first; the recorded document setup used CUDA 13.0 and a 96 GB Blackwell GPU.

```bash
python -m pip install -e '.[docjev]'
hf auth login
hf download SoMarkAI/mjev-doc --local-dir models/mjev-doc
python demo_docjev.py --model models/mjev-doc \
  --input examples/docjev/multiple.json --check-only
python demo_docjev.py --model models/mjev-doc \
  --input examples/docjev/multiple.json --output outputs/mjev-doc.json
```

**[mjev-doc weights](https://huggingface.co/SoMarkAI/mjev-doc)** include the full checkpoint, tokenizer, processor and chat template. Code, model documentation and evaluation remain in this repository. The example includes one real validation table, eight bilingual language records and a recorded trained-model output. [Browse the example →](../../examples/docjev/README.md)

## Keep asking

```bash
python demo_docjev.py --model /path/to/mjev-doc \
  --input examples/docjev/multiple.json --numerics stable \
  --prefix-cache --question-batch-size 2 --output outputs/docjev-cached.json
```

The shared media/context prefix is prefetched once, then independent question branches are scored. Cache/batch comparisons use `stable` numerics; reported accuracy studies use `native`, serial, uncached inference. Keep these protocols distinct.

## Learn and evaluate

The first mjev-doc training run consumed **15,658 language records / 7,829 bilingual question pairs / 1,223 images** over one epoch on eight GPUs. It freezes vision and updates language parameters with candidate-restricted RLCD training, without LoRA or a new decision head.

| Cohort | Base | mjev-doc |
| --- | ---: | ---: |
| First validation: 500 language records | 73.0% | 77.8% |
| Unified diagnostic: 3,024 language records | 82.04% | 84.36% |

The evaluation covers **1,512 questions** across 220 image files, with Chinese and English versions. Every candidate has a recorded raw logit and temperature-1 softmax probability. Scores measure reference-label agreement; bilingual versions are paired observations. [Complete results and probabilities →](results.md)

- [Installation](installation.md) · [Input/output and API](input-output.md)
- [Shared attention, logits and caching](architecture.md)
- [Evaluation, candidate permutations and paired model comparison](evaluation.md)
- [RLCD reward, stages and resume](training.md)

docjev-bench: [Immortal-Zhang/docjev-bench](https://huggingface.co/datasets/Immortal-Zhang/docjev-bench). Code is Apache-2.0; the example table keeps its [CC-BY-2.0 terms](../../THIRD_PARTY_LICENSES.md).
