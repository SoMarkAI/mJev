**English** · [简体中文](models.zh-CN.md)

[Recorded official-model validation](qwen3_vl_validation.md)

# Model selection and deployment

mJev selects the native model family from the local `config.json`. Start with the released [mJev-Qwen3-VL-4B-RLCD](https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD) for image/video, or official Qwen3-Omni Thinker weights for audio. Both paths retain the native processor, chat template, visual features, position encoding and LM Head. Scoring uses the existing output layer through `forward` or vLLM pooling.

| Capability | Qwen3-Omni-30B-A3B-Instruct | Qwen3-VL family (4B) |
| --- | --- | --- |
| Image + text | Yes | Yes |
| Video + text | Yes | Yes |
| Audio / video with audio | Yes | **No: rejected explicitly** |
| Dynamic candidates, logits, probabilities, decision | Same interface | Same interface |
| HF causal / candidate-isolated attention | Yes | Yes |
| HF shared-prefix KV + real question batches | Yes | Yes |
| vLLM pooling + mask + prefix cache | Optional backend | Optional backend |
| Default vLLM tensor parallelism | 4 GPUs | 1 GPU |
| Weights | Official, downloaded separately | Released RLCD or official base, downloaded separately |

The released RLCD checkpoint is fine-tuned for candidate selection; its [training and reward design](training.md) is documented separately. The archived GPU validation records and pinned benchmark configurations use **Qwen/Qwen3-VL-4B-Instruct**. Keep the checkpoint identity with each result.

Qwen3-VL is a dense vision-language model. Its smaller parameter count is not a measured speedup guarantee. It does not replace Omni on audio tasks. In particular, do not remove audio from an audio-dependent benchmark and report that as equivalent evaluation.

## Install and run the 4B model

Linux, Python 3.11+, NVIDIA GPU and system FFmpeg are required for the documented GPU/video workflow. Activate a virtual environment; HF does not require vLLM or custom CUDA compilation. The CPU codec decodes video while PyTorch runs model inference on CUDA.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf-vl,test]' huggingface_hub
export MODEL_DIR="$HOME/models/mJev-Qwen3-VL-4B-RLCD"
hf download SoMarkAI/mJev-Qwen3-VL-4B-RLCD --local-dir "$MODEL_DIR"
python demo_hf.py --model "$MODEL_DIR" --input examples/multiple.json --check-only
python demo_hf.py --model "$MODEL_DIR" --input examples/multiple.json \
  --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 2 --output outputs/vl-image.json
```

Use the same `HFMJevEngine` API and input JSON as [HF deployment](hf.md). The model family is detected, not supplied as an unchecked CLI override. `audio` and `audio_video` inputs fail before media decoding. Video sampling retains native frame timestamps and `mm_token_type_ids`; no synthetic replacement position encoding is introduced.

For the optional pinned vLLM environment, follow [vLLM deployment](vllm.md), then run from the checkout:

```bash
VLLM_BATCH_INVARIANT=1 python demo.py --model "$MODEL_DIR" \
  --input examples/multiple.json --mode causal --tensor-parallel-size 1 \
  --output outputs/vl-vllm-image.json
```

The vLLM environment uses the existing custom pooling/attention hooks. This is not a stock OpenAI-compatible vLLM server. Container commands use `python3`.

## Reproducible image + video smoke workflow

This workflow uses the **official base model** pinned in `configs/qwen3_vl_4b_hf.json`, independently of the checkpoint downloaded for the demo above. To download that exact base revision explicitly:

```bash
hf download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17
```

This workflow creates a red rectangle image and a short static video, with three authored questions per media item. It downloads **no third-party media**. The questions and labels are deliberately synthetic integration fixtures; their scores must never be presented as natural-data benchmark accuracy.

```bash
python -m mjev.benchmark.visual_smoke --root data/visual-smoke
python -m mjev.benchmark.grouped --manifest data/visual-smoke/manifest.jsonl \
  --config configs/qwen3_vl_4b_hf.json --out outputs/vl-hf-smoke
```

The grouped runner resolves the fixed Hub revision (using the standard Hub cache), scores all questions, and writes `run.json`, `predictions.jsonl`, `raw.jsonl`, `report.json`, `REPORT.md` and `_SUCCESS`. It records actual source hashes, Git state where available, dependencies, media hashes, numerical settings and timing boundaries. A `--local-dir` demo download is separate from the standard Hub cache used by this runner.

For a real dataset, supply a frozen [unified manifest](benchmark.md), preserving original questions, candidates and labels, with `modality=image` or `video`. The same runner computes Accuracy, NLL, multiclass Brier Score and ECE. Audio-containing manifests are rejected for VL instead of filtered or relabeled. Select a new output directory for each run.

Within the pinned vLLM environment, substitute `configs/qwen3_vl_4b_vllm.json`. Keep both runtime hook flags disabled when starting the grouped runner; it enables the appropriate hooks itself.

## Cache and batch validation

```bash
PYTHONPATH=. python benchmarks/integrated/validate_model.py \
  --manifest data/visual-smoke/manifest.jsonl \
  --config configs/qwen3_vl_4b_hf.json --out outputs/vl-hf-cache \
  --repeats 2 --atol 0.0001
bash scripts/test.sh --suite hf -q
```

The full-weight probe compares serial, batched, cache-serial and cache-batched execution in both causal and isolated modes. It verifies cache reuse, reversed question order, probability sums, decision agreement and maximum raw-logit differences. HF measurements include fresh prefill per call; vLLM can retain warm cache across calls and uses prepared inputs. Those backend timing boundaries differ and are recorded explicitly.

Candidate-order changes may change model preferences; candidate isolation is not permutation invariance. Tiny native-model tests verify that changing one candidate cannot alter another candidate's hidden states under the isolated mask, while the answer suffix can still respond. The official tokenizer checks each label as exactly one token at the actual answer boundary.

## Validation status

See [current validation](validation_current.md) for actual completed checks and limitations. Passing tiny random-weight tests establishes mechanics, not pretrained accuracy. The experimental HTTP/Tree-KV runtime is separate and is not automatically certified for Qwen3-VL by this adapter.

Official model cards: [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct).

### Input and recovery contracts

Both main backends default to `causal`; candidate-isolated attention requires an explicit `isolated` argument. Independent question branches are separate from candidate isolation within one question. Candidate isolation is an experimental structure for controlled comparisons and potential future training, not evidence of a higher achievable accuracy ceiling.

CLI preflight and inference share structural validation with the scoring APIs: candidate dictionaries must use ordered A/B/... labels and question lists must be nonempty. Grouped evaluation durably saves each completed media group in `groups/`, plus `progress.json` and cumulative raw/prediction files. A later failure retains those files without a `_SUCCESS` marker; automatic resume is not implemented.
