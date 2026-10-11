# Hugging Face inference

[简体中文](hf.zh-CN.md) · [Document model](docjev/README.md) · [Service](docjev/serving.md)

Linux, Python 3.11+ and an NVIDIA GPU. The package pins PyTorch 2.11.0 and Transformers 5.13.1. Scoring uses native model forwards to return candidate logits.

<a id="models"></a>

## Models and installation

- `SoMarkAI/mjev-doc`: document images; follow the [document guide](docjev/README.md).
- `SoMarkAI/mJev-Qwen3-VL-4B-RLCD`: general images and video, based on Qwen3-VL.
- `Qwen/Qwen3-Omni-30B-A3B-Instruct`: also handles audio and audio-video. Its historical configuration uses four 24 GB GPUs.

Inside a virtual environment; video also requires system FFmpeg:

```bash
python -m pip install torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf-vl]' huggingface_hub
hf download SoMarkAI/mJev-Qwen3-VL-4B-RLCD --local-dir models/mJev
```

For Omni, install `python -m pip install -e '.[hf]'` and download the complete Omni checkpoint. Keep configuration, tokenizer, processor, chat template and every weight shard.

## Images and video

```bash
python demo_hf.py --model models/mJev --input examples/multiple.json \
  --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 2 --output outputs/image-decisions.json
python demo_hf.py --model models/mJev --input examples/motion-demo/input.json \
  --numerics stable --projection full --output outputs/video-decisions.json
```

Paths resolve relative to the input JSON; see [examples](../examples/README.md). `--check-only` loads the processor/tokenizer, decodes media and checks the actual input without loading model weights. The default limit is 4000 tokens; oversized inputs fail.

Use the `audio`, `video` or `audio_video` input fields. `video_options` can specify `fps`, `min_pixels`, `max_pixels` and `max_frames`. Qwen3-VL rejects audio tasks rather than discarding their audio; `audio_video` requires Omni.

## Scoring and caching

The default mode is `causal`; `isolated` explicitly enables within-question candidate isolation. Labels must be single tokens at the actual template boundary. Results include raw logits, candidate softmax and deterministic tie handling.

`stable` is the general HF default numerical profile. Hold numerics and projection fixed when comparing batch and prefix caching; see [numerics and caching](stability.md). `--projection full` computes the full vocabulary projection; the default `selected` computes candidate rows. Configure `--device-map`, `--dtype` and `--max-input-tokens` explicitly when needed.

## Evaluation and validation

[General evaluation](benchmark.md) preserves original labels and candidate order. [Document evaluation](docjev/evaluation.md) accepts local checkpoints and tests candidate permutations.

```bash
PYTHONPATH=. python benchmarks/integrated/validate_model.py \
  --manifest data/visual-smoke/manifest.jsonl --config configs/qwen3_vl_4b_hf.json \
  --out outputs/cache-check
```

First generate the smoke inputs described in [general evaluation](benchmark.md). This probe uses real weights to compare serial, batch and cache results and record timings. Synthetic questions do not establish benchmark accuracy. CPU mechanism checks are in the [development guide](development.md).
