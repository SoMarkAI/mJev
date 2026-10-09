**English** · [简体中文](av_pilot.zh-CN.md)

# AV pilot (historical pre-GPU validation record)

> Historical results use their recorded configurations. See [current validation](validation_current.md) for later checks and settings that remain unverified.

> Historical pilot status. For subsequent full-weight results, see [paired evaluation](paired_backends.md) and [stability regression](stability.md).

This extends the existing raw LM-head pooling path to audio and audio-video inputs. It does not use generation or change model weights. The official Thinker audio/vision encoders, processor, MRoPE and vLLM scheduler remain in use. The attention patch is unchanged; its candidate suffix guard checks multimodal expansion before shifting spans.

The small test selects 10 original questions with complete media: one sound/music/speech MMAU question each (at most 30 seconds), four questions from the shortest complete Video-MME-v2 group, and three MMOU questions from distinct videos (at most 30 seconds). Selection does not inspect labels. Duration-warning files are excluded from this pilot only. Eight frames with maximum 65536 pixels are sampled across each full video; the entire audio track is used. No evidence timestamps or reference labels reach inference. These are feasibility tests, not representative benchmark scores.

Validation at that historical stage: 19 CPU unit tests pass, and the official processor preflight passes on all 10 selected questions (154–3027 tokens). GPU kernel, scoring, mask isolation and prefix-cache checks remain pending; no AV accuracy result is claimed.

Build from the repository root:

```bash
docker build -f experiments/av/Dockerfile -t mjev-av-pilot:0.1 .
mkdir -p outputs/av-pilot
```

Set `MODEL_DIR` to official local weights and `BENCH_ROOT` to the completed benchmark bundle, then run CPU preflight:

```bash
docker run --rm -v "$MODEL_DIR:/model:ro" -v "$BENCH_ROOT:/bench:ro" \
  -v "$PWD/outputs/av-pilot:/out" mjev-av-pilot:0.1 \
  --model /model --root /bench --output /out/preflight --check-only
```

Only with **four available GPUs**, run the seeded kernel witness and GPU pilot:

```bash
docker run --rm --gpus all --entrypoint python3 mjev-av-pilot:0.1 experiments/av/av_witness.py

docker run --rm --gpus all --ipc=host -e VLLM_BATCH_INVARIANT=0 \
  -v "$MODEL_DIR:/model:ro" -v "$BENCH_ROOT:/bench:ro" \
  -v "$PWD/outputs/av-pilot:/out" mjev-av-pilot:0.1 \
  --model /model --root /bench --output /out/gpu --numerics native --model-revision "$MODEL_REVISION"
```

The pilot compares causal and isolated scoring on identical prepared inputs. It also checks hidden-state isolation with a same-token-length intervention, a causal positive control, prefix-cache reuse, candidate reordering and three concurrent video questions. Synthetic mechanism-test options are excluded from accuracy metrics. A failure stops the run and saves `failure.json`; no automatic retry is performed. `_SUCCESS` is written only after the actual GPU assertions pass.

Reports include Accuracy/NLL/Brier/ECE, raw logits/probabilities and latency. Latency excludes decoding/preparation and model load, includes the encode request, and uses enabled caches with alternating mode order. These ten timings are descriptive only. The AV limit is 8192 tokens (image-only remains 4096); preflight rejects overflow. Longer-media handling is not implemented by silently truncating inputs.

Set MODEL_REVISION to the actual local model commit. Native is an explicit rerun profile, not proof of missing historical settings.
