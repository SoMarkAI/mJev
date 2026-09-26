**English** · [简体中文](integrated.zh-CN.md)

# Integrated experimental runtime

> See the [current validation index](validation_current.md) for backend boundaries and latest checks. Historical results retain their recorded configuration; missing model revisions, complete dependencies or run settings are unverified, never inferred from current defaults.

Composition: main `ca755f8` reference package/evaluator plus mJev `59b38ea`
service/Tree-KV runtime, with shared deterministic decision and stable candidate
softmax. The `mjev/` dense-SDPA reference remains available alongside the runtime.
Service causal is default; masked Tree-KV is experimental. Runtime and reference
hooks must not be enabled simultaneously. Runtime still uses vLLM generate as
an internal single-step score transport, not a text decoding loop; it does NOT
satisfy a literal prohibition on calling generate. There is no new model head
or model training.

The CPU selective-projection test compares with full FP32 projection of the
same BF16 weights, not a full 30B/Tensor Parallel parity certificate. Full-model
Tree-KV versus dense-mask parity remains pending. Different prompts/backends
must not be conflated as an exact logits comparison.

## CPU verification

```bash
docker run --rm -w /workspace -e PYTHONPATH=/workspace/runtime:/workspace \
  -v "$PWD:/workspace:ro" --entrypoint python3 mjev-av-pilot:0.1 \
  -m pytest -q -p no:cacheprovider tests/runtime
```

## GPU deployment

Only after four GPUs have been made available:

```bash
export MODEL_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct
export DATA_DIR=/path/to/mjev_multiquestion_le4000_v1
bash scripts/serve_integrated.sh
curl -f http://127.0.0.1:17005/health
```

The existing `mjev-av-pilot:0.1` reference image must be present (otherwise
run `docker build -f experiments/av/Dockerfile -t mjev-av-pilot:0.1 .` from the repository root). This launcher creates a separate container and never
stops an existing service. It binds the API to loopback.

## Evaluation

Run within the reference CPU image with code on PYTHONPATH and data readable:

```bash
python3 benchmarks/integrated/evaluate_service.py --root /data --out /results/prepared --prepare-only
python3 benchmarks/integrated/evaluate_service.py --root /data --out /results/smoke --limit-per-modality 2
python3 benchmarks/integrated/evaluate_service.py --root /data --out /results/full
```

Outputs must be new directories. Errors stop immediately with an ERROR.json;
no retries or silent omissions. The runner sends only media, question, choices
and IDs. 374 native/disagreeing Clotho questions are excluded from MCQ metrics;
226 unanimous original yes/no questions use the separately documented adapter.
Audio-video requests explicitly enable the native audio-in-video path.
The prior <=4000 lengths apply to the original processor/template settings:
they are not a new-runtime length certificate. No trimming is introduced;
overlength or decoder errors must be investigated, not silently dropped.
Accuracy/NLL/Brier/ECE are per-question measures, not official leaderboard scores.
Reported latency is client wall time per grouped request, not per question.
The two modes use alternating order; these are not certified cold-cache timings.

The HTTP evaluator now records client source and dependencies. Server model revision, numerics and dependencies are not attested and remain explicitly unverified_not_attested; this is not complete server reproduction evidence.

## Media access and startup checks

The experimental HTTP service enforces its own media boundary before vLLM sees a request:

- Local paths and `file:` URLs must resolve inside `MJEV_ALLOWED_MEDIA_ROOTS` (default `/data`; multiple roots use the platform path separator). Regular files only; symlink escapes are rejected. The launch script mounts `/data` read-only.
- Remote reads are disabled by default. To opt in, set `MJEV_ALLOWED_MEDIA_HOSTS` to comma-separated exact hostnames. Only HTTPS on port 443 is accepted; credentials, non-public IP destinations and redirects are rejected. The connection uses the validated IP with hostname-verified TLS.
- `MJEV_MAX_MEDIA_BYTES` defaults to 536870912 bytes per item. File sizes and encoded data-URL lengths are checked before reading/decoding; reads are bounded. vLLM receives the validated bytes rather than reopening a path or fetching again.

Startup requires vLLM 0.25.1 and all runtime hooks. Hook failure aborts the process, including spawned Python workers. `/v1/mjev/health` checks frontend hook installation; use the standard vLLM `/health` endpoint as well for engine health. These checks are not a substitute for full-model Tree-KV parity validation.
