**English** · [简体中文](validation_current.zh-CN.md)

# Current validation entry point

Latest controlled performance evidence: [HF prefix-cache scaling](cache_scaling.md): one public animation, 63/2,266-token prefixes, 1/3/8/16 questions, five repeats per configuration. Includes full prefill, memory and same-input parity checks; not an accuracy benchmark.

Latest software fixes and regression checks: [reliability review](reliability_review.md). Attention defaults are now consistently `causal`; prior GPU records retain their own configurations.

This is the authoritative **index of evidence and its scope**, not a blanket success certificate. The latest model expansion is the [Qwen3-VL-4B HF/vLLM validation](qwen3_vl_validation.md), including cache/batch checks and an Omni HF regression. The earlier [clean Linux HF verification](clean_linux_validation.md) completed the public six-question workflow. Its code base is `04cc2e0a8fd723d678f44b0c69c2e74b4fe94ea4` plus the installation/provenance fixes identified by actual source hashes; the unmodified commit is not claimed to pass.

## Implementation boundaries

| Path | Execution | Numerical/cache scope | Evidence |
| --- | --- | --- | --- |
| HF main | Official Omni Thinker / Qwen3-VL `forward`, no vLLM or `generate` | Explicit stable/native; cloned per-question prefix KV, real row batches | [HF](hf.md), [numerical regression](stability.md) |
| vLLM pooling main | `AsyncLLM.encode`, original LM Head; custom hooks | Fixed supported vLLM, Omni TP4 / VL TP1; mask-aware APC, dense SDPA | [Deployment](vllm.md), [numerical regression](stability.md) |
| Experimental HTTP / Tree-KV | vLLM single-step `generate` used as score transport | Different scheduler/block-table path; full-model Tree-KV/dense parity pending | [Experimental runtime](integrated.md) |
| Historical AV pilot / release packaging | Evidence from their recorded stages only | “GPU pending” refers to that historical stage, not current HF/pooling capability | [AV pilot](av_pilot.md), [0.1.0 packaging](validation.md) |

Do not transfer main-backend validation claims to HTTP/Tree-KV. No model weights, attention equations or candidate-scoring behavior were changed in this repair.

## Earlier Omni Linux verification (2026-09-26)

Clean Python 3.11 installation, dependency check, four-GPU CUDA witness, 105 CPU tests (3 optional skips), original H.264/AV1 decoding, 6/6 official-model HF predictions and report generation passed. Full input lengths were 920–1,255 tokens. Image demo processor preflight also passed; full image inference and new vLLM/cache regressions were not run. See [complete results and retained failures](clean_linux_validation.md) and [machine-readable evidence](validation_evidence/linux-clean-install.json).

## Earlier macOS repair checks (2026-09-26)

The table below is the earlier host-specific record, not the latest Linux status.

| Check | Actual outcome |
| --- | --- |
| Fresh Python 3.11 macOS ARM64 environment, documented HF extra | **Failed dependency resolution:** no decord distribution for this platform |
| Same isolated environment, `pip install -e '.[test]'` | Passed; base/tiny-CPU test dependencies installed; `pip check` clean |
| Unified CPU pytest | **105 passed, 3 module skips** for absent optional vLLM; one expected native-cache warning |
| Suite collection | base 44, HF 47, runtime 14; vLLM-only collection skipped both modules and exited 5; GPU wrapper collected 1 test, not executed |
| Minimal CLI and tiny model | `demo_hf.py --help`, public mini CLI and tiny random-weight HF forwards passed; no official-weight demo inference |
| Public MMOU preparation | 2 selected videos / 6 original questions / 834,580 media bytes downloaded and receipt/manifest hashes verified |
| Public inference/report contracts | Tested with an explicitly fake scorer; preserved labels, excluded references from input, rejected incomplete reports; **not measured model accuracy** |
| Public official-model HF / vLLM inference | **Not run:** no NVIDIA CUDA device on this host; official weights were not downloaded for this check |
| Linux installation / Docker build | **Not run:** this host is macOS ARM64 and Docker daemon is unavailable |
| Complete media decode / processor preflight | **Not run:** FFmpeg and complete HF AV dependency stack unavailable |

Exact installed versions: [package inventory](validation_evidence/macos-arm64-python311-packages.json). Reproduction commands and limitations: [installation check](installation_validation.md), [test suites](testing.md), [public mini workflow](reproduce.md). Download integrity is not full decode, and tiny random-weight correctness is not pretrained accuracy.

## Historical results: preserve, do not reinterpret

- [Uncached paired evaluation](paired_backends.md): reported BF16 weights, HF layer placement vs vLLM TP4, serial/no cache/full readout. The table remains unchanged. Exact model revision, full environment lock and explicit numerical-profile evidence are **unverified from the public repository**. Do not assign the current stable default to these scores.
- [Stable numerical regression](stability.md): reported HF stable / vLLM invariant profiles and a small 27-question regression, not accuracy. Exact historical model revision and full environment lock remain unverified here; this repair did not rerun the GPU experiment.
- [Native cache failure](hf.md): remains a historical failed acceptance, not silently superseded by successful tiny CPU tests.
- [Original data audit](benchmark_validation.md): dataset integrity only; the newer public mini is a different selection.

New local-bundle runs require explicit numerics and a declared model commit, write actual source/dependency evidence, and mark local revision identity unverified. Reports without a `run.json` retain `unverified_historical_configuration`. The new public workflow resolves a pinned Hub revision directly. Neither mechanism retroactively verifies old artifacts.
