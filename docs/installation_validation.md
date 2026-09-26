**English** · [简体中文](installation_validation.zh-CN.md)

# Isolated installation check — 2026-09-26

**Latest:** the subsequent [clean Linux official-model run](clean_linux_validation.md) completed all six public questions after fixing Python/media dependencies. The record below is the earlier macOS attempt and retains its failures and limits.

Host: macOS ARM64, Python 3.11. The supported full-model deployment target remains Linux/NVIDIA. No existing environment packages were inherited; the venv was created without `--system-site-packages`.

## Actual commands and results

```bash
python3.11 -m venv /tmp/mjev-clean-20260926
/tmp/mjev-clean-20260926/bin/python -m pip install --retries 0 -e '.[hf]' pytest requests pyarrow
```

**Failed**, during dependency resolution: `No matching distribution found for decord; extra == "decord"`. The package was not declared successfully installed with its HF extra. This is not evidence that the Linux install fails; Linux installation was not tested here.

For the independently supported CPU mechanism checks:

```bash
/tmp/mjev-clean-20260926/bin/python -m pip install --retries 0 -e '.[test]'
/tmp/mjev-clean-20260926/bin/python -m pip check
PYTHON=/tmp/mjev-clean-20260926/bin/python bash scripts/test.sh -q
/tmp/mjev-clean-20260926/bin/python demo_hf.py --help
/tmp/mjev-clean-20260926/bin/python -m mjev.benchmark.public_mini --help
```

Final result: installation and `pip check` passed; **105 CPU tests passed, three modules skipped for absent vLLM**. The initial test attempt exposed missing accelerate and an optional runtime vLLM dependency. The test extra now installs accelerate; optional vLLM imports produce visible skips. No numerical assertions were weakened and no fake model was presented as official-weight inference.

Minimal forwards are the tiny random-weight Thinker tests, including checkpoint loading, modalities, masks and cache behavior. `--help` verifies CLI parsing only. The new public data preparation downloaded and hash-verified two videos / six original questions (834,580 media bytes), independent of the internal benchmark bundle. See [public workflow](reproduce.md).

Pinned core versions in this environment are torch 2.11.0, Transformers 5.13.1, pytest 8.4.2 and accelerate 1.15.0. The [full installed package inventory](validation_evidence/macos-arm64-python311-packages.json) is evidence, **not a portable Linux lockfile**. Per-run `run.json` records the actual environment; reinstalling later with unconstrained transitive dependencies may yield a different environment.

## Not executed

- Full HF AV installation, official processor preflight and official 30B demo.
- Official-model public inference, accuracy report and vLLM GPU regressions.
- Docker build or clean Linux installation: Docker daemon unavailable, host is not Linux.
- Full audio/video decode: no FFmpeg or complete AV stack.

Before claiming clean Linux support or publishing new model results, run the documented installation and public pipeline in a fresh Linux/NVIDIA environment, retain its dependency inventory and run evidence, and report any failures without reclassifying this CPU validation as GPU success.
