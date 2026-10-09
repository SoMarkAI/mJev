# Evaluation

[Project](../../README.md) · [Recorded study](results.md)

## Run

```bash
python -m docjev.evaluate --model models/Qwen3-VL-4B-Instruct \
  --input examples/docjev/multiple.json --protocol circular --output outputs/evaluation.json
```

Inputs can be a labelled document JSON or JSONL with one document per line, each carrying `questions[]`. Flat Jev records also work. Images resolve relative to the input file. The evaluator never sends the reference label, language or metadata as prompt text.

The output report contains original-order metrics, all-variant metrics, language breakdowns, label histograms and semantic consistency. A companion `.predictions.jsonl` records order mappings, raw logits, probabilities and semantic predictions. Local image/server paths are not copied into prediction records.

Offline rescoring requires no model:

```bash
python -m docjev.evaluate --predictions outputs/evaluation.predictions.jsonl \
  --output outputs/recomputed.json
```

## docjev-bench

[docjev-bench · v2](https://huggingface.co/datasets/Immortal-Zhang/docjev-bench) contains the unified document diagnostic: 220 images, 1,512 paired Chinese/English questions and 3,024 language records. Images come from Infinity and OLM-TFR.

Use its `grouped.jsonl` as `--input`, keeping the accompanying `images/` directory beside it. The Hugging Face dataset exposes a `validation` split in both the default `questions` configuration (3,024 records) and the `documents` configuration (220 image groups). Use `--protocol none` for the original candidate order reported in the [comparison](results.md#unified-1512-question-evaluation), or `--protocol circular` for a separate candidate-order evaluation.

The first RLCD study's 40-image / 500-record validation cohort remains available at the pinned [pilot-v1 revision](https://huggingface.co/datasets/Immortal-Zhang/docjev-bench/tree/f3848f96b0553d08c80cf6115dd3db0d64e2bc4e). Set `revision="f3848f96b0553d08c80cf6115dd3db0d64e2bc4e"` when loading that specific cohort; the default branch contains v2.

## Metrics

| Metric | Definition |
| :--- | :--- |
| Accuracy | Fraction whose highest-probability candidate matches the reference |
| Permutation Consistency | Fraction of questions selecting the same semantic candidate in every tested order |
| All-variants Accuracy | Fraction correct for every tested order |
| Identity Agreement | Fraction of nonidentity predictions matching their original-order semantic prediction |


Consistency is not accuracy: consistently selecting the wrong candidate scores 100% consistency and 0% all-variants accuracy. Original-only evaluations contain no order intervention, so their consistency of 1 is trivial and should not be reported as robustness evidence.

## Candidate order

- `--protocol none`: original order only; appropriate for reproducing the first study.
- `--protocol circular`: all K cyclic rotations, including identity. Every semantic candidate visits every position. This is not all K! permutations.
- `--protocol random --random-count 4 --seed 42`: identity plus up to four distinct seeded random orders. Small candidate sets may exhaust the available permutations first.

Labels are reassigned for each order. Predictions are mapped back to original zero-based candidate indices before consistency or reference comparisons. The original question and candidate strings never change.

## Benchmark design

Split by exact image content and document/paper components. All questions and language versions from one image stay in one split. Keep independent test data separate from both gradient updates and model selection. SHA256 detects exact duplicates; perceptual duplicates and document-level leakage require additional checks before claiming an independent benchmark.

Inspect label histograms by candidate count. A global A/B histogram can hide biases caused by varying candidate counts. Use CircularEval to evaluate position effects rather than silently rewriting existing annotations.

The public repository includes one Infinity validation image and its eight language records as a demo; it does not distribute the full training or validation dataset. Recorded validation pairs are correlated language versions; 500 rows are not 500 independent semantic questions.

## Compare original and trained weights

Use the same labelled JSON/JSONL input for both complete local checkpoints:

```bash
python -m docjev.compare --base-model models/Qwen3-VL-4B-Instruct \
  --trained-model /path/to/mjev-doc \
  --input examples/docjev/multiple.json --out outputs/docjev-comparison
```

The comparison runs each model sequentially on one visible GPU. It freezes image/input hashes, checks every actual processor tensor and prompt for equality, saves raw candidate logits and probabilities, and recomputes paired Accuracy. It uses causal/native/full projection, temperature 1, original candidate order and no prefix cache. A failure preserves partial files without `_SUCCESS`; use a fresh output directory for a new run. Keep local outputs private when the input contains private media or questions.

## Unified diagnostic probability readout

The [1,512-question report](results.md#unified-1512-question-evaluation) covers Chinese and English versions of each question, totaling 3,024 language records. Each model records every candidate's raw label logit and temperature-1 softmax probability. The combined report and subgroup counts are [machine-readable](../validation_evidence/mjev-doc-unified1512.json). This is one fixed diagnostic, separate from the original 500-record training-validation cohort.

The package and console entrypoints retain their `docjev` names for compatibility; the document project is displayed as **mJev-Doc**, with published weights at `SoMarkAI/mjev-doc`.
