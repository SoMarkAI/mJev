# Evaluation

[Project](../../README.md) &nbsp;&nbsp; [Recorded study](results.md)

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

## DocJev-Bench

[DocJev-Bench](https://huggingface.co/datasets/Immortal-Zhang/DocJev-Bench) contains 608 images, 2617 bilingual pairs and 5234 language records. Use `grouped.jsonl` as `--input`, with the downloaded `images/` directory alongside it. The default Hugging Face `questions` configuration exposes validation and test splits.

Use `--protocol none` for original-order accuracy or `--protocol circular` for a separate candidate-order experiment. [Current scores](results.md) are archived-prediction aggregation on the model-outcome-selected collection, not a fresh blind test.

## Metrics

| Metric | Definition |
| :--- | :--- |
| Accuracy | Fraction whose highest-probability candidate matches the reference |
| Permutation Consistency | Fraction of questions selecting the same semantic candidate in every tested order |
| All-variants Accuracy | Fraction correct for every tested order |
| Identity Agreement | Fraction of nonidentity predictions matching their original-order semantic prediction |


Consistency is not accuracy: consistently selecting the wrong candidate scores 100% consistency and 0% all-variants accuracy. Original-only evaluations contain no order intervention, so their consistency of 1 is trivial and should not be reported as robustness evidence.

## Candidate order

- `--protocol none`: original order only; original-order scoring.
- `--protocol circular`: all K cyclic rotations, including identity. Every semantic candidate visits every position. This is not all K! permutations.
- `--protocol random --random-count 4 --seed 42`: identity plus up to four distinct seeded random orders. Small candidate sets may exhaust the available permutations first.

Labels are reassigned for each order. Predictions are mapped back to original zero-based candidate indices before consistency or reference comparisons. The original question and candidate strings never change.

## Benchmark design

Split by exact image content and document/paper components. All questions and language versions from one image stay in one split. Keep independent test data separate from both gradient updates and model selection. SHA256 detects exact duplicates; perceptual duplicates and document-level leakage require additional checks before claiming an independent benchmark.

Inspect label histograms by candidate count. A global A/B histogram can hide biases caused by varying candidate counts. Use CircularEval to evaluate position effects rather than silently rewriting existing annotations.

The repository includes one licensed document image and eight language records as a runnable example. The full dataset is hosted separately on Hugging Face; bilingual records are correlated.

## Compare original and trained weights

Use the same labelled JSON/JSONL input for both complete local checkpoints:

```bash
python -m docjev.compare --base-model models/Qwen3-VL-4B-Instruct \
  --trained-model /path/to/mjev-doc \
  --input examples/docjev/multiple.json --out outputs/docjev-comparison
```

The comparison runs each model sequentially on one visible GPU. It freezes image/input hashes, checks every actual processor tensor and prompt for equality, saves raw candidate logits and probabilities, and recomputes paired Accuracy. It uses causal/native/full projection, temperature 1, original candidate order and no prefix cache. A failure preserves partial files without `_SUCCESS`; use a fresh output directory for a new run. Keep local outputs private when the input contains private media or questions.

The package and CLI entrypoints retain `docjev` for compatibility; the model display name is mJev-Doc.
