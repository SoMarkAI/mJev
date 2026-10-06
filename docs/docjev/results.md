# mjev-doc · Training and evaluation

[Project](../../README.md) · [Machine-readable results](../validation_evidence/docjev-rlcd-first-epoch.json) · [docjev-bench](https://huggingface.co/datasets/Immortal-Zhang/docjev-bench)

## Unified 1,512-question evaluation

The evaluation covers **1,512 questions** across 220 image files. Chinese and English versions yield 3,024 language records. Both checkpoints use the shared mJev HF core with identical processor tensors, prompts and candidate orders; all records completed with zero inference errors.

Inference uses causal attention, native BF16, full LM Head projection, original candidate order, temperature 1, one question per forward and no prefix cache. Decisions use the highest raw candidate-label logit. No reference labels, OCR text or generation evidence enter either model's prompt.

| Group | Original Qwen | mjev-doc | Delta |
| --- | ---: | ---: | ---: |
| All 3,024 records | 2,481/3,024 (82.04%) | 2,551/3,024 (84.36%) | +2.31 pp |
| Chinese | 1,245/1,512 (82.34%) | 1,269/1,512 (83.93%) | +1.59 pp |
| English | 1,236/1,512 (81.75%) | 1,282/1,512 (84.79%) | +3.04 pp |

194 language records change from wrong to correct; 124 change from correct to wrong. Chinese and English are paired versions of the same semantic questions, not independent samples.

### Candidate probabilities

Every record retains each candidate's **raw logit and probability**. The same readout is used for both checkpoints: `p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))`. Temperature is 1; subtracting the maximum only provides numerical stability. No variance normalization, additional scaling or calibration is applied.

| Model | Mean reference-candidate probability | Mean selected-candidate probability | Maximum probability-sum error |
| --- | ---: | ---: | ---: |
| Original Qwen | 0.805995 | 0.917514 | 1.45e-07 |
| mjev-doc | 0.842157 | 0.963156 | 1.34e-07 |

The reference probability is the probability assigned to the stored answer; the selected probability is the largest candidate probability. Means cover all 3,024 records. These are candidate-restricted probabilities, not calibrated confidence or correctness guarantees. All candidate vectors were independently recomputed from the recorded logits within 1e-5 tolerance.

Exact image hashes, source paths and supplied paper IDs found no training overlap. Perceptual near-duplicates were not assessed. Reference labels combine generated-and-screened questions with transform-grounded rotation targets; the reported accuracy measures agreement with those references.

[Complete task breakdown, probability summaries and fingerprints](../validation_evidence/mjev-doc-unified1512.json) includes improved, unchanged and regressed tasks; the overall gain does not imply improvement on every task. Raw per-record logits, probabilities, media and private run artifacts remain outside the code repository.

## First training study

### Protocol

| Item | Configuration |
| :--- | :--- |
| Trained model | mjev-doc |
| Base | Official Qwen3-VL-4B-Instruct |
| Base revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Source | Infinity document images with paired Chinese/English candidate questions |
| Total | 1,263 images, 8,079 semantic pairs, 16,158 language records |
| Training | 1,223 images, 7,829 pairs, 15,658 language records |
| Validation | [docjev-bench · pilot-v1](https://huggingface.co/datasets/Immortal-Zhang/docjev-bench/tree/f3848f96b0553d08c80cf6115dd3db0d64e2bc4e): 40 images, 250 pairs, 500 language records |
| Independent test | Not included in this study |
| Split | Complete connected paper/exact-image groups; bilingual pairs kept together |
| Parameters | 4,022,468,096 trainable language parameters; 415,347,712 frozen visual parameters |
| Optimization | AdamW, LR 2e-6, KL 0.02, group size 8, one epoch, 1,958 steps |
| Execution | Eight-GPU FSDP; FP32 master parameters, BF16 compute |
| Inference | Native BF16, causal attention, full LM Head projection, no KV reuse |
| Input limits | 4,000 tokens; 3,136–501,760 image pixels; observed maximum 685 tokens |

All 15,658 training language records were consumed once. Six additional forwards were collective padding with zero loss and gradient, not extra training examples. Eight sampled candidate labels per record are actions, not newly generated questions.

The pinned pilot-v1 revision's `validation` split preserves this study's validation images and annotations. It is the study’s validation cohort, rather than an additional independent test set. The default dataset branch hosts the 1,512-question v2 diagnostic. See the [evaluation guide](evaluation.md#docjev-bench) for the input file and revision selection.

## Accuracy

| Language | Before correct | After correct | Before | After | Delta |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Chinese | 185/250 | 195/250 | 74.0% | 78.0% | +4.0 pp |
| English | 180/250 | 194/250 | 72.0% | 77.6% | +5.6 pp |
| Combined | 365/500 | 389/500 | 73.0% | **77.8%** | **+4.8 pp** |

49 language records changed from wrong to correct; 25 changed from correct to wrong.

## Checkpoint verification

Exported weights were independently reloaded. Eight post-training witness rows matched the final FSDP candidate logits with maximum absolute error **0.0**. All 315 compared visual tensors remained byte-identical to the official base; 371 of 398 compared language tensors changed. The tied output head and input embedding remained equal.

Dataset/reference hashes and configuration are retained in the machine-readable summary. Model weights, original document images, source questions, user operation logs and private review artifacts are not part of this repository.

## Scope

Reference questions and labels were model-generated and screened. Reported scores measure agreement with those references, not independently established human ground truth. Validation did not enter the optimizer. One seed and one epoch were evaluated; there is no independent test set, confidence-interval claim, human-verified benchmark-wide result or measured permutation improvement in this study.

Exact image and supplied paper IDs are separated across train/validation. Perceptual duplicate checks were not performed. One repeated question/candidate mapping on different images was observed across splits; this is not proof of answer leakage, and broader duplication checks remain necessary for a formal benchmark.
