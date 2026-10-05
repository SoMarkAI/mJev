# DocJev-Qwen3-VL-4B-RLCD

First DocJev training study.

[Project](../../README.md) · [Machine-readable results](../validation_evidence/docjev-rlcd-first-epoch.json) · [DocJev-Bench](https://huggingface.co/datasets/Immortal-Zhang/DocJev-Bench)

## Protocol

| Item | Configuration |
| :--- | :--- |
| Trained model | DocJev-Qwen3-VL-4B-RLCD |
| Base | Official Qwen3-VL-4B-Instruct |
| Base revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Source | Infinity document images with paired Chinese/English candidate questions |
| Total | 1,263 images, 8,079 semantic pairs, 16,158 language records |
| Training | 1,223 images, 7,829 pairs, 15,658 language records |
| Validation | [DocJev-Bench · pilot-v1](https://huggingface.co/datasets/Immortal-Zhang/DocJev-Bench): 40 images, 250 pairs, 500 language records |
| Independent test | Not included in this study |
| Split | Complete connected paper/exact-image groups; bilingual pairs kept together |
| Parameters | 4,022,468,096 trainable language parameters; 415,347,712 frozen visual parameters |
| Optimization | AdamW, LR 2e-6, KL 0.02, group size 8, one epoch, 1,958 steps |
| Execution | Eight-GPU FSDP; FP32 master parameters, BF16 compute |
| Inference | Native BF16, causal attention, full LM Head projection, no KV reuse |
| Input limits | 4,000 tokens; 3,136–501,760 image pixels; observed maximum 685 tokens |

All 15,658 training language records were consumed once. Six additional forwards were collective padding with zero loss and gradient, not extra training examples. Eight sampled candidate labels per record are actions, not newly generated questions.

The benchmark's `validation` split preserves this study's validation images and annotations. It is the study’s validation cohort, rather than an additional independent test set. See the [evaluation guide](evaluation.md#docjev-bench) for the input file.

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

## Expanded multitask diagnostic

A separate frozen cohort contains **200 images** (100 Infinity, 100 OLM-TFR), **1,492 bilingual question pairs** and **2,984 language records** across 21 task types. Both models received identical image tensors, prompts and candidate orders. Inference used causal attention, native BF16, full LM Head projection, raw candidate-logit argmax, temperature 1, serial questions and no prefix cache. All 2,984 records completed for both models with zero errors.

| Group | Base correct | DocJev correct | Base | DocJev | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Combined | 2,453/2,984 | 2,526/2,984 | 82.21% | 84.65% | +2.45 pp |
| Chinese | 1,233/1,492 | 1,256/1,492 | 82.64% | 84.18% | +1.54 pp |
| English | 1,220/1,492 | 1,270/1,492 | 81.77% | 85.12% | +3.35 pp |

192 language records changed from wrong to right and 119 from right to wrong. Exact image hashes, source paths and supplied paper IDs found no training overlap; perceptual near-duplicates were not assessed. The questions use generated-and-screened references; this diagnostic is distinct from the first validation study and does not establish general superiority.

[Aggregate results and hashes](../validation_evidence/docjev-multitask200.json). Raw media, questions and private run records remain outside the code repository.

## Nonzero rotation and paper appearance

This diagnostic uses **20 documents / 20 semantic questions / 40 aligned Chinese-English records**: ten nonzero rotations (90°, 180°, 270°) and ten paper-appearance questions. It derives from parent images in the 200-image diagnostic, so it is not another independent test set. Both models use the same controlled native protocol described above, with identical actual input tensors and zero errors.

| Group | Records | Base | DocJev | Delta |
| --- | ---: | ---: | ---: | ---: |
| Combined | 40 | 28/40 (70.0%) | 25/40 (62.5%) | −7.5 pp |
| Nonzero rotation | 20 | 8/20 (40.0%) | 5/20 (25.0%) | −15.0 pp |
| Paper appearance, positives only | 20 | 20/20 (100.0%) | 20/20 (100.0%) | 0.0 pp |
| Chinese | 20 | 12/20 (60.0%) | 13/20 (65.0%) | +5.0 pp |
| English | 20 | 16/20 (80.0%) | 12/20 (60.0%) | −20.0 pp |

Two language records improve and five regress. Two model-record outputs have tied top native-BF16 logits; ties follow the documented first-candidate policy. Rotation accuracy declines in this small cohort, while the paper questions match in both models. The rotation labels follow the recorded pixel transform; paper labels describe visible physical-paper appearance and do not establish camera/scanner acquisition history. All ten paper examples are positives, so their score cannot establish balanced paper-vs-digital discrimination.

[Aggregate counts, model fingerprints and cohort hashes](../validation_evidence/docjev-visual20.json). Raw diagnostic media and questions are not distributed with the code.
