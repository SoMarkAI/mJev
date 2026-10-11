# DocJev-bench · Document evidence, business decisions

[Dataset](https://huggingface.co/datasets/Immortal-Zhang/DocJev-Bench) &nbsp;&nbsp; [简体中文](benchmark.zh-CN.md) &nbsp;&nbsp; [mJev-Doc](README.md)

**One document. Several decisions. Clear rules.**

DocJev-bench evaluates the capabilities that connect document understanding to a useful next step: identifying evidence, comparing fields, checking stated requirements and choosing an action. Use it to evaluate mJev-Doc or compare other document models.

```text
Document image + question-level criteria + candidate actions → decision
```

The current collection contains 608 images, 2,617 bilingual question pairs and 5,234 language records. Each page can support multiple distinct questions; candidate sets vary with the task.

## Where it fits in a business workflow

| Workflow component | What the benchmark asks a model to do |
| --- | --- |
| Material intake | Recognize a document's category, purpose and relevant visible elements |
| Information review | Compare fields and reconcile values across tables or document regions |
| Readiness checks | Decide whether the fields required by a stated checklist are present |
| Rule-based screening | Apply supplied thresholds and combinations of conditions to document facts |
| Review routing | Choose a processing or review route under the rules given in the question |
| Action selection | Select the prescribed next action, annotation or handling outcome |

These are evaluation targets for document-processing components. The benchmark measures answers to individual questions; it does not execute an operational approval system. Rules are stated in each applicable question, rather than inferred as real organizational, legal or financial policy.

## Task design

Each page supports several distinct questions grounded in its content and the stated rules. Candidate counts vary by question. Chinese and English versions share the image and reference answer, allowing comparisons by language or task type.

## Dataset groups

| Group | Images | Question pairs | Language records |
|---|---:|---:|---:|
| Document understanding and decisions | 388 | 1848 | 3696 |
| Cross-industry classification | 220 | 769 | 1538 |

[Current results](results.md) · [Dataset notes](https://huggingface.co/datasets/Immortal-Zhang/DocJev-Bench/blob/main/DATASET_NOTES.md)

## Use it to evaluate a document decision component

Supply the image, question and candidates to a model, keep the reference label out of the prompt, and compare its selected candidate with the stored answer. Report Accuracy by language and task type. Candidate-order experiments can additionally report Permutation Consistency.

The current task format is Choice. A model may return candidate-relative probabilities; the dataset does not supply reference probability distributions or continuous business scores. Chinese and English versions are paired observations. Use the [evaluation guide](evaluation.md) for the existing protocol and distinguish image-based evaluation from an OCR-plus-text-model pipeline.
