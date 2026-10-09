# DocJev-bench · Document evidence, business decisions

[Dataset](https://huggingface.co/datasets/Immortal-Zhang/docjev-bench) · [简体中文](benchmark.zh-CN.md) · [mJev-Doc](README.md)

**One document. Several decisions. Clear rules.**

DocJev-bench evaluates the capabilities that connect document understanding to a useful next step: identifying evidence, comparing fields, checking stated requirements and choosing an action. It is a separately usable benchmark asset alongside the mJev-Doc model.

```text
Document image + question-level criteria + candidate actions → decision
```

The current collection contains 220 images, 1,512 bilingual question pairs and 21 task types. Each page can support multiple distinct questions; candidate sets vary with the task.

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

## Design strengths

- **Evidence and decision in one task.** An answer must connect the document's visible information to the requested judgment.
- **Several decisions per page.** The same page is tested through different task types, helping expose strengths and weaknesses within a document.
- **Rules as inputs.** Explicit criteria make conditional choices reproducible and let a reader inspect why a particular action is selected.
- **Dynamic candidate sets.** Questions retain 2, 3, 4, 5 or 7 candidates, with reference answers suitable for Jev-style choice scoring.
- **Paired languages.** Chinese and English versions use the same image and reference candidate, enabling language-specific and paired comparisons.

## Current decision-task coverage

| Task type | Question pairs |
| --- | ---: |
| Conditional reasoning | 92 |
| Action decisions | 72 |
| Routing | 64 |
| Completeness | 46 |
| Compliance with stated requirements | 8 |
| Priority | 8 |
| **Total for these six task types** | **290** |

The collection also includes 476 field-relation questions and 70 consistency questions, alongside perception, classification and other understanding tasks. Counts refer to semantic question pairs, not the two language records separately.

## A page with several business decisions

Document `000002` is a financial-statement page in the current benchmark. Its questions illustrate distinct steps on the same evidence:

| Question pair | Stated decision rule | Reference choice |
| --- | --- | --- |
| `000002_08` | Select years from 2022E–2024E with a quick ratio ≥ 1 and a debt-to-assets ratio < 35% | 2023E and 2024E |
| `000002_09` | For 2023E, route to cash-flow review when operating cash flow is below net profit and the working-capital change is negative; the question defines the other branches | Cash Flow Review Team |
| `000002_10` | Use 2024E operating cash flow to cover investing and financing outflows, then choose the action based on the remaining funds | Retain the balance |

The rules are benchmark assumptions written in the questions. The full wording, candidate sets and bilingual versions are available in the [dataset](https://huggingface.co/datasets/Immortal-Zhang/docjev-bench); the page is `images/000002.jpg`.

## Use it to evaluate a document decision component

Supply the image, question and candidates to a model, keep the reference label out of the prompt, and compare its selected candidate with the stored answer. Report Accuracy by language and task type. Candidate-order experiments can additionally report Permutation Consistency.

The current task format is Choice. A model may return candidate-relative probabilities; the dataset does not supply reference probability distributions or continuous business scores. Chinese and English versions are paired observations. Use the [evaluation guide](evaluation.md) for the existing protocol and distinguish image-based evaluation from an OCR-plus-text-model pipeline.
