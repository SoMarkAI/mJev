# RLCD training

[Project](../../README.md) · [First study](results.md)

**mjev-doc** starts from official Qwen3-VL weights. Inference alone is training-free; the optional RLCD pipeline updates the full language model while freezing the visual tower and aligner. It adds no LoRA or decision head.

## Reward and objective

A frozen base model scores the same image, question and candidates. Let `p*` be its candidate-softmax probability of the reference label. For every sampled candidate action:

```text
reward = +p*  if action == reference label
         -p*  otherwise
advantage = reward - group_mean(reward)
loss = clipped_candidate_policy_loss + beta × KL(policy || frozen_reference)
```

There is no division by group standard deviation: that would cancel the intended `p*` weighting for binary signed rewards. All-correct and all-wrong groups have zero centered policy advantage. KL is computed exactly over the supplied candidate action space. Actions are sampled from the policy distribution without using the reference label.

The validated first-epoch loop applies one optimizer update to each collected action group. The old/current policy coincide before that update, so PPO clipping is approximately inactive in this protocol. RLCD uses a single-decision on-policy update with group-centered advantages and a KL penalty, rather than multi-epoch PPO. Probabilities are computed directly from the LM Head; `generate()` is never called.

## Frozen bilingual source

The first pipeline accepts a JSONL file with one accepted document per line. For every semantic question, provide aligned Chinese/English question objects, identical candidate label keys and the same reference label. The held-out example below illustrates the schema; use separate documents for training:

```json
{
  "id": "infinity-validation-table",
  "status": "accepted",
  "split": "dev",
  "image": "validation-table.jpg",
  "image_sha256": "c5bc6f4e69f7b8bf446c1ae35e2e74904bc3dac2e2dcf58e2c7dba67c53fb595",
  "paper_id": "page:c5bc6f4e69f7b8bf",
  "source_dataset": "Infinity",
  "source_rows": [
    267436
  ],
  "question_ids": [
    "c5bc6f4e69f7b8bf_r0q03"
  ],
  "questions": [
    {
      "image": "validation-table.jpg",
      "question": {
        "instructions": "表中哪些项目的P值均标为“< 0.001”？",
        "criteria": {
          "A": "中度营养不良与重度营养不良",
          "B": "重度营养不良与治疗史",
          "C": "诊断时分期与治疗史",
          "D": "中度营养不良与诊断时分期"
        }
      },
      "label": "B"
    }
  ],
  "questions_en": [
    {
      "image": "validation-table.jpg",
      "question": {
        "instructions": "Which items in the table have P-values marked “< 0.001”?",
        "criteria": {
          "A": "Moderately malnourished and severely malnourished",
          "B": "Severely malnourished and treatment history",
          "C": "Stage at diagnosis and treatment history",
          "D": "Moderately malnourished and stage at diagnosis"
        }
      },
      "label": "B"
    }
  ]
}
```

The corresponding image must exist under `--image-root`; traversal outside that root is rejected. SHA256 must match. Supply truthful `reference_provenance` in the configuration; the held-out example uses `codex_generated_bilingual_model_screened_reference`. Preparation preserves your configured reference provenance in row and run metadata; it does not infer or assert human ground truth.

Question IDs must be globally unique. `paper_id` connects pages of a document; identical image SHA256 connects copies. Connected groups are indivisible. An exact 250-pair validation cohort is selected by default, giving 500 language records. If that number cannot be reached with whole groups, preparation fails rather than splitting images. Adapt `validation_pairs` to your own corpus. No independent test split is created by this first-epoch pipeline.

The schema above illustrates the bilingual training input. Supply separate training documents; keep the bundled held-out example out of training.

## Four-stage pipeline

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_docjev.sh \
  --model models/Qwen3-VL-4B-Instruct \
  --source /path/to/documents.jsonl --image-root /path/to/images \
  --config configs/docjev_rlcd.json --out outputs/rlcd-run
```

1. **Prepare.** Copy a source snapshot, hash media, preserve reference text, check both language inputs through the real processor, freeze connected train/dev groups and derive the one-epoch step count.
2. **Reference.** One native BF16 model per visible GPU scores the frozen train/dev cohort. Store label IDs, prompt/processor tensor hashes and full reference candidate distributions.
3. **Train.** FSDP shards full FP32 language parameters and AdamW state, computes in BF16 and freezes visual parameters. Each real row contributes once. Equal-collective padding contributes zero gradient.
4. **Evaluate.** Reload `train/checkpoint-final`, score the same validation cohort, compare final-forward witnesses and write `comparison.json`.

A copy of your config is stored under the output directory. Preparation updates that copy with the derived step count; it never edits the repository's template. Frozen config/data/reference hashes are enforced in later stages. Output directories must be fresh. A failed stage exits without a success marker; inspect `stage-N.log` before resuming.

The default config requires exactly eight visible GPUs. Other world sizes require changing the config before freezing references and have not been covered by the eight-card study. The wrapper rejects multiple epochs because the validated row schedule is one epoch. Training uses FP32 optimizer state; inference memory requirements are not training memory requirements.

## Partial checkpoint and resume

Lower-level commands are available for stage-by-stage control. A deliberately bounded run saves model, AdamW, rank-local sampling RNG, CPU/CUDA RNG, layout and contract hashes:

```bash
torchrun --standalone --nproc_per_node=8 -m docjev.rlcd.train \
  --model models/Qwen3-VL-4B-Instruct --data outputs/rlcd-run/data \
  --references outputs/rlcd-run/reference --config outputs/rlcd-run/config.json \
  --out outputs/rlcd-run/train --stop-after 200
```

Use the path recorded in `train/partial-run.json` as `--resume`:

```bash
torchrun --standalone --nproc_per_node=8 -m docjev.rlcd.train \
  --model models/Qwen3-VL-4B-Instruct --data outputs/rlcd-run/data \
  --references outputs/rlcd-run/reference --config outputs/rlcd-run/config.json \
  --out outputs/rlcd-run/train --resume /path/from/partial-run.json
```

The rank count, config, source manifest, references and checkpoint layout must be unchanged. Resume uses locally produced trusted checkpoints; do not load arbitrary external Python checkpoint objects. An already completed training directory cannot be resumed.

## Scope of validation

The underlying trainer completed the recorded eight-GPU first epoch and a step-200 save/resume. Release checks cover the migrated objective, RNG/optimizer restore, grouped splitting, zero-gradient padding and demo parity. The new four-stage convenience wrapper is not a second full training run. It orchestrates the same stage modules; the original result remains the separately recorded experiment.
