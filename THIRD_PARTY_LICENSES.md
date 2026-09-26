**English** · [简体中文](THIRD_PARTY_LICENSES.zh-CN.md)

# Third-party benchmark licenses

Scope: the current mJev `mjev_multiquestion_le4000_v1` bundle: 391 media / 1,335 questions. Checked 2026-09-25. This notice does not relicense third-party annotations or media. mJev's Apache-2.0 code license does not apply automatically to any dataset.

## Fields and redistribution semantics

All question records (flat and nested) and media-group records carry:

- `source_dataset`: upstream dataset name.
- `source_id`: object preserving original media/question identifiers. Clotho has no upstream question ID; its exact question text, filename and split identify the source row group.
- `source_url`: canonical annotation dataset page; `media_source_url` separately identifies the original media.
- `license`: annotation license identifier or a documented LicenseRef.
- `media_license`: per-file media license identifier; `UNKNOWN` means not established, not public domain.
- `redistribution_allowed`: **string enum**, for the combined annotation + media sample: `conditional`, `requires_permission`, or `unknown`. It is not a Boolean or blanket export authorization; check the value explicitly. `conditional` requires compliance with the terms in `license_details`; `unknown` must not be treated as permission.
- `license_details`: evidence URLs, upstream declarations, attribution and limitations.

## Clotho-AQA — 100 audio / 600 questions

Annotations: **MIT**, copyright (c) 2022 Tampere University. Preserve its copyright and permission notice. The official record explicitly separates question/answer licensing from sound licensing.

Media: use the **per-file** `license` column in official `clotho_aqa_metadata.csv`, not MIT. This selected set contains 46 CC0-1.0, 38 CC-BY-3.0, 13 CC-BY-NC-3.0, 2 Sampling Plus 1.0, and 1 unspecified license. Original Freesound URL, uploader, filename and excerpt boundaries are retained in `license_details.media_attribution`. The missing license remains `UNKNOWN`.

For CC-BY retain attribution and license notice and identify changes. CC-BY-NC also restricts commercial use. Sampling Plus permits distribution of copies of the whole work for noncommercial purposes under its terms; do not equate it with CC-BY or approve commercial clip distribution automatically. Its two files are marked `LicenseRef-CC-Sampling-Plus-1.0` with the canonical terms URL. Known licenses are `conditional`; the blank entry is `unknown`.

Sources: https://zenodo.org/records/6473207 ; https://zenodo.org/records/6473207/files/LICENSE.txt ; https://creativecommons.org/licenses/sampling+/1.0/

## MuChoMusic — 100 audio / 300 questions

Annotations: **CC-BY-SA-4.0**, as stated by the original Zenodo record and the project's dataset license section. Its repository's MIT code license is **not** the dataset license. Preserve attribution and license notices; share adapted annotation material under applicable ShareAlike terms and identify modifications.

Media: **UNKNOWN** for these individual recordings. All selected clips are MusicCaps-derived, obtained through a public decoded-audio mirror. MuChoMusic's official release does not include audio. Its annotation license does not establish rights to redistribute the underlying YouTube music. Original MusicCaps identifiers, YouTube URLs, and mirror revision are preserved. Combined sample redistribution is `unknown` pending recording-specific rights verification.

Sources: https://zenodo.org/records/12709974 ; https://github.com/mulab-mir/muchomusic#license ; https://creativecommons.org/licenses/by-sa/4.0/ ; media mirror https://huggingface.co/datasets/lmms-lab-audio/muchomusic

## Video-MME-v2 — 4 video / 15 questions

**Conflicting upstream declarations:** the pinned Hugging Face card says `license: mit`, while the official GitHub dataset section imposes academic-research-only use, prohibits commercial use, and requires prior approval for distribution, publication, copying, dissemination or modification, in whole or part. Video copyright remains with the original owners.

We do not resolve this conflict by silently selecting MIT. Records use `LicenseRef-Video-MME-v2-Research-Only-Permission-Required`, retain the HF MIT declaration in `license_details`, and mark `redistribution_allowed: requires_permission`. `media_license: UNKNOWN` preserves the separate original-owner rights issue. Obtain upstream clarification/approval and relevant media rights before redistribution. This document itself grants no permission and does not certify existing conversion as authorized.

Sources: https://github.com/MME-Benchmarks/Video-MME-v2#-dataset ; https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2/blob/6e4bebb03202e1ddbf3d37703e560e51c5aa2d64/README.md

## MMOU Test Mini — 187 video / 420 questions

Annotations: **Apache-2.0**, declared by the original NVIDIA MMOU dataset card, not applied by mJev. Retain applicable upstream notices and identify modifications when distributing annotations.

Media: the companion `sonalkum/MMOU-Videos` card also declares Apache-2.0. We record that declaration explicitly, but have not verified that it licenses each underlying third-party YouTube video/audio track. Consequently `media_license: UNKNOWN`, `media_repository_declared_license: Apache-2.0`, and combined redistribution `unknown`. The dataset/mirror card alone is not treated as proof of each original rights holder's grant.

Sources: https://huggingface.co/datasets/nvidia/MMOU/blob/60fddffb699443e618148f6e3ef84bb63f039cf5/README.md ; https://huggingface.co/datasets/sonalkum/MMOU-Videos/blob/main/README.md ; https://www.apache.org/licenses/LICENSE-2.0

## Evidence and integrity

The prepared bundle includes `license_evidence/` with retrieved upstream records, license text/cards, per-file Clotho metadata, evidence URLs and SHA256 hashes. `license_audit.json` records completeness and verifies that original question/candidate/answer data and media hashes did not change. Pre-update manifests are retained separately under `provenance/pre_license_fields/` and are historical evidence, not current licensed exports.

No blanket dataset LICENSE is added. `grouped.jsonl` and `manifest.jsonl` remain mixed-license collections, and must not be published as a single Apache-2.0 dataset. Existing source bundles outside this current filtered benchmark are not relabeled by this update.
