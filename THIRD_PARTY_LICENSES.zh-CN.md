[English](THIRD_PARTY_LICENSES.md) · **简体中文**

# 第三方 benchmark 许可证

本页是英文通知的中文说明，保留原始条款与证据引用，不新增授权；适用上游许可证原文。范围为 2026-09-25 检查的 `mjev_multiquestion_le4000_v1` 数据包：391 个媒体／1,335 道题，不代表之后所有数据集版本。mJev的 Apache-2.0 代码许可证不会自动覆盖数据标注或媒体。

## 字段与再分发语义

扁平/嵌套问题记录及媒体组记录均含：

- `source_dataset`：上游数据集名称。
- `source_id`：保留原媒体/问题 ID 的对象。Clotho 没有上游问题 ID，用精确题文、文件名、split 标识源行组。
- `source_url`：标准标注数据页；`media_source_url` 单独标识媒体来源。
- `license`：标注许可证标识或有说明的 LicenseRef。
- `media_license`：逐媒体许可证；`UNKNOWN` 是尚未确立，不是公有领域。
- `redistribution_allowed`：标注与媒体组合样本的**字符串枚举**：`conditional`、`requires_permission`、`unknown`。不是布尔值或整体导出授权；必须显式检查。conditional 需遵守 `license_details` 条款，unknown 不能当作允许。
- `license_details`：证据 URL、上游声明、署名和限制。

## Clotho-AQA：100 音频／600 题

标注为 **MIT**，copyright (c) 2022 Tampere University，保留版权与许可通知。官方记录明确区分问答与声音许可。

媒体使用官方 `clotho_aqa_metadata.csv` 的逐文件 `license`，不是 MIT。该子集含 46 CC0-1.0、38 CC-BY-3.0、13 CC-BY-NC-3.0、2 Sampling Plus 1.0 和 1 未注明许可。原 Freesound URL、上传者、文件名及片段边界保存在 `license_details.media_attribution`；缺失许可保持 UNKNOWN。

CC-BY 保留署名/许可并标明修改；CC-BY-NC 还限制商业用途。Sampling Plus 按其条款允许非商业目的分发完整作品副本，不等同 CC-BY，不能自动允许商业片段分发。两文件标为 `LicenseRef-CC-Sampling-Plus-1.0` 并附官方条款 URL。已知许可为 conditional，空白项为 unknown。

来源：[官方记录](https://zenodo.org/records/6473207)、[LICENSE.txt](https://zenodo.org/records/6473207/files/LICENSE.txt)、[Sampling Plus](https://creativecommons.org/licenses/sampling+/1.0/)。

## MuChoMusic：100 音频／300 题

标注为 **CC-BY-SA-4.0**，依据原 Zenodo 与项目数据许可。仓库 MIT 代码许可不是数据许可。保留署名和许可通知，改编标注遵守适用的 ShareAlike 并标明修改。

逐录音媒体许可 **UNKNOWN**。选中片段来自 MusicCaps，通过公开解码音频镜像取得。MuChoMusic 官方不附音频，标注许可不能证明底层 YouTube 音乐再分发权。保留 MusicCaps ID、YouTube URL、镜像 revision。组合样本为 unknown，待逐录音确认权利。

来源：[Zenodo](https://zenodo.org/records/12709974)、[项目许可](https://github.com/mulab-mir/muchomusic#license)、[CC-BY-SA](https://creativecommons.org/licenses/by-sa/4.0/)、[媒体镜像](https://huggingface.co/datasets/lmms-lab-audio/muchomusic)。

## Video-MME-v2：4 视频／15 题

**上游声明冲突：**固定 HF card 写 `license: mit`，但官方 GitHub dataset 段落限制学术研究、禁止商业用途，全部或部分分发/出版/复制/传播/修改需事先批准。视频版权仍归原权利人。

不静默选择 MIT 消解冲突。记录用 `LicenseRef-Video-MME-v2-Research-Only-Permission-Required`，在 `license_details` 保留 HF MIT 声明，再分发标为 requires_permission。`media_license: UNKNOWN` 保留原媒体权利问题。再分发前需上游澄清/批准及相关媒体权利。本文件不授予许可，也不认证已有转换已获授权。

来源：[GitHub 条款](https://github.com/MME-Benchmarks/Video-MME-v2#-dataset)、[固定 HF card](https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2/blob/6e4bebb03202e1ddbf3d37703e560e51c5aa2d64/README.md)。

## MMOU Test Mini：187 视频／420 题

标注为原 NVIDIA 数据卡声明的 **Apache-2.0**，不是 mJev 重新赋予。分发标注时保留适用上游通知并标识修改。

`sonalkum/MMOU-Videos` 配套卡也声明 Apache-2.0；明确记录这一声明，但未核实它授权了每个底层第三方 YouTube 视频/音轨。因此媒体许可 UNKNOWN、`media_repository_declared_license: Apache-2.0`、组合分发 unknown。数据/镜像卡本身不视为各原权利人授权证明。

来源：[固定 NVIDIA card](https://huggingface.co/datasets/nvidia/MMOU/blob/60fddffb699443e618148f6e3ef84bb63f039cf5/README.md)、[媒体 card](https://huggingface.co/datasets/sonalkum/MMOU-Videos/blob/main/README.md)、[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)。

## 证据与完整性

数据包 `license_evidence/` 保留已获取的上游记录、条款/cards、逐文件 Clotho 元信息、证据 URL 与 SHA256。`license_audit.json` 记录完整性，并核对原问题/候选/答案与媒体哈希不变。更新前 manifest 在 `provenance/pre_license_fields/` 单独留存，是历史证据，不是当前授权导出。

不添加一份覆盖全部数据的 LICENSE。`grouped.jsonl` 和 `manifest.jsonl` 仍为混合许可集合，不得作为统一 Apache-2.0 数据集发布。本次不为其他源数据包重新标注许可证。

## mjev-doc validation example

`examples/docjev/validation-table.jpg` is Table 3 of Gupta et al. (2008), *Can subjective global assessment of nutritional status predict survival in ovarian cancer?*, Journal of Ovarian Research 1, 5. © 2008 Gupta et al., licensee BioMed Central Ltd. [Source](https://link.springer.com/article/10.1186/1757-2215-1-5) · [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). The crop is supplied by Infinity and its demo bytes are unchanged. It is not relicensed under Apache-2.0. See `examples/docjev/metadata.json` for attribution and hashes.
