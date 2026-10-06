# mjev-doc · 让文档信息变成清晰决策

[English](README.md) · 简体中文 · [返回 mJev](../../README.zh-CN.md)

给一张文档，提出多个问题，为每题定义有意义的候选答案。**mjev-doc** 面向文档视觉属性、类别、字段关系和明确规则下的业务决策，返回每个候选的原始分数、概率和最终选择。

## 两个模型，共享推理核心

| 模型 | 定位 | 权重 |
| --- | --- | --- |
| mJev-Qwen3-VL-4B-RLCD | 通用视觉候选决策 | [Hugging Face](https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD) |
| mjev-doc | 文档理解与候选决策 | Hugging Face，待发布 |

两者均以 Qwen3-VL-4B-Instruct 为基础，来自不同训练数据和训练任务。mjev-doc 直接复用 `mjev/` 的 HF processor、模板、attention、LM Head 和 prefix cache；文档预处理、训练及评测入口独立保留。两个模型的权重分别发布，代码和说明统一维护在本仓库。

## 开始使用

Linux、Python 3.11+、NVIDIA GPU，先安装适合驱动的 PyTorch 2.11。文档实测环境为 CUDA 13.0、96 GB Blackwell GPU。

```bash
python -m pip install -e '.[docjev]'
hf download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --local-dir models/Qwen3-VL-4B-Instruct
python demo_docjev.py --model models/Qwen3-VL-4B-Instruct \
  --input examples/docjev/multiple.json --check-only
python demo_docjev.py --model /path/to/mjev-doc \
  --input examples/docjev/multiple.json --output outputs/docjev.json
```

mjev-doc 权重将与 mJev 一样通过 Hugging Face 发布。当前可使用完整训练 checkpoint 或原始 Qwen 模型运行示例。[真实验证集示例](../../examples/docjev/README.md)包含一张研究表格、四对中英文题目及训练后模型的实际输出。

同页多题可以使用 `--numerics stable --prefix-cache --question-batch-size 2`。这与准确率评测使用的 `native`、串行、无缓存设置不同，比较时需保持协议一致。

## 训练与结果

mjev-doc 首轮使用 **1,223 张图、7,829 对中英文题、15,658 条语言记录**训练一轮：八卡 FSDP、冻结视觉、更新语言参数，不使用 LoRA 或新增决策头。

| 评测 | 原始 Qwen | mjev-doc |
| --- | ---: | ---: |
| 首轮验证，500 条语言记录 | 73.0% | 77.8% |
| 统一诊断，3,024 条语言记录 | 82.04% | 84.36% |

评测包含 **1,512 道题目**，覆盖 220 张评测图片，提供中英文版本。每个候选均保存原始 logit 与温度 1 softmax 概率。分数表示与参考标签的一致率，中英文不能视为独立样本；纸质专项仅含正例。[完整结果、概率与视觉分组](results.md)

[安装](installation.md) · [输入输出](input-output.md) · [架构](architecture.md) · [评测](evaluation.md) · [RLCD 训练](training.md)

Benchmark：[Immortal-Zhang/DocJev-Bench](https://huggingface.co/datasets/Immortal-Zhang/DocJev-Bench)。代码采用 Apache-2.0，示例图片保留 CC-BY-2.0 许可与署名。

旋转／纸质专项的参考标签分别来自已知旋转角度与筛选后的纸质外观判断；这部分分数按固定参考标签计算。
