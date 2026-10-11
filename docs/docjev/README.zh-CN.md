# mJev-Doc · 让文档信息变成清晰决策

[English](README.md) &nbsp;&nbsp; 简体中文 &nbsp;&nbsp; [返回 mJev](../../README.zh-CN.md)

给一张文档，提出多个问题，为每题定义有意义的候选答案。**mJev-Doc** 面向文档视觉属性、类别、字段关系和明确规则下的业务决策，返回每个候选的原始分数、概率和最终选择。

模型基于 Qwen3-VL-4B-Instruct，与[通用 mJev](../models.zh-CN.md) 共用 HF 推理核心，使用独立的文档训练权重。

## DocJev-bench · 评测文档中的业务判断

配套的 [DocJev-bench](https://huggingface.co/datasets/Immortal-Zhang/docjev-bench) 可用于比较不同文档模型，覆盖感知、字段关系、条件筛选、完整性、业务路由和行动选择，包含 **220 张图片上的 1,512 对中英文题目**。

[任务覆盖与财务报表样例 →](benchmark.zh-CN.md)

## 开始使用

Linux、Python 3.11+、NVIDIA GPU，先安装适合驱动的 PyTorch 2.11。文档实测环境为 CUDA 13.0、96 GB Blackwell GPU。

```bash
python -m pip install -e '.[docjev]'
hf auth login
hf download SoMarkAI/mjev-doc --local-dir models/mjev-doc
python demo_docjev.py --model models/mjev-doc \
  --input examples/docjev/multiple.json --check-only
python demo_docjev.py --model models/mjev-doc \
  --input examples/docjev/multiple.json --output outputs/docjev.json
```

**[mJev-Doc 权重](https://huggingface.co/SoMarkAI/mjev-doc)**包含完整模型、tokenizer、processor 和 chat template。下载后即可运行示例。[真实验证集示例](../../examples/docjev/README.md)包含一张研究表格、四对中英文题目及训练后模型的实际输出。

同页多题可以使用 `--numerics stable --prefix-cache --question-batch-size 2`。这与准确率评测使用的 `native`、串行、无缓存设置不同，比较时需保持协议一致。

## 训练与结果

mJev-Doc 首轮使用 **1,223 张图、7,829 对中英文题、15,658 条语言记录**训练一轮：八卡 FSDP、冻结视觉、更新语言参数，不使用 LoRA 或新增决策头。

| 评测 | 原始 Qwen | mJev-Doc |
| --- | ---: | ---: |
| 首轮验证，500 条语言记录 | 73.0% | 77.8% |
| 统一诊断，3,024 条语言记录 | 82.04% | 84.36% |

评测包含 **1,512 道题目**，覆盖 220 张评测图片，提供中英文版本。每个候选均保存原始 logit 与温度 1 softmax 概率。分数表示与参考标签的一致率，中英文不能视为独立样本。[完整结果与概率](results.md)

[安装](installation.md) &nbsp;&nbsp; [输入输出](input-output.md) &nbsp;&nbsp; [架构](architecture.md) &nbsp;&nbsp; [评测](evaluation.md) &nbsp;&nbsp; [RLCD 训练](training.md)

[HF 多图片服务：动态 batch、多 GPU 副本和 Windows 原生安装](serving.zh-CN.md)

Benchmark：[Immortal-Zhang/docjev-bench](https://huggingface.co/datasets/Immortal-Zhang/docjev-bench)。代码采用 Apache-2.0，示例图片保留 CC-BY-2.0 许可与署名。
