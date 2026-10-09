[English](stability.md) · **简体中文**

Docker 之外的命令均假定已激活 Python 3.11+ 虚拟环境：先用 `python3 -m venv .venv` 创建，再运行 `source .venv/bin/activate`。激活后安装和运行统一使用 `python`；Docker 内使用 `python3`。Shell 脚本也支持 `PYTHON=/path/to/venv/bin/python`。历史执行记录保留原始命令。

# 数值模式与多问题评分

> 历史结果按当时配置记录；后续检查与缺失的配置证据见[最新验证](validation_current.zh-CN.md)。

本实现不训练模型、不改变权重、不调整输出分数，也不缓存生成答案。评分不读取参考答案。稳定模式保留预期的 attention 可见性和原 LM Head，但改变浮点计算方式。因此，应比较稳定模式下的串行、batch 和缓存结果，不应直接沿用旧 native BF16 的 logits 或准确率。

## HF

默认 `HFMJevEngine(..., numerics="stable")`。文本计算与 MoE 路由采用 FP32，参数保持加载时的 dtype。线性投影使用固定行分块，SDPA 分块对齐绝对 token 位置，避免 batch 大小和缓存边界改变逐 token 归约形状。

媒体使用官方编码器；各媒体独立处理，同一次 forward 中完全重复的媒体复用特征。保留原生视频小数位置。无需 vLLM 或自定义内核编译。

`score_many(..., batch_size=3, use_prefix_cache=True)` 与 `score_questions(context, questions, batch_size=3)` 执行真实三行 forward。各行有独立可见性和克隆的公共 KV；尾批次可小于三行。这不是零复制 Tree-KV，也不把多个问题合成一段对话。单行和 batch 都支持完整或选定词表行的 LM Head 投影，问题可有不同候选数；比较执行策略时保持投影方式一致。

```bash
python demo_hf.py --model /path/to/model --input task.json \
  --numerics stable --question-batch-size 3 --prefix-cache --projection full
```

`--numerics native` 保留早期计算路径，用于速度和参考研究，但 batch/cache 可能改变答案。稳定模式增加时间和显存开销；缓存可能摊薄前缀计算。processor 与媒体解码不等同于评分。

## vLLM pooling

进程启动前设置 `VLLM_BATCH_INVARIANT=1`，启用 vLLM 已有的稳定线性/MoE 内核。mJev hook 还使 residual RMSNorm 稳定，并将 SDPA 分块对齐绝对 token 位置，使完整 prefill 与缓存后缀采用相同逐 token attention 形状。候选 mask 与 scheduler 保持有效，不会静默串行化请求。

```bash
VLLM_BATCH_INVARIANT=1 python demo.py --model /path/to/model \
  --input examples/multiple.json --mode causal
```

音视频特征按原生特征类型、宽度与 token ID 放置，不再假设整个打包 batch 是一段连续交错音视频区域，修复了不同请求之间文本间隔的放置问题。本页仅适用于 `mjev` pooling，不适用于独立 HTTP Tree-KV runtime。

## 复现检查

已有固定依赖环境中，从仓库根目录执行：

```bash
PYTHONPATH=.:runtime MJEV_ENABLE=0 MJEV_ENABLE_PATCHES=0 \
  OMP_NUM_THREADS=4 python -m pytest tests -q -p no:cacheprovider
```

CPU 测试覆盖 mask、小数位置、真实 batch、分支独立、权重不变、媒体类型放置、对齐 attention 和失败时的 hook 安装清理。发布 CPU 套件在已有固定环境通过 102 项；全新安装未验证。小规模一致性通过不等于数据集准确率或生产负载验证。历史 native 失败见 [HF 文档](hf.zh-CN.md)。

## 全权重回归：2026-09-26

四张 24 GB NVIDIA GPU，冻结数值核心，9 个媒体／27 道题（2 图片、4 音频、3 带音轨视频），causal/isolated 两种 mask，每种策略一次，完整 LM Head 投影：

| 模式 | 串行/batch/cache 对照 | 热缓存顺序检查 | 最大 logit / 概率差 |
| --- | ---: | ---: | ---: |
| HF stable | 216/216 | 54/54 | 0 / 0 |
| vLLM invariant | 270/270 | 54/54 | 0 / 0 |

这是 27 题的重复比较，不是准确率。参考为同一数值模式的串行评分，不宣称跨后端相等或普遍逐位确定性。顺序检查重放已预热的问题，不是未缓存的逆序测试。

三题一组的 causal 平均耗时：HF 无缓存串行 11.710 秒、热 KV batch 3.350 秒、包含冷前缀构建 6.816 秒；vLLM 分别为 1.716、0.707、1.198 秒。计时从已准备输入开始，排除媒体解码、processor、启动和网络。两后端精度与并行方式不同。HF 实际 batch=3；vLLM 提交三请求，观察到同一步最多调度两个。每种 mask 两后端均 27/27 热缓存命中；HF 缓存后缀不调用媒体编码器，基础 KV 不变。

回归后的发布改动增加失败安全的 hook 安装/启动，并使 HF 真 batch 正确支持 selected projection；完整投影分支保持不变，新增部分另行做 CPU 与 demo 检查。

发布版 `demo_hf.py` 还对一个真实带音轨视频的三个问题运行 selected projection、batch=3 和 prefix cache：三题各复用 523 tokens，未导入 vLLM。双 mask API 检查 18/18 通过：selected 串行/batch/cache 的 12 项评分完全相等；selected/full 的六个答案一致，最大 logit 差 1.63e-5、概率差 3.34e-6。该小规模发布差异检查不替代上面的完整投影回归。
