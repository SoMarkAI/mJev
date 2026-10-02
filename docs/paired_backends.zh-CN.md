[English](paired_backends.md) · **简体中文**

Docker 之外的命令均假定已激活 Python 3.11+ 虚拟环境：先用 `python3 -m venv .venv` 创建，再运行 `source .venv/bin/activate`。激活后安装和运行统一使用 `python`；Docker 内使用 `python3`。Shell 脚本也支持 `PYTHON=/path/to/venv/bin/python`。历史执行记录保留原始命令。

# HF / vLLM 配对评测

> 历史结果按当时配置记录；后续检查与缺失的配置证据见[最新验证](validation_current.zh-CN.md)。

使用官方本地 Qwen3-Omni Thinker，比较 HF 与主包 vLLM pooling，不是 HTTP runtime。未引入生成、训练权重或 Decision Head，测试 causal 与候选隔离两种模式。

需要 Linux 和固定的 HF/可选 vLLM 依赖。已检查的预装镜像记录于本地计算记录中，不意味着任意全新安装均已验证。

```bash
PYTHONPATH=. python benchmarks/integrated/compare_backends.py --freeze --root /path/to/av-benchmark --infinity /path/to/infinity-eval --out outputs/paired
VLLM_BATCH_INVARIANT=0 MJEV_ENABLE=1 VLLM_USE_V2_MODEL_RUNNER=0 PYTHONPATH=. python benchmarks/integrated/compare_backends.py --backend vllm --model /path/to/model --out outputs/paired --numerics native --model-revision "$MODEL_REVISION"
MJEV_ENABLE=0 PYTHONPATH=. python benchmarks/integrated/compare_backends.py --backend hf --model /path/to/model --out outputs/paired --numerics native --model-revision "$MODEL_REVISION"
PYTHONPATH=. python benchmarks/integrated/report_comparison.py --out outputs/paired
```

GPU 步骤依次执行，每步需要四张空闲 GPU。使用独立输出目录；运行器不覆盖已有后端结果目录。冒烟测试可把冻结 manifest 复制到另一输出目录，再为每个后端加 `--smoke`。

## 输入、计时与指标

运行前冻结输入和参考答案，运行时复核文件哈希。两后端使用相同 qwen-omni-utils 解码路径；图像/视频只缩放一次，明确帧数和像素配置，音频右侧补零至特征提取器 hop 大小。使用完整音频与采样视频帧，参考答案不进入模型。展开后的 prompt token 哈希须匹配公共 HF processor 输出；这不等于独立校验全部 vLLM 编码器特征张量。

两后端都禁用前缀 KV 复用；vLLM 也禁用多模态 processor 缓存。HF 在最后位置读取不变的完整 LM Head 后选择单 token 标签；vLLM 使用原 LM Head/pooling。HF 四卡分层，vLLM TP4，内核差异仍存在。

并发为一、逐题串行计时，包含问题模板/processor 到 CUDA 同步后的评分完成。公共媒体解码和模型启动单独记录，不计入逐题评分。还执行不计时的公共预处理校验。总评测墙钟时间包含校验、解码、预热与记录，不代表生产吞吐。每模态/模式的预热不计入评分样本；每题一次观测，P95 是跨问题分位数，不是同请求重复或并发压力测试的 P95。

Infinity 参考标签自动生成且未人工审核，只报告参考一致率。音视频使用原始多选标签及一致同意的 Clotho 二元适配，明确排除自由回答/分歧参考。公共解码/processor 拒绝保存在 `rejected.json`，配对报告要求两后端接收相同 ID。

逐题/模式保存原 logits、概率、决策、耗时、token 哈希与参考答案。报告原始标签 Accuracy（或生成参考一致率）、NLL、Brier、ECE、平均/中位/P95 耗时、串行评分速率、参考并列与答案变化。第三方标签和媒体不随 Apache-2.0 代码发布。

## 未缓存实测：2026-09-26

四张 24 GB NVIDIA GPU，官方 BF16 Thinker，串行请求，每题/模式一次计时。vLLM 的 causal/isolated 都使用 mJev 自定义 dense SDPA，因此不是 stock vLLM 性能测试。未训练或更改权重。

| 数据 | 题数 | 模式 | vLLM 指标 | HF 指标 | vLLM 秒/题 | HF 秒/题 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 音频 | 526 | causal | 81.37% | 81.37% | 0.129 | 0.309 |
| 音频 | 526 | isolated | 76.81% | 78.33% | 0.129 | 0.315 |
| 带音轨视频 | 423 | causal | 51.77% | 52.25% | 0.929 | 0.819 |
| 带音轨视频 | 423 | isolated | 11.11% | 10.64% | 0.932 | 1.031 |

Infinity 是生成参考一致率，不是人工真值准确率。音频为 226 道 Clotho-AQA 一致参考二元题和 300 道 MuChoMusic；视频为 411 道 MMOU 和 12 道 Video-MME-v2。音视频按原数据标签评分，不是官方榜单聚合分数。

原始 1,335 道音视频题先排除 374 道自由回答/分歧 Clotho 题，冻结 961 道音视频题和 300 道图片题。两后端再拒绝相同 12 题：10 题超过完整输入 4000 tokens；一个视频解码失败影响两题（torchcodec 帧耗尽，随后 torchvision read_video 回退不可用）。最终配对 1,249 题，每后端 2,498 行，共 4,996 行。

独立重算核对覆盖率、token 哈希、决策、softmax、指标聚合及零缓存 tokens，不证明全部编码器特征或每个 checkpoint 分片相同。仍有生成图片参考、筛选子集、无重复运行、无并发吞吐测量等限制。私人结果与第三方媒体不发布。当前结果偏向 causal，但未证明隔离性能下降的具体原因。后续稳定配置与缓存测试另见 [稳定性文档](stability.zh-CN.md)。

> 命令中的 MODEL_REVISION 必须设为该本地模型真实对应的 40 字符 commit。这里的 native 命令是显式选择的重跑配置，不是对历史配置缺失项的补填，也不保证重现旧成绩。公开可下载流程请用 [公开小型评测](reproduce.zh-CN.md)。
