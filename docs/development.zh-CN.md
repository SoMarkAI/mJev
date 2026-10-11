[English](development.md) &nbsp;&nbsp; **简体中文**

# mJev 开发者指南

[返回 README](../README.zh-CN.md) &nbsp;&nbsp; [参与贡献](../CONTRIBUTING.zh-CN.md)

## 入口与职责

| 需要改进的行为 | 实现入口 |
| --- | --- |
| HF 命令行、输入输出 | [`demo_hf.py`](../demo_hf.py) |
| 官方模板、候选及标签校验 | [`prompt.py`](../mjev/prompt.py) |
| HF forward、媒体处理和缓存接口 | [`hf.py`](../mjev/hf.py) |
| HF 问题 batch 与缓存分支 | [`hf_batch.py`](../mjev/hf_batch.py) |
| vLLM 请求与评分 | [`engine.py`](../mjev/engine.py) |
| vLLM attention 与缓存 hooks | [`patch.py`](../mjev/patch.py) |
| 决策与并列策略 | [`decision.py`](../mjev/decision.py) |
| 文档适配、配对评测与 RLCD | [`docjev/`](../docjev/) &nbsp;&nbsp; [文档指南](docjev/README.zh-CN.md) |

文档中的项目名使用 **mJev**，Python 导入使用 `mjev`，项目环境变量使用 `MJEV_` 前缀。

## 本地开发

从仓库根目录执行；模型推理环境要求见 [HF 教程](hf.zh-CN.md)。

```bash
python -m pip install -e '.[docjev,test]'
bash scripts/test.sh -q
```

HF 测试使用小型随机权重模型检查机制，不代表官方 30B 模型的准确率。vLLM 参考环境的 CPU 与 GPU 检查命令见 [vLLM 指南](vllm.zh-CN.md)。运行 GPU 检查前自行确认资源可用。

## 实现约束

保留官方权重加载、processor、chat template 和多模态位置编码。不要把图像专用的位置逻辑直接套到音频和视频。候选标签必须在实际模板边界验证为单 token；概率只对实际候选做 softmax。

修改缓存时检查完整前缀、媒体参数、MRoPE、mask 兼容性和分支独立性。HF 当前复制分支 KV；vLLM 使用另一套调度与缓存机制，不能互相推定验证结果。

## 验证与报告

按改动选择机制检查，再决定是否需要全权重回归。对 mask/cache/numerics 改动，比较相同配置下的串行、batch、缓存、顺序变换，保存 raw logits、概率差、答案变化和计时边界。不得仅凭答案相同宣称数值一致。

当前证据与限制见 [稳定性记录](stability.zh-CN.md)、[配对评测](https://github.com/SoMarkAI/mJev/blob/9ff8af6e91c46c04b2984b1a05f637ecb599ed7a/docs/paired_backends.zh-CN.md) 和 [发布验证](https://github.com/SoMarkAI/mJev/blob/9ff8af6e91c46c04b2984b1a05f637ecb599ed7a/docs/validation.zh-CN.md)。新增结果应标明日期、版本、样本范围、精度与失败项。

## 区分两种隔离

- **问题分支独立**：同一媒体的不同问题使用独立请求／batch 行，可以复用公共前缀，但不读取其他问题分支。
- **题内候选隔离**：显式选择 `isolated` 时，同一道题的候选只读取公共上下文、问题和自身历史；最终答案位置读取全部候选。

两种机制不等价。主 HF 与 vLLM API、CLI 默认统一为 `causal`；题内候选隔离必须显式选择。候选隔离提供可控的注意力结构，可用于交互方式比较和后续定向训练实验；效果或扩展性上限仍需训练与消融验证。

输入结构校验集中在 `mjev/inputs.py`，由预检查、CLI 和评分接口共用。分组评测在 `groups/` 中逐组保存完整结果，同时更新 `progress.json`、预测与原始输出；后续失败不删除已完成组，但当前不支持自动断点续跑。


<a id="tests"></a>


Docker 之外的命令均假定已激活 Python 3.11+ 虚拟环境：先用 `python3 -m venv .venv` 创建，再运行 `source .venv/bin/activate`。激活后安装和运行统一使用 `python`；Docker 内使用 `python3`。Shell 脚本也支持 `PYTHON=/path/to/venv/bin/python`。历史执行记录保留原始命令。

## 测试命令

全新 Python 3.11+ 环境执行 `python -m pip install -e '.[docjev,test]'`，安装固定 torch/Transformers、pytest、requests 及 tiny checkpoint 测试所需 accelerate。这**不等于完整 HF 媒体部署依赖**。从 checkout 执行：

```bash
bash scripts/test.sh --suite base -q
bash scripts/test.sh --suite hf -q
bash scripts/test.sh --suite runtime -q
bash scripts/test.sh --suite vllm -q
bash scripts/test.sh -q
```

可用 `PYTHON=/path/to/venv/bin/python bash scripts/test.sh ...` 指定解释器。统一 pytest，递归发现 unittest 类与函数式测试。默认 all 包含基础、HF、runtime 与可选 vLLM CPU 套件，不运行 GPU。可选 vLLM 缺失会显示 **skip，不是通过**。直接 python -m pytest tests 使用相同配置，推荐脚本以明确关闭 runtime hooks。

| 套件 | 范围 | 限制/依赖 |
| --- | --- | --- |
| base | 数据/指标、mask、决策、报告和来源记录 | 无预训练模型评分 |
| hf | tiny 随机 Thinker forward、cache/batch、checkpoint 加载 | 不是官方 30B 准确率；需要 accelerate |
| runtime | 实验服务/Tree-KV 纯辅助逻辑、mask oracle、投影 | 投影依赖 vLLM；不是全模型 Tree-KV 一致性 |
| vllm | pooling 媒体放置及 prefix-bypass CPU 合约 | 需要固定 vLLM 环境 |
| gpu | 官方模型 vLLM 端到端机制验证 | 显式开启、四张空闲 GPU、本地权重 |

```bash
MJEV_MODEL=/path/to/official/model bash scripts/test_gpu.sh -q
```

GPU 脚本通过 pytest GPU 套件执行原 tests/e2e.py，保留原断言；未显式覆盖时设置 VLLM_BATCH_INVARIANT=1，不停止服务。HF 全权重和并发缓存实验仍为 benchmarks/integrated 的独立命令，不暗中纳入 CPU pytest，也不由本 GPU wrapper 认证。其外部数据和 numerics/revision 要求另行说明。

`bash scripts/test.sh --collect-only -q` 只检查收集。--suite 在导入前过滤，base 不强制要求其他套件依赖。不要再用 unittest discover 作仓库门槛，它会漏掉函数式测试和子目录。实际结果见 [最新验证入口](development.zh-CN.md#validation-evidence)。

### 文档模型检查

`tests/docjev/` 纳入 base suite，覆盖文档输入、双语分组、候选顺序映射、RLCD 奖励、collective padding 和 checkpoint/resume。实际双权重 GPU 推理入口见 [mJev-Doc 评测](docjev/evaluation.md)。

<a id="validation-evidence"></a>

## 保留的验证依据

CPU 机制检查不代表预训练模型准确率。历史原始报告保留在 [validation_evidence/](validation_evidence/)，沿用当时模型和配置，不能当作当前版本重跑结果。[当前文档评测](docjev/results.zh-CN.md)与[缓存测试](cache_scaling.zh-CN.md)分别展示。
