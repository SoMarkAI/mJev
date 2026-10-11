# 开发与测试

[English](development.md) · [首页](../README.zh-CN.md) · [参与贡献](../CONTRIBUTING.zh-CN.md)

## 代码入口

| 功能 | 位置 |
|---|---|
| 通用 HF 推理 | `demo_hf.py`、`mjev/hf.py` |
| 文档输入、推理、HTTP 服务 | `docjev/io.py`、`docjev/engine.py`、`docjev/service.py` |
| 模板、标签与输入校验 | `mjev/prompt.py`、`mjev/inputs.py` |
| 批处理、前缀缓存、数值模式 | `mjev/hf_batch.py`、`mjev/hf_request_cache.py`、`mjev/hf_numerics.py` |
| vLLM 评分与 hooks | `demo.py`、`mjev/engine.py`、`mjev/patch.py` |
| 文档评测、模型比较 | `docjev/evaluate.py`、`docjev/compare.py` |
| 通用评测、缓存检查 | `mjev/benchmark/`、`benchmarks/integrated/validate_model.py` |

保留官方权重加载、processor、chat template 和位置编码。候选标签在实际模板边界校验，概率只对实际候选做 softmax；参考答案不得进入推理输入。缓存改动需检查分支独立、mask、位置编码和媒体参数，并在相同数值模式下比较原始 logits。

<a id="tests"></a>

## 测试

Python 3.11+ 虚拟环境，从仓库根目录执行：

```bash
python -m pip install -e '.[docjev,test]'
bash scripts/test.sh -q
```

可用 `PYTHON=/path/to/venv/bin/python bash scripts/test.sh -q` 指定解释器。默认执行所有 CPU 测试；GPU 测试显式开启。使用 `--suite base`、`--suite hf`、`--suite vllm` 选择套件，`--collect-only` 只检查收集。vLLM 缺失显示 skip。

- base：输入与指标、服务调度、决策、文档评测和通用评测合约。
- hf：随机小模型 forward、batch、缓存、数值与 checkpoint 加载；不代表预训练准确率。
- vllm：固定 vLLM 环境中的媒体放置和 prefix-bypass 合约。
- gpu：官方 Omni 模型的 vLLM 端到端机制检查，需要四张空闲 GPU 和本地完整权重。

```bash
MJEV_MODEL=/path/to/official/model bash scripts/test_gpu.sh -q
```

[服务测试](../tests/docjev/test_service.py)覆盖 HTTP 协议、任务队列和缓存调度。真实 checkpoint 的评分与性能需要另行使用[文档评测](docjev/evaluation.md)和[通用评测](benchmark.zh-CN.md)验证。

<a id="validation-evidence"></a>

## 验证依据

[当前文档成绩](docjev/results.zh-CN.md)和[缓存测试](cache_scaling.zh-CN.md)保留各自的样本及运行条件。缓存原始数据和核验器保留在 `docs/validation_evidence/`；历史 harness 保持原样；新测量按缓存报告中的步骤导出所选源码提交。

其他历史验证报告、实验 HTTP runtime、AV 数据下载和 RLCD 训练工具见[精简前固定提交](https://github.com/SoMarkAI/mJev/tree/a313f135041cf893db27a81f40052b9e66b2b5a2)。这些历史记录不是当前代码的新一轮 GPU 测试。
