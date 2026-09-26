[English](testing.md) · **简体中文**

Docker 之外的命令均假定已激活 Python 3.11+ 虚拟环境：先用 `python3 -m venv .venv` 创建，再运行 `source .venv/bin/activate`。激活后安装和运行统一使用 `python`；Docker 内使用 `python3`。Shell 脚本也支持 `PYTHON=/path/to/venv/bin/python`。历史执行记录保留原始命令。

# 统一测试入口

全新 Python 3.11+ 环境执行 `python -m pip install -e '.[test]'`，安装固定 torch/Transformers、pytest、requests 及 tiny checkpoint 测试所需 accelerate。这**不等于完整 HF 媒体部署依赖**。从 checkout 执行：

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

`bash scripts/test.sh --collect-only -q` 只检查收集。--suite 在导入前过滤，base 不强制要求其他套件依赖。不要再用 unittest discover 作仓库门槛，它会漏掉函数式测试和子目录。实际结果见 [最新验证入口](validation_current.zh-CN.md)。
