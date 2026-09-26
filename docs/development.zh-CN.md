[English](development.md) · **简体中文**

# mJev 开发者指南

[返回 README](../README.zh-CN.md) · [参与贡献](../CONTRIBUTING.zh-CN.md)

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

对外项目名统一为 **mJev**，大小写固定。下方的小写包名／导入标识符及大写环境变量名属于需要保持兼容的技术名称。

Python 包与导入路径统一为 `mjev`，环境变量统一为 `MJEV_` 前缀；调用方需使用当前命名。

## 本地开发

从仓库根目录执行；模型推理环境要求见 [HF 教程](hf.zh-CN.md)。

```bash
python -m pip install -e '.[test]'
bash scripts/test.sh -q
```

HF 测试使用小型随机权重模型检查机制，不代表官方 30B 模型的准确率。vLLM 参考环境的 CPU 与 GPU 检查命令见 [vLLM 指南](vllm.zh-CN.md)。运行 GPU 检查前自行确认资源可用。

## 实现约束

保留官方权重加载、processor、chat template 和多模态位置编码。不要把图像专用的位置逻辑直接套到音频和视频。候选标签必须在实际模板边界验证为单 token；概率只对实际候选做 softmax。

修改缓存时检查完整前缀、媒体参数、MRoPE、mask 兼容性和分支独立性。HF 当前复制分支 KV；vLLM 使用另一套调度与缓存机制，不能互相推定验证结果。

## 验证与报告

按改动选择机制检查，再决定是否需要全权重回归。对 mask/cache/numerics 改动，比较相同配置下的串行、batch、缓存、顺序变换，保存 raw logits、概率差、答案变化和计时边界。不得仅凭答案相同宣称数值一致。

当前证据与限制见 [稳定性记录](stability.zh-CN.md)、[配对评测](paired_backends.zh-CN.md) 和 [发布验证](validation.zh-CN.md)。新增结果应标明日期、版本、样本范围、精度与失败项。

## 复现与最新验证

- [公开小型评测：数据准备 → 推理 → 报告](reproduce.zh-CN.md)
- [统一最新验证入口与后端边界](validation_current.zh-CN.md)
- [pytest 测试范围与入口](testing.zh-CN.md)
- [隔离环境安装实测与限制](installation_validation.zh-CN.md)

## 区分两种隔离

- **问题分支独立**：同一媒体的不同问题使用独立请求／batch 行，可以复用公共前缀，但不读取其他问题分支。
- **题内候选隔离**：显式选择 `isolated` 时，同一道题的候选只读取公共上下文、问题和自身历史；最终答案位置读取全部候选。

两种机制不等价。主 HF 与 vLLM API、CLI 默认统一为 `causal`；题内候选隔离必须显式选择。候选隔离提供可控的注意力结构，可用于交互方式比较和后续定向训练实验；效果或扩展性上限仍需训练与消融验证。

输入结构校验集中在 `mjev/inputs.py`，由预检查、CLI 和评分接口共用。分组评测在 `groups/` 中逐组保存完整结果，同时更新 `progress.json`、预测与原始输出；后续失败不删除已完成组，但当前不支持自动断点续跑。
