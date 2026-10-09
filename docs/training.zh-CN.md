[English](training.md) · **简体中文**

[返回 mJev](../README.zh-CN.md) · [模型选择](models.zh-CN.md)

# 训练与奖励设计

发布的 **mJev-Qwen3-VL-4B-RLCD** 模型使用 RLCD 针对候选选择进行训练。训练前，冻结的评分器为每条样本计算目标概率 $p^{\ast}$，即正确选项在候选集合内的概率。每次采样的回答按下式获得奖励：

```math
r(\hat{y}) = \begin{cases}
+p^{\ast}, & \hat{y} = y^{\ast} \\
-p^{\ast}, & \text{otherwise}
\end{cases}
```

其中 $y^{\ast}$ 为正确标签，错误或无效回答获得负奖励。奖励范围为 $[-1, 1]$，目标概率越高，训练信号越强。输出被限制为单个候选标签，因此不另加格式奖励；关闭 reward scaling 以保留这一权重，并通过独立的 KL 惩罚限制策略偏离参考模型。训练更新语言模型，视觉塔和对齐模块保持冻结。

## 文档模型训练

mJev-Doc 使用独立的文档数据与 checkpoint。[mJev-Doc 训练说明](docjev/training.md)包含完整阶段与奖励定义；入口为 `scripts/train_docjev.sh`，配置为 `configs/docjev_rlcd.json`。代码整合不会重新训练或改变已发布的 mJev 模型。
