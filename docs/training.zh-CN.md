[English](training.md) &nbsp;&nbsp; **简体中文**

[返回 mJev](../README.zh-CN.md) &nbsp;&nbsp; [模型选择](hf.zh-CN.md#models)

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

mJev-Doc 选用全参 SFT 第 990 步 checkpoint。[模型背景](docjev/training.md)说明该权重与历史 RLCD 实验的区别；上方通用 mJev 的 RLCD 权重是另一模型。
