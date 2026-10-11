**English** &nbsp;&nbsp; [简体中文](training.zh-CN.md)

[Back to mJev](../README.md) &nbsp;&nbsp; [Model selection](hf.md#models)

# Training and reward

The released **mJev-Qwen3-VL-4B-RLCD** model uses RLCD training for candidate selection. Before training, a frozen scorer assigns each example a target probability $p^{\ast}$: the probability of the correct option within its candidate set. Each sampled answer receives:

```math
r(\hat{y}) = \begin{cases}
+p^{\ast}, & \hat{y} = y^{\ast} \\
-p^{\ast}, & \text{otherwise}
\end{cases}
```

Here $y^{\ast}$ is the correct label; incorrect or invalid answers receive the negative reward. Rewards lie in $[-1, 1]$, with higher target probabilities producing stronger signals. Outputs are constrained to one candidate label, so no separate format reward is used. Reward scaling is disabled to preserve this weighting, and a separate KL penalty limits drift from the reference model. Training updates the language model while freezing the vision tower and aligner.

## Recorded evaluation

On the same 195-question mJev-Compositional-VQA subset, the model card records 152/195 (77.95%) for the base model and 156/195 (80.00%) after training. These are archived subset results, not a new run or full-benchmark scores. [Model-card record](https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD/blob/7af7cda55ea36e76bce64e87a02c9faea1919d64/README.md#evaluation).

## Document training

mJev-Doc selects a full-SFT checkpoint at step 990. Its [model background](docjev/training.md) distinguishes this checkpoint from the archived RLCD experiments. The general mJev RLCD model above is a separate checkpoint.
