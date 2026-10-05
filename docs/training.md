**English** · [简体中文](training.zh-CN.md)

[Back to mJev](../README.md) · [Model selection](models.md)

# Training and reward

The released **mJev-Qwen3-VL-4B-RLCD** model is fine-tuned with [GRPO](https://arxiv.org/abs/2402.03300) for candidate selection. Before training, a frozen scorer assigns each example a target probability $p^{\ast}$: the probability of the correct option within its candidate set. Each sampled answer receives:

```math
r(\hat{y}) = \begin{cases}
+p^{\ast}, & \hat{y} = y^{\ast} \\
-p^{\ast}, & \text{otherwise}
\end{cases}
```

Here $y^{\ast}$ is the correct label; incorrect or invalid answers receive the negative reward. Rewards lie in $[-1, 1]$, with higher target probabilities producing stronger signals. Outputs are constrained to one candidate label, so no separate format reward is used. Reward scaling is disabled to preserve this weighting, and a separate KL penalty limits drift from the reference model. Training updates the language model while freezing the vision tower and aligner.

## Document training

mjev-doc uses a separate document corpus and checkpoint. Its runnable stage pipeline and exact objective are in [mjev-doc training](docjev/training.md); use `scripts/train_docjev.sh` with `configs/docjev_rlcd.json`. Adding these modules does not rerun or change the released mJev model.
