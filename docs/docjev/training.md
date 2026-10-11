# Model training background

[Project](../../README.md) · [Evaluation](evaluation.md)

The selected mJev-Doc checkpoint is a full-SFT checkpoint at optimizer step 990, including visual merger updates. It is a complete checkpoint without an RL adapter. Check the Hugging Face model card and immutable revision to identify the published weights.

The full-SFT training pipeline is maintained separately and is not included in this inference repository. The earlier candidate-restricted RLCD tools do not reproduce that SFT run; their code and experiment details remain in the [historical checkout](https://github.com/SoMarkAI/mJev/tree/a313f135041cf893db27a81f40052b9e66b2b5a2/docjev/rlcd) and [training record](https://github.com/SoMarkAI/mJev/tree/a313f135041cf893db27a81f40052b9e66b2b5a2/docs/docjev/training.md).

The general mJev-Qwen3-VL-4B-RLCD model is a separate checkpoint; see its [training background](../training.md). Use [paired evaluation](evaluation.md#compare-original-and-trained-weights) to compare complete local checkpoints under the same input and scoring settings.
