**English** · [简体中文](CONTRIBUTING.zh-CN.md)

# Contributing to mJev

Thank you for helping improve mJev. We welcome documentation fixes, reproducible bug reports, tests, backend integrations and performance improvements with clearly defined measurement boundaries.

## Report an issue

Include the code version, operating system, GPU, dependency versions, backend, numerical profile, minimal input, reproduction command, expected behavior and actual behavior. Remove credentials, private media and internal server details from logs before sharing them.

## Submit a change

1. Create a focused branch that addresses one issue.
2. Read the [developer guide](docs/development.md) and update the relevant implementation and documentation.
3. Run checks appropriate to the change and describe the results in your Merge Request or Pull Request.
4. For model behavior changes, explain whether they affect prompts, candidate order, masks, positional encoding, caching or weight loading.
5. Submit for review with the necessary reproduction steps and known limitations.

## Results and assets

- Distinguish tiny/random-weight unit tests, real-weight tests and benchmark accuracy.
- Fix samples, numerical precision and processing parameters in performance comparisons. State whether timing includes model loading, media decoding and prefill.
- Do not treat a cache hit or matching final answers as proof that all logits match.
- Do not submit API keys, tokens, internal host details, model weights, checkpoints or restricted media.
- Preserve the original licenses and attribution of third-party code and data; do not relicense them all as Apache-2.0.

By contributing, you confirm that you have the right to provide the submitted material and agree to distribute your original code under this project's Apache-2.0 license. Third-party assets remain subject to their original terms.
