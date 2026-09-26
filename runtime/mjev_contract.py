"""Validation and prompt helpers shared by the mJev service and tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping


MODEL_NAME = "Qwen3-Omni-30B-A3B-Instruct"
MIN_QUESTIONS = 1
MAX_QUESTIONS = 6
MIN_CANDIDATES = 2
MAX_CANDIDATES = 26
VALID_MODES = frozenset({"causal", "masked"})
VALID_MEDIA_TYPES = frozenset({"image_url", "video_url", "audio_url"})


def expected_labels(count: int) -> list[str]:
    if not MIN_CANDIDATES <= count <= MAX_CANDIDATES:
        raise ValueError(f"candidate count must be {MIN_CANDIDATES}--{MAX_CANDIDATES}")
    return [chr(ord("A") + index) for index in range(count)]


def validate_criteria(criteria: Mapping[str, str]) -> None:
    labels = list(criteria)
    expected = expected_labels(len(labels))
    if labels != expected:
        raise ValueError(f"criteria labels must be contiguous: {expected}")
    if any(not isinstance(value, str) or not value.strip() for value in criteria.values()):
        raise ValueError("candidate text must not be blank")


def build_question_text(instructions: str, criteria: Mapping[str, str]) -> str:
    """Return the fixed, minimal user text. The checkpoint adds chat framing."""
    validate_criteria(criteria)
    if not instructions.strip():
        raise ValueError("question instructions must not be blank")
    candidates = "\n".join(f"{label}. {text}" for label, text in criteria.items())
    return f"问题：{instructions}\n\n{candidates}\n\n仅输出一个正确选项字母。"


def validate_single_token_labels(tokenizer: object) -> dict[str, int]:
    """Resolve A--Z from the loaded checkpoint; never assume token IDs."""
    result: dict[str, int] = {}
    encode = getattr(tokenizer, "encode")
    decode = getattr(tokenizer, "decode")
    for label in expected_labels(MAX_CANDIDATES):
        token_ids = encode(label, add_special_tokens=False)
        if len(token_ids) != 1 or decode(token_ids) != label:
            raise RuntimeError(
                f"checkpoint label {label!r} is not one round-tripping token: {token_ids}"
            )
        result[label] = int(token_ids[0])
    if len(set(result.values())) != MAX_CANDIDATES:
        raise RuntimeError("A--Z do not map to distinct checkpoint tokens")
    return result


def common_prefix_length(rows: Iterable[list[int]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    limit = min(map(len, rows))
    for index in range(limit):
        value = rows[0][index]
        if any(row[index] != value for row in rows[1:]):
            return index
    return limit


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class LabelSet:
    labels: tuple[str, ...]
    token_ids: tuple[int, ...]

    @classmethod
    def for_criteria(
        cls, criteria: Mapping[str, str], checkpoint_labels: Mapping[str, int]
    ) -> "LabelSet":
        validate_criteria(criteria)
        labels = tuple(criteria)
        return cls(labels, tuple(checkpoint_labels[label] for label in labels))
