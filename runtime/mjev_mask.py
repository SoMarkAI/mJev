"""Masked-attention semantics and Tree-KV merge descriptors.

The dense mask is an oracle for short tests only.  Production consumes compact
``MaskedQuestionLayout`` metadata and expresses the same visibility through
PagedAttention block tables plus token-level copies for partial pages.  It
must never materialize a quadratic attention mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mjev_contract import build_question_text, validate_criteria


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("token spans must be non-empty and non-negative")


@dataclass(frozen=True)
class TreeKVPlan:
    common: TokenSpan
    candidates: tuple[TokenSpan, ...]
    decision: TokenSpan

    def __post_init__(self) -> None:
        if len(self.candidates) < 2 or len(self.candidates) > 26:
            raise ValueError("a masked question needs 2--26 candidates")
        cursor = self.common.end
        for candidate in self.candidates:
            if candidate.start != cursor:
                raise ValueError("candidate spans must exactly partition the prompt")
            cursor = candidate.end
        if self.decision.start != cursor:
            raise ValueError("decision suffix must follow the final candidate")

    @property
    def length(self) -> int:
        return self.decision.end

    @property
    def logical_merge_order(self) -> tuple[TokenSpan, ...]:
        return (self.common, *self.candidates)


@dataclass(frozen=True)
class MaskedQuestionLayout:
    """Token-exact layout of one native-template, fully processed prompt."""

    prompt_token_ids: tuple[int, ...]
    plan: TreeKVPlan

    def __post_init__(self) -> None:
        if self.plan.common.start != 0:
            raise ValueError("the common span must start at token zero")
        if self.plan.length != len(self.prompt_token_ids):
            raise ValueError("Tree-KV spans must cover the complete native prompt")

    def compact_candidate(self, index: int) -> list[int]:
        span = self.plan.candidates[index]
        return [
            *self.prompt_token_ids[: self.plan.common.end],
            *self.prompt_token_ids[span.start : span.end],
        ]

    def to_payload(self) -> dict[str, Any]:
        return {
            "prompt_token_ids": list(self.prompt_token_ids),
            "common_end": self.plan.common.end,
            "candidate_spans": [
                [span.start, span.end] for span in self.plan.candidates
            ],
            "decision_span": [self.plan.decision.start, self.plan.decision.end],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MaskedQuestionLayout":
        token_ids = payload.get("prompt_token_ids")
        common_end = payload.get("common_end")
        candidate_spans = payload.get("candidate_spans")
        decision_span = payload.get("decision_span")
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(not isinstance(token, int) or isinstance(token, bool) for token in token_ids)
            or not isinstance(common_end, int)
            or isinstance(common_end, bool)
            or not isinstance(candidate_spans, list)
            or not isinstance(decision_span, list)
            or len(decision_span) != 2
        ):
            raise ValueError("invalid masked question payload")
        candidates = tuple(TokenSpan(*values) for values in candidate_spans)
        return cls(
            tuple(token_ids),
            TreeKVPlan(
                common=TokenSpan(0, common_end),
                candidates=candidates,
                decision=TokenSpan(*decision_span),
            ),
        )


def _find_last_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    if not needle or len(needle) > len(haystack):
        raise ValueError("question token sequence is absent from the native prompt")
    for start in range(len(haystack) - len(needle), -1, -1):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return start
    raise ValueError(
        "native prompt tokenization changed at the question boundary; "
        "masked mode refuses an inexact split"
    )


def _token_boundary(
    offsets: Sequence[tuple[int, int]], char_offset: int, text: str
) -> int:
    """Translate a character boundary into a safe token boundary.

    BPE tokenizers may merge the final character of a candidate with the
    following line separator (for example ``"%\\n\\n"``).  The separator has
    no semantic content from the next candidate, so the complete token can be
    assigned to the preceding span.  A token that carries non-whitespace text
    across the boundary still fails closed: exact Tree-KV isolation cannot be
    represented without changing the native token sequence in that case.
    """
    if char_offset == 0:
        return 0
    for index, (start, end) in enumerate(offsets):
        if start < char_offset < end:
            if not text[char_offset:end].strip():
                return index + 1
            raise ValueError(
                f"character boundary {char_offset} splits tokenizer token {index}"
            )
        if start >= char_offset:
            return index
    if offsets and offsets[-1][1] <= char_offset:
        return len(offsets)
    raise ValueError(f"cannot map character boundary {char_offset} to tokens")


def build_masked_layout(
    full_prompt_token_ids: Sequence[int],
    tokenizer: Any,
    instructions: str,
    criteria: Mapping[str, str],
) -> MaskedQuestionLayout:
    """Split one native prompt without changing a single model-visible token.

    Candidate line separators are assigned to the preceding candidate.  The
    blank line and the fixed decision instruction belong to the decision
    suffix.  Every boundary is checked against tokenizer offsets; if a future
    checkpoint merges across a boundary, masked execution fails closed.
    """

    validate_criteria(criteria)
    question_text = build_question_text(instructions, criteria)
    common_text = f"问题：{instructions}\n\n"
    candidate_lines = [f"{label}. {text}" for label, text in criteria.items()]
    decision_text = "\n\n仅输出一个正确选项字母。"
    if question_text != common_text + "\n".join(candidate_lines) + decision_text:
        raise AssertionError("masked layout diverged from the fixed prompt contract")

    encoded = tokenizer(
        question_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    question_ids = [int(token) for token in encoded["input_ids"]]
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    base = _find_last_subsequence(full_prompt_token_ids, question_ids)

    common_char_end = len(common_text)
    candidate_char_spans: list[tuple[int, int]] = []
    cursor = common_char_end
    for index, line in enumerate(candidate_lines):
        end = cursor + len(line)
        if index + 1 < len(candidate_lines):
            end += 1  # The single line separator stays with this branch.
        candidate_char_spans.append((cursor, end))
        cursor = end
    decision_char_start = cursor
    if question_text[decision_char_start:] != decision_text:
        raise AssertionError("decision suffix boundary is inconsistent")

    common_end = base + _token_boundary(offsets, common_char_end, question_text)
    candidates = tuple(
        TokenSpan(
            base + _token_boundary(offsets, start, question_text),
            base + _token_boundary(offsets, end, question_text),
        )
        for start, end in candidate_char_spans
    )
    decision_start = base + _token_boundary(offsets, decision_char_start, question_text)
    plan = TreeKVPlan(
        common=TokenSpan(0, common_end),
        candidates=candidates,
        decision=TokenSpan(decision_start, len(full_prompt_token_ids)),
    )

    # This is the critical invariant: branches are views of the already
    # processed native prompt, never independently re-tokenized alternatives.
    rebuilt = list(full_prompt_token_ids[:common_end])
    for span in candidates:
        rebuilt.extend(full_prompt_token_ids[span.start : span.end])
    rebuilt.extend(full_prompt_token_ids[decision_start:])
    if rebuilt != list(full_prompt_token_ids):
        raise ValueError("masked spans do not reconstruct the native prompt exactly")
    return MaskedQuestionLayout(tuple(full_prompt_token_ids), plan)


def dense_oracle_mask(plan: TreeKVPlan) -> list[list[bool]]:
    """Build the KEV-style block-causal oracle for parity tests only."""
    length = plan.length
    mask = [[False] * length for _ in range(length)]
    # Common prefix is ordinary causal attention.
    for query in range(plan.common.start, plan.common.end):
        for key in range(plan.common.start, query + 1):
            mask[query][key] = True
    # A candidate sees common + itself causally, never another candidate.
    for candidate in plan.candidates:
        for query in range(candidate.start, candidate.end):
            for key in range(plan.common.start, plan.common.end):
                mask[query][key] = True
            for key in range(candidate.start, query + 1):
                mask[query][key] = True
    # The decision suffix sees the merged common/candidate KV and itself.
    for query in range(plan.decision.start, plan.decision.end):
        for key in range(plan.common.start, plan.decision.start):
            mask[query][key] = True
        for key in range(plan.decision.start, query + 1):
            mask[query][key] = True
    return mask


def original_position_ids(plan: TreeKVPlan) -> dict[str, object]:
    """Positions each branch must retain from the unmodified full prompt."""
    return {
        "common": tuple(range(plan.common.start, plan.common.end)),
        "candidates": tuple(
            tuple(range(span.start, span.end)) for span in plan.candidates
        ),
        "decision": tuple(range(plan.decision.start, plan.decision.end)),
    }
