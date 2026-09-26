from mjev_mask import (
    MaskedQuestionLayout,
    TokenSpan,
    TreeKVPlan,
    build_masked_layout,
    dense_oracle_mask,
    original_position_ids,
)


def example_plan():
    return TreeKVPlan(
        common=TokenSpan(0, 3),
        candidates=(TokenSpan(3, 5), TokenSpan(5, 8), TokenSpan(8, 9)),
        decision=TokenSpan(9, 11),
    )


def test_candidates_are_isolated_and_decision_sees_all():
    mask = dense_oracle_mask(example_plan())
    assert mask[4][3] and not mask[4][5]
    assert mask[7][5] and not mask[7][3]
    assert mask[8][8] and not mask[8][5]
    assert all(mask[9][key] for key in range(9))


def test_original_positions_are_not_renumbered():
    positions = original_position_ids(example_plan())
    assert positions["candidates"] == ((3, 4), (5, 6, 7), (8,))
    assert positions["decision"] == (9, 10)


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens and return_offsets_mapping
        return {
            "input_ids": [1000 + ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_native_prompt_is_split_without_retokenizing_candidates():
    tokenizer = CharacterTokenizer()
    text = "问题：颜色？\n\nA. 红\nB. 蓝\n\n仅输出一个正确选项字母。"
    text_ids = tokenizer(text, return_offsets_mapping=True)["input_ids"]
    full = [7, 8, *text_ids, 9, 10]
    layout = build_masked_layout(full, tokenizer, "颜色？", {"A": "红", "B": "蓝"})

    assert layout.prompt_token_ids == tuple(full)
    assert layout.compact_candidate(0) != layout.compact_candidate(1)
    assert layout.plan.decision.end == len(full)
    assert MaskedQuestionLayout.from_payload(layout.to_payload()) == layout


class TrailingSeparatorTokenizer:
    """Character tokenizer with one token merged across a whitespace boundary."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens and return_offsets_mapping
        boundary = text.index("\nB.")
        offsets = []
        cursor = 0
        while cursor < len(text):
            end = cursor + 1
            if cursor == boundary - 1:
                end = boundary + 1  # final A character + the line separator
            offsets.append((cursor, end))
            cursor = end
        return {
            "input_ids": [2000 + index for index in range(len(offsets))],
            "offset_mapping": offsets,
        }


def test_whitespace_suffix_merged_into_previous_candidate_is_safe():
    tokenizer = TrailingSeparatorTokenizer()
    text = "问题：颜色？\n\nA. 红\nB. 蓝\n\n仅输出一个正确选项字母。"
    text_ids = tokenizer(text, return_offsets_mapping=True)["input_ids"]
    full = [*text_ids]
    layout = build_masked_layout(full, tokenizer, "颜色？", {"A": "红", "B": "蓝"})

    assert layout.plan.candidates[0].end == layout.plan.candidates[1].start
    assert layout.compact_candidate(0)[-1] == text_ids[layout.plan.candidates[0].end - 1]


def test_tree_kv_attention_matches_dense_block_mask():
    torch = __import__("torch")
    plan = example_plan()
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(plan.length, 4, generator=generator)
    key = torch.randn(plan.length, 4, generator=generator)
    value = torch.randn(plan.length, 3, generator=generator)

    def attend(query_row, key_rows, value_rows):
        weights = torch.softmax(query_row @ key_rows.T / 2.0, dim=-1)
        return weights @ value_rows

    mask = dense_oracle_mask(plan)
    dense = []
    for row in range(plan.length):
        visible = torch.tensor(mask[row], dtype=torch.bool)
        dense.append(attend(query[row], key[visible], value[visible]))
    dense = torch.stack(dense)

    # Candidate branches contain common + exactly one candidate.  Decision
    # rows read the logical Tree-KV order common + A + B + C + prior decision.
    tree = torch.empty_like(dense)
    for row in range(plan.common.end):
        visible = list(range(plan.common.start, row + 1))
        tree[row] = attend(query[row], key[visible], value[visible])
    for candidate in plan.candidates:
        for row in range(candidate.start, candidate.end):
            visible = [*range(plan.common.start, plan.common.end), *range(candidate.start, row + 1)]
            tree[row] = attend(query[row], key[visible], value[visible])
    for row in range(plan.decision.start, plan.decision.end):
        visible = [*range(plan.common.start, plan.decision.start), *range(plan.decision.start, row + 1)]
        tree[row] = attend(query[row], key[visible], value[visible])

    torch.testing.assert_close(tree, dense, rtol=1e-6, atol=1e-6)
