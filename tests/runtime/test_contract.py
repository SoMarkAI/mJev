from mjev_contract import build_question_text, common_prefix_length, expected_labels


def test_prompt_is_exact_and_has_no_system_material():
    prompt = build_question_text("颜色？", {"A": "红", "B": "蓝"})
    assert prompt == "问题：颜色？\n\nA. 红\nB. 蓝\n\n仅输出一个正确选项字母。"


def test_labels_are_contiguous():
    assert expected_labels(4) == ["A", "B", "C", "D"]


def test_common_prefix_ragged():
    assert common_prefix_length([[1, 2, 3], [1, 2, 4], [1, 2]]) == 2
