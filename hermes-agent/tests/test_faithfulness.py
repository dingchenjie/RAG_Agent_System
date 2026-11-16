"""忠实度：引用支撑度的阈值与边界。"""
from __future__ import annotations

from hermes.eval.faithfulness import citation_support_rate


def test_supported_answer_scores_one():
    contexts = ["LoRA 的秩 r 常用取值是 8 或 16。"]
    answer = "LoRA 的秩 r 常用取值是 8 或 16。[1]"
    assert citation_support_rate(answer, contexts) == 1.0


def test_unsupported_answer_scores_zero():
    contexts = ["LoRA 的秩 r 常用取值是 8 或 16。"]
    answer = "超导量子比特的退相干时间大约是一百微秒。"
    assert citation_support_rate(answer, contexts) == 0.0


def test_empty_inputs_and_partial_support():
    assert citation_support_rate("", ["任意上下文"]) == 0.0
    assert citation_support_rate("任意回答", []) == 0.0

    contexts = ["LoRA 的秩 r 常用取值是 8。"]
    answer = "LoRA 的秩 r 常用取值是 8。完全无关的编造结论在这里。"
    assert citation_support_rate(answer, contexts) == 0.5
