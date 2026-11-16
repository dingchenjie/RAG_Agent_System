"""忠实度评测：离线的"引用支撑度" + 可选 RAGAS。

为什么默认不用 RAGAS：
- RAGAS 需要 LLM 判官（额外成本、非确定性、离线不可跑）；
- Agent 场景下更关键的问题是"答案有没有落在检索到的证据上"，这可以用
  逐句 n-gram 覆盖率离线近似；
- 有 Key 想对齐学术界口径时，再通过 ``[ragas]`` extra 打开即可。
"""
from __future__ import annotations

import re

from ..knowledge.embedding import tokenize

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;.\n])")
_CITATION_RE = re.compile(r"\[\d+\]")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if len(s.strip()) > 1]


def citation_support_rate(
    answer: str,
    contexts: list[str],
    *,
    threshold: float = 0.6,
) -> float:
    """答案逐句被引用片段支撑的比例（0..1）。

    单句的 token 在任一引用片段中的覆盖率 ≥ ``threshold`` 即视为"有据可依"；
    引用标记 ``[1]`` 先剔除，避免把编号当成内容。
    """
    sentences = _sentences(_CITATION_RE.sub("", answer or ""))
    if not sentences:
        return 0.0
    context_tokens = [set(tokenize(context)) for context in contexts or [] if context]
    if not context_tokens:
        return 0.0

    supported = 0
    for sentence in sentences:
        tokens = set(tokenize(sentence))
        if not tokens:
            supported += 1  # 纯标点/数字分隔，不构成幻觉
            continue
        best = max(len(tokens & context) / len(tokens) for context in context_tokens)
        if best >= threshold:
            supported += 1
    return round(supported / len(sentences), 4)


def ragas_faithfulness(
    answer: str,
    contexts: list[str],
    *,
    question: str = "",
) -> float | None:
    """可选 RAGAS 适配：未安装或运行失败都返回 ``None``，绝不阻断评测主流程。"""
    try:  # pragma: no cover - 依赖可选包与外部服务
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import Faithfulness
    except ImportError:
        return None
    try:  # pragma: no cover
        sample = SingleTurnSample(
            user_input=question or answer, response=answer,
            retrieved_contexts=list(contexts),
        )
        result = evaluate(EvaluationDataset(samples=[sample]), metrics=[Faithfulness()])
        return float(result["faithfulness"])
    except Exception:  # noqa: BLE001 评测降级不抛错
        return None
