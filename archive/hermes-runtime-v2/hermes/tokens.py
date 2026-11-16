"""Token 估算与截断。

不依赖 tiktoken 的下载，保证离线、确定性；中文按 1 字 ≈ 1 token，英文按 4 字符 ≈ 1 token。
对上下文预算控制来说，稳定且偏保守的估算比"精确"更重要。
"""
from __future__ import annotations

_CJK_RANGES = (
    ("\u4e00", "\u9fff"),
    ("\u3000", "\u303f"),
    ("\uff00", "\uffef"),
)


def _is_cjk(ch: str) -> bool:
    return any(lo <= ch <= hi for lo, hi in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return max(1, int(cjk + other / 4) + 1)


def truncate_tokens(text: str, max_tokens: int, *, marker: str = " …[已截断]… ") -> str:
    """保留头尾、按 token 预算截断，避免把关键结论（常在末尾）丢掉。"""
    if max_tokens <= 0 or estimate_tokens(text) <= max_tokens:
        return text
    marker_tokens = estimate_tokens(marker)
    if max_tokens <= marker_tokens + 2:
        return text[: max(1, max_tokens)]

    # 头 60% / 尾 40%
    head_budget = int((max_tokens - marker_tokens) * 0.6)
    tail_budget = max_tokens - marker_tokens - head_budget

    head_chars = _chars_for_tokens(text, head_budget)
    tail_chars = _chars_for_tokens(text[::-1], tail_budget)
    return text[:head_chars] + marker + (text[-tail_chars:] if tail_chars else "")


def _chars_for_tokens(text: str, token_budget: int) -> int:
    """按平均密度换算字符数，取保守值。"""
    if token_budget <= 0:
        return 0
    if not text:
        return 0
    tokens = estimate_tokens(text)
    if tokens <= token_budget:
        return len(text)
    return max(1, int(len(text) * token_budget / tokens))


def count_message_tokens(content: str) -> int:
    """Chat 消息除正文外的固定开销（role/分隔符），按 4 tokens 估算。"""
    return estimate_tokens(content) + 4
