"""评测指标：工具 F1、关键字覆盖、引用率、延迟分位数。"""
from __future__ import annotations


def tool_f1(expected: list[str], used: list[str]) -> tuple[float, float, float]:
    exp, used_set = set(expected), set(used)
    tp = len(exp & used_set)
    precision = tp / len(used_set) if used_set else 0.0
    recall = tp / len(exp) if exp else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def keyword_coverage(keywords: list[str], text: str) -> float:
    if not keywords:
        return 1.0
    hit = sum(1 for kw in keywords if kw in text)
    return round(hit / len(keywords), 4)


def citation_rate(text: str) -> float:
    """报告里带 [S#] 引用的段落占比。"""
    paragraphs = [p for p in text.split("\n") if p.strip() and not p.startswith("#")]
    if not paragraphs:
        return 0.0
    cited = sum(1 for p in paragraphs if "[S" in p)
    return round(cited / len(paragraphs), 4)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
    return round(ordered[idx], 3)
