"""记忆候选抽取。

当前实现是"规则 + 可替换接口"：
- 规则抽取保证离线可测、可解释，适合高确定性的用户画像字段（姓名/偏好/邮箱/项目）；
- ``MemoryExtractor`` 是协议，生产可换成 LLM 抽取（同样的输出结构），不影响下游确认与召回。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass
class MemoryCandidate:
    mkey: str
    value: str
    mtype: str = "fact"
    confidence: float = 0.8


class MemoryExtractor(Protocol):
    def extract(self, text: str) -> list[MemoryCandidate]: ...


_PATTERNS: list[tuple[re.Pattern, str, str, float]] = [
    (re.compile(r"我叫([\u4e00-\u9fffA-Za-z]{2,10})"), "name", "profile", 0.95),
    (re.compile(r"我的(邮箱|电话|手机|公司|职位|城市|学校|专业)是?\s*([^\s，。；;]{2,40})"), "profile.{0}", "profile", 0.9),
    (re.compile(r"我(?:更)?(?:喜欢|偏好|习惯)([^，。；;\n]{2,40})"), "preference", "preference", 0.75),
    (re.compile(r"(?:请)?记住[:：,，]?\s*([^。\n]{2,80})"), "note", "fact", 0.8),
]


class RuleMemoryExtractor:
    def extract(self, text: str) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        for pattern, key, mtype, confidence in _PATTERNS:
            for match in pattern.finditer(text):
                if "{0}" in key:
                    field = match.group(1)
                    value = match.group(2)
                    candidates.append(MemoryCandidate(mkey=f"profile.{field}", value=value.strip(),
                                                      mtype=mtype, confidence=confidence))
                else:
                    value = match.group(1).strip()
                    if value:
                        candidates.append(MemoryCandidate(mkey=key, value=value, mtype=mtype,
                                                          confidence=confidence))
        # 同一 key 保留置信度最高的一条
        best: dict[str, MemoryCandidate] = {}
        for candidate in candidates:
            if candidate.mkey not in best or candidate.confidence > best[candidate.mkey].confidence:
                best[candidate.mkey] = candidate
        return list(best.values())
