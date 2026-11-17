"""记忆候选抽取。

两种实现共用同一个输出结构，下游确认/召回完全不感知差异：
- ``RuleMemoryExtractor``：规则抽取，离线可测、可解释，适合高确定性的画像字段；
- ``LLMMemoryExtractor``：一次 LLM 调用做结构化抽取，能覆盖规则表达不了的语义
  （例如"我们团队周四发版"），但必须在证据与置信度上设闸，失败时返回空而不是阻塞回答。

``propose_from_text_async`` 会优先走 ``extract_async``，规则实现没有异步版本时退化为线程池，
因此同步的 MemoryService 接口保持不变。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from ..llm.base import LLMProvider
from ..types import Message

logger = logging.getLogger(__name__)


@dataclass
class MemoryCandidate:
    mkey: str
    value: str
    mtype: str = "fact"
    confidence: float = 0.8


class MemoryExtractor(Protocol):
    def extract(self, text: str) -> list[MemoryCandidate]: ...


_PROFILE_FIELDS = {
    "邮箱": "email", "电话": "phone", "手机": "phone", "公司": "company",
    "职位": "title", "城市": "city", "学校": "school", "专业": "major",
}

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
                    # key 统一用英文（profile.email），与 LLM 抽取器保持同一套 schema
                    english = _PROFILE_FIELDS.get(field, field)
                    candidates.append(MemoryCandidate(mkey=f"profile.{english}", value=value.strip(),
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

    async def extract_async(self, text: str) -> list[MemoryCandidate]:
        return self.extract(text)


_ALLOWED_TYPES = {"profile", "preference", "fact", "note"}
_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


def parse_candidates(raw: str, *, min_confidence: float = 0.5,
                     max_candidates: int = 8) -> list[MemoryCandidate]:
    """解析并校验模型输出；任何不合法条目直接丢弃，不抛异常。"""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        payload: Any = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    best: dict[str, MemoryCandidate] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or item.get("mkey") or "").strip()
        value = str(item.get("value") or "").strip()
        if not _KEY_RE.match(key) or not value or len(value) > 200:
            continue
        try:
            confidence = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            continue
        if confidence < min_confidence:
            continue
        mtype = str(item.get("type") or item.get("mtype") or "fact").strip()
        if mtype not in _ALLOWED_TYPES:
            mtype = "fact"
        candidate = MemoryCandidate(mkey=key, value=value, mtype=mtype,
                                    confidence=min(1.0, max(0.0, confidence)))
        if key not in best or candidate.confidence > best[key].confidence:
            best[key] = candidate
    return list(best.values())[:max_candidates]


class LLMMemoryExtractor:
    """用一次结构化抽取调用把用户输入变成记忆候选。

    失败策略是"返回空"：记忆是增强能力，不能因为模型不听话就阻塞主回答。
    """

    _SYSTEM = (
        "你是长期记忆抽取器。只输出 JSON 数组，不要任何解释或 Markdown。\n"
        "从用户这句话里抽取值得长期记住的稳定信息：身份画像、偏好、习惯、长期项目/业务对象。\n"
        "规则：\n"
        "1. 只抽用户明确陈述的信息；寒暄、一次性任务指令、猜测、别人的信息都不要。\n"
        "2. key 用英文蛇形命名且语义稳定，例如 name、profile.email、preference.language、"
        "project.codename、customer.contact_channel。\n"
        "3. value 只保留最小事实值，不加解释、不编造。\n"
        "4. type 只能是 profile|preference|fact|note；confidence 是 0~1 的小数。\n"
        "5. 没有可抽取内容就输出 []。\n"
        "示例输入：我叫丁辰杰，做后端，偏好 Python，最近在负责 Apollo 项目。\n"
        "示例输出：[{\"key\":\"name\",\"value\":\"丁辰杰\",\"type\":\"profile\",\"confidence\":0.95},"
        "{\"key\":\"preference.language\",\"value\":\"Python\",\"type\":\"preference\",\"confidence\":0.9},"
        "{\"key\":\"project.codename\",\"value\":\"Apollo\",\"type\":\"fact\",\"confidence\":0.85}]"
    )

    def __init__(self, llm: LLMProvider, *, min_confidence: float = 0.5,
                 max_candidates: int = 8, max_input_chars: int = 2000):
        self.llm = llm
        self.min_confidence = min_confidence
        self.max_candidates = max_candidates
        self.max_input_chars = max_input_chars

    async def extract_async(self, text: str) -> list[MemoryCandidate]:
        if not text.strip():
            return []
        try:
            response = await self.llm.chat(
                [
                    Message(role="system", content=self._SYSTEM),
                    Message(role="user", content=text[: self.max_input_chars]),
                ],
                temperature=0.0,
                max_tokens=600,
            )
        except Exception as exc:  # noqa: BLE001 抽取失败不能影响主链路
            logger.warning("LLM 记忆抽取失败，按无候选处理: %s", exc)
            return []
        return parse_candidates(response.content or "", min_confidence=self.min_confidence,
                                max_candidates=self.max_candidates)

    def extract(self, text: str) -> list[MemoryCandidate]:
        raise RuntimeError("LLMMemoryExtractor 仅支持异步抽取，请调用 extract_async()")


def build_memory_extractor(backend: str, llm: LLMProvider, *,
                           min_confidence: float = 0.5) -> MemoryExtractor:
    backend = (backend or "rule").strip().lower()
    if backend == "rule":
        return RuleMemoryExtractor()
    if backend == "llm":
        return LLMMemoryExtractor(llm, min_confidence=min_confidence)
    raise ValueError(f"未知的记忆抽取后端: {backend}（可选 rule|llm）")
