"""LLM 抽象层：Runtime 只依赖这里的协议，不依赖任何厂商 SDK。"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..types import Message, ToolCall, Usage


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str = "stop"
    raw: dict[str, Any] | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "usage": self.usage.to_dict(),
            "model": self.model,
            "finish_reason": self.finish_reason,
        }


class LLMProvider(ABC):
    """所有 provider 的统一契约：对话 + 向量化。"""

    name: str = "base"

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(f"{self.name} 未实现 embed")


def parse_arguments(raw: Any) -> dict:
    """模型返回的 arguments 可能是 dict 字符串、JSON 字符串或空。"""
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (TypeError, ValueError):
        return {}
