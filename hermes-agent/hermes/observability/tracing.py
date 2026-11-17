"""链路追踪与用量核算。

- ``Tracer.span`` 用上下文管理器包裹一次 LLM 调用/工具调用，自动产出 span 事件与耗时；
- ``UsageMeter`` 汇总 token 与成本，支持按 run 预算熔断（超预算直接终止，避免失控花费）。
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from ..errors import BudgetExceeded
from ..ids import new_span_id
from ..types import Usage
from .events import Event, EventBus, EventType

# 每 1K token 的美元单价（可换成实时价目表）
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "glm-4-flash": (0.0001, 0.0001),
    "glm-4-air": (0.0005, 0.0005),
    "mock-1": (0.0, 0.0),
}


@dataclass
class Span:
    id: str
    name: str
    parent_id: str | None
    started_at: float
    attributes: dict[str, Any] = field(default_factory=dict)


class Tracer:
    def __init__(self, bus: EventBus, run_id: str):
        self.bus = bus
        self.run_id = run_id
        self._stack: list[Span] = []

    @property
    def current_span_id(self) -> str | None:
        return self._stack[-1].id if self._stack else None

    @asynccontextmanager
    async def span(self, name: str, **attributes: Any) -> AsyncIterator[Span]:
        parent = self.current_span_id
        span = Span(id=new_span_id(), name=name, parent_id=parent, started_at=time.perf_counter(), attributes=attributes)
        self._stack.append(span)
        await self.bus.emit(Event(
            run_id=self.run_id,
            type=EventType.SPAN_STARTED,
            span_id=span.id,
            parent_span_id=parent,
            payload={"name": name, **attributes},
        ))
        try:
            yield span
        finally:
            duration_ms = round((time.perf_counter() - span.started_at) * 1000, 2)
            self._stack.pop()
            await self.bus.emit(Event(
                run_id=self.run_id,
                type=EventType.SPAN_ENDED,
                span_id=span.id,
                parent_span_id=parent,
                payload={"name": name, "duration_ms": duration_ms, **span.attributes},
            ))


class UsageMeter:
    """单次 run 的用量与预算守卫。"""

    def __init__(self, *, max_tokens: int = 60_000, max_cost_usd: float = 0.5):
        self.total = Usage()
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.by_model: dict[str, Usage] = {}

    def record(self, usage: Usage, model: str) -> None:
        self.total.add(usage)
        self.by_model.setdefault(model, Usage()).add(usage)

    def check_budget(self) -> None:
        if self.total.total_tokens > self.max_tokens:
            raise BudgetExceeded(f"run token 预算超限: {self.total.total_tokens} > {self.max_tokens}")
        if self.total.cost_usd > self.max_cost_usd:
            raise BudgetExceeded(f"run 成本预算超限: {self.total.cost_usd:.4f} > {self.max_cost_usd}")

    def snapshot(self) -> dict:
        return {"total": self.total.to_dict(), "by_model": {k: v.to_dict() for k, v in self.by_model.items()}}


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_price, out_price = PRICE_TABLE.get(model, (0.0, 0.0))
    return (prompt_tokens / 1000) * in_price + (completion_tokens / 1000) * out_price
