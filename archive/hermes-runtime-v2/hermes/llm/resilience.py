"""LLM 调用的可靠性包装：超时、指数退避重试、熔断。

这三件事是 JD 里"处理断线、取消、重复请求、进程重启"的最小落点：
- 超时：``asyncio.wait_for``，避免单次调用挂死整个 run；
- 重试：只对 ``retryable`` 错误退避重试，非重试错误立即上抛；
- 熔断：连续失败达阈值后直接快速失败，保护上游配额与尾延迟。
"""
from __future__ import annotations

import asyncio
import random
import time

from ..errors import LLMError, LLMTimeout
from ..observability.events import Event, EventBus, EventType
from ..types import Message
from .base import LLMProvider, LLMResponse


class CircuitBreaker:
    def __init__(self, threshold: int = 5, reset_s: float = 30.0):
        self.threshold = threshold
        self.reset_s = reset_s
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at >= self.reset_s:
            # 半开：允许一次试探
            self.opened_at = None
            self.failures = 0
            return False
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic()


class ResilientLLM(LLMProvider):
    def __init__(
        self,
        inner: LLMProvider,
        *,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        breaker: CircuitBreaker | None = None,
        bus: EventBus | None = None,
        run_id: str = "",
        backoff_base_s: float = 0.4,
        backoff_cap_s: float = 8.0,
    ):
        self.inner = inner
        self.name = inner.name
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.breaker = breaker or CircuitBreaker()
        self.bus = bus
        self.run_id = run_id
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if self.breaker.is_open:
            raise LLMError("上游模型连续失败，熔断中", code="circuit_open", retryable=False)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self.inner.chat(messages, tools, temperature=temperature, max_tokens=max_tokens),
                    timeout=self.timeout_s,
                )
                self.breaker.record_success()
                return response
            except asyncio.TimeoutError as exc:
                last_error = LLMTimeout(f"模型调用超时（>{self.timeout_s}s）")
            except LLMError as exc:
                last_error = exc
                if not exc.retryable:
                    self.breaker.record_failure()
                    raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # 未知异常按可重试处理一次
                last_error = LLMError(f"模型调用异常: {exc}", retryable=True)

            self.breaker.record_failure()
            if attempt >= self.max_retries:
                break
            delay = min(self.backoff_cap_s, self.backoff_base_s * (2 ** attempt)) * (0.5 + random.random() * 0.5)
            if self.bus is not None:
                await self.bus.emit(Event(
                    run_id=self.run_id,
                    type=EventType.LLM_RETRY,
                    payload={"attempt": attempt + 1, "delay_s": round(delay, 3), "error": str(last_error)},
                ))
            await asyncio.sleep(delay)

        assert last_error is not None
        raise last_error

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self.inner.embed(texts)
