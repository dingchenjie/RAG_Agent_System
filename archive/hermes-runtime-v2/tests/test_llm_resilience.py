"""模型调用可靠性：超时、退避重试、非重试错误、熔断、重试事件。"""
from __future__ import annotations

import asyncio

import pytest

from hermes.errors import LLMError, LLMTimeout
from hermes.llm.base import LLMProvider, LLMResponse
from hermes.llm.mock import FailingLLM
from hermes.llm.resilience import CircuitBreaker, ResilientLLM
from hermes.observability.events import EventBus, EventStore, EventType
from hermes.storage.db import Database
from hermes.types import Message


class SlowLLM(LLMProvider):
    name = "slow"

    async def chat(self, messages, tools=None, *, temperature=0.1, max_tokens=None) -> LLMResponse:
        await asyncio.sleep(1.0)
        return LLMResponse(content="late")


def _fast(inner: LLMProvider, **kwargs) -> ResilientLLM:
    kwargs.setdefault("backoff_base_s", 0.01)
    kwargs.setdefault("backoff_cap_s", 0.02)
    return ResilientLLM(inner, **kwargs)


def test_retry_then_success():
    inner = FailingLLM(fail_times=2)
    llm = _fast(inner, timeout_s=2, max_retries=3)
    response = asyncio.run(llm.chat([Message(role="user", content="hi")]))
    assert response.content == "ok"
    assert inner.calls == 3


def test_non_retryable_error_fails_fast():
    inner = FailingLLM(fail_times=1, retryable=False)
    llm = _fast(inner, timeout_s=2, max_retries=3)
    with pytest.raises(LLMError) as info:
        asyncio.run(llm.chat([Message(role="user", content="hi")]))
    assert info.value.retryable is False
    assert inner.calls == 1  # 非重试错误不消耗重试次数


def test_timeout_is_retryable_and_bounded():
    llm = _fast(SlowLLM(), timeout_s=0.05, max_retries=1)
    with pytest.raises(LLMTimeout):
        asyncio.run(llm.chat([Message(role="user", content="hi")]))


def test_circuit_breaker_opens_after_threshold():
    inner = FailingLLM(fail_times=100)
    breaker = CircuitBreaker(threshold=2, reset_s=60)
    llm = _fast(inner, timeout_s=2, max_retries=0, breaker=breaker)

    for _ in range(2):
        with pytest.raises(LLMError):
            asyncio.run(llm.chat([Message(role="user", content="hi")]))
    calls_before = inner.calls

    with pytest.raises(LLMError) as info:
        asyncio.run(llm.chat([Message(role="user", content="hi")]))
    assert info.value.code == "circuit_open"
    assert inner.calls == calls_before  # 熔断期间不再打上游


def test_retry_events_are_recorded(tmp_path):
    db = Database(str(tmp_path / "evt.db"))
    bus = EventBus(EventStore(db))
    llm = _fast(FailingLLM(fail_times=1), timeout_s=2, max_retries=2, bus=bus, run_id="run-x")
    asyncio.run(llm.chat([Message(role="user", content="hi")]))

    events = bus.store.history("run-x")
    retries = [e for e in events if e.type == EventType.LLM_RETRY]
    assert len(retries) == 1
    assert retries[0].payload["attempt"] == 1
