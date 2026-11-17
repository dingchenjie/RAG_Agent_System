"""Runtime：工具循环、取消、确认恢复、幂等、断点重载。"""
from __future__ import annotations

import asyncio
import time

from hermes.api.app import build_state
from hermes.llm.base import LLMProvider, LLMResponse
from hermes.llm.mock import MockLLM
from hermes.runtime.cancel import CancelToken
from hermes.types import ToolCall
from tests.conftest import ingest_seed_docs


class SlowLLM(LLMProvider):
    name = "slow"

    async def chat(self, messages, tools=None, *, temperature=0.1, max_tokens=None) -> LLMResponse:
        await asyncio.sleep(5)
        return LLMResponse(content="迟到的回答")


def test_tool_loop_with_knowledge(settings):
    state = build_state(settings, MockLLM())
    ingest_seed_docs(state)

    result = asyncio.run(state.runtime.run(user_input="LoRA 的秩 r 通常设置为多少？"))
    assert result.state == "succeeded"
    assert "8" in (result.answer or "")
    assert result.citations, "回答应带检索引用"
    assert result.steps >= 1

    events = state.event_store.history(result.run_id)
    types = [e.type.value for e in events]
    assert "tool.started" in types
    assert "context.built" in types
    assert "run.completed" in types


def test_cancellation_interrupts_llm(settings):
    state = build_state(settings, SlowLLM())

    async def scenario():
        token = CancelToken()
        task = asyncio.create_task(state.runtime.run(user_input="慢问题", cancel=token))
        await asyncio.sleep(0.2)
        started = time.perf_counter()
        token.cancel()
        result = await task
        return result, time.perf_counter() - started

    result, elapsed = asyncio.run(scenario())
    assert result.state == "cancelled"
    assert elapsed < 2.0
    assert state.runs.get_run(result.run_id)["state"] == "cancelled"


def test_pause_then_confirm(settings):
    scripted = MockLLM(script=[
        LLMResponse(content="", finish_reason="tool_calls",
                    tool_calls=[ToolCall(id="call_1", name="memory_write",
                                         arguments={"key": "profile.city", "value": "南京"})]),
    ])
    state = build_state(settings, scripted)
    scopes = frozenset({"tool:read", "memory:write"})

    paused = asyncio.run(state.runtime.run(user_input="记住我住在南京", scopes=scopes))
    assert paused.state == "paused"
    assert paused.pending and paused.pending["name"] == "memory_write"
    assert state.memory.list_memories(tenant_id="default", user_id="anonymous", status="active") == []

    resumed = asyncio.run(state.runtime.resume(paused.run_id, confirmed=True))
    assert resumed.state == "succeeded"
    active = state.memory.list_memories(tenant_id="default", user_id="anonymous", status="active")
    assert any(r.value == "南京" for r in active)


def test_idempotency_returns_same_run(settings):
    state = build_state(settings, MockLLM())

    async def scenario():
        first = await state.runtime.run(user_input="你好", idempotency_key="k-1")
        second = await state.runtime.run(user_input="你好", idempotency_key="k-1")
        return first, second

    first, second = asyncio.run(scenario())
    assert first.run_id == second.run_id


def test_resume_after_process_restart(settings):
    scripted = MockLLM(script=[
        LLMResponse(content="", finish_reason="tool_calls",
                    tool_calls=[ToolCall(id="call_1", name="memory_write",
                                         arguments={"key": "profile.phone", "value": "13800000000"})]),
    ])
    state = build_state(settings, scripted)
    paused = asyncio.run(state.runtime.run(
        user_input="记住我的电话", scopes=frozenset({"tool:read", "memory:write"})))
    assert paused.state == "paused"

    # 模拟进程重启：同一个 DB，全新的运行时对象
    restarted = build_state(settings, MockLLM())
    resumed = asyncio.run(restarted.runtime.resume(paused.run_id, confirmed=True))
    assert resumed.state == "succeeded"
    active = restarted.memory.list_memories(tenant_id="default", user_id="anonymous", status="active")
    assert any(r.mkey == "profile.phone" for r in active)


def test_context_report_is_recorded(settings):
    state = build_state(settings, MockLLM())
    ingest_seed_docs(state)
    result = asyncio.run(state.runtime.run(user_input="LoRA 的 alpha 一般怎么设置？"))
    assert result.context["final_tokens"] <= result.context["budget"]
    assert result.context["budget"] > 0


def test_follow_up_uses_previous_question_for_retrieval(settings):
    """同一会话里的「那上海呢？」应带上上一轮主题再检索。"""
    state = build_state(settings, MockLLM())
    ingest_seed_docs(state)
    scopes = frozenset({"tool:read", "kb:group:finance"})

    async def scenario():
        first = await state.runtime.run(
            user_input="差旅住宿一线城市的上限是多少？", scopes=scopes)
        session_id = state.runs.get_run(first.run_id)["session_id"]
        return await state.runtime.run(
            user_input="那上海呢？", session_id=session_id, scopes=scopes)

    second = asyncio.run(scenario())
    events = [e for e in state.event_store.history(second.run_id)
              if e.type.value == "knowledge.retrieved"]
    assert events
    payload = events[0].payload
    assert payload["rewritten"] is True
    assert "一线城市" in payload["query"]
    assert payload["count"] >= 1
