"""多 Agent：规划拆解、并行委派、黑板汇总、单点失败隔离。"""
from __future__ import annotations

import asyncio

from hermes.agents.orchestrator import RulePlanner
from hermes.api.app import build_state
from hermes.llm.mock import MockLLM
from hermes.observability.events import EventType
from tests.conftest import ingest_seed_docs


def test_rule_planner_splits_by_capability():
    planner = RulePlanner()
    plan = planner.plan("根据知识库说明 LoRA 的秩 r，并计算 32/4 的结果",
                        {"kb_search", "memory_recall", "calculator"})
    kinds = {s.kind for s in plan.subtasks}
    assert kinds == {"knowledge", "calc"}
    assert all(s.tools for s in plan.subtasks), "每个子任务必须绑定最小工具集"


def test_orchestrator_end_to_end(settings):
    state = build_state(settings, MockLLM())
    ingest_seed_docs(state)

    result = asyncio.run(state.orchestrator.run(
        goal="根据知识库说明 LoRA 的秩 r，并计算 32/4 的结果",
        parent_run_id="orch-test",
    ))
    assert result.state == "succeeded"
    assert len(result.workers) == 2
    assert all(w.ok for w in result.workers)
    assert set(result.blackboard) == {"finding.s1", "finding.s2"}
    assert result.answer

    types = [e.type for e in state.event_store.history("orch-test")]
    assert EventType.AGENT_PLANNED in types
    assert EventType.AGENT_DELEGATED in types
    assert EventType.AGENT_COMPLETED in types
    assert EventType.BLACKBOARD_UPDATED in types


def test_worker_failure_is_isolated(settings, monkeypatch):
    state = build_state(settings, MockLLM())
    ingest_seed_docs(state)

    original = state.runtime.with_tools

    class CrashingWorker:
        async def run(self, **kwargs):
            raise RuntimeError("子 Agent 崩了")

    def fake_with_tools(names, **kwargs):
        if "calculator" in list(names):
            return CrashingWorker()
        return original(names, **kwargs)

    monkeypatch.setattr(state.runtime, "with_tools", fake_with_tools)
    result = asyncio.run(state.orchestrator.run(
        goal="根据知识库说明 LoRA 的秩 r，并计算 32/4 的结果", parent_run_id="orch-fail"))

    # 一个子任务崩溃不影响整体：整体仍成功，失败被显式记录
    assert result.state == "succeeded"
    status = {w.subtask_id: w.ok for w in result.workers}
    assert status == {"s1": True, "s2": False}
    assert any("崩了" in (w.error or "") for w in result.workers)
