"""编排失败隔离：用故障注入验证"单个子任务失败不拖垮整体"。

两个场景：
1. 部分失败：知识子任务成功、计算子任务超时 → 整体仍 succeeded，答案保留成功结论并显式标注失败；
2. 全部失败：所有子任务异常 → 整体 failed，答案给出"任务未完成"与逐项原因。

故障注入只替换 ``AgentRuntime.with_tools`` 返回的 worker，不改生产 Orchestrator 逻辑；
因此它验证的是真实的分派、汇总与事件链路，而不是另一套实现。
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..agents.orchestrator import Orchestrator
from ..observability.events import EventType
from ..runtime.runtime import AgentRuntime

PARTIAL_GOAL = "LoRA 的秩 r 通常设置为多少？另外计算 8/8 的结果"
ALL_FAIL_GOAL = "LoRA 的秩 r 通常设置为多少？另外计算 8/8 的结果"


class _FailingWorker:
    """替身 worker：被分派时直接抛异常，模拟子 Agent 超时/崩溃。"""

    def __init__(self, kind: str):
        self.kind = kind

    async def run(self, **kwargs: Any):
        raise TimeoutError(f"子 Agent「{self.kind}」超时（故障注入）")


class _FaultyRuntime:
    """只拦截 with_tools，其余属性透传给真实 runtime。"""

    def __init__(self, runtime: AgentRuntime, failing_kinds: set[str]):
        self._runtime = runtime
        self._failing_kinds = failing_kinds

    def __getattr__(self, name: str):
        return getattr(self._runtime, name)

    def with_tools(self, tools, **kwargs):
        kind = str(kwargs.get("agent_name", "")).removeprefix("worker.")
        if kind in self._failing_kinds:
            return _FailingWorker(kind)
        return self._runtime.with_tools(tools, **kwargs)


async def _run_scenario(state, *, name: str, goal: str, failing: set[str]) -> dict:
    run_id = f"eval_orch_failure_{name}"
    orchestrator = Orchestrator(
        runtime=_FaultyRuntime(state.runtime, failing),
        llm=state.runtime.llm,
        bus=state.bus,
    )
    result = await orchestrator.run(
        goal=goal, tenant_id="eval", user_id="eval-user",
        scopes=frozenset({"tool:read"}), parent_run_id=run_id,
    )
    events = await asyncio.to_thread(state.event_store.history, run_id, 0, 500)
    types = [event.type for event in events]
    failed_workers = [worker for worker in result.workers if not worker.ok]
    ok_workers = [worker for worker in result.workers if worker.ok]

    if name == "partial":
        passed = (
            result.state == "succeeded"
            and len(result.workers) == 2
            and len(ok_workers) == 1
            and len(failed_workers) == 1
            and "超时" in (failed_workers[0].error or "")
            and "失败" in result.answer
            and "超时" in result.answer
            and EventType.AGENT_FAILED in types
            and EventType.RUN_COMPLETED in types
            and EventType.RUN_FAILED not in types
        )
    else:
        passed = (
            result.state == "failed"
            and len(failed_workers) == len(result.workers) == 2
            and result.answer.startswith("任务未完成")
            and EventType.RUN_FAILED in types
            and EventType.RUN_COMPLETED not in types
        )

    return {
        "name": name,
        "passed": passed,
        "state": result.state,
        "workers": len(result.workers),
        "ok_workers": len(ok_workers),
        "answer": result.answer[:200],
        "answer_has_failure": "失败" in result.answer,
        "answer_has_timeout": "超时" in result.answer,
        "answer_starts_incomplete": result.answer.startswith("任务未完成"),
        "events": len(events),
    }


async def evaluate_orchestration_failures(state) -> dict:
    """跑两条失败路径场景，返回逐项结果与通过数。"""
    details = [
        await _run_scenario(state, name="partial", goal=PARTIAL_GOAL, failing={"calc"}),
        await _run_scenario(state, name="all_fail", goal=ALL_FAIL_GOAL,
                            failing={"knowledge", "calc"}),
    ]
    passed = sum(1 for item in details if item["passed"])
    return {"scenarios": len(details), "passed": passed, "details": details}
