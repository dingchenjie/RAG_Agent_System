"""主 Agent：规划 -> 并行委派子 Agent -> 汇总。

与"一个 Agent 硬扛所有工具"相比，这里的收益是可解释的：
- 每个子 Agent 只拿到完成子任务所需的工具与上下文（更少的 token、更小的出错面）；
- 子任务并行执行，缩短端到端时延；
- 单个子任务失败不会拖垮整体，最终答案会显式标注降级来源。
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMProvider
from ..observability.events import Event, EventBus, EventType
from ..observability.tracing import Usage, UsageMeter
from ..runtime.cancel import CancelToken
from ..runtime.runtime import AgentRuntime, RunResult
from ..types import Message
from .blackboard import Blackboard

KB_RE = re.compile(r"论文|文档|资料|知识库|检索|LoRA|RAG|原理|实验|政策|流程")
MEMORY_RE = re.compile(r"记得|还记得|我叫|我的.*(偏好|习惯|名字|邮箱|城市)|之前说过|上次")
CALC_RE = re.compile(r"\d+\s*[+\-*/^%]\s*\d+|计算|算一下|等于多少")


@dataclass
class Subtask:
    id: str
    goal: str
    tools: tuple[str, ...]
    kind: str = "general"

    def to_dict(self) -> dict:
        return {"id": self.id, "goal": self.goal, "tools": list(self.tools), "kind": self.kind}


@dataclass
class Plan:
    goal: str
    subtasks: list[Subtask]
    rationale: str = ""

    def to_dict(self) -> dict:
        return {"goal": self.goal, "rationale": self.rationale,
                "subtasks": [s.to_dict() for s in self.subtasks]}


@dataclass
class WorkerResult:
    subtask_id: str
    agent: str
    ok: bool
    answer: str = ""
    error: str | None = None
    run_id: str | None = None
    steps: int = 0
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"subtask_id": self.subtask_id, "agent": self.agent, "ok": self.ok,
                "answer": self.answer, "error": self.error, "run_id": self.run_id,
                "steps": self.steps, "usage": self.usage}


@dataclass
class OrchestratorResult:
    state: str
    answer: str
    plan: Plan
    workers: list[WorkerResult]
    usage: dict = field(default_factory=dict)
    blackboard: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"state": self.state, "answer": self.answer, "plan": self.plan.to_dict(),
                "workers": [w.to_dict() for w in self.workers], "usage": self.usage,
                "blackboard": self.blackboard}


class RulePlanner:
    """确定性规划器：先保证可测、可控，再谈让模型自由规划。

    生产上可以替换成 LLM 规划器（同样的 ``plan`` 契约），
    但规则规划在可解释性、延迟与稳定性上更适合企业高频任务。
    """

    def plan(self, goal: str, available: set[str]) -> Plan:
        subtasks: list[Subtask] = []
        if KB_RE.search(goal) and "kb_search" in available:
            subtasks.append(Subtask(id="s1", goal=f"检索并总结与问题相关的资料：{goal}",
                                    tools=("kb_search",), kind="knowledge"))
        if MEMORY_RE.search(goal) and "memory_recall" in available:
            subtasks.append(Subtask(id=f"s{len(subtasks)+1}",
                                    goal=f"召回与该问题相关的用户长期记忆：{goal}",
                                    tools=("memory_recall",), kind="memory"))
        if CALC_RE.search(goal) and "calculator" in available:
            subtasks.append(Subtask(id=f"s{len(subtasks)+1}",
                                    goal=f"完成问题中的数值计算：{goal}",
                                    tools=("calculator",), kind="calc"))
        if not subtasks:
            default_tools = tuple(t for t in ("kb_search", "memory_recall") if t in available)
            subtasks.append(Subtask(id="s1", goal=goal, tools=default_tools, kind="general"))
        return Plan(goal=goal, subtasks=subtasks,
                    rationale="按能力域拆分为最小子任务，各自只暴露必要工具")


class Orchestrator:
    def __init__(self, *, runtime: AgentRuntime, llm: LLMProvider, bus: EventBus,
                 planner: RulePlanner | None = None, max_workers: int = 3):
        self.runtime = runtime
        self.llm = llm
        self.bus = bus
        self.planner = planner or RulePlanner()
        self.max_workers = max_workers

    async def run(
        self,
        *,
        goal: str,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        scopes: frozenset[str] = frozenset({"tool:read"}),
        cancel: CancelToken | None = None,
        parent_run_id: str | None = None,
    ) -> OrchestratorResult:
        available = {spec.name for spec in getattr(self.runtime.registry, "all_specs", lambda: [])()}
        plan = self.planner.plan(goal, available)
        run_id = parent_run_id or f"orch_{int(asyncio.get_running_loop().time()*1000)}"

        await self.bus.emit(Event(run_id=run_id, type=EventType.AGENT_PLANNED, payload=plan.to_dict()))
        blackboard = Blackboard()

        results = await self._dispatch(plan, blackboard, tenant_id, user_id, scopes,
                                       cancel or CancelToken(), run_id)

        answer, usage = await self._synthesize(goal, plan, results, run_id)
        state = "succeeded" if any(r.ok for r in results) else "failed"
        if state == "failed" and results:
            answer = "任务未完成：" + "；".join(f"{r.subtask_id}: {r.error}" for r in results)

        await self.bus.emit(Event(
            run_id=run_id, type=EventType.RUN_COMPLETED if state == "succeeded" else EventType.RUN_FAILED,
            payload={"orchestrator": True, "state": state, "workers": len(results)},
        ))
        return OrchestratorResult(state=state, answer=answer, plan=plan, workers=results,
                                  usage=usage, blackboard=blackboard.snapshot())

    async def _dispatch(self, plan: Plan, blackboard: Blackboard, tenant_id: str, user_id: str,
                        scopes: frozenset[str], cancel: CancelToken, run_id: str) -> list[WorkerResult]:
        semaphore = asyncio.Semaphore(self.max_workers)

        async def run_one(subtask: Subtask) -> WorkerResult:
            async with semaphore:
                agent_name = f"worker.{subtask.kind}"
                await self.bus.emit(Event(run_id=run_id, type=EventType.AGENT_DELEGATED,
                                          payload={"subtask": subtask.to_dict(), "agent": agent_name}))
                worker = self.runtime.with_tools(
                    list(subtask.tools) or [s.name for s in self.runtime.registry.all_specs()],
                    agent_name=agent_name,
                    system_prompt=(
                        f"你是子 Agent「{subtask.kind}」，只负责一个子任务：{subtask.goal}\n"
                        "只使用分配给你的工具；完成后用简洁结论回答，不要展开无关内容。"
                    ),
                    max_steps=4,
                )
                try:
                    result: RunResult = await worker.run(
                        user_input=subtask.goal, tenant_id=tenant_id, user_id=user_id,
                        scopes=scopes, cancel=cancel,
                    )
                    ok = result.state == "succeeded"
                    answer = result.answer or ""
                    blackboard.put(f"finding.{subtask.id}", answer, owner=agent_name)
                    await self.bus.emit(Event(
                        run_id=run_id,
                        type=EventType.AGENT_COMPLETED if ok else EventType.AGENT_FAILED,
                        payload={"subtask_id": subtask.id, "agent": agent_name, "ok": ok,
                                 "answer": answer[:400], "usage": result.usage},
                    ))
                    await self.bus.emit(Event(run_id=run_id, type=EventType.BLACKBOARD_UPDATED,
                                              payload={"key": f"finding.{subtask.id}", "owner": agent_name}))
                    return WorkerResult(subtask_id=subtask.id, agent=agent_name, ok=ok, answer=answer,
                                        run_id=result.run_id, steps=result.steps, usage=result.usage,
                                        error=result.error)
                except Exception as exc:  # noqa: BLE001
                    await self.bus.emit(Event(run_id=run_id, type=EventType.AGENT_FAILED,
                                              payload={"subtask_id": subtask.id, "error": str(exc)}))
                    return WorkerResult(subtask_id=subtask.id, agent=agent_name, ok=False,
                                        error=str(exc))

        return list(await asyncio.gather(*(run_one(s) for s in plan.subtasks)))

    async def _synthesize(self, goal: str, plan: Plan, results: list[WorkerResult],
                          run_id: str) -> tuple[str, dict]:
        meter = UsageMeter(max_tokens=self.runtime.settings.max_run_tokens,
                           max_cost_usd=self.runtime.settings.max_run_cost_usd)
        findings = "\n\n".join(
            f"[{r.subtask_id}/{'成功' if r.ok else '失败'}] {r.answer or r.error}"
            for r in results
        )
        messages = [
            Message(role="system", content="你是主编排 Agent。请基于子 Agent 的结论，"
                                           "用中文给出最终答复；标注不确定之处；保留来源编号。"),
            Message(role="user", content=f"用户问题：{goal}\n\n子 Agent 结论：\n{findings}"),
        ]
        try:
            response = await self.llm.chat(messages, None, temperature=0.1)
            meter.record(response.usage, response.model or "unknown")
            return (response.content or findings), meter.snapshot()["total"]
        except Exception:
            # 汇总失败时降级为拼接，保证有可用输出
            return findings, meter.snapshot()["total"]
