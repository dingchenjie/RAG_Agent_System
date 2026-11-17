"""评测执行器。

同一套 Runtime 代码，既可以跑交互请求，也可以跑离线评测：
评测只是"批量提交 + 从事件流里取事实"，不存在两套逻辑。

任务可以声明 ``mode``：
- ``single``（默认）：直接调用 Runtime；
- ``orchestrate``：走主 Agent 的规划-委派-汇总链路，从父子事件流合并事实。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..observability.events import EventType
from ..runtime.cancel import CancelToken
from ..runtime.runtime import AgentRuntime
from .faithfulness import citation_support_rate
from .metrics import EvalReport, TaskResult


class EvalRunner:
    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        event_store: Any,
        dataset_path: str | Path,
        orchestrator: Any | None = None,
    ):
        self.runtime = runtime
        self.event_store = event_store
        self.dataset_path = Path(dataset_path)
        self.orchestrator = orchestrator

    def load_tasks(self) -> list[dict]:
        tasks = []
        for line in self.dataset_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
        return tasks

    async def run(self, *, limit: int | None = None) -> EvalReport:
        tasks = self.load_tasks()[: limit or None]
        report = EvalReport(meta=self._describe(tasks))
        for task in tasks:
            report.results.append(await self.run_task(task))
        return report

    async def run_task(self, task: dict) -> TaskResult:
        started = time.perf_counter()
        mode = task.get("mode", "single")
        try:
            if mode == "orchestrate":
                return await self._run_orchestrated(task, started)
            return await self._run_single(task, started)
        except Exception as exc:  # noqa: BLE001 单条用例失败不应中断整批评测
            return TaskResult(
                task_id=task.get("id", "unknown"), category=task.get("category", "general"),
                mode=mode, ok=False, answer="", expected_contains=task.get("expected_contains", []),
                expected_tools=task.get("expected_tools", []), called_tools=[],
                latency_ms=(time.perf_counter() - started) * 1000, steps=0, tokens=0,
                cost_usd=0.0, error=str(exc),
            )

    # ------------------------------------------------------------------ 单 Agent
    async def _run_single(self, task: dict, started: float) -> TaskResult:
        result = await self.runtime.run(
            user_input=task["input"],
            tenant_id=task.get("tenant_id", "eval"),
            user_id=task.get("user_id", "eval-user"),
            scopes=frozenset(task.get("scopes", ["tool:read"])),
            cancel=CancelToken(),
        )
        events = await asyncio.to_thread(self.event_store.history, result.run_id, 0, 2000)
        usage = result.usage or {}
        return self._score(
            task, mode="single", started=started, state=result.state,
            answer=result.answer or "", events=events, steps=result.steps,
            tokens=int(usage.get("total_tokens", 0)),
            cost_usd=float(usage.get("cost_usd", 0.0)),
            citations=len(result.citations), error=result.error,
        )

    # ------------------------------------------------------------------ 多 Agent
    async def _run_orchestrated(self, task: dict, started: float) -> TaskResult:
        if self.orchestrator is None:
            raise RuntimeError("orchestrate 任务需要注入 orchestrator")
        parent_run_id = f"eval_orch_{task['id']}"
        result = await self.orchestrator.run(
            goal=task["input"],
            tenant_id=task.get("tenant_id", "eval"),
            user_id=task.get("user_id", "eval-user"),
            scopes=frozenset(task.get("scopes", ["tool:read"])),
            cancel=CancelToken(),
            parent_run_id=parent_run_id,
        )
        # 父轨迹记规划/派发，子 Agent 的工具调用与检索各自在自己的 run 下，需要合并
        events = await asyncio.to_thread(self.event_store.history, parent_run_id, 0, 2000)
        for worker in result.workers:
            if worker.run_id:
                events += await asyncio.to_thread(self.event_store.history, worker.run_id, 0, 2000)
        events.sort(key=lambda e: e.ts)
        usage = result.usage or {}
        subtasks_ok = sum(1 for w in result.workers if w.ok)
        return self._score(
            task, mode="orchestrate", started=started, state=result.state,
            answer=result.answer or "", events=events,
            steps=sum(w.steps for w in result.workers),
            tokens=int(usage.get("total_tokens", 0)),
            cost_usd=float(usage.get("cost_usd", 0.0)),
            citations=0, error=None,
            subtasks=len(result.workers), subtasks_ok=subtasks_ok,
        )

    # ------------------------------------------------------------------ 打分
    def _score(
        self,
        task: dict,
        *,
        mode: str,
        started: float,
        state: str,
        answer: str,
        events: list,
        steps: int,
        tokens: int,
        cost_usd: float,
        citations: int,
        error: str | None,
        subtasks: int = 0,
        subtasks_ok: int = 0,
    ) -> TaskResult:
        called = [
            e.payload.get("tool", "")
            for e in events
            if e.type == EventType.TOOL_CALL_STARTED
        ]
        expected_tools = task.get("expected_tools", [])
        expected_contains = task.get("expected_contains", [])
        ok = state == "succeeded" and all(kw in answer for kw in expected_contains)
        if expected_tools:
            ok = ok and set(expected_tools).issubset(set(called))
        if "expected_subtasks" in task:
            ok = ok and subtasks == int(task["expected_subtasks"]) and subtasks_ok == subtasks

        contexts: list[str] = []
        for event in events:
            if event.type == EventType.KNOWLEDGE_RETRIEVED and event.payload.get("contexts"):
                contexts = list(event.payload["contexts"])
        faithfulness = citation_support_rate(answer, contexts) if contexts else None
        return TaskResult(
            task_id=task["id"],
            category=task.get("category", "general"),
            mode=mode,
            ok=ok,
            answer=answer,
            expected_contains=expected_contains,
            expected_tools=expected_tools,
            called_tools=called,
            latency_ms=(time.perf_counter() - started) * 1000,
            steps=steps,
            tokens=tokens,
            cost_usd=cost_usd,
            citations=citations,
            faithfulness=faithfulness,
            error=error,
            subtasks=subtasks,
            subtasks_ok=subtasks_ok,
        )

    def _describe(self, tasks: list[dict]) -> dict:
        settings = getattr(self.runtime, "settings", None)
        prompt = getattr(self.runtime, "system_prompt", "") or ""
        return {
            "dataset": self.dataset_path.name,
            "dataset_size": len(tasks),
            "modes": sorted({t.get("mode", "single") for t in tasks}),
            "llm_provider": getattr(settings, "llm_provider", "unknown"),
            "llm_model": getattr(settings, "llm_model", "unknown"),
            "embedding": getattr(settings, "embedding_backend", "unknown"),
            "rerank": getattr(settings, "rerank_backend", "unknown"),
            "prompt_version": hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8],
        }
