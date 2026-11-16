"""评测执行器。

同一套 Runtime 代码，既可以跑交互请求，也可以跑离线评测：
评测只是"批量提交 + 从事件流里取事实"，不存在两套逻辑。
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from ..observability.events import EventType
from ..runtime.cancel import CancelToken
from ..runtime.runtime import AgentRuntime
from .metrics import EvalReport, TaskResult


class EvalRunner:
    def __init__(self, *, runtime: AgentRuntime, event_store: Any, dataset_path: str | Path):
        self.runtime = runtime
        self.event_store = event_store
        self.dataset_path = Path(dataset_path)

    def load_tasks(self) -> list[dict]:
        tasks = []
        for line in self.dataset_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
        return tasks

    async def run(self, *, limit: int | None = None) -> EvalReport:
        report = EvalReport()
        for task in self.load_tasks()[: limit or None]:
            report.results.append(await self.run_task(task))
        return report

    async def run_task(self, task: dict) -> TaskResult:
        started = time.perf_counter()
        token = CancelToken()
        try:
            result = await self.runtime.run(
                user_input=task["input"],
                tenant_id=task.get("tenant_id", "eval"),
                user_id=task.get("user_id", "eval-user"),
                scopes=frozenset(task.get("scopes", ["tool:read"])),
                cancel=token,
            )
            answer = result.answer or ""
            events = await asyncio.to_thread(self.event_store.history, result.run_id, 0, 2000)
            called = [
                e.payload.get("tool", "")
                for e in events
                if e.type == EventType.TOOL_CALL_STARTED
            ]
            expected_tools = task.get("expected_tools", [])
            expected_contains = task.get("expected_contains", [])
            ok = result.state == "succeeded" and all(kw in answer for kw in expected_contains)
            if expected_tools:
                ok = ok and set(expected_tools).issubset(set(called))
            citations = len(result.citations)
            usage = result.usage or {}
            return TaskResult(
                task_id=task["id"],
                category=task.get("category", "general"),
                ok=ok,
                answer=answer,
                expected_contains=expected_contains,
                expected_tools=expected_tools,
                called_tools=called,
                latency_ms=(time.perf_counter() - started) * 1000,
                steps=result.steps,
                tokens=int(usage.get("total_tokens", 0)),
                cost_usd=float(usage.get("cost_usd", 0.0)),
                citations=citations,
                error=result.error,
            )
        except Exception as exc:  # noqa: BLE001 单条用例失败不应中断整批评测
            return TaskResult(
                task_id=task.get("id", "unknown"), category=task.get("category", "general"),
                ok=False, answer="", expected_contains=task.get("expected_contains", []),
                expected_tools=task.get("expected_tools", []), called_tools=[],
                latency_ms=(time.perf_counter() - started) * 1000, steps=0, tokens=0,
                cost_usd=0.0, error=str(exc),
            )
