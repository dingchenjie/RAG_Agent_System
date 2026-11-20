"""运行引擎：装配依赖 → 建图 → 执行 → 落库与落盘。"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .graph import Runtime, build_graph
from .llm import build_llm
from .store import Store
from .tools import ToolRegistry, build_registry
from .trace import Tracer


@dataclass
class Engine:
    settings: Settings
    store: Store
    registry: ToolRegistry
    llm: Any
    run_id: str
    tracer: Tracer


def create_engine(settings: Settings, *, run_id: str | None = None, llm: Any = None,
                  store: Store | None = None) -> Engine:
    store = store or Store(settings.db_path)
    run_id = run_id or uuid.uuid4().hex[:12]
    tracer = Tracer(store, run_id, settings)
    return Engine(settings=settings, store=store, registry=build_registry(settings),
                  llm=llm or build_llm(settings), run_id=run_id, tracer=tracer)


async def run_question(engine: Engine, question: str, *,
                       require_approval: bool = False) -> dict[str, Any]:
    s = engine.settings
    started = time.time()
    engine.store.create_run(engine.run_id, question)
    engine.tracer.emit("run.started", question=question, model=getattr(engine.llm, "model", "mock"),
                       prompt_version=s.prompt_version, approval=require_approval)
    runtime = Runtime(settings=s, llm=engine.llm, registry=engine.registry,
                      store=engine.store, tracer=engine.tracer, run_id=engine.run_id)
    graph = build_graph(runtime, require_approval=require_approval)
    config = {"recursion_limit": 40, "configurable": {"thread_id": engine.run_id}}
    initial = {"question": question, "sections": [], "findings": [], "evidence": [],
               "tools_used": [], "errors": [], "pages_used": 0, "revisions": 0}
    status, report, error = "succeeded", "", ""
    try:
        await graph.ainvoke(initial, config)
        snapshot = await graph.aget_state(config)
        if snapshot.next:
            status = "paused"
            report = snapshot.values.get("draft", "")
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        engine.tracer.emit("run.error", error=error[:400])

    snapshot = await graph.aget_state(config)
    values = snapshot.values
    if not report:
        report = values.get("report") or values.get("draft") or ""
    meta = {
        "model": getattr(engine.llm, "model", "mock"),
        "prompt_version": s.prompt_version,
        "mode": values.get("mode", ""),
        "tools_used": sorted(set(values.get("tools_used", []))),
        # 并行分支各自算增量会重复计数，按 ToolContext 的真实计数取值
        "pages_used": runtime.ctx.pages_used,
        "loop_detected": bool(values.get("loop_detected")),
        "citations": sorted({e["source_id"] for e in values.get("evidence", [])}),
        "tokens": {"prompt": getattr(engine.llm, "prompt_tokens", 0),
                   "completion": getattr(engine.llm, "completion_tokens", 0)},
        "cost_usd": round(float(getattr(engine.llm, "cost_usd", 0.0)), 6),
        "elapsed_s": round(time.time() - started, 2),
        "error": error,
    }
    meta["budget_exceeded"] = meta["cost_usd"] > s.max_run_cost_usd

    if status == "succeeded":
        os.makedirs(s.output_dir, exist_ok=True)
        path = os.path.join(s.output_dir, f"{engine.run_id}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report)
        meta["report_path"] = path

    engine.store.finish_run(engine.run_id, status, report, meta)
    engine.tracer.emit("run.finished", status=status, elapsed_s=meta["elapsed_s"],
                       tokens=meta["tokens"], cost_usd=meta["cost_usd"], error=error)
    engine.tracer.flush()
    return {"run_id": engine.run_id, "status": status, "report": report, "meta": meta}


async def resume_run(engine: Engine) -> dict[str, Any]:
    """人工审批后继续：从 checkpoint 恢复，只补跑 publish。"""
    runtime = Runtime(settings=engine.settings, llm=engine.llm, registry=engine.registry,
                      store=engine.store, tracer=engine.tracer, run_id=engine.run_id)
    graph = build_graph(runtime, require_approval=True)
    config = {"recursion_limit": 40, "configurable": {"thread_id": engine.run_id}}
    await graph.ainvoke(None, config)
    snapshot = await graph.aget_state(config)
    report = snapshot.values.get("report", "")
    row = engine.store.get_run(engine.run_id) or {}
    meta = row.get("meta") or {}
    engine.store.finish_run(engine.run_id, "succeeded", report, meta)
    engine.tracer.emit("run.finished", status="succeeded", approved=True)
    return {"run_id": engine.run_id, "status": "succeeded", "report": report}


def run_sync(settings: Settings, question: str, **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(run_question(create_engine(settings), question, **kwargs))
