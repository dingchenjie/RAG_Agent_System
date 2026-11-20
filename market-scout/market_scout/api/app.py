"""HTTP API：提交调研、查询状态、SSE 事件流、报告下载。"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from ..config import Settings
from ..demo import seed_demo_sources
from ..engine import create_engine, run_question

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class RunRequest(BaseModel):
    question: str
    provider: str | None = None


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="market-scout", version="0.1.0")
    engine_boot = create_engine(settings)
    store = engine_boot.store
    app.state.engines = {}

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as fh:
            return fh.read()

    @app.post("/api/runs")
    async def submit(req: RunRequest, request: Request) -> dict[str, Any]:
        local = Settings.from_env()
        local.db_path = settings.db_path
        local.output_dir = settings.output_dir
        if req.provider:
            local.llm_provider = req.provider
        engine = create_engine(local)
        if local.llm_provider == "mock":
            # 离线演示模式：种入内置竞品资料，Web 端也能看到带 [S#] 引用的报告
            seed_demo_sources(engine.store, engine.run_id)
        app.state.engines[engine.run_id] = engine
        task = asyncio.create_task(run_question(engine, req.question))
        app.state.__dict__.setdefault("_tasks", []).append(task)
        return {"run_id": engine.run_id, "status": "running"}

    @app.get("/api/runs")
    async def list_runs() -> list[dict[str, Any]]:
        return store.list_runs(20)

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        row = store.get_run(run_id)
        if not row:
            raise HTTPException(404, "run 不存在")
        return row

    @app.get("/api/runs/{run_id}/events")
    async def get_events(run_id: str, after: int = 0) -> list[dict[str, Any]]:
        return store.list_events(run_id, after)

    @app.get("/api/runs/{run_id}/report")
    async def get_report(run_id: str) -> Any:
        row = store.get_run(run_id)
        if not row or not row.get("report"):
            raise HTTPException(404, "报告尚未生成")
        return {"run_id": run_id, "report": row["report"]}

    @app.get("/api/runs/{run_id}/stream")
    async def stream(run_id: str, request: Request) -> StreamingResponse:
        async def generator():
            seq = 0
            idle = 0
            while True:
                for event in store.list_events(run_id, seq):
                    seq = event["seq"]
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                row = store.get_run(run_id)
                if row and row["status"] != "running":
                    yield f"data: {json.dumps({'type': 'run.closed', 'data': {'status': row['status']}}, ensure_ascii=False)}\n\n"
                    break
                idle += 1
                if idle > 600 or await request.is_disconnected():
                    break
                await asyncio.sleep(0.4)

        return StreamingResponse(generator(), media_type="text/event-stream")

    @app.get("/api/traces")
    async def traces(limit: int = 50) -> list[dict[str, Any]]:
        path = settings.trace_jsonl
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()[-limit:]
        return [json.loads(line) for line in lines]

    @app.get("/favicon.ico")
    async def favicon() -> FileResponse:
        raise HTTPException(404)

    return app
