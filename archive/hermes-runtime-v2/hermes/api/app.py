"""HTTP 服务层。

对外只暴露资源化的接口：run 是资源，事件流是 run 的从属资源，
取消/确认是对 run 的状态迁移。这样 SSE、轮询、重放用的是同一套事件数据。
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..agents.orchestrator import Orchestrator
from ..config import Settings
from ..ids import new_run_id
from ..knowledge.embedding import HashingEmbedder
from ..knowledge.index import KnowledgeIndex
from ..knowledge.service import KnowledgeService
from ..llm.base import LLMProvider
from ..llm.mock import MockLLM
from ..llm.resilience import CircuitBreaker, ResilientLLM
from ..llm.zhipu import ZhipuLLM
from ..memory.service import MemoryService
from ..observability.events import EventBus, EventStore, EventType
from ..observability.metrics import Metrics
from ..runtime.cancel import CancelToken
from ..runtime.context import ContextBuilder
from ..runtime.runtime import AgentRuntime
from ..runtime.scheduler import RunScheduler
from ..runtime.store import RunStore, SessionStore
from ..storage.db import Database
from ..tools.builtin import build_default_tools
from ..tools.registry import ToolRegistry
from ..tools.sandbox import SandboxRunner

TERMINAL_EVENTS = {
    EventType.RUN_COMPLETED.value,
    EventType.RUN_FAILED.value,
    EventType.RUN_CANCELLED.value,
    EventType.RUN_PAUSED.value,
}


# ------------------------------------------------------------------ 请求模型
class CreateSessionBody(BaseModel):
    tenant_id: str = "default"
    user_id: str = "anonymous"
    title: str | None = None


class CreateRunBody(BaseModel):
    input: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    tenant_id: str = "default"
    user_id: str = "anonymous"
    scopes: list[str] = Field(default_factory=lambda: ["tool:read"])
    mode: str = Field("single", description="single | orchestrate")


class IngestBody(BaseModel):
    doc_id: str
    text: str
    version: str = "1"
    title: str | None = None
    acl: list[str] = Field(default_factory=list)


class ProposeMemoryBody(BaseModel):
    text: str
    tenant_id: str = "default"
    user_id: str = "anonymous"


class CorrectMemoryBody(BaseModel):
    value: str
    reason: str | None = None


# ------------------------------------------------------------------ 应用状态
@dataclass
class AppState:
    settings: Settings
    db: Database
    bus: EventBus
    event_store: EventStore
    sessions: SessionStore
    runs: RunStore
    metrics: Metrics
    registry: ToolRegistry
    knowledge: KnowledgeService
    memory: MemoryService
    runtime: AgentRuntime
    orchestrator: Orchestrator
    scheduler: RunScheduler
    cancel_tokens: dict[str, CancelToken] = field(default_factory=dict)
    scheduler_futures: dict[str, asyncio.Future] = field(default_factory=dict)


def build_llm(settings: Settings, override: LLMProvider | None = None) -> LLMProvider:
    if override is not None:
        return override
    if settings.llm_provider == "zhipu":
        return ZhipuLLM(api_key=settings.llm_api_key, model=settings.llm_model,
                        base_url=settings.llm_base_url, timeout_s=settings.llm_timeout_s)
    return MockLLM()


def build_state(settings: Settings, llm_provider: LLMProvider | None = None) -> AppState:
    db = Database(settings.db_path)
    metrics = Metrics()
    event_store = EventStore(db)
    bus = EventBus(event_store)
    sessions = SessionStore(db)
    runs = RunStore(db)

    embedder = HashingEmbedder(settings.embedding_dim)
    knowledge = KnowledgeService(
        KnowledgeIndex(db, embedder), default_top_k=settings.rerank_top_k)
    memory = MemoryService(db, embedder=embedder, ttl_days=settings.memory_ttl_days,
                           min_confidence=settings.memory_min_confidence)

    registry = ToolRegistry(db=db, metrics=metrics,
                            default_timeout_s=settings.tool_timeout_s,
                            default_retries=settings.tool_max_retries)
    registry.register_many(build_default_tools(
        knowledge=knowledge, memory=memory, sandbox=SandboxRunner(),
        tool_timeout_s=settings.tool_timeout_s,
    ))

    base_llm = build_llm(settings, llm_provider)
    resilient = ResilientLLM(
        base_llm, timeout_s=settings.llm_timeout_s, max_retries=settings.llm_max_retries,
        breaker=CircuitBreaker(settings.llm_breaker_threshold, settings.llm_breaker_reset_s),
        bus=bus,
    )

    runtime = AgentRuntime(
        settings=settings, sessions=sessions, runs=runs, bus=bus, registry=registry,
        llm=resilient, metrics=metrics, memory=memory, knowledge=knowledge,
        builder=ContextBuilder(settings.context),
    )
    orchestrator = Orchestrator(runtime=runtime, llm=resilient, bus=bus)
    scheduler = RunScheduler(settings.max_concurrency)

    return AppState(
        settings=settings, db=db, bus=bus, event_store=event_store, sessions=sessions, runs=runs,
        metrics=metrics, registry=registry, knowledge=knowledge, memory=memory, runtime=runtime,
        orchestrator=orchestrator, scheduler=scheduler,
    )


# ------------------------------------------------------------------ 工厂
def create_app(settings: Settings | None = None, llm_provider: LLMProvider | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = build_state(settings, llm_provider)
        app.state.hermes = state
        await state.scheduler.start()
        try:
            yield
        finally:
            await state.scheduler.shutdown()
            state.db.close()

    app = FastAPI(title="Hermes Agent Runtime", version="0.1.0", lifespan=lifespan)

    def st(request: Request) -> AppState:
        state = getattr(request.app.state, "hermes", None)
        if state is None:  # 方便测试直接注入
            state = build_state(settings, llm_provider)
            request.app.state.hermes = state
        return state

    # --------------------------------------------------------------- 健康/指标
    @app.get("/healthz")
    async def healthz(request: Request) -> dict:
        state = st(request)
        return {"status": "ok", "llm": state.runtime.llm.name,
                "knowledge": state.knowledge.stats(), "scheduler_pending": state.scheduler.pending}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics(request: Request) -> str:
        return st(request).metrics.render_prometheus()

    @app.get("/v1/metrics/json")
    async def metrics_json(request: Request) -> dict:
        state = st(request)
        return {"metrics": state.metrics.snapshot(),
                "scheduler": {"completed": state.scheduler.completed, "failed": state.scheduler.failed,
                              "pending": state.scheduler.pending}}

    # --------------------------------------------------------------- 会话
    @app.post("/v1/sessions")
    async def create_session(body: CreateSessionBody, request: Request) -> dict:
        session = st(request).sessions.create_session(
            tenant_id=body.tenant_id, user_id=body.user_id, title=body.title)
        return {"session_id": session.id, "tenant_id": session.tenant_id, "user_id": session.user_id}

    # --------------------------------------------------------------- 运行
    @app.post("/v1/runs", status_code=202)
    async def create_run(
        body: CreateRunBody,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="X-Idempotency-Key"),
    ) -> dict:
        state = st(request)
        if idempotency_key:
            existing = state.runs.find_by_idempotency(idempotency_key)
            if existing:
                return {"run_id": existing["id"], "state": existing["state"], "deduplicated": True}

        session = state.sessions.get_session(body.session_id) if body.session_id else None
        if session is None:
            session = state.sessions.create_session(
                tenant_id=body.tenant_id, user_id=body.user_id, title=body.input[:40])
        run_id = new_run_id()
        state.runs.create_run(session_id=session.id, tenant_id=session.tenant_id,
                              user_id=session.user_id, user_input=body.input,
                              idempotency_key=idempotency_key, run_id=run_id)
        state.runs.update(run_id, state="queued")
        token = CancelToken()
        state.cancel_tokens[run_id] = token
        scopes = frozenset(body.scopes)

        if body.mode == "orchestrate":
            async def job():
                result = await state.orchestrator.run(
                    goal=body.input, tenant_id=session.tenant_id, user_id=session.user_id,
                    scopes=scopes, cancel=token, parent_run_id=run_id)
                state.runs.update(run_id, state=result.state, answer=result.answer,
                                  usage=result.usage)
                return result
        else:
            async def job():
                return await state.runtime.run(
                    user_input=body.input, session_id=session.id, tenant_id=session.tenant_id,
                    user_id=session.user_id, scopes=scopes, cancel=token, run_id=run_id)

        state.scheduler_futures[run_id] = await state.scheduler.submit(job)
        return {"run_id": run_id, "session_id": session.id, "state": "queued"}

    @app.get("/v1/runs/{run_id}")
    async def get_run(run_id: str, request: Request) -> dict:
        state = st(request)
        row = state.runs.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="run 不存在")
        checkpoint = state.runs.load_checkpoint(run_id) or {}
        return {
            "run_id": row["id"], "state": row["state"], "answer": row["answer"],
            "error": row["error"], "steps": row["steps"],
            "usage": json.loads(row["usage"] or "{}"),
            "pending": checkpoint.get("pending"),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    @app.get("/v1/runs/{run_id}/events")
    async def stream_events(run_id: str, request: Request, after_seq: int = Query(0, ge=0)):
        state = st(request)

        async def generator():
            queue = state.bus.subscribe(run_id)
            try:
                history = await asyncio.to_thread(state.event_store.history, run_id, after_seq)
                last = after_seq
                for event in history:
                    last = event.seq
                    yield {"event": event.type.value,
                           "data": json.dumps(event.to_dict(), ensure_ascii=False)}
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield {"comment": "ping"}
                        continue
                    if event.seq <= last:
                        continue
                    last = event.seq
                    yield {"event": event.type.value,
                           "data": json.dumps(event.to_dict(), ensure_ascii=False)}
                    if event.type.value in TERMINAL_EVENTS:
                        break
            finally:
                state.bus.unsubscribe(run_id, queue)

        return EventSourceResponse(generator())

    @app.get("/v1/runs/{run_id}/trace")
    async def get_trace(run_id: str, request: Request, limit: int = Query(2000, le=5000)) -> dict:
        state = st(request)
        events = await asyncio.to_thread(state.event_store.history, run_id, 0, limit)
        return {"run_id": run_id, "count": len(events), "events": [e.to_dict() for e in events]}

    @app.post("/v1/runs/{run_id}/cancel")
    async def cancel_run(run_id: str, request: Request) -> dict:
        state = st(request)
        row = state.runs.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="run 不存在")
        token = state.cancel_tokens.get(run_id)
        if token is None:
            raise HTTPException(status_code=409, detail="run 已结束或不可取消")
        token.cancel()
        return {"run_id": run_id, "state": "cancelling"}

    @app.post("/v1/runs/{run_id}/confirm")
    async def confirm_run(run_id: str, request: Request) -> dict:
        state = st(request)
        row = state.runs.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="run 不存在")
        if row["state"] != "paused":
            raise HTTPException(status_code=409, detail=f"当前状态 {row['state']} 不可确认")
        token = CancelToken()
        state.cancel_tokens[run_id] = token
        future = await state.scheduler.submit(
            lambda: state.runtime.resume(run_id, confirmed=True, cancel=token))
        state.scheduler_futures[run_id] = future
        return {"run_id": run_id, "state": "resuming"}

    @app.post("/v1/runs/{run_id}/resume")
    async def resume_run(run_id: str, request: Request) -> dict:
        state = st(request)
        row = state.runs.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="run 不存在")
        token = CancelToken()
        state.cancel_tokens[run_id] = token
        future = await state.scheduler.submit(
            lambda: state.runtime.resume(run_id, confirmed=False, cancel=token))
        state.scheduler_futures[run_id] = future
        return {"run_id": run_id, "state": "resuming"}

    # --------------------------------------------------------------- 多 Agent
    @app.post("/v1/orchestrate")
    async def orchestrate(body: CreateRunBody, request: Request) -> dict:
        state = st(request)
        result = await state.orchestrator.run(
            goal=body.input, tenant_id=body.tenant_id, user_id=body.user_id,
            scopes=frozenset(body.scopes))
        return result.to_dict()

    # --------------------------------------------------------------- 知识库
    @app.post("/v1/knowledge/documents")
    async def ingest(body: IngestBody, request: Request) -> dict:
        state = st(request)
        report = await asyncio.to_thread(
            state.knowledge.ingest_text, body.doc_id, body.text, version=body.version,
            title=body.title, acl=body.acl,
            chunk_size=state.settings.chunk_size, overlap=state.settings.chunk_overlap,
        )
        return {"doc_id": report.doc_id, "version": report.version, "chunks": report.chunks,
                "replaced_versions": report.replaced_versions, "chars": report.chars}

    @app.get("/v1/knowledge/search")
    async def search(q: str, request: Request, top_k: int = Query(4, ge=1, le=10),
                     acl_group: list[str] | None = Query(None)) -> dict:
        state = st(request)
        hits = await asyncio.to_thread(state.knowledge.search, q, top_k=top_k,
                                       acl_groups=set(acl_group) if acl_group else None)
        return {"query": q, "hits": [h.to_dict() for h in hits]}

    # --------------------------------------------------------------- 记忆
    @app.get("/v1/memory")
    async def list_memory(request: Request, tenant_id: str = "default", user_id: str = "anonymous",
                          status: str | None = None) -> dict:
        state = st(request)
        records = state.memory.list_memories(tenant_id=tenant_id, user_id=user_id, status=status)
        return {"items": [r.to_dict() for r in records], "stats": state.memory.stats(
            tenant_id=tenant_id, user_id=user_id)}

    @app.post("/v1/memory/propose")
    async def propose_memory(body: ProposeMemoryBody, request: Request) -> dict:
        state = st(request)
        created = state.memory.propose_from_text(
            tenant_id=body.tenant_id, user_id=body.user_id, text=body.text)
        return {"items": [r.to_dict() for r in created]}

    @app.post("/v1/memory/{memory_id}/confirm")
    async def confirm_memory(memory_id: str, request: Request,
                             tenant_id: str = "default", user_id: str = "anonymous") -> dict:
        state = st(request)
        try:
            record = state.memory.confirm(memory_id, tenant_id=tenant_id, user_id=user_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"item": record.to_dict()}

    @app.post("/v1/memory/{memory_id}/correct")
    async def correct_memory(memory_id: str, body: CorrectMemoryBody, request: Request,
                             tenant_id: str = "default", user_id: str = "anonymous") -> dict:
        state = st(request)
        try:
            record = state.memory.correct(memory_id, tenant_id=tenant_id, user_id=user_id,
                                          value=body.value, reason=body.reason)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"item": record.to_dict()}

    @app.delete("/v1/memory/{memory_id}")
    async def delete_memory(memory_id: str, request: Request,
                            tenant_id: str = "default", user_id: str = "anonymous") -> dict:
        state = st(request)
        try:
            state.memory.forget(memory_id, tenant_id=tenant_id, user_id=user_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"deleted": memory_id}

    return app


app = create_app()
