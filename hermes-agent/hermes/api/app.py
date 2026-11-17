"""HTTP 服务层。

对外只暴露资源化的接口：run 是资源，事件流是 run 的从属资源，
取消/确认是对 run 的状态迁移。这样 SSE、轮询、重放用的是同一套事件数据。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, Query, Request, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..agents.orchestrator import Orchestrator
from ..config import Settings
from ..errors import ApiError, HermesError
from ..ids import new_run_id
from ..knowledge.embedding import build_embedder
from ..knowledge.index import KnowledgeIndex
from ..knowledge.rerank import build_reranker
from ..knowledge.service import KnowledgeService
from ..learning.lessons import LessonService
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
from ..security.auth import Principal, resolve_principal
from ..storage.db import Database
from ..tools.builtin import build_default_tools
from ..tools.registry import ToolRegistry
from ..tools.sandbox import SandboxRunner
from .errors import install_error_handlers

logger = logging.getLogger("hermes.api")

ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".md", ".txt", ".docx", ".html", ".htm"}
UPLOAD_CHUNK_BYTES = 1024 * 1024

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
    lessons: LessonService
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

    embedder = build_embedder(
        settings.embedding_backend, dim=settings.embedding_dim,
        api_key=settings.llm_api_key, model=settings.embedding_model,
        api_model=settings.embedding_api_model,
    )
    knowledge = KnowledgeService(
        KnowledgeIndex(db, embedder),
        default_top_k=settings.rerank_top_k,
        reranker=build_reranker(settings.rerank_backend, model=settings.rerank_model),
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        dedupe_threshold=settings.dedupe_threshold,
    )
    memory = MemoryService(db, embedder=embedder, ttl_days=settings.memory_ttl_days,
                           min_confidence=settings.memory_min_confidence)
    lessons = LessonService(db)

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
        builder=ContextBuilder(settings.context), lessons=lessons,
    )
    orchestrator = Orchestrator(runtime=runtime, llm=resilient, bus=bus)
    scheduler = RunScheduler(settings.max_concurrency)

    return AppState(
        settings=settings, db=db, bus=bus, event_store=event_store, sessions=sessions, runs=runs,
        metrics=metrics, registry=registry, knowledge=knowledge, memory=memory,
        lessons=lessons, runtime=runtime,
        orchestrator=orchestrator, scheduler=scheduler,
    )


# ------------------------------------------------------------------ 工厂
def create_app(settings: Settings | None = None, llm_provider: LLMProvider | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = build_state(settings, llm_provider)
        app.state.hermes = state
        if settings.auth_mode == "off":
            logger.warning("HERMES_AUTH_MODE=off：单租户管理员模式，仅用于本地演示/测试/CI")
        await state.scheduler.start()
        try:
            yield
        finally:
            await state.scheduler.shutdown()
            state.db.close()

    app = FastAPI(title="Hermes Agent Runtime", version="0.1.0", lifespan=lifespan)
    install_error_handlers(app)

    def st(request: Request) -> AppState:
        state = getattr(request.app.state, "hermes", None)
        if state is None:  # 方便测试直接注入
            state = build_state(settings, llm_provider)
            request.app.state.hermes = state
        return state

    # --------------------------------------------------------------- 身份
    def who(request: Request) -> Principal:
        state = st(request)
        return resolve_principal(request.headers.get("authorization"),
                                 mode=state.settings.auth_mode,
                                 raw_keys=state.settings.api_keys)

    def effective_tenant(principal: Principal, requested: str | None) -> str:
        """租户只能由凭证决定；管理员可显式指定，普通调用方越权直接 403。"""
        if principal.is_admin:
            return requested or principal.tenant_id
        if requested and requested not in ("default", principal.tenant_id):
            raise ApiError("禁止访问其他租户的数据", code="tenant_override_denied", status=403)
        return principal.tenant_id

    def effective_user(principal: Principal, requested: str | None) -> str:
        if principal.is_admin:
            return requested or principal.user_id
        if requested and requested not in ("anonymous", principal.user_id):
            raise ApiError("禁止冒用其他用户", code="tenant_override_denied", status=403)
        return principal.user_id

    def effective_scopes(principal: Principal, requested: list[str] | None) -> frozenset[str]:
        """普通调用方只能用凭证里的 scope；管理员可按请求体降权，但不能升权。"""
        if principal.is_admin and requested:
            wanted = frozenset(requested) & principal.scopes
            return wanted or principal.scopes
        return principal.scopes

    def require_scope(principal: Principal, scope: str) -> None:
        if not principal.has_scope(scope):
            raise ApiError(f"缺少权限 {scope}", code="scope_denied", status=403)

    def visible_run(state: AppState, run_id: str, principal: Principal) -> dict:
        """不存在与无权访问统一返回 404，避免用状态码枚举 run id。"""
        row = state.runs.get_run(run_id)
        if row is None or (not principal.is_admin and row.get("tenant_id") != principal.tenant_id):
            raise ApiError("run 不存在", code="run_not_found", status=404)
        return row

    @app.middleware("http")
    async def http_metrics(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        state = getattr(request.app.state, "hermes", None)
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        if state is not None and path != "/metrics":  # 抓取自身不计入，避免自激
            state.metrics.inc("http_requests_total", path=path, status=str(response.status_code))
            state.metrics.observe("http_request_duration_ms",
                                  (time.perf_counter() - started) * 1000, path=path)
        return response

    # --------------------------------------------------------------- 健康/指标
    @app.get("/healthz")
    async def healthz(request: Request) -> dict:
        state = st(request)
        return {"status": "ok", "llm": state.runtime.llm.name,
                **state.knowledge.describe(),
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
    async def create_session(body: CreateSessionBody, request: Request,
                             principal: Principal = Depends(who)) -> dict:
        session = st(request).sessions.create_session(
            tenant_id=effective_tenant(principal, body.tenant_id),
            user_id=effective_user(principal, body.user_id), title=body.title)
        return {"session_id": session.id, "tenant_id": session.tenant_id, "user_id": session.user_id}

    # --------------------------------------------------------------- 运行
    @app.post("/v1/runs", status_code=202)
    async def create_run(
        body: CreateRunBody,
        request: Request,
        principal: Principal = Depends(who),
        idempotency_key: str | None = Header(default=None, alias="X-Idempotency-Key"),
    ) -> dict:
        state = st(request)
        require_scope(principal, "tool:read")
        tenant_id = effective_tenant(principal, body.tenant_id)
        user_id = effective_user(principal, body.user_id)
        if idempotency_key:
            existing = state.runs.find_by_idempotency(idempotency_key)
            if existing:
                if not principal.is_admin and existing.get("tenant_id") != tenant_id:
                    raise ApiError("幂等键已被其他租户使用", code="idempotency_conflict", status=409)
                return {"run_id": existing["id"], "state": existing["state"], "deduplicated": True}

        session = state.sessions.get_session(body.session_id) if body.session_id else None
        if session is not None and not principal.is_admin and session.tenant_id != tenant_id:
            raise ApiError("session 不存在", code="session_not_found", status=404)
        if session is None:
            session = state.sessions.create_session(
                tenant_id=tenant_id, user_id=user_id, title=body.input[:40])
        run_id = new_run_id()
        state.runs.create_run(session_id=session.id, tenant_id=session.tenant_id,
                              user_id=session.user_id, user_input=body.input,
                              idempotency_key=idempotency_key, run_id=run_id)
        state.runs.update(run_id, state="queued")
        token = CancelToken()
        state.cancel_tokens[run_id] = token
        scopes = effective_scopes(principal, body.scopes)

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
    async def get_run(run_id: str, request: Request,
                      principal: Principal = Depends(who)) -> dict:
        state = st(request)
        row = visible_run(state, run_id, principal)
        checkpoint = state.runs.load_checkpoint(run_id) or {}
        return {
            "run_id": row["id"], "state": row["state"], "answer": row["answer"],
            "error": row["error"], "steps": row["steps"],
            "usage": json.loads(row["usage"] or "{}"),
            "pending": checkpoint.get("pending"),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    @app.get("/v1/runs/{run_id}/events")
    async def stream_events(run_id: str, request: Request, after_seq: int = Query(0, ge=0),
                            principal: Principal = Depends(who)):
        state = st(request)
        visible_run(state, run_id, principal)

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
    async def get_trace(run_id: str, request: Request, limit: int = Query(2000, le=5000),
                        principal: Principal = Depends(who)) -> dict:
        state = st(request)
        visible_run(state, run_id, principal)
        events = await asyncio.to_thread(state.event_store.history, run_id, 0, limit)
        return {"run_id": run_id, "count": len(events), "events": [e.to_dict() for e in events]}

    @app.post("/v1/runs/{run_id}/cancel")
    async def cancel_run(run_id: str, request: Request,
                         principal: Principal = Depends(who)) -> dict:
        state = st(request)
        visible_run(state, run_id, principal)
        token = state.cancel_tokens.get(run_id)
        if token is None:
            raise ApiError("run 已结束或不可取消", code="run_not_cancellable", status=409)
        token.cancel()
        return {"run_id": run_id, "state": "cancelling"}

    @app.post("/v1/runs/{run_id}/confirm")
    async def confirm_run(run_id: str, request: Request,
                          principal: Principal = Depends(who)) -> dict:
        state = st(request)
        row = visible_run(state, run_id, principal)
        if row["state"] != "paused":
            raise ApiError(f"当前状态 {row['state']} 不可确认", code="invalid_state", status=409)
        token = CancelToken()
        state.cancel_tokens[run_id] = token
        future = await state.scheduler.submit(
            lambda: state.runtime.resume(run_id, confirmed=True, cancel=token))
        state.scheduler_futures[run_id] = future
        return {"run_id": run_id, "state": "resuming"}

    @app.post("/v1/runs/{run_id}/resume")
    async def resume_run(run_id: str, request: Request,
                         principal: Principal = Depends(who)) -> dict:
        state = st(request)
        visible_run(state, run_id, principal)
        token = CancelToken()
        state.cancel_tokens[run_id] = token
        future = await state.scheduler.submit(
            lambda: state.runtime.resume(run_id, confirmed=False, cancel=token))
        state.scheduler_futures[run_id] = future
        return {"run_id": run_id, "state": "resuming"}

    # --------------------------------------------------------------- 多 Agent
    @app.post("/v1/orchestrate")
    async def orchestrate(body: CreateRunBody, request: Request,
                          principal: Principal = Depends(who)) -> dict:
        state = st(request)
        require_scope(principal, "tool:read")
        result = await state.orchestrator.run(
            goal=body.input,
            tenant_id=effective_tenant(principal, body.tenant_id),
            user_id=effective_user(principal, body.user_id),
            scopes=effective_scopes(principal, body.scopes))
        return result.to_dict()

    # --------------------------------------------------------------- 知识库
    @app.post("/v1/knowledge/documents")
    async def ingest(body: IngestBody, request: Request,
                     principal: Principal = Depends(who)) -> dict:
        state = st(request)
        require_scope(principal, "knowledge:write")
        report = await asyncio.to_thread(
            state.knowledge.ingest_text, body.doc_id, body.text, version=body.version,
            title=body.title, acl=body.acl,
            chunk_size=state.settings.chunk_size, overlap=state.settings.chunk_overlap,
        )
        return {"doc_id": report.doc_id, "version": report.version, "chunks": report.chunks,
                "replaced_versions": report.replaced_versions, "chars": report.chars}

    @app.get("/v1/knowledge/search")
    async def search(q: str, request: Request, top_k: int = Query(4, ge=1, le=10),
                     acl_group: list[str] | None = Query(None),
                     principal: Principal = Depends(who)) -> dict:
        state = st(request)
        require_scope(principal, "knowledge:read")
        hits = await asyncio.to_thread(state.knowledge.search, q, top_k=top_k,
                                       acl_groups=set(acl_group) if acl_group else None)
        return {"query": q, "hits": [h.to_dict() for h in hits]}

    @app.post("/v1/knowledge/files")
    async def upload_document(
        request: Request,
        file: UploadFile = File(...),
        doc_id: str | None = Query(None),
        acl_group: list[str] | None = Query(None),
        principal: Principal = Depends(who),
    ) -> dict:
        """上传 PDF/MD/TXT/HTML/DOCX 并入库（走完整的 loader → 分块 → 去冗余链路）。

        防护顺序：后缀白名单 → Content-Length 预检 → 分块落盘并累计计数。
        任一步失败都会删掉半成品文件，不留孤儿。
        """
        state = st(request)
        require_scope(principal, "knowledge:write")
        raw_name = Path(file.filename or "upload.txt").name
        suffix = Path(raw_name).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            raise ApiError(f"不支持的文件类型 {suffix or '<无后缀>'}",
                           code="unsupported_file_type", status=415,
                           details={"allowed": sorted(ALLOWED_UPLOAD_SUFFIXES)})
        max_bytes = state.settings.max_upload_mb * 1024 * 1024
        declared = (request.headers.get("content-length") or "").strip()
        if declared.isdigit() and int(declared) > max_bytes + 64 * 1024:
            raise ApiError(f"文件超过 {state.settings.max_upload_mb} MiB 上限",
                           code="file_too_large", status=413)
        upload_dir = Path(state.settings.upload_dir)
        await asyncio.to_thread(upload_dir.mkdir, parents=True, exist_ok=True)
        target = upload_dir / f"{uuid4().hex[:8]}-{raw_name}"
        written = 0
        ok = False
        try:
            with target.open("wb") as handle:
                while True:
                    chunk = await file.read(UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise ApiError(f"文件超过 {state.settings.max_upload_mb} MiB 上限",
                                       code="file_too_large", status=413)
                    handle.write(chunk)
            try:
                report = await asyncio.to_thread(
                    state.knowledge.ingest_path, target,
                    doc_id=doc_id or Path(raw_name).stem, acl=acl_group or [])
            except HermesError as exc:
                raise ApiError(f"解析失败：{exc.message}", code="parse_failed", status=422) from exc
            ok = True
            return {"doc_id": report.doc_id, "doc_ids": report.doc_ids, "version": report.version,
                    "chunks": report.chunks, "chars": report.chars,
                    "replaced_versions": report.replaced_versions}
        finally:
            await file.close()
            if not ok:
                await asyncio.to_thread(target.unlink, missing_ok=True)

    @app.get("/v1/knowledge/documents")
    async def list_documents(request: Request, principal: Principal = Depends(who)) -> dict:
        state = st(request)
        require_scope(principal, "knowledge:read")
        return {"items": state.knowledge.documents(), "stats": state.knowledge.stats()}

    @app.delete("/v1/knowledge/documents/{doc_id}")
    async def delete_document(doc_id: str, request: Request,
                              principal: Principal = Depends(who)) -> dict:
        state = st(request)
        require_scope(principal, "knowledge:write")
        removed = await asyncio.to_thread(state.knowledge.index.delete, doc_id)
        if not removed:
            raise ApiError("文档不存在", code="document_not_found", status=404)
        return {"deleted": doc_id, "chunks": removed}

    # --------------------------------------------------------------- 记忆
    @app.get("/v1/memory")
    async def list_memory(request: Request, tenant_id: str = "default", user_id: str = "anonymous",
                          status: str | None = None,
                          principal: Principal = Depends(who)) -> dict:
        state = st(request)
        tenant_id = effective_tenant(principal, tenant_id)
        user_id = effective_user(principal, user_id)
        records = state.memory.list_memories(tenant_id=tenant_id, user_id=user_id, status=status)
        return {"items": [r.to_dict() for r in records], "stats": state.memory.stats(
            tenant_id=tenant_id, user_id=user_id)}

    @app.post("/v1/memory/propose")
    async def propose_memory(body: ProposeMemoryBody, request: Request,
                             principal: Principal = Depends(who)) -> dict:
        state = st(request)
        created = state.memory.propose_from_text(
            tenant_id=effective_tenant(principal, body.tenant_id),
            user_id=effective_user(principal, body.user_id), text=body.text)
        return {"items": [r.to_dict() for r in created]}

    @app.post("/v1/memory/{memory_id}/confirm")
    async def confirm_memory(memory_id: str, request: Request,
                             tenant_id: str = "default", user_id: str = "anonymous",
                             principal: Principal = Depends(who)) -> dict:
        state = st(request)
        try:
            record = state.memory.confirm(
                memory_id, tenant_id=effective_tenant(principal, tenant_id),
                user_id=effective_user(principal, user_id))
        except Exception as exc:
            raise ApiError(str(exc), code="memory_not_found", status=404) from exc
        return {"item": record.to_dict()}

    @app.post("/v1/memory/{memory_id}/correct")
    async def correct_memory(memory_id: str, body: CorrectMemoryBody, request: Request,
                             tenant_id: str = "default", user_id: str = "anonymous",
                             principal: Principal = Depends(who)) -> dict:
        state = st(request)
        try:
            record = state.memory.correct(
                memory_id, tenant_id=effective_tenant(principal, tenant_id),
                user_id=effective_user(principal, user_id),
                value=body.value, reason=body.reason)
        except Exception as exc:
            raise ApiError(str(exc), code="memory_not_found", status=404) from exc
        return {"item": record.to_dict()}

    @app.delete("/v1/memory/{memory_id}")
    async def delete_memory(memory_id: str, request: Request,
                            tenant_id: str = "default", user_id: str = "anonymous",
                            principal: Principal = Depends(who)) -> dict:
        state = st(request)
        try:
            state.memory.forget(memory_id,
                                tenant_id=effective_tenant(principal, tenant_id),
                                user_id=effective_user(principal, user_id))
        except Exception as exc:
            raise ApiError(str(exc), code="memory_not_found", status=404) from exc
        return {"deleted": memory_id}

    return app


app = create_app()
