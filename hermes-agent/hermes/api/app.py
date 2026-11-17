"""HTTP 服务层。

对外只暴露资源化的接口：run 是资源，事件流是 run 的从属资源，
取消/确认是对 run 的状态迁移。这样 SSE、轮询、重放用的是同一套事件数据。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager, suppress
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
from ..eval.replay import Replay, diff_traces
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
from ..observability.events import Event, EventBus, EventStore, EventType
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
from ..triggers import Trigger, TriggerService
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


def _as_seq(raw: str | None) -> int:
    """Last-Event-ID 可能被客户端写成任意字符串，非法值按 0 处理而不是报错。"""
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


# ------------------------------------------------------------------ 请求模型
class CreateSessionBody(BaseModel):
    tenant_id: str = "default"
    user_id: str = "anonymous"
    title: str | None = None


class RunContext(BaseModel):
    """调用方附带的业务上下文，进入上下文预算但不写入会话历史。"""

    page: str | None = Field(None, max_length=2000, description="当前页面/业务对象摘要")
    rules: list[str] = Field(default_factory=list, max_length=10,
                             description="企业规则，每条建议不超过 200 字")


class CreateRunBody(BaseModel):
    input: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    tenant_id: str = "default"
    user_id: str = "anonymous"
    scopes: list[str] = Field(default_factory=lambda: ["tool:read"])
    mode: str = Field("single", description="single | orchestrate")
    context: RunContext | None = None


class CreateTriggerBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    input: str = Field(..., min_length=1, max_length=4000)
    interval_s: int = Field(60, ge=10, le=86400, description="触发周期（秒），最小 10")
    mode: str = Field("single", description="single | orchestrate")
    tenant_id: str = "default"
    user_id: str = "anonymous"
    scopes: list[str] | None = None
    session_id: str | None = None


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
    subject_type: str | None = Field(None, max_length=40)
    subject_id: str | None = Field(None, max_length=120)


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
    triggers: TriggerService
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
    triggers = TriggerService(db)

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
        lessons=lessons, triggers=triggers, runtime=runtime,
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
        await _recover_runs(state)
        trigger_task: asyncio.Task | None = None
        if settings.trigger_tick_s > 0:
            trigger_task = asyncio.create_task(_trigger_loop(state))
        try:
            yield
        finally:
            if trigger_task is not None:
                trigger_task.cancel()
                with suppress(asyncio.CancelledError):
                    await trigger_task
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

    # --------------------------------------------------------------- 统一提交入口
    def _build_job(state: AppState, *, run_id: str, session_id: str, tenant_id: str,
                   user_id: str, user_input: str, mode: str, scopes: frozenset[str],
                   token: CancelToken, extra_context: dict | None):
        """HTTP、触发器和重启恢复共用的执行体，避免出现第二套跑 run 的代码。"""
        if mode == "orchestrate":
            async def job():
                result = await state.orchestrator.run(
                    goal=user_input, tenant_id=tenant_id, user_id=user_id,
                    scopes=scopes, cancel=token, parent_run_id=run_id,
                    extra_context=extra_context)
                state.runs.update(run_id, state=result.state, answer=result.answer,
                                  usage=result.usage)
                return result
        else:
            async def job():
                return await state.runtime.run(
                    user_input=user_input, session_id=session_id, tenant_id=tenant_id,
                    user_id=user_id, scopes=scopes, cancel=token, run_id=run_id,
                    extra_context=extra_context)
        return job

    async def enqueue_run(
        state: AppState,
        *,
        user_input: str,
        tenant_id: str,
        user_id: str,
        scopes: frozenset[str],
        mode: str = "single",
        idempotency_key: str | None = None,
        session_id: str | None = None,
        title: str | None = None,
        is_admin: bool = False,
        extra_context: dict | None = None,
    ) -> dict:
        """HTTP 请求与周期触发共用的提交逻辑。

        只有这一处会创建 run：幂等、session 归属、调度提交都在这里，
        触发器因此不需要（也不允许）复制一份"建 run"的代码。
        """
        if idempotency_key:
            existing = state.runs.find_by_idempotency(idempotency_key)
            if existing:
                if not is_admin and existing.get("tenant_id") != tenant_id:
                    raise ApiError("幂等键已被其他租户使用", code="idempotency_conflict", status=409)
                return {"run_id": existing["id"], "state": existing["state"], "deduplicated": True}

        session = state.sessions.get_session(session_id) if session_id else None
        if session is not None and not is_admin and session.tenant_id != tenant_id:
            raise ApiError("session 不存在", code="session_not_found", status=404)
        if session is None:
            session = state.sessions.create_session(
                tenant_id=tenant_id, user_id=user_id, title=title or user_input[:40])

        run_id = new_run_id()
        state.runs.create_run(
            session_id=session.id, tenant_id=session.tenant_id, user_id=session.user_id,
            user_input=user_input, idempotency_key=idempotency_key, run_id=run_id,
            request={"mode": mode, "scopes": sorted(scopes), "context": extra_context},
        )
        state.runs.update(run_id, state="queued")
        token = CancelToken()
        state.cancel_tokens[run_id] = token
        job = _build_job(state, run_id=run_id, session_id=session.id,
                         tenant_id=session.tenant_id, user_id=session.user_id,
                         user_input=user_input, mode=mode, scopes=scopes,
                         token=token, extra_context=extra_context)
        state.scheduler_futures[run_id] = await state.scheduler.submit(job)
        return {"run_id": run_id, "session_id": session.id, "state": "queued"}

    async def _recover_runs(state: AppState) -> None:
        """进程重启对账：queued 重新入队，running 标记中断等人工恢复。"""
        rows = await asyncio.to_thread(state.runs.list_unfinished)
        for row in rows:
            run_id = row["id"]
            try:
                request = json.loads(row.get("request") or "{}")
            except (TypeError, json.JSONDecodeError):
                request = {}
            extra_context = request.get("context") if isinstance(request, dict) else None

            if row["state"] == "queued":
                token = CancelToken()
                state.cancel_tokens[run_id] = token
                job = _build_job(
                    state, run_id=run_id, session_id=row["session_id"],
                    tenant_id=row["tenant_id"], user_id=row["user_id"],
                    user_input=row["input"], mode=request.get("mode", "single"),
                    scopes=frozenset(request.get("scopes") or ["tool:read"]),
                    token=token, extra_context=extra_context)
                try:
                    state.scheduler_futures[run_id] = await state.scheduler.submit(job)
                    logger.info("重启恢复：重新入队 run=%s", run_id)
                except Exception as exc:  # noqa: BLE001
                    await asyncio.to_thread(
                        state.runs.mark_interrupted, run_id, f"重启恢复失败: {exc}")
                    logger.warning("重启恢复失败 run=%s: %s", run_id, exc)
                continue

            reason = "服务重启导致运行中断，可使用 /v1/runs/{id}/resume 从 checkpoint 恢复"
            marked = await asyncio.to_thread(state.runs.mark_interrupted, run_id, reason)
            if marked:
                await state.bus.emit(Event(
                    run_id=run_id, type=EventType.RUN_FAILED,
                    payload={"error": reason, "kind": "process_restart", "recoverable": True},
                ))
                logger.warning("重启对账：run=%s 标记为 interrupted", run_id)

    async def _trigger_loop(state: AppState) -> None:
        """周期扫描到期触发器；单个触发器失败不影响其他触发器。"""
        while True:
            try:
                due = await asyncio.to_thread(state.triggers.due)
                for trigger in due:
                    try:
                        submitted = await enqueue_run(
                            state, user_input=trigger.input, tenant_id=trigger.tenant_id,
                            user_id=trigger.user_id, scopes=frozenset(trigger.scopes),
                            mode=trigger.mode, session_id=trigger.session_id,
                            idempotency_key=f"trigger:{trigger.id}:{int(trigger.next_fire_at)}",
                            title=f"[定时] {trigger.name}", is_admin=True)
                        await asyncio.to_thread(state.triggers.mark_fired,
                                                trigger.id, submitted["run_id"],
                                                trigger.next_fire_at)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("触发器 %s 提交失败: %s", trigger.id, exc)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("触发器扫描失败: %s", exc)
            await asyncio.sleep(max(1.0, settings.trigger_tick_s))

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

    @app.get("/v1/sessions")
    async def list_sessions(request: Request, tenant_id: str | None = Query(None),
                            user_id: str | None = Query(None),
                            limit: int = Query(20, ge=1, le=100),
                            principal: Principal = Depends(who)) -> dict:
        state = st(request)
        tenant = effective_tenant(principal, tenant_id)
        # 普通调用方只能看自己的会话；管理员可以按 user_id 过滤，不传则看整个租户
        user_filter = user_id if principal.is_admin else principal.user_id
        sessions = await asyncio.to_thread(
            state.sessions.list_sessions, tenant_id=tenant, user_id=user_filter, limit=limit)
        return {"items": [{"session_id": s.id, "tenant_id": s.tenant_id, "user_id": s.user_id,
                           "title": s.title, "created_at": s.created_at,
                           "updated_at": s.updated_at} for s in sessions]}

    @app.get("/v1/sessions/{session_id}/messages")
    async def session_messages(session_id: str, request: Request,
                               limit: int = Query(200, ge=1, le=500),
                               principal: Principal = Depends(who)) -> dict:
        state = st(request)
        session = state.sessions.get_session(session_id)
        if session is None or (not principal.is_admin
                               and session.tenant_id != principal.tenant_id):
            raise ApiError("session 不存在", code="session_not_found", status=404)
        messages = await asyncio.to_thread(state.sessions.load_messages, session_id, limit)
        return {"session_id": session_id, "items": [m.to_dict() for m in messages]}

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
        return await enqueue_run(
            state, user_input=body.input, tenant_id=tenant_id, user_id=user_id,
            scopes=effective_scopes(principal, body.scopes), mode=body.mode,
            idempotency_key=idempotency_key, session_id=body.session_id,
            title=body.input[:40], is_admin=principal.is_admin,
            extra_context=body.context.model_dump() if body.context else None,
        )

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
                            last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
                            principal: Principal = Depends(who)):
        state = st(request)
        visible_run(state, run_id, principal)
        # 浏览器 EventSource 断线重连会自动带 Last-Event-ID，取两者较大值作为续传起点
        start = max(after_seq, _as_seq(last_event_id))

        async def generator():
            queue = state.bus.subscribe(run_id)
            finished = False
            try:
                history = await asyncio.to_thread(state.event_store.history, run_id, start)
                last = start
                for event in history:
                    if event.seq <= last:
                        continue
                    last = event.seq
                    yield {"id": str(event.seq), "event": event.type.value,
                           "data": json.dumps(event.to_dict(), ensure_ascii=False)}
                    if event.type.value in TERMINAL_EVENTS:
                        # 已结束的 run 回放完就关流，否则客户端会一直挂着收心跳
                        finished = True
                        break
                while not finished:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield {"comment": "ping"}
                        continue
                    if event.seq <= last:
                        continue
                    last = event.seq
                    yield {"id": str(event.seq), "event": event.type.value,
                           "data": json.dumps(event.to_dict(), ensure_ascii=False)}
                    if event.type.value in TERMINAL_EVENTS:
                        finished = True
            finally:
                state.bus.unsubscribe(run_id, queue)

        return EventSourceResponse(generator(), ping=15)

    @app.get("/v1/runs/{run_id}/replay")
    async def replay_run(run_id: str, request: Request,
                         principal: Principal = Depends(who)) -> dict:
        state = st(request)
        visible_run(state, run_id, principal)
        timeline = await asyncio.to_thread(Replay(state.event_store).timeline, run_id)
        return {"run_id": run_id, "timeline": timeline}

    @app.get("/v1/runs/{run_id}/diff")
    async def diff_run(run_id: str, request: Request, against: str = Query(..., min_length=1),
                       principal: Principal = Depends(who)) -> dict:
        """对比两次运行：换模型 / 换 Prompt / 换工具后，差异到底在哪。"""
        state = st(request)
        visible_run(state, run_id, principal)
        visible_run(state, against, principal)
        baseline = await asyncio.to_thread(state.event_store.history, against, 0, 2000)
        candidate = await asyncio.to_thread(state.event_store.history, run_id, 0, 2000)
        return {"baseline_run_id": against, "candidate_run_id": run_id,
                "diff": diff_traces(baseline, candidate)}

    # --------------------------------------------------------------- 周期触发
    def visible_trigger(state: AppState, trigger_id: str, principal: Principal) -> Trigger:
        try:
            trigger = state.triggers.get(trigger_id)
        except KeyError:
            raise ApiError("触发器不存在", code="trigger_not_found", status=404) from None
        if not principal.is_admin and trigger.tenant_id != principal.tenant_id:
            raise ApiError("触发器不存在", code="trigger_not_found", status=404)
        return trigger

    @app.post("/v1/triggers", status_code=201)
    async def create_trigger(body: CreateTriggerBody, request: Request,
                             principal: Principal = Depends(who)) -> dict:
        state = st(request)
        require_scope(principal, "trigger:write")
        try:
            trigger = state.triggers.create(
                input=body.input, name=body.name, interval_s=body.interval_s,
                tenant_id=effective_tenant(principal, body.tenant_id),
                user_id=effective_user(principal, body.user_id), mode=body.mode,
                scopes=sorted(effective_scopes(principal, body.scopes)),
                session_id=body.session_id,
            )
        except ValueError as exc:
            raise ApiError(str(exc), code="invalid_trigger", status=422) from exc
        return trigger.to_dict()

    @app.get("/v1/triggers")
    async def list_triggers(request: Request, tenant_id: str | None = Query(default=None),
                            principal: Principal = Depends(who)) -> dict:
        state = st(request)
        require_scope(principal, "trigger:read")
        tenant = effective_tenant(principal, tenant_id)
        return {"triggers": [t.to_dict() for t in state.triggers.list(tenant_id=tenant)]}

    @app.post("/v1/triggers/{trigger_id}/enable")
    async def enable_trigger(trigger_id: str, request: Request,
                             principal: Principal = Depends(who)) -> dict:
        state = st(request)
        require_scope(principal, "trigger:write")
        visible_trigger(state, trigger_id, principal)
        state.triggers.set_enabled(trigger_id, True)
        return {"trigger_id": trigger_id, "enabled": True}

    @app.post("/v1/triggers/{trigger_id}/disable")
    async def disable_trigger(trigger_id: str, request: Request,
                              principal: Principal = Depends(who)) -> dict:
        state = st(request)
        require_scope(principal, "trigger:write")
        visible_trigger(state, trigger_id, principal)
        state.triggers.set_enabled(trigger_id, False)
        return {"trigger_id": trigger_id, "enabled": False}

    @app.delete("/v1/triggers/{trigger_id}")
    async def delete_trigger(trigger_id: str, request: Request,
                             principal: Principal = Depends(who)) -> dict:
        state = st(request)
        require_scope(principal, "trigger:write")
        visible_trigger(state, trigger_id, principal)
        state.triggers.remove(trigger_id)
        return {"trigger_id": trigger_id, "deleted": True}

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
        row = visible_run(state, run_id, principal)
        if row["state"] not in ("paused", "interrupted"):
            raise ApiError(f"当前状态 {row['state']} 不可恢复", code="invalid_state", status=409)
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
            scopes=effective_scopes(principal, body.scopes),
            extra_context=body.context.model_dump() if body.context else None)
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
                          subject_type: str | None = None, subject_id: str | None = None,
                          principal: Principal = Depends(who)) -> dict:
        state = st(request)
        tenant_id = effective_tenant(principal, tenant_id)
        user_id = effective_user(principal, user_id)
        records = state.memory.list_memories(
            tenant_id=tenant_id, user_id=user_id, status=status,
            subject_type=subject_type, subject_id=subject_id)
        return {"items": [r.to_dict() for r in records], "stats": state.memory.stats(
            tenant_id=tenant_id, user_id=user_id)}

    @app.post("/v1/memory/propose")
    async def propose_memory(body: ProposeMemoryBody, request: Request,
                             principal: Principal = Depends(who)) -> dict:
        state = st(request)
        created = state.memory.propose_from_text(
            tenant_id=effective_tenant(principal, body.tenant_id),
            user_id=effective_user(principal, body.user_id), text=body.text,
            subject_type=body.subject_type, subject_id=body.subject_id)
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
