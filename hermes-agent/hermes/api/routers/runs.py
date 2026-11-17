"""运行域路由：提交、查询、事件流、重放/对比、取消/确认/恢复与多 Agent。"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ...errors import ApiError
from ...eval.replay import Replay, diff_traces
from ...observability.events import Event, EventType
from ...runtime.cancel import CancelToken
from ...security.auth import Principal
from ..deps import effective_scopes, effective_tenant, effective_user, get_state, require_scope, visible_run, who
from ..jobs import enqueue_run, tracked_job

router = APIRouter()

TERMINAL_EVENTS = {
    EventType.RUN_COMPLETED.value,
    EventType.RUN_FAILED.value,
    EventType.RUN_CANCELLED.value,
    EventType.RUN_PAUSED.value,
}


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


def _as_seq(raw: str | None) -> int:
    """Last-Event-ID 可能被客户端写成任意字符串，非法值按 0 处理而不是报错。"""
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def _collect_artifacts(events: list[Event]) -> tuple[list[dict], list[dict]]:
    """从事件流聚合"答案引用了什么、调用过哪些工具"。

    终态事件里已经有 Runtime 写好的快照，优先用它；run 还没结束时退化为按
    工具完成事件聚合。events 仍是唯一事实源，不额外建表。
    """
    citations: list[dict] = []
    tools: list[dict] = []
    for event in events:
        payload = event.payload or {}
        if event.type in (EventType.RUN_COMPLETED, EventType.RUN_FAILED):
            citations = payload.get("citations") or []
            tools = payload.get("tools") or []
        elif event.type in (EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED):
            tools.append({
                "tool": payload.get("tool"),
                "version": payload.get("version"),
                "ok": payload.get("ok"),
                "code": payload.get("code"),
                "latency_ms": payload.get("latency_ms"),
            })
    return citations, tools


@router.post("/v1/runs", status_code=202)
async def create_run(
    body: CreateRunBody,
    request: Request,
    principal: Principal = Depends(who),
    idempotency_key: str | None = Header(default=None, alias="X-Idempotency-Key"),
) -> dict:
    state = get_state(request)
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


@router.get("/v1/runs/{run_id}")
async def get_run(run_id: str, request: Request,
                  principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    row = visible_run(state, run_id, principal)
    checkpoint = state.runs.load_checkpoint(run_id) or {}
    events = await asyncio.to_thread(state.event_store.history, run_id, 0, 5000)
    citations, tool_calls = _collect_artifacts(events)
    return {
        "run_id": row["id"], "state": row["state"], "answer": row["answer"],
        "error": row["error"], "steps": row["steps"],
        "usage": json.loads(row["usage"] or "{}"),
        "pending": checkpoint.get("pending"),
        "citations": citations, "tool_calls": tool_calls,
        "created_at": row["created_at"], "updated_at": row["updated_at"],
    }


@router.get("/v1/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request, after_seq: int = Query(0, ge=0),
                        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
                        principal: Principal = Depends(who)):
    state = get_state(request)
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


@router.get("/v1/runs/{run_id}/replay")
async def replay_run(run_id: str, request: Request,
                     principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    visible_run(state, run_id, principal)
    timeline = await asyncio.to_thread(Replay(state.event_store).timeline, run_id)
    return {"run_id": run_id, "timeline": timeline}


@router.get("/v1/runs/{run_id}/diff")
async def diff_run(run_id: str, request: Request, against: str = Query(..., min_length=1),
                   principal: Principal = Depends(who)) -> dict:
    """对比两次运行：换模型 / 换 Prompt / 换工具后，差异到底在哪。"""
    state = get_state(request)
    visible_run(state, run_id, principal)
    visible_run(state, against, principal)
    baseline = await asyncio.to_thread(state.event_store.history, against, 0, 2000)
    candidate = await asyncio.to_thread(state.event_store.history, run_id, 0, 2000)
    return {"baseline_run_id": against, "candidate_run_id": run_id,
            "diff": diff_traces(baseline, candidate)}


@router.get("/v1/runs/{run_id}/trace")
async def get_trace(run_id: str, request: Request, limit: int = Query(2000, le=5000),
                    principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    visible_run(state, run_id, principal)
    events = await asyncio.to_thread(state.event_store.history, run_id, 0, limit)
    return {"run_id": run_id, "count": len(events), "events": [e.to_dict() for e in events]}


@router.post("/v1/runs/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request,
                     principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    visible_run(state, run_id, principal)
    token = state.cancel_tokens.get(run_id)
    if token is None:
        raise ApiError("run 已结束或不可取消", code="run_not_cancellable", status=409)
    token.cancel()
    return {"run_id": run_id, "state": "cancelling"}


@router.post("/v1/runs/{run_id}/confirm")
async def confirm_run(run_id: str, request: Request,
                      principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    row = visible_run(state, run_id, principal)
    if row["state"] != "paused":
        raise ApiError(f"当前状态 {row['state']} 不可确认", code="invalid_state", status=409)
    token = CancelToken()
    state.cancel_tokens[run_id] = token
    state.limiter.track(row["tenant_id"])
    try:
        future = await state.scheduler.submit(tracked_job(
            state, row["tenant_id"],
            lambda: state.runtime.resume(run_id, confirmed=True, cancel=token)))
    except BaseException:
        state.limiter.release(row["tenant_id"])
        raise
    state.scheduler_futures[run_id] = future
    return {"run_id": run_id, "state": "resuming"}


@router.post("/v1/runs/{run_id}/resume")
async def resume_run(run_id: str, request: Request,
                     principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    row = visible_run(state, run_id, principal)
    if row["state"] not in ("paused", "interrupted"):
        raise ApiError(f"当前状态 {row['state']} 不可恢复", code="invalid_state", status=409)
    token = CancelToken()
    state.cancel_tokens[run_id] = token
    state.limiter.track(row["tenant_id"])
    try:
        future = await state.scheduler.submit(tracked_job(
            state, row["tenant_id"],
            lambda: state.runtime.resume(run_id, confirmed=False, cancel=token)))
    except BaseException:
        state.limiter.release(row["tenant_id"])
        raise
    state.scheduler_futures[run_id] = future
    return {"run_id": run_id, "state": "resuming"}


@router.post("/v1/orchestrate")
async def orchestrate(body: CreateRunBody, request: Request,
                      principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    require_scope(principal, "tool:read")
    result = await state.orchestrator.run(
        goal=body.input,
        tenant_id=effective_tenant(principal, body.tenant_id),
        user_id=effective_user(principal, body.user_id),
        scopes=effective_scopes(principal, body.scopes),
        extra_context=body.context.model_dump() if body.context else None)
    return result.to_dict()
