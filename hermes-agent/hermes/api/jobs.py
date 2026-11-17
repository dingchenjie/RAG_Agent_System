"""统一的 run 提交入口。

HTTP、周期触发器和重启恢复共用这里的执行体：只有这一处会创建 run，
幂等、session 归属、限流配额与调度提交都在这里收敛。
"""
from __future__ import annotations

from ..errors import ApiError
from ..ids import new_run_id
from ..runtime.cancel import CancelToken
from .state import AppState


def build_job(state: AppState, *, run_id: str, session_id: str, tenant_id: str,
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


def tracked_job(state: AppState, tenant_id: str, job):
    """run 结束时释放在途配额；无论成功、失败还是取消都会走到。"""
    async def wrapped():
        try:
            return await job()
        finally:
            state.limiter.release(tenant_id)
    return wrapped


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
            # 幂等重放不消耗配额，避免客户端重试把额度打光
            return {"run_id": existing["id"], "state": existing["state"], "deduplicated": True}

    decision = state.limiter.acquire(tenant_id)
    if not decision.allowed:
        state.metrics.inc("run_rejected_total", reason=decision.reason, tenant=tenant_id)
        raise ApiError(
            f"租户提交超出配额（{decision.reason}），请 {decision.retry_after_s} 秒后重试",
            code=decision.reason, status=429,
            details={"retry_after_s": decision.retry_after_s, "limit": decision.limit},
            headers={"Retry-After": str(decision.retry_after_s)},
        )

    try:
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
        job = build_job(state, run_id=run_id, session_id=session.id,
                        tenant_id=session.tenant_id, user_id=session.user_id,
                        user_input=user_input, mode=mode, scopes=scopes,
                        token=token, extra_context=extra_context)
        state.scheduler_futures[run_id] = await state.scheduler.submit(
            tracked_job(state, tenant_id, job))
        return {"run_id": run_id, "session_id": session.id, "state": "queued"}
    except BaseException:
        state.limiter.release(tenant_id)
        raise
