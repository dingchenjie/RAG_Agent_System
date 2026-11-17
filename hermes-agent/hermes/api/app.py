"""HTTP 服务层：只负责组装。

路由按资源域拆到 ``hermes.api.routers.*``，这里保留 lifespan、中间件与
生命周期辅助函数（重启对账、触发器扫描、事件保留清理）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request

from ..config import Settings
from ..llm.base import LLMProvider
from ..observability.events import Event, EventType
from ..runtime.cancel import CancelToken
from .errors import install_error_handlers
from .jobs import build_job, enqueue_run, tracked_job
from .routers import knowledge, memory, runs, sessions, system, triggers
from .state import AppState, build_llm, build_state

# 向后兼容再导出：大量测试与脚本从 hermes.api.app 导入这些符号
__all__ = ["AppState", "app", "build_llm", "build_state", "create_app"]

logger = logging.getLogger("hermes.api")


def _prune_events(state: AppState) -> int:
    """按保留天数清理过期中间事件；配置为 0 时关闭。"""
    days = state.settings.event_retention_days
    if days <= 0:
        return 0
    deleted = state.event_store.prune(before_ts=time.time() - days * 86400)
    if deleted:
        logger.info("事件保留策略：清理 %s 条过期中间事件（终态事件保留）", deleted)
    return deleted


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
            job = build_job(
                state, run_id=run_id, session_id=row["session_id"],
                tenant_id=row["tenant_id"], user_id=row["user_id"],
                user_input=row["input"], mode=request.get("mode", "single"),
                scopes=frozenset(request.get("scopes") or ["tool:read"]),
                token=token, extra_context=extra_context)
            state.limiter.track(row["tenant_id"])
            try:
                state.scheduler_futures[run_id] = await state.scheduler.submit(
                    tracked_job(state, row["tenant_id"], job))
                logger.info("重启恢复：重新入队 run=%s", run_id)
            except Exception as exc:  # noqa: BLE001
                state.limiter.release(row["tenant_id"])
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


async def _trigger_loop(state: AppState, settings: Settings) -> None:
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
        await asyncio.to_thread(_prune_events, state)
        trigger_task: asyncio.Task | None = None
        if settings.trigger_tick_s > 0:
            trigger_task = asyncio.create_task(_trigger_loop(state, settings))
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

    # 没有 lifespan 时（例如测试直接调用路由）依赖兜底构建需要这两项
    app.state.hermes_settings = settings
    app.state.hermes_llm_override = llm_provider

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

    app.include_router(system.router)
    app.include_router(sessions.router)
    app.include_router(runs.router)
    app.include_router(triggers.router)
    app.include_router(knowledge.router)
    app.include_router(memory.router)
    return app


app = create_app()
