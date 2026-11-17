"""系统域路由：健康检查、指标、用量聚合与事件清理。"""
from __future__ import annotations

import asyncio
import re
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse

from ...errors import ApiError
from ...security.auth import Principal
from ..deps import effective_tenant, get_state, who

router = APIRouter()


def _parse_window(raw: str) -> float:
    """把 ``24h`` / ``7d`` / ``30m`` 解析成秒；非法值直接 422，不做猜测。"""
    match = re.fullmatch(r"(\d+)([mhd])", (raw or "").strip().lower())
    if not match:
        raise ApiError("window 需形如 30m / 24h / 7d", code="invalid_window", status=422)
    value, unit = int(match.group(1)), match.group(2)
    return float(value * {"m": 60, "h": 3600, "d": 86400}[unit])


@router.get("/healthz")
async def healthz(request: Request) -> dict:
    state = get_state(request)
    return {"status": "ok", "llm": state.runtime.llm.name,
            **state.knowledge.describe(),
            "knowledge": state.knowledge.stats(), "scheduler_pending": state.scheduler.pending}


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics(request: Request) -> str:
    return get_state(request).metrics.render_prometheus()


@router.get("/v1/metrics/json")
async def metrics_json(request: Request) -> dict:
    state = get_state(request)
    return {"metrics": state.metrics.snapshot(),
            "scheduler": {"completed": state.scheduler.completed, "failed": state.scheduler.failed,
                          "pending": state.scheduler.pending}}


@router.get("/v1/usage")
async def get_usage(request: Request, window: str = "7d",
                    tenant_id: str | None = None,
                    principal: Principal = Depends(who)) -> dict:
    """按租户聚合窗口内的 run 数、token 与成本；普通用户只能看自己租户。"""
    state = get_state(request)
    effective = effective_tenant(principal, tenant_id)
    summary = await asyncio.to_thread(
        state.runs.usage_summary,
        tenant_id=effective,
        since=time.time() - _parse_window(window),
    )
    summary["window"] = window
    return summary


@router.post("/v1/admin/prune")
async def admin_prune(request: Request, days: int | None = None,
                      principal: Principal = Depends(who)) -> dict:
    """手动触发事件清理（仅管理员）；长驻进程配合定时任务使用。"""
    state = get_state(request)
    if not principal.is_admin:
        raise ApiError("需要管理员权限", code="forbidden", status=403)
    keep = days if days is not None else state.settings.event_retention_days
    if keep <= 0:
        return {"deleted": 0, "retention_days": keep}
    deleted = await asyncio.to_thread(
        state.event_store.prune, before_ts=time.time() - keep * 86400
    )
    return {"deleted": deleted, "retention_days": keep}
