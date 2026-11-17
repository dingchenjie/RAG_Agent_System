"""触发器域路由：周期任务的创建、列表与启停/删除。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from ...errors import ApiError
from ...security.auth import Principal
from ..deps import effective_scopes, effective_tenant, effective_user, get_state, require_scope, visible_trigger, who

router = APIRouter()


class CreateTriggerBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    input: str = Field(..., min_length=1, max_length=4000)
    interval_s: int = Field(60, ge=10, le=86400, description="触发周期（秒），最小 10")
    mode: str = Field("single", description="single | orchestrate")
    tenant_id: str = "default"
    user_id: str = "anonymous"
    scopes: list[str] | None = None
    session_id: str | None = None


@router.post("/v1/triggers", status_code=201)
async def create_trigger(body: CreateTriggerBody, request: Request,
                         principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
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


@router.get("/v1/triggers")
async def list_triggers(request: Request, tenant_id: str | None = Query(default=None),
                        principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    require_scope(principal, "trigger:read")
    tenant = effective_tenant(principal, tenant_id)
    return {"triggers": [t.to_dict() for t in state.triggers.list(tenant_id=tenant)]}


@router.post("/v1/triggers/{trigger_id}/enable")
async def enable_trigger(trigger_id: str, request: Request,
                         principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    require_scope(principal, "trigger:write")
    visible_trigger(state, trigger_id, principal)
    state.triggers.set_enabled(trigger_id, True)
    return {"trigger_id": trigger_id, "enabled": True}


@router.post("/v1/triggers/{trigger_id}/disable")
async def disable_trigger(trigger_id: str, request: Request,
                          principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    require_scope(principal, "trigger:write")
    visible_trigger(state, trigger_id, principal)
    state.triggers.set_enabled(trigger_id, False)
    return {"trigger_id": trigger_id, "enabled": False}


@router.delete("/v1/triggers/{trigger_id}")
async def delete_trigger(trigger_id: str, request: Request,
                         principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    require_scope(principal, "trigger:write")
    visible_trigger(state, trigger_id, principal)
    state.triggers.remove(trigger_id)
    return {"trigger_id": trigger_id, "deleted": True}
