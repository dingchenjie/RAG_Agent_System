"""记忆域路由：候选记忆的列出、提出、确认、纠正与删除。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ...errors import ApiError
from ...security.auth import Principal
from ..deps import effective_tenant, effective_user, get_state, who

router = APIRouter()


class ProposeMemoryBody(BaseModel):
    text: str
    tenant_id: str = "default"
    user_id: str = "anonymous"
    subject_type: str | None = Field(None, max_length=40)
    subject_id: str | None = Field(None, max_length=120)


class CorrectMemoryBody(BaseModel):
    value: str
    reason: str | None = None


@router.get("/v1/memory")
async def list_memory(request: Request, tenant_id: str = "default", user_id: str = "anonymous",
                      status: str | None = None,
                      subject_type: str | None = None, subject_id: str | None = None,
                      principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    tenant_id = effective_tenant(principal, tenant_id)
    user_id = effective_user(principal, user_id)
    records = state.memory.list_memories(
        tenant_id=tenant_id, user_id=user_id, status=status,
        subject_type=subject_type, subject_id=subject_id)
    return {"items": [r.to_dict() for r in records], "stats": state.memory.stats(
        tenant_id=tenant_id, user_id=user_id)}


@router.post("/v1/memory/propose")
async def propose_memory(body: ProposeMemoryBody, request: Request,
                         principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    created = await state.memory.propose_from_text_async(
        tenant_id=effective_tenant(principal, body.tenant_id),
        user_id=effective_user(principal, body.user_id), text=body.text,
        subject_type=body.subject_type, subject_id=body.subject_id)
    return {"items": [r.to_dict() for r in created]}


@router.post("/v1/memory/{memory_id}/confirm")
async def confirm_memory(memory_id: str, request: Request,
                         tenant_id: str = "default", user_id: str = "anonymous",
                         principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    try:
        record = state.memory.confirm(
            memory_id, tenant_id=effective_tenant(principal, tenant_id),
            user_id=effective_user(principal, user_id))
    except Exception as exc:
        raise ApiError(str(exc), code="memory_not_found", status=404) from exc
    return {"item": record.to_dict()}


@router.post("/v1/memory/{memory_id}/correct")
async def correct_memory(memory_id: str, body: CorrectMemoryBody, request: Request,
                         tenant_id: str = "default", user_id: str = "anonymous",
                         principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    try:
        record = state.memory.correct(
            memory_id, tenant_id=effective_tenant(principal, tenant_id),
            user_id=effective_user(principal, user_id),
            value=body.value, reason=body.reason)
    except Exception as exc:
        raise ApiError(str(exc), code="memory_not_found", status=404) from exc
    return {"item": record.to_dict()}


@router.delete("/v1/memory/{memory_id}")
async def delete_memory(memory_id: str, request: Request,
                        tenant_id: str = "default", user_id: str = "anonymous",
                        principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    try:
        state.memory.forget(memory_id,
                            tenant_id=effective_tenant(principal, tenant_id),
                            user_id=effective_user(principal, user_id))
    except Exception as exc:
        raise ApiError(str(exc), code="memory_not_found", status=404) from exc
    return {"deleted": memory_id}
