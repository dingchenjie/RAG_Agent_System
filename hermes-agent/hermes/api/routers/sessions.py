"""会话域路由：会话创建/列表与消息回放。"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from ...errors import ApiError
from ...security.auth import Principal
from ..deps import effective_tenant, effective_user, get_state, who

router = APIRouter()


class CreateSessionBody(BaseModel):
    tenant_id: str = "default"
    user_id: str = "anonymous"
    title: str | None = None


@router.post("/v1/sessions")
async def create_session(body: CreateSessionBody, request: Request,
                         principal: Principal = Depends(who)) -> dict:
    session = get_state(request).sessions.create_session(
        tenant_id=effective_tenant(principal, body.tenant_id),
        user_id=effective_user(principal, body.user_id), title=body.title)
    return {"session_id": session.id, "tenant_id": session.tenant_id, "user_id": session.user_id}


@router.get("/v1/sessions")
async def list_sessions(request: Request, tenant_id: str | None = Query(None),
                        user_id: str | None = Query(None),
                        limit: int = Query(20, ge=1, le=100),
                        principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    tenant = effective_tenant(principal, tenant_id)
    # 普通调用方只能看自己的会话；管理员可以按 user_id 过滤，不传则看整个租户
    user_filter = user_id if principal.is_admin else principal.user_id
    sessions = await asyncio.to_thread(
        state.sessions.list_sessions, tenant_id=tenant, user_id=user_filter, limit=limit)
    return {"items": [{"session_id": s.id, "tenant_id": s.tenant_id, "user_id": s.user_id,
                       "title": s.title, "created_at": s.created_at,
                       "updated_at": s.updated_at} for s in sessions]}


@router.get("/v1/sessions/{session_id}/messages")
async def session_messages(session_id: str, request: Request,
                           limit: int = Query(200, ge=1, le=500),
                           principal: Principal = Depends(who)) -> dict:
    state = get_state(request)
    session = state.sessions.get_session(session_id)
    if session is None or (not principal.is_admin
                           and session.tenant_id != principal.tenant_id):
        raise ApiError("session 不存在", code="session_not_found", status=404)
    messages = await asyncio.to_thread(state.sessions.load_messages, session_id, limit)
    return {"session_id": session_id, "items": [m.to_dict() for m in messages]}
