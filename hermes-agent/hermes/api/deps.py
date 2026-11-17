"""路由共享依赖：状态获取、身份解析、租户/权限校验。

所有路由统一用 ``Depends(who)`` 拿 ``Principal``，再通过 ``effective_*`` 把
凭证收敛成真正的租户/用户/scope；越权一律 403，不存在与无权访问统一 404。
"""
from __future__ import annotations

from fastapi import Request

from ..errors import ApiError
from ..security.auth import Principal, resolve_principal
from ..triggers import Trigger
from .state import AppState, build_state


def get_state(request: Request) -> AppState:
    """取当前应用状态；没有 lifespan 时按 settings/override 现场构建，方便测试注入。"""
    state = getattr(request.app.state, "hermes", None)
    if state is None:
        settings = getattr(request.app.state, "hermes_settings", None)
        state = build_state(settings, getattr(request.app.state, "hermes_llm_override", None))
        request.app.state.hermes = state
    return state


def who(request: Request) -> Principal:
    state = get_state(request)
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


def visible_trigger(state: AppState, trigger_id: str, principal: Principal) -> Trigger:
    try:
        trigger = state.triggers.get(trigger_id)
    except KeyError:
        raise ApiError("触发器不存在", code="trigger_not_found", status=404) from None
    if not principal.is_admin and trigger.tenant_id != principal.tenant_id:
        raise ApiError("触发器不存在", code="trigger_not_found", status=404)
    return trigger
