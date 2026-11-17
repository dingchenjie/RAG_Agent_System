"""调用方身份与权限解析。

设计原则：**权限不能由被授权方声明**。客户端只能出示凭证（API Key），
租户、用户、scope 一律由服务端根据凭证解析；请求体里的同名字段最多用于
管理员"降权演示"，不允许任何形式的升权。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..errors import ApiError

AUTH_MODES = ("off", "api_key")


@dataclass(frozen=True)
class Principal:
    """一次请求的身份快照。"""

    tenant_id: str
    user_id: str
    scopes: frozenset[str]
    is_admin: bool = False

    def has_scope(self, scope: str) -> bool:
        if self.is_admin or "*" in self.scopes or scope in self.scopes:
            return True
        domain = scope.split(":", 1)[0]
        return f"{domain}:*" in self.scopes


def parse_api_keys(raw: str) -> dict[str, Principal]:
    """解析 ``HERMES_API_KEYS``。

    格式：``key1:tenantA:alice:tool:read|knowledge:read;key2:tenantB:bob:*``
    - 条目之间用 ``;``，字段之间用 ``:``（只切前三个冒号，scope 本身含冒号）
    - scope 之间用 ``|``；``*`` 表示管理员（跨租户）
    """
    registry: dict[str, Principal] = {}
    for entry in (raw or "").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":", 3)
        if len(parts) < 3:
            raise RuntimeError(f"HERMES_API_KEYS 格式错误：{entry!r}（需 key:tenant:user[:scopes]）")
        key, tenant_id, user_id = parts[0], parts[1], parts[2]
        raw_scopes = parts[3] if len(parts) > 3 else "tool:read"
        scopes = frozenset(s.strip() for s in raw_scopes.split("|") if s.strip())
        if not key or not tenant_id or not user_id or not scopes:
            raise RuntimeError(f"HERMES_API_KEYS 条目字段为空：{entry!r}")
        registry[key] = Principal(tenant_id=tenant_id, user_id=user_id, scopes=scopes,
                                  is_admin="*" in scopes)
    if not registry:
        raise RuntimeError("HERMES_AUTH_MODE=api_key 时 HERMES_API_KEYS 不能为空")
    return registry


def _bearer(authorization: str | None) -> str:
    if not authorization:
        return ""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return ""
    return token.strip()


def resolve_principal(authorization: str | None, *, mode: str, raw_keys: str) -> Principal:
    """把 `Authorization` 头解析成 Principal；失败直接抛 401。"""
    if mode == "off":
        # 本地演示/测试/CI：视为单租户管理员，启动时会打 warning
        return Principal(tenant_id="default", user_id="anonymous",
                         scopes=frozenset({"*"}), is_admin=True)
    if mode not in AUTH_MODES:
        raise RuntimeError(f"未知的 HERMES_AUTH_MODE={mode!r}，可选 {AUTH_MODES}")

    token = _bearer(authorization)
    if not token:
        raise ApiError("缺少 Bearer Token", code="unauthorized", status=401)
    principal = parse_api_keys(raw_keys).get(token)
    if principal is None:
        raise ApiError("API Key 无效", code="unauthorized", status=401)
    return principal
