"""工具契约。

设计要点（对应 JD 第 3 条）：
- 参数用 Pydantic 模型声明，模型调用前强校验，非法参数不会进入执行体；
- 每个工具带 version / scopes（最小权限）/ timeout / idempotent / side_effect / 是否需要确认；
- 结果统一为 ``ToolResult``，错误带稳定 code，模型和上层都能消费。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from pydantic import BaseModel

if TYPE_CHECKING:  # 避免循环导入
    from ..runtime.cancel import CancelToken


@dataclass
class ToolContext:
    run_id: str = ""
    tenant_id: str = "default"
    user_id: str = "anonymous"
    scopes: frozenset[str] = frozenset({"tool:read"})
    cancel: "CancelToken | None" = None
    deadline: float | None = None
    confirmed: bool = False
    dry_run: bool = False

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    error: str | None = None
    code: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1
    cached: bool = False
    needs_confirmation: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """渲染成给模型看的 tool message。"""
        if self.ok:
            return self.content
        return f"ERROR[{self.code or 'tool_error'}]: {self.error or self.content}"

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "content": self.content,
            "error": self.error,
            "code": self.code,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "cached": self.cached,
            "needs_confirmation": self.needs_confirmation,
        }


ToolHandler = Callable[[Any, ToolContext], Awaitable[str | ToolResult] | str | ToolResult]


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler
    version: str = "1.0"
    scopes: tuple[str, ...] = ("tool:read",)
    timeout_s: float | None = None
    max_retries: int = 0
    idempotent: bool = False
    side_effect: bool = False
    requires_confirmation: bool = False
    max_result_chars: int = 4000
    tags: tuple[str, ...] = ()

    def schema(self) -> dict:
        params = self.args_model.model_json_schema()
        _strip_titles(params)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }


def _strip_titles(node: Any) -> None:
    """部分厂商对 JSON Schema 里的 title 兼容性差，这里递归去掉（保留 $defs/$ref）。"""
    if isinstance(node, dict):
        node.pop("title", None)
        for value in list(node.values()):
            _strip_titles(value)
    elif isinstance(node, list):
        for item in node:
            _strip_titles(item)


def arguments_preview(arguments: dict) -> str:
    return json.dumps(arguments, ensure_ascii=False, default=str)
