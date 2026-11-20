"""工具注册表：参数校验、超时、重试、结果截断、审计、页面预算，一处收口。"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..store import Store
from ..trace import Tracer


@dataclass
class ToolContext:
    store: Store
    run_id: str
    tracer: Tracer
    settings: Settings
    pages_used: int = 0
    pages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pages_left(self) -> int:
        return max(0, self.settings.max_pages_per_run - self.pages_used)


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[[ToolContext, dict[str, Any]], Awaitable[dict[str, Any]]]
    timeout_s: float = 20.0
    max_retries: int = 2
    max_result_chars: int = 4000

    def to_openai(self) -> dict[str, Any]:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.schema}}


class ToolRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai() for t in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        spec = self._tools.get(name)
        if spec is None:
            return {"ok": False, "error": f"未知工具: {name}", "code": "unknown_tool"}

        missing = [k for k in (spec.schema.get("required") or []) if k not in arguments]
        if missing:
            return {"ok": False, "error": f"缺少参数: {', '.join(missing)}", "code": "invalid_arguments"}

        if name in ("fetch_page", "browse_page") and ctx.pages_left <= 0:
            return {"ok": False, "code": "page_budget_exhausted",
                    "error": f"本次调研最多抓取 {self.settings.max_pages_per_run} 个页面"}

        started = asyncio.get_event_loop().time()
        ctx.tracer.emit("tool.started", tool=name, arguments=arguments)
        error: str | None = None
        result: dict[str, Any] = {}
        for attempt in range(spec.max_retries + 1):
            try:
                result = await asyncio.wait_for(spec.handler(ctx, arguments),
                                                timeout=spec.timeout_s)
                error = None
                break
            except asyncio.TimeoutError:
                error = f"工具超时（>{spec.timeout_s}s）"
                result = {"ok": False, "error": error, "code": "timeout", "retryable": True}
            except Exception as exc:  # 工具异常不能打穿主循环
                error = f"{type(exc).__name__}: {exc}"
                result = {"ok": False, "error": error, "code": "tool_exception",
                          "retryable": False}
            if not result.get("retryable") or attempt >= spec.max_retries:
                break
            await asyncio.sleep(0.5 * (attempt + 1))

        payload = json.dumps(result, ensure_ascii=False)
        if len(payload) > spec.max_result_chars:
            result = _shrink(result, spec.max_result_chars)
        elapsed = round(asyncio.get_event_loop().time() - started, 3)
        ctx.tracer.emit("tool.finished", tool=name, elapsed_s=elapsed, ok=result.get("ok", False),
                        error=error, result=result)
        return result


def _shrink(result: dict[str, Any], limit: int) -> dict[str, Any]:
    """超长时只裁长文本字段，保留 source_id/url/title 等结构化字段。

    早期实现把整个结果换成 preview 字符串，模型因此拿不到引用编号——
    看起来"工具调用成功"，实际证据全部丢失。
    """
    payload = json.dumps(result, ensure_ascii=False)
    if len(payload) <= limit:
        return result
    for key in ("text", "formatted"):
        value = result.get(key)
        if isinstance(value, str):
            keep = max(0, len(value) - (len(payload) - limit) - 40)
            result = {**result, key: value[:keep], "truncated": True}
            payload = json.dumps(result, ensure_ascii=False)
            if len(payload) <= limit:
                return result
    return {"ok": result.get("ok", False), "truncated": True, "preview": payload[:limit]}


def check_loop(history: list[tuple[str, str]], name: str, arguments: dict[str, Any],
               threshold: int = 2) -> bool:
    """同一工具 + 同一参数重复出现 threshold 次，判定为原地打转。"""
    signature = (name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
    history.append(signature)
    return history.count(signature) >= threshold
