"""工具注册表：权限、校验、幂等、超时、重试、审计、结果截断的唯一入口。

执行顺序（缺一不可）：
    unknown -> permission -> schema validate -> dry-run/confirm
    -> idempotency cache -> invoke(timeout) -> retry -> truncate -> audit
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
from typing import Any

from pydantic import ValidationError

from ..errors import ToolError
from ..observability.metrics import Metrics
from ..security.redact import redact_obj
from ..storage.db import Database
from .base import ToolContext, ToolResult, ToolSpec


class ToolRegistry:
    def __init__(
        self,
        *,
        db: Database | None = None,
        metrics: Metrics | None = None,
        default_timeout_s: float = 15.0,
        default_retries: int = 2,
    ):
        self._tools: dict[str, ToolSpec] = {}
        self.db = db
        self.metrics = metrics or Metrics()
        self.default_timeout_s = default_timeout_s
        self.default_retries = default_retries
        self._memory_idem: dict[str, str] = {}

    # ---------- 注册 ----------
    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"工具重复注册: {spec.name}")
        self._tools[spec.name] = spec

    def register_many(self, specs: list[ToolSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def all_specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def schemas(self, scopes: frozenset[str] | None = None) -> list[dict]:
        return [spec.schema() for spec in self._tools.values() if self._allowed(spec, scopes)]

    def subset(self, names: tuple[str, ...] | list[str], *, extra_scopes: frozenset[str] | None = None) -> ToolView:
        return ToolView(self, frozenset(names), extra_scopes)

    @staticmethod
    def _allowed(spec: ToolSpec, scopes: frozenset[str] | None) -> bool:
        if scopes is None or "*" in scopes:  # "*" 表示管理员，等价于全部 scope
            return True
        return set(spec.scopes).issubset(scopes)

    # ---------- 执行 ----------
    async def execute(self, name: str, arguments: dict, ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(ok=False, code="unknown_tool", error=f"未注册的工具: {name}",
                              latency_ms=_ms(started))

        if not self._allowed(spec, ctx.scopes):
            missing = sorted(set(spec.scopes) - set(ctx.scopes))
            self.metrics.inc("hermes_tool_denied_total", tool=name)
            return ToolResult(ok=False, code="permission_denied",
                              error=f"缺少权限: {', '.join(missing)}", latency_ms=_ms(started))

        try:
            args = spec.args_model.model_validate(arguments)
        except ValidationError as exc:
            self.metrics.inc("hermes_tool_invalid_args_total", tool=name)
            return ToolResult(ok=False, code="invalid_arguments",
                              error=f"参数校验失败: {exc.errors(include_url=False)}",
                              latency_ms=_ms(started))

        # 预演 / 人工确认：有副作用的工具不允许"先斩后奏"
        if spec.requires_confirmation and not ctx.confirmed:
            preview = json.dumps(args.model_dump(), ensure_ascii=False, default=str)
            return ToolResult(
                ok=True,
                content=f"待确认操作：{name} {preview}",
                needs_confirmation=True,
                meta={"tool": name, "arguments": args.model_dump(), "dry_run": ctx.dry_run},
                latency_ms=_ms(started),
            )

        idem_key = self._idem_key(spec, args, ctx.run_id) if spec.idempotent else None
        if idem_key:
            cached = self._idem_get(idem_key)
            if cached is not None:
                self.metrics.inc("hermes_tool_cache_hit_total", tool=name)
                return ToolResult(ok=True, content=cached, cached=True, latency_ms=_ms(started))

        timeout = spec.timeout_s or self.default_timeout_s
        attempts_allowed = 1 + max(spec.max_retries, self.default_retries if spec.max_retries else 0)
        result: ToolResult | None = None
        last_error: ToolError | None = None

        for attempt in range(1, attempts_allowed + 1):
            if ctx.cancel is not None:
                ctx.cancel.raise_if_cancelled()
            try:
                raw = await asyncio.wait_for(self._invoke(spec, args, ctx), timeout=timeout)
                result = _coerce_result(raw)
                break
            except asyncio.TimeoutError:
                last_error = ToolError(f"工具执行超时（>{timeout}s）", code="tool_timeout", retryable=True)
            except ToolError as exc:
                last_error = exc
                if not exc.retryable:
                    break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 边界处兜底，避免异常击穿 runtime
                last_error = ToolError(f"工具内部异常: {exc}", code="tool_error", retryable=False)
                break

            if attempt < attempts_allowed:
                await asyncio.sleep(min(1.0, 0.05 * (2 ** (attempt - 1))))

        if result is None:
            last_error = last_error or ToolError("工具执行失败", code="tool_error")
            self.metrics.inc("hermes_tool_failed_total", tool=name, code=last_error.code)
            result = ToolResult(ok=False, code=last_error.code, error=last_error.message,
                                attempts=attempts_allowed)

        if len(result.content) > spec.max_result_chars:
            result.content = result.content[: spec.max_result_chars] + "\n…[工具结果已截断]"

        result.latency_ms = _ms(started)
        self.metrics.inc("hermes_tool_calls_total", tool=name, ok=str(result.ok).lower())
        self.metrics.observe("hermes_tool_latency_ms", result.latency_ms, tool=name)

        if idem_key and result.ok:
            self._idem_put(idem_key, result.content)
        self._audit(spec, arguments, result, ctx)
        return result

    async def _invoke(self, spec: ToolSpec, args: Any, ctx: ToolContext) -> Any:
        out = spec.handler(args, ctx)
        if inspect.isawaitable(out):
            return await out
        return out

    def _idem_key(self, spec: ToolSpec, args: Any, run_id: str = "") -> str:
        """幂等键按 run 作用域：只消除同一 run 内的重复调用（模型重试、重复工具调用）。

        不能跨 run 复用：读类工具（知识库、记忆、时钟）的结果会随时间变化，
        跨 run 命中旧缓存会返回过期事实——这类"缓存幻觉"比性能问题更危险。
        """
        payload = json.dumps(args.model_dump(), sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(f"{spec.name}|{spec.version}|{run_id}|{payload}".encode()).hexdigest()

    def _idem_get(self, key: str) -> str | None:
        if self.db is not None:
            row = self.db.query_one("SELECT payload FROM idempotency WHERE key=? AND kind='tool'", (key,))
            return json.loads(row["payload"])["content"] if row else None
        return self._memory_idem.get(key)

    def _idem_put(self, key: str, content: str) -> None:
        payload = json.dumps({"content": content}, ensure_ascii=False)
        if self.db is not None:
            self.db.execute(
                "INSERT OR REPLACE INTO idempotency(key, kind, payload, created_at) VALUES (?, 'tool', ?, ?)",
                (key, payload, time.time()),
            )
        else:
            self._memory_idem[key] = content

    def _audit(self, spec: ToolSpec, arguments: dict, result: ToolResult, ctx: ToolContext) -> None:
        if self.db is None:
            return
        try:
            self.db.execute(
                "INSERT INTO tool_audit(run_id, tool, version, arguments, ok, error_code, latency_ms, attempts, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ctx.run_id,
                    spec.name,
                    spec.version,
                    json.dumps(redact_obj(arguments), ensure_ascii=False, default=str)[:2000],
                    1 if result.ok else 0,
                    result.code,
                    result.latency_ms,
                    result.attempts,
                    time.time(),
                ),
            )
        except Exception:
            # 审计失败不能影响主流程
            pass


class ToolView:
    """子 Agent 的工具视图：只暴露被授权的工具与 scope，实现最小权限。"""

    def __init__(self, parent: ToolRegistry, names: frozenset[str], extra_scopes: frozenset[str] | None):
        self.parent = parent
        self.names = names
        self.extra_scopes = extra_scopes

    def schemas(self, scopes: frozenset[str] | None = None) -> list[dict]:
        effective = (scopes or frozenset()) | (self.extra_scopes or frozenset())
        return [
            spec.schema()
            for spec in self.parent.all_specs()
            if spec.name in self.names and ToolRegistry._allowed(spec, effective)
        ]

    def get(self, name: str) -> ToolSpec | None:
        # 版本等元信息不受子 Agent 权限影响；未授权工具仍返回 None，避免泄露存在性
        return self.parent.get(name) if name in self.names else None

    async def execute(self, name: str, arguments: dict, ctx: ToolContext) -> ToolResult:
        if name not in self.names:
            return ToolResult(ok=False, code="permission_denied", error=f"子 Agent 未获授权使用工具: {name}")
        if self.extra_scopes:
            ctx.scopes = ctx.scopes | self.extra_scopes
        return await self.parent.execute(name, arguments, ctx)


def _coerce_result(raw: Any) -> ToolResult:
    if isinstance(raw, ToolResult):
        return raw
    if isinstance(raw, str):
        return ToolResult(ok=True, content=raw)
    if raw is None:
        return ToolResult(ok=True, content="")
    return ToolResult(ok=True, content=json.dumps(raw, ensure_ascii=False, default=str))


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
