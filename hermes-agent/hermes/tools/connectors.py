"""声明式连接器：把已有的 HTTP 接口与 CLI 命令，声明成 ToolSpec。

设计目标：接一个新接口是"写配置"，不是"改核心代码"，并且自动获得
ToolRegistry 的九道关（权限、校验、超时、重试、截断、审计），不产生第二套执行链路。

安全边界（有意为之）：
- HTTP 只允许 https（或显式 allow_http），host 必须精确命中 allowed_hosts，禁跟随重定向；
- CLI 命令由运营方在配置里固定，模型只能填 schema 声明的参数值，且不经 shell；
- 子进程只继承最小环境变量，不把整份 os.environ 交给外部命令。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from ..config import Settings
from .base import ToolContext, ToolResult, ToolSpec
from .mcp_bridge import model_from_json_schema
from .registry import ToolRegistry

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
_PLACEHOLDER_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
_CLI_RESULT_LIMIT = 8 * 1024
_CLI_BASE_ENV = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT")


def _expand_env(value: str, where: str) -> str:
    """把 ${ENV_NAME} 展开成环境变量；缺失直接失败，避免带着空密钥发请求。"""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"{where} 引用了未设置的环境变量 {name}")
        return os.environ[name]

    return _ENV_RE.sub(replace, value)


def load_connector_defs(raw: str) -> list[dict[str, Any]]:
    """解析并校验 HERMES_CONNECTORS；空字符串表示不启用任何连接器。"""
    if not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"HERMES_CONNECTORS 不是合法 JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError("HERMES_CONNECTORS 必须是 JSON 数组")

    defs: list[dict[str, Any]] = []
    for index, item in enumerate(parsed):
        where = f"连接器[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where} 必须是对象")
        name = str(item.get("name") or "")
        if not _NAME_RE.match(name):
            raise ValueError(f"{where} 的 name 非法：{name!r}（小写字母开头，允许数字与下划线）")
        kind = str(item.get("kind") or "")
        if kind not in {"http", "cli"}:
            raise ValueError(f"{where} 的 kind 必须是 http 或 cli")
        if not str(item.get("description") or "").strip():
            raise ValueError(f"{where} 缺少 description")

        if kind == "http":
            url = str(item.get("url") or "")
            if not url:
                raise ValueError(f"{where} 缺少 url")
            allowed = item.get("allowed_hosts") or []
            if not isinstance(allowed, list) or not allowed:
                raise ValueError(f"{where} 必须显式声明 allowed_hosts")
            parsed_url = urlsplit(url)
            if parsed_url.username or parsed_url.password:
                raise ValueError(f"{where} 的 url 不允许内嵌凭据，请用 headers + 环境变量")
            if parsed_url.scheme == "http" and not item.get("allow_http"):
                raise ValueError(f"{where} 的 url 是 http；如确需明文，请显式设置 allow_http=true")
            if parsed_url.scheme not in {"http", "https"}:
                raise ValueError(f"{where} 的 url 必须使用 http(s)")
            if parsed_url.netloc not in allowed:
                raise ValueError(f"{where} 的 host 未在 allowed_hosts 中声明")
            method = str(item.get("method") or "GET").upper()
            if method not in _HTTP_METHODS:
                raise ValueError(f"{where} 的 method 不受支持：{method}")
            headers = item.get("headers") or {}
            if not isinstance(headers, dict):
                raise ValueError(f"{where} 的 headers 必须是对象")
            item["resolved_headers"] = {
                str(key): _expand_env(str(value), f"{where}.headers.{key}")
                for key, value in headers.items()
            }
        else:
            command = item.get("command")
            if not isinstance(command, list) or not command or not all(isinstance(c, str) for c in command):
                raise ValueError(f"{where} 的 command 必须是非空字符串数组")
            cwd = Path(str(item.get("cwd") or ".")).resolve()
            root = Path.cwd().resolve()
            if root not in cwd.parents and cwd != root:
                raise ValueError(f"{where} 的 cwd 必须位于项目目录内")
            if not cwd.is_dir():
                raise ValueError(f"{where} 的 cwd 不存在：{cwd}")
            item["resolved_cwd"] = str(cwd)
            extra_env = item.get("env") or []
            if not isinstance(extra_env, list) or not all(isinstance(k, str) for k in extra_env):
                raise ValueError(f"{where} 的 env 必须是环境变量名数组")
            item["resolved_env"] = {
                key: os.environ[key] for key in extra_env if key in os.environ
            }
        defs.append(item)
    return defs


def _fill_url(template: str, values: dict[str, Any], consumed: set[str]) -> str:
    """只替换 {name} 形式的占位符，参数值做 URL 编码，避免路径注入。"""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values or values[key] is None:
            raise ValueError(f"url 占位符缺少参数：{key}")
        consumed.add(key)
        return quote(str(values[key]), safe="")

    return _PLACEHOLDER_RE.sub(replace, template)


def _build_http_handler(defn: dict[str, Any], transport: httpx.AsyncBaseTransport | None):
    method = str(defn.get("method") or "GET").upper()
    url_template = str(defn["url"])
    allowed_hosts = set(defn.get("allowed_hosts") or [])
    headers = dict(defn.get("resolved_headers") or defn.get("headers") or {})
    query_keys = list(defn.get("query") or [])
    body_keys = list(defn.get("body") or [])
    max_bytes = int(defn.get("max_response_bytes") or 262144)
    timeout_s = float(defn.get("timeout_s") or 10.0)
    allow_http = bool(defn.get("allow_http"))

    async def handler(args: Any, ctx: ToolContext) -> ToolResult:
        values = args.model_dump(exclude_none=True)
        consumed: set[str] = set()
        url = _fill_url(url_template, values, consumed)

        parsed = urlsplit(url)
        if parsed.scheme == "http" and not allow_http:
            return ToolResult(ok=False, code="connector_rejected", error="禁止明文 http 请求")
        if parsed.netloc not in allowed_hosts:
            return ToolResult(
                ok=False, code="connector_rejected",
                error=f"目标 host 未授权: {parsed.netloc}",
            )

        rest = {key: value for key, value in values.items() if key not in consumed}
        params = {key: rest[key] for key in query_keys if key in rest}
        payload = {key: rest[key] for key in body_keys if key in rest}
        remainder = {key: value for key, value in rest.items() if key not in query_keys + body_keys}
        if remainder:
            (params if method == "GET" else payload).update(remainder)

        kwargs: dict[str, Any] = {
            "params": params or None,
            "headers": headers or None,
            "timeout": timeout_s,
            "follow_redirects": False,
        }
        if method != "GET" and payload:
            kwargs["json"] = payload

        try:
            async with httpx.AsyncClient(transport=transport) as client:
                async with client.stream(method, url, **kwargs) as response:
                    chunks: list[bytes] = []
                    total = 0
                    truncated = False
                    async for chunk in response.aiter_bytes():
                        if total + len(chunk) > max_bytes:
                            chunks.append(chunk[: max_bytes - total])
                            truncated = True
                            break
                        chunks.append(chunk)
                        total += len(chunk)
                    body = b"".join(chunks)
                    status = response.status_code
        except httpx.TimeoutException:
            return ToolResult(ok=False, code="tool_timeout", error=f"连接器请求超时（>{timeout_s}s）")
        except httpx.HTTPError as exc:
            return ToolResult(ok=False, code="connector_error", error=f"请求失败: {type(exc).__name__}")

        text = body.decode("utf-8", errors="replace").strip()
        if not 200 <= status < 300:
            return ToolResult(ok=False, code=f"http_{status}", error=text[:300] or f"HTTP {status}")

        content = text
        try:
            content = json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            pass
        return ToolResult(
            ok=True, content=content,
            meta={"status": status, "truncated": truncated, "connector": defn["name"]},
        )

    return handler, timeout_s + 2


def _build_cli_handler(defn: dict[str, Any]):
    command = [str(part) for part in defn["command"]]
    cwd = str(defn.get("resolved_cwd") or Path(str(defn.get("cwd") or ".")).resolve())
    env_extra = dict(defn.get("resolved_env") or {})
    arg_keys = list(defn.get("args") or [])
    timeout_s = float(defn.get("timeout_s") or 10.0)

    async def handler(args: Any, ctx: ToolContext) -> ToolResult:
        values = args.model_dump(exclude_none=True)
        argv = list(command)
        for key in arg_keys:
            if key not in values:
                continue
            if isinstance(values[key], bool):
                if values[key]:
                    argv.append(f"--{key}")
                continue
            argv.extend([f"--{key}", str(values[key])])
        unknown = [key for key in values if key not in arg_keys]
        if unknown:
            return ToolResult(ok=False, code="invalid_arguments",
                              error=f"未声明的参数: {', '.join(unknown)}")

        env = {key: os.environ[key] for key in _CLI_BASE_ENV if key in os.environ}
        env.update(env_extra)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv, cwd=cwd, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return ToolResult(ok=False, code="connector_error", error=f"命令启动失败: {exc}")

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ToolResult(ok=False, code="cli_timeout", error=f"命令超时（>{timeout_s}s）已终止")

        out = stdout.decode("utf-8", errors="replace")[:_CLI_RESULT_LIMIT].strip()
        err = stderr.decode("utf-8", errors="replace")[:_CLI_RESULT_LIMIT].strip()
        if process.returncode != 0:
            return ToolResult(
                ok=False, code=f"cli_exit_{process.returncode}",
                error=(err or out or f"命令退出码 {process.returncode}")[:300],
                meta={"returncode": process.returncode},
            )
        return ToolResult(ok=True, content=out or "（无输出）",
                          meta={"returncode": 0, "connector": defn["name"]})

    return handler, timeout_s + 2


def build_connector_tools(
    defs: list[dict[str, Any]], *, http_transport: httpx.AsyncBaseTransport | None = None,
) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for defn in defs:
        name = str(defn["name"])
        kind = str(defn["kind"])
        schema = defn.get("args_schema") or {"type": "object", "properties": {}}
        if kind == "http":
            handler, timeout_s = _build_http_handler(defn, http_transport)
            default_idempotent = str(defn.get("method") or "GET").upper() == "GET"
        else:
            handler, timeout_s = _build_cli_handler(defn)
            default_idempotent = False
        specs.append(ToolSpec(
            name=name,
            version=str(defn.get("version") or "1.0"),
            description=str(defn["description"]),
            args_model=model_from_json_schema(name, schema),
            handler=handler,
            scopes=tuple(defn.get("scopes") or ("tool:read",)),
            timeout_s=timeout_s,
            max_retries=int(defn.get("max_retries") or 0),
            idempotent=bool(defn.get("idempotent", default_idempotent)),
            side_effect=bool(defn.get("side_effect") or False),
            requires_confirmation=bool(defn.get("requires_confirmation") or False),
            max_result_chars=int(defn.get("max_result_chars") or 4000),
            tags=("connector", kind),
        ))
    return specs


def register_connectors(registry: ToolRegistry, settings: Settings,
                        *, http_transport: httpx.AsyncBaseTransport | None = None) -> list[str]:
    """把 HERMES_CONNECTORS 里的声明注册进同一个 ToolRegistry，返回注册成功的工具名。"""
    specs = build_connector_tools(load_connector_defs(settings.connectors),
                                  http_transport=http_transport)
    for spec in specs:
        registry.register(spec)
    return [spec.name for spec in specs]
