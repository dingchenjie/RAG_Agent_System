"""MCP 客户端桥接：把外部 MCP Server 的工具接进本地注册表。

一个 MCP 工具进入 Runtime 后，与内置工具没有区别：
同样的参数校验、权限、超时、审计、事件流。这就是"连接器体系"的价值。
"""
from __future__ import annotations

import json
import os
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import Field, create_model

from .base import ToolContext, ToolResult, ToolSpec
from .registry import ToolRegistry

_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _sanitize(name: str) -> str:
    # 部分厂商不允许 function name 含点号
    return name.replace(".", "__").replace("-", "_")


def model_from_json_schema(tool_name: str, schema: dict[str, Any]) -> type:
    fields: dict[str, Any] = {}
    for prop, spec in (schema.get("properties") or {}).items():
        py_type = _TYPE_MAP.get(spec.get("type", "string"), str)
        description = spec.get("description", "")
        if "default" in spec:
            fields[prop] = (py_type, Field(default=spec["default"], description=description))
        elif prop in (schema.get("required") or []):
            fields[prop] = (py_type, Field(..., description=description))
        else:
            fields[prop] = (py_type | None, Field(default=None, description=description))
    return create_model(f"{tool_name}_Args", **fields)


class MCPToolBridge:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        command: str = "python",
        args: list[str] | None = None,
        server_label: str = "hermes",
        scopes: tuple[str, ...] = ("tool:read",),
        env: dict[str, str] | None = None,
    ):
        self.registry = registry
        self.command = command
        self.args = args or ["-m", "hermes.mcp_server.server"]
        self.server_label = server_label
        self.scopes = scopes
        # MCP SDK 默认只透传一份"安全"环境变量，业务配置（DB 路径、租户等）必须显式传入
        self.env = {**os.environ, **(env or {})}
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self.registered: list[str] = []

    async def start(self) -> list[str]:
        params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        self._stack = AsyncExitStack()
        read_stream, write_stream = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self._session.initialize()
        listing = await self._session.list_tools()

        for tool in listing.tools:
            await self._register(tool)
        return self.registered

    async def _register(self, tool: Any) -> None:
        remote_name = tool.name
        local_name = _sanitize(f"mcp__{self.server_label}__{remote_name}")
        args_model = model_from_json_schema(local_name, tool.inputSchema or {})
        session = self._session

        async def handler(args: Any, ctx: ToolContext) -> ToolResult:
            if session is None:
                return ToolResult(ok=False, code="mcp_disconnected", error="MCP 会话已关闭")
            response = await session.call_tool(remote_name, args.model_dump(exclude_none=True))
            texts = []
            for item in response.content:
                text = getattr(item, "text", None)
                if text:
                    texts.append(text)
            content = "\n".join(texts) if texts else json.dumps({"ok": True}, ensure_ascii=False)
            if getattr(response, "isError", False):
                return ToolResult(ok=False, code="mcp_tool_error", error=content)
            return ToolResult(ok=True, content=content, meta={"mcp": self.server_label, "remote_tool": remote_name})

        spec = ToolSpec(
            name=local_name,
            version="mcp-1",
            description=f"[MCP:{self.server_label}] {tool.description or remote_name}",
            args_model=args_model,
            handler=handler,
            scopes=self.scopes,
            timeout_s=20.0,
            tags=("mcp", self.server_label),
        )
        self.registry.register(spec)
        self.registered.append(local_name)

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None
