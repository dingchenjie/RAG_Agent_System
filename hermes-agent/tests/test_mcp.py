"""MCP：自建 stdio Server 被客户端桥接进注册表，并走统一的权限/审计/事件链路。"""
from __future__ import annotations

import asyncio
import sys

import pytest

from hermes.api.app import build_state
from hermes.llm.mock import MockLLM
from hermes.tools.base import ToolContext
from hermes.tools.mcp_bridge import MCPToolBridge
from tests.conftest import ingest_seed_docs


def test_mcp_bridge_exposes_remote_tools(settings, monkeypatch):
    state = build_state(settings, MockLLM())
    ingest_seed_docs(state)
    # 让子进程连到同一个库，验证"跨进程也能复用同一份知识与记忆"
    monkeypatch.setenv("HERMES_DB_PATH", settings.db_path)

    bridge = MCPToolBridge(state.registry, command=sys.executable,
                           args=["-m", "hermes.mcp_server.server"], server_label="kb")

    async def scenario():
        try:
            names = await bridge.start()
        except Exception as exc:  # 环境不支持子进程 stdio 时跳过，不掩盖其他失败
            await bridge.stop()
            pytest.skip(f"MCP stdio 不可用：{exc}")
        try:
            assert any(name.startswith("mcp__kb__") for name in names)
            result = await state.registry.execute(
                "mcp__kb__kb_search", {"query": "LoRA 的秩 r"},
                ToolContext(run_id="mcp-run", scopes=frozenset({"tool:read"})),
            )
            return result
        finally:
            await bridge.stop()

    result = asyncio.run(scenario())
    assert result.ok, result.error
    assert "8" in result.content

    rows = state.db.query("SELECT * FROM tool_audit WHERE tool LIKE 'mcp__kb__%'")
    assert rows, "MCP 工具调用同样要留审计"


def test_mcp_tool_respects_scopes(settings, monkeypatch):
    state = build_state(settings, MockLLM())
    ingest_seed_docs(state)
    monkeypatch.setenv("HERMES_DB_PATH", settings.db_path)
    bridge = MCPToolBridge(state.registry, command=sys.executable,
                           args=["-m", "hermes.mcp_server.server"], server_label="kb")

    async def scenario():
        try:
            await bridge.start()
        except Exception as exc:
            await bridge.stop()
            pytest.skip(f"MCP stdio 不可用：{exc}")
        try:
            denied = await state.registry.execute(
                "mcp__kb__kb_search", {"query": "LoRA"},
                ToolContext(run_id="mcp-run2", scopes=frozenset({"other:scope"})))
            await bridge.stop()
            return denied
        except Exception:
            await bridge.stop()
            raise

    result = asyncio.run(scenario())
    assert not result.ok
    assert result.code == "permission_denied"
