"""声明式连接器：HTTP / CLI 复用 ToolRegistry 的执行链路与安全边界。"""
from __future__ import annotations

import asyncio
import json
import sys

import httpx
import pytest

from hermes.tools.base import ToolContext
from hermes.tools.connectors import (
    _blocked_ip,
    build_connector_tools,
    load_connector_defs,
    register_connectors,
)
from hermes.tools.registry import ToolRegistry


def _http_def(**overrides) -> dict:
    base = {
        "name": "ticket_lookup",
        "kind": "http",
        "description": "查询工单",
        "method": "GET",
        "url": "https://api.example.com/v1/tickets/{ticket_id}",
        "allowed_hosts": ["api.example.com"],
        "args_schema": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}},
            "required": ["ticket_id"],
        },
    }
    base.update(overrides)
    return base


def _registry(specs) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_many(specs)
    return registry


def _ctx(scopes=frozenset({"tool:read"})) -> ToolContext:
    return ToolContext(run_id="run-1", scopes=scopes)


def test_load_rejects_bad_config():
    with pytest.raises(ValueError, match="name 非法"):
        load_connector_defs(json.dumps([_http_def(name="Bad-Name")]))
    with pytest.raises(ValueError, match="allowed_hosts"):
        load_connector_defs(json.dumps([_http_def(allowed_hosts=[])]))
    with pytest.raises(ValueError, match="http"):
        load_connector_defs(json.dumps([_http_def(url="http://api.example.com/x")]))
    with pytest.raises(ValueError, match="command"):
        load_connector_defs(json.dumps([{"name": "ls", "kind": "cli", "description": "x"}]))


def test_http_connector_returns_json_and_status():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/tickets/T-1"
        return httpx.Response(200, json={"status": "open", "id": "T-1"})

    specs = build_connector_tools([_http_def()], http_transport=httpx.MockTransport(handler))
    result = asyncio.run(_registry(specs).execute("ticket_lookup", {"ticket_id": "T-1"}, _ctx()))
    assert result.ok is True
    assert json.loads(result.content)["status"] == "open"
    assert result.meta["status"] == 200


def test_http_connector_surfaces_status_error():
    transport = httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
    specs = build_connector_tools([_http_def()], http_transport=transport)
    result = asyncio.run(_registry(specs).execute("ticket_lookup", {"ticket_id": "T-1"}, _ctx()))
    assert result.ok is False
    assert result.code == "http_500"
    assert "boom" in (result.error or "")


def test_http_connector_blocks_unlisted_host_before_sending():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, text="should-not-happen")

    # 绕过 load 的注册期校验，确认运行期仍会拒绝
    defn = _http_def(allowed_hosts=["other.example.com"])
    specs = build_connector_tools([defn], http_transport=httpx.MockTransport(handler))
    result = asyncio.run(_registry(specs).execute("ticket_lookup", {"ticket_id": "T-1"}, _ctx()))
    assert result.ok is False
    assert result.code == "connector_rejected"
    assert calls == []


def test_blocked_ip_classification():
    assert _blocked_ip("10.0.0.7") == "private"
    assert _blocked_ip("172.16.3.4") == "private"
    assert _blocked_ip("192.168.1.10") == "private"
    assert _blocked_ip("127.0.0.1") == "loopback"
    assert _blocked_ip("169.254.169.254") == "link_local"
    assert _blocked_ip("::1") == "loopback"
    assert _blocked_ip("fe80::1") == "link_local"
    assert _blocked_ip("::ffff:127.0.0.1") == "loopback"
    assert _blocked_ip("93.184.216.34") is None


def test_http_connector_rejects_private_dns(monkeypatch):
    async def fake_resolve(host: str, port: int) -> list[str]:
        return ["93.184.216.34", "10.0.0.5"]  # 混合解析：任一私网地址都拒绝

    monkeypatch.setattr("hermes.tools.connectors._resolve_addresses", fake_resolve)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, text="should-not-happen")

    # transport=None → 走真实网络路径的 DNS 校验；请求在发出去之前就被拦下
    specs = build_connector_tools([_http_def()])
    result = asyncio.run(_registry(specs).execute("ticket_lookup", {"ticket_id": "T-1"}, _ctx()))
    assert result.ok is False
    assert result.code == "connector_rejected"
    assert "10.0.0.5" in (result.error or "")
    assert calls == []


def test_http_connector_rejects_link_local_ipv6(monkeypatch):
    async def fake_resolve(host: str, port: int) -> list[str]:
        return ["fe80::1"]

    monkeypatch.setattr("hermes.tools.connectors._resolve_addresses", fake_resolve)
    specs = build_connector_tools([_http_def()])
    result = asyncio.run(_registry(specs).execute("ticket_lookup", {"ticket_id": "T-1"}, _ctx()))
    assert result.ok is False
    assert result.code == "connector_rejected"
    assert "link_local" in (result.error or "")


def test_injected_transport_skips_dns_lookup(monkeypatch):
    async def boom(host: str, port: int) -> list[str]:
        raise AssertionError("注入 transport 时不应触发 DNS 解析")

    monkeypatch.setattr("hermes.tools.connectors._resolve_addresses", boom)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="ok"))
    specs = build_connector_tools([_http_def()], http_transport=transport)
    result = asyncio.run(_registry(specs).execute("ticket_lookup", {"ticket_id": "T-1"}, _ctx()))
    assert result.ok is True


def test_connector_env_missing_fails_fast(monkeypatch):
    monkeypatch.delenv("TICKET_TOKEN_FOR_TEST", raising=False)
    defn = _http_def(headers={"Authorization": "Bearer ${TICKET_TOKEN_FOR_TEST}"})
    with pytest.raises(ValueError, match="TICKET_TOKEN_FOR_TEST"):
        load_connector_defs(json.dumps([defn]))


def test_cli_connector_runs_and_reports_exit_code():
    ok_def = {
        "name": "say_hello", "kind": "cli", "description": "输出 hello",
        "command": [sys.executable, "-c", "print('hello')"],
    }
    fail_def = {
        "name": "exit_three", "kind": "cli", "description": "退出码 3",
        "command": [sys.executable, "-c", "import sys; sys.exit(3)"],
    }
    registry = _registry(build_connector_tools([ok_def, fail_def]))
    ok = asyncio.run(registry.execute("say_hello", {}, _ctx()))
    assert ok.ok is True
    assert ok.content.strip() == "hello"

    failed = asyncio.run(registry.execute("exit_three", {}, _ctx()))
    assert failed.ok is False
    assert failed.code == "cli_exit_3"


def test_register_connectors_adds_schemas(settings):
    settings.connectors = json.dumps([{
        "name": "repo_status", "kind": "cli", "description": "仓库改动",
        "command": ["git", "status", "--porcelain"],
    }])
    registry = ToolRegistry()
    assert register_connectors(registry, settings) == ["repo_status"]
    assert "repo_status" in [spec["function"]["name"] for spec in registry.schemas()]


def test_connector_scope_is_enforced():
    specs = build_connector_tools([{
        "name": "admin_tool", "kind": "cli", "description": "需要写权限",
        "command": [sys.executable, "-c", "print(1)"], "scopes": ["tool:write"],
    }])
    result = asyncio.run(_registry(specs).execute("admin_tool", {}, _ctx()))
    assert result.ok is False
    assert result.code == "permission_denied"
