"""工具层：参数校验、权限、确认、幂等、超时、沙箱。"""
from __future__ import annotations

import asyncio

from pydantic import BaseModel

from hermes.tools.base import ToolContext, ToolSpec
from hermes.tools.builtin import build_default_tools, safe_eval
from hermes.tools.registry import ToolRegistry
from hermes.tools.sandbox import SandboxPolicy, SandboxRunner


def _registry(**kwargs) -> ToolRegistry:
    registry = ToolRegistry(default_timeout_s=1.0, default_retries=0, **kwargs)
    registry.register_many(build_default_tools())
    return registry


def test_safe_eval_blocks_escape():
    assert safe_eval("(3+4)*2") == 14
    for bad in ["__import__('os').system('echo hi')", "eval('1+1')", "2**9999", "open('/etc/passwd')"]:
        try:
            safe_eval(bad)
        except Exception:
            continue
        raise AssertionError(f"应当拒绝: {bad}")


def test_invalid_arguments_and_permission():
    registry = _registry()
    ctx = ToolContext(scopes=frozenset({"tool:read"}))
    result = asyncio.run(registry.execute("calculator", {}, ctx))
    assert result.code == "invalid_arguments"

    denied = asyncio.run(registry.execute("memory_write", {"key": "a", "value": "b"}, ctx))
    assert denied.code == "permission_denied"

    unknown = asyncio.run(registry.execute("nope", {}, ctx))
    assert unknown.code == "unknown_tool"


def test_confirmation_flow_and_write(settings):
    from hermes.api.app import build_state
    from hermes.llm.mock import MockLLM

    state = build_state(settings, MockLLM())
    ctx = ToolContext(scopes=frozenset({"tool:read", "memory:write"}))
    pending = asyncio.run(state.registry.execute("memory_write", {"key": "profile.city", "value": "南京"}, ctx))
    assert pending.needs_confirmation is True
    assert pending.ok is True

    ctx.confirmed = True
    done = asyncio.run(state.registry.execute("memory_write", {"key": "profile.city", "value": "南京"}, ctx))
    assert done.ok is True
    records = state.memory.list_memories(tenant_id="default", user_id="anonymous", status="active")
    assert any(r.mkey == "profile.city" and r.value == "南京" for r in records)


def test_idempotent_tool_uses_cache():
    registry = _registry()
    ctx = ToolContext()
    first = asyncio.run(registry.execute("calculator", {"expression": "2+2"}, ctx))
    second = asyncio.run(registry.execute("calculator", {"expression": "2+2"}, ctx))
    assert first.ok and second.ok
    assert second.cached is True


def test_timeout_is_enforced():
    class SlowArgs(BaseModel):
        seconds: float = 1.0

    async def slow(args, ctx):
        await asyncio.sleep(args.seconds)
        return "done"

    registry = ToolRegistry(default_timeout_s=0.1, default_retries=0)
    registry.register(ToolSpec(name="slow", description="slow", args_model=SlowArgs,
                               handler=slow, timeout_s=0.1))
    result = asyncio.run(registry.execute("slow", {"seconds": 1}, ToolContext()))
    assert result.ok is False
    assert result.code == "tool_timeout"


def test_sandbox_executes_and_blocks_import():
    runner = SandboxRunner()
    ok = asyncio.run(runner.run_python("print(6*7)", SandboxPolicy(timeout_s=10)))
    assert ok.ok and ok.stdout.strip() == "42"

    rejected = asyncio.run(runner.run_python("import os\nprint(os.listdir('.'))"))
    assert rejected.ok is False
    assert "禁止导入" in (rejected.rejected_reason or "")


def test_sandbox_kills_infinite_loop():
    runner = SandboxRunner()
    result = asyncio.run(runner.run_python("while True:\n    pass", SandboxPolicy(timeout_s=1.0)))
    assert result.ok is False
    assert result.killed is True
