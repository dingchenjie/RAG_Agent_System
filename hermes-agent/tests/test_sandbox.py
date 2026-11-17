"""代码执行沙箱行为：静态白名单、超时终止、输出截断、内存上限、Docker 参数。"""
from __future__ import annotations

import asyncio
import sys
import time

import pytest

from hermes.tools.sandbox import SandboxPolicy, SandboxResult, SandboxRunner, build_docker_args


def _run(code: str, policy: SandboxPolicy | None = None, **runner_kwargs) -> SandboxResult:
    """同步包装：用 asyncio.run 驱动异步接口，避免依赖 pytest-asyncio 插件。"""
    runner = SandboxRunner(**runner_kwargs)
    return asyncio.run(runner.run_python(code, policy))


# --------------------------------------------------------------- 正常执行
def test_正常执行_print_返回标准输出():
    result = _run('print("hello sandbox")')

    assert result.ok is True
    assert result.stdout.strip() == "hello sandbox"
    assert result.exit_code == 0
    assert result.killed is False
    assert result.rejected_reason is None


# --------------------------------------------------------------- 静态白名单
@pytest.mark.parametrize("module", ["os", "socket"])
def test_未授权导入被拒绝(module: str):
    result = _run(f"import {module}")

    assert result.ok is False
    assert result.rejected_reason is not None
    assert f"禁止导入模块: {module}" in result.rejected_reason
    assert result.stdout == ""
    assert result.exit_code is None


@pytest.mark.parametrize("code", ['print("x".__class__)', "print((1).__mro__)"])
def test_魔术属性访问被拒绝(code: str):
    result = _run(code)

    assert result.ok is False
    assert result.rejected_reason is not None
    assert "魔术属性" in result.rejected_reason


# --------------------------------------------------------------- 超时
def test_死循环在超时后被杀死():
    started = time.perf_counter()
    result = _run("while True:\n    pass", SandboxPolicy(timeout_s=0.5))
    elapsed = time.perf_counter() - started

    assert result.ok is False
    assert result.killed is True
    assert "超时" in result.stderr
    assert elapsed < 10  # 杀进程 + 回收要快，不能拖住调用方


# --------------------------------------------------------------- 输出截断
def test_输出超过上限被截断():
    result = _run('print("x" * 10000)', SandboxPolicy(max_output_chars=100))

    assert result.ok is True
    assert len(result.stdout) == 100
    assert result.stdout == "x" * 100


# --------------------------------------------------------------- POSIX 内存上限
@pytest.mark.skipif(sys.platform == "win32", reason="Windows 没有 resource 模块，内存上限不生效")
def test_posix_内存上限生效():
    code = "buf = bytearray(256 * 1024 * 1024)\nprint(len(buf))"
    result = _run(code, SandboxPolicy(max_memory_mb=64, timeout_s=10))

    assert result.ok is False
    assert result.killed is True or result.exit_code != 0


# --------------------------------------------------------------- Docker 参数
def test_build_docker_args_默认为无网络只读():
    args = build_docker_args(SandboxPolicy(max_memory_mb=128), "python:3.12-slim")

    assert args[:2] == ["docker", "run"]
    assert "--rm" in args
    assert "-i" in args
    assert args[args.index("--network") + 1] == "none"
    assert args[args.index("--memory") + 1] == "128m"
    assert "--cpus" in args
    assert "--pids-limit" in args
    assert "--read-only" in args
    assert args[args.index("--tmpfs") + 1] == "/tmp:rw,size=16m"
    assert args[args.index("--user") + 1] == "65534:65534"
    # 镜像之后固定以 python -I -S - 从 stdin 读代码
    assert args[-4:] == ["python", "-I", "-S", "-"]
    assert args[-5] == "python:3.12-slim"


def test_build_docker_args_允许网络时切换为_bridge():
    args = build_docker_args(SandboxPolicy(allow_network=True), "python:3.12-slim")

    assert args[args.index("--network") + 1] == "bridge"
    assert "none" not in args


# --------------------------------------------------------------- Docker 执行
class _FakeProc:
    """伪进程：记录 stdin 内容并返回预设输出。"""

    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.pid = 4242
        self.received: bytes | None = None

    async def communicate(self, input: bytes | None = None):
        self.received = input
        return self._stdout, self._stderr


def test_docker_模式通过_stdin_传代码并映射结果(monkeypatch):
    proc = _FakeProc(stdout=b"hello\n")
    captured: dict[str, object] = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    result = _run('print("hello")', mode="docker")

    assert result.ok is True
    assert result.stdout == "hello\n"
    assert result.exit_code == 0
    assert proc.received == b'print("hello")'
    assert captured["args"][0] == "docker"
    assert captured["args"][-1] == "-"
    assert captured["kwargs"]["stdin"] is asyncio.subprocess.PIPE


def test_docker_不可用时返回失败而不抛异常(monkeypatch):
    async def fake_exec(*args, **kwargs):
        raise FileNotFoundError("docker 未安装")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    result = _run("print(1)", mode="docker")

    assert result.ok is False
    assert "Docker" in result.stderr
    assert result.exit_code is None
