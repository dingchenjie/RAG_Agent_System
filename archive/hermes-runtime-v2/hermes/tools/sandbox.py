"""代码执行沙箱。

诚实说明边界：这是一个**子进程隔离 + 静态白名单**的方案，不是内核级沙箱。
它能挡住：越权 import、``__dunder__`` 逃逸、死循环、无限输出；
它挡不住：精心构造的 CPython 解释器漏洞。生产环境的正确做法是容器 + 只读挂载
+ 无网络 + seccomp/gVisor。README 里会明确写清这个边界。
"""
from __future__ import annotations

import ast
import asyncio
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_IMPORTS = {
    "math", "json", "statistics", "itertools", "collections",
    "datetime", "re", "decimal", "fractions", "typing", "random",
}
FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "input", "__import__", "globals", "locals", "vars", "getattr", "setattr", "delattr"}


class UnsafeCode(ValueError):
    pass


@dataclass
class SandboxPolicy:
    timeout_s: float = 5.0
    max_output_chars: int = 4000
    allowed_imports: frozenset[str] = field(default_factory=lambda: frozenset(ALLOWED_IMPORTS))
    allow_network: bool = False
    max_memory_mb: int = 256


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: float = 0.0
    killed: bool = False
    rejected_reason: str | None = None


def validate_code(code: str, policy: SandboxPolicy) -> None:
    if len(code) > 8000:
        raise UnsafeCode("代码过长（>8000 字符）")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCode(f"语法错误: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in policy.allowed_imports:
                    raise UnsafeCode(f"禁止导入模块: {root}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in policy.allowed_imports:
                raise UnsafeCode(f"禁止导入模块: {root}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise UnsafeCode(f"禁止访问魔术属性: {node.attr}")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise UnsafeCode(f"禁止使用名称: {node.id}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise UnsafeCode("禁止使用 global/nonlocal")


class SandboxRunner:
    """用独立解释器进程执行用户代码。"""

    def __init__(self, default_policy: SandboxPolicy | None = None):
        self.default_policy = default_policy or SandboxPolicy()

    async def run_python(self, code: str, policy: SandboxPolicy | None = None) -> SandboxResult:
        policy = policy or self.default_policy
        started = time.perf_counter()
        try:
            validate_code(code, policy)
        except UnsafeCode as exc:
            return SandboxResult(ok=False, rejected_reason=str(exc),
                                 duration_ms=round((time.perf_counter() - started) * 1000, 2))

        with tempfile.TemporaryDirectory(prefix="hermes_sbx_") as tmp:
            script = Path(tmp) / "snippet.py"
            script.write_text(code, encoding="utf-8")
            env = {
                "PATH": "",
                "PYTHONPATH": "",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HOME": tmp,
                "TMPDIR": tmp,
            }
            if sys.platform == "win32":
                import os
                env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
                env["PATH"] = os.environ.get("PATH", "")

            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", "-S", str(script),
                cwd=tmp, env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            killed = False
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=policy.timeout_s)
            except asyncio.TimeoutError:
                killed = True
                await _kill_tree(proc)
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=2)
                except asyncio.TimeoutError:
                    stdout, stderr = b"", b""

            out = stdout.decode("utf-8", errors="replace")[: policy.max_output_chars]
            err = stderr.decode("utf-8", errors="replace")[: policy.max_output_chars]
            duration = round((time.perf_counter() - started) * 1000, 2)
            if killed:
                return SandboxResult(ok=False, stdout=out, stderr="执行超时，进程已被终止", killed=True,
                                     exit_code=proc.returncode, duration_ms=duration)
            return SandboxResult(ok=proc.returncode == 0, stdout=out, stderr=err,
                                 exit_code=proc.returncode, duration_ms=duration)


async def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    if sys.platform == "win32":
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/F", "/T", "/PID", str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        except Exception:
            pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass
