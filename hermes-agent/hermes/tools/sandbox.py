"""代码执行沙箱。

诚实说明边界：默认后端是**子进程隔离 + 静态白名单**的方案，不是内核级沙箱。
它能挡住：越权 import、``__dunder__`` 逃逸、死循环、无限输出、内存/CPU 滥用；
它挡不住：精心构造的 CPython 解释器漏洞。生产环境的正确做法是容器 + 只读挂载
+ 无网络 + seccomp/gVisor，因此这里也提供可选的 ``docker`` 后端。
README 里会明确写清这个边界。
"""
from __future__ import annotations

import ast
import asyncio
import functools
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


def build_docker_args(policy: SandboxPolicy, image: str) -> list[str]:
    """构造 ``docker run`` 命令行（纯函数，便于单测）。

    代码通过 stdin 传给容器，容器内用 ``python -I -S -`` 从标准输入执行；
    默认无网络、只读根文件系统、非 root 用户、限制内存/CPU/进程数。
    """
    network = "bridge" if policy.allow_network else "none"
    return [
        "docker", "run", "--rm", "-i",
        "--network", network,
        "--memory", f"{policy.max_memory_mb}m",
        "--cpus", "0.5",
        "--pids-limit", "64",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=16m",
        "--user", "65534:65534",
        image,
        "python", "-I", "-S", "-",
    ]


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _apply_posix_limits(policy: SandboxPolicy) -> None:
    """在子进程 exec 之前施加 POSIX 资源上限（Windows 上不做任何事）。

    只能通过 ``preexec_fn`` 在 fork 之后、exec 之前调用。单项设置失败不影响
    其余限制，避免因平台差异（如某些容器禁改 hard limit）导致整个子进程起不来。
    """
    if sys.platform == "win32":
        return
    import resource  # 仅 POSIX 提供该模块

    memory_bytes = max(1, policy.max_memory_mb) * 1024 * 1024
    cpu_seconds = max(1, int(policy.timeout_s) + 1)
    limits = (
        (resource.RLIMIT_AS, memory_bytes),
        (resource.RLIMIT_CPU, cpu_seconds),
        (resource.RLIMIT_FSIZE, 1024 * 1024),
        (resource.RLIMIT_NOFILE, 32),
    )
    for which, value in limits:
        try:
            resource.setrlimit(which, (value, value))
        except (OSError, ValueError):
            pass


class SandboxRunner:
    """用独立解释器进程（或 Docker 容器）执行用户代码。"""

    def __init__(
        self,
        default_policy: SandboxPolicy | None = None,
        mode: str = "process",
        docker_image: str = "python:3.12-slim",
    ):
        if mode not in {"process", "docker"}:
            raise ValueError(f"未知的沙箱模式: {mode}（可选 process/docker）")
        self.default_policy = default_policy or SandboxPolicy()
        self.mode = mode
        self.docker_image = docker_image

    async def run_python(self, code: str, policy: SandboxPolicy | None = None) -> SandboxResult:
        policy = policy or self.default_policy
        started = time.perf_counter()
        try:
            validate_code(code, policy)
        except UnsafeCode as exc:
            return SandboxResult(ok=False, rejected_reason=str(exc), duration_ms=_elapsed_ms(started))

        if self.mode == "docker":
            return await self._run_docker(code, policy, started)
        return await self._run_process(code, policy, started)

    async def _run_process(self, code: str, policy: SandboxPolicy, started: float) -> SandboxResult:
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

            extra: dict[str, object] = {}
            if sys.platform != "win32":
                # POSIX：exec 前施加内存 / CPU / 文件大小 / 句柄数上限
                extra["preexec_fn"] = functools.partial(_apply_posix_limits, policy)

            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", "-S", str(script),
                cwd=tmp, env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **extra,
            )
            return await _collect(proc, policy, started, "执行超时，进程已被终止")

    async def _run_docker(self, code: str, policy: SandboxPolicy, started: float) -> SandboxResult:
        args = build_docker_args(policy, self.docker_image)
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:  # Docker 未安装 / 守护进程未启动
            return SandboxResult(ok=False, stderr=f"Docker 后端不可用: {exc}", duration_ms=_elapsed_ms(started))

        return await _collect(proc, policy, started, "执行超时，容器已被终止", stdin_data=code.encode("utf-8"))


async def _collect(
    proc: asyncio.subprocess.Process,
    policy: SandboxPolicy,
    started: float,
    timeout_message: str,
    stdin_data: bytes | None = None,
) -> SandboxResult:
    """等待子进程结束并按策略截断输出；超时则杀进程树。"""
    killed = False
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(stdin_data), timeout=policy.timeout_s)
    except asyncio.TimeoutError:
        killed = True
        await _kill_tree(proc)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=2)
        except asyncio.TimeoutError:
            stdout, stderr = b"", b""

    out = stdout.decode("utf-8", errors="replace")[: policy.max_output_chars]
    err = stderr.decode("utf-8", errors="replace")[: policy.max_output_chars]
    duration = _elapsed_ms(started)
    if killed:
        return SandboxResult(ok=False, stdout=out, stderr=timeout_message, killed=True,
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
