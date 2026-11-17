"""内置工具集。

每个工具都是"契约先行"：Pydantic 参数模型 + 明确 scope + 超时 + 幂等/副作用声明。
"""
from __future__ import annotations

import ast
import asyncio
import math
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..errors import HermesError
from .base import ToolContext, ToolResult, ToolSpec
from .sandbox import SandboxPolicy, SandboxRunner


# ---------------------------------------------------------------- 参数模型
class CalculatorArgs(BaseModel):
    expression: str = Field(..., description="要计算的数学表达式，如 (3+4)*2、sqrt(16)、log(2.718)", max_length=200)


class ClockArgs(BaseModel):
    tz_offset_hours: int = Field(0, description="相对 UTC 的小时偏移，例如中国为 8", ge=-12, le=14)


class KBSearchArgs(BaseModel):
    query: str = Field(..., description="检索问题或关键词", min_length=1, max_length=200)
    top_k: int = Field(4, description="返回条数", ge=1, le=10)
    acl_group: str | None = Field(None, description="需要访问的权限分组，例如 finance")


class MemoryRecallArgs(BaseModel):
    query: str = Field("", description="检索记忆的查询语句；留空则返回最近的记忆", max_length=200)
    top_k: int = Field(4, description="返回条数", ge=1, le=10)
    subject_type: str | None = Field(None, description="业务对象类型，如 customer", max_length=40)
    subject_id: str | None = Field(None, description="业务对象 ID，如 c_001", max_length=120)


class MemoryWriteArgs(BaseModel):
    key: str = Field(..., description="记忆的键，如 profile.city", min_length=1, max_length=64)
    value: str = Field(..., description="记忆的值", min_length=1, max_length=500)
    memory_type: str = Field("fact", description="记忆类型：fact / preference / profile")
    subject_type: str | None = Field(None, description="业务对象类型，如 customer", max_length=40)
    subject_id: str | None = Field(None, description="业务对象 ID，如 c_001", max_length=120)


class SandboxArgs(BaseModel):
    code: str = Field(..., description="要执行的 Python 代码（仅标准库白名单）", max_length=8000)
    timeout_s: float = Field(5.0, description="执行超时秒数", ge=0.5, le=30)


class KBIngestArgs(BaseModel):
    path: str = Field(..., description="待入库的本地文件路径（pdf/md/txt/html/docx）", max_length=500)
    doc_id: str | None = Field(None, description="自定义文档 ID，默认取文件名", max_length=120)
    acl_group: str | None = Field(None, description="需要限制到某个权限分组时填写", max_length=64)


# ---------------------------------------------------------------- 安全计算器
_ALLOWED_FUNCS: dict[str, Any] = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "floor": math.floor, "ceil": math.ceil,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def safe_eval(expression: str) -> float:
    """AST 白名单求值，替代原项目里的 ``eval``（原实现可被构造参数逃逸）。"""
    if len(expression) > 200:
        raise ValueError("表达式过长")
    tree = ast.parse(expression, mode="eval")

    def _ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("只支持数字常量")
            return float(node.value)
        if isinstance(node, ast.BinOp):
            left, right = _ev(node.left), _ev(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                if abs(right) > 100 or abs(left) > 1e6:
                    raise ValueError("幂运算超出安全范围")
                return left ** right
            raise ValueError("不支持的运算符")
        if isinstance(node, ast.UnaryOp):
            value = _ev(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func = _ALLOWED_FUNCS.get(node.func.id)
            if func is None:
                raise ValueError(f"不支持的函数: {node.func.id}")
            return float(func(*[_ev(arg) for arg in node.args]))
        if isinstance(node, ast.Name) and node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"不支持的表达式节点: {type(node).__name__}")

    return _ev(tree)


# ---------------------------------------------------------------- 工具实现
def build_default_tools(
    *,
    knowledge: Any | None = None,
    memory: Any | None = None,
    sandbox: SandboxRunner | None = None,
    tool_timeout_s: float = 15.0,
) -> list[ToolSpec]:
    sandbox = sandbox or SandboxRunner()

    async def calculator(args: CalculatorArgs, ctx: ToolContext) -> str:
        result = safe_eval(args.expression)
        formatted = int(result) if float(result).is_integer() else round(result, 6)
        return f"{args.expression} = {formatted}"

    async def clock(args: ClockArgs, ctx: ToolContext) -> str:
        ts = time.time() + args.tz_offset_hours * 3600
        return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts)) + f" (UTC{args.tz_offset_hours:+d})"

    async def kb_search(args: KBSearchArgs, ctx: ToolContext) -> ToolResult:
        if knowledge is None:
            return ToolResult(ok=False, code="not_configured", error="知识库未接入")
        groups = {s.split(":", 2)[2] for s in ctx.scopes if s.startswith("kb:group:")}
        if args.acl_group:
            groups.add(args.acl_group)
        hits = knowledge.search(args.query, top_k=args.top_k, acl_groups=groups or None)
        if not hits:
            return ToolResult(ok=True, content="知识库中没有检索到相关内容。",
                              meta={"citations": []})
        content = knowledge.render_context(hits, wrap=False)
        return ToolResult(ok=True, content=content, meta={"citations": knowledge.citations(hits)})

    async def kb_ingest(args: KBIngestArgs, ctx: ToolContext) -> ToolResult:
        if knowledge is None:
            return ToolResult(ok=False, code="not_configured", error="知识库未接入")
        try:
            report = await asyncio.to_thread(
                knowledge.ingest_path, Path(args.path), doc_id=args.doc_id,
                acl=[args.acl_group] if args.acl_group else [],
            )
        except HermesError as exc:
            return ToolResult(ok=False, code=exc.code, error=exc.message)
        return ToolResult(
            ok=True,
            content=f"已入库 {report.chunks} 个分块（doc_id={report.doc_id}，{report.chars} 字符）",
            meta={"doc_id": report.doc_id, "doc_ids": report.doc_ids,
                  "chunks": report.chunks, "chars": report.chars},
        )

    async def memory_recall(args: MemoryRecallArgs, ctx: ToolContext) -> ToolResult:
        if memory is None:
            return ToolResult(ok=False, code="not_configured", error="记忆服务未接入")
        if args.query:
            recalled = memory.recall(tenant_id=ctx.tenant_id, user_id=ctx.user_id,
                                     query=args.query, top_k=args.top_k,
                                     subject_type=args.subject_type, subject_id=args.subject_id)
            if not recalled:
                return ToolResult(ok=True, content="没有找到相关长期记忆。", meta={"memories": []})
            return ToolResult(
                ok=True,
                content="\n".join(f"- {m.record.mkey}: {m.record.value}" for m in recalled),
                meta={"memories": [m.to_dict() for m in recalled]},
            )
        records = memory.list_memories(tenant_id=ctx.tenant_id, user_id=ctx.user_id, status="active",
                                       subject_type=args.subject_type, subject_id=args.subject_id)
        content = "\n".join(f"- {r.mkey}: {r.value}" for r in records) or "当前没有已确认的长期记忆。"
        return ToolResult(ok=True, content=content, meta={"memories": [r.to_dict() for r in records]})

    async def memory_write(args: MemoryWriteArgs, ctx: ToolContext) -> ToolResult:
        if memory is None:
            return ToolResult(ok=False, code="not_configured", error="记忆服务未接入")
        from ..memory.extractor import MemoryCandidate

        created = memory.propose(
            tenant_id=ctx.tenant_id, user_id=ctx.user_id,
            candidates=[MemoryCandidate(mkey=args.key, value=args.value, mtype=args.memory_type, confidence=0.95)],
            source_run_id=ctx.run_id, subject_type=args.subject_type, subject_id=args.subject_id,
        )
        if not created:
            return ToolResult(ok=True, content="该记忆已存在，未重复写入。")
        record = memory.confirm(created[0].id, tenant_id=ctx.tenant_id, user_id=ctx.user_id)
        return ToolResult(ok=True, content=f"已记住：{record.mkey} = {record.value}",
                          meta={"memory_id": record.id})

    async def run_python(args: SandboxArgs, ctx: ToolContext) -> ToolResult:
        result = await sandbox.run_python(args.code, SandboxPolicy(timeout_s=args.timeout_s))
        if result.rejected_reason:
            return ToolResult(ok=False, code="sandbox_rejected", error=result.rejected_reason)
        if not result.ok:
            return ToolResult(ok=False, code="sandbox_failed",
                              error=(result.stderr or "执行失败").strip()[:600],
                              meta={"stdout": result.stdout, "killed": result.killed})
        return ToolResult(ok=True, content=result.stdout.strip() or "（无输出）",
                          meta={"duration_ms": result.duration_ms})

    return [
        ToolSpec(
            name="kb_search",
            version="1.1",
            description="在企业知识库中做 BM25+向量混合检索。回答事实类问题前先调用它，结果带 [编号] 引用。",
            args_model=KBSearchArgs,
            handler=kb_search,
            scopes=("tool:read",),
            timeout_s=tool_timeout_s,
            idempotent=True,
            max_result_chars=3000,
            tags=("knowledge", "rag"),
        ),
        ToolSpec(
            name="kb_ingest",
            version="1.0",
            description="把本地文档（PDF/MD/TXT/HTML/DOCX）入库到知识库。属于写操作，需要用户确认后才会真正执行。",
            args_model=KBIngestArgs,
            handler=kb_ingest,
            scopes=("tool:write", "knowledge:write"),
            timeout_s=60.0,
            side_effect=True,
            requires_confirmation=True,
            max_result_chars=2000,
            tags=("knowledge", "write"),
        ),
        ToolSpec(
            name="memory_recall",
            version="1.0",
            description="召回当前用户的长期记忆（已确认的偏好、身份、历史事实）。",
            args_model=MemoryRecallArgs,
            handler=memory_recall,
            scopes=("tool:read",),
            timeout_s=5.0,
            idempotent=True,
            tags=("memory",),
        ),
        ToolSpec(
            name="memory_write",
            version="1.0",
            description="把用户明确提供的稳定信息写入长期记忆。属于写操作，需要用户确认后才会真正执行。",
            args_model=MemoryWriteArgs,
            handler=memory_write,
            scopes=("tool:read", "memory:write"),
            timeout_s=5.0,
            side_effect=True,
            requires_confirmation=True,
            tags=("memory", "write"),
        ),
        ToolSpec(
            name="calculator",
            version="1.1",
            description="安全地计算数学表达式（AST 白名单，不使用 eval）。",
            args_model=CalculatorArgs,
            handler=calculator,
            scopes=("tool:read",),
            timeout_s=3.0,
            idempotent=True,
            tags=("utility",),
        ),
        ToolSpec(
            name="clock",
            version="1.0",
            description="获取当前日期与时间。",
            args_model=ClockArgs,
            handler=clock,
            scopes=("tool:read",),
            timeout_s=2.0,
            idempotent=True,
            tags=("utility",),
        ),
        ToolSpec(
            name="python_sandbox",
            version="1.0",
            description="在隔离子进程中执行只读 Python 片段（标准库白名单、超时强杀、输出限长）。",
            args_model=SandboxArgs,
            handler=run_python,
            scopes=("tool:read", "tool:code"),
            timeout_s=20.0,
            tags=("sandbox",),
        ),
    ]
