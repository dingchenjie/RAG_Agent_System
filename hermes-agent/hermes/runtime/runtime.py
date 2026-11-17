"""Agent Runtime 主循环。

一次 run 的生命周期：
    created -> running -> (paused -> running) -> succeeded / failed / cancelled

关键工程点（JD 第 1、2、5、6、7 条）：
- 结构化事件流：每一步都有事件，SSE/回放/评测共用同一份数据；
- 上下文工程：每步都重新构建上下文并记录预算报告；
- 取消：LLM 与工具调用都可被立即打断；
- 断点：每步落 checkpoint，进程重启后可 resume；
- 预演确认：有副作用的工具先暂停、等待确认再执行；
- 记忆：把本轮对话中的稳定事实抽取成 pending 记忆，确认后才生效。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..errors import BudgetExceeded, CancelledByUser, HermesError
from ..ids import new_run_id
from ..learning.lessons import LessonService
from ..llm.base import LLMProvider
from ..memory.service import MemoryService
from ..observability.events import Event, EventBus, EventType
from ..observability.metrics import Metrics
from ..observability.tracing import Tracer, UsageMeter, compute_cost
from ..security.untrusted import SAFETY_CLAUSE, wrap_untrusted
from ..tools.base import ToolContext
from ..types import Message
from .cancel import CancelToken, run_with_cancel
from .context import ContextBuilder, ContextReport
from .store import RunStore, SessionStore

DEFAULT_SYSTEM_PROMPT = (
    "你是 Hermes，一个严谨的企业级智能助手。你可以调用工具检索知识库、查询长期记忆、做数学计算、"
    "在受限沙箱中执行 Python。\n"
    "工作规则：\n"
    "1. 凡涉及具体事实、文档内容、论文细节、制度或政策条款的问题，第一轮必须先调用 kb_search，"
    "禁止凭自身记忆直接回答；检索结果为空或与问题无关时，直接回答\"知识库中没有相关依据\"，不要猜测；\n"
    "2. 答案里的每个事实都要能对应到检索结果，并在结论后保留 [编号] 引用；\n"
    "3. 同一个工具、同一组参数不要重复调用；拿到结果就继续推理，不要为了确认再查一遍；\n"
    "4. 问题同时包含\"查资料\"和\"算数\"时，先检索再计算，两步都必须做，不能跳过检索；\n"
    "5. 工具返回 ERROR 时先判断能否换一种参数重试，不要重复同样的失败调用；\n"
    "6. 需要写操作（如记住用户信息）时先说明将要做什么；\n"
    "7. 寒暄、自我介绍等与业务无关的问题直接回答，不要调用工具。"
) + SAFETY_CLAUSE

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    run_id: str
    state: str
    answer: str | None = None
    steps: int = 0
    usage: dict = field(default_factory=dict)
    citations: list[dict] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    pending: dict | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "state": self.state,
            "answer": self.answer,
            "steps": self.steps,
            "usage": self.usage,
            "citations": self.citations,
            "context": self.context,
            "pending": self.pending,
            "error": self.error,
        }


class AgentRuntime:
    def __init__(
        self,
        *,
        settings: Settings,
        sessions: SessionStore,
        runs: RunStore,
        bus: EventBus,
        registry: Any,                      # ToolRegistry | ToolView
        llm: LLMProvider,
        metrics: Metrics | None = None,
        memory: MemoryService | None = None,
        knowledge: Any | None = None,
        builder: ContextBuilder | None = None,
        lessons: LessonService | None = None,
        system_prompt: str | None = None,
        max_steps: int | None = None,
        agent_name: str = "main",
    ):
        self.settings = settings
        self.sessions = sessions
        self.runs = runs
        self.bus = bus
        self.registry = registry
        self.llm = llm
        self.metrics = metrics or Metrics()
        self.memory = memory
        self.knowledge = knowledge
        self.lessons = lessons
        self.builder = builder or ContextBuilder(settings.context)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_steps = max_steps or settings.max_steps
        self.agent_name = agent_name

    # ------------------------------------------------------------------ run
    async def run(
        self,
        *,
        user_input: str,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        session_id: str | None = None,
        scopes: frozenset[str] = frozenset({"tool:read"}),
        cancel: CancelToken | None = None,
        idempotency_key: str | None = None,
        run_id: str | None = None,
        extra_context: dict | None = None,
    ) -> RunResult:
        if idempotency_key:
            existing = self.runs.find_by_idempotency(idempotency_key)
            if existing:
                return self._result_from_row(existing)

        session = self.sessions.get_session(session_id) if session_id else None
        if session is None:
            session = self.sessions.create_session(tenant_id=tenant_id, user_id=user_id,
                                                   title=user_input[:40], session_id=session_id)
        tenant_id = session.tenant_id
        user_id = session.user_id

        run_id = run_id or new_run_id()
        if self.runs.get_run(run_id) is None:
            self.runs.create_run(session_id=session.id, tenant_id=tenant_id, user_id=user_id,
                                 user_input=user_input, idempotency_key=idempotency_key, run_id=run_id)
        else:
            # API 层为了先返回 run_id，会预先建一条 queued 记录
            self.runs.update(run_id, state="running")
        await self._emit(run_id, EventType.RUN_CREATED, {
            "session_id": session.id, "agent": self.agent_name, "input": user_input,
            "tenant_id": tenant_id, "user_id": user_id,
            "context": self._context_meta(extra_context),
        })
        await self._emit(run_id, EventType.RUN_STARTED, {"agent": self.agent_name})

        self.sessions.add_message(session.id, Message(role="user", content=user_input))
        history = self.sessions.load_messages(session.id)

        return await self._loop(
            run_id=run_id, session_id=session.id, tenant_id=tenant_id, user_id=user_id,
            user_input=user_input, history=history, start_step=0, scopes=scopes,
            cancel=cancel or CancelToken(), pending=None, extra_context=extra_context,
        )

    # --------------------------------------------------------------- resume
    async def resume(self, run_id: str, *, confirmed: bool = False,
                     cancel: CancelToken | None = None) -> RunResult:
        row = self.runs.get_run(run_id)
        if row is None:
            raise HermesError(f"run 不存在: {run_id}", code="not_found")
        checkpoint = self.runs.load_checkpoint(run_id)
        if checkpoint is None:
            raise HermesError(f"run 没有可恢复的 checkpoint: {run_id}", code="no_checkpoint")

        history = [Message.from_dict(m) for m in checkpoint.get("messages", [])]
        pending = checkpoint.get("pending")
        step = int(checkpoint.get("step", 0))
        await self._emit(run_id, EventType.RUN_RESUMED, {"step": step, "confirmed": confirmed,
                                                        "pending_tool": (pending or {}).get("name")})
        self.runs.update(run_id, state="running")

        return await self._loop(
            run_id=run_id, session_id=row["session_id"], tenant_id=row["tenant_id"],
            user_id=row["user_id"], user_input=row["input"], history=history, start_step=step,
            scopes=frozenset({"tool:read", "tool:write", "memory:write"}),
            cancel=cancel or CancelToken(), pending=pending, confirmed=confirmed,
            extra_context=checkpoint.get("context"),
        )

    # ----------------------------------------------------------------- core
    async def _loop(
        self,
        *,
        run_id: str,
        session_id: str,
        tenant_id: str,
        user_id: str,
        user_input: str,
        history: list[Message],
        start_step: int,
        scopes: frozenset[str],
        cancel: CancelToken,
        pending: dict | None,
        confirmed: bool = False,
        extra_context: dict | None = None,
    ) -> RunResult:
        tracer = Tracer(self.bus, run_id)
        meter = UsageMeter(max_tokens=self.settings.max_run_tokens,
                           max_cost_usd=self.settings.max_run_cost_usd)
        pending = pending or None

        # 本轮的知识检索与记忆召回只做一次，避免每步重复打 embedding
        injected: dict[str, str] = {}
        citations: list[dict] = []
        recalled_memories: list[dict] = []
        try:
            await self._prepare_context(user_input, tenant_id, user_id, scopes, injected,
                                        citations, recalled_memories, run_id)
        except Exception as exc:  # 检索失败不应阻塞对话
            await self._emit(run_id, EventType.KNOWLEDGE_RETRIEVED, {"error": str(exc), "degraded": True})

        # 页面信息/企业规则随请求进入本轮上下文，再交给统一的预算与裁剪逻辑
        rendered_context = self._render_run_context(extra_context)
        if rendered_context:
            injected["当前页面"] = rendered_context

        step = start_step
        answer: str | None = None
        state = "running"
        error: str | None = None
        context_report: ContextReport | None = None

        try:
            # 恢复时先执行被挂起的写操作
            if pending and confirmed:
                pending_spec = self.registry.get(pending["name"])
                result = await run_with_cancel(
                    self.registry.execute(
                        pending["name"], pending.get("arguments") or {},
                        ToolContext(run_id=run_id, tenant_id=tenant_id, user_id=user_id,
                                    scopes=scopes, cancel=cancel, confirmed=True),
                    ),
                    cancel,
                )
                history.append(Message(
                    role="tool",
                    content=self._truncate_tool(wrap_untrusted(
                        "tool_output", result.content,
                        {"tool": pending["name"], "agent": self.agent_name})),
                    tool_call_id=pending.get("id"), name=pending["name"]))
                self.sessions.add_message(session_id, history[-1])
                await self._emit(run_id, EventType.TOOL_CALL_COMPLETED if result.ok else EventType.TOOL_CALL_FAILED,
                                 {"tool": pending["name"],
                                  "version": pending_spec.version if pending_spec else "",
                                  "confirmed": True, "ok": result.ok,
                                  "content": result.content[:300]})
                pending = None

            while step < self.max_steps:
                cancel.raise_if_cancelled()
                await self._emit(run_id, EventType.STEP_STARTED, {"step": step, "agent": self.agent_name})

                messages, context_report = self.builder.build(
                    system=self.system_prompt, history=history, injected=injected,
                )
                await self._emit(run_id, EventType.CONTEXT_BUILT, context_report.to_dict())
                if context_report.compressed:
                    await self._emit(run_id, EventType.CONTEXT_COMPRESSED, {
                        "dropped_messages": context_report.dropped_messages,
                        "truncated_tools": context_report.truncated_tools,
                        "compression_ratio": context_report.compression_ratio,
                    })

                tools_schema = self.registry.schemas(scopes) or None
                await self._emit(run_id, EventType.LLM_REQUEST, {
                    "model": getattr(self.llm, "name", "unknown"),
                    "messages": len(messages),
                    "context_tokens": context_report.final_tokens,
                    "tools": [t["function"]["name"] for t in (tools_schema or [])],
                })

                async with tracer.span("llm.chat", model=getattr(self.llm, "name", "unknown")):
                    response = await run_with_cancel(
                        self.llm.chat(messages, tools_schema,
                                      temperature=self.settings.llm_temperature),
                        cancel,
                    )

                usage = response.usage
                if usage.cost_usd == 0 and usage.total_tokens > 0:
                    usage.cost_usd = compute_cost(response.model, usage.prompt_tokens, usage.completion_tokens)
                meter.record(usage, response.model or "unknown")
                meter.check_budget()
                await self._emit(run_id, EventType.LLM_RESPONSE, {
                    "content": (response.content or "")[:500],
                    "tool_calls": [tc.to_dict() for tc in response.tool_calls],
                    "usage": usage.to_dict(),
                    "finish_reason": response.finish_reason,
                })
                await self._emit(run_id, EventType.USAGE_RECORDED, {
                    "usage": usage.to_dict(), "run_total": meter.snapshot()["total"],
                })
                self.metrics.inc("hermes_llm_tokens_total", usage.total_tokens, model=response.model or "unknown")

                # 没有工具调用 = 最终答案
                if not response.tool_calls:
                    answer = response.content or ""
                    history.append(Message(role="assistant", content=answer))
                    self.sessions.add_message(session_id, history[-1])
                    state = "succeeded"
                    break

                history.append(Message(role="assistant", content=response.content or "",
                                       tool_calls=response.tool_calls))
                self.sessions.add_message(session_id, history[-1])

                paused = False
                for tool_call in response.tool_calls:
                    cancel.raise_if_cancelled()
                    spec = self.registry.get(tool_call.name)
                    version = spec.version if spec else ""
                    await self._emit(run_id, EventType.TOOL_CALL_STARTED, {
                        "tool": tool_call.name, "version": version,
                        "call_id": tool_call.id, "arguments": tool_call.arguments,
                    })
                    ctx = ToolContext(run_id=run_id, tenant_id=tenant_id, user_id=user_id,
                                      scopes=scopes, cancel=cancel, confirmed=confirmed)
                    async with tracer.span("tool.execute", tool=tool_call.name):
                        result = await run_with_cancel(
                            self.registry.execute(tool_call.name, tool_call.arguments, ctx), cancel)

                    if result.needs_confirmation:
                        pending = {"id": tool_call.id, "name": tool_call.name,
                                   "arguments": tool_call.arguments}
                        paused = True
                        await self._emit(run_id, EventType.TOOL_CONFIRM_REQUIRED, {
                            "tool": tool_call.name, "version": version,
                            "arguments": tool_call.arguments,
                            "preview": result.content,
                        })
                        continue

                    tool_message = Message(
                        role="tool",
                        content=self._truncate_tool(wrap_untrusted(
                            "tool_output", result.render(),
                            {"tool": tool_call.name, "agent": self.agent_name})),
                        tool_call_id=tool_call.id, name=tool_call.name)
                    history.append(tool_message)
                    self.sessions.add_message(session_id, tool_message)
                    await self._emit(
                        run_id,
                        EventType.TOOL_CALL_COMPLETED if result.ok else EventType.TOOL_CALL_FAILED,
                        {"tool": tool_call.name, "version": version, "ok": result.ok,
                         "code": result.code, "latency_ms": result.latency_ms,
                         "attempts": result.attempts, "cached": result.cached,
                         "content": result.content[:400]},
                    )

                step += 1
                self.runs.save_checkpoint(run_id, messages=history, step=step,
                                          pending=pending, usage=meter.snapshot(),
                                          context=extra_context)
                self.runs.update(run_id, state="running", steps=step)
                await self._emit(run_id, EventType.STEP_COMPLETED, {"step": step, "paused": paused})

                if paused:
                    state = "paused"
                    break
            else:
                state = "failed"
                error = f"超过最大步数 {self.max_steps}，任务未收敛"

        except CancelledByUser as exc:
            state, error = "cancelled", str(exc)
            await self._emit(run_id, EventType.RUN_CANCELLED, {"reason": error, "step": step})
        except BudgetExceeded as exc:
            state, error = "failed", str(exc)
            await self._emit(run_id, EventType.RUN_FAILED, {"error": error, "kind": "budget"})
        except Exception as exc:  # noqa: BLE001
            state, error = "failed", str(exc)
            await self._emit(run_id, EventType.RUN_FAILED, {"error": error, "kind": type(exc).__name__})

        usage_snapshot = meter.snapshot()
        self.runs.update(run_id, state=state, answer=answer, error=error, steps=step,
                         usage=usage_snapshot,
                         checkpoint={"messages": [m.to_dict() for m in history], "step": step,
                                     "pending": pending, "usage": usage_snapshot,
                                     "context": extra_context})

        if state == "succeeded":
            await self._emit(run_id, EventType.RUN_COMPLETED, {
                "steps": step, "usage": usage_snapshot["total"], "answer": (answer or "")[:500],
            })
            await self._maybe_propose_memory(run_id, tenant_id, user_id, user_input)
            self.metrics.inc("hermes_runs_total", state="succeeded")
        elif state == "paused":
            await self._emit(run_id, EventType.RUN_PAUSED, {"pending": pending})
            self.metrics.inc("hermes_runs_total", state="paused")
        elif state != "cancelled":
            self.metrics.inc("hermes_runs_total", state=state)

        self.metrics.observe("hermes_run_steps", step)
        await self._observe_lessons(run_id, state=state, answer=answer or "",
                                    steps=step, citations=len(citations), tenant_id=tenant_id)
        return RunResult(
            run_id=run_id, state=state, answer=answer, steps=step,
            usage=usage_snapshot["total"], citations=citations,
            context=context_report.to_dict() if context_report else {},
            pending=pending, error=error,
        )

    # ------------------------------------------------------------- helpers
    def with_tools(self, names: list[str], *, agent_name: str | None = None,
                   system_prompt: str | None = None, max_steps: int | None = None) -> AgentRuntime:
        """派生一个只暴露指定工具的子 Agent 运行时（最小权限）。"""
        registry = self.registry
        if hasattr(registry, "subset"):
            registry = registry.subset(names)
        return AgentRuntime(
            settings=self.settings,
            sessions=self.sessions,
            runs=self.runs,
            bus=self.bus,
            registry=registry,
            llm=self.llm,
            metrics=self.metrics,
            memory=self.memory,
            knowledge=self.knowledge,
            builder=self.builder,
            lessons=self.lessons,
            system_prompt=system_prompt or self.system_prompt,
            max_steps=max_steps or self.max_steps,
            agent_name=agent_name or f"{self.agent_name}.worker",
        )

    async def _prepare_context(self, query: str, tenant_id: str, user_id: str,
                               scopes: frozenset[str], injected: dict[str, str],
                               citations: list[dict], recalled: list[dict], run_id: str) -> None:
        if self.lessons is not None:
            guidance = self.lessons.render_active(limit=3, tenant_id=tenant_id)
            if guidance:
                injected["历史教训"] = guidance

        if self.memory is not None:
            memories = self.memory.recall(tenant_id=tenant_id, user_id=user_id, query=query,
                                          top_k=self.settings.memory_top_k)
            if memories:
                injected["长期记忆"] = MemoryService.render(memories)
                recalled.extend(m.to_dict() for m in memories)
                await self._emit(run_id, EventType.MEMORY_RECALLED, {
                    "count": len(memories),
                    "items": [{"key": m.record.mkey, "value": m.record.value, "score": round(m.score, 4)}
                              for m in memories],
                })

        if self.knowledge is not None:
            acl_groups = {s.split(":", 2)[2] for s in scopes if s.startswith("kb:group:")}
            # 检索是 CPU/IO 混合操作，放到线程里避免阻塞事件循环
            hits = await asyncio.to_thread(
                self.knowledge.search, query,
                top_k=self.settings.rerank_top_k, acl_groups=acl_groups or None,
            )
            if hits:
                injected["知识库检索结果"] = self.knowledge.render_context(hits)
                citations.extend(self.knowledge.citations(hits))
                await self._emit(run_id, EventType.KNOWLEDGE_RETRIEVED, {
                    "count": len(hits), "citations": citations,
                    "embedder": getattr(self.knowledge, "describe", lambda: {})().get("embedding", {}),
                    # 忠实度评测要拿原文比对，事件里保留上下文
                    "contexts": [h.text for h in hits],
                })
            else:
                await self._emit(run_id, EventType.KNOWLEDGE_RETRIEVED, {"count": 0})

    async def _maybe_propose_memory(self, run_id: str, tenant_id: str, user_id: str, user_input: str) -> None:
        if self.memory is None:
            return
        created = self.memory.propose_from_text(tenant_id=tenant_id, user_id=user_id,
                                                text=user_input, source_run_id=run_id)
        if created:
            await self._emit(run_id, EventType.MEMORY_PROPOSED, {
                "count": len(created),
                "items": [{"id": m.id, "key": m.mkey, "value": m.value, "status": m.status} for m in created],
            })

    async def _observe_lessons(self, run_id: str, *, state: str, answer: str,
                               steps: int, citations: int, tenant_id: str) -> None:
        """运行结束后提炼教训；提炼失败绝不能影响主流程。"""
        if self.lessons is None or state not in ("succeeded", "failed"):
            return
        try:
            proposed = self.lessons.observe_run(
                run_id=run_id, events=self.bus.buffered(run_id), state=state,
                answer=answer, steps=steps, max_steps=self.max_steps, citations=citations,
                tenant_id=tenant_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("教训提炼失败: %s", exc)
            return
        for lesson in proposed:
            await self._emit(run_id, EventType.LESSON_PROPOSED, {
                "id": lesson.id, "rule": lesson.rule, "status": lesson.status,
                "content": lesson.content[:300],
            })

    @staticmethod
    def _context_meta(extra: dict | None) -> dict | None:
        """只把形状写进事件，避免页面全文进轨迹后难以脱敏与回放。"""
        if not extra:
            return None
        return {"page_chars": len(str(extra.get("page") or "")),
                "rules": len(extra.get("rules") or [])}

    @staticmethod
    def _render_run_context(extra: dict | None) -> str | None:
        if not extra:
            return None
        page = str(extra.get("page") or "").strip()[:2000]
        rules = [str(rule).strip()[:200] for rule in (extra.get("rules") or [])
                 if str(rule).strip()][:10]
        parts: list[str] = []
        if page:
            parts.append(wrap_untrusted("page_context", page, {"source": "request"}))
        if rules:
            parts.append("企业规则（仅在与用户请求不冲突时参考）：\n"
                         + "\n".join(f"- {rule}" for rule in rules))
        text = "\n\n".join(parts)
        return text[:3000] if text else None

    def _truncate_tool(self, content: str) -> str:
        limit = self.settings.context.tool_result_max_chars
        return content if len(content) <= limit else content[:limit] + "\n…[已截断]"

    async def _emit(self, run_id: str, event_type: EventType, payload: dict) -> None:
        await self.bus.emit(Event(run_id=run_id, type=event_type, payload=payload))

    @staticmethod
    def _result_from_row(row: dict) -> RunResult:
        import json

        return RunResult(
            run_id=row["id"], state=row["state"], answer=row.get("answer"),
            steps=int(row.get("steps") or 0),
            usage=json.loads(row.get("usage") or "{}"),
            error=row.get("error"),
        )
