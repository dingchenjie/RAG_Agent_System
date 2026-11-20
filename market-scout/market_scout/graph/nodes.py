"""图节点：路由 → 计划 → 并行收集（ReAct）→ 校验 → 写作 → 批评 → 发布。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..llm import MockLLM
from ..prompts import get_prompt
from ..store import Store
from ..tools import ToolContext, ToolRegistry, check_loop
from ..trace import Tracer
from .state import ScoutState

CITE_RE = re.compile(r"\[S(\d+)\]")
REACTIVE_HINTS = ("是什么", "是多少", "多少钱", "价格", "定义", "官网")
DELIBERATIVE_HINTS = ("对比", "调研", "分析", "报告", "竞品", "评估")


@dataclass
class Runtime:
    settings: Settings
    llm: Any
    registry: ToolRegistry
    store: Store
    tracer: Tracer
    run_id: str

    @property
    def ctx(self) -> ToolContext:
        if not hasattr(self, "_ctx"):
            self._ctx = ToolContext(store=self.store, run_id=self.run_id,
                                    tracer=self.tracer, settings=self.settings)
        return self._ctx


def classify(question: str) -> tuple[str, str]:
    """反应式 vs 深思熟虑：先用可解释的规则，避免为一次分类多花一轮模型调用。"""
    length = len(question)
    if length <= 40 and any(k in question for k in REACTIVE_HINTS) and not any(
            k in question for k in DELIBERATIVE_HINTS):
        return "reactive", "短问题 + 单点事实诉求，走一次 ReAct 即可"
    return "deliberative", "涉及多维度对比或报告产出，需要计划、并行收集与校验"


def heuristic_sections(limit: int) -> list[str]:
    base = ["定价与套餐", "核心功能", "目标用户与市场定位"]
    return base[: max(1, limit)]


class Nodes:
    def __init__(self, rt: Runtime):
        self.rt = rt

    # ---------- 路由 ----------
    async def route(self, state: ScoutState) -> dict[str, Any]:
        mode, reason = classify(state["question"])
        self.rt.tracer.emit("route.decided", mode=mode, reason=reason)
        return {"mode": mode, "route_reason": reason}

    def dispatch_route(self, state: ScoutState) -> str:
        return "quick" if state.get("mode") == "reactive" else "plan"

    # ---------- 计划 ----------
    async def plan(self, state: ScoutState) -> dict[str, Any]:
        sections = heuristic_sections(self.rt.settings.max_sections)
        if not isinstance(self.rt.llm, MockLLM):
            prompt = get_prompt(self.rt.settings.prompt_version, "planner").format(
                question=state["question"])
            try:
                resp = await self.rt.llm.chat([{"role": "user", "content": prompt}])
                data = json.loads(self._extract_json(resp["content"]))
                parsed = [str(s) for s in data.get("sections", []) if str(s).strip()]
                if parsed:
                    sections = parsed[: self.rt.settings.max_sections]
            except Exception as exc:
                self.rt.tracer.emit("plan.fallback", error=str(exc)[:200])
        self.rt.tracer.emit("plan.created", sections=sections)
        return {"sections": sections}

    def dispatch_sections(self, state: ScoutState) -> list[Any]:
        from langgraph.types import Send

        return [Send("collect", {"question": state["question"], "section": s})
                for s in state.get("sections", [])]

    # ---------- ReAct 子循环 ----------
    async def _react(self, question: str, section: str | None, max_steps: int) -> dict[str, Any]:
        system = get_prompt(self.rt.settings.prompt_version, "researcher").format(
            section=f"{question}｜{section}" if section else question)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system},
                                          {"role": "user",
                                           "content": f"{question}\n本节聚焦：{section}" if section else question}]
        history: list[tuple[str, str]] = []
        evidence: list[dict[str, Any]] = []
        tools_used: list[str] = []
        pages_before = self.rt.ctx.pages_used
        content, looped = "", False
        for step in range(max_steps):
            resp = await self.rt.llm.chat(messages, self.rt.registry.openai_schemas())
            calls = resp.get("tool_calls") or []
            content = resp.get("content") or ""
            if not calls:
                break
            messages.append({
                "role": "assistant", "content": content,
                "tool_calls": [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"],
                                             "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                               for c in calls]})
            for call in calls:
                if check_loop(history, call["name"], call["arguments"]):
                    looped = True
                    self.rt.tracer.emit("loop.detected", tool=call["name"], step=step,
                                        arguments=call["arguments"])
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": "重复动作已拦截：请基于已有证据直接作答。"})
                    continue
                tools_used.append(call["name"])
                result = await self.rt.registry.execute(call["name"], call["arguments"], self.rt.ctx)
                if result.get("ok") and result.get("source_id"):
                    evidence.append({"source_id": result["source_id"], "url": result.get("url", ""),
                                     "title": result.get("title", ""),
                                     "text": (result.get("text") or "")[:600]})
                for hit in (result.get("hits") or []):
                    evidence.append({"source_id": hit["source_id"], "url": hit["url"],
                                     "title": hit["title"], "text": hit["text"][:400]})
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result, ensure_ascii=False)[:2500]})
            if looped:
                break
        return {"notes": content.strip(), "evidence": evidence, "tools_used": tools_used,
                "pages": self.rt.ctx.pages_used - pages_before, "loop": looped}

    async def collect(self, state: ScoutState) -> dict[str, Any]:
        section = state.get("section", "")
        self.rt.tracer.emit("collect.started", section=section)
        result = await self._react(state["question"], section, self.rt.settings.max_steps)
        cites = sorted({e["source_id"] for e in result["evidence"]})
        self.rt.tracer.emit("collect.finished", section=section, chars=len(result["notes"]),
                            citations=[f"S{c}" for c in cites], loop_detected=result["loop"])
        return {"findings": [{"section": section, "notes": result["notes"],
                              "citations": cites}],
                "evidence": result["evidence"], "tools_used": result["tools_used"],
                "pages_used": result["pages"], "loop_detected": result["loop"]}

    async def quick(self, state: ScoutState) -> dict[str, Any]:
        result = await self._react(state["question"], None, max_steps=3)
        cites = sorted({e["source_id"] for e in result["evidence"]})
        answer = result["notes"] or "未找到相关公开信息。"
        if cites:
            answer += "\n\n来源：" + "、".join(f"[S{c}]" for c in cites)
        return {"draft": answer, "report": answer, "evidence": result["evidence"],
                "tools_used": result["tools_used"], "pages_used": result["pages"],
                "loop_detected": result["loop"]}

    # ---------- 写作 / 校验 ----------
    async def merge(self, state: ScoutState) -> dict[str, Any]:
        sources = sorted({e["source_id"] for e in state.get("evidence", [])})
        self.rt.tracer.emit("evidence.merged", sources=[f"S{s}" for s in sources],
                            pages=state.get("pages_used", 0))
        return {}

    async def write(self, state: ScoutState) -> dict[str, Any]:
        notes = "\n\n".join(f"### {f['section']}\n{f['notes']}\n引用：{[f'S{c}' for c in f['citations']]}"
                           for f in state.get("findings", []))
        if isinstance(self.rt.llm, MockLLM):
            draft = self._compose(state["question"], notes, state.get("evidence", []))
        else:
            prompt = get_prompt(self.rt.settings.prompt_version, "writer").format(
                question=state["question"], notes=notes)
            resp = await self.rt.llm.chat([{"role": "user", "content": prompt}])
            draft = resp.get("content") or ""
        draft = self._ensure_sources(draft, state.get("evidence", []))
        self.rt.tracer.emit("draft.written", chars=len(draft))
        return {"draft": draft}

    async def critique(self, state: ScoutState) -> dict[str, Any]:
        draft = state.get("draft", "")
        valid = {e["source_id"] for e in state.get("evidence", [])}
        used = {int(m) for m in CITE_RE.findall(draft)}
        issues: list[str] = []
        if not used:
            issues.append("全文没有任何 [S#] 引用，无法验证事实来源")
        unknown = sorted(used - valid)
        if unknown:
            issues.append(f"引用了不存在的来源：{[f'S{i}' for i in unknown]}")
        if not isinstance(self.rt.llm, MockLLM) and not issues:
            prompt = get_prompt(self.rt.settings.prompt_version, "critic").format(draft=draft)
            try:
                resp = await self.rt.llm.chat([{"role": "user", "content": prompt}])
                data = json.loads(self._extract_json(resp["content"]))
                issues.extend(str(x) for x in data.get("issues", [])[:5])
            except Exception:
                pass
        self.rt.tracer.emit("critique.finished", issues=issues, citations=sorted(used))
        return {"critique": issues}

    async def revise(self, state: ScoutState) -> dict[str, Any]:
        draft = state.get("draft", "")
        issues = state.get("critique", [])
        if issues and not isinstance(self.rt.llm, MockLLM):
            prompt = (get_prompt(self.rt.settings.prompt_version, "writer").format(
                question=state["question"],
                notes="\n".join(f"[S{e['source_id']}] {e['title']} {e['text'][:300]}"
                                for e in state.get("evidence", []))) +
                "\n上一版的问题，请修正：\n" + "\n".join(issues))
            try:
                resp = await self.rt.llm.chat([{"role": "user", "content": prompt}])
                draft = self._ensure_sources(resp.get("content") or draft, state.get("evidence", []))
            except Exception:
                pass
        self.rt.tracer.emit("draft.revised", revisions=state.get("revisions", 0) + 1)
        return {"draft": draft, "revisions": state.get("revisions", 0) + 1}

    async def publish(self, state: ScoutState) -> dict[str, Any]:
        report = state.get("draft", "")
        self.rt.tracer.emit("report.published", chars=len(report),
                            loop_detected=bool(state.get("loop_detected")))
        return {"report": report}

    # ---------- 工具函数 ----------
    @staticmethod
    def _extract_json(text: str) -> str:
        start, end = text.find("{"), text.rfind("}")
        return text[start:end + 1] if start >= 0 and end > start else "{}"

    @staticmethod
    def _compose(question: str, notes: str, evidence: list[dict[str, Any]]) -> str:
        cites = sorted({e["source_id"] for e in evidence})
        body = notes or "未找到公开信息。"
        if cites:
            body += "\n\n引用：" + "、".join(f"[S{c}]" for c in cites)
        return f"## {question}\n\n{body}"

    @staticmethod
    def _ensure_sources(draft: str, evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            return draft
        # 来源清单由代码生成：模型写的那段常常是空标题/瞎编 URL，直接丢掉重写
        if "## 证据来源" in draft:
            draft = draft.split("## 证据来源")[0].rstrip()
        seen: dict[int, dict[str, Any]] = {}
        for item in evidence:
            seen.setdefault(item["source_id"], item)

        def label(item: dict[str, Any]) -> str:
            title = item.get("title") or item.get("url") or "未命名来源"
            url = item.get("url") or ""
            return f"- [S{item['source_id']}] {title}" + (f" — {url}" if url and url != title else "")

        return draft + "\n\n## 证据来源\n" + "\n".join(label(it) for _, it in sorted(seen.items()))
