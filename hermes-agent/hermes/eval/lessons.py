"""教训注入 A/B：同一批任务，在"无 active 教训"与"注入已确认教训"下各跑一次。

离线默认使用 ``LessonAwareLLM``：它只在系统提示里出现 ``历史教训`` 时才按规范作答，
因此能确定性验证 **失败轨迹 → 提炼 → 人工确认 → 注入上下文 → 行为改变** 这条链路。
它验证的是机制是否真的接通（提示词确实到达模型、且能改变输出），不是模型能力上限；
真实效果量级应在接入真实模型后用同一套用例复测。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..learning.lessons import LessonService
from ..llm.base import LLMProvider, LLMResponse
from ..tokens import estimate_tokens
from ..types import Message, ToolCall, Usage
from .runner import EvalRunner


@dataclass
class LessonCase:
    """一条 A/B 用例：失败模式 + 修复它的教训 + 判定标准。"""

    case_id: str
    input: str
    lesson_rule: str
    lesson_content: str
    expected_tools: list[str] = field(default_factory=list)
    expected_contains: list[str] = field(default_factory=list)
    category: str = "citation"

    def to_task(self) -> dict:
        return {
            "id": self.case_id,
            "input": self.input,
            "category": self.category,
            "expected_tools": list(self.expected_tools),
            "expected_contains": list(self.expected_contains),
            "tenant_id": "eval",
            "user_id": "eval-user",
        }


def load_lesson_cases(path: str | Path) -> list[LessonCase]:
    cases: list[LessonCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        cases.append(LessonCase(
            case_id=data["id"],
            input=data["input"],
            lesson_rule=data["lesson_rule"],
            lesson_content=data["lesson_content"],
            expected_tools=list(data.get("expected_tools") or []),
            expected_contains=list(data.get("expected_contains") or []),
            category=data.get("category", "citation"),
        ))
    return cases


class LessonAwareLLM(LLMProvider):
    """离线 A/B 专用替身：系统提示里有 ``历史教训`` 才按规范作答。

    - 未注入教训：检索后直接给结论，不标引用编号（复现 ``unsupported_answer``）；
    - 已注入教训：同一份检索结果，按教训要求带上 ``[n]`` 引用。
    """

    name = "lesson-aware"

    def __init__(self, model: str = "lesson-aware-1", *, cite: str = "[1]"):
        self.model = model
        self.cite = cite

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        available = {t["function"]["name"] for t in (tools or []) if t.get("type") == "function"}
        system = "\n".join(m.content for m in messages if m.role == "system" and m.content)
        guided = "历史教训" in system
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "") or ""
        tool_msgs = [m for m in messages if m.role == "tool" and m.name]

        if "kb_search" in available and not tool_msgs:
            return self._respond(
                messages,
                LLMResponse(
                    content="先去知识库检索相关资料。",
                    tool_calls=[ToolCall(id="call_lesson_kb", name="kb_search",
                                         arguments={"query": last_user})],
                    finish_reason="tool_calls",
                ),
            )

        evidence = (tool_msgs[-1].content or "")[:200] if tool_msgs else ""
        if guided:
            content = f"依据编号 {self.cite}《LoRA 笔记》：{evidence}"
        else:
            content = f"根据资料：{evidence}"
        return self._respond(messages, LLMResponse(content=content))

    def _respond(self, messages: list[Message], response: LLMResponse) -> LLMResponse:
        response.model = self.model
        prompt = sum(estimate_tokens(m.content) + 4 for m in messages)
        response.usage = Usage(prompt_tokens=prompt,
                               completion_tokens=estimate_tokens(response.content or ""))
        return response


async def evaluate_lessons_ab(
    *,
    runner: EvalRunner,
    lessons: LessonService,
    cases: list[LessonCase],
    tenant_id: str = "eval",
) -> dict[str, Any]:
    """先在无 active 教训时跑一遍，激活教训后再跑一遍，返回前后对比。"""
    if lessons.active(limit=1, tenant_id=tenant_id):
        raise ValueError("教训 A/B 需要干净的教训库，请使用独立临时数据库")

    before = [await runner.run_task(case.to_task()) for case in cases]

    seeded: set[str] = set()
    for case in cases:
        lesson = lessons.propose(
            case.lesson_rule, case.lesson_content,
            {"source": "lessons_ab", "case": case.case_id}, tenant_id=tenant_id,
        )
        if lessons.set_status(lesson.id, "active", tenant_id=tenant_id):
            seeded.add(lesson.id)

    after = [await runner.run_task(case.to_task()) for case in cases]

    def success(results: list) -> int:
        return sum(1 for item in results if item.ok)

    before_ok, after_ok = success(before), success(after)
    total = len(cases)
    return {
        "cases": total,
        "before_success": before_ok,
        "after_success": after_ok,
        "before_success_rate": before_ok / total if total else 0.0,
        "after_success_rate": after_ok / total if total else 0.0,
        "delta": (after_ok - before_ok) / total if total else 0.0,
        "lessons_injected": len(seeded),
        "rules": sorted({case.lesson_rule for case in cases}),
        "per_case": [
            {
                "id": case.case_id,
                "before_ok": before_result.ok,
                "after_ok": after_result.ok,
                "before_steps": before_result.steps,
                "after_steps": after_result.steps,
            }
            for case, before_result, after_result in zip(cases, before, after, strict=True)
        ],
    }
