"""确定性 Mock Provider。

存在的意义不只是"省钱"：
1. 单元测试可以在没有网络/Key 的情况下断言完整调用链（工具顺序、取消、恢复）；
2. 评测集可以跑回归，不受模型随机性干扰；
3. 演示环境不依赖外部服务，随时可跑。
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from ..tokens import estimate_tokens
from ..types import Message, ToolCall, Usage
from .base import LLMProvider, LLMResponse

_CALC_RE = re.compile(r"\d+\s*[+\-*/^%]\s*\d+|计算|算一下|等于多少")
_MEMORY_RE = re.compile(r"记得|还记得|我叫|我的.*(偏好|习惯|名字|邮箱)|之前说过|上次")
_KB_RE = re.compile(r"论文|文档|资料|知识库|检索|LoRA|RAG|原理|方法|实验|政策|规定|流程|上限|报销|制度")

ScriptStep = LLMResponse | Callable[[list[Message], list[dict] | None], LLMResponse]


@dataclass
class MockLLM(LLMProvider):
    """两种模式：

    - 脚本模式：``script=[LLMResponse(...), ...]`` 按顺序返回，用于精确测试；
    - 自动模式：根据最后一条用户消息与工具结果，按规则决定"调工具"或"给答案"。
    """

    script: list[ScriptStep] | None = None
    model: str = "mock-1"
    name: str = "mock"

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if self.script:
            step = self.script.pop(0)
            response = step(messages, tools) if callable(step) else step
            response.model = response.model or self.model
            if response.usage.total_tokens == 0:
                response.usage = self._estimate_usage(messages, response.content or "")
            return response

        response = self._auto(messages, tools or [])
        response.model = self.model
        response.usage = self._estimate_usage(messages, response.content or "")
        return response

    # ---------- 自动决策 ----------
    def _auto(self, messages: list[Message], tools: list[dict]) -> LLMResponse:
        available = {t["function"]["name"] for t in tools if t.get("type") == "function"}
        last_user_idx = max((i for i, m in enumerate(messages) if m.role == "user"), default=-1)
        last_user = messages[last_user_idx].content if last_user_idx >= 0 else ""
        tool_msgs = [m for m in messages[last_user_idx + 1:] if m.role == "tool" if m.name]

        # 主编排 Agent 的汇总请求：没有工具可调，直接把子 Agent 结论整理成答复
        marker = "子 Agent 结论："
        if not available and marker in last_user:
            return LLMResponse(content="[mock] 综合子 Agent 结论：\n"
                                       + last_user.split(marker, 1)[1].strip()[:800])

        used = {m.name for m in tool_msgs if m.name}

        # 需要哪些能力，就按顺序补齐（支持多步：先检索、再计算）
        needs: list[tuple[str, tuple[str, ...]]] = []
        if _MEMORY_RE.search(last_user):
            needs.append(("memory", ("memory_recall", "memory_search", "recall_memory")))
        if _KB_RE.search(last_user):
            needs.append(("kb", ("kb_search", "knowledge_search", "search_knowledge")))
        if _CALC_RE.search(last_user):
            needs.append(("calc", ("calculator", "calc", "calculate")))

        for kind, candidates in needs:
            name = self._pick(available, candidates)
            if not name or name in used:
                continue
            if kind == "memory":
                return self._call(name, {"query": last_user}, "先查一下长期记忆里有没有相关信息。")
            if kind == "kb":
                return self._call(name, {"query": last_user}, "先去知识库检索相关资料。")
            return self._call(name, {"expression": self._extract_expression(last_user)}, "需要做数值计算。")

        if tool_msgs:
            return LLMResponse(content=self._compose_answer(last_user, tool_msgs))

        return LLMResponse(content=f"[mock] 已收到问题：{last_user}。当前没有需要调用的工具。")

    @staticmethod
    def _pick(available: set[str], candidates: tuple[str, ...]) -> str | None:
        for candidate in candidates:
            if candidate in available:
                return candidate
        for name in available:
            if any(name.endswith(f".{c}") or name.endswith(c) for c in candidates):
                return name
        return None

    def _call(self, name: str, arguments: dict, thought: str) -> LLMResponse:
        return LLMResponse(
            content=thought,
            tool_calls=[ToolCall(id=f"call_{abs(hash((name, str(arguments)))) % 10**8}", name=name, arguments=arguments)],
            finish_reason="tool_calls",
        )

    @staticmethod
    def _extract_expression(text: str) -> str:
        match = re.search(r"[\d\.\s+\-*/^%()]{3,}", text)
        return match.group(0).strip() if match else text

    @staticmethod
    def _compose_answer(question: str, tool_msgs: list[Message]) -> str:
        parts = []
        for msg in tool_msgs:
            label = msg.name or "tool"
            content = (msg.content or "").strip()
            if content:
                parts.append(f"[{label}] {content[:600]}")
        joined = "\n".join(parts) if parts else "（工具没有返回有效内容）"
        return f"[mock] 针对「{question}」的结论：\n{joined}"

    def _estimate_usage(self, messages: list[Message], content: str) -> Usage:
        prompt = sum(estimate_tokens(m.content) + 4 for m in messages)
        return Usage(prompt_tokens=prompt, completion_tokens=estimate_tokens(content), cost_usd=0.0)


class FailingLLM(LLMProvider):
    """测试用：前 N 次抛可重试错误，用于验证重试/熔断。"""

    name = "failing"

    def __init__(self, fail_times: int, *, retryable: bool = True):
        self.fail_times = fail_times
        self.retryable = retryable
        self.calls = 0

    async def chat(self, messages, tools=None, *, temperature=0.1, max_tokens=None) -> LLMResponse:
        from ..errors import LLMError

        self.calls += 1
        if self.calls <= self.fail_times:
            raise LLMError(f"模拟失败 #{self.calls}", retryable=self.retryable)
        return LLMResponse(content="ok", model="mock-1")
