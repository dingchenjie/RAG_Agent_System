"""上下文工程。

LLM 的上下文是有限预算资源，Runtime 必须显式管理它，而不是"把什么都拼进去"：
1. 预算分配：system / 记忆 / 知识 / 近期对话 / 历史摘要各有上限；
2. 压缩：超出窗口的旧对话抽取成摘要，保留最近 N 轮原文；
3. 裁剪：旧工具结果只留头尾，避免一条长输出挤掉所有历史；
4. 有效信息保留：最新的工具结果、用户最新诉求、引用来源始终优先保留。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import ContextConfig
from ..tokens import estimate_tokens, truncate_tokens
from ..types import Message


@dataclass
class ContextReport:
    budget: int
    original_tokens: int
    final_tokens: int
    sections: list[dict] = field(default_factory=list)
    compressed: bool = False
    summary_tokens: int = 0
    dropped_messages: int = 0
    truncated_tools: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens <= 0:
            return 0.0
        return round(max(0.0, 1 - self.final_tokens / self.original_tokens), 4)

    def to_dict(self) -> dict:
        return {
            "budget": self.budget,
            "original_tokens": self.original_tokens,
            "final_tokens": self.final_tokens,
            "compressed": self.compressed,
            "compression_ratio": self.compression_ratio,
            "summary_tokens": self.summary_tokens,
            "dropped_messages": self.dropped_messages,
            "truncated_tools": self.truncated_tools,
            "sections": self.sections,
        }


def _default_summarize(messages: list[Message]) -> str:
    """确定性抽取式摘要：保留每轮的关键片段，不额外调用模型（不花 token）。"""
    parts: list[str] = []
    for msg in messages:
        if msg.role == "tool":
            continue
        snippet = msg.content.strip().replace("\n", " ")[:100]
        if snippet:
            prefix = {"user": "用户", "assistant": "助手", "system": "系统"}.get(msg.role, msg.role)
            parts.append(f"{prefix}:{snippet}")
    return " | ".join(parts)


class ContextBuilder:
    def __init__(self, cfg: ContextConfig, summarizer: Callable[[list[Message]], str] | None = None):
        self.cfg = cfg
        self.summarizer = summarizer or _default_summarize

    def build(
        self,
        *,
        system: str,
        history: list[Message],
        injected: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> tuple[list[Message], ContextReport]:
        injected = injected or {}
        budget = (max_tokens or self.cfg.total_tokens) - self.cfg.reserve_output_tokens
        budget = max(256, budget)

        original_tokens = estimate_tokens(system) + sum(
            estimate_tokens(m.content) + 4 for m in history
        )
        report = ContextReport(budget=budget, original_tokens=original_tokens, final_tokens=0)

        # 当前轮 = 最后一条 user 消息及其之后的内容，必须保持原文（否则模型看不到本次任务的依据）
        last_user_idx = max(
            (i for i, m in enumerate(history) if m.role == "user"), default=len(history) - 1
        )

        # 1) 旧工具结果瘦身（只处理当前轮之前的历史，保留头尾）
        trimmed: list[Message] = []
        for i, msg in enumerate(history):
            if (i < last_user_idx and msg.role == "tool"
                    and len(msg.content) > self.cfg.old_tool_result_chars):
                report.truncated_tools += 1
                trimmed.append(Message(
                    role=msg.role,
                    content=truncate_tokens(msg.content, self.cfg.old_tool_result_chars // 2),
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                    meta={**msg.meta, "truncated": True},
                ))
            else:
                trimmed.append(msg)
        history = trimmed

        # 2) 历史压缩：最近 N 轮原文 + 更早内容摘要；切分点不得越过当前轮起点
        keep = max(2, self.cfg.recent_turns_keep)
        cut = min(max(0, len(history) - keep), last_user_idx)
        # 避免 recent 以孤立的 tool 消息开头（其 assistant 调用已进摘要，会被 provider 拒绝）
        while cut < len(history) and history[cut].role == "tool":
            cut += 1
        older, recent = history[:cut], history[cut:]
        summary_text = ""
        if older:
            raw_summary = self.summarizer(older)
            summary_text = truncate_tokens(raw_summary, self.cfg.summary_max_tokens)
            report.compressed = True

        # 3) 注入块按优先级分配剩余预算：记忆 > 知识
        system_tokens = estimate_tokens(system)
        summary_tokens = estimate_tokens(summary_text)
        remaining = budget - system_tokens - summary_tokens
        blocks: list[tuple[str, str, int]] = []
        for name in sorted(injected, key=lambda n: 0 if "记忆" in n else 1):
            content = injected[name]
            if not content:
                continue
            share = max(64, int(remaining * (0.35 if "记忆" in name else 0.5)))
            cap = min(share, max(0, remaining))
            text = truncate_tokens(content, cap)
            used = estimate_tokens(text)
            remaining -= used
            if used > 0:
                blocks.append((name, text, used))
                report.sections.append({"name": name, "tokens": used, "kept": True})

        system_content = system
        if summary_text:
            system_content += f"\n\n[历史摘要]\n{summary_text}"
            report.summary_tokens = summary_tokens
            report.sections.append({"name": "history_summary", "tokens": summary_tokens, "kept": True})
        for name, text, _used in blocks:
            system_content += f"\n\n[{name}]\n{text}"
        report.sections.insert(0, {"name": "system", "tokens": system_tokens, "kept": True})

        messages: list[Message] = [Message(role="system", content=system_content)] + list(recent)

        # 4) 仍然超预算：从最旧的对话开始丢。
        #    规则：当前轮（最后一条 user 及其之后）整段保护；更早的按"问 + 答" / "调用 + 结果"成组丢弃，
        #    不留下孤立的 tool 消息或断头的回答，否则会被 OpenAI 兼容接口判为非法请求。
        total = sum(estimate_tokens(m.content) + 4 for m in messages)
        current_turn = max((i for i, m in enumerate(messages) if m.role == "user"), default=1)
        while total > budget:
            target = 1 if current_turn > 1 else None
            if target is None:
                break
            group = [messages[target]]
            nxt = target + 1
            head = messages[target]
            if head.role == "assistant" and head.tool_calls:
                ids = {tc.id for tc in head.tool_calls}
                while nxt < len(messages) and messages[nxt].role == "tool" \
                        and messages[nxt].tool_call_id in ids:
                    group.append(messages[nxt])
                    nxt += 1
            elif head.role == "user" and nxt < current_turn and messages[nxt].role == "assistant":
                group.append(messages[nxt])
            del messages[target:target + len(group)]
            for dropped in group:
                total -= estimate_tokens(dropped.content) + 4
                report.dropped_messages += 1
            report.compressed = True
            current_turn = max((i for i, m in enumerate(messages) if m.role == "user"), default=1)

        # 5) system 本身极端超长时兜底截断
        if estimate_tokens(messages[0].content) > budget:
            messages[0].content = truncate_tokens(messages[0].content, budget)
            report.compressed = True

        report.final_tokens = sum(estimate_tokens(m.content) + 4 for m in messages)
        return messages, report
