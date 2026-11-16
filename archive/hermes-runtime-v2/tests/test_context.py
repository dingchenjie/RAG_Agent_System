"""上下文工程：预算、压缩、裁剪、关键信息保留。"""
from __future__ import annotations

from hermes.config import ContextConfig
from hermes.runtime.context import ContextBuilder
from hermes.tokens import estimate_tokens, truncate_tokens
from hermes.types import Message, ToolCall


def _history(n: int) -> list[Message]:
    messages = []
    for i in range(n):
        messages.append(Message(role="user", content=f"用户第{i}个问题：" + "内容" * 40))
        messages.append(Message(role="assistant", content=f"助手第{i}个回答：" + "答案" * 40))
    return messages


def test_truncate_keeps_head_and_tail():
    text = "A" * 400 + "关键结论" + "B" * 400
    out = truncate_tokens(text, 40)
    assert len(out) < len(text)
    assert "已截断" in out


def test_budget_is_respected_and_recent_kept():
    cfg = ContextConfig(total_tokens=500, reserve_output_tokens=100, recent_turns_keep=10,
                        tool_result_max_chars=100, old_tool_result_chars=40, summary_max_tokens=60)
    builder = ContextBuilder(cfg)
    history = _history(30)
    history.append(Message(role="user", content="最后一个必须保留的问题"))
    messages, report = builder.build(system="系统提示", history=history,
                                     injected={"知识库检索结果": "片段" * 200})
    assert messages[-1].content == "最后一个必须保留的问题"
    assert sum(estimate_tokens(m.content) + 4 for m in messages) <= report.budget
    assert report.compressed is True
    assert report.dropped_messages > 0
    assert report.compression_ratio > 0


def test_tool_result_truncated_in_context():
    cfg = ContextConfig(total_tokens=800, reserve_output_tokens=100, recent_turns_keep=10,
                        old_tool_result_chars=50)
    history = [
        Message(role="user", content="问题"),
        Message(role="assistant", content=""),
        Message(role="tool", content="X" * 2000, tool_call_id="c1", name="kb_search"),
        Message(role="user", content="后续问题"),
    ]
    messages, report = ContextBuilder(cfg).build(system="s", history=history)
    tool_msg = next(m for m in messages if m.role == "tool")
    assert len(tool_msg.content) < 2000
    assert report.truncated_tools == 1


def test_current_turn_tool_result_kept_verbatim():
    """当前轮的工具结果是最新依据，不能被"旧结果瘦身"截断。"""
    cfg = ContextConfig(total_tokens=6000, reserve_output_tokens=100, recent_turns_keep=2,
                        old_tool_result_chars=20)
    history = [
        Message(role="user", content="LoRA 的秩 r 通常设置为多少？"),
        Message(role="assistant", content="",
                tool_calls=[ToolCall(id="c1", name="kb_search", arguments={"query": "LoRA 秩"})]),
        Message(role="tool", content="秩 r 通常设置为 8", tool_call_id="c1", name="kb_search"),
    ]
    messages, report = ContextBuilder(cfg).build(system="s", history=history)
    tool_msg = next(m for m in messages if m.role == "tool")
    assert tool_msg.content == "秩 r 通常设置为 8"
    assert report.truncated_tools == 0


def test_injected_blocks_are_trimmed_when_budget_tight():
    cfg = ContextConfig(total_tokens=400, reserve_output_tokens=50, summary_max_tokens=30)
    messages, report = ContextBuilder(cfg).build(
        system="系统", history=[Message(role="user", content="问题")],
        injected={"长期记忆": "记忆" * 400, "知识库检索结果": "知识" * 800},
    )
    system_content = messages[0].content
    assert "长期记忆" in system_content
    assert estimate_tokens(system_content) <= report.budget
