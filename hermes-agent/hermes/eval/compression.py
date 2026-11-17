"""上下文压缩评测：长对话里埋关键事实，压缩后检查事实是否还在。

判定不依赖模型：直接调用 Runtime 使用的 ``ContextBuilder``，对压缩后的消息做断言，
同时统计 token 压缩比。这样同一份数据集在 Mock 与真实模型下都可用。

数据集里有一条**负对照**（``expect_retained: false``）：事实被放在摘要窗口之外的深处，
预期就是丢失。如果评测把它判成"保留"，说明判据本身失效——负对照就是用来防这个的。
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ..config import ContextConfig
from ..runtime.context import ContextBuilder
from ..types import Message, ToolCall

_DEFAULT_FILLER = "客户在周三确认了交付节奏，运营团队同步了库存与排期，本轮没有新的风险项。"


def load_compression_tasks(path: str | Path) -> list[dict]:
    tasks: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    return tasks


def _body(task: dict) -> str:
    filler = str(task.get("filler") or _DEFAULT_FILLER)
    repeat = max(1, int(task.get("filler_chars", 120)) // max(1, len(filler)))
    return filler * repeat


def _build_history(task: dict) -> tuple[list[Message], str]:
    """按任务声明拼出一段长对话，返回历史与当前问题。"""
    turns = int(task.get("turns", 6))
    needle = str(task["needle"])
    where = str(task.get("needle_location", "early_user"))
    body = _body(task)
    messages: list[Message] = []
    for index in range(turns):
        # 工具结果放在最后一轮之前：仍在 recent 窗口内，会走"旧结果留头尾"的裁剪路径
        if where == "recent_tool" and index == turns - 1:
            messages.append(Message(
                role="assistant", content="",
                tool_calls=[ToolCall(id="call-1", name="kb_search", arguments={"query": "订单状态"})],
            ))
            messages.append(Message(
                role="tool", content=f"{needle}。{body}",
                tool_call_id="call-1", name="kb_search",
            ))

        if index == 0 and where == "early_user":
            user_content = f"{needle}。{body}"
        elif index == 0 and where == "deep_in_early_user":
            # 前 600 字与事实无关，落在默认抽取式摘要的 100 字窗口之外
            user_content = f"{body[:600]}{needle}。{body[600:]}"
        elif index == 1 and where == "summary_fact":
            # 摘要保留每条消息的前 100 字，事实必须放在开头才谈得上"保留"
            user_content = f"{needle}。{body}"
        else:
            user_content = f"第 {index} 轮：{body}"
        messages.append(Message(role="user", content=user_content))

        assistant_content = f"好的，第 {index} 轮已记录。{body}"
        if index == 1 and where == "summary_fact":
            assistant_content = f"{needle}，已记录。{body}"
        messages.append(Message(role="assistant", content=assistant_content))

    question = str(task["question"])
    if where == "current_user":
        question = f"{needle}。{question}"
    return messages, question


def evaluate_compression(
    tasks: list[dict],
    *,
    cfg: ContextConfig | None = None,
) -> dict:
    """逐条压缩并检查事实保留，返回保留率、压缩比与逐条证据。"""
    base = cfg or ContextConfig()
    results: list[dict] = []
    for task in tasks:
        history, question = _build_history(task)
        history.append(Message(role="user", content=question))
        local = replace(
            base,
            total_tokens=int(task.get("budget", base.total_tokens)),
            reserve_output_tokens=int(task.get("reserve", base.reserve_output_tokens)),
            recent_turns_keep=int(task.get("recent_turns_keep", base.recent_turns_keep)),
            summary_max_tokens=int(task.get("summary_max_tokens", base.summary_max_tokens)),
        )
        messages, report = ContextBuilder(local).build(
            system=str(task.get("system") or "你是企业助手，回答时优先引用已确认的事实。"),
            history=history,
            injected=task.get("injected"),
        )
        needle = str(task["needle"])
        retained = any(needle in message.content for message in messages)
        expected = bool(task.get("expect_retained", True))
        results.append({
            "task_id": task.get("id", ""),
            "category": task.get("category", "general"),
            "budget": report.budget,
            "original_tokens": report.original_tokens,
            "final_tokens": report.final_tokens,
            "compression_ratio": report.compression_ratio,
            "dropped_messages": report.dropped_messages,
            "summary_tokens": report.summary_tokens,
            "retained": retained,
            "expected_retained": expected,
            "passed": retained == expected,
        })
    return _summarize(results)


def _summarize(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    retained = sum(1 for item in results if item["retained"])
    groups: dict[str, list[dict]] = {}
    for item in results:
        groups.setdefault(item["category"], []).append(item)

    by_category = {}
    for category, items in sorted(groups.items()):
        ok = sum(1 for item in items if item["passed"])
        by_category[category] = {
            "cases": len(items),
            "passed": ok,
            "pass_rate": ok / len(items),
            "avg_compression_ratio": round(
                sum(item["compression_ratio"] for item in items) / len(items), 4),
        }

    ratios = [item["compression_ratio"] for item in results]
    finals = [item["final_tokens"] for item in results]
    return {
        "cases": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "retained": retained,
        "retention_rate": retained / total if total else 0.0,
        "avg_compression_ratio": round(sum(ratios) / len(ratios), 4) if ratios else 0.0,
        "avg_final_tokens": round(sum(finals) / len(finals), 1) if finals else 0.0,
        "by_category": by_category,
        "details": results,
    }
