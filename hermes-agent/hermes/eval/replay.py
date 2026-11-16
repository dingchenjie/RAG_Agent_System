"""轨迹回放与版本对比。

两个版本的 Agent（换模型、换 Prompt、换工具、换上下文策略）差异到底在哪？
不看"感觉"，看轨迹：工具序列、步数、token、延迟、失败点。
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from ..observability.events import EventType


def summarize_trace(events: list) -> dict:
    if not events:
        return {"events": 0}
    tools = [e.payload.get("tool") for e in events if e.type == EventType.TOOL_CALL_STARTED]
    tool_stats = Counter(tools)
    llm_calls = [e for e in events if e.type == EventType.LLM_REQUEST]
    failures = [e for e in events if e.type in (EventType.TOOL_CALL_FAILED, EventType.LLM_ERROR)]
    compressed = [e for e in events if e.type == EventType.CONTEXT_COMPRESSED]
    start = min(e.ts for e in events)
    end = max(e.ts for e in events)
    return {
        "events": len(events),
        "duration_s": round(end - start, 3),
        "tools": dict(tool_stats),
        "tool_sequence": tools,
        "llm_calls": len(llm_calls),
        "tool_failures": len(failures),
        "context_compressions": len(compressed),
        "final_state": _final_state(events),
    }


def diff_traces(baseline: list, candidate: list) -> dict:
    base, cand = summarize_trace(baseline), summarize_trace(candidate)
    base_tools, cand_tools = set(base.get("tools", {})), set(cand.get("tools", {}))
    return {
        "baseline": base,
        "candidate": cand,
        "tools_added": sorted(cand_tools - base_tools),
        "tools_removed": sorted(base_tools - cand_tools),
        "llm_calls_delta": cand.get("llm_calls", 0) - base.get("llm_calls", 0),
        "duration_delta_s": round(cand.get("duration_s", 0) - base.get("duration_s", 0), 3),
        "tool_failures_delta": cand.get("tool_failures", 0) - base.get("tool_failures", 0),
        "state_changed": base.get("final_state") != cand.get("final_state"),
    }


def _final_state(events: list) -> str:
    for event in reversed(events):
        if event.type in (EventType.RUN_COMPLETED, EventType.RUN_FAILED,
                          EventType.RUN_CANCELLED, EventType.RUN_PAUSED):
            return event.type.value
    return "unknown"


class Replay:
    """把事件流按时间顺序重放成可读时间线。"""

    def __init__(self, event_store: Any):
        self.event_store = event_store

    def timeline(self, run_id: str) -> list[dict[str, Any]]:
        events = self.event_store.history(run_id)
        base = events[0].ts if events else 0
        timeline = []
        for event in events:
            if event.type in (EventType.SPAN_STARTED, EventType.SPAN_ENDED):
                continue
            timeline.append({
                "seq": event.seq,
                "t": round(event.ts - base, 3),
                "type": event.type.value,
                "payload": event.payload,
            })
        return timeline
