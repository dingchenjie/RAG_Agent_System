"""轨迹回放：事件时间线、摘要、两版对比。"""
from __future__ import annotations

from hermes.eval.replay import Replay, diff_traces, summarize_trace
from hermes.observability.events import Event, EventStore, EventType
from hermes.storage.db import Database


def _trace(run_id: str, tools: list[str], *, failed: bool = False) -> list[Event]:
    events = [Event(run_id=run_id, type=EventType.RUN_STARTED, ts=1.0)]
    ts = 2.0
    for name in tools:
        events.append(Event(run_id=run_id, type=EventType.TOOL_CALL_STARTED,
                            payload={"tool": name}, ts=ts))
        ts += 0.5
    if failed:
        events.append(Event(run_id=run_id, type=EventType.TOOL_CALL_FAILED,
                            payload={"tool": tools[-1] if tools else "x"}, ts=ts))
        events.append(Event(run_id=run_id, type=EventType.RUN_FAILED, ts=ts + 0.1))
    else:
        events.append(Event(run_id=run_id, type=EventType.RUN_COMPLETED, ts=ts + 0.1))
    return events


def test_summarize_and_diff():
    base = _trace("a", ["kb_search", "calculate"])
    cand = _trace("b", ["kb_search"], failed=True)
    summary = summarize_trace(base)
    assert summary["tool_sequence"] == ["kb_search", "calculate"]
    assert summary["final_state"] == EventType.RUN_COMPLETED.value

    diff = diff_traces(base, cand)
    assert diff["tools_removed"] == ["calculate"]
    assert diff["tool_failures_delta"] == 1
    assert diff["state_changed"] is True


def test_replay_timeline_from_event_store(tmp_path):
    store = EventStore(Database(tmp_path / "events.db"))
    for event in _trace("run_1", ["kb_search"]):
        store.append(event)
    timeline = Replay(store).timeline("run_1")
    assert timeline
    assert timeline[0]["type"] == EventType.RUN_STARTED.value
    assert all("t" in item for item in timeline)
