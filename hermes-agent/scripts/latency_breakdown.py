"""按事件轨迹拆分 run 耗时：LLM / 工具 / 其他（上下文、记忆、落库、调度）。

不新增埋点，只读 ``events`` 表已记录的时间戳与 ``latency_ms``：
- LLM 段 = 相邻 ``llm.request`` → ``llm.response`` 的耗时；
- 工具段 = ``tool.completed`` / ``tool.failed`` 的 ``latency_ms`` 之和；
- 其他段 = 墙钟时间 - LLM - 工具（上下文构建、记忆召回、checkpoint、SSE 派发等）。

用法：
    python scripts/latency_breakdown.py --db data/loadtest.db --limit 20
    python scripts/latency_breakdown.py --db data/hermes.db --limit 10 --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes.observability.events import EventStore, EventType  # noqa: E402
from hermes.storage.db import Database  # noqa: E402

TERMINAL = {EventType.RUN_COMPLETED, EventType.RUN_FAILED,
            EventType.RUN_CANCELLED, EventType.RUN_PAUSED}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def _segments(events) -> dict[str, float]:
    llm = tool = retrieval = 0.0
    request_ts: float | None = None
    for event in events:
        if event.type is EventType.LLM_REQUEST:
            request_ts = event.ts
        elif event.type is EventType.LLM_RESPONSE and request_ts is not None:
            llm += max(0.0, event.ts - request_ts)
            request_ts = None
        elif event.type in (EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED):
            seconds = float(event.payload.get("latency_ms") or 0.0) / 1000
            tool += seconds
            if event.payload.get("tool") == "kb_search":
                retrieval += seconds

    start = next((event.ts for event in events
                  if event.type in (EventType.RUN_CREATED, EventType.RUN_STARTED)), None)
    end = next((event.ts for event in reversed(events) if event.type in TERMINAL), None)
    wall = max(0.0, end - start) if start and end else 0.0
    other = max(0.0, wall - llm - tool)
    return {
        "wall_ms": wall * 1000,
        "llm_ms": llm * 1000,
        "tool_ms": tool * 1000,
        "retrieval_ms": retrieval * 1000,
        "other_ms": other * 1000,
    }


def _summary(rows: list[dict], key: str) -> dict[str, float]:
    values = [row[key] for row in rows]
    return {
        "mean": round(sum(values) / len(values), 2) if values else 0.0,
        "p50": round(_percentile(values, 0.50), 2),
        "p95": round(_percentile(values, 0.95), 2),
        "max": round(max(values), 2) if values else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="按事件轨迹拆分 run 耗时")
    parser.add_argument("--db", default=os.getenv("HERMES_DB_PATH", "data/hermes.db"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    db = Database(args.db)
    try:
        runs = db.query(
            "SELECT id, state FROM runs WHERE state IN ('succeeded','failed') "
            "ORDER BY updated_at DESC LIMIT ?",
            (args.limit,),
        )
        store = EventStore(db)
        rows = []
        for run in runs:
            events = store.history(run["id"], 0, 5000)
            if not events:
                continue
            rows.append({"id": run["id"], "state": run["state"], **_segments(events)})
    finally:
        db.close()

    if not rows:
        print(f"没有可分析的历史 run：{args.db}")
        return 1

    report = {
        "db": args.db,
        "runs": len(rows),
        "succeeded": sum(1 for row in rows if row["state"] == "succeeded"),
        "wall_ms": _summary(rows, "wall_ms"),
        "llm_ms": _summary(rows, "llm_ms"),
        "tool_ms": _summary(rows, "tool_ms"),
        "retrieval_ms": _summary(rows, "retrieval_ms"),
        "other_ms": _summary(rows, "other_ms"),
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    wall = report["wall_ms"]["p50"] or 1.0
    print(f"db={args.db} runs={report['runs']} succeeded={report['succeeded']}")
    for key, label in (("wall_ms", "总耗时"), ("llm_ms", "LLM"),
                       ("tool_ms", "工具"), ("retrieval_ms", "  └ kb_search"),
                       ("other_ms", "其他")):
        item = report[key]
        share = item["p50"] / wall * 100
        print(f"{label:<10} mean={item['mean']:>8.1f}ms  P50={item['p50']:>8.1f}ms  "
              f"P95={item['p95']:>8.1f}ms  max={item['max']:>8.1f}ms  P50占比={share:>5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
