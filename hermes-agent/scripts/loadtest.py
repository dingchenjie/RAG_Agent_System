"""并发压测：对本机 API 打 N 个 run，统计成功率与尾延迟。

用法：
    python scripts/loadtest.py --runs 20 --concurrency 5
    python scripts/loadtest.py --runs 10 --mode orchestrate --base-url http://127.0.0.1:8000
    python scripts/loadtest.py --runs 8 --concurrency 1,2,4,8   # 容量扫描
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from collections import Counter

import httpx

DONE_STATES = {"succeeded", "failed", "cancelled", "paused"}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * q))
    return ordered[index]


async def _one_run(client: httpx.AsyncClient, index: int, mode: str, poll_interval: float) -> tuple[str, float]:
    started = time.perf_counter()
    if mode == "orchestrate":
        response = await client.post("/v1/orchestrate", json={"input": f"计算 {index} + 1 并说明结果"})
        elapsed = time.perf_counter() - started
        return ("succeeded" if response.status_code == 200 else f"http_{response.status_code}"), elapsed

    response = await client.post("/v1/runs", json={"input": f"计算 {index} + 1"})
    if response.status_code not in (200, 202):
        return f"http_{response.status_code}", time.perf_counter() - started
    run_id = response.json()["run_id"]
    state = "unknown"
    for _ in range(600):
        polled = await client.get(f"/v1/runs/{run_id}")
        state = polled.json().get("state", "unknown")
        if state in DONE_STATES:
            break
        await asyncio.sleep(poll_interval)
    return state, time.perf_counter() - started


async def main() -> int:
    parser = argparse.ArgumentParser(description="Hermes API 并发压测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--concurrency", default="5",
                        help="并发数；逗号分隔做容量扫描，如 1,2,4,8")
    parser.add_argument("--mode", choices=["single", "orchestrate"], default="single")
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    args = parser.parse_args()

    headers = {}
    api_key = os.getenv("HERMES_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    levels = _parse_levels(args.concurrency)
    results = []
    for level in levels:
        if len(levels) > 1:
            print(f"\n--- concurrency={level} ---")
        results.append(await _run_level(args, level, headers))

    if len(results) > 1:
        print("\n并发扫描：")
        print("concurrency  runs  success_rate  throughput(run/s)  P50(s)  P95(s)  P99(s)")
        for item in results:
            print(f"{item['concurrency']:>11}  {item['runs']:>4}  {item['rate']:>11.2%}  "
                  f"{item['throughput']:>16.2f}  {item['p50']:>6.2f}  {item['p95']:>6.2f}  "
                  f"{item['p99']:>6.2f}")
    return 0 if all(item["rate"] >= args.min_success_rate for item in results) else 1


async def _run_level(args: argparse.Namespace, concurrency: int, headers: dict) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    states: Counter[str] = Counter()

    async with httpx.AsyncClient(base_url=args.base_url, headers=headers, timeout=120.0) as client:
        async def task(index: int) -> None:
            async with semaphore:
                state, elapsed = await _one_run(client, index, args.mode, 0.1)
                states[state] += 1
                latencies.append(elapsed)

        started = time.perf_counter()
        await asyncio.gather(*(task(i) for i in range(args.runs)))
        wall = time.perf_counter() - started

    success = states.get("succeeded", 0)
    rate = success / max(1, args.runs)
    print(f"mode={args.mode} runs={args.runs} concurrency={concurrency} wall={wall:.1f}s")
    print(f"states={dict(states)}")
    print(f"success_rate={rate:.2%}  throughput={args.runs / wall:.2f} run/s")
    print(f"latency  P50={_percentile(latencies, 0.50):.2f}s  "
          f"P95={_percentile(latencies, 0.95):.2f}s  P99={_percentile(latencies, 0.99):.2f}s  "
          f"max={max(latencies):.2f}s")
    return {
        "concurrency": concurrency,
        "runs": args.runs,
        "rate": rate,
        "throughput": args.runs / wall,
        "p50": _percentile(latencies, 0.50),
        "p95": _percentile(latencies, 0.95),
        "p99": _percentile(latencies, 0.99),
        "states": dict(states),
    }


def _parse_levels(value: str) -> list[int]:
    levels = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            level = int(part)
        except ValueError as exc:
            raise SystemExit(f"并发数必须是整数：{part}") from exc
        if level <= 0:
            raise SystemExit(f"并发数必须大于 0：{part}")
        levels.append(level)
    return levels or [5]


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
