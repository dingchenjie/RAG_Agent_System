"""调度器：并发上限、优先级、失败传播。"""
from __future__ import annotations

import asyncio

import pytest

from hermes.runtime.scheduler import RunScheduler


def test_concurrency_limit_is_respected():
    async def scenario():
        scheduler = RunScheduler(max_concurrency=2)
        await scheduler.start()
        active = 0
        peak = 0

        async def job():
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1
            return "ok"

        futures = [await scheduler.submit(job) for _ in range(6)]
        results = await asyncio.gather(*futures)
        await scheduler.shutdown()
        return results, peak

    results, peak = asyncio.run(scenario())
    assert results == ["ok"] * 6
    assert peak == 2


def test_priority_orders_waiting_jobs():
    async def scenario():
        scheduler = RunScheduler(max_concurrency=1)
        await scheduler.start()
        order: list[str] = []
        gate = asyncio.Event()

        async def blocker():
            await gate.wait()
            order.append("blocker")
            return "b"

        def make(tag: str):
            async def job():
                order.append(tag)
                return tag
            return job

        first = await scheduler.submit(blocker, priority=0)
        await asyncio.sleep(0.05)
        low = await scheduler.submit(make("low"), priority=9)
        high = await scheduler.submit(make("high"), priority=1)
        await asyncio.sleep(0.05)
        gate.set()
        await asyncio.gather(first, low, high)
        await scheduler.shutdown()
        return order

    order = asyncio.run(scenario())
    assert order[0] == "blocker"
    assert order.index("high") < order.index("low")


def test_failure_is_propagated_and_queue_drains():
    async def scenario():
        scheduler = RunScheduler(max_concurrency=1)
        await scheduler.start()

        async def boom():
            raise ValueError("bad")

        with pytest.raises(ValueError):
            await (await scheduler.submit(boom))
        await scheduler.shutdown()
        return scheduler.pending == 0

    assert asyncio.run(scenario()) is True
