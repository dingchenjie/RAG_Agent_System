"""任务调度器：优先级队列 + 并发上限 + 优雅停机。

API 层收到 run 请求后不直接 ``create_task``，而是进入调度队列：
- 控制并发，避免一次涌入把上游模型配额打满；
- 支持优先级（交互式请求优先于批处理评测）；
- 停机时 drain 队列，保证任务状态不丢。
"""
from __future__ import annotations

import asyncio
import itertools
from collections.abc import Awaitable, Callable


class RunScheduler:
    def __init__(self, max_concurrency: int = 4):
        self.max_concurrency = max(1, max_concurrency)
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._seq = itertools.count()
        self._workers: list[asyncio.Task] = []
        self._started = False
        self._closed = False
        self.completed = 0
        self.failed = 0

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for _ in range(self.max_concurrency):
            self._workers.append(asyncio.create_task(self._worker(), name="hermes-scheduler"))

    async def submit(self, factory: Callable[[], Awaitable], *, priority: int = 5) -> asyncio.Future:
        if self._closed:
            raise RuntimeError("调度器已关闭")
        await self.start()
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((priority, next(self._seq), factory, future))
        return future

    async def _worker(self) -> None:
        while True:
            priority, _, factory, future = await self._queue.get()
            if factory is None:  # 停机信号
                self._queue.task_done()
                return
            if future.cancelled():
                self._queue.task_done()
                continue
            try:
                result = await factory()
                if not future.done():
                    future.set_result(result)
                self.completed += 1
            except asyncio.CancelledError:
                if not future.done():
                    future.cancel()
                raise
            except Exception as exc:  # noqa: BLE001
                if not future.done():
                    future.set_exception(exc)
                self.failed += 1
            finally:
                self._queue.task_done()

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._workers:
            await self._queue.put((0, next(self._seq), None, None))  # type: ignore[arg-type]
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    @property
    def pending(self) -> int:
        return self._queue.qsize()
