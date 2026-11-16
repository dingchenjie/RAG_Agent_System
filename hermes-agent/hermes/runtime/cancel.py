"""取消控制。

Agent 的取消不是"停个循环"那么简单：必须能打断正在进行的 LLM 请求与工具调用，
并保证状态落库为 ``cancelled``，不留下半截的副作用。
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, TypeVar

from ..errors import CancelledByUser

T = TypeVar("T")


class CancelToken:
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.reason: str | None = None

    def cancel(self, reason: str = "用户取消") -> None:
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledByUser(self.reason or "用户取消")


async def run_with_cancel(coro: Awaitable[T], token: CancelToken | None) -> T:
    """执行协程，期间一旦收到取消信号就立刻取消它。"""
    if token is None:
        return await coro

    token.raise_if_cancelled()
    task = asyncio.ensure_future(coro)
    waiter = asyncio.ensure_future(token.wait())
    try:
        done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return task.result()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        raise CancelledByUser(token.reason or "用户取消")
    finally:
        if not waiter.done():
            waiter.cancel()
        if not task.done():
            task.cancel()
