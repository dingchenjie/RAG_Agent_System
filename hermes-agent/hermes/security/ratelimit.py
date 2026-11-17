"""租户级限流与并发配额。

两道彼此独立的闸门，都由同一个入口检查：
- ``rpm``：每租户令牌桶，限制提交速率，防止单租户把入口打满；
- ``max_inflight``：每租户在途 run 数，防止长任务占满工作线程。

默认 0 表示关闭，保持本地/CI 零配置；开启后超限统一返回 429 + ``Retry-After``，
被拒绝的请求不创建 run，也不会进入调度队列。
"""
from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    retry_after_s: int = 0
    limit: int = 0

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "retry_after_s": self.retry_after_s,
            "limit": self.limit,
        }


class TenantRateLimiter:
    def __init__(
        self,
        *,
        rpm: int = 0,
        max_inflight: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.rpm = max(0, int(rpm))
        self.max_inflight = max(0, int(max_inflight))
        self._clock = clock
        self._lock = threading.Lock()
        self._tokens: dict[str, float] = {}
        self._refilled_at: dict[str, float] = {}
        self._inflight: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self.rpm > 0 or self.max_inflight > 0

    def acquire(self, tenant_id: str) -> Decision:
        """尝试占用一次提交额度；成功时在途计数 +1，调用方必须配对 ``release``。"""
        if not self.enabled:
            return Decision(allowed=True)
        now = self._clock()
        with self._lock:
            active = self._inflight.get(tenant_id, 0)
            if self.max_inflight and active >= self.max_inflight:
                return Decision(False, "inflight_limited", 1, self.max_inflight)
            if self.rpm:
                tokens = self._refill(tenant_id, now)
                if tokens < 1.0:
                    rate_per_s = self.rpm / 60.0
                    wait = max(1, math.ceil((1.0 - tokens) / rate_per_s))
                    return Decision(False, "rate_limited", wait, self.rpm)
                self._tokens[tenant_id] = tokens - 1.0
            self._inflight[tenant_id] = active + 1
            return Decision(True)

    def track(self, tenant_id: str) -> None:
        """只登记在途、不消耗限流额度（用于重启恢复/续跑等非新提交场景）。"""
        with self._lock:
            self._inflight[tenant_id] = self._inflight.get(tenant_id, 0) + 1

    def release(self, tenant_id: str) -> None:
        with self._lock:
            self._inflight[tenant_id] = max(0, self._inflight.get(tenant_id, 0) - 1)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "rpm": self.rpm,
                "max_inflight": self.max_inflight,
                "inflight": dict(self._inflight),
            }

    def _refill(self, tenant_id: str, now: float) -> float:
        capacity = float(self.rpm)
        tokens = self._tokens.get(tenant_id, capacity)
        last = self._refilled_at.get(tenant_id)
        if last is not None:
            tokens = min(capacity, tokens + (now - last) * self.rpm / 60.0)
        self._tokens[tenant_id] = tokens
        self._refilled_at[tenant_id] = now
        return tokens
