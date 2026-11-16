"""ID 生成：带前缀、可按时间排序，便于日志检索。"""
from __future__ import annotations

import secrets
import time


def new_id(prefix: str) -> str:
    # 毫秒时间戳 + 随机后缀，字典序约等于时间序
    return f"{prefix}_{int(time.time() * 1000):013d}{secrets.token_hex(4)}"


def new_run_id() -> str:
    return new_id("run")


def new_session_id() -> str:
    return new_id("sess")


def new_event_id() -> str:
    return new_id("evt")


def new_memory_id() -> str:
    return new_id("mem")


def new_span_id() -> str:
    return new_id("sp")
