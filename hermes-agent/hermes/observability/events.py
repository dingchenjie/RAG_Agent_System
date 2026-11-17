"""结构化事件流。

一次 run 的所有行为都落成事件（LLM 请求、工具调用、上下文压缩、记忆写入、
Agent 委派……）。事件同时用于：
1. SSE 实时推送（前端可观测）；
2. 落库后做轨迹回放 / 版本对比 / 自动评测；
3. 出错时的根因定位（每条事件带 span 关系）。
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..ids import new_event_id
from ..security.redact import redact_obj
from ..storage.db import Database


class EventType(str, Enum):
    RUN_CREATED = "run.created"
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    RUN_PAUSED = "run.paused"
    RUN_RESUMED = "run.resumed"

    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"

    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_RETRY = "llm.retry"
    LLM_ERROR = "llm.error"
    USAGE_RECORDED = "usage.recorded"

    TOOL_CALL_STARTED = "tool.started"
    TOOL_CALL_COMPLETED = "tool.completed"
    TOOL_CALL_FAILED = "tool.failed"
    TOOL_CALL_DENIED = "tool.denied"
    TOOL_CONFIRM_REQUIRED = "tool.confirm_required"

    CONTEXT_BUILT = "context.built"
    CONTEXT_COMPRESSED = "context.compressed"

    MEMORY_PROPOSED = "memory.proposed"
    MEMORY_RECALLED = "memory.recalled"
    MEMORY_CONFIRMED = "memory.confirmed"
    MEMORY_CORRECTED = "memory.corrected"
    MEMORY_EXPIRED = "memory.expired"

    KNOWLEDGE_INGESTED = "knowledge.ingested"
    KNOWLEDGE_RETRIEVED = "knowledge.retrieved"

    AGENT_PLANNED = "agent.planned"
    AGENT_DELEGATED = "agent.delegated"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    BLACKBOARD_UPDATED = "blackboard.updated"

    LESSON_PROPOSED = "lesson.proposed"

    SPAN_STARTED = "span.started"
    SPAN_ENDED = "span.ended"


@dataclass
class Event:
    run_id: str
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=new_event_id)
    span_id: str | None = None
    parent_span_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "seq": self.seq,
            "ts": self.ts,
            "type": self.type.value if isinstance(self.type, EventType) else str(self.type),
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "payload": self.payload,
        }

    @classmethod
    def from_row(cls, row: Any) -> Event:
        return cls(
            run_id=row["run_id"],
            seq=row["seq"],
            ts=row["ts"],
            type=EventType(row["type"]),
            span_id=row["span_id"],
            parent_span_id=row["parent_span_id"],
            payload=json.loads(row["payload"] or "{}"),
        )


class EventStore:
    """事件持久化。seq 在库内串行分配，保证同一 run 的全序。"""

    def __init__(self, db: Database):
        self.db = db
        # MAX(seq)+1 与 INSERT 必须原子，否则并发子 Agent 会争到同一个 seq
        self._lock = threading.Lock()

    def append(self, event: Event) -> int:
        with self._lock:
            # 事件是唯一事实源，凭据一旦落进来就会被 trace/replay/eval 反复读取，
            # 因此在入口处就脱敏，并同步改写事件对象，让实时订阅者看到同样的内容。
            event.payload = redact_obj(event.payload)
            seq = int(self.db.scalar(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id = ?", (event.run_id,)
            ) or 1)
            event.seq = seq
            self.db.execute(
                "INSERT INTO events(run_id, seq, ts, type, span_id, parent_span_id, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    event.run_id,
                    seq,
                    event.ts,
                    event.type.value if isinstance(event.type, EventType) else str(event.type),
                    event.span_id,
                    event.parent_span_id,
                    json.dumps(event.payload, ensure_ascii=False, default=str),
                ),
            )
        return seq

    def history(self, run_id: str, after_seq: int = 0, limit: int = 5000) -> list[Event]:
        rows = self.db.query(
            "SELECT * FROM events WHERE run_id = ? AND seq > ? ORDER BY seq LIMIT ?",
            (run_id, after_seq, limit),
        )
        return [Event.from_row(r) for r in rows]


class EventBus:
    """进程内发布订阅 + 落库 + 环形缓冲（SSE 重放用）。"""

    def __init__(self, store: EventStore | None = None, buffer_size: int = 2000):
        self.store = store
        self._buffer_size = buffer_size
        self._buffers: dict[str, deque[Event]] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def emit(self, event: Event) -> Event:
        if self.store is not None:
            await asyncio.to_thread(self.store.append, event)
        buf = self._buffers.setdefault(event.run_id, deque(maxlen=self._buffer_size))
        buf.append(event)
        for queue in list(self._subscribers.get(event.run_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # 慢消费者直接丢弃最旧事件，保证运行时不被拖死
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except Exception:
                    pass
        return event

    def emit_sync(self, event: Event) -> None:
        """同步上下文（如工具线程）中的轻量派发，不含落库。"""
        buf = self._buffers.setdefault(event.run_id, deque(maxlen=self._buffer_size))
        buf.append(event)
        for queue in list(self._subscribers.get(event.run_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def subscribe(self, run_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.setdefault(run_id, set()).add(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        self._subscribers.get(run_id, set()).discard(queue)

    def buffered(self, run_id: str) -> list[Event]:
        return list(self._buffers.get(run_id, ()))
