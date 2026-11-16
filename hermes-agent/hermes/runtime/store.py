"""会话与任务状态持久化。

进程重启后能恢复，靠的是三张表：
- ``sessions/messages``：短期记忆（会话历史）
- ``runs``：任务状态机 + checkpoint
- ``events``：完整轨迹（在 observability 层写入）
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from ..ids import new_run_id, new_session_id
from ..storage.db import Database
from ..types import Message


@dataclass
class Session:
    id: str
    tenant_id: str
    user_id: str
    title: str | None
    created_at: float
    updated_at: float


class SessionStore:
    def __init__(self, db: Database):
        self.db = db

    def create_session(self, *, tenant_id: str, user_id: str, title: str | None = None,
                       session_id: str | None = None) -> Session:
        now = time.time()
        session = Session(
            id=session_id or new_session_id(), tenant_id=tenant_id, user_id=user_id,
            title=title, created_at=now, updated_at=now,
        )
        self.db.execute(
            "INSERT OR IGNORE INTO sessions(id, tenant_id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (session.id, tenant_id, user_id, title, now, now),
        )
        return session

    def get_session(self, session_id: str) -> Session | None:
        row = self.db.query_one("SELECT * FROM sessions WHERE id=?", (session_id,))
        if row is None:
            return None
        return Session(id=row["id"], tenant_id=row["tenant_id"], user_id=row["user_id"],
                       title=row["title"], created_at=row["created_at"], updated_at=row["updated_at"])

    def add_message(self, session_id: str, message: Message) -> None:
        payload = {
            "tool_calls": [tc.to_dict() for tc in message.tool_calls],
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "meta": message.meta,
        }
        now = time.time()
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, payload, created_at) VALUES (?,?,?,?,?)",
                (session_id, message.role, message.content, json.dumps(payload, ensure_ascii=False), now),
            )
            conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))

    def load_messages(self, session_id: str, limit: int = 200) -> list[Message]:
        rows = self.db.query(
            "SELECT * FROM (SELECT * FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (session_id, limit),
        )
        return [self._row_to_message(r) for r in rows]

    @staticmethod
    def _row_to_message(row) -> Message:
        payload = json.loads(row["payload"] or "{}")
        return Message.from_dict({
            "role": row["role"],
            "content": row["content"],
            "tool_calls": payload.get("tool_calls") or [],
            "tool_call_id": payload.get("tool_call_id"),
            "name": payload.get("name"),
            "meta": payload.get("meta") or {},
        })


class RunStore:
    def __init__(self, db: Database):
        self.db = db

    def create_run(self, *, session_id: str, tenant_id: str, user_id: str, user_input: str,
                   idempotency_key: str | None = None, run_id: str | None = None) -> str:
        run_id = run_id or new_run_id()
        now = time.time()
        self.db.execute(
            "INSERT INTO runs(id, session_id, tenant_id, user_id, input, state, created_at, updated_at, idempotency_key) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, session_id, tenant_id, user_id, user_input, "running", now, now, idempotency_key),
        )
        return run_id

    def get_run(self, run_id: str) -> dict | None:
        row = self.db.query_one("SELECT * FROM runs WHERE id=?", (run_id,))
        return dict(row) if row else None

    def find_by_idempotency(self, key: str) -> dict | None:
        row = self.db.query_one("SELECT * FROM runs WHERE idempotency_key=?", (key,))
        return dict(row) if row else None

    def update(self, run_id: str, *, state: str | None = None, answer: str | None = None,
               error: str | None = None, steps: int | None = None, usage: dict | None = None,
               checkpoint: dict | None = None) -> None:
        fields = ["updated_at=?"]
        params: list = [time.time()]
        for name, value in (("state", state), ("answer", answer), ("error", error), ("steps", steps)):
            if value is not None:
                fields.append(f"{name}=?")
                params.append(value)
        if usage is not None:
            fields.append("usage=?")
            params.append(json.dumps(usage, ensure_ascii=False))
        if checkpoint is not None:
            fields.append("checkpoint=?")
            params.append(json.dumps(checkpoint, ensure_ascii=False, default=str))
        params.append(run_id)
        self.db.execute(f"UPDATE runs SET {', '.join(fields)} WHERE id=?", params)

    def save_checkpoint(self, run_id: str, *, messages: list[Message], step: int,
                        pending: dict | None, usage: dict) -> None:
        checkpoint = {
            "messages": [m.to_dict() for m in messages],
            "step": step,
            "pending": pending,
            "usage": usage,
        }
        self.db.execute(
            "UPDATE runs SET checkpoint=?, steps=?, updated_at=? WHERE id=?",
            (json.dumps(checkpoint, ensure_ascii=False, default=str), step, time.time(), run_id),
        )

    def load_checkpoint(self, run_id: str) -> dict | None:
        row = self.db.query_one("SELECT checkpoint FROM runs WHERE id=?", (run_id,))
        if not row or not row["checkpoint"]:
            return None
        return json.loads(row["checkpoint"])

    def list_runs(self, *, session_id: str | None = None, limit: int = 20) -> list[dict]:
        if session_id:
            rows = self.db.query("SELECT * FROM runs WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
                                 (session_id, limit))
        else:
            rows = self.db.query("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]
