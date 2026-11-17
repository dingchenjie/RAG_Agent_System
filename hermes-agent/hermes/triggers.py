"""周期触发：让 Agent 从"被调用"变成"会自己跑"。

设计取舍：
- 只做 interval，不做 cron 表达式：真正难的是"触发→去重→调度→可观测"这条链路，
  表达式解析可以在此之上叠加，但会引入大量边界；
- 触发循环跑在 API 进程内，SQLite 单写者保证同机不重复触发；
  多副本部署需要分布式锁，这是明确记录的边界；
- 触发器不自己执行 run，只产出"该触发了"，由 API 层复用 enqueue_run 落库提交，
  避免出现第二套"建 run"逻辑。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from .ids import new_id
from .storage.db import Database


@dataclass
class Trigger:
    id: str
    tenant_id: str
    user_id: str
    name: str
    input: str
    mode: str
    interval_s: int
    scopes: list[str]
    session_id: str | None
    enabled: bool
    next_fire_at: float
    last_run_id: str | None
    created_at: float
    updated_at: float

    def to_dict(self) -> dict:
        return {
            "id": self.id, "tenant_id": self.tenant_id, "user_id": self.user_id,
            "name": self.name, "input": self.input, "mode": self.mode,
            "interval_s": self.interval_s, "scopes": self.scopes,
            "session_id": self.session_id, "enabled": self.enabled,
            "next_fire_at": self.next_fire_at, "last_run_id": self.last_run_id,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }


class TriggerService:
    def __init__(self, db: Database):
        self.db = db

    def create(
        self,
        *,
        input: str,
        name: str,
        interval_s: int,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        mode: str = "single",
        scopes: list[str] | None = None,
        session_id: str | None = None,
        now: float | None = None,
    ) -> Trigger:
        if interval_s < 10:
            raise ValueError("interval_s 不能小于 10 秒")
        now = now or time.time()
        trigger_id = new_id("trg")
        self.db.execute(
            "INSERT INTO triggers(id, tenant_id, user_id, name, input, mode, interval_s, "
            "scopes, session_id, enabled, next_fire_at, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trigger_id, tenant_id, user_id, name, input, mode, interval_s,
             json.dumps(scopes or ["tool:read"], ensure_ascii=False),
             session_id, 1, now + interval_s, now, now),
        )
        return self.get(trigger_id)

    def get(self, trigger_id: str) -> Trigger:
        row = self.db.query_one("SELECT * FROM triggers WHERE id=?", (trigger_id,))
        if row is None:
            raise KeyError(trigger_id)
        return self._row(row)

    def list(self, *, tenant_id: str | None = None, limit: int = 50) -> list[Trigger]:
        if tenant_id:
            rows = self.db.query(
                "SELECT * FROM triggers WHERE tenant_id=? ORDER BY created_at DESC LIMIT ?",
                (tenant_id, limit))
        else:
            rows = self.db.query("SELECT * FROM triggers ORDER BY created_at DESC LIMIT ?", (limit,))
        return [self._row(r) for r in rows]

    def set_enabled(self, trigger_id: str, enabled: bool, *, tenant_id: str | None = None) -> bool:
        sql = "UPDATE triggers SET enabled=?, updated_at=? WHERE id=?"
        params: list[Any] = [1 if enabled else 0, time.time(), trigger_id]
        if tenant_id:
            sql += " AND tenant_id=?"
            params.append(tenant_id)
        return self.db.execute(sql, params).rowcount > 0

    def mark_fired(self, trigger_id: str, run_id: str, next_fire_at: float) -> None:
        self.db.execute(
            "UPDATE triggers SET last_run_id=?, next_fire_at=?, updated_at=? WHERE id=?",
            (run_id, next_fire_at, time.time(), trigger_id),
        )

    def remove(self, trigger_id: str, *, tenant_id: str | None = None) -> bool:
        sql = "DELETE FROM triggers WHERE id=?"
        params: list[Any] = [trigger_id]
        if tenant_id:
            sql += " AND tenant_id=?"
            params.append(tenant_id)
        return self.db.execute(sql, params).rowcount > 0

    def due(self, now: float | None = None, limit: int = 20) -> list[Trigger]:
        """取到期触发器，并原子推进 next_fire_at（先占位，再交给调用方执行）。

        推进发生在返回之前，因此即使上一次 run 还没跑完，也不会在同一周期重复触发。
        """
        now = now or time.time()
        rows = self.db.query(
            "SELECT * FROM triggers WHERE enabled=1 AND next_fire_at<=? "
            "ORDER BY next_fire_at ASC LIMIT ?", (now, limit))
        fired: list[Trigger] = []
        for row in rows:
            trigger = self._row(row)
            next_fire_at = trigger.next_fire_at + trigger.interval_s
            # 如果服务停了很久，跳过已经错过的周期，避免启动瞬间补跑一堆
            while next_fire_at <= now:
                next_fire_at += trigger.interval_s
            updated = self.db.execute(
                "UPDATE triggers SET next_fire_at=?, updated_at=? WHERE id=? AND enabled=1",
                (next_fire_at, now, trigger.id),
            )
            if updated.rowcount > 0:
                trigger.next_fire_at = next_fire_at
                fired.append(trigger)
        return fired

    @staticmethod
    def _row(row) -> Trigger:
        return Trigger(
            id=row["id"], tenant_id=row["tenant_id"], user_id=row["user_id"],
            name=row["name"], input=row["input"], mode=row["mode"],
            interval_s=row["interval_s"], scopes=json.loads(row["scopes"] or "[]"),
            session_id=row["session_id"], enabled=bool(row["enabled"]),
            next_fire_at=row["next_fire_at"], last_run_id=row["last_run_id"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )
