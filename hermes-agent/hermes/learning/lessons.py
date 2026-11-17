"""经验沉淀：把失败轨迹提炼成可复用的规则，下次运行前注入。

三个刻意选择：
1. 不在线学习——规则是确定性的，失败轨迹进、文本结论出，可复现、可审计；
2. 不自动生效——新规则是 ``pending``，人工 ``--activate`` 后才进入上下文，
   避免一次偶发失败污染后续所有任务；
3. 只注入最近 3 条 active——教训是稀缺资源，多了反而挤占正常指令的预算。
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..ids import new_lesson_id
from ..observability.events import EventType
from ..storage.db import Database

TOOL_INSTABILITY = "tool_instability"
UNSUPPORTED_ANSWER = "unsupported_answer"
PLAN_TOO_LONG = "plan_too_long"

STATUSES = ("pending", "active", "rejected")


@dataclass
class Lesson:
    id: str
    tenant_id: str
    rule: str
    content: str
    evidence: dict
    status: str
    hits: int
    created_at: float
    updated_at: float

    def to_dict(self) -> dict:
        return {
            "id": self.id, "tenant_id": self.tenant_id, "rule": self.rule,
            "content": self.content, "evidence": self.evidence, "status": self.status,
            "hits": self.hits, "created_at": self.created_at, "updated_at": self.updated_at,
        }


class LessonService:
    def __init__(self, db: Database, tenant_id: str = "default"):
        self.db = db
        self.tenant_id = tenant_id

    # ------------------------------------------------------------- 提炼
    def observe_run(
        self,
        *,
        run_id: str,
        events: list[Any],
        state: str,
        answer: str,
        steps: int,
        max_steps: int,
        citations: int = 0,
        tenant_id: str | None = None,
    ) -> list[Lesson]:
        """按固定规则检查一次运行的轨迹，命中则提出（或累加）一条教训。"""
        tenant = tenant_id or self.tenant_id
        proposed: list[Lesson] = []

        failures = Counter(
            event.payload.get("tool", "?") for event in events
            if event.type == EventType.TOOL_CALL_FAILED
        )
        for tool, count in failures.items():
            if count >= 2:
                proposed.append(self.propose(
                    TOOL_INSTABILITY,
                    f"工具 {tool} 在一次任务中失败 {count} 次：调用前先校验参数与前置条件，"
                    f"或改用其它工具完成同类任务。",
                    {"run_id": run_id, "tool": tool, "failures": count},
                    tenant_id=tenant,
                ))

        retrieved = any(
            event.type == EventType.KNOWLEDGE_RETRIEVED and event.payload.get("contexts")
            for event in events
        )
        if state == "succeeded" and retrieved and citations <= 0:
            proposed.append(self.propose(
                UNSUPPORTED_ANSWER,
                "检索到了资料却没有在回答里标注来源编号：结论必须引用 [n] 编号，否则无法追溯。",
                {"run_id": run_id},
                tenant_id=tenant,
            ))

        plan_events = [e for e in events if e.type == EventType.AGENT_PLANNED]
        if plan_events:
            subtasks = len(plan_events[-1].payload.get("subtasks", []))
            if subtasks > 3:
                proposed.append(self.propose(
                    PLAN_TOO_LONG,
                    f"一次规划拆出了 {subtasks} 个子任务：合并同能力域的子任务，"
                    "或直接单 Agent 完成，避免委派开销超过收益。",
                    {"run_id": run_id, "subtasks": subtasks},
                    tenant_id=tenant,
                ))
        elif state == "failed" and steps >= max_steps:
            proposed.append(self.propose(
                PLAN_TOO_LONG,
                f"任务在 {steps} 步内仍未收敛：先把问题拆小，或改用确定性工具链，"
                "不要在单次运行里硬扛多个未知。",
                {"run_id": run_id, "steps": steps},
                tenant_id=tenant,
            ))

        return proposed

    def propose(self, rule: str, content: str, evidence: dict | None = None,
                *, tenant_id: str | None = None) -> Lesson:
        """同一租户同一规则只保留一条：命中即刷新内容与证据，不重复膨胀。"""
        tenant = tenant_id or self.tenant_id
        now = time.time()
        existing = self.db.query_one(
            "SELECT * FROM lessons WHERE tenant_id=? AND rule=?", (tenant, rule))
        if existing is not None:
            self.db.execute(
                "UPDATE lessons SET content=?, evidence=?, hits=hits+1, updated_at=? WHERE id=?",
                (content, json.dumps(evidence or {}, ensure_ascii=False),
                 now, existing["id"]),
            )
            return self._row(self.db.query_one("SELECT * FROM lessons WHERE id=?", (existing["id"],)))
        lesson_id = new_lesson_id()
        self.db.execute(
            "INSERT INTO lessons(id, tenant_id, rule, content, evidence, status, hits, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (lesson_id, tenant, rule, content,
             json.dumps(evidence or {}, ensure_ascii=False), "pending", 1, now, now),
        )
        return self._row(self.db.query_one("SELECT * FROM lessons WHERE id=?", (lesson_id,)))

    # ------------------------------------------------------------- 管理
    def list(self, *, status: str | None = None, limit: int = 20,
             tenant_id: str | None = None) -> list[Lesson]:
        tenant = tenant_id or self.tenant_id
        if status:
            rows = self.db.query(
                "SELECT * FROM lessons WHERE tenant_id=? AND status=? "
                "ORDER BY updated_at DESC LIMIT ?", (tenant, status, limit))
        else:
            rows = self.db.query(
                "SELECT * FROM lessons WHERE tenant_id=? ORDER BY updated_at DESC LIMIT ?",
                (tenant, limit))
        return [self._row(r) for r in rows]

    def set_status(self, lesson_id: str, status: str,
                   *, tenant_id: str | None = None) -> bool:
        if status not in STATUSES:
            raise ValueError(f"未知的教训状态: {status}")
        cur = self.db.execute(
            "UPDATE lessons SET status=?, updated_at=? WHERE id=? AND tenant_id=?",
            (status, time.time(), lesson_id, tenant_id or self.tenant_id),
        )
        return cur.rowcount > 0

    def active(self, limit: int = 3, *, tenant_id: str | None = None) -> list[Lesson]:
        return self.list(status="active", limit=limit, tenant_id=tenant_id)

    def render_active(self, limit: int = 3, *, tenant_id: str | None = None) -> str:
        lessons = self.active(limit, tenant_id=tenant_id)
        if not lessons:
            return ""
        lines = ["历史教训（已确认，需主动规避）："]
        lines += [f"{i}. {lesson.content}" for i, lesson in enumerate(lessons, start=1)]
        return "\n".join(lines)

    @staticmethod
    def _row(row) -> Lesson:
        return Lesson(
            id=row["id"], tenant_id=row["tenant_id"], rule=row["rule"],
            content=row["content"], evidence=json.loads(row["evidence"] or "{}"),
            status=row["status"], hits=row["hits"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )
