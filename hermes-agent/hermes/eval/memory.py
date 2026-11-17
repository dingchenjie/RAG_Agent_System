"""长期记忆评测：召回命中、冲突归档、TTL 过期、租户隔离。

与检索评测一样，直接驱动真实的 ``MemoryService``，只从 ``recall`` 的返回值做断言，
不复制业务逻辑，因此 Mock embedding 与真实 embedding 共用同一套判定。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..memory.extractor import MemoryCandidate


def load_memory_tasks(path: str | Path) -> list[dict]:
    tasks: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    return tasks


def evaluate_memory(memory: Any, tasks: list[dict]) -> dict:
    """逐条写入种子记忆并召回，返回通过率与逐条证据。

    用例字段：
    - ``seed``：``key`` / ``value`` / 可选 ``confirmed``（默认 true）、``ttl_days``、
      ``subject_type`` / ``subject_id``；
    - ``query``：召回语句；``query_tenant`` / ``query_user`` 可指定"另一个身份"来测隔离；
    - ``expect_keys`` / ``expect_values``：必须命中；
    - ``absent_keys`` / ``absent_values``：不得命中；
    - ``expect_empty``：召回结果必须为空（TTL、待确认、跨租户场景）。
    """
    results: list[dict] = []
    for task in tasks:
        tenant_id = str(task.get("tenant_id") or "eval-memory")
        user_id = str(task.get("user_id") or "eval-user")
        _seed(memory, task.get("seed") or [], tenant_id=tenant_id, user_id=user_id)

        hits = memory.recall(
            tenant_id=str(task.get("query_tenant") or tenant_id),
            user_id=str(task.get("query_user") or user_id),
            query=str(task["query"]),
            top_k=int(task.get("top_k", 4)),
        )
        keys = [hit.record.mkey for hit in hits]
        values = [hit.record.value for hit in hits]
        missing_keys = [key for key in task.get("expect_keys") or [] if key not in keys]
        missing_values = [value for value in task.get("expect_values") or [] if value not in values]
        leaked_keys = [key for key in task.get("absent_keys") or [] if key in keys]
        leaked_values = [value for value in task.get("absent_values") or [] if value in values]
        empty_expected = bool(task.get("expect_empty", False))

        ok = not missing_keys and not missing_values and not leaked_keys and not leaked_values
        if empty_expected:
            ok = ok and not hits
        elif not (task.get("expect_keys") or task.get("expect_values")):
            ok = ok and bool(hits)

        results.append({
            "task_id": task.get("id", ""),
            "category": task.get("category", "general"),
            "hits": [{"key": key, "value": value, "score": round(hit.score, 4)}
                     for key, value, hit in zip(keys, values, hits, strict=True)],
            "missing_keys": missing_keys,
            "missing_values": missing_values,
            "leaked_keys": leaked_keys,
            "leaked_values": leaked_values,
            "expect_empty": empty_expected,
            "passed": ok,
        })

    return _summarize(results)


def _seed(memory: Any, seeds: list[dict], *, tenant_id: str, user_id: str) -> None:
    for item in seeds:
        records = memory.propose(
            tenant_id=tenant_id,
            user_id=user_id,
            candidates=[MemoryCandidate(
                mkey=str(item["key"]), value=str(item["value"]), confidence=1.0,
            )],
            ttl_days=item.get("ttl_days"),
            subject_type=item.get("subject_type"),
            subject_id=item.get("subject_id"),
        )
        if records and item.get("confirmed", True):
            memory.confirm(records[0].id, tenant_id=tenant_id, user_id=user_id)


def _summarize(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    groups: dict[str, list[dict]] = {}
    for item in results:
        groups.setdefault(item["category"], []).append(item)

    by_category = {}
    for category, items in sorted(groups.items()):
        ok = sum(1 for item in items if item["passed"])
        by_category[category] = {
            "cases": len(items),
            "passed": ok,
            "pass_rate": ok / len(items),
        }

    return {
        "cases": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "by_category": by_category,
        "details": results,
    }
