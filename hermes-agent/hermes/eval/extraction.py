"""记忆抽取评测：该抽的抽到没、不该抽的是否乱抽。

直接调用抽取器（规则或 LLM），与写入/召回解耦，两个后端共用同一套判定：
- precision：抽出的候选里有多少命中标注；
- recall：标注的稳定信息里抽到了多少；
- false_positive_rate / abstain_accuracy：负例（寒暄、一次性指令、注入、代码）
  是否保持沉默；
- by_category：按 profile / preference / project / note / negative 分组，
  便于定位是"该抽没抽"还是"不该抽却抽了"。

用例字段：``text`` 为输入；``expect`` 是期望条目列表，每项 ``any_keys``（命中的
key 之一）与 ``value_contains``（值包含的片段）；``expect_empty`` 表示应无候选。
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from ..memory.extractor import MemoryCandidate

ExtractFn = Callable[[str], Awaitable[list[MemoryCandidate]]]


def load_extraction_tasks(path: str | Path) -> list[dict]:
    tasks: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    return tasks


async def evaluate_extraction(extract: ExtractFn, tasks: list[dict]) -> dict:
    results: list[dict] = []
    total_tp = total_fp = total_fn = 0
    negative_cases = negative_silent = 0

    for task in tasks:
        predicted = await extract(str(task.get("text") or ""))
        expects = task.get("expect") or []
        used: set[int] = set()
        matched = 0
        missing: list[dict] = []

        for slot in expects:
            keys = set(slot.get("any_keys") or ([slot["key"]] if slot.get("key") else []))
            needles = [str(item).casefold() for item in
                       (slot.get("value_any") or [slot.get("value_contains") or ""])]
            needles = [item for item in needles if item]
            hit = next(
                (index for index, candidate in enumerate(predicted)
                 if index not in used and candidate.mkey in keys
                 and (not needles or any(item in candidate.value.casefold() for item in needles))),
                None,
            )
            if hit is None:
                missing.append({"any_keys": sorted(keys),
                                "value_any": [str(item) for item in needles]})
            else:
                used.add(hit)
                matched += 1

        extra = [candidate for index, candidate in enumerate(predicted) if index not in used]
        tp, fn, fp = matched, len(expects) - matched, len(extra)
        total_tp += tp
        total_fn += fn
        total_fp += fp

        is_negative = bool(task.get("expect_empty", False)) or not expects
        if is_negative:
            negative_cases += 1
            if not predicted:
                negative_silent += 1

        results.append({
            "task_id": task.get("id", ""),
            "category": task.get("category", "general"),
            "predicted": [{"key": c.mkey, "value": c.value, "confidence": round(c.confidence, 3)}
                          for c in predicted],
            "missing": missing,
            "extra": [{"key": c.mkey, "value": c.value} for c in extra],
            "passed": not missing and not extra,
        })

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    return {
        "cases": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "pass_rate": sum(1 for item in results if item["passed"]) / len(results) if results else 0.0,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "negative_cases": negative_cases,
        "false_positive_rate": (negative_cases - negative_silent) / negative_cases if negative_cases else 0.0,
        "abstain_accuracy": negative_silent / negative_cases if negative_cases else 1.0,
        "by_category": _by_category(results),
        "details": results,
    }


def _by_category(results: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for item in results:
        groups.setdefault(item["category"], []).append(item)
    return {
        category: {
            "cases": len(items),
            "passed": sum(1 for item in items if item["passed"]),
            "pass_rate": sum(1 for item in items if item["passed"]) / len(items),
        }
        for category, items in sorted(groups.items())
    }
