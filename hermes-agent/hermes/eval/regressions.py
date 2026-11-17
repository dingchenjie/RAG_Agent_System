"""真实模型失败样本回归。

可复现的 Mock 评测保证 CI 稳定，但真实模型的问题往往来自"这次模型/提示词版本"。
这里把失败用例连同运行环境写进 JSONL，下次迭代直接作为回归集重放，
避免"改完 prompt 凭感觉说变好了"。

去重键为 ``task_id + provider + model + prompt_version``：同一版本重复跑只记一次，
换模型或改提示词才会新增记录。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def record_failures(report: dict, *, tasks: list[dict], path: str | Path) -> int:
    """把报告里的失败任务追加到 ``path``，返回本次新增条数。"""
    path = Path(path)
    meta = report.get("meta") or {}
    existing = _existing_keys(path)
    task_by_id = {task.get("id"): task for task in tasks}

    written = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for item in report.get("tasks") or []:
            if item.get("ok"):
                continue
            task_id = item.get("task_id", "")
            marker = _marker(meta, task_id)
            if marker in existing:
                continue
            record = {
                "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "meta": meta,
                "task": task_by_id.get(task_id, {"id": task_id}),
                "answer": item.get("answer", ""),
                "called_tools": item.get("called_tools") or [],
                "missing_keywords": item.get("missing_keywords") or [],
                "error": item.get("error"),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing.add(marker)
            written += 1
    return written


def _existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add(_marker(item.get("meta") or {}, (item.get("task") or {}).get("id", "")))
    return keys


def _marker(meta: dict, task_id: str) -> str:
    return "|".join([
        task_id,
        str(meta.get("llm_provider", "")),
        str(meta.get("llm_model", "")),
        str(meta.get("prompt_version", "")),
    ])
