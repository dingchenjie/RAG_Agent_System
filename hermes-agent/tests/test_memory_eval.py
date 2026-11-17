"""记忆评测：召回命中、冲突归档、TTL、待确认与跨租户/跨用户隔离。"""
from __future__ import annotations

from pathlib import Path

from hermes.eval.memory import evaluate_memory, load_memory_tasks

EVAL = Path(__file__).resolve().parents[1] / "eval"


def test_memory_dataset_passes(state):
    tasks = load_memory_tasks(EVAL / "memory.jsonl")
    result = evaluate_memory(state.memory, tasks)

    assert result["cases"] == len(tasks) == 6
    assert result["pass_rate"] == 1.0

    by_id = {item["task_id"]: item for item in result["details"]}
    assert by_id["mem-recall-basic"]["hits"][0]["key"] == "preferred_language"
    assert by_id["mem-conflict-new-wins"]["missing_values"] == []
    assert by_id["mem-ttl-expired"]["hits"] == []
    assert by_id["mem-pending-invisible"]["hits"] == []
    assert by_id["mem-tenant-isolation"]["hits"] == []
    assert by_id["mem-user-isolation"]["hits"] == []
