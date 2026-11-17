"""失败样本回归：写入、去重与版本口径。"""
from __future__ import annotations

import json

from hermes.eval.regressions import record_failures

META = {
    "llm_provider": "zhipu", "llm_model": "glm-4-air", "prompt_version": "abcd1234",
}


def _report() -> dict:
    return {
        "meta": META,
        "tasks": [
            {"task_id": "t1", "ok": False, "answer": "答错了", "called_tools": ["kb_search"],
             "missing_keywords": ["600"], "error": None},
            {"task_id": "t2", "ok": True, "answer": "对的", "called_tools": [], "missing_keywords": []},
        ],
    }


def test_record_failures_writes_only_failed_tasks(tmp_path):
    path = tmp_path / "failures.jsonl"
    tasks = [{"id": "t1", "input": "一线城市住宿上限?", "expected_contains": ["600"]},
             {"id": "t2", "input": "你好"}]
    assert record_failures(_report(), tasks=tasks, path=path) == 1

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["task"]["input"] == "一线城市住宿上限?"
    assert record["answer"] == "答错了"
    assert record["meta"]["prompt_version"] == "abcd1234"


def test_record_failures_is_idempotent_per_version(tmp_path):
    path = tmp_path / "failures.jsonl"
    tasks = [{"id": "t1", "input": "一线城市住宿上限?"}]
    assert record_failures(_report(), tasks=tasks, path=path) == 1
    assert record_failures(_report(), tasks=tasks, path=path) == 0

    changed = dict(_report(), meta=dict(META, prompt_version="ffff0000"))
    assert record_failures(changed, tasks=tasks, path=path) == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
