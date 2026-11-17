"""检索质量专项评测：recall@k / MRR 与 ACL 过滤。"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes.eval.retrieval import evaluate_retrieval

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def knowledge(state):
    state.knowledge.ingest_file(ROOT / "knowledge_base" / "lora_notes.md",
                                doc_id="lora_notes", version="1")
    state.knowledge.ingest_file(ROOT / "knowledge_base" / "finance_policy.md",
                                doc_id="finance_policy", version="1", acl=["finance"])
    return state.knowledge


def test_retrieval_metrics_hit_miss_and_skip(knowledge):
    tasks = [
        {"id": "hit", "input": "LoRA 的秩 r 常用取值是多少？",
         "gold": [{"doc_id": "lora_notes", "contains": "最常用的取值是 8 或 16"}]},
        {"id": "miss", "input": "LoRA 的秩 r",
         "gold": [{"doc_id": "lora_notes", "contains": "不存在的片段"}]},
        {"id": "skip", "input": "你好"},
    ]
    result = evaluate_retrieval(knowledge, tasks, k=5)
    assert result["scored_tasks"] == 2
    assert result["recall_at_k"] == 0.5
    assert result["mrr"] > 0
    assert [item["ok"] for item in result["details"]] == [True, False]


def test_retrieval_respects_acl(knowledge):
    task = {
        "id": "finance", "input": "差旅住宿一线城市的上限是多少？",
        "scopes": ["tool:read", "kb:group:finance"],
        "gold": [{"doc_id": "finance_policy", "contains": "一线城市住宿上限为每晚 600 元"}],
    }
    assert evaluate_retrieval(knowledge, [task])["recall_at_k"] == 1.0

    without_acl = dict(task, id="finance-no-acl", scopes=["tool:read"])
    assert evaluate_retrieval(knowledge, [without_acl])["recall_at_k"] == 0.0


def test_retrieval_breakdown_by_k_category_and_empty(knowledge):
    tasks = [
        {"id": "lora", "category": "synonym", "input": "LoRA 的秩 r 常用取值是多少？",
         "gold": [{"doc_id": "lora_notes", "contains": "最常用的取值是 8 或 16"}]},
        {"id": "finance", "category": "acl", "input": "一线城市住宿上限是多少？",
         "scopes": ["tool:read", "kb:group:finance"],
         "gold": [{"doc_id": "finance_policy", "contains": "一线城市住宿上限为每晚 600 元"}]},
        {"id": "empty", "category": "out_of_domain", "input": "请推荐几部科幻电影", "expect_empty": True},
    ]
    result = evaluate_retrieval(knowledge, tasks, k=5, ks=(1, 5), gate_thresholds=(0.99,))
    assert result["ks"] == [1, 5]
    assert result["scored_tasks"] == 2
    assert set(result["by_k"]) == {"1", "5"}
    assert set(result["by_category"]) == {"acl", "synonym"}
    assert result["empty_tasks"] == 1
    assert result["empty_ok"] == 1
    assert result["empty_rate"] == 1.0
    assert result["gate_sweep"][0]["min_coverage"] == 0.99

