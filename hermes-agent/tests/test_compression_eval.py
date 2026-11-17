"""压缩评测：关键事实保留率、压缩比与负对照的有效性。"""
from __future__ import annotations

from pathlib import Path

from hermes.eval.compression import evaluate_compression, load_compression_tasks

EVAL = Path(__file__).resolve().parents[1] / "eval"


def test_compression_dataset_measures_retention():
    tasks = load_compression_tasks(EVAL / "compression.jsonl")
    result = evaluate_compression(tasks)

    assert result["cases"] == len(tasks) == 6
    assert result["pass_rate"] == 1.0
    assert result["retention_rate"] > 0.5
    assert result["avg_compression_ratio"] > 0.3

    by_id = {item["task_id"]: item for item in result["details"]}
    assert by_id["cmp-early-fact"]["retained"] is True
    assert by_id["cmp-summary-fact"]["retained"] is True
    assert by_id["cmp-tool-head"]["retained"] is True
    assert by_id["cmp-current-turn"]["retained"] is True
    assert by_id["cmp-injected-memory"]["retained"] is True
    # 负对照：放在摘要窗口之外的事实应当丢失，证明判据能测出损失
    assert by_id["cmp-deep-needle"]["retained"] is False
    assert all(item["final_tokens"] <= item["budget"] for item in result["details"])
