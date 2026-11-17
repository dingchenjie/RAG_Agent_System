"""离线评测：任务成功率、工具选择质量、延迟与成本。"""
from __future__ import annotations

import asyncio
from pathlib import Path

from hermes.api.app import build_state
from hermes.eval.runner import EvalRunner
from hermes.llm.mock import MockLLM
from tests.conftest import ingest_seed_docs

ROOT = Path(__file__).resolve().parents[1]


def test_eval_report_metrics(settings):
    state = build_state(settings, MockLLM())
    ingest_seed_docs(state)

    runner = EvalRunner(runtime=state.runtime, event_store=state.event_store,
                        dataset_path=ROOT / "eval" / "dataset.jsonl",
                        orchestrator=state.orchestrator)
    tasks = runner.load_tasks()
    assert len(tasks) >= 5

    report = asyncio.run(runner.run())
    summary = report.summary()

    assert summary["tasks"] == len(tasks)
    assert summary["success_rate"] >= 0.8
    assert summary["tool_recall"] >= 0.8
    assert summary["avg_steps"] >= 0.5
    assert summary["latency_p50_ms"] > 0
    assert summary["faithfulness_tasks"] >= 1
    assert 0 < summary["faithfulness_avg"] <= 1
    assert "忠实度" in report.markdown()
    assert set(summary["by_mode"]) == {"single", "orchestrate"}
    assert report.meta["dataset_size"] == len(tasks)


def test_single_task_trace_based_tool_check(settings):
    state = build_state(settings, MockLLM())
    ingest_seed_docs(state)
    runner = EvalRunner(runtime=state.runtime, event_store=state.event_store,
                        dataset_path=ROOT / "eval" / "dataset.jsonl")
    result = asyncio.run(runner.run_task({
        "id": "x1", "category": "rag", "input": "LoRA 的秩 r 通常设置为多少？",
        "expected_tools": ["kb_search"], "expected_contains": ["8"],
    }))
    assert result.ok
    assert result.called_tools == ["kb_search"]
    assert result.citations >= 1
    assert result.faithfulness is not None and result.faithfulness > 0


def test_orchestrated_task_aggregates_worker_traces(settings):
    state = build_state(settings, MockLLM())
    ingest_seed_docs(state)
    runner = EvalRunner(runtime=state.runtime, event_store=state.event_store,
                        dataset_path=ROOT / "eval" / "dataset.jsonl",
                        orchestrator=state.orchestrator)
    result = asyncio.run(runner.run_task({
        "id": "o1", "mode": "orchestrate", "category": "orchestrate_rag_calc",
        "input": "LoRA 的秩 r 常用取值是多少？另外计算 8/8 的结果",
        "expected_tools": ["kb_search", "calculator"], "expected_contains": ["8"],
        "expected_subtasks": 2,
    }))
    assert result.mode == "orchestrate"
    assert result.subtasks == 2 and result.subtasks_ok == 2
    assert set(result.expected_tools) <= set(result.called_tools)
    assert result.ok, result.error
