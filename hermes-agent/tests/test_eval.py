"""离线评测：任务成功率、工具选择质量、延迟与成本。"""
from __future__ import annotations

import asyncio
from pathlib import Path

from hermes.api.app import build_state
from hermes.eval.metrics import aggregate_summaries
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


def test_aggregate_summaries_reports_mean_and_std():
    """多轮评测要报均值与标准差，避免把单次真实模型结果当成稳定值。"""
    merged = aggregate_summaries([
        {"success_rate": 1.0, "tool_f1": 0.9, "latency_p95_ms": 100.0,
         "total_cost_usd": 0.02, "faithfulness_avg": 0.8},
        {"success_rate": 0.5, "tool_f1": 0.7, "latency_p95_ms": 300.0,
         "total_cost_usd": 0.04, "faithfulness_avg": 0.6},
    ])
    assert merged["repeats"] == 2
    assert merged["success_rate"] == 0.75
    assert merged["success_rate_std"] == 0.25
    assert merged["success_rate_min"] == 0.5
    assert merged["success_rate_max"] == 1.0
    assert merged["latency_p95_ms"] == 200.0
    assert merged["total_cost_usd"] == 0.03
    assert merged["faithfulness_avg"] == 0.7


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


def test_keyword_match_ignores_whitespace_differences():
    """真实模型会把 "10 个工作日" 写成 "10个工作日内"，不应判失败。"""
    runner = EvalRunner(runtime=None, event_store=None, dataset_path=Path("unused"))
    result = runner._score({"id": "ws", "expected_contains": ["10 个工作日"]},
                           mode="single", started=0.0, state="succeeded",
                           answer="需要在出差结束后10个工作日内提交。", events=[], steps=1,
                           tokens=0, cost_usd=0.0, citations=0, error=None)
    assert result.ok


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
