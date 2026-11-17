"""离线评测入口（可做 CI 门禁）。

运行：``python scripts/run_eval.py --fail-under 0.8``
退出码非 0 表示成功率低于阈值，可直接挂到 CI / pre-push 钩子上。

除了端到端任务，还会跑两组独立评测：
- 检索质量：独立索引 + 独立语料，算 recall@k / MRR、分类指标与门控扫描；
- 注入红队：把攻击载荷放进检索结果、长期记忆、页面上下文，统计抵抗率。

``--record-failures`` 会把本次失败用例连同模型/提示词版本追加到 JSONL，
下一次迭代直接作为回归集重放。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hermes.api.app import build_state  # noqa: E402
from hermes.config import Settings  # noqa: E402
from hermes.eval.injection import evaluate_injection, load_injection_tasks  # noqa: E402
from hermes.eval.regressions import record_failures  # noqa: E402
from hermes.eval.retrieval import evaluate_retrieval, load_retrieval_tasks  # noqa: E402
from hermes.eval.runner import EvalRunner  # noqa: E402

KB = ROOT / "knowledge_base"
EVAL = ROOT / "eval"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(EVAL / "dataset.jsonl"))
    parser.add_argument("--fail-under", type=float, default=0.0,
                        help="任务成功率下限，低于则退出码 1")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--record-failures", nargs="?", default="",
                        const=str(EVAL / "regressions" / "failures.jsonl"),
                        help="把失败任务追加到 JSONL；不带路径时写入默认位置")
    args = parser.parse_args()

    base = Settings.from_env()
    state = build_state(base)
    state.knowledge.ingest_path(KB / "lora_notes.md", doc_id="lora_notes", version="1")
    state.knowledge.ingest_path(KB / "finance_policy.md", doc_id="finance_policy",
                                version="1", acl=["finance"])

    runner = EvalRunner(runtime=state.runtime, event_store=state.event_store,
                        dataset_path=args.dataset, orchestrator=state.orchestrator)
    tasks = runner.load_tasks()
    report = await runner.run()
    print(report.markdown())

    retrieval = _run_retrieval(base)
    _print_retrieval(retrieval)

    injection = await _run_injection(base)
    _print_injection(injection)

    summary = report.summary()
    faith = summary["faithfulness_avg"]
    print(f"\n成功率 {summary['success_rate']:.2%} / 工具 F1 {summary['tool_f1']:.4f} / "
          f"忠实度 {'-' if faith is None else f'{faith:.4f}'} / "
          f"P95 {summary['latency_p95_ms']:.0f}ms / 成本 ${summary['total_cost_usd']:.6f}")

    if args.record_failures:
        written = record_failures(report.to_dict(), tasks=tasks, path=args.record_failures)
        print(f"失败样本：新增 {written} 条 → {args.record_failures}")

    if args.json_out:
        payload = report.to_dict()
        payload["retrieval"] = retrieval
        payload["injection"] = injection
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细已写入 {args.json_out}")

    state.db.close()
    return 0 if summary["success_rate"] >= args.fail_under else 1


def _run_retrieval(base: Settings) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        state = build_state(_isolated(base, tmp, "retrieval"))
        corpus = json.loads((EVAL / "retrieval_corpus.json").read_text(encoding="utf-8"))
        for doc in corpus:
            if doc.get("path"):
                state.knowledge.ingest_path(
                    ROOT / doc["path"], doc_id=doc["doc_id"], acl=doc.get("acl") or [],
                )
            else:
                state.knowledge.ingest_text(
                    doc["doc_id"], doc["text"], title=doc.get("title"),
                    acl=doc.get("acl") or [],
                )
        tasks = load_retrieval_tasks(EVAL / "retrieval.jsonl")
        result = evaluate_retrieval(
            state.knowledge, tasks, k=5, ks=(1, 3, 5),
            gate_thresholds=(0.0, 0.1, 0.2, 0.34, 0.5, 0.7),
        )
        state.db.close()
        return result


async def _run_injection(base: Settings) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        state = build_state(_isolated(base, tmp, "injection"))
        tasks = load_injection_tasks(EVAL / "injection.jsonl")
        result = await evaluate_injection(
            state.runtime, state.event_store, tasks,
            knowledge=state.knowledge, memory=state.memory,
        )
        state.db.close()
        return result


def _isolated(base: Settings, tmp: str, name: str) -> Settings:
    return replace(
        base,
        db_path=str(Path(tmp) / f"{name}.db"),
        upload_dir=str(Path(tmp) / f"{name}-uploads"),
    )


def _print_retrieval(result: dict) -> None:
    by_k = " / ".join(
        f"recall@{k} {result['by_k'][str(k)]['recall_at_k']:.2%}" for k in result["ks"]
    )
    print(f"\n检索质量（独立语料，{result['scored_tasks']} 条计分）：{by_k} / "
          f"MRR {result['mrr']:.4f}")
    print("分类：" + "；".join(
        f"{name} {item['recall_at_k']:.2%}（{item['scored_tasks']} 条）"
        for name, item in result["by_category"].items()
    ))
    if result["empty_tasks"]:
        print(f"越界问题返回空结果：{result['empty_ok']}/{result['empty_tasks']}"
              f"（{result['empty_rate']:.2%}）")
    if result["gate_sweep"]:
        print("门控扫描：" + "；".join(
            f"min_coverage={item['min_coverage']}: recall@5 {item['recall_at_k']:.2%}"
            for item in result["gate_sweep"]
        ))


def _print_injection(result: dict) -> None:
    print(f"\n注入红队：抵抗率 {result['resistance_rate']:.2%}"
          f"（{result['resisted']}/{result['cases']}），"
          f"越权工具调用 {result['unsafe_tool_calls']}，标记泄漏 {result['leaks']}")
    print("分类：" + "；".join(
        f"{name} {item['resistance_rate']:.2%}（{item['cases']} 条）"
        for name, item in result["by_category"].items()
    ))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
