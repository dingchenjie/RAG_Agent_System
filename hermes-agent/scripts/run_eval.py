"""离线评测入口（可做 CI 门禁）。

运行：``python scripts/run_eval.py --fail-under 0.8``
退出码非 0 表示成功率低于阈值，可直接挂到 CI / pre-push 钩子上。

除了端到端任务，还会跑五组独立评测：
- 检索质量：独立索引 + 独立语料，算 recall@k / MRR / nDCG@k、分类指标与门控扫描；
- 注入红队：把攻击载荷放进检索结果、长期记忆、页面上下文，统计抵抗率；
- 上下文压缩：长对话埋关键事实，统计事实保留率与 token 压缩比；
- 长期记忆：召回命中、冲突归档、TTL 过期与跨租户/跨用户隔离；
- 记忆抽取：规则 vs LLM 的 precision / recall / F1 与负例误抽率；
- 教训注入：`mock` 模式下用确定性替身跑"无教训 / 注入教训"的 A/B，验证链路接通。

``--real`` 强制真实模型（provider 不是 mock 且必须有 key）；``--save-baseline``
把成功率、成本、P95 与各专项指标落盘，``--baseline`` 与历史基线对比，成功率
下降超过 5 个百分点时退出码为 1。``--record-failures`` 会把本次失败用例连同
模型/提示词版本追加到 JSONL，下一次迭代直接作为回归集重放。
``--repeats`` 让端到端任务重复多轮并报告均值 ± 标准差；``--rerank-ab`` 额外用
cross-encoder 重排跑一遍检索，对比 recall@k / MRR / nDCG（依赖缺失时跳过）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hermes.api.app import build_llm, build_state  # noqa: E402
from hermes.config import Settings  # noqa: E402
from hermes.eval.compression import evaluate_compression, load_compression_tasks  # noqa: E402
from hermes.eval.extraction import evaluate_extraction, load_extraction_tasks  # noqa: E402
from hermes.eval.injection import evaluate_injection, load_injection_tasks  # noqa: E402
from hermes.eval.lessons import (  # noqa: E402
    LessonAwareLLM,
    evaluate_lessons_ab,
    load_lesson_cases,
)
from hermes.eval.memory import evaluate_memory, load_memory_tasks  # noqa: E402
from hermes.eval.metrics import aggregate_summaries  # noqa: E402
from hermes.eval.orchestration import evaluate_orchestration_failures  # noqa: E402
from hermes.eval.regressions import record_failures  # noqa: E402
from hermes.eval.retrieval import evaluate_retrieval, load_retrieval_tasks  # noqa: E402
from hermes.eval.runner import EvalRunner  # noqa: E402
from hermes.memory.extractor import build_memory_extractor  # noqa: E402

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
    parser.add_argument("--real", action="store_true",
                        help="强制真实模型：provider 为 mock 或缺 API key 时直接失败")
    parser.add_argument("--memory-extractor", choices=("rule", "llm"), default="",
                        help="覆盖 HERMES_MEMORY_EXTRACTOR，用于规则/LLM 抽取 A/B")
    parser.add_argument("--save-baseline", default="",
                        help="把本次结果写入基线 JSON（按 provider/model/prompt_version 记录）")
    parser.add_argument("--baseline", default="",
                        help="与历史基线对比；成功率下降超过 0.05 时退出码 1")
    parser.add_argument("--retrieval-ab", action="store_true",
                        help="额外用 hashing 或云端 embedding 跑一遍检索，输出前后对比")
    parser.add_argument("--rerank-ab", action="store_true",
                        help="额外用 cross-encoder 重排跑一遍检索；依赖不可用时跳过")
    parser.add_argument("--repeats", type=int, default=1,
                        help="端到端评测重复轮数，报告均值 ± 标准差（真实模型建议 3）")
    args = parser.parse_args()

    base = Settings.from_env()
    if args.memory_extractor:
        base = replace(base, memory_extractor=args.memory_extractor)
    if args.real:
        problem = _real_guard(base)
        if problem:
            print(problem, file=sys.stderr)
            return 2

    state = build_state(base)
    state.knowledge.ingest_path(KB / "lora_notes.md", doc_id="lora_notes", version="1")
    state.knowledge.ingest_path(KB / "finance_policy.md", doc_id="finance_policy",
                                version="1", acl=["finance"])

    runner = EvalRunner(runtime=state.runtime, event_store=state.event_store,
                        dataset_path=args.dataset, orchestrator=state.orchestrator)
    tasks = runner.load_tasks()
    rounds = max(1, args.repeats)
    reports = []
    for index in range(rounds):
        reports.append(await runner.run())
        if rounds > 1:
            item = reports[-1].summary()
            print(f"第 {index + 1}/{rounds} 轮：成功率 {item['success_rate']:.2%} / "
                  f"P95 {item['latency_p95_ms']:.0f}ms / 成本 ${item['total_cost_usd']:.6f}")
    report = reports[0]
    if rounds == 1:
        print(report.markdown())
        summary = report.summary()
    else:
        summary = aggregate_summaries([item.summary() for item in reports])
        print(f"\n{rounds} 轮汇总：成功率 {summary['success_rate']:.2%} ± "
              f"{summary['success_rate_std']:.2%}"
              f"（min {summary['success_rate_min']:.2%} / max {summary['success_rate_max']:.2%}） / "
              f"P95 {summary['latency_p95_ms']:.0f} ± {summary['latency_p95_ms_std']:.0f}ms / "
              f"成本 ${summary['total_cost_usd']:.6f} ± ${summary['total_cost_usd_std']:.6f}")

    retrieval = _run_retrieval(base)
    _print_retrieval(retrieval)

    injection = await _run_injection(base)
    _print_injection(injection)

    compression = _run_compression(base)
    _print_compression(compression)

    memory = _run_memory(base)
    _print_memory(memory)

    extraction = await _run_extraction(base)
    _print_extraction(extraction)

    lessons_ab = None
    orchestration_failures = None
    if base.llm_provider == "mock":
        lessons_ab = await _run_lessons_ab(base)
        _print_lessons_ab(lessons_ab)
        orchestration_failures = await _run_orchestration_failures(base)
        _print_orchestration_failures(orchestration_failures)

    faith = summary["faithfulness_avg"]
    print(f"\n成功率 {summary['success_rate']:.2%} / 工具 F1 {summary['tool_f1']:.4f} / "
          f"忠实度 {'-' if faith is None else f'{faith:.4f}'} / "
          f"P95 {summary['latency_p95_ms']:.0f}ms / 成本 ${summary['total_cost_usd']:.6f}")

    retrieval_ab = None
    if args.retrieval_ab:
        retrieval_ab = _run_retrieval_ab(base, retrieval)
    rerank_ab = None
    if args.rerank_ab:
        rerank_ab = _run_rerank_ab(base, retrieval)

    if args.record_failures:
        written = record_failures(report.to_dict(), tasks=tasks, path=args.record_failures)
        print(f"失败样本：新增 {written} 条 → {args.record_failures}")

    baseline_payload = _baseline_payload(report, retrieval, extraction, base, summary=summary)
    # 先对比再落盘：两者指向同一文件时，比较的应是上一轮的基线
    baseline_ok = True
    if args.baseline:
        baseline_ok = _compare_baseline(baseline_payload, Path(args.baseline))
    if args.save_baseline:
        path = Path(args.save_baseline)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(baseline_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"基线已写入 {path}（{baseline_payload['provider']}/{baseline_payload['model']}）")

    if args.json_out:
        payload = report.to_dict()
        payload["summary"] = summary
        if rounds > 1:
            payload["rounds"] = [item.summary() for item in reports]
        payload["retrieval"] = retrieval
        payload["injection"] = injection
        payload["compression"] = compression
        payload["memory"] = memory
        payload["extraction"] = extraction
        if lessons_ab:
            payload["lessons_ab"] = lessons_ab
        if orchestration_failures:
            payload["orchestration_failures"] = orchestration_failures
        if retrieval_ab:
            payload["retrieval_ab"] = retrieval_ab
        if rerank_ab:
            payload["rerank_ab"] = rerank_ab
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细已写入 {args.json_out}")

    state.db.close()
    if summary["success_rate"] < args.fail_under:
        return 1
    return 0 if baseline_ok else 1


def _real_guard(settings: Settings) -> str | None:
    if settings.llm_provider == "mock":
        return "已指定 --real，但 HERMES_LLM_PROVIDER 仍为 mock；请设置为 zhipu 等真实后端。"
    if not settings.llm_api_key:
        return ("已指定 --real，但缺少 API key；请设置 ZHIPU_API_KEY 或 "
                "HERMES_LLM_API_KEY。")
    return None


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


def _run_retrieval_ab(base: Settings, primary: dict) -> dict | None:
    """对比 hashing 与当前 embedding 后端；后端不可用时跳过而不是中断评测。"""
    alt_backend = "hashing" if base.embedding_backend != "hashing" else "openai"
    if alt_backend == base.embedding_backend:
        return None
    try:
        alt = _run_retrieval(replace(base, embedding_backend=alt_backend))
    except Exception as exc:  # noqa: BLE001 外部 embedding 不可用不应影响主评测
        print(f"\n检索 A/B 跳过：{alt_backend} 后端不可用（{exc}）")
        return None
    _print_retrieval_ab(primary, alt, primary_name=base.embedding_backend, alt_name=alt_backend)
    return {
        "primary": {"backend": base.embedding_backend,
                    "recall_at_k": primary["recall_at_k"], "mrr": primary["mrr"],
                    "ndcg_at_k": primary["ndcg_at_k"]},
        "alternative": {"backend": alt_backend,
                        "recall_at_k": alt["recall_at_k"], "mrr": alt["mrr"],
                        "ndcg_at_k": alt["ndcg_at_k"]},
    }


def _run_rerank_ab(base: Settings, primary: dict) -> dict | None:
    """对比 RRF 截断与 cross-encoder 重排；可选依赖缺失时跳过而不是中断评测。"""
    primary_backend = base.rerank_backend
    alt_backend = "rrf" if primary_backend == "cross-encoder" else "cross-encoder"
    try:
        alt = _run_retrieval(replace(base, rerank_backend=alt_backend))
    except Exception as exc:  # noqa: BLE001 可选依赖不可用不应影响主评测
        print(f"\n重排 A/B 跳过：{alt_backend} 不可用（{exc}）")
        return None
    _print_rerank_ab(primary, alt, primary_name=primary_backend, alt_name=alt_backend)
    metrics = ("recall_at_k", "mrr", "ndcg_at_k")
    return {
        "primary": {"backend": primary_backend, **{key: primary[key] for key in metrics},
                    "by_k": primary["by_k"]},
        "alternative": {"backend": alt_backend, **{key: alt[key] for key in metrics},
                        "by_k": alt["by_k"]},
    }


async def _run_extraction(base: Settings) -> dict:
    extractor = build_memory_extractor(base.memory_extractor, build_llm(base),
                                       min_confidence=base.memory_min_confidence)
    tasks = load_extraction_tasks(EVAL / "extraction.jsonl")
    result = await evaluate_extraction(extractor.extract_async, tasks)
    result["backend"] = base.memory_extractor
    return result


async def _run_lessons_ab(base: Settings) -> dict:
    """教训注入 A/B：独立临时库 + 确定性替身，只验证链路，不冒充实测效果。"""
    with tempfile.TemporaryDirectory() as tmp:
        state = build_state(_isolated(base, tmp, "lessons"), LessonAwareLLM())
        try:
            state.knowledge.ingest_path(KB / "lora_notes.md", doc_id="lora_notes", version="1")
            runner = EvalRunner(runtime=state.runtime, event_store=state.event_store,
                                dataset_path=EVAL / "lessons_ab.jsonl")
            return await evaluate_lessons_ab(
                runner=runner, lessons=state.lessons,
                cases=load_lesson_cases(EVAL / "lessons_ab.jsonl"),
            )
        finally:
            state.db.close()


async def _run_orchestration_failures(base: Settings) -> dict:
    """编排失败隔离：故障注入两条场景，验证部分失败降级与全部失败显式报错。"""
    with tempfile.TemporaryDirectory() as tmp:
        state = build_state(_isolated(base, tmp, "orch"))
        try:
            state.knowledge.ingest_path(KB / "lora_notes.md", doc_id="lora_notes", version="1")
            return await evaluate_orchestration_failures(state)
        finally:
            state.db.close()


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


def _run_compression(base: Settings) -> dict:
    tasks = load_compression_tasks(EVAL / "compression.jsonl")
    return evaluate_compression(tasks, cfg=base.context)


def _run_memory(base: Settings) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        state = build_state(_isolated(base, tmp, "memory"))
        try:
            return evaluate_memory(state.memory, load_memory_tasks(EVAL / "memory.jsonl"))
        finally:
            state.db.close()


def _isolated(base: Settings, tmp: str, name: str) -> Settings:
    return replace(
        base,
        db_path=str(Path(tmp) / f"{name}.db"),
        upload_dir=str(Path(tmp) / f"{name}-uploads"),
    )


def _baseline_payload(report, retrieval: dict, extraction: dict, base: Settings,
                      *, summary: dict | None = None) -> dict:
    summary = summary or report.summary()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": base.llm_provider,
        "model": base.llm_model,
        "prompt_version": report.meta.get("prompt_version", ""),
        "embedding": base.embedding_backend,
        "memory_extractor": base.memory_extractor,
        "tasks": summary.get("tasks", 0),
        "repeats": summary.get("repeats", 1),
        "success_rate": summary.get("success_rate", 0.0),
        "tool_f1": summary.get("tool_f1", 0.0),
        "latency_p95_ms": summary.get("latency_p95_ms", 0.0),
        "total_cost_usd": summary.get("total_cost_usd", 0.0),
        "retrieval": {key: retrieval[key] for key in ("recall_at_k", "mrr", "ndcg_at_k")},
        "extraction": {key: extraction[key] for key in ("precision", "recall", "f1")},
    }
    for key in ("success_rate", "latency_p95_ms", "total_cost_usd"):
        for suffix in ("std", "min", "max"):
            name = f"{key}_{suffix}"
            if name in summary:
                payload[name] = summary[name]
    return payload


def _compare_baseline(current: dict, path: Path) -> bool:
    baseline = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n基线对比（{path}，{baseline.get('provider')}/{baseline.get('model')}）：")
    if (baseline.get("provider"), baseline.get("model")) != (current["provider"], current["model"]):
        print("  注意：provider/model 与基线不同，成功率对比仅供参考。")
    delta = current["success_rate"] - float(baseline.get("success_rate", 0.0))
    print(f"  成功率 {baseline.get('success_rate', 0.0):.2%} → {current['success_rate']:.2%}"
          f"（{delta:+.2%}）")
    print(f"  P95 {baseline.get('latency_p95_ms', 0.0):.0f}ms → {current['latency_p95_ms']:.0f}ms")
    print(f"  成本 ${baseline.get('total_cost_usd', 0.0):.6f} → ${current['total_cost_usd']:.6f}")
    if delta < -0.05:
        print("  回归：成功率下降超过 5 个百分点。")
        return False
    return True


def _print_retrieval(result: dict) -> None:
    by_k = " / ".join(
        f"recall@{k} {result['by_k'][str(k)]['recall_at_k']:.2%}" for k in result["ks"]
    )
    print(f"\n检索质量（独立语料，{result['scored_tasks']} 条计分）：{by_k} / "
          f"MRR {result['mrr']:.4f} / nDCG@5 {result['ndcg_at_k']:.4f}")
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


def _print_retrieval_ab(primary: dict, alt: dict, *, primary_name: str, alt_name: str) -> None:
    print(f"\n检索 A/B（{primary_name} vs {alt_name}）：")
    for label, key in (("recall@5", "recall_at_k"), ("MRR", "mrr"), ("nDCG@5", "ndcg_at_k")):
        print(f"  {label} {primary[key]:.4f} → {alt[key]:.4f}（{alt[key] - primary[key]:+.4f}）")


def _print_rerank_ab(primary: dict, alt: dict, *, primary_name: str, alt_name: str) -> None:
    k = primary["k"]
    print(f"\n重排 A/B（{primary_name} vs {alt_name}）：")
    for label, key in ((f"recall@{k}", "recall_at_k"), ("MRR", "mrr"),
                       (f"nDCG@{k}", "ndcg_at_k")):
        print(f"  {label} {primary[key]:.4f} → {alt[key]:.4f}（{alt[key] - primary[key]:+.4f}）")
    print("  recall@k 曲线：" + "；".join(
        f"k={kk} {primary['by_k'][str(kk)]['recall_at_k']:.2%} → "
        f"{alt['by_k'][str(kk)]['recall_at_k']:.2%}"
        for kk in primary["ks"]
    ))


def _print_injection(result: dict) -> None:
    print(f"\n注入红队：抵抗率 {result['resistance_rate']:.2%}"
          f"（{result['resisted']}/{result['cases']}），"
          f"越权工具调用 {result['unsafe_tool_calls']}，标记泄漏 {result['leaks']}")
    print("分类：" + "；".join(
        f"{name} {item['resistance_rate']:.2%}（{item['cases']} 条）"
        for name, item in result["by_category"].items()
    ))


def _print_compression(result: dict) -> None:
    print(f"\n压缩评测：通过率 {result['pass_rate']:.2%}（{result['passed']}/{result['cases']}），"
          f"事实保留 {result['retained']}/{result['cases']}，"
          f"平均压缩比 {result['avg_compression_ratio']:.2%}，"
          f"平均上下文 {result['avg_final_tokens']:.0f} tokens")
    print("分类：" + "；".join(
        f"{name} {item['pass_rate']:.0%}（压缩比 {item['avg_compression_ratio']:.0%}）"
        for name, item in result["by_category"].items()
    ))


def _print_memory(result: dict) -> None:
    print(f"\n记忆评测：通过率 {result['pass_rate']:.2%}"
          f"（{result['passed']}/{result['cases']}）")
    print("分类：" + "；".join(
        f"{name} {item['pass_rate']:.0%}（{item['cases']} 条）"
        for name, item in result["by_category"].items()
    ))


def _print_orchestration_failures(result: dict) -> None:
    print(f"\n编排失败隔离：{result['passed']}/{result['scenarios']} 通过"
          "（部分失败降级成功 / 全部失败显式报错，故障注入）")


def _print_lessons_ab(result: dict) -> None:
    print(f"\n教训注入 A/B（离线替身，{result['cases']} 条用例）：成功率 "
          f"{result['before_success_rate']:.2%} → {result['after_success_rate']:.2%}"
          f"（{result['delta']:+.2%}），注入教训 {result['lessons_injected']} 条")


def _print_extraction(result: dict) -> None:
    print(f"\n记忆抽取（{result['backend']}）：精确率 {result['precision']:.2%} / "
          f"召回率 {result['recall']:.2%} / F1 {result['f1']:.2%}，"
          f"负例误抽率 {result['false_positive_rate']:.2%}，"
          f"沉默准确率 {result['abstain_accuracy']:.2%}"
          f"（{result['negative_cases']} 条负例）")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
