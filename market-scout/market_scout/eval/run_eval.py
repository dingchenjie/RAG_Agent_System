"""离线评测入口：python -m market_scout.eval.run_eval [--provider mock] [--prompt v2]"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile

from ..config import Settings
from ..demo import seed_demo_sources
from ..engine import create_engine, run_question
from .metrics import citation_rate, keyword_coverage, percentile, tool_f1

DATASET = os.path.join(os.path.dirname(__file__), "dataset.json")


async def _run_case(case: dict, provider: str, prompt: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings.from_env()
        settings.llm_provider = provider
        settings.prompt_version = prompt
        settings.db_path = os.path.join(tmp, "eval.db")
        settings.trace_jsonl = os.path.join(tmp, "trace.jsonl")
        settings.output_dir = os.path.join(tmp, "reports")
        engine = create_engine(settings)
        try:
            seed_demo_sources(engine.store, engine.run_id)
            result = await run_question(engine, case["question"])
        finally:
            engine.store.close()  # Windows 上不关连接，临时目录删不掉
        report, meta = result["report"], result["meta"]
        precision, recall, f1 = tool_f1(case["expected_tools"], meta["tools_used"])
        coverage = keyword_coverage(case["keywords"], report)
        cite = citation_rate(report)
        recall_ok = coverage >= 1.0
        # 应拒答用例：正确行为就是给 0 引用，不能用"必须带引用"去判它失败
        cited_ok = cite > 0 if case["type"] != "abstain" else not meta.get("citations")
        return {"id": case["id"], "type": case["type"], "status": result["status"],
                "precision": precision, "recall": recall, "f1": f1,
                "keyword_coverage": coverage, "citation_rate": cite,
                "citations": len(meta.get("citations") or []),
                "loop_detected": meta["loop_detected"],
                "elapsed_s": meta["elapsed_s"], "cost_usd": meta["cost_usd"],
                "success": bool(recall_ok and cited_ok and result["status"] == "succeeded")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="mock", choices=["mock", "openai"])
    parser.add_argument("--prompt", default="v2", choices=["v1", "v2"])
    parser.add_argument("--out", default="data/eval_report.md")
    args = parser.parse_args(argv)

    cases = json.load(open(DATASET, encoding="utf-8"))
    rows = [asyncio.run(_run_case(case, args.provider, args.prompt)) for case in cases]

    total = len(rows)
    success = sum(1 for r in rows if r["success"])
    avg_f1 = round(sum(r["f1"] for r in rows) / total, 4)
    avg_cite = round(sum(r["citation_rate"] for r in rows) / total, 4)
    p50 = percentile([r["elapsed_s"] for r in rows], 50)
    p95 = percentile([r["elapsed_s"] for r in rows], 95)
    cost = round(sum(r["cost_usd"] for r in rows), 6)
    loops = sum(1 for r in rows if r["loop_detected"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = [
        f"# 评测报告（provider={args.provider}, prompt={args.prompt}）",
        "",
        f"- 任务成功率：**{success}/{total} = {success / total:.2%}**",
        f"- 工具 F1 均值：{avg_f1}",
        f"- 引用段落占比：{avg_cite}",
        f"- 延迟 P50 / P95：{p50}s / {p95}s",
        f"- 总成本：${cost}",
        f"- 触发死循环拦截：{loops} 次",
        "",
        "| case | 类型 | 成功 | 工具F1 | 关键字覆盖 | 引用率 | 引用数 | 耗时(s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(f"| {r['id']} | {r['type']} | {'Y' if r['success'] else 'N'} | {r['f1']} | "
                     f"{r['keyword_coverage']} | {r['citation_rate']} | {r['citations']} | {r['elapsed_s']} |")
    report = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report)
    with open(args.out.replace(".md", ".json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": {"success": success, "total": total, "avg_f1": avg_f1,
                               "citation_rate": avg_cite, "p50": p50, "p95": p95,
                               "cost_usd": cost, "loops": loops}, "rows": rows},
                  fh, ensure_ascii=False, indent=2)
    print(report)
    return 0 if success == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
