"""离线评测入口（可做 CI 门禁）。

运行：``python scripts/run_eval.py --fail-under 0.8``
退出码非 0 表示成功率低于阈值，可直接挂到 CI / pre-push 钩子上。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hermes.api.app import build_state  # noqa: E402
from hermes.config import Settings  # noqa: E402
from hermes.eval.runner import EvalRunner  # noqa: E402

KB = ROOT / "knowledge_base"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "eval" / "dataset.jsonl"))
    parser.add_argument("--fail-under", type=float, default=0.0,
                        help="任务成功率下限，低于则退出码 1")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    state = build_state(Settings.from_env())
    state.knowledge.ingest_path(KB / "lora_notes.md", doc_id="lora_notes", version="1")
    state.knowledge.ingest_path(KB / "finance_policy.md", doc_id="finance_policy",
                                version="1", acl=["finance"])

    runner = EvalRunner(runtime=state.runtime, event_store=state.event_store,
                        dataset_path=args.dataset, orchestrator=state.orchestrator)
    report = await runner.run()
    print(report.markdown())

    summary = report.summary()
    faith = summary["faithfulness_avg"]
    print(f"\n成功率 {summary['success_rate']:.2%} / 工具 F1 {summary['tool_f1']:.4f} / "
          f"忠实度 {'-' if faith is None else f'{faith:.4f}'} / "
          f"P95 {summary['latency_p95_ms']:.0f}ms / 成本 ${summary['total_cost_usd']:.6f}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细已写入 {args.json_out}")

    state.db.close()
    return 0 if summary["success_rate"] >= args.fail_under else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
