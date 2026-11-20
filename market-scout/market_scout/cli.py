"""命令行入口：scout ask / demo / serve / eval / runs / events。"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .config import Settings
from .engine import create_engine, run_question


def _print_result(result: dict[str, Any]) -> None:
    meta = result["meta"]
    print(f"\n[run {result['run_id']}] status={result['status']} "
          f"mode={meta.get('mode')} pages={meta.get('pages_used')} "
          f"tools={','.join(meta.get('tools_used') or []) or '-'} "
          f"loop={meta.get('loop_detected')} tokens={meta.get('tokens')} "
          f"cost=${meta.get('cost_usd')} elapsed={meta.get('elapsed_s')}s")
    if meta.get("report_path"):
        print(f"report: {meta['report_path']}")
    print("\n" + (result["report"] or meta.get("error") or "(空)"))


def _cmd_ask(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.provider:
        settings.llm_provider = args.provider
    engine = create_engine(settings)
    if settings.llm_provider == "mock":
        from .demo import seed_demo_sources

        # mock 只能检索已入库页面，先种入内置竞品资料，否则会如实"查不到"
        seed_demo_sources(engine.store, engine.run_id)
    result = asyncio.run(run_question(engine, args.question, require_approval=args.approve))
    _print_result(result)
    if result["status"] == "paused":
        answer = input("\n报告已生成，是否发布？[y/N] ").strip().lower()
        if answer == "y":
            from .engine import resume_run

            _print_result(asyncio.run(resume_run(engine)))
    return 0 if result["status"] in ("succeeded", "paused") else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import seed_demo_sources

    settings = Settings.from_env()
    settings.llm_provider = "mock"
    engine = create_engine(settings)
    seeds = seed_demo_sources(engine.store, engine.run_id)
    engine.tracer.emit("demo.seeded", sources=[s["source_id"] for s in seeds])
    question = args.question or "对比 Acme CRM 与 Beta CRM 的定价与核心功能，给出选型建议"
    result = asyncio.run(run_question(engine, question))
    _print_result(result)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.app import create_app

    settings = Settings.from_env()
    if args.provider:
        settings.llm_provider = args.provider
    uvicorn.run(create_app(settings), host=settings.api_host, port=settings.api_port)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .eval.run_eval import main as eval_main

    return eval_main(["--provider", args.provider or "mock"])


def _cmd_runs(_: argparse.Namespace) -> int:
    settings = Settings.from_env()
    engine = create_engine(settings)
    for row in engine.store.list_runs(20):
        print(f"{row['id']}  {row['status']:<10} {row['mode']:<12} {row['question'][:60]}")
    return 0


def _cmd_events(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    engine = create_engine(settings)
    for event in engine.store.list_events(args.run_id):
        print(f"{event['seq']:>3}  {event['type']:<20} {json.dumps(event['data'], ensure_ascii=False)[:160]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scout", description="竞品调研 Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="发起一次调研")
    ask.add_argument("question")
    ask.add_argument("--provider", choices=["mock", "openai"])
    ask.add_argument("--approve", action="store_true", help="发布前人工审批（LangGraph interrupt）")
    ask.set_defaults(func=_cmd_ask)

    demo = sub.add_parser("demo", help="离线演示：内置证据 + mock 模型")
    demo.add_argument("--question")
    demo.set_defaults(func=_cmd_demo)

    serve = sub.add_parser("serve", help="启动 HTTP API 与 Web 页面")
    serve.add_argument("--provider", choices=["mock", "openai"])
    serve.set_defaults(func=_cmd_serve)

    ev = sub.add_parser("eval", help="运行评测集")
    ev.add_argument("--provider", choices=["mock", "openai"])
    ev.set_defaults(func=_cmd_eval)

    runs = sub.add_parser("runs", help="列出最近的 run")
    runs.set_defaults(func=_cmd_runs)

    events = sub.add_parser("events", help="打印某个 run 的事件流")
    events.add_argument("run_id")
    events.set_defaults(func=_cmd_events)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
