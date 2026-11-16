"""命令行入口：``hermes ask / chat / ingest / eval / serve``。

CLI 是唯一的人工入口——旧项目有 ``main_agent.py`` / ``main_cli.py`` / ``main_api.py``
三个脚本，合并后统一到这里，避免"演示一条路、服务另一条路"。
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .api.app import AppState, build_state, create_app
from .config import Settings
from .eval.runner import EvalRunner
from .observability.events import EventType

ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "knowledge_base"
DEFAULT_DATASET = ROOT / "eval" / "dataset.jsonl"


def _ensure_seed(state: AppState) -> None:
    """首次运行时把自带的 md 种子文档入库，保证 clone 后 ``hermes ask`` 直接可用。"""
    if state.knowledge.stats().get("documents", 0) > 0 or not SEED_DIR.exists():
        return
    reports = state.knowledge.ingest_dir(SEED_DIR)
    total = sum(report.chunks for report in reports)
    print(f"[seed] 已入库 {len(reports)} 个文档、{total} 个分块")


def _describe_knowledge(state: AppState) -> str:
    info = state.knowledge.describe()
    stats = state.knowledge.stats()
    return (f"embedding={info['embedding']['provider']}({info['embedding']['dim']}d) "
            f"rerank={info['rerank']['provider']} "
            f"docs={stats['documents']} chunks={stats['chunks']}")


async def _ask(state: AppState, question: str, args: argparse.Namespace) -> int:
    _ensure_seed(state)
    result = await state.runtime.run(
        user_input=question, tenant_id=args.tenant, user_id=args.user,
        scopes=frozenset(args.scopes),
    )
    events = state.event_store.history(result.run_id, 0, 2000)
    tools = [event.payload.get("tool", "?") for event in events
             if event.type == EventType.TOOL_CALL_STARTED]

    print(f"\n答案：{result.answer or '（无）'}")
    print(f"\n轨迹：state={result.state} steps={result.steps} tools={tools or '-'}")
    if result.citations:
        print("引用：" + ", ".join(c["citation"] for c in result.citations))
    if result.context:
        print(f"上下文：{result.context.get('final_tokens')} tokens "
              f"(丢弃 {result.context.get('dropped_messages')} 条消息)")
    return 0 if result.state == "succeeded" else 1


def _chat(state: AppState, args: argparse.Namespace) -> int:
    _ensure_seed(state)
    print(f"Hermes REPL · {_describe_knowledge(state)}")
    print("命令：/memory 查看长期记忆 · /ingest <路径> 入库 · /exit 退出")
    session_id = None
    while True:
        try:
            line = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"/exit", "/quit"}:
            return 0
        if line == "/memory":
            records = state.memory.list_memories(tenant_id=args.tenant, user_id=args.user,
                                                 status="active")
            print("\n".join(f"- {r.mkey}: {r.value}" for r in records) or "（暂无长期记忆）")
            continue
        if line.startswith("/ingest "):
            target = Path(line.split(" ", 1)[1].strip())
            reports = (state.knowledge.ingest_dir(target) if target.is_dir()
                       else [state.knowledge.ingest_path(target)])
            print(f"已入库 {len(reports)} 个文档，{sum(r.chunks for r in reports)} 个分块")
            continue

        result = asyncio.run(state.runtime.run(
            user_input=line, session_id=session_id, tenant_id=args.tenant,
            user_id=args.user, scopes=frozenset(args.scopes),
        ))
        session_id = state.runs.get_run(result.run_id)["session_id"]
        print(f"Hermes> {result.answer or '（无答案）'}")


def _eval(args: argparse.Namespace) -> int:
    async def run() -> int:
        state = build_state(Settings.from_env())
        _ensure_seed(state)
        runner = EvalRunner(runtime=state.runtime, event_store=state.event_store,
                            dataset_path=args.dataset)
        report = await runner.run()
        print(report.markdown())
        summary = report.summary()
        state.db.close()
        print(f"\n成功率 {summary['success_rate']:.2%}（门禁 {args.fail_under:.0%}）")
        return 0 if summary["success_rate"] >= args.fail_under else 1

    return asyncio.run(run())


def _ingest(args: argparse.Namespace) -> int:
    state = build_state(Settings.from_env())
    target = Path(args.path)
    reports = (state.knowledge.ingest_dir(target, version=args.version) if target.is_dir()
               else [state.knowledge.ingest_path(target, doc_id=args.doc_id, version=args.version,
                                                 acl=args.acl or [])])
    for report in reports:
        print(f"{report.doc_id} v{report.version}: {report.chunks} chunks / "
              f"{report.chars} chars / {report.duration_ms}ms")
    state.db.close()
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(create_app(Settings.from_env()), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes", description="Hermes Agent 命令行入口")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tenant", default="default")
        p.add_argument("--user", default="cli-user")
        p.add_argument("--scopes", nargs="*",
                       default=["tool:read", "tool:write", "memory:write"])

    ask = sub.add_parser("ask", help="单次提问，打印答案与工具轨迹")
    ask.add_argument("question")
    add_common(ask)

    chat = sub.add_parser("chat", help="交互式 REPL")
    add_common(chat)

    ingest = sub.add_parser("ingest", help="入库文件或目录")
    ingest.add_argument("path")
    ingest.add_argument("--doc-id", default=None)
    ingest.add_argument("--version", default="1")
    ingest.add_argument("--acl", nargs="*", default=None)

    eval_cmd = sub.add_parser("eval", help="离线评测（可做 CI 门禁）")
    eval_cmd.add_argument("--dataset", default=str(DEFAULT_DATASET))
    eval_cmd.add_argument("--fail-under", type=float, default=0.0)

    serve = sub.add_parser("serve", help="启动 HTTP 服务")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ask":
        state = build_state(Settings.from_env())
        try:
            return asyncio.run(_ask(state, args.question, args))
        finally:
            state.db.close()
    if args.command == "chat":
        state = build_state(Settings.from_env())
        try:
            return _chat(state, args)
        finally:
            state.db.close()
    if args.command == "ingest":
        return _ingest(args)
    if args.command == "eval":
        return _eval(args)
    if args.command == "serve":
        return _serve(args)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
