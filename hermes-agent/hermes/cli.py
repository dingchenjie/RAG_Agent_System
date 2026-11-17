"""命令行入口：``hermes ask / chat / ingest / eval / serve / lessons``。

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
                            dataset_path=args.dataset, orchestrator=state.orchestrator)
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


def _kb(args: argparse.Namespace) -> int:
    """查看文档历史版本，或把某个历史版本切回线上。"""
    state = build_state(Settings.from_env())
    try:
        if args.activate:
            ok = state.knowledge.activate_version(args.doc_id, args.activate)
            if not ok:
                print(f"版本不存在: {args.doc_id} v{args.activate}")
                return 1
            print(f"已切换到: {args.doc_id} v{args.activate}")
            return 0
        items = state.knowledge.versions(args.doc_id)
        if not items:
            print(f"文档不存在或没有版本: {args.doc_id}")
            return 1
        for item in items:
            flag = "active" if item["active"] else "      "
            print(f"v{item['version']:<8} [{flag}] chunks={item['chunks']:<4} "
                  f"title={item['title'] or '-'}")
        return 0
    finally:
        state.db.close()


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(create_app(Settings.from_env()), host=args.host, port=args.port)
    return 0


def _lessons(args: argparse.Namespace) -> int:
    from .learning.lessons import LessonService

    state = build_state(Settings.from_env())
    try:
        service = LessonService(state.db, tenant_id=args.tenant)
        if args.activate or args.reject:
            lesson_id = args.activate or args.reject
            status = "active" if args.activate else "rejected"
            ok = service.set_status(lesson_id, status)
            print(f"{'已设为 ' + status if ok else '未找到该教训'}: {lesson_id}")
            return 0 if ok else 1
        records = service.list(status=args.status)
        if not records:
            print("（暂无教训：先跑几条失败任务，或换 --status 查看）")
            return 0
        for record in records:
            print(f"{record.id} [{record.status}] {record.rule} hits={record.hits}\n  {record.content}")
        return 0
    finally:
        state.db.close()


def _replay(args: argparse.Namespace) -> int:
    """打印时间线；带 --against 时改为对比两次运行的差异。"""
    from .eval.replay import Replay, diff_traces

    state = build_state(Settings.from_env())
    try:
        if state.runs.get_run(args.run_id) is None:
            print(f"run 不存在: {args.run_id}")
            return 1
        if args.against:
            if state.runs.get_run(args.against) is None:
                print(f"基线 run 不存在: {args.against}")
                return 1
            baseline = state.event_store.history(args.against, 0, 2000)
            candidate = state.event_store.history(args.run_id, 0, 2000)
            diff = diff_traces(baseline, candidate)
            print(f"基线 {args.against} → 候选 {args.run_id}")
            print(f"工具新增: {diff['tools_added'] or '-'} / 工具移除: {diff['tools_removed'] or '-'}")
            print(f"LLM 调用变化: {diff['llm_calls_delta']:+d} / "
                  f"耗时变化: {diff['duration_delta_s']:+.3f}s / "
                  f"工具失败变化: {diff['tool_failures_delta']:+d}")
            print(f"终态是否变化: {'是' if diff['state_changed'] else '否'}")
            return 0
        timeline = Replay(state.event_store).timeline(args.run_id)
        for item in timeline:
            payload = item["payload"] or {}
            detail = payload.get("tool") or payload.get("model") or payload.get("preview") or ""
            print(f"{item['t']:>8.3f}s  {item['type']:<22} {detail}")
        return 0
    finally:
        state.db.close()


def _triggers(args: argparse.Namespace) -> int:
    state = build_state(Settings.from_env())
    try:
        service = state.triggers
        if args.enable or args.disable or args.delete:
            trigger_id = args.enable or args.disable or args.delete
            try:
                service.get(trigger_id)
            except KeyError:
                print(f"触发器不存在: {trigger_id}")
                return 1
            if args.delete:
                service.remove(trigger_id)
                print(f"已删除: {trigger_id}")
            else:
                enabled = bool(args.enable)
                service.set_enabled(trigger_id, enabled)
                print(f"{'已启用' if enabled else '已停用'}: {trigger_id}")
            return 0
        records = service.list(tenant_id=args.tenant)
        if not records:
            print("（暂无触发器：hermes triggers 或 POST /v1/triggers 创建）")
            return 0
        for record in records:
            flag = "on " if record.enabled else "off"
            print(f"{record.id} [{flag}] {record.name} every={record.interval_s}s "
                  f"mode={record.mode} last_run={record.last_run_id or '-'}")
        return 0
    finally:
        state.db.close()


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

    lessons = sub.add_parser("lessons", help="查看/启用/驳回自动提炼的教训")
    lessons.add_argument("--tenant", default="default")
    lessons.add_argument("--status", default=None, choices=["pending", "active", "rejected"])
    lessons.add_argument("--activate", default=None, metavar="LESSON_ID")
    lessons.add_argument("--reject", default=None, metavar="LESSON_ID")

    replay = sub.add_parser("replay", help="回放 run 轨迹，或用 --against 对比两次运行")
    replay.add_argument("run_id")
    replay.add_argument("--against", default=None, metavar="BASELINE_RUN_ID")

    triggers = sub.add_parser("triggers", help="列出/启停/删除周期触发器")
    triggers.add_argument("--tenant", default="default")
    triggers.add_argument("--enable", default=None, metavar="TRIGGER_ID")
    triggers.add_argument("--disable", default=None, metavar="TRIGGER_ID")
    triggers.add_argument("--delete", default=None, metavar="TRIGGER_ID")

    kb = sub.add_parser("kb", help="查看知识文档的历史版本，或回滚到指定版本")
    kb.add_argument("doc_id")
    kb.add_argument("--activate", default=None, metavar="VERSION")

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
    if args.command == "lessons":
        return _lessons(args)
    if args.command == "replay":
        return _replay(args)
    if args.command == "triggers":
        return _triggers(args)
    if args.command == "kb":
        return _kb(args)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
