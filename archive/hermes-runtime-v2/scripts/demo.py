"""端到端演示：一次多步工具调用 + 一次主从多 Agent 协作。

运行：``python scripts/demo.py``
默认使用 MockLLM（离线可跑）；设置 ``HERMES_LLM_PROVIDER=zhipu`` 与 ``ZHIPU_API_KEY`` 可切真实模型。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hermes.api.app import build_state  # noqa: E402
from hermes.config import Settings  # noqa: E402

KB = ROOT / "knowledge_base"


def ingest(state) -> None:
    state.knowledge.ingest_file(KB / "lora_notes.md", doc_id="lora_notes", version="1")
    state.knowledge.ingest_file(KB / "finance_policy.md", doc_id="finance_policy",
                                version="1", acl=["finance"])


def print_events(state, run_id: str) -> None:
    for event in state.event_store.history(run_id):
        payload = event.payload
        brief = ""
        if event.type.value == "llm.response":
            brief = f"tool_calls={[c['name'] for c in payload.get('tool_calls', [])]}"
        elif event.type.value.startswith("tool."):
            brief = f"{payload.get('tool')} {payload.get('code') or ''}"
        elif event.type.value == "context.built":
            brief = (f"tokens {payload['final_tokens']}/{payload['budget']} "
                     f"压缩 {payload['compression_ratio']:.0%}")
        elif event.type.value == "context.compressed":
            brief = f"丢弃 {payload['dropped_messages']} 条"
        elif event.type.value == "llm.retry":
            brief = f"第 {payload['attempt']} 次重试"
        print(f"  [{event.seq:>2}] {event.type.value:<22} {brief}")


async def main() -> None:
    settings = Settings.from_env()
    state = build_state(settings)
    ingest(state)
    print(f"LLM Provider: {state.runtime.llm.name}   知识库: {state.knowledge.stats()}\n")

    question = "LoRA 的秩 r 一般设为多少？如果 r=8、alpha=8，计算 8/8 的结果"
    print(f"[单 Agent] 问题：{question}")
    result = await state.runtime.run(user_input=question)
    print(f"  状态：{result.state}  步数：{result.steps}  引用：{len(result.citations)}")
    print(f"  回答：{(result.answer or '').strip()[:220]}\n")
    print("  事件轨迹：")
    print_events(state, result.run_id)

    print("\n[多 Agent] 主 Agent 规划 → 并行子 Agent → 汇总")
    orch = await state.orchestrator.run(goal="根据知识库解释 LoRA 的秩 r，并计算 32/4 的结果")
    print(f"  规划：{[(s['id'], s['kind'], s['tools']) for s in orch.plan.to_dict()['subtasks']]}")
    for worker in orch.workers:
        print(f"  子 Agent {worker.subtask_id} ({worker.agent}) ok={worker.ok} "
              f"steps={worker.steps} → {worker.answer[:80]}")
    print(f"  黑板：{list(orch.blackboard)}")
    print(f"  最终答复：{orch.answer[:220]}")

    state.db.close()


if __name__ == "__main__":
    asyncio.run(main())
