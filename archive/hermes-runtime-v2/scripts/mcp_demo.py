"""MCP 演示：启动本地自建 MCP Server，把远端工具接进注册表后像本地工具一样调用。

运行：``python scripts/mcp_demo.py``
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hermes.api.app import build_state  # noqa: E402
from hermes.config import Settings  # noqa: E402
from hermes.tools.base import ToolContext  # noqa: E402
from hermes.tools.mcp_bridge import MCPToolBridge  # noqa: E402


async def main() -> None:
    settings = Settings.from_env()
    state = build_state(settings)
    state.knowledge.ingest_file(ROOT / "knowledge_base" / "lora_notes.md",
                                doc_id="lora_notes", version="1")

    bridge = MCPToolBridge(
        state.registry, command=sys.executable,
        args=["-m", "hermes.mcp_server.server"], server_label="kb",
        env={"HERMES_DB_PATH": settings.db_path},
    )
    names = await bridge.start()
    print("MCP Server 已连接，注册工具：", names)
    print("注册表可见工具：", [s.name for s in state.registry.all_specs()])

    ctx = ToolContext(scopes=frozenset({"tool:read"}))
    for name, args in (("mcp__kb__knowledge_stats", {}),
                       ("mcp__kb__kb_search", {"query": "LoRA 的秩 r"})):
        result = await state.registry.execute(name, args, ctx)
        print(f"\n{name} -> ok={result.ok} latency={result.latency_ms:.1f}ms")
        print((result.content or result.error or "")[:300])

    await bridge.stop()
    state.db.close()


if __name__ == "__main__":
    asyncio.run(main())
