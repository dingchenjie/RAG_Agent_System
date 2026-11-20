"""MCP Server：把浏览器/抓取/检索能力以标准 MCP 工具暴露给任意客户端。

运行：python -m market_scout.mcp_server.server
配置：SCOUT_DB_PATH 指向同一个数据库，证据写入 run_id="mcp" 的命名空间。
"""
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from ..config import Settings
from ..store import Store
from ..tools.browser import build_browser_tool
from ..tools.evidence import search_evidence
from ..tools.registry import ToolContext, ToolRegistry
from ..tools.web import build_fetch_tool

settings = Settings.from_env()
store = Store(settings.db_path)
registry = ToolRegistry(settings)
registry.register(build_fetch_tool(settings))
registry.register(build_browser_tool(settings))

mcp = FastMCP("market-scout")


def _context() -> ToolContext:
    from ..trace import Tracer

    return ToolContext(store=store, run_id="mcp", tracer=Tracer(store, "mcp", settings),
                       settings=settings)


@mcp.tool()
async def fetch_page(url: str) -> str:
    """抓取静态网页并抽取正文，返回 JSON（含 citation 编号与正文）。"""
    result = await registry.execute("fetch_page", {"url": url}, _context())
    return json.dumps(result, ensure_ascii=False)[:6000]


@mcp.tool()
async def browse_page(url: str) -> str:
    """用无头浏览器打开需要 JS 渲染的页面，返回 JSON（含正文与截图路径）。"""
    result = await registry.execute("browse_page", {"url": url}, _context())
    return json.dumps(result, ensure_ascii=False)[:6000]


@mcp.tool(name="search_evidence")
async def mcp_search_evidence(query: str, k: int = 4) -> str:
    """在已抓取的页面中检索证据，返回带 [S<编号>] 的片段。"""
    result = await search_evidence(_context(), {"query": query, "k": k})
    return json.dumps(result, ensure_ascii=False)[:4000]


if __name__ == "__main__":
    mcp.run()
