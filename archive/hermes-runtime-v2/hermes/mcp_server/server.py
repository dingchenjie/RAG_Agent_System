"""自建 MCP Server（stdio）。

暴露知识库与记忆能力，供任意 MCP 客户端（包括本项目的 MCPToolBridge）连接。
启动：``python -m hermes.mcp_server.server``
"""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from ..config import Settings
from ..knowledge.embedding import HashingEmbedder
from ..knowledge.index import KnowledgeIndex
from ..knowledge.service import KnowledgeService
from ..memory.service import MemoryService
from ..storage.db import Database

mcp = FastMCP("hermes-knowledge")

_settings = Settings.from_env()
_db = Database(_settings.db_path)
_embedder = HashingEmbedder(_settings.embedding_dim)
_knowledge = KnowledgeService(KnowledgeIndex(_db, _embedder), default_top_k=_settings.rerank_top_k)
_memory = MemoryService(_db, embedder=_embedder, ttl_days=_settings.memory_ttl_days)
_default_tenant = os.getenv("HERMES_TENANT_ID", "default")
_default_user = os.getenv("HERMES_USER_ID", "anonymous")


@mcp.tool()
def kb_search(query: str, top_k: int = 4) -> str:
    """检索 Hermes 知识库，返回带来源引用的片段。"""
    hits = _knowledge.search(query, top_k=top_k, acl_groups=None)
    if not hits:
        return "知识库中没有检索到相关内容。"
    return _knowledge.render_context(hits)


@mcp.tool()
def memory_search(query: str, top_k: int = 4) -> str:
    """检索当前用户的长期记忆。"""
    recalled = _memory.recall(tenant_id=_default_tenant, user_id=_default_user, query=query, top_k=top_k)
    if not recalled:
        return "没有找到相关长期记忆。"
    return "\n".join(f"- {item.record.mkey}: {item.record.value}" for item in recalled)


@mcp.tool()
def knowledge_stats() -> str:
    """返回知识库统计信息。"""
    stats = _knowledge.stats()
    return f"documents={stats['documents']} chunks={stats['chunks']}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
