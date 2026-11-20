"""知识库接入检查：工具契约、服务装配、检索与 ACL 的真实链路。"""
from __future__ import annotations

import asyncio

from hermes.tools.base import ToolContext
from tests.conftest import ingest_seed_docs


def test_kb_tools_are_registered_with_expected_contract(state):
    search = state.registry.get("kb_search")
    ingest = state.registry.get("kb_ingest")
    assert search is not None, "kb_search 未注册到工具表"
    assert ingest is not None, "kb_ingest 未注册到工具表"
    assert search.scopes == ("tool:read",)
    assert search.idempotent is True
    assert search.max_result_chars > 0
    assert ingest.side_effect is True and ingest.requires_confirmation is True
    assert "knowledge:write" in ingest.scopes


def test_knowledge_service_is_wired_into_runtime(state):
    assert state.runtime.knowledge is state.knowledge
    stats = state.knowledge.stats()
    assert {"documents", "chunks"} <= set(stats)


def test_kb_search_returns_citations_and_respects_acl(state):
    ingest_seed_docs(state)
    ctx = ToolContext(scopes=frozenset({"tool:read"}))

    hits = asyncio.run(state.registry.execute("kb_search", {"query": "LoRA 的秩 r 取值", "top_k": 2}, ctx))
    assert hits.ok, hits.error
    assert "[1]" in hits.content
    citations = hits.meta.get("citations") or []
    assert citations and citations[0]["doc_id"] == "lora_notes"

    # finance_policy 入库时带 acl=["finance"]：无 group 时不可见
    blocked = asyncio.run(state.registry.execute("kb_search", {"query": "差旅住宿上限", "top_k": 3}, ctx))
    assert blocked.ok and "finance_policy" not in blocked.content

    allowed_ctx = ToolContext(scopes=frozenset({"tool:read", "kb:group:finance"}))
    allowed = asyncio.run(state.registry.execute("kb_search", {"query": "差旅住宿上限", "top_k": 3}, allowed_ctx))
    assert allowed.ok and "finance_policy" in allowed.content
