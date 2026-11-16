"""知识服务：版本、ACL、引用。"""
from __future__ import annotations

from hermes.api.app import build_state
from hermes.llm.mock import MockLLM


def test_ingest_and_version_replacement(settings):
    state = build_state(settings, MockLLM())
    state.knowledge.ingest_text("doc", "LoRA 的秩 r 常用 8 或 16，alpha 设为 r 的两倍。", version="1")
    hits = state.knowledge.search("LoRA 的秩 r 取值", top_k=3)
    assert hits and hits[0].version == "1"

    state.knowledge.ingest_text("doc", "新版本说明：LoRA 的秩 r 常用 32。", version="2")
    hits2 = state.knowledge.search("LoRA 的秩 r 取值", top_k=3)
    assert hits2 and hits2[0].version == "2"
    assert "32" in hits2[0].text
    assert state.knowledge.stats()["documents"] == 1


def test_acl_filtering(settings):
    state = build_state(settings, MockLLM())
    state.knowledge.ingest_text("policy", "差旅住宿上限为每晚 600 元。", version="1", acl=["finance"])
    assert state.knowledge.search("差旅住宿上限", top_k=3) == []
    allowed = state.knowledge.search("差旅住宿上限", top_k=3, acl_groups={"finance"})
    assert allowed and "600" in allowed[0].text


def test_citations_are_structured(settings):
    state = build_state(settings, MockLLM())
    state.knowledge.ingest_text("doc", "LoRA 使用低秩矩阵 A 与 B。", version="1", title="lora.md")
    hits = state.knowledge.search("低秩矩阵", top_k=3)
    citations = state.knowledge.citations(hits)
    assert citations[0]["doc_id"] == "doc"
    assert citations[0]["citation"].startswith("doc@v1#chunk-")
    rendered = state.knowledge.render_context(hits)
    assert "[1]" in rendered


def test_chunking_keeps_sentence_boundaries():
    from hermes.knowledge.index import chunk_text

    text = "第一句话。" * 120
    chunks = chunk_text(text, size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)
