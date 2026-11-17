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


def test_cross_encoder_rerank_expands_candidate_pool(settings):
    """交叉重排只能从召回候选里挑，候选池必须大于最终 top-k，否则只是换顺序。"""
    from hermes.knowledge.rerank import CrossEncoderReranker, RRFReranker

    state = build_state(settings, MockLLM())
    calls: list[int] = []

    def recording_search(query, *, top_k, **kwargs):
        calls.append(top_k)
        return []

    state.knowledge.index.search = recording_search
    state.knowledge.reranker = RRFReranker()
    state.knowledge.search("LoRA", top_k=4)
    state.knowledge.reranker = CrossEncoderReranker()
    state.knowledge.search("LoRA", top_k=4)
    assert calls == [4, 12]


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


def test_zhipu_embedder_uses_api_model_not_local_repo_name():
    """本地 bge 仓库名不能被当成云端 embedding 的 model（真实 400 的根因）。"""
    from hermes.knowledge.embedding import build_embedder

    embedder = build_embedder(
        "zhipu", api_key="fake-key",
        model="BAAI/bge-small-zh-v1.5", api_model="embedding-3",
    )
    assert embedder.name == "zhipu"
    assert embedder.model == "embedding-3"


def test_search_rejects_dim_mismatch(settings):
    """换过 embedding 后端却没重建索引时，必须显式报错，而不是 zip 截断后静默错分。"""
    import pytest

    from hermes.errors import HermesError
    from hermes.knowledge.embedding import HashingEmbedder

    state = build_state(settings, MockLLM())
    state.knowledge.ingest_text("doc", "LoRA 的秩 r 常用 8 或 16。", version="1")
    state.knowledge.index.embedder = HashingEmbedder(64)
    with pytest.raises(HermesError) as exc:
        state.knowledge.search("LoRA 的秩 r")
    assert exc.value.code == "embedding_mismatch"


class CountingEmbedder:
    """包一层真实 embedder，记录每批实际送去向量化的文本。"""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self.batches: list[list[str]] = []

    def embed(self, texts):
        self.batches.append(list(texts))
        return self.inner.embed(texts)


def test_incremental_ingest_reuses_unchanged_chunks(settings):
    """同一文档升版本时，文本没变的 chunk 不应重复调 embedding。"""
    state = build_state(settings, MockLLM())
    first = state.knowledge.index.ingest_chunks(
        doc_id="doc", chunks=["A 段", "B 段"], version="1")
    assert (first.embedded_chunks, first.reused_chunks) == (2, 0)

    counter = CountingEmbedder(state.knowledge.index.embedder)
    state.knowledge.index.embedder = counter
    second = state.knowledge.index.ingest_chunks(
        doc_id="doc", chunks=["A 段", "B 段", "C 段"], version="2")

    assert second.chunks == 3
    assert (second.embedded_chunks, second.reused_chunks) == (1, 2)
    assert counter.batches == [["C 段"]]
    # 复用后新版本仍可被检索到
    hits = state.knowledge.search("A 段", top_k=3)
    assert hits and hits[0].version == "2"
