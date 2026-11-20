"""向量后端：归一化、auto 降级、非法配置、维度不一致保护。"""
from __future__ import annotations

import math

import pytest

from hermes.errors import HermesError
from hermes.knowledge import embedding as embedding_module
from hermes.knowledge.embedding import HashingEmbedder, build_embedder
from hermes.knowledge.index import KnowledgeIndex
from hermes.storage.db import Database


def test_hashing_embedder_is_normalized():
    vector = HashingEmbedder(64).embed(["LoRA 低秩矩阵"])[0]
    assert len(vector) == 64
    assert math.isclose(sum(value * value for value in vector), 1.0, rel_tol=1e-6)


def test_auto_falls_back_to_hashing_then_openai(monkeypatch):
    monkeypatch.setattr(embedding_module, "_available", lambda module: False)
    assert build_embedder("auto").name == "hashing"
    assert build_embedder("auto", api_key="test-key", base_url="https://example.com/v1").name == "openai"


def test_unknown_backend_is_rejected():
    with pytest.raises(HermesError) as exc:
        build_embedder("tfidf")
    assert exc.value.code == "invalid_config"


def test_embedding_dimension_mismatch_is_rejected(tmp_path):
    db = Database(tmp_path / "knowledge.db")
    KnowledgeIndex(db, HashingEmbedder(32)).ingest(doc_id="d1", text="LoRA 使用低秩矩阵。")
    with pytest.raises(HermesError) as exc:
        KnowledgeIndex(db, HashingEmbedder(16)).ingest(doc_id="d2", text="另一段无关文本。")
    assert exc.value.code == "embedding_mismatch"
