"""HTTP 知识库接口：上传入库、文档列表、删除、健康检查暴露向量/重排后端。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes.api.app import create_app
from hermes.llm.mock import MockLLM


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, MockLLM())) as test_client:
        yield test_client


def test_healthz_reports_embedder_and_reranker(client):
    body = client.get("/healthz").json()
    assert body["embedding"]["provider"] == "hashing"
    assert body["embedding"]["dim"] == 384
    assert body["rerank"]["provider"] == "rrf"


def test_upload_list_and_delete_document(client):
    content = "# 上传文档\nLoRA 的秩 r 常用 8 或 16。".encode()
    response = client.post(
        "/v1/knowledge/files",
        files={"file": ("upload.md", content, "text/markdown")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["doc_id"] == "upload"
    assert body["chunks"] >= 1

    items = client.get("/v1/knowledge/documents").json()["items"]
    uploaded = [item for item in items if item["doc_id"] == "upload"]
    assert len(uploaded) == 1
    assert uploaded[0]["embedding_model"] == "hashing"
    assert uploaded[0]["embedding_dim"] == 384

    deleted = client.delete("/v1/knowledge/documents/upload").json()
    assert deleted["chunks"] >= 1

    remaining = client.get("/v1/knowledge/documents").json()["items"]
    assert all(item["doc_id"] != "upload" for item in remaining)
