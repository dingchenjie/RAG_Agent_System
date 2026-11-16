"""HTTP 层：异步提交、轮询、事件轨迹、幂等、知识库 ACL、记忆生命周期。"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from hermes.api.app import create_app
from hermes.llm.mock import MockLLM
from tests.conftest import ingest_seed_docs


def _wait(client: TestClient, run_id: str, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = client.get(f"/v1/runs/{run_id}").json()
        if body["state"] in {"succeeded", "failed", "cancelled", "paused"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} 未在 {timeout_s}s 内结束")


@pytest.fixture
def client(settings):
    app = create_app(settings, MockLLM())
    with TestClient(app) as test_client:
        ingest_seed_docs(test_client.app.state.hermes)
        yield test_client


def test_health_and_metrics(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["knowledge"]["documents"] >= 2
    assert "hermes" in client.get("/metrics").text


def test_run_lifecycle_and_trace(client):
    response = client.post("/v1/runs", json={"input": "LoRA 的秩 r 通常设置为多少？"})
    assert response.status_code == 202
    run_id = response.json()["run_id"]

    body = _wait(client, run_id)
    assert body["state"] == "succeeded"
    assert "8" in (body["answer"] or "")
    assert body["steps"] >= 1

    trace = client.get(f"/v1/runs/{run_id}/trace").json()
    types = [event["type"] for event in trace["events"]]
    assert "run.started" in types
    assert "context.built" in types
    assert "tool.started" in types
    assert "run.completed" in types


def test_idempotency_header_deduplicates(client):
    headers = {"X-Idempotency-Key": "api-idem-1"}
    first = client.post("/v1/runs", json={"input": "你好"}, headers=headers).json()
    second = client.post("/v1/runs", json={"input": "你好"}, headers=headers).json()
    assert first["run_id"] == second["run_id"]
    assert second["deduplicated"] is True
    _wait(client, first["run_id"])


def test_knowledge_ingest_and_acl(client):
    client.post("/v1/knowledge/documents", json={
        "doc_id": "travel", "version": "1", "acl": ["finance"],
        "text": "差旅住宿上限为每晚上限 600 元。",
    })

    public_hits = client.get("/v1/knowledge/search",
                             params={"q": "差旅住宿上限"}).json()["hits"]
    assert all(hit["doc_id"] != "travel" for hit in public_hits)

    finance_hits = client.get("/v1/knowledge/search",
                              params={"q": "差旅住宿上限", "acl_group": "finance"}).json()["hits"]
    assert any(hit["doc_id"] == "travel" for hit in finance_hits)


def test_memory_lifecycle(client):
    created = client.post("/v1/memory/propose", json={
        "text": "我叫丁辰杰，我的邮箱是 dcj@example.com", "tenant_id": "t1", "user_id": "u1",
    }).json()["items"]
    assert created, "应当从自然语言里抽取到候选记忆"
    memory_id = created[0]["id"]

    pending = client.get("/v1/memory", params={"tenant_id": "t1", "user_id": "u1"}).json()
    assert len(pending["items"]) == len(created)

    confirmed = client.post(f"/v1/memory/{memory_id}/confirm",
                            params={"tenant_id": "t1", "user_id": "u1"}).json()["item"]
    assert confirmed["status"] == "active"

    corrected = client.post(f"/v1/memory/{memory_id}/correct",
                            params={"tenant_id": "t1", "user_id": "u1"},
                            json={"value": "dcj2@example.com", "reason": "用户更正"}).json()["item"]
    assert corrected["value"] == "dcj2@example.com"

    client.delete(f"/v1/memory/{memory_id}", params={"tenant_id": "t1", "user_id": "u1"})
    active = client.get("/v1/memory", params={"tenant_id": "t1", "user_id": "u1"}).json()["items"]
    assert memory_id not in {item["id"] for item in active if item["status"] == "active"}


def test_orchestrate_endpoint(client):
    body = client.post("/v1/orchestrate", json={
        "input": "根据知识库说明 LoRA 的秩 r，并计算 32/4 的结果",
    }).json()
    assert body["state"] == "succeeded"
    assert len(body["plan"]["subtasks"]) == 2
    assert set(body["blackboard"]) == {"finding.s1", "finding.s2"}
