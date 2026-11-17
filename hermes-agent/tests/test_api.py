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


def test_sse_stream_closes_after_terminal_event(client):
    run_id = client.post("/v1/runs", json={"input": "你好"}).json()["run_id"]
    _wait(client, run_id)

    # 已结束的 run 必须回放完就关流，否则客户端会一直挂心跳
    with client.stream("GET", f"/v1/runs/{run_id}/events") as response:
        assert response.status_code == 200
        events = [line for line in response.iter_lines() if line.startswith("event:")]
    assert any("run.completed" in line for line in events)


def test_sse_last_event_id_resumes(client):
    run_id = client.post("/v1/runs", json={"input": "LoRA 的秩 r 通常设置为多少？"}).json()["run_id"]
    _wait(client, run_id)
    trace = client.get(f"/v1/runs/{run_id}/trace").json()["events"]
    first_seq = trace[0]["seq"]

    with client.stream("GET", f"/v1/runs/{run_id}/events",
                       headers={"Last-Event-ID": str(first_seq)}) as response:
        ids = [int(line.split(":", 1)[1]) for line in response.iter_lines()
               if line.startswith("id:")]
    assert ids, "续传应当补发 Last-Event-ID 之后的事件"
    assert min(ids) == first_seq + 1


def test_replay_and_diff_endpoints(client):
    baseline = client.post("/v1/runs", json={"input": "你好"}).json()["run_id"]
    candidate = client.post("/v1/runs", json={"input": "你好"}).json()["run_id"]
    _wait(client, baseline)
    _wait(client, candidate)

    timeline = client.get(f"/v1/runs/{candidate}/replay").json()["timeline"]
    assert timeline and timeline[0]["type"] == "run.created"
    assert any(item["type"] == "run.completed" for item in timeline)

    diff = client.get(f"/v1/runs/{candidate}/diff", params={"against": baseline}).json()["diff"]
    assert diff["state_changed"] is False
    assert diff["tools_added"] == [] and diff["tools_removed"] == []


def test_trigger_crud_and_validation(client):
    created = client.post("/v1/triggers", json={
        "name": "每日摘要", "input": "总结今天的知识库更新", "interval_s": 30,
    })
    assert created.status_code == 201
    trigger_id = created.json()["id"]

    listing = client.get("/v1/triggers").json()["triggers"]
    assert any(item["id"] == trigger_id for item in listing)

    client.post(f"/v1/triggers/{trigger_id}/disable")
    listing = client.get("/v1/triggers").json()["triggers"]
    assert next(item for item in listing if item["id"] == trigger_id)["enabled"] is False

    client.post(f"/v1/triggers/{trigger_id}/enable")
    client.delete(f"/v1/triggers/{trigger_id}")
    assert all(item["id"] != trigger_id for item in client.get("/v1/triggers").json()["triggers"])

    too_fast = client.post("/v1/triggers", json={"name": "x", "input": "y", "interval_s": 5})
    assert too_fast.status_code == 422
    assert client.post("/v1/triggers/trg_missing/enable").status_code == 404


def test_run_context_injection_and_tool_version(client):
    page = "客户详情页：客户 A，偏好邮件联系"
    run_id = client.post("/v1/runs", json={
        "input": "LoRA 的秩 r 通常设置为多少？",
        "context": {"page": page, "rules": ["回答必须给出处"]},
    }).json()["run_id"]
    _wait(client, run_id)

    events = client.get(f"/v1/runs/{run_id}/trace").json()["events"]
    created = next(event for event in events if event["type"] == "run.created")
    assert created["payload"]["context"] == {"page_chars": len(page), "rules": 1}

    built = next(event for event in events if event["type"] == "context.built")
    sections = [section["name"] for section in built["payload"]["sections"]]
    assert "当前页面" in sections

    started = next(event for event in events if event["type"] == "tool.started")
    assert started["payload"]["version"], "工具事件必须带版本，replay/diff 才能发现版本变化"


def test_run_context_rejects_too_long_page(client):
    response = client.post("/v1/runs", json={
        "input": "你好", "context": {"page": "x" * 2001},
    })
    assert response.status_code == 422


def test_sessions_list_and_messages(client):
    created = client.post("/v1/runs", json={"input": "你好"}).json()
    run_id, session_id = created["run_id"], created["session_id"]
    body = _wait(client, run_id)

    listing = client.get("/v1/sessions").json()["items"]
    assert any(item["session_id"] == session_id for item in listing)

    messages = client.get(f"/v1/sessions/{session_id}/messages").json()["items"]
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[1]["content"] == body["answer"]

    assert client.get("/v1/sessions/sess_missing/messages").status_code == 404


def test_memory_subject_association(client):
    created = client.post("/v1/memory/propose", json={
        "text": "我叫丁辰杰，我的邮箱是 dcj@example.com",
        "tenant_id": "t1", "user_id": "u1",
        "subject_type": "customer", "subject_id": "c_001",
    }).json()["items"]
    assert created and all(item["subject_type"] == "customer" for item in created)

    matched = client.get("/v1/memory", params={
        "tenant_id": "t1", "user_id": "u1",
        "subject_type": "customer", "subject_id": "c_001",
    }).json()["items"]
    assert len(matched) == len(created)

    other = client.get("/v1/memory", params={
        "tenant_id": "t1", "user_id": "u1",
        "subject_type": "customer", "subject_id": "c_002",
    }).json()["items"]
    assert other == []


def test_run_detail_links_citations_and_tools(client):
    created = client.post("/v1/runs", json={"input": "LoRA 的秩 r 通常设置为多少？"}).json()
    body = _wait(client, created["run_id"])
    assert body["state"] == "succeeded"

    assert body["citations"], "终态应携带引用，回答才能追溯到文档"
    assert body["citations"][0]["doc_id"] == "lora_notes"

    tools = {item["tool"]: item for item in body["tool_calls"]}
    assert tools["kb_search"]["version"] == "1.1"
    assert tools["kb_search"]["ok"] is True


def test_run_detail_without_terminal_event_has_empty_artifacts(client):
    state = client.app.state.hermes
    session = state.sessions.create_session(tenant_id="default", user_id="anonymous", title="排队中")
    state.runs.create_run(session_id=session.id, tenant_id="default", user_id="anonymous",
                          user_input="排队中", run_id="run_queued_test")

    body = client.get("/v1/runs/run_queued_test").json()
    assert body["citations"] == []
    assert body["tool_calls"] == []


def test_knowledge_versions_and_rollback(client):
    for version, text in (("1", "旧版：差旅报销上限 500 元。"), ("2", "新版：差旅报销上限 800 元。")):
        resp = client.post("/v1/knowledge/documents", json={
            "doc_id": "policy", "version": version, "title": f"差旅制度 {version}", "text": text,
        })
        assert resp.status_code == 200

    items = client.get("/v1/knowledge/documents/policy/versions").json()["items"]
    assert [item["version"] for item in items] == ["2", "1"]
    assert items[0]["active"] is True and items[1]["active"] is False
    assert items[0]["chunks"] >= 1

    hit = client.get("/v1/knowledge/search", params={"q": "差旅报销上限"}).json()["hits"][0]
    assert "800" in hit["text"]

    resp = client.post("/v1/knowledge/documents/policy/versions/1/activate")
    assert resp.status_code == 200
    hit = client.get("/v1/knowledge/search", params={"q": "差旅报销上限"}).json()["hits"][0]
    assert "500" in hit["text"]

    items = client.get("/v1/knowledge/documents/policy/versions").json()["items"]
    assert {item["version"]: item["active"] for item in items} == {"1": True, "2": False}

    assert client.get("/v1/knowledge/documents/nope/versions").status_code == 404
    assert client.post("/v1/knowledge/documents/policy/versions/9/activate").status_code == 404
