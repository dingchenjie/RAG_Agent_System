"""用量聚合与事件保留：都不新增事实源，只读 runs / events。"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from hermes.api.app import create_app
from hermes.llm.mock import MockLLM
from hermes.observability.events import Event, EventType
from tests.conftest import ingest_seed_docs


@pytest.fixture
def client(settings):
    app = create_app(settings, MockLLM())
    with TestClient(app) as test_client:
        ingest_seed_docs(test_client.app.state.hermes)
        yield test_client


def _wait(client, run_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/v1/runs/{run_id}").json()
        if body["state"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            return body
        time.sleep(0.05)
    raise AssertionError("run 未在超时内结束")


def test_usage_summary_by_tenant(client):
    created = client.post("/v1/runs", json={"input": "1+1 等于几？"}).json()
    _wait(client, created["run_id"])

    body = client.get("/v1/usage", params={"window": "24h"}).json()
    assert body["tenant_id"] == "default"
    assert body["window"] == "24h"
    assert body["runs"] >= 1
    assert body["total_tokens"] > 0
    assert body["by_state"]["succeeded"]["runs"] >= 1
    assert body["cost_usd"] >= 0

    assert client.get("/v1/usage", params={"window": "7x"}).status_code == 422


def test_event_retention_keeps_terminal_events(client):
    state = client.app.state.hermes
    session = state.sessions.create_session(tenant_id="default", user_id="anonymous", title="保留策略")
    for run_id in ("run_old", "run_new"):
        state.runs.create_run(session_id=session.id, tenant_id="default", user_id="anonymous",
                              user_input=run_id, run_id=run_id)
        state.event_store.append(Event(run_id=run_id, type=EventType.RUN_STARTED, payload={}))
        state.event_store.append(Event(run_id=run_id, type=EventType.RUN_COMPLETED,
                                       payload={"citations": [], "tools": []}))
        state.runs.update(run_id, state="succeeded")

    old_ts = time.time() - 60 * 86400
    state.db.execute("UPDATE runs SET updated_at=? WHERE id=?", (old_ts, "run_old"))
    state.db.execute("UPDATE events SET ts=? WHERE run_id=?", (old_ts, "run_old"))

    deleted = state.event_store.prune(before_ts=time.time() - 30 * 86400)
    assert deleted == 1
    assert [event.type for event in state.event_store.history("run_old")] == [EventType.RUN_COMPLETED]
    assert len(state.event_store.history("run_new")) == 2

    # 终态仍在，所以 run 详情还能拿到工具/引用（这里本来就是空数组）
    body = client.get("/v1/runs/run_old").json()
    assert body["state"] == "succeeded"
    assert body["citations"] == []


def test_admin_prune_requires_admin(settings):
    settings.auth_mode = "api_key"
    settings.api_keys = "user-key:tenant-a:u1:run:read"
    app = create_app(settings, MockLLM())
    with TestClient(app) as client:
        resp = client.post("/v1/admin/prune", headers={"Authorization": "Bearer user-key"})
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "forbidden"
