"""进程重启对账：queued 自动重新入队，running 标记 interrupted 后可恢复。"""
from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from hermes.api.app import create_app
from hermes.llm.mock import MockLLM
from hermes.runtime.store import RunStore, SessionStore
from hermes.storage.db import Database
from hermes.types import Message
from tests.conftest import make_settings


def _wait(client: TestClient, run_id: str, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = client.get(f"/v1/runs/{run_id}").json()
        if body["state"] in {"succeeded", "failed", "cancelled", "paused"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} 未在 {timeout_s}s 内结束")


def _preset(tmp_path) -> tuple[str, str]:
    """在应用启动前直接写入两条在途 run，模拟进程崩溃现场。"""
    settings = make_settings(tmp_path)
    db = Database(settings.db_path)
    sessions = SessionStore(db)
    runs = RunStore(db)

    session = sessions.create_session(tenant_id="default", user_id="anonymous", title="恢复用例")
    queued_id = "run_recover_queued"
    runs.create_run(session_id=session.id, tenant_id="default", user_id="anonymous",
                    user_input="你好", run_id=queued_id,
                    request={"mode": "single", "scopes": ["tool:read"], "context": None})
    runs.update(queued_id, state="queued")

    running_id = "run_recover_running"
    runs.create_run(session_id=session.id, tenant_id="default", user_id="anonymous",
                    user_input="你好", run_id=running_id,
                    request={"mode": "single", "scopes": ["tool:read"], "context": None})
    checkpoint = {
        "messages": [Message(role="user", content="你好").to_dict()],
        "step": 0,
        "pending": None,
        "usage": {},
        "context": None,
    }
    db.execute("UPDATE runs SET checkpoint=?, state='running' WHERE id=?",
               (json.dumps(checkpoint, ensure_ascii=False), running_id))
    db.close()
    return queued_id, running_id


def test_recovery_requeues_and_interrupts(tmp_path):
    queued_id, running_id = _preset(tmp_path)
    app = create_app(make_settings(tmp_path), MockLLM())
    with TestClient(app) as client:
        assert _wait(client, queued_id)["state"] == "succeeded"

        running = client.get(f"/v1/runs/{running_id}").json()
        assert running["state"] == "interrupted"
        assert "checkpoint" in (running["error"] or "")

        trace = client.get(f"/v1/runs/{running_id}/trace").json()["events"]
        failed = [event for event in trace if event["type"] == "run.failed"]
        assert failed and failed[0]["payload"]["kind"] == "process_restart"
        assert failed[0]["payload"]["recoverable"] is True

        resumed = client.post(f"/v1/runs/{running_id}/resume")
        assert resumed.status_code == 200
        assert _wait(client, running_id)["state"] == "succeeded"


def test_mark_interrupted_ignores_finished_runs(settings):
    db = Database(settings.db_path)
    sessions = SessionStore(db)
    runs = RunStore(db)
    session = sessions.create_session(tenant_id="default", user_id="anonymous")
    runs.create_run(session_id=session.id, tenant_id="default", user_id="anonymous",
                    user_input="hi", run_id="run_done")
    runs.update("run_done", state="succeeded")

    assert runs.mark_interrupted("run_done", "不应生效") is False
    assert runs.get_run("run_done")["state"] == "succeeded"
    db.close()


def test_mock_delay_setting(monkeypatch):
    """恢复演练靠 Mock LLM 延迟制造可 kill 窗口，配置必须从环境变量读得到。"""
    from hermes.config import Settings

    monkeypatch.setenv("HERMES_MOCK_DELAY_S", "1.5")
    assert Settings.from_env().mock_delay_s == 1.5
