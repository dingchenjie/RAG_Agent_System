"""存储层：run 状态机、checkpoint 恢复、幂等键、会话消息。"""
from __future__ import annotations

from hermes.runtime.store import RunStore, SessionStore
from hermes.storage.db import Database
from hermes.types import Message


def _stores(tmp_path):
    db = Database(tmp_path / "store.db")
    return db, SessionStore(db), RunStore(db)


def test_checkpoint_roundtrip_keeps_pending_confirmation(tmp_path):
    _, _, runs = _stores(tmp_path)
    session = SessionStore(Database(tmp_path / "store.db")).create_session(
        tenant_id="t1", user_id="u1", title="demo")
    runs.create_run(session_id=session.id, tenant_id="t1", user_id="u1",
                    user_input="转账 100 元", run_id="run_x")
    runs.save_checkpoint("run_x", messages=[Message(role="user", content="转账 100 元")],
                         step=2, pending={"id": "call_1", "name": "transfer"}, usage={})
    loaded = runs.load_checkpoint("run_x")
    assert loaded["step"] == 2
    assert loaded["pending"]["name"] == "transfer"
    assert loaded["messages"][0]["content"] == "转账 100 元"


def test_idempotency_key_and_state_updates(tmp_path):
    _, sessions, runs = _stores(tmp_path)
    session = sessions.create_session(tenant_id="t1", user_id="u1")
    runs.create_run(session_id=session.id, tenant_id="t1", user_id="u1",
                    user_input="hi", idempotency_key="idem-1", run_id="run_a")
    hit = runs.find_by_idempotency("idem-1")
    assert hit["id"] == "run_a"
    runs.update("run_a", state="succeeded", answer="done")
    row = runs.get_run("run_a")
    assert (row["state"], row["answer"]) == ("succeeded", "done")
    assert runs.get_run("nope") is None


def test_messages_are_loaded_in_order(tmp_path):
    _, sessions, _ = _stores(tmp_path)
    session = sessions.create_session(tenant_id="t1", user_id="u1")
    sessions.add_message(session.id, Message(role="user", content="第一句"))
    sessions.add_message(session.id, Message(role="assistant", content="第二句"))
    messages = sessions.load_messages(session.id)
    assert [m.content for m in messages] == ["第一句", "第二句"]
