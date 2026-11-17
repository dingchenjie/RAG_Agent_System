"""经验沉淀：提炼规则、人工确认、注入上下文。"""
from __future__ import annotations

import asyncio

from hermes.api.app import build_state
from hermes.learning.lessons import (
    PLAN_TOO_LONG,
    TOOL_INSTABILITY,
    UNSUPPORTED_ANSWER,
    LessonService,
)
from hermes.llm.mock import MockLLM
from hermes.observability.events import Event, EventType
from hermes.storage.db import Database
from tests.conftest import ingest_seed_docs


def _events() -> list[Event]:
    return [
        Event(run_id="r1", type=EventType.TOOL_CALL_FAILED, payload={"tool": "kb_search"}),
        Event(run_id="r1", type=EventType.TOOL_CALL_FAILED, payload={"tool": "kb_search"}),
        Event(run_id="r1", type=EventType.KNOWLEDGE_RETRIEVED, payload={"contexts": ["LoRA 秩 r 常用 8"]}),
        Event(run_id="r1", type=EventType.AGENT_PLANNED,
              payload={"subtasks": [{"id": f"s{i}"} for i in range(1, 5)]}),
    ]


def test_observe_run_extracts_three_rules(settings):
    db = Database(settings.db_path)
    try:
        service = LessonService(db)
        proposed = service.observe_run(
            run_id="r1", events=_events(), state="succeeded", answer="", steps=2,
            max_steps=6, citations=0, tenant_id="t1",
        )
        assert {lesson.rule for lesson in proposed} == {
            TOOL_INSTABILITY, UNSUPPORTED_ANSWER, PLAN_TOO_LONG}
        assert all(lesson.status == "pending" for lesson in proposed)
        assert len(service.list(tenant_id="t1")) == 3
    finally:
        db.close()


def test_propose_dedupes_and_status_gates_injection(settings):
    db = Database(settings.db_path)
    try:
        service = LessonService(db)
        first = service.propose(TOOL_INSTABILITY, "第一版", {}, tenant_id="t1")
        service.propose(TOOL_INSTABILITY, "第二版", {}, tenant_id="t1")
        records = service.list(tenant_id="t1")
        assert len(records) == 1
        assert records[0].hits == 2 and records[0].content == "第二版"

        # pending 不注入
        assert service.render_active(tenant_id="t1") == ""
        assert service.set_status(first.id, "active", tenant_id="t1")
        rendered = service.render_active(tenant_id="t1")
        assert "第二版" in rendered and "[历史教训]" not in rendered

        assert service.set_status(first.id, "rejected", tenant_id="t1")
        assert service.render_active(tenant_id="t1") == ""
        assert not service.set_status("missing", "active", tenant_id="t1")
    finally:
        db.close()


def test_runtime_injects_active_lessons(settings):
    state = build_state(settings, MockLLM())
    ingest_seed_docs(state)
    lesson = state.lessons.propose(TOOL_INSTABILITY, "调用 kb_search 前先确认查询词存在。")
    state.lessons.set_status(lesson.id, "active")

    result = asyncio.run(state.runtime.run(
        user_input="LoRA 的秩 r 通常设置为多少？", tenant_id="default", user_id="u1",
    ))
    sections = [section["name"] for section in result.context.get("sections", [])]
    assert "历史教训" in sections
    assert result.state == "succeeded"
