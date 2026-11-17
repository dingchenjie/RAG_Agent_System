"""教训注入 A/B：无教训时失败，注入已确认教训后通过。"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hermes.api.app import build_state
from hermes.eval.lessons import LessonAwareLLM, evaluate_lessons_ab, load_lesson_cases
from hermes.eval.runner import EvalRunner
from hermes.learning.lessons import TOOL_INSTABILITY
from tests.conftest import ingest_seed_docs

EVAL = Path(__file__).resolve().parents[1] / "eval"


def _runner(state) -> EvalRunner:
    return EvalRunner(runtime=state.runtime, event_store=state.event_store,
                      dataset_path=EVAL / "lessons_ab.jsonl")


def test_lessons_ab_turns_failure_into_success(settings):
    state = build_state(settings, LessonAwareLLM())
    ingest_seed_docs(state)
    try:
        cases = load_lesson_cases(EVAL / "lessons_ab.jsonl")
        result = asyncio.run(evaluate_lessons_ab(
            runner=_runner(state), lessons=state.lessons, cases=cases))
        assert result["cases"] == len(cases) == 3
        assert result["before_success"] == 0
        assert result["after_success"] == 3
        assert result["delta"] == 1.0
        assert result["lessons_injected"] == 1
        # 注入后教训确实处于 active，且能渲染进上下文
        assert state.lessons.render_active(tenant_id="eval")
        assert all(item["after_ok"] for item in result["per_case"])
    finally:
        state.db.close()


def test_lessons_ab_requires_clean_store(settings):
    state = build_state(settings, LessonAwareLLM())
    try:
        lesson = state.lessons.propose(TOOL_INSTABILITY, "已有教训", {}, tenant_id="eval")
        state.lessons.set_status(lesson.id, "active", tenant_id="eval")
        with pytest.raises(ValueError):
            asyncio.run(evaluate_lessons_ab(
                runner=_runner(state), lessons=state.lessons,
                cases=load_lesson_cases(EVAL / "lessons_ab.jsonl")))
    finally:
        state.db.close()
