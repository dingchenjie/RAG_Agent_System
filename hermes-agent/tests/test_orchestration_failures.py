"""编排失败隔离：部分失败降级成功，全部失败显式报错。"""
from __future__ import annotations

import asyncio

from hermes.api.state import build_state
from hermes.eval.orchestration import evaluate_orchestration_failures
from tests.conftest import ingest_seed_docs


def test_orchestration_partial_and_total_failure(settings):
    state = build_state(settings)
    ingest_seed_docs(state)
    try:
        result = asyncio.run(evaluate_orchestration_failures(state))
        assert result["scenarios"] == 2
        assert result["passed"] == 2, result["details"]

        partial = next(item for item in result["details"] if item["name"] == "partial")
        assert partial["state"] == "succeeded"
        assert (partial["workers"], partial["ok_workers"]) == (2, 1)
        assert partial["answer_has_failure"] and partial["answer_has_timeout"]

        total = next(item for item in result["details"] if item["name"] == "all_fail")
        assert total["state"] == "failed"
        assert total["ok_workers"] == 0
        assert total["answer_starts_incomplete"]
    finally:
        state.db.close()
